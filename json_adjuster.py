import json
import argparse
import random
from pathlib import Path
import shutil
from typing import List, Tuple, Optional

from PIL import Image

# -----------------------------
# File helpers
# -----------------------------
def load_json(path: Path):
    # Read a JSON file using UTF-8 encoding and parse it to a Python object.
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(obj, path: Path):
    # Write a JSON file with stable, human-friendly formatting (indentation).
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

def img_name(n: int, ext: str):
    # Create the canonical image filename used by the dataset.
    # Example: n=5, ext="jpeg" -> "IMG000005.jpeg"
    return f"IMG{n:06d}.{ext}"

# -----------------------------
# Normalization and class mapping
# -----------------------------
def _normalize_class_name(name: str) -> str:
    # Normalize strings so they can be compared against folder names.
    # This matches how folder names are structured in generated_images.
    # Example: "simple bone cyst" -> "simple_bone_cyst"
    return "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower())

def _class_label_map():
    # Map normalized class keys to the label text stored in annotation JSONs.
    # These labels must match the labels in your existing annotation files,
    # because CustomDataset uses them to build the class index.
    return {
        "osteochondroma": "osteochondroma",
        "osteosarcoma": "osteosarcoma",
        "multiple_osteochondromas": "multiple osteochondromas",
        "simple_bone_cyst": "simple bone cyst",
        "giant_cell_tumor": "giant cell tumor",
        "synovial_osteochondroma": "synovial osteochondroma",
        "osteofibroma": "osteofibroma",
    }

# -----------------------------
# CLI parsing helpers
# -----------------------------
def _parse_additions(raw_additions):
    # Parse repeated --add class=count arguments into a dict.
    # Example: --add osteofibroma=9 --add synovial_osteochondroma=31
    # Result: {"osteofibroma": 9, "synovial_osteochondroma": 31}
    additions = {}
    if not raw_additions:
        return additions
    label_map = _class_label_map()
    for item in raw_additions:
        if "=" not in item:
            raise ValueError(f"Invalid --add format: {item!r} (expected class=count)")
        # Split into "class" and "count" parts.
        cls, count = item.split("=", 1)
        cls_norm = _normalize_class_name(cls)
        if cls_norm not in label_map:
            raise ValueError(f"Unknown class in --add: {cls}")
        try:
            count_int = int(count)
        except ValueError as exc:
            raise ValueError(f"Invalid count in --add {item!r}") from exc
        if count_int <= 0:
            raise ValueError(f"Count in --add must be > 0: {item!r}")
        # Allow multiple --add entries for the same class (sum them up).
        additions[cls_norm] = additions.get(cls_norm, 0) + count_int
    return additions

# -----------------------------
# File collection helpers
# -----------------------------
def _collect_files_by_class(gen_root: Path, class_names):
    # Scan all subfolders under gen_root and group images by class.
    # A subfolder is assigned to the first class whose normalized name
    # appears in the normalized folder name (substring match).
    #
    # Example:
    #   Folder: "upper_limb_osteofibroma_gamma6_snapshot15800_trunc1.0"
    #   Normalized: "upper_limb_osteofibroma_gamma6_snapshot15800_trunc1_0"
    #   Match: class "osteofibroma" -> images go to that class pool.
    allowed_suffixes = {".jpg", ".jpeg", ".png"}
    files_by_class = {cls: [] for cls in class_names}
    if not gen_root.exists():
        raise FileNotFoundError(f"gen-root not found: {gen_root}")
    for subdir in gen_root.iterdir():
        if not subdir.is_dir():
            continue
        subdir_norm = _normalize_class_name(subdir.name)
        for cls in class_names:
            if cls in subdir_norm:
                # Collect all image files in this matching folder.
                files = [p for p in subdir.iterdir() if p.suffix.lower() in allowed_suffixes]
                files_by_class[cls].extend(files)
                break
    return files_by_class

# -----------------------------
# Annotation helpers
# -----------------------------
def _labelme_payload(label: str, image_filename: str, width: int, height: int):
    # Build a minimal Labelme-style annotation with a full-image rectangle.
    # This keeps the label usable by the existing dataset loader.
    #
    # Note: CustomDataset uses the first shape label as the image label.
    # We create a single rectangle covering the full image so the label exists.
    return {
        "version": "5.4.1",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": [[0, 0], [max(width - 1, 1), max(height - 1, 1)]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            }
        ],
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }

# -----------------------------
# Main CLI entry point
# -----------------------------
def main():
    # CLI configuration.
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True, help="split.json input")
    ap.add_argument("--out", type=Path, default=None, help="output json (default: same folder)")
    ap.add_argument("--n", type=int, default=None, help="how many new train indices to add")
    ap.add_argument("--gen-dir", type=Path, default=None, help="folder with generated images (optional)")
    ap.add_argument("--gen-root", type=Path, default=None, help="root folder with class subfolders")
    ap.add_argument(
        "--add",
        action="append",
        default=None,
        help="Add class=count (e.g. --add osteofibroma=9). Can be repeated.",
    )
    ap.add_argument("--label", type=str, default=None, help="label for --gen-dir when writing annotations")
    ap.add_argument("--target-dir", type=Path, default=None, help="optional output dir for selected images")
    ap.add_argument("--anno-dir", type=Path, default=None, help="optional output dir for annotation jsons")
    ap.add_argument("--move", action="store_true", help="move files instead of copy when using --target-dir")
    ap.add_argument("--seed", type=int, default=None, help="random seed for sampling")
    ap.add_argument("--start-idx", type=int, required=True,help="next index to start (e.g. 1707)")
    ap.add_argument("--start-img", type=int, required=True, help="next image number (e.g. 1868 for IMG001868)")
    ap.add_argument("--ext", default="jpeg")
    ap.add_argument("--rename", action="store_true", help="rename files in --gen-dir")
    args = ap.parse_args()

    # Load existing split and validate basic structure.
    split = load_json(args.split)
    if "train" not in split or not isinstance(split["train"], list):
        raise ValueError("Input split JSON must contain a list under 'train'.")

    # If the output path is not provided, create it next to the input file.
    if args.out is None:
        args.out = args.split.parent / (args.split.stem + "_new.json")

    # Ensure the caller didn't select conflicting modes.
    if args.gen_dir is not None and args.gen_root is not None:
        raise ValueError("Use either --gen-dir or --gen-root, not both.")

    # This list will hold (file_path, label_or_none) for every selected image.
    # The label is only known when we pick via --gen-root.
    selected_files: List[Tuple[Path, Optional[str]]] = []
    if args.gen_root is not None:
        # Multi-class sampling from a root folder that has many class subfolders.
        additions = _parse_additions(args.add)
        if not additions:
            raise ValueError("When using --gen-root, provide at least one --add class=count.")
        label_map = _class_label_map()
        # Use a local RNG so random selection can be controlled with --seed.
        rng = random.Random(args.seed)
        files_by_class = _collect_files_by_class(args.gen_root, additions.keys())
        for cls, count in additions.items():
            available = files_by_class.get(cls, [])
            if len(available) < count:
                raise ValueError(
                    f"Not enough images for class '{cls}': requested {count}, found {len(available)}"
                )
            # Random, non-repeating selection from the pool for this class.
            picked = rng.sample(available, count)
            # Store the file and the correct annotation label for each pick.
            selected_files.extend([(p, label_map[cls]) for p in picked])
        n_new = len(selected_files)
    elif args.gen_dir is not None:
        # Simple mode: use all images from a single folder (one class or mixed).
        allowed_suffixes = {".jpg", ".jpeg", ".png"}
        files = sorted([p for p in args.gen_dir.iterdir() if p.suffix.lower() in allowed_suffixes])
        n_new = len(files)
        if n_new == 0:
            raise ValueError("No images found in gen-dir.")
        # Label is unknown in this mode unless --label is provided.
        selected_files = [(p, None) for p in files]
    else:
        # Pure index-only mode: just add N indices without touching any files.
        if args.n is None or args.n <= 0:
            raise ValueError("Use --n > 0 OR provide --gen-dir.")
        n_new = args.n

    # Add indices and optionally move/copy/rename files + create annotations.
    for i in range(n_new):
        # Assign a new dataset index and image filename for this synthetic image.
        new_idx = args.start_idx + i
        new_imgnum = args.start_img + i
        new_filename = img_name(new_imgnum, args.ext)

        # Always extend the training split with the new indices.
        split["train"].append(new_idx)

        # If requested, rename and move/copy the image into the target dataset folder.
        if args.rename and args.gen_dir is not None:
            src = selected_files[i][0]
            dst = args.gen_dir / new_filename
            if dst.exists():
                raise FileExistsError(f"Refusing to overwrite: {dst}")
            # In gen-dir mode, rename happens inside the same folder.
            shutil.move(str(src), str(dst))
        elif args.rename and args.gen_root is not None and args.target_dir is not None:
            args.target_dir.mkdir(parents=True, exist_ok=True)
            src = selected_files[i][0]
            dst = args.target_dir / new_filename
            if dst.exists():
                raise FileExistsError(f"Refusing to overwrite: {dst}")
            if args.move:
                # Move removes the original file (so it cannot be sampled again).
                shutil.move(str(src), str(dst))
            else:
                # Copy keeps the original in generated_images.
                shutil.copy2(str(src), str(dst))

        # If requested, write a minimal Labelme-style annotation JSON.
        # The label is taken from the class mapping (gen_root mode) or
        # from --label (gen_dir mode).
        if args.anno_dir is not None:
            label = selected_files[i][1]
            if label is None:
                if args.label is None:
                    raise ValueError("Use --label when writing annotations from --gen-dir.")
                # Map the provided label to the canonical annotation label if possible.
                label = _class_label_map().get(_normalize_class_name(args.label), args.label)
            args.anno_dir.mkdir(parents=True, exist_ok=True)
            image_path = new_filename
            ann_path = args.anno_dir / Path(new_filename).with_suffix(".json").name
            if ann_path.exists():
                raise FileExistsError(f"Refusing to overwrite: {ann_path}")
            if args.target_dir is None and args.gen_dir is None and args.gen_root is None:
                raise ValueError("Annotations require image files to be selected.")
            # Resolve the actual image file path so we can read width/height.
            image_file = (args.target_dir / new_filename) if args.target_dir else selected_files[i][0]
            with Image.open(image_file) as img:
                width, height = img.size
            # Create and save the Labelme JSON payload.
            payload = _labelme_payload(label, image_path, width, height)
            save_json(payload, ann_path)

    # Write the updated split JSON.
    save_json(split, args.out)
    print(f"Saved: {args.out}")
    print(f"Added {n_new} new train indices: {args.start_idx}..{args.start_idx + n_new - 1}")

if __name__ == "__main__":
    main()
