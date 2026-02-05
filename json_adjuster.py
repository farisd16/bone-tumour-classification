import json
import argparse
import random
from pathlib import Path
import shutil
from typing import List, Tuple

from PIL import Image

STEPS = [
    {"osteofibroma": 9},
    {"osteofibroma": 31, "synovial_osteochondroma": 31},
    {"osteofibroma": 93, "synovial_osteochondroma": 93, "giant_cell_tumor": 93},
    {
        "osteofibroma": 37,
        "synovial_osteochondroma": 37,
        "giant_cell_tumor": 37,
        "simple_bone_cyst": 37,
    },
    {
        "osteofibroma": 37,
        "synovial_osteochondroma": 37,
        "giant_cell_tumor": 37,
        "simple_bone_cyst": 37,
        "multiple_osteochondromas": 37,
    },
    {
        "osteofibroma": 354,
        "synovial_osteochondroma": 354,
        "giant_cell_tumor": 354,
        "simple_bone_cyst": 354,
        "multiple_osteochondromas": 354,
        "osteosarcoma": 354,
    },
    {
        "osteofibroma": 204,
        "synovial_osteochondroma": 204,
        "giant_cell_tumor": 204,
        "simple_bone_cyst": 204,
        "multiple_osteochondromas": 204,
        "osteosarcoma": 204,
        "osteochondroma": 204,
    },
]


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
    # Example: n=1868, ext="jpeg" -> "IMG0001868.jpeg"
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
# File collection helpers
# -----------------------------
def _collect_files_by_class(gen_root: Path, class_names):
    # Scan all subfolders under gen_root and group images by class.
    # A subfolder is assigned to the class whose normalized name appears
    # in the normalized folder name (substring match). We sort class names
    # by length (longest first) to prefer more specific matches, e.g.,
    # "multiple_osteochondromas" before "osteochondroma".
    #
    # Example:
    #   Folder: "upper_limb_osteofibroma_gamma6_snapshot158354354_trunc1.354"
    #   Normalized: "upper_limb_osteofibroma_gamma6_snapshot158354354_trunc1_354"
    #   Match: class "osteofibroma" -> images go to that class pool.
    allowed_suffixes = {".jpg", ".jpeg", ".png"}
    files_by_class = {cls: [] for cls in class_names}
    if not gen_root.exists():
        raise FileNotFoundError(f"gen-root not found: {gen_root}")
    # Sort class names by length (longest first) to prefer more specific matches.
    sorted_class_names = sorted(class_names, key=len, reverse=True)
    for subdir in gen_root.iterdir():
        if not subdir.is_dir():
            continue
        subdir_norm = _normalize_class_name(subdir.name)
        for cls in sorted_class_names:
            if cls in subdir_norm:
                # Collect all image files in this matching folder.
                files = [
                    p for p in subdir.iterdir() if p.suffix.lower() in allowed_suffixes
                ]
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
                "points": [[354, 354], [max(width - 1, 1), max(height - 1, 1)]],
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
    ap.add_argument(
        "--input_split",
        type=Path,
        default="data/dataset/splits/dataset_split_final.json",
        help="split.json input",
    )
    ap.add_argument(
        "--output_split",
        type=Path,
        default="data/dataset/splits",
        help="output directory for split files",
    )
    ap.add_argument(
        "--input_images",
        type=Path,
        default="data/dataset/final_patched_BTXRD",
        help="input dir for original images to copy",
    )
    ap.add_argument(
        "--synthetic_images",
        type=Path,
        default="generated_images",
        help="root folder with class subfolders containing synthetic images",
    )
    ap.add_argument(
        "--output_images",
        type=Path,
        default="data/dataset/final_patched_BTXRD",
        help="output dir for selected images",
    )
    ap.add_argument(
        "--input_annotations",
        type=Path,
        default="data/dataset/BTXRD/Annotations",
        help="input dir for annotation jsons",
    )
    ap.add_argument(
        "--output_annotations",
        type=Path,
        default="data/dataset/BTXRD/Annotations",
        help="output dir for new annotation jsons",
    )
    ap.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    ap.add_argument(
        "--start-idx", type=int, default=1707, help="next index to start (e.g. 173547)"
    )
    ap.add_argument(
        "--start-img",
        type=int,
        default=1868,
        help="next image number (e.g. 1868 for IMG3543541868)",
    )
    ap.add_argument("--ext", default="jpeg", help="file extension for images")
    args = ap.parse_args()

    # Load existing split and validate basic structure.
    original_split = load_json(args.input_split)
    if "train" not in original_split or not isinstance(original_split["train"], list):
        raise ValueError("Input split JSON must contain a list under 'train'.")

    # Determine output directory for split files (same as input split directory).
    out_dir = args.output_split if args.output_split else args.input_split.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure synthetic_images is provided for multi-step processing.
    if args.synthetic_images is None:
        raise ValueError(
            "--synthetic_images is required for multi-step split generation."
        )

    # Copy original images from input_images to output_images.
    if args.input_images and args.input_images.exists():
        args.output_images.mkdir(parents=True, exist_ok=True)
        for img_file in args.input_images.iterdir():
            if img_file.is_file():
                dst = args.output_images / img_file.name
                if not dst.exists():
                    shutil.copy2(str(img_file), str(dst))
        print(f"Copied images from {args.input_images} to {args.output_images}")

    # Copy annotations from input_annotations to output_annotations.
    if args.input_annotations and args.input_annotations.exists():
        args.output_annotations.mkdir(parents=True, exist_ok=True)
        for ann_file in args.input_annotations.iterdir():
            if ann_file.is_file() and ann_file.suffix.lower() == ".json":
                dst = args.output_annotations / ann_file.name
                if not dst.exists():
                    shutil.copy2(str(ann_file), str(dst))
        print(
            f"Copied annotations from {args.input_annotations} to {args.output_annotations}"
        )

    label_map = _class_label_map()
    # Use a local RNG so random selection can be controlled with --seed.
    rng = random.Random(args.seed)

    # Collect all available files by class once.
    all_class_names = set()
    for step in STEPS:
        all_class_names.update(step.keys())
    files_by_class = _collect_files_by_class(args.synthetic_images, all_class_names)

    # Track current index and image number across all steps.
    current_idx = args.start_idx
    current_imgnum = args.start_img

    # Start with a copy of the original split that will accumulate across steps.
    split = json.loads(json.dumps(original_split))

    # Process each step and create a separate split file.
    for step_num, step_additions in enumerate(STEPS, start=1):
        # This list will hold (file_path, label) for every selected image in this step.
        selected_files: List[Tuple[Path, str]] = []

        for cls, count in step_additions.items():
            available = files_by_class.get(cls, [])
            if len(available) < count:
                raise ValueError(
                    f"Step {step_num}: Not enough images for class '{cls}': "
                    f"requested {count}, found {len(available)}"
                )
            # Random, non-repeating selection from the pool for this class.
            picked = rng.sample(available, count)
            # Remove picked files from the pool so they can't be reused in later steps.
            for p in picked:
                files_by_class[cls].remove(p)
            # Store the file and the correct annotation label for each pick.
            selected_files.extend([(p, label_map[cls]) for p in picked])

        n_new = len(selected_files)

        # Add indices and copy files + create annotations for this step.
        for i in range(n_new):
            # Assign a new dataset index and image filename for this synthetic image.
            new_idx = current_idx + i
            new_imgnum = current_imgnum + i
            new_filename = img_name(new_imgnum, args.ext)

            # Extend the training split with the new index.
            split["train"].append(new_idx)

            # Copy the image into the target dataset folder with standardized name.
            args.output_images.mkdir(parents=True, exist_ok=True)
            src = selected_files[i][0]
            dst = args.output_images / new_filename
            if dst.exists():
                raise FileExistsError(f"Refusing to overwrite: {dst}")
            shutil.copy2(str(src), str(dst))

            # Write a minimal Labelme-style annotation JSON.
            label = selected_files[i][1]
            args.output_annotations.mkdir(parents=True, exist_ok=True)
            image_path = new_filename
            ann_path = (
                args.output_annotations / Path(new_filename).with_suffix(".json").name
            )
            if ann_path.exists():
                raise FileExistsError(f"Refusing to overwrite: {ann_path}")
            # Read width/height from the copied image.
            with Image.open(dst) as img:
                width, height = img.size
            # Create and save the Labelme JSON payload.
            payload = _labelme_payload(label, image_path, width, height)
            save_json(payload, ann_path)

        # Update counters for the next step.
        current_idx += n_new
        current_imgnum += n_new

        # Write the split JSON for this step.
        split_path = out_dir / f"split_step{step_num}.json"
        save_json(split, split_path)
        print(f"Step {step_num}: Saved {split_path}")
        print(
            f"  Added {n_new} new train indices: {current_idx - n_new}..{current_idx - 1}"
        )

    print(f"\nTotal images added across all steps: {current_idx - args.start_idx}")


if __name__ == "__main__":
    main()
