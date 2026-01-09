import json
import argparse
from pathlib import Path
import shutil

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

def img_name(n: int, ext: str):
    return f"IMG{n:06d}.{ext}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True, help="split.json input")
    ap.add_argument("--out", type=Path, default=None, help="output json (default: same folder)")
    ap.add_argument("--n", type=int, default=None, help="how many new train indices to add")
    ap.add_argument("--gen-dir", type=Path, default=None, help="folder with generated images (optional)")
    ap.add_argument("--start-idx", type=int, required=True,help="next index to start (e.g. 1707)")
    ap.add_argument("--start-img", type=int, required=True, help="next image number (e.g. 1868 for IMG001868)")
    ap.add_argument("--ext", default="jpeg")
    ap.add_argument("--rename", action="store_true", help="rename files in --gen-dir")
    args = ap.parse_args()

    split = load_json(args.split)

    if args.out is None:
        args.out = args.split.parent / (args.split.stem + "_new.json")

    # Number of items
    if args.gen_dir is not None:
        files = sorted([p for p in args.gen_dir.iterdir() if p.suffix.lower() in ".jpeg"])
        n_new = len(files)
        if n_new == 0:
            raise ValueError("No images found in gen-dir.")
    else:
        if args.n is None or args.n <= 0:
            raise ValueError("Use --n > 0 OR provide --gen-dir.")
        files = []
        n_new = args.n

    # add indices + (optional) rename files
    for i in range(n_new):
        new_idx = args.start_idx + i
        new_imgnum = args.start_img + i
        new_filename = img_name(new_imgnum, args.ext)

        split["train"].append(new_idx)

        if args.rename and args.gen_dir is not None:
            src = files[i]
            dst = args.gen_dir / new_filename
            if dst.exists():
                raise FileExistsError(f"Refusing to overwrite: {dst}")
            shutil.move(str(src), str(dst))

    save_json(split, args.out)
    print(f"Saved: {args.out}")
    print(f"Added {n_new} new train indices: {args.start_idx}..{args.start_idx + n_new - 1}")

if __name__ == "__main__":
    main()
