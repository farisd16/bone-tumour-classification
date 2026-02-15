from pathlib import Path
from PIL import Image
import json
import re
import argparse

TARGET_SIZE_DEFAULT = 512

# Regex to extract 4-digit index from "IMGxxxxxx.jpeg"
INDEX_PATTERN = re.compile(r'IMG0*(\d{4})\.jpeg$', re.IGNORECASE)


def resize_train_split(src: Path, dst: Path, train_indices: list, target_size: int):
    """
    Resize only images whose index (extracted from IMGxxxxxx.jpeg)
    is in train_indices.
    """
    dst.mkdir(parents=True, exist_ok=True)
    count, skipped, no_match = 0, 0, 0

    train_set = set(str(idx).zfill(4) for idx in train_indices)

    all_jpegs = list(src.rglob("*.jpeg"))
    print(f"Total JPEGs: {len(all_jpegs)}")

    matching = [
        INDEX_PATTERN.search(p.name).group(1)
        for p in all_jpegs
        if INDEX_PATTERN.search(p.name)
    ]

    print(f"Matching indices in source: {set(matching)}")
    print(f"Train indices missing in source: {train_set - set(matching)}")

    for p in src.rglob("*.jpeg"):
        match = INDEX_PATTERN.search(p.name)
        if not match:
            no_match += 1
            continue

        img_index = match.group(1)
        if img_index not in train_set:
            continue

        rel = p.relative_to(src)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                img = img.resize((target_size, target_size), Image.BICUBIC)
                img.save(out_path)
                count += 1
        except Exception as e:
            print(f"Skipping bad image: {p} | error: {e}")
            skipped += 1

    print(f"Resized {count} train-split images.")
    print(f"(skipped {skipped} bad, {no_match} non-matching filenames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_src", type=Path, required=True)
    parser.add_argument("--split_json", type=Path, required=True)
    parser.add_argument("--real_dst", type=Path, required=True)
    parser.add_argument("--size", type=int, default=TARGET_SIZE_DEFAULT)

    args = parser.parse_args()

    with open(args.split_json) as f:
        split_data = json.load(f)

    train_indices = split_data["train"]
    print(f"Loaded {len(train_indices)} train indices.")

    resize_train_split(
        src=args.real_src,
        dst=args.real_dst,
        train_indices=train_indices,
        target_size=args.size
    )

    print("Done.")
    print(f"Output folder: {args.real_dst}")


if __name__ == "__main__":
    main()
