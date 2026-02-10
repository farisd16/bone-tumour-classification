"""
Prepare DreamBooth dataset folder structure from BTXRD dataset.

Creates:
    dreambooth/
    ├── healthy/                 # 200 healthy X-rays from images folder
    ├── osteochondroma/
    ├── osteosarcoma/
    ├── multiple_osteochondromas/
    ├── simple_bone_cyst/
    ├── giant_cell_tumor/
    ├── synovial_osteochondroma/
    └── osteofibroma/
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Mapping from annotation labels to standardized folder names
LABEL_TO_FOLDER = {
    "osteochondroma": "osteochondroma",
    "osteosarcoma": "osteosarcoma",
    "multiple osteochondromas": "multiple_osteochondromas",
    "simple bone cyst": "simple_bone_cyst",
    "giant cell tumor": "giant_cell_tumor",
    "synovial osteochondroma": "synovial_osteochondroma",
    "osteofibroma": "osteofibroma",
}

# Labels to skip (not tumor classes we want)
SKIP_LABELS = {"other mt", "other", "normal", "healthy"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare DreamBooth dataset from BTXRD"
    )
    parser.add_argument(
        "--tumor_images_dir",
        type=str,
        default="data/dataset/BTXRD/final_patched_BTXRD",
        help="Directory containing tumor images",
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="data/dataset/BTXRD/Annotations",
        help="Directory containing annotation JSON files",
    )
    parser.add_argument(
        "--healthy_images_dir",
        type=str,
        default="data/dataset/BTXRD/images",
        help="Directory containing healthy/all images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/dataset/dreambooth",
        help="Output directory for DreamBooth dataset",
    )
    parser.add_argument(
        "--healthy_start",
        type=int,
        default=3000,
        help="Starting image number for healthy images (default: 3000)",
    )
    parser.add_argument(
        "--num_healthy",
        type=int,
        default=200,
        help="Number of healthy images to copy (default: 200)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without copying files",
    )
    return parser.parse_args()


def get_tumor_type_from_annotation(annotation_path):
    """Extract tumor type(s) from annotation JSON file."""
    try:
        with open(annotation_path, "r") as f:
            data = json.load(f)

        tumor_types = set()
        for shape in data.get("shapes", []):
            label = shape.get("label", "").lower().strip()
            if label in SKIP_LABELS:
                continue
            if label in LABEL_TO_FOLDER:
                tumor_types.add(LABEL_TO_FOLDER[label])
            else:
                print(f"Warning: Unknown label '{label}' in {annotation_path}")

        return tumor_types
    except Exception as e:
        print(f"Error reading {annotation_path}: {e}")
        return set()


def main():
    args = parse_args()

    tumor_images_dir = Path(args.tumor_images_dir)
    annotations_dir = Path(args.annotations_dir)
    healthy_images_dir = Path(args.healthy_images_dir)
    output_dir = Path(args.output_dir)

    # Create output directories
    all_folders = list(set(LABEL_TO_FOLDER.values())) + ["healthy"]
    for folder in all_folders:
        folder_path = output_dir / folder
        if not args.dry_run:
            folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder_path}")

    # Statistics
    stats = defaultdict(int)

    # Process tumor images
    print("\n" + "=" * 60)
    print("Processing tumor images...")
    print("=" * 60)

    tumor_images = (
        sorted(tumor_images_dir.glob("*.jpeg"))
        + sorted(tumor_images_dir.glob("*.jpg"))
        + sorted(tumor_images_dir.glob("*.png"))
    )

    for image_path in tqdm(tumor_images, desc="Processing tumor images"):
        # Get corresponding annotation
        image_name = image_path.stem  # e.g., "IMG000100"
        annotation_path = annotations_dir / f"{image_name}.json"

        if not annotation_path.exists():
            print(f"Warning: No annotation for {image_path.name}")
            continue

        tumor_types = get_tumor_type_from_annotation(annotation_path)

        if not tumor_types:
            stats["skipped_no_valid_label"] += 1
            continue

        # Copy image to each tumor type folder (an image might have multiple labels)
        for tumor_type in tumor_types:
            dest_folder = output_dir / tumor_type
            dest_path = dest_folder / image_path.name

            if not args.dry_run:
                shutil.copy2(image_path, dest_path)

            stats[tumor_type] += 1

    # Process healthy images
    print("\n" + "=" * 60)
    print("Processing healthy images...")
    print("=" * 60)

    healthy_count = 0
    for i in range(args.healthy_start, args.healthy_start + args.num_healthy):
        image_name = f"IMG{i:06d}.jpeg"
        image_path = healthy_images_dir / image_name

        if not image_path.exists():
            # Try .jpg extension
            image_name = f"IMG{i:06d}.jpg"
            image_path = healthy_images_dir / image_name

        if not image_path.exists():
            print(f"Warning: Healthy image not found: {image_name}")
            continue

        dest_path = output_dir / "healthy" / image_path.name

        if not args.dry_run:
            shutil.copy2(image_path, dest_path)

        healthy_count += 1

    stats["healthy"] = healthy_count

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for folder in sorted(stats.keys()):
        print(f"  {folder}: {stats[folder]} images")
    print("=" * 60)

    if args.dry_run:
        print("\nDRY RUN - no files were copied")
    else:
        print(f"\nDataset created at: {output_dir}")


if __name__ == "__main__":
    main()
