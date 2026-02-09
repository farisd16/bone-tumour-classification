import json
import os
import argparse
import shutil

import pandas as pd

from custom_dataset_class import CustomDataset

parser = argparse.ArgumentParser(
    description="Generate metadata.jsonl for HuggingFace ImageFolder dataset"
)
parser.add_argument(
    "--image-dir",
    type=str,
    default="./data/dataset/final_patched_BTXRD",
    help="Path to the folder containing images",
)
parser.add_argument(
    "--xlsx-path",
    type=str,
    default="./data/dataset/BTXRD/dataset.xlsx",
    help="Path to xlsx file containing metadata",
)
parser.add_argument(
    "--annotations-dir",
    type=str,
    default="./data/dataset/BTXRD/Annotations",
    help="Path to JSON annotations folder (for tumor subtype)",
)
parser.add_argument(
    "--split-path",
    type=str,
    default="./data/dataset/splits/dataset_split_final.json",
    help="Path to JSON file containing split indices (with 'train', 'val', 'test' keys)",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./data/dataset/hf_dataset",
    help="Output directory for the HF dataset (defaults to hf_<dataset_name> in parent of image-dir)",
)
args = parser.parse_args()

# === Column definitions ===
BODY_PARTS = ["upper limb", "lower limb", "pelvis"]

ANATOMICAL_LOCATIONS = [
    "hand",
    "ulna",
    "radius",
    "humerus",
    "foot",
    "tibia",
    "fibula",
    "femur",
    "hip-bone",
    "ankle-joint",
    "knee-joint",
    "hip-joint",
    "wrist-joint",
    "elbow-joint",
    "shoulder-joint",
]

VIEWS = ["frontal", "lateral", "oblique"]


# === Helper to normalize column names ===
def normalize_col_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


# === Load xlsx metadata ===
xlsx_df = pd.read_excel(args.xlsx_path)

# Build mapping from normalized column names to original
normalized_to_original = {
    normalize_col_name(c): c for c in xlsx_df.columns if str(c).strip() != ""
}

# Find image_id column
image_id_col = normalized_to_original.get("imageid")
if image_id_col is None:
    raise ValueError("Expected an 'image_id' column in xlsx file")


# Helper function to generate metadata entry for an image
def generate_metadata_entry(image_path):
    image_file = os.path.basename(image_path)
    image_key = image_file.strip().lower()

    # body_part = body_part_by_image.get(image_key, "unknown")
    anatomical_location = anatomical_location_by_image.get(image_key, "unknown")
    tumor_subtype = tumor_by_image.get(image_key, "unknown")
    view = view_by_image.get(image_key, "unknown")

    text = f"X-ray image of {tumor_subtype} in the {anatomical_location}, {view} view"
    return {"file_name": image_file, "text": text}


def get_one_hot_value(row, locations, normalized_to_original):
    """Find which location has value 1 in the one-hot encoded columns."""
    for loc in locations:
        original_col = normalized_to_original.get(normalize_col_name(loc))
        if original_col is None:
            continue
        value = row.get(original_col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        try:
            if int(value) == 1:
                return loc
        except (TypeError, ValueError):
            continue
    return "unknown"


# === Build lookup dictionaries ===
body_part_by_image = {}
anatomical_location_by_image = {}
view_by_image = {}

for _, row in xlsx_df.iterrows():
    image_id_raw = row.get(image_id_col)
    if image_id_raw is None or (
        isinstance(image_id_raw, float) and pd.isna(image_id_raw)
    ):
        continue
    image_id = os.path.basename(str(image_id_raw).strip()).lower()
    if not image_id:
        continue

    body_part_by_image[image_id] = get_one_hot_value(
        row, BODY_PARTS, normalized_to_original
    )
    anatomical_location_by_image[image_id] = get_one_hot_value(
        row, ANATOMICAL_LOCATIONS, normalized_to_original
    )

    view_by_image[image_id] = get_one_hot_value(row, VIEWS, normalized_to_original)


# === Load tumor subtypes from JSON annotations ===
tumor_by_image = {}
for json_file in os.listdir(args.annotations_dir):
    if not json_file.endswith(".json"):
        continue
    json_path = os.path.join(args.annotations_dir, json_file)
    with open(json_path, "r") as f:
        data = json.load(f)
    image_name = json_file.replace(".json", ".jpeg").lower()
    label = data["shapes"][0]["label"].lower()
    tumor_by_image[image_name] = label


# === Load split indices from JSON file ===
with open(args.split_path, "r") as f:
    split_indices = json.load(f)

output_dir = args.output_dir

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

base_dataset = CustomDataset(
    image_dir=args.image_dir,
    json_dir=args.annotations_dir,
    transform=None,
)

# Process train split
train_metadata = []
for idx in split_indices["train"]:
    img_path, label = base_dataset.samples[idx]
    image_file = os.path.basename(img_path)
    # Copy image to train directory
    shutil.copy2(img_path, os.path.join(train_dir, image_file))
    # Generate metadata
    train_metadata.append(generate_metadata_entry(img_path))

# Process test split
test_metadata = []
for idx in split_indices["val"]:
    img_path, label = base_dataset.samples[idx]
    image_file = os.path.basename(img_path)
    # Copy image to test directory
    shutil.copy2(img_path, os.path.join(test_dir, image_file))
    # Generate metadata
    test_metadata.append(generate_metadata_entry(img_path))

for idx in split_indices["test"]:
    img_path, label = base_dataset.samples[idx]
    image_file = os.path.basename(img_path)
    # Copy image to test directory
    shutil.copy2(img_path, os.path.join(test_dir, image_file))
    # Generate metadata
    test_metadata.append(generate_metadata_entry(img_path))

# Write metadata.jsonl files
train_metadata_path = os.path.join(train_dir, "metadata.jsonl")
with open(train_metadata_path, "w") as f:
    for entry in train_metadata:
        f.write(json.dumps(entry) + "\n")

test_metadata_path = os.path.join(test_dir, "metadata.jsonl")
with open(test_metadata_path, "w") as f:
    for entry in test_metadata:
        f.write(json.dumps(entry) + "\n")

# Save split indices for reproducibility
split_info_path = os.path.join(output_dir, "split_info.json")
with open(split_info_path, "w") as f:
    json.dump(
        {
            "train": split_indices["train"],
            "test": split_indices["val"] + split_indices["test"],
            "source_split_file": args.split_path,
        },
        f,
        indent=2,
    )

print(f"✓ Created HuggingFace dataset structure at: {output_dir}")
print(f"  - Train: {len(train_metadata)} images → {train_dir}")
print(f"  - Test: {len(test_metadata)} images → {test_dir}")
print(f"  - Split info saved to: {split_info_path}")

# Print sample entries
print("\nSample train entries:")
for entry in train_metadata[:3]:
    print(f"  {entry}")
print("\nSample test entries:")
for entry in test_metadata[:3]:
    print(f"  {entry}")
