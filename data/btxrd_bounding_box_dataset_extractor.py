import json
import cv2
import numpy as np
import os
import pandas as pd
from tumour_bounding_box import bounding_box_creator


"""
Unified script for extracting bounding-box patches 
from both normal and padded (106) BTXRD images.
"""


# === Paths ===
base_dir = os.path.dirname(__file__)
json_folder = os.path.join(base_dir, "dataset", "BTXRD", "Annotations")
image_folder = os.path.join(base_dir, "dataset", "BTXRD", "images")
output_folder = os.path.join(
    base_dir,
    "dataset",
    "final_patched_BTXRD",
)

# Special 106 padded images
squared_image_folder = os.path.join(base_dir, "dataset", "squared_padded")
csv_path = os.path.join(base_dir, "after_bounding_box_issues.csv")
padding_info_path = os.path.join(squared_image_folder, "padding_info.csv")

# === Load metadata for 106 images ===
df_106 = pd.read_csv(csv_path)
padding_info = pd.read_csv(padding_info_path)
squared_json_files = set(df_106["filename"])  # for quick lookup

# === Prepare output ===
os.makedirs(output_folder, exist_ok=True)

# === Collect JSON files ===
json_files = sorted([f for f in os.listdir(json_folder) if f.endswith(".json")])

classes = [
    "osteochondroma",
    "osteosarcoma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "synovial osteochondroma",
    "osteofibroma",
]

skipped = 0
processed = 0

# === Loop through all JSON files ===
for json_name in json_files:
    json_path = os.path.join(json_folder, json_name)
    image_name = json_name.replace(".json", ".jpeg")

    # Check if image belongs to the 106 special ones
    if json_name in squared_json_files:
        image_path = os.path.join(squared_image_folder, image_name)
        use_padding = True
    else:
        image_path = os.path.join(image_folder, image_name)
        use_padding = False

    # --- Load JSON and image ---
    if not os.path.exists(image_path):
        print(f"[MISSING] Image not found: {image_path}")
        continue

    with open(json_path, "r") as f:
        data = json.load(f)
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        continue

    H, W, _ = image.shape
    label = data["shapes"][0]["label"].lower()
    if label not in classes:
        continue

    # --- Collect tumour points ---
    all_pts = []
    for s in data["shapes"]:
        pts = np.array(s["points"], np.int32)

        # If padded, apply offset correction
        if use_padding:
            pad_row = padding_info[padding_info["filename"] == image_name]
            if not pad_row.empty:
                pad_left = int(pad_row["pad_left"].values[0])
                pad_top = int(pad_row["pad_top"].values[0])
                pts = pts + np.array([pad_left, pad_top])

        all_pts.append(pts)

    # --- Combine points ---
    all_pts = np.concatenate(all_pts, axis=0)

    x1, y1, x2, y2 = bounding_box_creator(
        all_pts, original_image=image, label=label, margin=0.10
    )

    # --- Skip invalid bounding boxes ---
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H or x2 <= x1 or y2 <= y1:
        skipped += 1
        print(f"[SKIPPED] Invalid box in: {json_name}")
        continue

    # --- Crop and save patch ---
    patch = image[y1:y2, x1:x2]
    patch_filename = os.path.join(output_folder, image_name)
    cv2.imwrite(patch_filename, patch)
    processed += 1

print(f"\n Finished processing!")
print(f"→ Saved {processed} patched images to: {output_folder}")
print(f"→ Skipped {skipped} invalid or missing entries.\n")
