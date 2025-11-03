import json
import cv2
import numpy as np
import os
from tumour_bounding_box import bounding_box_creator


"""
This code extracts the patched dataset without 106 images
"""

# === Paths ===

base_dir = os.path.dirname(__file__)
json_folder = os.path.join(base_dir, "dataset", "BTXRD", "Annotations")
image_folder = os.path.join(base_dir, "dataset", "BTXRD", "images")
output_folder = os.path.join(base_dir, "dataset", "patched_BTXRD")


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

i = 0

# === Loop through dataset ===
for json_name in json_files:
    json_path = os.path.join(json_folder, json_name)
    image_name = json_name.replace(".json", ".jpeg")
    image_path = os.path.join(image_folder, image_name)

    # --- Load JSON and image ---
    with open(json_path, "r") as f:
        data = json.load(f)
    
    image = cv2.imread(image_path)

    H,W, _ = image.shape

    label = data["shapes"][0]["label"].lower()
    if label not in classes:
        continue

    # --- Collect all tumour points ---
    all_pts = []
    for s in data["shapes"]:
        pts = np.array(s["points"], np.int32)
        all_pts.append(pts)

        # --- Combine all shapes’ points into one big array ---
    all_pts = np.concatenate(all_pts, axis=0)

    x1,y1,x2,y2 = bounding_box_creator(all_pts, original_image=image , label = label, margin=0.10)

    if x1 < 0 or y1 < 0 or x2 > W or y2 > H or x2 <= x1 or y2 <= y1:
        i += 1
        print(f"[SKIPPED] Out-of-bounds or invalid box in: {json_name}")
        continue
    
    w, h = x2 - x1, y2 - y1

    patch = image[y1:y2, x1:x2]
    patch_filename = os.path.join(output_folder, image_name)
    cv2.imwrite(patch_filename, patch)

print(f"Files saved to {output_folder}")
print(i)
