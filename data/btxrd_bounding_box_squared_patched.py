import json
import cv2
import numpy as np
import os
from tumour_bounding_box import bounding_box_creator
import pandas as pd

"""
This code extracts the bounding boxes for the problematic 106 images 
"""

# === Paths ===
base_dir = os.path.dirname(__file__)
json_folder = os.path.join(base_dir, "dataset","BTXRD", "Annotations")
image_folder = os.path.join(base_dir, "dataset","BTXRD", "images")

# For the unsquared 106 images
squared_image_folder = os.path.join(base_dir, "dataset", "squared_padded")
squared_output_folder = os.path.join(base_dir,"dataset", "squared_patched_106")
csv_path = os.path.join(base_dir, "after_bounding_box_issues.csv")
df = pd.read_csv(csv_path)
squared_json_files = sorted(list(df["filename"]))
padding_info = pd.read_csv(os.path.join(squared_image_folder, "padding_info.csv"))

os.makedirs(squared_output_folder, exist_ok=True)

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
for json_name in squared_json_files:
    json_path = os.path.join(json_folder, json_name)
    image_name = json_name.replace(".json", ".jpeg")
    image_path = os.path.join(squared_image_folder, image_name)

    # --- Load JSON and image ---
    with open(json_path, "r") as f:
        data = json.load(f)
    
    image = cv2.imread(image_path)

    H,W, _ = image.shape

    label = data["shapes"][0]["label"].lower()
    if label not in classes:
        continue

    # Get this image's padding offsets 
    pad_row = padding_info[padding_info["filename"] == image_name]
    pad_left = int(pad_row["pad_left"].values[0])
    pad_top = int(pad_row["pad_top"].values[0])
    
    # --- Collect all tumour points ---
    all_pts = []
    for s in data["shapes"]:
        pts = np.array(s["points"], np.int32)
        pts_padded = pts + np.array([pad_left, pad_top])  
        all_pts.append(pts_padded)


    # Combine all shapes’ points into one big array 
    all_pts = np.concatenate(all_pts, axis=0)

    x1,y1,x2,y2 = bounding_box_creator(all_pts, original_image=image , label = label, margin=0.10)

    if x1 < 0 or y1 < 0 or x2 > W or y2 > H or x2 <= x1 or y2 <= y1:
        i += 1
        print(f"[SKIPPED] Out-of-bounds or invalid box in: {json_name}")
        continue
    
    w, h = x2 - x1, y2 - y1

    patch = image[y1:y2, x1:x2]
    patch_filename = os.path.join(squared_output_folder, image_name)
    cv2.imwrite(patch_filename, patch)


print(i)