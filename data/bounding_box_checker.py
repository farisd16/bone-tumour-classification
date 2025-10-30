import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from random import sample
import pandas as pd
from tumour_bounding_box import bounding_box_creator


# === Paths ===
base_dir = os.path.dirname(__file__)
json_folder = os.path.join(base_dir, "BTXRD", "Annotations")
image_folder = os.path.join(base_dir, "BTXRD", "images")

# === Collect all JSON files ===
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

unsquared_images = []

# === Loop through dataset ===
for json_name in json_files:

    json_path = os.path.join(json_folder, json_name)
    image_name = json_name.replace(".json", ".jpeg")
    image_path = os.path.join(image_folder, image_name)

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Get label from first shape
    label = data["shapes"][0]["label"].lower()

    # Skip if label not in target classes
    if label not in classes:
        continue

    # --- Load and visualize ---
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = image.copy()


    # To store all of the tumour points
    all_pts = []

    for s in data["shapes"]:    
        pts = np.array(s["points"], np.int32)
        all_pts.append(pts)

    # Combine all shapes’ points into one big array 
    all_pts = np.concatenate(all_pts, axis=0)
    
    x1,y1,x2,y2 = bounding_box_creator(all_pts, original_image=overlay , label = label, margin=0.10)

    # Tumour region after margin
    w, h = x2 - x1, y2 - y1

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 3)

    # Label text
    cv2.putText(overlay,label,(x1, max(0, y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0),2,)

    # Blend overlay
    alpha = 0.4
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Check if bounding box exceeds image boundaries
    H_orig, W_orig, _ = overlay.shape
    out_of_bounds = x1 < 0 or y1 < 0 or x2 > W_orig or y2 > H_orig

    if out_of_bounds:
        unsquared_images.append({
            "filename": json_name,
            "label": label,
            "Original_image_width": W_orig,
            "Original_image_height": H_orig,
            "x1_tumour": x1,
            "y1_tumour": y1,
            "x2_tumour": x2,
            "y2_tumour": y2,
            "box_width": w,
            "box_height": h
        })


# After the loop 
print("\n=== SUMMARY ===")
print(f"Total images checked: {len(json_files)}")
print(f"Images with bounding box exceeding bounds: {len(unsquared_images)}")

# === Save CSV report ===
csv_path = os.path.join(base_dir, "after_bounding_box_issues.csv")
df = pd.DataFrame(unsquared_images)
df.to_csv(csv_path, index=False)
print(f"\nReport saved to: {csv_path}")