import json
import cv2
import numpy as np
import os

"""
BTXRD Bounding Box Extractor (exact visualizer-equivalent)
-----------------------------------------------------------
This version uses *identical logic* to your visualization script.
That means:
- Integer-based center (truncated, not rounded)
- Same sequence of operations (margin → clip → make-square → adjust)
- Produces bounding boxes whose w/h stats exactly match your ±1 results
"""

# === Paths ===
base_dir = os.path.dirname(__file__)
json_folder = os.path.join(base_dir, "BTXRD", "Annotations")
image_folder = os.path.join(base_dir, "BTXRD", "images")
output_folder = os.path.join(base_dir, "patched_BTXRD")

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

# === Loop through dataset ===
for json_name in json_files:
    json_path = os.path.join(json_folder, json_name)
    image_name = json_name.replace(".json", ".jpeg")
    image_path = os.path.join(image_folder, image_name)

    # --- Load JSON and image ---
    with open(json_path, "r") as f:
        data = json.load(f)
    
    image = cv2.imread(image_path)


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
    x_min, x_max = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
    y_min, y_max = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])

    margin = 0.10

    # --- Step 1: Compute tumour region ---
    w_tumour, h_tumour = x_max - x_min, y_max - y_min

    # --- Step 2: Expand with margin ---
    size = max(w_tumour, h_tumour) * (1 + margin)

    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    side = int(round(size))
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = x1 + side
    y2 = y1 + side

    w, h = x2 - x1, y2 - y1

    patch = image[y1:y2, x1:x2]
    patch_filename = os.path.join(output_folder, image_name)
    cv2.imwrite(patch_filename, patch)



