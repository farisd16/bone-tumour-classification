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
    if image is None:
        print(f"⚠️ Could not load image: {image_name}")
        continue

    label = data["shapes"][0]["label"].lower()
    if label not in classes:
        continue

    # --- Collect all tumour points ---
    all_pts = []
    for s in data["shapes"]:
        pts = np.array(s["points"], np.int32)
        all_pts.append(pts)

    all_pts = np.concatenate(all_pts, axis=0)
    x_min, x_max = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
    y_min, y_max = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])

    # --- Initial box with 10% margin (IDENTICAL to visualizer) ---
    margin = 0.10
    w, h = x_max - x_min, y_max - y_min
    size = max(w, h) * (1 + margin)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2  # <-- float, but truncated later

    x1, y1 = int(cx - size / 2), int(cy - size / 2)
    x2, y2 = int(cx + size / 2), int(cy + size / 2)

    H, W, _ = image.shape

    # --- Clip to image boundaries first (IDENTICAL) ---
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    # --- Compute new width & height ---
    w = x2 - x1
    h = y2 - y1

    # --- Make square by expanding to larger side ---
    if w != h:
        side = max(w, h)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Recalculate a centered square
        x1 = int(cx - side / 2)
        y1 = int(cy - side / 2)
        x2 = int(cx + side / 2)
        y2 = int(cy + side / 2)

        # --- Adjust if box goes beyond image boundaries ---
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > W:
            diff = x2 - W
            x1 -= diff
            x2 = W
        if y2 > H:
            diff = y2 - H
            y1 -= diff
            y2 = H

    # --- Final safety clip (just in case) ---
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    # --- Extract and save patch ---
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        print(f"Empty patch for {json_name}: ({x1},{y1},{x2},{y2})")
        continue

    patch_filename = os.path.join(output_folder, image_name)
    cv2.imwrite(patch_filename, patch)

print("Patch extraction complete (visualizer-equivalent).")
