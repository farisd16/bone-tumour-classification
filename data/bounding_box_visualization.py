import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from random import sample
from tumour_bounding_box import bounding_box_creator



# === Paths ===
base_dir = os.path.dirname(__file__)
json_folder = os.path.join(base_dir,"dataset", "BTXRD", "Annotations")
image_folder = os.path.join(base_dir,"dataset", "BTXRD", "images")
patched_dataset = os.path.join(base_dir,"dataset","patched_BTXRD")


# === Collect all JSON files ===
json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

classes = [
    "osteochondroma",
    "osteosarcoma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "synovial osteochondroma",
    "osteofibroma",
]

# Loop through dataset 
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
        if s["shape_type"] == "polygon":
            cv2.polylines(overlay, [pts], True, (255, 0, 0), 3)
            cv2.fillPoly(overlay, [pts], (255, 0, 0, 50))
        elif s["shape_type"] == "rectangle":
            (x1, y1), (x2, y2) = pts
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    # Combine all shapes’ points into one big array 
    all_pts = np.concatenate(all_pts, axis=0)

    x1,y1,x2,y2 = bounding_box_creator(all_pts, original_image=overlay , label = label, margin=0.10)

    # Tumour region after margin
    w, h = x2 - x1, y2 - y1

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 3)

    # Label text
    cv2.putText(
        overlay,
        label,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )

    # Blend overlay
    alpha = 0.4
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    

    # Plot
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(blended)
    plt.xlabel(f"{image_name}")
    plt.title(f"Class: {label},\n Original image shape: {overlay.shape}, Tumour bounding box shape: {w, h}")
    
    plt.subplot(122)
    image_patched_path = os.path.join(patched_dataset,image_name)
    image_patched = cv2.imread(image_patched_path)
    plt.imshow(image_patched)
    plt.xlabel(f"{image_name}")
    plt.title(f"Extracted patch shape: {w, h}")
    plt.show()