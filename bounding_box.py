import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# === Paths ===
json_folder = "/Users/bartu/Desktop/Bartu/RCI/3.Semester/ADLM/repo/BTXRD/Annotations/"
image_folder = "/Users/bartu/Desktop/Bartu/RCI/3.Semester/ADLM/repo/BTXRD/images/"


# === Collect all JSON files ===
json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

# === Pick 10 random ones ===
random_jsons = random.sample(json_files, 10)

classes = ["osteochondroma", "osteosarcoma", "multiple osteochondromas", "simple bone cyst", "giant cell tumor", "synovial osteochondroma", "osteofibroma"]

# === Keep track of which classes we've already shown ===
shown_classes = set()

# === Collect all JSON files ===
json_files = sorted([f for f in os.listdir(json_folder) if f.endswith(".json")])


# === Loop through dataset ===
for json_name in json_files:
    json_path = os.path.join(json_folder, json_name)
    image_name = json_name.replace(".json", ".jpeg")
    image_path = os.path.join(image_folder, image_name)

    # Skip missing images
    if not os.path.exists(image_path):
        continue

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Get label from first shape
    label = data["shapes"][0]["label"].lower()

    # Skip if label not in target classes
    if label not in classes:
        continue

    # Skip if we've already shown this class
    if label in shown_classes:
        continue

    # --- Load and visualize ---
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = image.copy()

    # Draw shapes
    for shape in data["shapes"]:
        shape_label = shape["label"]
        pts = np.array(shape["points"], np.int32)
        shape_type = shape["shape_type"]

        # Bounding box around polygon
        x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
        y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])

        if shape_type == "polygon":
            cv2.polylines(overlay, [pts], True, (255, 0, 0), 3)
            cv2.fillPoly(overlay, [pts], (255, 0, 0, 50))
        elif shape_type == "rectangle":
            (x1, y1), (x2, y2) = pts
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    # Enlarged bounding box (yellow)
    margin = 0.20
    w, h = x_max - x_min, y_max - y_min
    size = max(w, h) * (1 + margin)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    x1, y1 = int(cx - size / 2), int(cy - size / 2)
    x2, y2 = int(cx + size / 2), int(cy + size / 2)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 3)

    # Label text
    cv2.putText(
        overlay, label, (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2
    )

    # Blend overlay
    alpha = 0.4
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(blended)
    plt.axis("off")
    plt.title(f"Class: {label}")
    plt.show()

    # Mark this class as shown
    shown_classes.add(label)

    # Stop once all classes are shown
    if shown_classes == set(classes):
        break