import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from random import sample

# === Paths ===
original_image_folder = "/Users/bartu/Desktop/Bartu/RCI/3.Semester/ADLM/bone-tumour-classification/data/BTXRD/images"
patched_images_folder = "/Users/bartu/Desktop/Bartu/RCI/3.Semester/ADLM/bone-tumour-classification/data/patched_BTXRD"
json_folder = "/Users/bartu/Desktop/Bartu/RCI/3.Semester/ADLM/bone-tumour-classification/data/BTXRD/Annotations"

# === Collect all patched images ===
patched_image_files = [f for f in os.listdir(patched_images_folder) if f.endswith(".jpeg")]
random_patched_image_files = sample(patched_image_files,20)

# === Iterate over first 10 images ===
for i, image_file in enumerate(random_patched_image_files):
    if i == 20:
        break

    path_patched = os.path.join(patched_images_folder, image_file)
    original_path = os.path.join(original_image_folder, image_file)
    json_path = os.path.join(json_folder, image_file.replace(".jpeg", ".json"))

    # === Load JSON data ===
    with open(json_path, "r") as f:
        data = json.load(f)

    label = data["shapes"][0]["label"]
    print(f"Image {image_file}: {label}")

    # === Load images ===
    orig = cv2.imread(original_path)
    patched_image = cv2.imread(path_patched)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    patched_image = cv2.cvtColor(patched_image, cv2.COLOR_BGR2RGB)

    # === Handle multi-shape case ===
    if label == "multiple osteochondromas":
        all_pts = []
        for s in data["shapes"]:
            if s["label"].lower() == "multiple osteochondromas":
                pts = np.array(s["points"], np.int32)
                all_pts.append(pts)

                overlay = orig.copy()
                if s["shape_type"] == "polygon":
                    cv2.fillPoly(overlay, [pts], (255, 0, 0))
                elif s["shape_type"] == "rectangle":
                    (x1, y1), (x2, y2) = pts
                    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), -1)

                # Blend transparent overlay
                cv2.addWeighted(overlay, 0.4, orig, 0.6, 0, orig)

        # Combine all shape points to compute bounding box
        all_pts = np.concatenate(all_pts, axis=0)
        x_min, x_max = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
        y_min, y_max = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])

    else:
        all_pts = []
        for s in data["shapes"]:
            pts = np.array(s["points"], np.int32)
            all_pts.append(pts)

            overlay = orig.copy()
            if s["shape_type"] == "polygon":
                cv2.fillPoly(overlay, [pts], (255, 0, 0))
            elif s["shape_type"] == "rectangle":
                (x1, y1), (x2, y2) = pts
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), -1)

            cv2.addWeighted(overlay, 0.4, orig, 0.6, 0, orig)
        
        # Combine all shape points to compute bounding box
        all_pts = np.concatenate(all_pts, axis=0)
        x_min, x_max = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
        y_min, y_max = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])

    # === Bounding box with small margin ===
    margin = 0.10
    w, h = x_max - x_min, y_max - y_min
    size = max(w, h) * (1 + margin)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    x1, y1 = int(cx - size / 2), int(cy - size / 2)
    x2, y2 = int(cx + size / 2), int(cy + size / 2)

    # Clip to image boundaries
    H, W, _ = orig.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W - 1, x2)
    y2 = min(H - 1, y2)

    # Draw bounding box
    cv2.rectangle(orig, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # === Display ===
    print(f"Original shape: {orig.shape}")
    print(f"Patched shape: {patched_image.shape}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(orig)
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(patched_image)
    axs[1].set_title("Extracted Patch")
    axs[1].axis('off')

    plt.show()
