import pandas as pd
import os
import cv2
import numpy as np

# === Paths ===
base_dir = os.path.dirname(__file__)

csv_path = os.path.join(base_dir, "after_bounding_box_issues.csv")
df = pd.read_csv(csv_path)

image_folder = os.path.join(base_dir, "dataset", "BTXRD", "images")
output_folder = os.path.join(base_dir, "dataset", "squared_padded")

os.makedirs(output_folder, exist_ok=True)

# Find largest dimension (width or height) across all 106 images 
max_dim = max(df["Original_image_width"].max(), df["Original_image_height"].max())
target_dim = int(max_dim * 1.05)  # +5% margin
print(f"Largest dimension among images: {max_dim}, Target padded dimension: {target_dim}")

# Track per-image padding values 
padding_records = []

# --- Padding loop ---
for _, row in df.iterrows():
    filename = row["filename"].replace(".json", ".jpeg")
    img_path = os.path.join(image_folder, filename)

    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    # Compute how much padding is needed to reach target_dim 
    pad_vert = max(0, target_dim - H)
    pad_horiz = max(0, target_dim - W)

    # Apply padding symmetrically (centered) 
    pad_top = pad_vert // 2
    pad_bottom = pad_vert - pad_top
    pad_left = pad_horiz // 2
    pad_right = pad_horiz - pad_left

    # --- Apply padding ---
    img_padded = cv2.copyMakeBorder(
        img,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, img_padded)

    # Record padding info for this image 
    padding_records.append({
        "filename": filename,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom
    })

    print(f"Padded: {filename} | New size: {img_padded.shape[1]}x{img_padded.shape[0]} | "
          f"Padding (L,T,R,B): {pad_left},{pad_top},{pad_right},{pad_bottom}")

# Save all padding info for later 
padding_df = pd.DataFrame(padding_records)
padding_info_path = os.path.join(output_folder, "padding_info.csv")
padding_df.to_csv(padding_info_path, index=False)
print(f"\Saved per-image padding info → {padding_info_path}")
