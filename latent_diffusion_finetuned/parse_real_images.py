import os
import shutil
import pandas as pd

# ---------------- CONFIG ----------------
IMAGE_DIR = os.environ.get("IMAGE_DIR", "./real_images")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./organized_real_images")
EXCEL_PATH = os.environ.get("EXCEL_PATH", "./dataset.xlsx")

IMAGE_EXT = ".jpeg"
# ---------------------------------------

# Tumor subtype columns → normalized folder name
SUBTYPE_COLUMNS = {
    "osteochondroma": "osteochondroma",
    "multiple osteochondromas": "multiple_osteochondromas",
    "simple bone cyst": "simple_bone_cyst",
    "giant cell tumor": "giant_cell_tumor",
    "osteofibroma": "osteofibroma",
    "synovial osteochondroma": "synovial_osteochondroma",
    "osteosarcoma": "osteosarcoma",
}

# Gross location columns → normalized name
LOCATION_COLUMNS = {
    "upper limb": "upper_limb",
    "lower limb": "lower_limb",
    "pelvis": "pelvis",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Excel
df = pd.read_excel(EXCEL_PATH)

# Ensure image_id is string-like
df["image_id"] = df["image_id"].astype(str)

def get_single_positive(row, column_map):
    """Return mapped value if exactly one column == 1, else None"""
    positives = [mapped for col, mapped in column_map.items() if row.get(col, 0) == 1]
    if len(positives) == 1:
        return positives[0]
    return None

moved = 0
skipped = 0

for _, row in df.iterrows():
    raw_id = str(row["image_id"]).strip()

    # Remove prefix/suffix if already present
    raw_id = raw_id.replace("IMG", "")
    raw_id = raw_id.replace(".jpeg", "").replace(".jpg", "")

    image_id = raw_id.zfill(6)
    img_name = f"IMG{image_id}{IMAGE_EXT}"
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not os.path.isfile(img_path):
        print(f"Missing image: {img_name}")
        skipped += 1
        continue

    subtype = get_single_positive(row, SUBTYPE_COLUMNS)
    location = get_single_positive(row, LOCATION_COLUMNS)

    if subtype is None or location is None:
        print(
            f"Skipped {img_name}: "
            f"subtype={subtype}, location={location}"
        )
        skipped += 1
        continue

    target_dir = os.path.join(OUTPUT_DIR, f"{subtype}_{location}")
    os.makedirs(target_dir, exist_ok=True)

    shutil.move(img_path, os.path.join(target_dir, img_name))
    moved += 1

    print(f"Moved {img_name} → {target_dir}")

print("\nDone.")
print(f"Moved:   {moved}")
print(f"Skipped: {skipped}")
