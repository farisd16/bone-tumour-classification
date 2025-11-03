import os
import sys
import pandas as pd

# --------- CONFIG (relative to project root) ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset", "BTXRD")
EXCEL_PATH = os.path.join(DATASET_DIR, "dataset.xlsx")
CSV_OUTPUT_PATH = os.path.join(DATASET_DIR, "dataset_singlelabel.csv")
PATCHED_DIR = os.path.join(PROJECT_ROOT, "data", "dataset", "patched_BTXRD")

# The seven tumor classes (must match your Excel column names exactly)
TUMOR_COLS = [
    "osteochondroma",
    "osteosarcoma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "synovial osteochondroma",
    "osteofibroma",
]

# Map tumor names to numeric labels 0..6
TUMOR_LABELS = {name: i for i, name in enumerate(TUMOR_COLS)}


def main():
    print(f"[INFO] DATASET_DIR : {DATASET_DIR}")
    print(f"[INFO] EXCEL_PATH  : {EXCEL_PATH}")
    print(f"[INFO] PATCHED_DIR : {PATCHED_DIR}")

    # Basic checks
    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: dataset folder not found: {DATASET_DIR}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(EXCEL_PATH):
        print(f"ERROR: Excel file not found at: {EXCEL_PATH}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(PATCHED_DIR):
        print(f"ERROR: patched images folder not found: {PATCHED_DIR}", file=sys.stderr)
        sys.exit(1)

    # Load annotation spreadsheet
    df = pd.read_excel(EXCEL_PATH)

    # Ensure tumor columns exist
    missing = [c for c in TUMOR_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: Missing tumor columns: {missing}", file=sys.stderr)

    # Keep only image_id + tumor columns that actually exist
    keep_cols = ["image_id"] + [c for c in TUMOR_COLS if c in df.columns]
    out = df[keep_cols].copy()

    # Normalize tumor cols to 0/1 ints
    for c in TUMOR_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # Assign a single label 0..6 for rows with exactly one positive tumor
    def assign_label(row):
        positives = [TUMOR_LABELS[col] for col in TUMOR_COLS if col in row and row[col] == 1]
        if len(positives) == 1:
            return positives[0]    # valid tumor: 0..6
        elif len(positives) == 0:
            return -1              # no tumor -> drop later
        else:
            return -2              # multiple tumors -> drop later

    out["label"] = out.apply(assign_label, axis=1)

    # Keep only valid single-tumor rows (label in 0..6)
    total_rows = len(out)
    out = out[(out["label"] >= 0) & (out["label"] <= 6)]
    after_label_filter = len(out)
    dropped_label = total_rows - after_label_filter

    # Build set of available patched image filenames (lowercased basenames)
    patched_files = {
        fn.lower()
        for fn in os.listdir(PATCHED_DIR)
        if os.path.isfile(os.path.join(PATCHED_DIR, fn))
    }

    def row_basename_lower(img_id: str) -> str:
        # Support values like "images/IMG000123.jpeg" by taking basename
        return os.path.basename(str(img_id)).lower()

    # Filter rows to those whose image_id basename exists in patched_BTXRD
    before_patched_filter = len(out)
    out = out[out["image_id"].apply(lambda x: row_basename_lower(x) in patched_files)]
    after_patched_filter = len(out)
    dropped_missing_patched = before_patched_filter - after_patched_filter

    # Final CSV: only image_id and label (saved next to dataset.xlsx)
    out = out[["image_id", "label"]]
    out.to_csv(CSV_OUTPUT_PATH, index=False)

    print(f"✅ Saved: {CSV_OUTPUT_PATH}")
    print(f"Rows in Excel: {total_rows}")
    print(f"Kept after single-label filter: {after_label_filter} (dropped {dropped_label})")
    print(
        f"Kept after restricting to patched_BTXRD: {after_patched_filter} "
        f"(dropped {dropped_missing_patched} not present in patched folder)"
    )
    print("Label mapping (0..6):")
    for name, idx in TUMOR_LABELS.items():
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()

