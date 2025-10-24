import os
import sys
import pandas as pd

# --------- CONFIG (relative to this script) ----------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))       # folder containing create_csv.py
DATASET_DIR = os.path.join(ROOT_DIR, "BTXRD")               # repo/BTXRD
EXCEL_PATH = os.path.join(DATASET_DIR, "dataset.xlsx")      # repo/BTXRD/dataset.xlsx
CSV_OUTPUT_PATH = os.path.join(DATASET_DIR, "dataset_singlelabel.csv")

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
    print(f"[INFO] DATASET_DIR: {DATASET_DIR}")
    print(f"[INFO] EXCEL_PATH : {EXCEL_PATH}")

    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: dataset folder not found: {DATASET_DIR}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(EXCEL_PATH):
        print(f"ERROR: Excel file not found at: {EXCEL_PATH}", file=sys.stderr)
        sys.exit(1)

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
    before = len(out)
    out = out[(out["label"] >= 0) & (out["label"] <= 6)]
    after = len(out)
    dropped = before - after

    # Final CSV: only image_id and label (saved next to dataset.xlsx)
    out = out[["image_id", "label"]]
    out.to_csv(CSV_OUTPUT_PATH, index=False)

    print(f"✅ Saved: {CSV_OUTPUT_PATH}")
    print(f"Kept rows: {after} (dropped {dropped} rows with no or multiple tumors)")
    print("Label mapping (0..6):")
    for name, idx in TUMOR_LABELS.items():
        print(f"  {idx}: {name}")

if __name__ == "__main__":
    main()