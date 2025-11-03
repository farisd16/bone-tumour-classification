import os
import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "dataset" / "BTXRD" / "dataset_singlelabel.csv"
PATCHED_DIR = PROJECT_ROOT / "data" / "dataset" / "patched_BTXRD"


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)
    if not PATCHED_DIR.is_dir():
        print(f"ERROR: patched dir not found: {PATCHED_DIR}", file=sys.stderr)
        sys.exit(1)

    patched_files = {p.name.lower() for p in PATCHED_DIR.iterdir() if p.is_file()}

    with CSV_PATH.open(newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        csv_files = {os.path.basename(row['image_id']).lower() for row in rdr}

    missing_in_csv = sorted(patched_files - csv_files)
    extra_in_csv = sorted(csv_files - patched_files)

    print(f"patched files: {len(patched_files)}")
    print(f"csv rows (unique image_id): {len(csv_files)}")
    print(f"patched - csv: {len(missing_in_csv)}")
    if missing_in_csv:
        print("Examples missing in csv (up to 20):")
        print("\n".join(missing_in_csv[:20]))
    print(f"csv - patched: {len(extra_in_csv)}")
    if extra_in_csv:
        print("Examples missing in patched (up to 20):")
        print("\n".join(extra_in_csv[:20]))


if __name__ == "__main__":
    main()

