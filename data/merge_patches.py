from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def merge_patches(
    patched_dir: Path,
    squared_dir: Path,
    out_dir: Path,
    overwrite: bool = True,
) -> None:
    if not patched_dir.is_dir():
        print(f"ERROR: patched dir not found: {patched_dir}", file=sys.stderr)
        sys.exit(1)
    if not squared_dir.is_dir():
        print(f"ERROR: squared dir not found: {squared_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    def list_images(d: Path) -> list[Path]:
        exts = {".jpeg", ".jpg", ".png"}
        return [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]

    patched_imgs = list_images(patched_dir)
    squared_imgs = list_images(squared_dir)

    # Copy order: first all from patched, then squared (squared overwrites on conflicts)
    copied = 0
    skipped = 0
    overwritten = 0

    for src in patched_imgs:
        dst = out_dir / src.name
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        if dst.exists() and overwrite:
            overwritten += 1
        shutil.copy2(src, dst)
        copied += 1

    for src in squared_imgs:
        dst = out_dir / src.name
        if dst.exists() and not overwrite:
            # If not overwriting, keep existing patched version
            skipped += 1
            continue
        if dst.exists() and overwrite:
            overwritten += 1
        shutil.copy2(src, dst)
        copied += 1

    unique_names = {p.name for p in patched_imgs} | {s.name for s in squared_imgs}

    print("Merge summary:")
    print(f"  Patched images:            {len(patched_imgs)} from {patched_dir}")
    print(f"  Squared patched (106):     {len(squared_imgs)} from {squared_dir}")
    print(f"  Unique output filenames:   {len(unique_names)}")
    print(f"  Copied files (incl. overwrites): {copied}")
    print(f"  Overwritten due to conflict:      {overwritten}")
    if skipped:
        print(f"  Skipped (exists and overwrite=False): {skipped}")
    print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Merge patched_BTXRD and squared_patched_106 into a single folder.\n"
            "On filename conflicts, squared images take precedence (default)."
        )
    )
    parser.add_argument(
        "--patched-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "patched_BTXRD",
        help="Directory with standard patched images",
    )
    parser.add_argument(
        "--squared-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "squared_patched_106",
        help="Directory with squared patched images for the 106 cases",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset" / "patched_BTXRD_merged",
        help="Output directory for merged images",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing files in out-dir",
    )
    args = parser.parse_args()

    merge_patches(
        patched_dir=args.patched_dir,
        squared_dir=args.squared_dir,
        out_dir=args.out_dir,
        overwrite=not args.no_overwrite,
    )

