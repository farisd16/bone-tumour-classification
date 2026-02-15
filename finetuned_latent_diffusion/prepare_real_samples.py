# prepare_fid_data.py

from pathlib import Path
from PIL import Image

# ----- CONFIG -----
# Original folders
REAL_SRC = Path(
    "/vol/miltank/projects/practical_wise2526/"
    "bone-tumor-classification-gen-models/dataset/"
    "hf_entire_final_patched_BTXRD/train"
)

# Output folders for FID-ready images
REAL_DST = Path(
    "/vol/miltank/users/carre/bone-tumour-classification/"
    "latent_diffusion_finetuned/real_samples/real_512"
)

TARGET_SIZE = (512, 512)  # or (299, 299) if you prefer
# -------------------

def resize_folder(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    count, skipped = 0, 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                img = img.resize(TARGET_SIZE, Image.BICUBIC)
                img.save(out_path)
                count += 1
        except Exception as e:
            print(f"Skipping bad image: {p} | error: {e}")
            skipped += 1
    print(f"Resized {count} images from {src} to {dst} (skipped {skipped}).")

def main():
    print("Preparing real images for FID...")
    resize_folder(REAL_SRC, REAL_DST)
    print("Done. FID-ready folders:")
    print(f"  Real: {REAL_DST}")

if __name__ == "__main__":
    main()
