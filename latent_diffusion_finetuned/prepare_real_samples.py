from pathlib import Path
from PIL import Image
import json
import re

# ----- CONFIG -----
# Original folders
REAL_SRC = Path(
    "/vol/miltank/projects/practical_wise2526/"
    "bone-tumor-classification-gen-models/dataset/"
    "final_patched_BTXRD"
)

# Path to the JSON file listing train split images
SPLIT_JSON = Path("/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/data_split.json")  

# Output folders for FID-ready images
REAL_DST = Path(
    "/vol/miltank/users/carre/bone-tumour-classification/"
    "latent_diffusion_finetuned/real_samples/real_512"
)

TARGET_SIZE = (512, 512)

# Regex to extract 4-digit index from "IMGxxxxxx.jpeg" (e.g., IMG001466.jpeg -> 1466)
INDEX_PATTERN = re.compile(r'IMG0*(\d{4})\.jpeg$', re.IGNORECASE)
# -------------------

def resize_train_split(src: Path, dst: Path, train_indices: list):
    """
    Resize only images whose index (extracted from IMGxxxxxx.jpeg) is in train_indices.
    """
    dst.mkdir(parents=True, exist_ok=True)
    count, skipped, no_match = 0, 0, 0
    
    # Create set of train indices as strings
    train_set = set(str(idx).zfill(4) for idx in train_indices)
    
    all_jpegs = list(REAL_SRC.rglob("*.jpeg"))
    print(f"Total JPEGs: {len(all_jpegs)}")

    matching = [INDEX_PATTERN.search(p.name).group(1) for p in all_jpegs if INDEX_PATTERN.search(p.name)]
    print(f"Matching indices in source: {set(matching)}")
    print(f"Train indices missing in source: {train_set - set(matching)}")
    print(f"Sample source filenames: {[p.name for p in all_jpegs[:10]]}")
    
    for p in src.rglob("*.jpeg"):
        # Extract index from filename
        match = INDEX_PATTERN.search(p.name)
        if not match:
            no_match += 1
            continue
        
        img_index = match.group(1)
        if img_index not in train_set:
            continue  # Skip non-train images
            
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
    
    print(f"Resized {count} train-split images from {src} to {dst}.")
    print(f"(skipped {skipped} bad, {no_match} non-matching filenames)")


def main():
    # Load split from JSON
    with open(SPLIT_JSON) as f:
        split_data = json.load(f)
    
    train_indices = split_data["train"]
    print(f"Loaded {len(train_indices)} train indices from {SPLIT_JSON}")
    
    print("Preparing real train-split images for FID...")
    resize_train_split(REAL_SRC, REAL_DST, train_indices)
    print("Done. FID-ready folders:")
    print(f"  Real (train only): {REAL_DST}")


if __name__ == "__main__":
    main()

