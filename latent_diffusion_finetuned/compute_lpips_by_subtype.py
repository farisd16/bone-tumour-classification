# compute_lpips_by_subtype.py

import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import lpips
from PIL import Image

# Paths - UPDATE BTXRD_XLSX
REAL_DIR = Path("/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/real_samples/real_512")
FAKE_DIR = Path("/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/fid_evaluation_samples/fid_1000_stable_diffusion_v1_5_lora_0_7_ckp_5000")
BTXRD_XLSX = Path("/vol/miltank/projects/practical_wise2526/bone-tumor-classification-gen-models/dataset/BTXRD/dataset.xlsx")

NET = "alex"
N_PAIRS_PER_SUBTYPE = 300  # Reduced for rare subtypes
BATCH_SIZE = 16
SEED = 42

TUMOR_COLS = {
    "osteochondroma": "osteochondroma",
    "multiple_osteochondromas": "multiple osteochondromas",
    "simple_bone_cyst": "simple bone cyst",
    "giant_cell_tumor": "giant cell tumor",
    "osteofibroma": "osteofibroma",
    "synovial_osteochondroma": "synovial osteochondroma",
    "osteosarcoma": "osteosarcoma"
}

LOCATION_COLS = [
    "hand", "ulna", "radius", "humerus", "foot", "tibia", "fibula", "femur",
    "hip bone", "ankle-joint", "knee-joint", "hip-joint", "wrist-joint",
    "elbow-joint", "shoulder-joint"
]

VIEW_COLS = ["frontal", "lateral", "oblique"]

def parse_fake_metadata(file_path):
    """Parse fake filename"""
    name = file_path.name.lower()
    parts = re.split(r'_|\.', name)
    if len(parts) >= 4:
        tumor = parts[1].replace('multipleosteochondromas', 'multiple osteochondromas')
        loc = parts[2]
        view = parts[3]
        return tumor, loc, view
    return None, None, None

def load_real_metadata(xlsx_path, real_dir):
    """Load XLSX, map image_id -> (tumor, loc, view) using binary columns."""
    df = pd.read_excel(xlsx_path, sheet_name=0)
    # Clean image_id column (assume first col)
    df['image_id'] = df['image_id'].astype(str).str.strip()
    
    groups = defaultdict(list)
    for idx, row in df.iterrows():
        img_id = row['image_id']
        img_paths = list(real_dir.glob(f"*{img_id}*"))
        if not img_paths:
            continue
        
        # Find tumor (first 1 in tumor cols)
        tumor_key = None
        for k, col in TUMOR_COLS.items():
            if pd.notna(row.get(col, 0)) and row[col] == 1:
                tumor_key = k
                break
        
        if not tumor_key:
            continue  # Skip non-matching tumors
        
        # Find location (first 1)
        loc_key = None
        for loc_col in LOCATION_COLS:
            if pd.notna(row.get(loc_col, 0)) and row[loc_col] == 1:
                loc_key = loc_col.replace('-', '_').replace(' ', '_')
                break
        
        # Find view
        view_key = None
        for v in VIEW_COLS:
            if pd.notna(row.get(v, 0)) and row[v] == 1:
                view_key = v
                break
        
        if loc_key and view_key:
            key = (tumor_key, loc_key, view_key)
            groups[key].extend(img_paths)
    
    return {k: list(set(v)) for k, v in groups.items()}  # Dedup paths

def load_image(path, device):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr * 2.0) - 1.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    real_files = list(REAL_DIR.rglob("*"))
    fake_files = list(FAKE_DIR.rglob("*.png"))
    
    print(f"Found {len(real_files)} real, {len(fake_files)} fake images.")

    if not BTXRD_XLSX.exists():
        print("Error: XLSX not found. Run: BTXRD_XLSX = Path('/vol/.../dataset.xlsx')")
        return

    real_groups = load_real_metadata(BTXRD_XLSX, REAL_DIR)
    fake_groups = defaultdict(list)
    for f in fake_files:
        tumor, loc, view = parse_fake_metadata(f)
        if tumor and loc and view:
            key = (tumor, loc, view)
            fake_groups[key].append(f)

    print(f"Real groups: {len(real_groups)}, Fake groups: {len(fake_groups)}")
    print("Sample keys:", list(real_groups.keys())[:3])

    loss_fn = lpips.LPIPS(net=NET).to(device).eval()

    results = []
    for key in set(list(real_groups) + list(fake_groups)):
        tumor, loc, view = key
        real_list = real_groups.get(key, [])
        fake_list = fake_groups.get(key, [])
        
        if len(real_list) < 5 or len(fake_list) < 5:  # Min threshold
            continue
        
        n_pairs = min(N_PAIRS_PER_SUBTYPE, len(real_list) * len(fake_list))
        pairs = [(random.choice(real_list), random.choice(fake_list)) for _ in range(n_pairs)]

        subtype_dists = []
        for i in range(0, n_pairs, BATCH_SIZE):
            batch_real, batch_fake = [], []
            for rf, ff in pairs[i:i+BATCH_SIZE]:
                try:
                    batch_real.append(load_image(rf, device))
                    batch_fake.append(load_image(ff, device))
                except Exception:
                    continue
            
            if batch_real:
                real_b = torch.cat(batch_real)
                fake_b = torch.cat(batch_fake)
                with torch.no_grad():
                    d = loss_fn(real_b, fake_b).view(-1).cpu().numpy()
                subtype_dists.extend(d)

        if subtype_dists:
            mean_d, std_d = np.mean(subtype_dists), np.std(subtype_dists)
            results.append({
                'tumor': tumor.replace('_', ' '),
                'location': loc.replace('_', ' '),
                'view': view,
                'n_real': len(real_list),
                'n_fake': len(fake_list),
                'n_pairs': len(subtype_dists),
                'mean_lpips': mean_d,
                'std_lpips': std_d
            })
            print(f"{tumor}-{loc}-{view}: {mean_d:.4f} ± {std_d:.4f} (real={len(real_list)}, fake={len(fake_list)})")

    if results:
        df = pd.DataFrame(results).round(4)
        print("\nSummary Table:")
        print(df)
        df.to_csv("lpips_by_subtype.csv", index=False)
        print(f"\nOverall mean LPIPS: {df['mean_lpips'].mean():.4f}")

if __name__ == "__main__":
    main()

