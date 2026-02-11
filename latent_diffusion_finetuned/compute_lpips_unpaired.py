# compute_lpips_unpaired.py

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import lpips  # pip install lpips

REAL_DIR = Path("/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/real_samples/real_512")
FAKE_DIR = Path("/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/fid_evaluation_samples/sd-1-5-lora-rank-32-batch-4-resolution-512-1000-samples-chkp-45000")

NET = "alex"        # 'alex' is standard for LPIPS
N_PAIRS = 5000      # number of random real–fake pairs to sample
BATCH_SIZE = 16

def load_image(path, device):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)      # HWC -> CHW
    arr = (arr * 2.0) - 1.0           # [0,1] -> [-1,1]
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
    return tensor

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    real_files = [p for p in REAL_DIR.rglob("*") if p.is_file()]
    fake_files = [p for p in FAKE_DIR.rglob("*") if p.is_file()]

    if not real_files or not fake_files:
        print("No images found in one of the directories.")
        return

    print(f"Found {len(real_files)} real and {len(fake_files)} fake images.")

    loss_fn = lpips.LPIPS(net=NET).to(device)
    loss_fn.eval()

    distances = []
    pairs = []

    # Pre-sample indices for reproducibility and speed
    for _ in range(N_PAIRS):
        rf = random.choice(real_files)
        ff = random.choice(fake_files)
        pairs.append((rf, ff))

    for i in range(0, N_PAIRS, BATCH_SIZE):
        batch_real = []
        batch_fake = []
        for rf, ff in pairs[i : i + BATCH_SIZE]:
            try:
                im0 = load_image(rf, device)
                im1 = load_image(ff, device)
            except Exception as e:
                print(f"Skipping bad pair ({rf}, {ff}): {e}")
                continue
            batch_real.append(im0)
            batch_fake.append(im1)

        if not batch_real:
            continue

        real_batch = torch.cat(batch_real, dim=0)
        fake_batch = torch.cat(batch_fake, dim=0)

        with torch.no_grad():
            d = loss_fn(real_batch, fake_batch)  # (B,1,1,1) or (B,)
        distances.extend(d.view(-1).cpu().numpy().tolist())

    distances = np.array(distances)
    print(f"Unpaired LPIPS ({NET}) over {len(distances)} random real–fake pairs:")
    print(f"  mean: {distances.mean():.4f}")
    print(f"  std:  {distances.std():.4f}")

if __name__ == "__main__":
    main()
