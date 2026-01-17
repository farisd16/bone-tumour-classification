# lpips_eval.py
import os
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from PIL import Image
import torchvision.transforms as T
import lpips


IMG_EXTS = {".jpg",".jpeg"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def load_img_as_tensor(path: Path, size: int, device: str) -> torch.Tensor:
    """
    Returns tensor shaped (1,3,H,W) in [-1,1] as expected by lpips.LPIPS.
    Handles grayscale by converting to RGB (channel repeat).
    """
    img = Image.open(path).convert("RGB")  # ensures 3 channels
    tfm = T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),                 # [0,1]
        T.Lambda(lambda x: x * 2 - 1)  # [-1,1]
    ])
    x = tfm(img).unsqueeze(0).to(device)  # (1,3,H,W)
    return x


@torch.no_grad()
def lpips_intra_class(
    class_folder: Path,
    loss_fn: lpips.LPIPS,
    size: int,
    num_pairs: int,
    device: str,
    seed: int = 0,
) -> Tuple[float, float, int]:
    """
    Randomly samples num_pairs pairs from images inside class_folder, computes LPIPS for each.
    Returns (mean, std, used_pairs).
    """
    paths = list_images(class_folder)
    n = len(paths)
    if n < 2:
        return float("nan"), float("nan"), 0

    rng = random.Random(seed)
    # If num_pairs is too large relative to n, we still sample with replacement safely.
    scores = []

    for _ in range(num_pairs):
        a, b = rng.sample(paths, 2)  # without replacement within the pair
        xa = load_img_as_tensor(a, size=size, device=device)
        xb = load_img_as_tensor(b, size=size, device=device)
        d = loss_fn(xa, xb)  # shape (1,1,1,1) or (1,1)
        scores.append(float(d.squeeze().cpu()))

    t = torch.tensor(scores)
    return float(t.mean()), float(t.std(unbiased=True)), len(scores)


def eval_model_root(
    root: Path,
    classes: List[str],
    backbone: str,
    size: int,
    num_pairs: int,
    device: str,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    loss_fn = lpips.LPIPS(net=backbone).to(device)
    loss_fn.eval()

    results = {}
    per_class_means = []
    per_class_stds = []

    for c in classes:
        folder = root / c
        mean, std, k = lpips_intra_class(folder, loss_fn, size, num_pairs, device, seed)
        results[c] = {"mean": mean, "std": std, "pairs": k}
        if k > 0 and not (mean != mean):  # not nan
            per_class_means.append(mean)
            per_class_stds.append(std)

    # Macro-average across classes (class-balanced)
    if len(per_class_means) > 0:
        macro_mean = float(torch.tensor(per_class_means).mean())
        macro_std = float(torch.tensor(per_class_stds).mean())  # simple summary
    else:
        macro_mean, macro_std = float("nan"), float("nan")

    results["_macro"] = {"mean": macro_mean, "std": macro_std, "pairs": num_pairs}
    return results


def print_results(title: str, res: Dict[str, Dict[str, float]]):
    print(f"\n== {title} ==")
    for k, v in res.items():
        if k == "_macro":
            continue
        print(f"{k:>15s}: LPIPS mean={v['mean']:.4f}  std={v['std']:.4f}  pairs={int(v['pairs'])}")
    m = res["_macro"]
    print(f"{'MACRO-AVG':>15s}: LPIPS mean={m['mean']:.4f}  (avg std={m['std']:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stylegan_root", type=str, required=True,
                    help="Root folder containing class subfolders for StyleGAN samples")
    ap.add_argument("--diffusion_root", type=str, required=True,
                    help="Root folder containing class subfolders for Diffusion samples")
    ap.add_argument("--classes", type=str, nargs="+", required=True,
                    help="List of class subfolder names, e.g. benign malignant")
    ap.add_argument("--pairs", type=int, default=10000, help="Random pairs per class")
    ap.add_argument("--img_size", type=int, default=256, help="Resize images to this")
    ap.add_argument("--backbone", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
                    help="LPIPS backbone (alex is fast, vgg is heavier)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stylegan_res = eval_model_root(
        Path(args.stylegan_root), args.classes, args.backbone,
        args.img_size, args.pairs, args.device, args.seed
    )
    diffusion_res = eval_model_root(
        Path(args.diffusion_root), args.classes, args.backbone,
        args.img_size, args.pairs, args.device, args.seed
    )

    print_results("StyleGAN2-ADA", stylegan_res)
    print_results("Diffusion", diffusion_res)


if __name__ == "__main__":
    main()
