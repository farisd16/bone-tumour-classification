# lpips_eval.py
import os
import re
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from PIL import Image
import torchvision.transforms as T
import lpips


IMG_EXTS = {".jpeg", ".png"}


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


def normalize_class_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def strip_generated_suffix(name: str) -> str:
    # Matches folders like: class_name_gamma6_snapshot15800_trunc1.0
    return re.split(r"_gamma\d+.*$", name)[0]


def build_generated_class_map(gen_root: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for p in gen_root.iterdir():
        if not p.is_dir():
            continue
        base = normalize_class_name(strip_generated_suffix(p.name))
        mapping.setdefault(base, []).append(p)
    return mapping


def choose_best_generated_folder(paths: List[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    # Prefer highest snapshot number if present
    def snapshot_num(p: Path) -> int:
        m = re.search(r"snapshot(\d+)", p.name)
        return int(m.group(1)) if m else -1
    return sorted(paths, key=snapshot_num)[-1]


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
    ap.add_argument("--real_root", type=str, required=True,
                    help="Root folder containing class subfolders for real samples")
    ap.add_argument("--gen_root", type=str, required=True,
                    help="Root folder containing generated class subfolders")
    ap.add_argument("--classes", type=str, nargs="+", default=None,
                    help="Optional list of class subfolder names (defaults to all under real_root)")
    ap.add_argument("--pairs", type=int, default=10000, help="Random pairs per class")
    ap.add_argument("--img_size", type=int, default=256, help="Resize images to this")
    ap.add_argument("--backbone", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
                    help="LPIPS backbone (alex is fast, vgg is heavier)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    real_root = Path(args.real_root)
    gen_root = Path(args.gen_root)

    if args.classes:
        classes = args.classes
    else:
        classes = sorted([p.name for p in real_root.iterdir() if p.is_dir()])

    gen_map = build_generated_class_map(gen_root)
    resolved_classes = []
    for c in classes:
        key = normalize_class_name(c)
        if key not in gen_map:
            print(f"Warning: no generated folder found for class '{c}' (key: {key})")
            continue
        if len(gen_map[key]) > 1:
            picked = choose_best_generated_folder(gen_map[key])
            print(f"Warning: multiple generated folders for '{c}', using '{picked.name}'")
        resolved_classes.append(c)

    real_res = eval_model_root(
        real_root, resolved_classes, args.backbone,
        args.img_size, args.pairs, args.device, args.seed
    )

    # Evaluate generated folders using resolved mapping
    gen_res = {}
    loss_fn = lpips.LPIPS(net=args.backbone).to(args.device)
    loss_fn.eval()
    for c in resolved_classes:
        key = normalize_class_name(c)
        gen_folder = choose_best_generated_folder(gen_map[key])
        mean, std, k = lpips_intra_class(gen_folder, loss_fn, args.img_size, args.pairs, args.device, args.seed)
        gen_res[c] = {"mean": mean, "std": std, "pairs": k}

    if len(gen_res) > 0:
        means = [v["mean"] for v in gen_res.values() if not (v["mean"] != v["mean"])]
        stds = [v["std"] for v in gen_res.values() if not (v["std"] != v["std"])]
        macro_mean = float(torch.tensor(means).mean()) if means else float("nan")
        macro_std = float(torch.tensor(stds).mean()) if stds else float("nan")
        gen_res["_macro"] = {"mean": macro_mean, "std": macro_std, "pairs": args.pairs}
    else:
        gen_res["_macro"] = {"mean": float("nan"), "std": float("nan"), "pairs": args.pairs}

    print_results("Real", real_res)
    print_results("Generated", gen_res)


if __name__ == "__main__":
    main()
