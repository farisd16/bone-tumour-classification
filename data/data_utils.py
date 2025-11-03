from __future__ import annotations

from pathlib import Path
import json
import math
import random
from typing import Dict, List, Tuple, Optional

from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES: List[str] = [
    "osteochondroma",
    "osteosarcoma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "synovial osteochondroma",
    "osteofibroma",
]
LABEL_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class BTXRDDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(image_path)


def load_label_from_annotation(annotation_path: Path) -> Optional[int]:
    with annotation_path.open("r") as f:
        data = json.load(f)

    for shape in data.get("shapes", []):
        label_name = shape.get("label")
        if not label_name:
            continue
        key = label_name.strip().lower()
        if key in LABEL_TO_IDX:
            return LABEL_TO_IDX[key]
    return None


def collect_samples(images_root: Path, annotations_root: Path) -> List[Tuple[Path, int]]:
    if not images_root.exists():
        raise FileNotFoundError(f"Images folder not found: {images_root}")
    if not annotations_root.exists():
        raise FileNotFoundError(f"Annotations folder not found: {annotations_root}")
    samples: List[Tuple[Path, int]] = []
    image_paths: set[Path] = set()
    for pattern in ("*.jpeg", "*.jpg", "*.png"):
        image_paths.update(images_root.glob(pattern))
    for image_path in sorted(image_paths):
        annotation_path = annotations_root / f"{image_path.stem}.json"
        if not annotation_path.exists():
            continue
        label_idx = load_label_from_annotation(annotation_path)
        if label_idx is None:
            continue
        samples.append((image_path, label_idx))

    if not samples:
        raise RuntimeError("No labeled samples found. Ensure annotations exist for patched images.")
    return samples


def stratified_split(samples, ratios, seed):
    per_label: Dict[int, List[Tuple[Path, int]]] = {}
    for sample in samples:
        per_label.setdefault(sample[1], []).append(sample)

    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    ratio_order = sorted(ratios, key=ratios.get, reverse=True)

    for items in per_label.values():
        items_copy = items[:]
        rng.shuffle(items_copy)
        n = len(items_copy)

        counts = {key: int(n * ratios[key]) for key in ratios}
        assigned = sum(counts.values())
        remainder = n - assigned

        idx = 0
        while remainder > 0:
            key = ratio_order[idx % len(ratio_order)]
            counts[key] += 1
            remainder -= 1
            idx += 1

        start = 0
        for key in ("train", "validation", "test"):
            end = start + counts[key]
            splits[key].extend(items_copy[start:end])
            start = end

    for key in splits:
        rng.shuffle(splits[key])
    return splits


def build_dataloaders(
    images_root,
    annotations_root,
    ratios=None,
    batch_sizes=None,
    seed: int = 42,
):
    if ratios is None:
        ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    if batch_sizes is None:
        batch_sizes = {"train": 32, "validation": 32, "test": 32}

    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    samples = collect_samples(Path(images_root), Path(annotations_root))
    splits = stratified_split(samples, ratios, seed)

    train_tfm = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )
    eval_tfm = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )

    datasets = {
        "train": BTXRDDataset(splits["train"], train_tfm),
        "validation": BTXRDDataset(splits["validation"], eval_tfm),
        "test": BTXRDDataset(splits["test"], eval_tfm),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_sizes["train"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_sizes["validation"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_sizes["test"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        ),
    }

    for split_name, dataset in datasets.items():
        print(f"{split_name.capitalize()} samples: {len(dataset)}")

    return datasets, loaders

