from pathlib import Path
import argparse
import json
import random
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPOCHS = 10

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


def training_resnet(model_name):
    images_root = PROJECT_ROOT / "data" / "patched_BTXRD"
    annotations_root = PROJECT_ROOT / "data" / "BTXRD" / "Annotations"
    datasets, loaders = build_dataloaders(images_root, annotations_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match model_name:
        case "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, 7),  # 7 classes (0..6) according to create_csv.py
            )
        case "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 7),  # 7 classes (0..6) according to create_csv.py
            )
        case "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 7),  # 7 classes (0..6) according to create_csv.py
            )

    model.to(device)
    print(f"Using model {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=5e-3, weight_decay=2e-2, momentum=0.9)
    best_acc = 0.0
    output_dir = PROJECT_ROOT / "checkpoints" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        for phase in ("train", "validation"):
            if len(datasets[phase]) == 0:
                print(f"Skipping {phase} phase: no samples available.")
                continue
            model.train(phase == "train")
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, _ in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                preds = outputs.argmax(dim=1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum()

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])
            print(f"{phase:>11s} loss {epoch_loss:.4f}  acc {epoch_acc:.4f}")

            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), output_dir / "best.pt")

    final_path = output_dir / "final.pt"
    torch.save(model.state_dict(), final_path)

    # Testen
    if len(datasets["validation"]) > 0 or len(datasets["test"]) > 0:
        best_path = output_dir / "best.pt"
        load_path = best_path if best_path.exists() else final_path
        if not best_path.exists():
            print("Best weights not found (no validation split); using final weights for evaluation.")

        try:
            state_dict = torch.load(load_path, map_location=device, weights_only=True)
        except TypeError:
            # Fallback for older PyTorch without weights_only
            state_dict = torch.load(load_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        if len(datasets["test"]) == 0:
            print("Skipping test evaluation: no samples available.")
            return

        softmax = nn.Softmax(dim=1)
        all_labels, all_preds, all_probs, all_paths = [], [], [], []

        with torch.no_grad():
            for inputs, labels, paths in loaders["test"]:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = softmax(outputs)
                preds = outputs.argmax(dim=1).cpu().numpy()

                all_labels.extend(labels.numpy())
                all_preds.extend(preds)
                all_probs.extend(
                    probs[:, 1].cpu().numpy()
                )  # ROC for grade 1, adjust for 7 grades
                all_paths.extend(paths)

        acc = (np.array(all_labels) == np.array(all_preds)).mean()
        print(f"Test accuracy: {acc:.4f}")

        report = list(zip(all_paths, all_labels, all_preds))
        np.save(output_dir / "test_predictions.npy", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training pipeline with a variant of the ResNet model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet34", "resnet18"],
        help="ResNet model variant to train with",
    )
    args = parser.parse_args()
    training_resnet(args.model)
