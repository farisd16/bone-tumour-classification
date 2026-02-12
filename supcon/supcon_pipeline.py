"""
SupCon + Linear Evaluation Pipeline (Training Script)

This script implements a two-stage training pipeline for image classification:

1) Supervised Contrastive (SupCon) pretraining:
   - Trains an encoder backbone with a projection head using supervised
     contrastive loss (Khosla et al., 2020).
   - Requires two augmented views per image and uses label information
     to define positive pairs.

2) Linear evaluation / probing:
   - Freezes the pretrained encoder.
   - Trains a linear classifier head on top of the frozen features.
   - Evaluates on validation and test splits.

Additional features:
- Stratified train/val/test splits.
- Optional minority-class-specific augmentations during training.
- Logging of training and evaluation metrics to Weights & Biases (W&B).
- Saving checkpoints and split information, and logging them as W&B artifacts.

Expected directory structure (relative):
- data/dataset/final_patched_BTXRD
- data/dataset/BTXRD/Annotations

Outputs:
- checkpoints_supcon/<run_name>/encoder_supcon.pth
- checkpoints_supcon/<run_name>/classifier.pth
- checkpoints_supcon/<run_name>/split.json
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
import wandb
from tqdm import tqdm

# Ensure project root is on path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import WANDB_ENTITY, WANDB_PROJECT
from data.custom_dataset_class import CustomDataset
from sup_contrastive import SupConLoss, SupConModel, TwoViewDataset
from train_utils import make_minority_transform, default_minority_classes, make_transforms


DEFAULT_CONFIG: Dict[str, Any] = {
    "run_name_prefix": "supcon",
    "random_state": 42,
    "test_size": 0.2,
    "val_size": 0.1,
    "num_classes": 7,
    # SupCon phase
    "temperature": 0.1,
    "feature_dim": 128,
    "supcon_lr": 3e-4,
    "supcon_weight_decay": 1e-5,
    "supcon_batch_size": 32,
    "supcon_epochs": 50,
    # Linear head phase
    "linear_lr": 1e-3,
    "linear_weight_decay": 0.0,
    "linear_batch_size": 32,
    "linear_epochs": 20,
    # Augmentation knobs
    "color_jitter": 0.2,
    "random_rotation": 15,
    "apply_minority_aug": False,
    "minority_classes": default_minority_classes(),
}


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    val = value.lower()
    if val in {"true", "t", "1", "yes", "y"}:
        return True
    if val in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SupCon + linear evaluation pipeline")
    parser.add_argument("--run-name-prefix", "--run_name_prefix", dest="run_name_prefix", default=DEFAULT_CONFIG["run_name_prefix"])
    parser.add_argument("--random-state", "--random_state", dest="random_state", type=int, default=DEFAULT_CONFIG["random_state"])
    parser.add_argument("--test-size", "--test_size", dest="test_size", type=float, default=DEFAULT_CONFIG["test_size"])
    parser.add_argument("--val-size", "--val_size", dest="val_size", type=float, default=DEFAULT_CONFIG["val_size"])
    parser.add_argument("--num-classes", "--num_classes", dest="num_classes", type=int, default=DEFAULT_CONFIG["num_classes"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"])
    parser.add_argument("--feature-dim", "--feature_dim", dest="feature_dim", type=int, default=DEFAULT_CONFIG["feature_dim"])
    parser.add_argument("--supcon-lr", "--supcon_lr", dest="supcon_lr", type=float, default=DEFAULT_CONFIG["supcon_lr"])
    parser.add_argument(
        "--supcon-weight-decay",
        "--supcon_weight_decay",
        dest="supcon_weight_decay",
        type=float,
        default=DEFAULT_CONFIG["supcon_weight_decay"],
    )
    parser.add_argument(
        "--supcon-batch-size",
        "--supcon_batch_size",
        dest="supcon_batch_size",
        type=int,
        default=DEFAULT_CONFIG["supcon_batch_size"],
    )
    parser.add_argument("--supcon-epochs", "--supcon_epochs", dest="supcon_epochs", type=int, default=DEFAULT_CONFIG["supcon_epochs"])
    parser.add_argument("--linear-lr", "--linear_lr", dest="linear_lr", type=float, default=DEFAULT_CONFIG["linear_lr"])
    parser.add_argument(
        "--linear-weight-decay",
        "--linear_weight_decay",
        dest="linear_weight_decay",
        type=float,
        default=DEFAULT_CONFIG["linear_weight_decay"],
    )
    parser.add_argument(
        "--linear-batch-size",
        "--linear_batch_size",
        dest="linear_batch_size",
        type=int,
        default=DEFAULT_CONFIG["linear_batch_size"],
    )
    parser.add_argument("--linear-epochs", "--linear_epochs", dest="linear_epochs", type=int, default=DEFAULT_CONFIG["linear_epochs"])
    parser.add_argument("--color-jitter", "--color_jitter", dest="color_jitter", type=float, default=DEFAULT_CONFIG["color_jitter"])
    parser.add_argument(
        "--random-rotation", "--random_rotation", dest="random_rotation", type=float, default=DEFAULT_CONFIG["random_rotation"]
    )
    parser.add_argument(
        "--apply-minority-aug",
        "--apply_minority_aug",
        dest="apply_minority_aug",
        type=_str2bool,
        nargs="?",
        const=True,
        default=DEFAULT_CONFIG["apply_minority_aug"],
        help="Apply stronger augmentation only to configured minority classes during training phases",
    )
    parser.add_argument(
        "--minority-classes",
        "--minority_classes",
        dest="minority_classes",
        type=str,
        help="Comma-separated list of minority class names to receive stronger augmentation",
    )
    return parser.parse_args()


def get_base_dirs() -> Tuple[Path, Path]:
    dataset_dir = Path("data") / "dataset"
    image_dir = dataset_dir / "final_patched_BTXRD"
    json_dir = dataset_dir / "BTXRD" / "Annotations"
    return image_dir, json_dir


def stratified_splits(
    labels: np.ndarray, test_size: float, val_size: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stratified train/val/test indices."""
    all_indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=test_size, stratify=labels, random_state=random_state
    )
    train_labels = labels[train_idx]
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        stratify=train_labels,
        random_state=random_state,
    )
    return train_idx, val_idx, test_idx


def make_contrastive_transform(color_jitter: float, random_rotation: float):
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(random_rotation),
            transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_contrastive_minority_transform(color_jitter: float, random_rotation: float):
    """Stronger contrastive aug for minority classes."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(random_rotation + 5),
            transforms.ColorJitter(brightness=color_jitter + 0.1, contrast=color_jitter + 0.1),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.7),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class TwoViewMinorityDataset(torch.utils.data.Dataset):
    """Two-view dataset with minority-class-specific transform."""

    def __init__(
        self,
        base_dataset: CustomDataset,
        transform,
        minority_transform,
        minority_class_indices: List[int],
        apply_minority: bool,
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        self.minority_transform = minority_transform
        self.minority_class_indices = set(minority_class_indices)
        self.apply_minority = apply_minority

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]  # returns PIL image, label index
        use_minority = self.apply_minority and (label in self.minority_class_indices)
        t = self.minority_transform if use_minority else self.transform
        view1 = t(img)
        view2 = t(img)
        return torch.stack([view1, view2], dim=0), label


def make_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _minority_class_indices(minority_classes: List[str], class_to_idx: Dict[str, int]) -> List[int]:
    return [class_to_idx[c.lower()] for c in minority_classes if c.lower() in class_to_idx]


def train_supcon(
    cfg: Dict[str, Any],
    run_dir: Path,
    base_dataset: CustomDataset,
    train_idx: np.ndarray,
) -> Path:
    """Train SupCon encoder and return encoder checkpoint path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    minority_classes = cfg.get("minority_classes", default_minority_classes())
    minority_idx = _minority_class_indices(minority_classes, base_dataset.class_to_idx)

    contrastive_dataset = TwoViewMinorityDataset(
        Subset(base_dataset, train_idx),
        make_contrastive_transform(cfg["color_jitter"], cfg["random_rotation"]),
        make_contrastive_minority_transform(cfg["color_jitter"], cfg["random_rotation"]),
        minority_class_indices=minority_idx,
        apply_minority=bool(cfg.get("apply_minority_aug", False)),
    )
    train_loader = DataLoader(
        contrastive_dataset,
        batch_size=int(cfg["supcon_batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    encoder.fc = nn.Identity()
    model = SupConModel(encoder, feature_dim=int(cfg["feature_dim"])).to(device)

    criterion = SupConLoss(temperature=float(cfg["temperature"]))
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["supcon_lr"]),
        weight_decay=float(cfg["supcon_weight_decay"]),
    )

    print(f"[SupCon] Training for {cfg['supcon_epochs']} epochs "
          f"with batch_size={cfg['supcon_batch_size']}, temperature={cfg['temperature']}")

    for epoch in range(int(cfg["supcon_epochs"])):
        model.train()
        total_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"SupCon Epoch {epoch + 1}/{cfg['supcon_epochs']}",
            leave=False,
        )
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)
            images = images.view(-1, *images.shape[2:])

            features = model(images)
            features = features.view(batch_size, 2, -1)

            loss = criterion(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(len(train_loader), 1)
        wandb.log({"supcon/avg_loss": avg_loss, "supcon/epoch": epoch + 1})
        print(f"[SupCon] Epoch {epoch + 1}/{cfg['supcon_epochs']} avg_loss={avg_loss:.4f}")

    encoder_path = run_dir / "encoder_supcon.pth"
    torch.save(model.encoder.state_dict(), encoder_path)
    return encoder_path


def train_linear(
    cfg: Dict[str, Any],
    run_dir: Path,
    encoder_path: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    class_names: List[str],
) -> Tuple[float, float]:
    """Train linear head on frozen encoder. Returns (best_val_acc, test_acc)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir, json_dir = get_base_dirs()
    train_transform, val_transform = make_transforms()
    eval_transform = make_eval_transform()

    apply_minority = bool(cfg.get("apply_minority_aug", False))
    minority_classes = cfg.get("minority_classes", default_minority_classes())

    if apply_minority:
        minority_transform = make_minority_transform()
        train_dataset_full = CustomDataset(
            image_dir=str(image_dir),
            json_dir=str(json_dir),
            transform=train_transform,
            minority_transform=minority_transform,
            minority_classes=minority_classes,
        )
    else:
        train_dataset_full = CustomDataset(
            image_dir=str(image_dir),
            json_dir=str(json_dir),
            transform=train_transform,
        )

    val_dataset_full = CustomDataset(
        image_dir=str(image_dir),
        json_dir=str(json_dir),
        transform=val_transform,
    )
    test_dataset_full = CustomDataset(
        image_dir=str(image_dir),
        json_dir=str(json_dir),
        transform=eval_transform,
    )

    train_subset = Subset(train_dataset_full, train_idx)
    val_subset = Subset(val_dataset_full, val_idx)
    test_subset = Subset(test_dataset_full, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=int(cfg["linear_batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=int(cfg["linear_batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=int(cfg["linear_batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    encoder = models.resnet34(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    classifier = nn.Linear(512, int(cfg["num_classes"])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        classifier.parameters(),
        lr=float(cfg["linear_lr"]),
        weight_decay=float(cfg["linear_weight_decay"]),
    )

    best_val_acc = 0.0
    best_model_path = run_dir / "classifier.pth"

    print(f"[Linear] Training for {cfg['linear_epochs']} epochs "
          f"with batch_size={cfg['linear_batch_size']}, lr={cfg['linear_lr']}")

    def evaluate(loader, collect_metrics: bool = False):
        classifier.eval()
        correct, total, loss_sum = 0, 0, 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        class_correct = [0 for _ in class_names]
        class_total = [0 for _ in class_names]
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    feats = encoder(images)
                logits = classifier(feats)
                loss = criterion(logits, labels)
                loss_sum += loss.item()
                _, preds = logits.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
                if collect_metrics:
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                    for p, l in zip(preds, labels):
                        class_total[int(l.item())] += 1
                        if p == l:
                            class_correct[int(l.item())] += 1
        avg_loss = loss_sum / max(len(loader), 1)
        acc = 100.0 * correct / max(total, 1)
        metrics = {}
        if collect_metrics:
            class_acc = {}
            for idx, name in enumerate(class_names):
                total_c = class_total[idx]
                if total_c > 0:
                    class_acc[name] = 100.0 * class_correct[idx] / total_c
            metrics = {
                "precision_weighted": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
                "recall_weighted": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
                "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
                "accuracy": accuracy_score(all_labels, all_preds),
                "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
                "class_accuracy": class_acc,
            }
        return avg_loss, acc, metrics

    # Proper transform application within DataLoader using dataset-transform override
    train_subset.dataset.transform = eval_transform  # type: ignore
    val_subset.dataset.transform = eval_transform  # type: ignore
    test_subset.dataset.transform = eval_transform  # type: ignore

    for epoch in range(int(cfg["linear_epochs"])):
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0
        progress = tqdm(
            train_loader,
            desc=f"Linear Epoch {epoch + 1}/{cfg['linear_epochs']}",
            leave=False,
        )
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                feats = encoder(images)
            logits = classifier(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = logits.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            progress.set_postfix(loss=loss.item())

        train_acc = 100.0 * correct / max(total, 1)
        train_loss = total_loss / max(len(train_loader), 1)
        # Validation: only accuracy/loss to match baseline logging
        val_loss, val_acc, _ = evaluate(val_loader, collect_metrics=False)

        wandb.log(
            {
                "linear/train_loss": train_loss,
                "linear/train_acc": train_acc,
                "linear/val_loss": val_loss,
                "linear/val_acc": val_acc,
                "linear/epoch": epoch + 1,
                # Duplicate keys matching CE/WCE/Focal runs
                "Accuracy/Train": train_acc,
                "Accuracy/Val": val_acc,
                "Loss/Train": train_loss,
                "Loss/Val": val_loss,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), best_model_path)
        print(
            f"[Linear] Epoch {epoch + 1}/{cfg['linear_epochs']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

    # Evaluate best classifier on test
    classifier.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_metrics = evaluate(test_loader, collect_metrics=True)
    wandb.log(
        {
            # Test metrics (single evaluation after training)
            "Test Loss": test_loss,
            "Test Accuracy": test_metrics.get("accuracy", 0.0),
            "Weighted Precision": test_metrics.get("precision_weighted", 0.0),
            "Weighted Recall": test_metrics.get("recall_weighted", 0.0),
            "Weighted F1": test_metrics.get("f1_weighted", 0.0),
            "Balanced Accuracy": test_metrics.get("balanced_accuracy", 0.0),
            **{f"Class Accuracy/{k}": v for k, v in test_metrics.get("class_accuracy", {}).items()},
        }
    )
    return best_val_acc, test_acc, test_metrics


def main(config: Dict[str, Any] = None) -> float:
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    else:
        args = parse_args()
        cfg.update(vars(args))

    # Normalize minority classes config (supports comma-separated CLI string or list)
    minority_classes = cfg.get("minority_classes", default_minority_classes())
    if minority_classes is None or minority_classes == "":
        minority_classes = default_minority_classes()
    if isinstance(minority_classes, str):
        minority_classes = [c.strip().lower() for c in minority_classes.split(",") if c.strip()]
    else:
        minority_classes = [c.lower() for c in minority_classes]
    if not minority_classes:
        minority_classes = default_minority_classes()
    cfg["minority_classes"] = minority_classes

    image_dir, json_dir = get_base_dirs()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    aug_tag = "aug" if cfg.get("apply_minority_aug") else "noaug"
    run_name = f"{cfg['run_name_prefix']}_{aug_tag}_{timestamp}"
    run_dir = Path("checkpoints_supcon") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = CustomDataset(
        image_dir=str(image_dir),
        json_dir=str(json_dir),
        transform=lambda x: x,  # keep PIL images for contrastive transforms
    )
    labels = np.array([base_dataset.class_to_idx[label] for _, label in base_dataset.samples])
    train_idx, val_idx, test_idx = stratified_splits(
        labels=labels,
        test_size=float(cfg["test_size"]),
        val_size=float(cfg["val_size"]),
        random_state=int(cfg["random_state"]),
    )

    split_path = run_dir / "split.json"
    with open(split_path, "w") as f:
        json.dump(
            {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()},
            f,
            indent=4,
        )

    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=cfg,
        name=run_name,
    ) as run:
        cfg = dict(wandb.config)
        encoder_path = train_supcon(cfg, run_dir, base_dataset, train_idx)
        best_val_acc, test_acc, test_metrics = train_linear(
            cfg,
            run_dir,
            encoder_path,
            train_idx,
            val_idx,
            test_idx,
            base_dataset.classes,
        )

        artifact = wandb.Artifact(name=f"{run_name}_artifacts", type="model")
        artifact.add_file(str(encoder_path))
        artifact.add_file(str(run_dir / "classifier.pth"))
        artifact.add_file(str(split_path))
        run.log_artifact(artifact)

        run.summary["val_acc"] = best_val_acc
        run.summary["test_acc"] = test_acc
        for k, v in test_metrics.items():
            run.summary[f"test_{k}"] = v
        # Also mirror common summary keys for comparison
        run.summary["Accuracy/Val"] = best_val_acc
        run.summary["Accuracy"] = test_metrics.get("accuracy", test_acc)
        return best_val_acc


if __name__ == "__main__":
    main()
