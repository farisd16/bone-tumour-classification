import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from data.custom_dataset_class import CustomDataset


DEFAULT_RUNS: List[str] = [
    "resnet_wce_aug_2025-11-14_15-45-32",
    "resnet_ce_noaug_2025-11-14_18-04-03",
    "resnet_ce_aug_2025-11-14_17-38-51",
    "resnet_wce_noaug_2025-11-14_17-08-47",
    "resnet_focal_aug_2025-12-08_19-51-10",
    "resnet_wfocal_noaug_2025-12-08_21-52-40",
    "supcon_noaug_2025-12-11_20-44-55",
    "resnet_wfocal_aug_2025-12-08_19-40-31",
    "supcon_aug_2025-12-11_17-54-40",
    "resnet_focal_noaug_2025-12-04_11-56-09",
]


def _val_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_resnet_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet34(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))
    model.to(device)
    model.eval()
    return model


def _compute_metrics(all_labels: List[int], all_preds: List[int]) -> Dict[str, float]:
    """Compute the requested evaluation metrics."""
    return {
        "weighted_recall": recall_score(
            all_labels, all_preds, average="weighted", zero_division=0
        ),
        "weighted_precision": precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(
            all_labels, all_preds, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            all_labels, all_preds, average="weighted", zero_division=0
        ),
    }


def _evaluate_resnet_run(
    run_dir: Path, batch_size: int, device: torch.device
) -> Tuple[Dict[str, float], int]:
    checkpoint = run_dir / "best_model.pth"
    split_path = run_dir / "data_split.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    with open(split_path, "r") as f:
        split = json.load(f)
    test_idx = split.get("test")
    if test_idx is None:
        raise KeyError("Split file does not contain 'test' indices")

    dataset_root = Path("data") / "dataset"
    image_dir = dataset_root / "final_patched_BTXRD"
    json_dir = dataset_root / "BTXRD" / "Annotations"
    dataset = CustomDataset(
        image_dir=str(image_dir), json_dir=str(json_dir), transform=_val_transform()
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=batch_size, shuffle=False
    )

    model = _load_resnet_model(num_classes=7, device=device)
    try:
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = _compute_metrics(all_labels, all_preds)
    return metrics, len(all_labels)


def _evaluate_supcon_run(
    run_dir: Path, batch_size: int, device: torch.device
) -> Tuple[Dict[str, float], int]:
    encoder_path = run_dir / "encoder_supcon.pth"
    classifier_path = run_dir / "classifier.pth"
    split_path = run_dir / "split.json"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing encoder checkpoint: {encoder_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Missing classifier checkpoint: {classifier_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    with open(split_path, "r") as f:
        split = json.load(f)
    test_idx = split.get("test")
    if test_idx is None:
        raise KeyError("Split file does not contain 'test' indices")

    dataset_root = Path("data") / "dataset"
    image_dir = dataset_root / "final_patched_BTXRD"
    json_dir = dataset_root / "BTXRD" / "Annotations"
    dataset = CustomDataset(
        image_dir=str(image_dir), json_dir=str(json_dir), transform=_val_transform()
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=batch_size, shuffle=False
    )

    encoder = models.resnet34(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    classifier = nn.Linear(512, 7)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            feats = encoder(images)
            logits = classifier(feats)
            _, preds = logits.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = _compute_metrics(all_labels, all_preds)
    return metrics, len(all_labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run saved checkpoints to compute evaluation metrics."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
        help="List of W&B display names / checkpoint folder names to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation dataloaders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recomputed_metrics_table.txt"),
        help="Path to save the markdown table with recomputed metrics.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for run_name in args.runs:
        resnet_dir = Path("checkpoints") / run_name
        supcon_dir = Path("checkpoints_supcon") / run_name
        try:
            if resnet_dir.exists():
                metrics, n = _evaluate_resnet_run(
                    resnet_dir, batch_size=args.batch_size, device=device
                )
                run_type = "resnet"
            elif supcon_dir.exists():
                metrics, n = _evaluate_supcon_run(
                    supcon_dir, batch_size=args.batch_size, device=device
                )
                run_type = "supcon"
            else:
                print(f"[SKIP] Run folder not found for {run_name}")
                continue
            print(
                f"[OK] {run_name} ({run_type}) "
                f"Macro F1: {metrics['macro_f1']:.4f} on {n} samples"
            )
            results.append({"name": run_name, "metrics": metrics})
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {run_name}: {exc}")

    if results:
        print("\nSummary (Macro F1):")
        for entry in results:
            print(f"- {entry['name']}: {entry['metrics']['macro_f1']:.4f}")

        table_lines = [
            "| Model | Weighted Recall | Weighted Precision | Balanced Accuracy | Macro F1 | Weighted F1 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for entry in results:
            m = entry["metrics"]
            table_lines.append(
                "| {name} | {wr:.4f} | {wp:.4f} | {ba:.4f} | {mf1:.4f} | {wf1:.4f} |".format(
                    name=entry["name"],
                    wr=m["weighted_recall"],
                    wp=m["weighted_precision"],
                    ba=m["balanced_accuracy"],
                    mf1=m["macro_f1"],
                    wf1=m["weighted_f1"],
                )
            )

        table_text = "\n".join(table_lines)
        args.output.write_text(table_text + "\n", encoding="utf-8")
        print(f"\nSaved metrics table to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
