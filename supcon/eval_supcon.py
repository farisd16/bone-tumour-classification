"""
Evaluation Script for SupCon Encoder with Linear Classifier.

This script evaluates a pretrained Supervised Contrastive (SupCon) encoder
combined with a trained linear classifier on the BTXRD dataset test split.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

import wandb
from config import WANDB_ENTITY, WANDB_PROJECT

from data.custom_dataset_class import CustomDataset
from sklearn.metrics import (
    f1_score,
    precision_score,
    accuracy_score,
    recall_score,
    balanced_accuracy_score,
)
from utils import display_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SupCon encoder + linear classifier on test split.")
    p.add_argument("--encoder-path", type=str, required=True, help="Path to encoder_supcon.pth")
    p.add_argument("--classifier-path", type=str, required=True, help="Path to classifier.pth")
    p.add_argument("--split-path", type=str, required=True, help="Path to split.json (from train_supcon run)")
    p.add_argument("--dataset-dir", type=str, default="data/dataset", help="Root dataset dir (default: data/dataset)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-name", type=str, default=None, help="Optional wandb run name")
    return p.parse_args()


def main():
    args = parse_args()

    encoder_path = Path(args.encoder_path)
    classifier_path = Path(args.classifier_path)
    split_path = Path(args.split_path)

    if not encoder_path.exists():
        raise FileNotFoundError(f"encoder_path does not exist: {encoder_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"classifier_path does not exist: {classifier_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"split_path does not exist: {split_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load split
    with open(split_path, "r") as f:
        split = json.load(f)
    test_idx = split["test"]

    # Create test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset_dir = Path(args.dataset_dir)
    image_dir = dataset_dir / "final_patched_BTXRD"
    json_dir = dataset_dir / "BTXRD" / "Annotations"

    full_data = CustomDataset(str(image_dir), str(json_dir), transform=transform)
    test_dataset = Subset(full_data, test_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Load encoder & classifier
    encoder = models.resnet34(weights=None)
    encoder.fc = nn.Identity()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    classifier = nn.Linear(512, 7)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing SupCon"):
            images = images.to(device)
            labels = labels.to(device)

            feats = encoder(images)
            logits = classifier(feats)
            preds = logits.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    display_confusion_matrix(all_labels, all_preds)

    metrics = {
        "weighted_precision": precision_score(all_labels, all_preds, average="weighted"),
        "weighted_recall": recall_score(all_labels, all_preds, average="weighted"),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
    }

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    if args.wandb:
        run_name = args.wandb_name or f"supcon-eval-{encoder_path.parent.name}"
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name)
        wandb.config.update({
            "encoder_path": str(encoder_path),
            "classifier_path": str(classifier_path),
            "split_path": str(split_path),
            "dataset_dir": str(dataset_dir),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        })
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()