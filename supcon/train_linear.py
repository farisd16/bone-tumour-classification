import os
import json
import argparse
import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

from data.custom_dataset_class import CustomDataset


"""
To train the classifier part
"""


def parse_args():
    p = argparse.ArgumentParser(description="Train linear classifier on frozen SupCon encoder.")
    p.add_argument("--encoder-path", type=str, required=True, help="Path to encoder_supcon.pth")
    p.add_argument("--split-path", type=str, required=True, help="Path to split.json (from train_supcon run)")
    p.add_argument("--dataset-dir", type=str, default="data/dataset", help="Root dataset dir (default: data/dataset)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-root", type=str, default="checkpoints_linear", help="Where to create timestamped run folder")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_path = Path(args.encoder_path)
    split_path = Path(args.split_path)

    if not encoder_path.exists():
        raise FileNotFoundError(f"encoder_path does not exist: {encoder_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"split_path does not exist: {split_path}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(args.save_root) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving classifier to: {save_dir}")

    # Save run metadata (for reproducibility)
    run_meta = {
        "encoder_path": str(encoder_path.resolve()),
        "split_path": str(split_path.resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "num_classes": args.num_classes,
        "num_workers": args.num_workers,
        "device": str(device),
    }
    with open(save_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=4)

    # Load split
    with open(split_path, "r") as f:
        split = json.load(f)

    train_idx = split["train"]  # same train indices as train_supcon

    # Normal transforms
    normal_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Dataset
    dataset_dir = Path(args.dataset_dir)
    image_dir = dataset_dir / "final_patched_BTXRD"
    json_dir = dataset_dir / "BTXRD" / "Annotations"

    full = CustomDataset(image_dir=str(image_dir), json_dir=str(json_dir), transform=normal_transform)
    train_subset = Subset(full, train_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Load encoder & freeze it
    encoder = models.resnet34(weights=None)
    encoder.fc = nn.Identity()  # remove classification head
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False

    # Linear classifier (trainable)
    classifier = nn.Linear(512, args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                feats = encoder(images)  # 512-d vector

            logits = classifier(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            preds = logits.argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        acc = correct / total * 100
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Acc={acc:.2f}%")

    # Save classifier
    clf_path = save_dir / "classifier.pth"
    torch.save(classifier.state_dict(), clf_path)
    print(f"Saved classifier to: {clf_path}")


if __name__ == "__main__":
    main()