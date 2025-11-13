import os
import json
import datetime
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.custom_dataset_class import CustomDataset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import wandb

from config import WANDB_ENTITY, WANDB_PROJECT

import argparse
from early_stopping import EarlyStopper

# CLI args (early stopping)
parser = argparse.ArgumentParser()
parser.add_argument("--early-stop", action="store_true",
                    help="Enable early stopping on validation loss")
parser.add_argument("--early-stop-patience", type=int, default=5,
                    help="Epochs without improvement before stopping")
parser.add_argument("--early-stop-min-delta", type=float, default=0.0,
                    help="Minimum improvement in val loss to reset patience")
args = parser.parse_args()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder structure and Tensorboard
checkpoints_base_dir = "checkpoints"
os.makedirs(checkpoints_base_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(checkpoints_base_dir, f"resnet_{timestamp}")
best_model_path = os.path.join(run_dir, "best_model.pth")
os.makedirs(run_dir, exist_ok=True)


# TensorBoard writer
writer = SummaryWriter(log_dir=run_dir)

# Transformations
# - Train: with augmentations
# - Val/Test: deterministic only
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # Color jitter: brightness, contrast, saturation, hue
        transforms.ColorJitter(
            brightness=0.2,  # ±20% brightness variation
            contrast=0.2,  # ±20% contrast variation
            saturation=0.2,  # ±20% saturation variation
            hue=0.1,  # ±0.1 hue shift
        ),
        transforms.RandomPerspective(
            distortion_scale=0.2, p=0.5
        ),  # Random perspective distortion (scale 0.2, probability 0.5)
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5], std=[0.5]
        ),  # Normalize to mean=0 (and std=1 by default)
        # TODO: Investigate if normalization should be done like with ResNet
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5], std=[0.5]
        ),  # Normalize to mean=0 (and std=1 by default)
        # TODO: Investigate if normalization should be done like with ResNet
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

DATASET_DIR = os.path.join("data", "dataset")
image_dir = (
    Path(DATASET_DIR) / "patched_BTXRD_merged"
)  # Folder might have to be changed
json_dir = Path(DATASET_DIR) / "BTXRD" / "Annotations"

# Build a base dataset to create splits (no transform needed for indexing)
dataset_base = CustomDataset(
    image_dir=str(image_dir), json_dir=str(json_dir), transform=None
)

# targets = [label for _, label in dataset_base]
targets = np.array(
    [dataset_base.class_to_idx[label] for _, label in dataset_base.samples]
)
indices = np.arange(len(dataset_base))

# Stratified shuffling
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, temp_idx = next(sss.split(indices, targets))

sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
# val_idx, test_idx = next(sss_val.split(temp_idx, np.array(targets)[temp_idx]))
val_rel, test_rel = next(sss_val.split(temp_idx, np.array(targets)[temp_idx]))
val_idx = temp_idx[val_rel]
test_idx = temp_idx[test_rel]

split_indices = {
    "train": train_idx.tolist(),
    "val": val_idx.tolist(),
    "test": test_idx.tolist(),
}

split_save_path = f"{run_dir}/data_split.json"

with open(split_save_path, "w") as f:
    json.dump(split_indices, f)
print(f"Saved split to {split_save_path}")

# === Create subsets using the same indices ===
train_ds_full = CustomDataset(
    image_dir=str(image_dir), json_dir=str(json_dir), transform=train_transform
)
val_ds_full = CustomDataset(
    image_dir=str(image_dir), json_dir=str(json_dir), transform=val_transform
)

train_dataset = Subset(train_ds_full, split_indices["train"])
val_dataset = Subset(val_ds_full, split_indices["val"])

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 7),
)
model.to(device)

# (Weighted) Cross Entropy Loss, Optimizer, Scheduler
weighted_cross_entropy = True

if weighted_cross_entropy:
    targets = [label for _, label in train_dataset]
    # train_targets = targets[train_idx]
    class_counts = Counter(targets)
    weights = 1.0 / np.array([class_counts[i] for i in range(7)])
    weights = weights / weights.sum() * 7
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()

lr = 1e-4
weight_decay = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# Training loop
num_epochs = 30
best_val_acc = 0.0

# Early stopping state
if args.early_stop:
    best_val_loss = float("inf")
    early_stopper = EarlyStopper(patience=args.early_stop_patience,
                                 min_delta=args.early_stop_min_delta)
else:
    best_val_loss = None
    early_stopper = None

# WandB run
run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    config={
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "architecture": "ResNet34",
        "epochs": num_epochs,
    },
    name=f"resnet_{timestamp}",
)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(
        train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"
    ):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100 * correct / total
    avg_train_loss = train_loss / len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(
            val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_dataloader)

    scheduler.step(avg_val_loss)

    # Logging
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% "
        f"| Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )

    # Log to TensorBoard
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Loss/Val", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_acc, epoch)
    writer.add_scalar("Accuracy/Val", val_acc, epoch)

    # Log to WandB
    run.log(
        {
            "Loss/Train": avg_train_loss,
            "Loss/Val": avg_val_loss,
            "Accuracy/Train": train_acc,
            "Accuracy/Val": val_acc,
        }
    )

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print("Saved new best model")

    artifact = wandb.Artifact(name=f"resnet_{timestamp}", type="model")
    artifact.add_file(best_model_path)
    artifact.add_file(split_save_path)
    run.log_artifact(artifact)

    # Early stopping check (based on validation loss)
    if args.early_stop:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        should_stop, improved = early_stopper.step(avg_val_loss)
        if improved and run is not None:
            run.summary["best_val_loss"] = early_stopper.best
        if should_stop:
            print(
                f"Early stopping: no val loss improvement in {args.early_stop_patience} epochs. "
                f"Best val loss: {early_stopper.best:.4f}"
            )
            if run is not None:
                run.summary["early_stopped"] = True
                run.summary["early_stop_epoch"] = epoch + 1
            break

writer.close()
run.finish()
print("Training complete")
