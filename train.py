import os
import json
import datetime
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.custom_dataset_class import CustomDataset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import wandb

from config import WANDB_ENTITY, WANDB_PROJECT

import argparse
from utils import EarlyStopper
from train_utils import make_transforms, build_splits_and_loaders

# CLI args (early stopping)
parser = argparse.ArgumentParser()
parser.add_argument("--early-stop", action="store_true",                            # default is false, when called then true
                    help="Enable early stopping on validation loss")
parser.add_argument("--early-stop-patience", type=int, default=5,
                    help="Epochs without improvement before stopping")
parser.add_argument("--early-stop-min-delta", type=float, default=0.0,
                    help="Minimum improvement in val loss to reset patience")
parser.add_argument("--weighted-ce", action="store_true",                           # default is false, when called then true
                    help="Use class-weighted cross entropy loss")
parser.add_argument("--apply-minority-aug", action="store_true",                    # default is false, when called then true
                    help="Apply stronger augmentation only to minority classes")
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

DATASET_DIR = os.path.join("data", "dataset")
image_dir = (
    Path(DATASET_DIR) / "patched_BTXRD_merged"
)  # Folder might have to be changed
json_dir = Path(DATASET_DIR) / "BTXRD" / "Annotations"

# TensorBoard writer
writer = SummaryWriter(log_dir=run_dir)

train_transform, val_transform = make_transforms()

(
    train_dataset,
    val_dataset,
    train_dataloader,
    val_dataloader,
    split_save_path,
    split_indices,
) = build_splits_and_loaders(
    image_dir=str(image_dir),
    json_dir=str(json_dir),
    run_dir=run_dir,
    batch_size=16,
    test_size=0.2,
    random_state=42,
    apply_minority_aug=args.apply_minority_aug,
)

# =====================================Model================================================
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 7),
)
model.to(device)

# (Weighted) Cross Entropy Loss, Optimizer, Scheduler
weighted_cross_entropy = args.weighted_ce

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

lr = 5e-5
weight_decay = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# ===================================== Training loop ===========================================
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
