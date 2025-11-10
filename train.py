import os
import json
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from data.custom_dataset_class import CustomDataset


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
        # TODO: Adjust the mean and std to match pretrained dataset
        transforms.Normalize(
            mean=[0.5], std=[0.5]
        ),  # Normalize to mean=0 (and std=1 by default)
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # TODO: Adjust the mean and std to match pretrained dataset
        transforms.Normalize(
            mean=[0.5], std=[0.5]
        ),  # Normalize to mean=0 (and std=1 by default)
    ]
)


# Dataset
dataset_folder_path = os.path.join("data", "dataset")
dataset = CustomDataset(
    image_dir=os.path.join(dataset_folder_path, "patched_BTXRD"),
    json_dir=os.path.join(dataset_folder_path, "BTXRD", "Annotations"),
)

# Train val test dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
train_dataset.transform = train_transform
val_dataset.transform = val_transform

# Save the split indices
split_indices = {
    "train": train_dataset.indices,
    "val": val_dataset.indices,
    "test": test_dataset.indices,
}

split_path = os.path.join(run_dir, "data_split.json")
with open(split_path, "w") as f:
    json.dump(split_indices, f)

# Train val test dataloader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 7),
)
model.to(device)

# Loss, Optimizere, Scheduler
criterion = nn.CrossEntropyLoss()
lr = 1e-4
weight_decay = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# Training loop
num_epochs = 5
best_val_acc = 0.0

# WandB run
run = wandb.init(
    entity="faris-demirovic-tum-technical-university-of-munich",
    project="bone-tumor-classification",
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
        print("✅ Saved new best model")

artifact = wandb.Artifact(name=f"resnet_{timestamp}", type="model")
artifact.add_file(best_model_path)
artifact.add_file(split_path)
run.log_artifact(artifact)

writer.close()
run.finish()
print("Training complete")
