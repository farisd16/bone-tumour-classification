import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
import datetime
from data.custom_dataset_class import CustomDataset
from collections import Counter
from torch.utils.data import DataLoader
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Folder structure and Tensorboard
base_dir = "checkpoints"
os.makedirs(base_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(base_dir, f"run_weighted_cross_entropy{timestamp}")
os.makedirs(run_dir, exist_ok=True)

writer = SummaryWriter(log_dir=run_dir)


# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),    
    
    # Color jitter: brightness, contrast, saturation, hue
    transforms.ColorJitter(
        brightness=0.2,     # ±20% brightness variation
        contrast=0.2,       # ±20% contrast variation
        saturation=0.2,     # ±20% saturation variation
        hue=0.1             # ±0.1 hue shift
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Random perspective distortion (scale 0.2, probability 0.5)
    transforms.GaussianBlur(kernel_size=3),   
    transforms.ToTensor(),
        
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to mean=0 (and std=1 by default)
])


# Dataset
dataset = CustomDataset(
    image_dir="/Users/bartu/Desktop/Bartu/RCI/3.Semester/ADLM/bone-tumour-classification/data/dataset/patched_BTXRD",
    json_dir="/Users/bartu/Desktop/Bartu/RCI/3.Semester/ADLM/bone-tumour-classification/data/dataset/BTXRD/Annotations",
    transform=transform
)

# Dataset length
total_len = len(dataset)
train_size = int(0.8 * total_len)
val_size   = int(0.10 * total_len)
test_size  = total_len - train_size - val_size

# Train val test dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Save the split indices
split_indices = {
    "train": train_dataset.indices,
    "val": val_dataset.indices,
    "test": test_dataset.indices
}

split_path = os.path.join(run_dir, "data_split.json")
with open(split_path, "w") as f:
    json.dump(split_indices, f)

# Train val test dataloader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Model
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 7),  
)

# Weighted Cross Entropy Loss, Optimizer, Scheduler

targets = [label for _, label in train_dataset]  
class_counts = Counter(targets)
weights = 1.0 / np.array([class_counts[i] for i in range(7)])
weights = weights / weights.sum() * 7
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# Training loop
num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
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

    # === Validation ===
    model.eval()
    val_loss = 0.0
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
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
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% "
          f"| Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Log to TensorBoard
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Loss/Val", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_acc, epoch)
    writer.add_scalar("Accuracy/Val", val_acc, epoch)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{run_dir}/best_model.pth")
        print("✅ Saved new best model")

writer.close()
print("Training complete")
