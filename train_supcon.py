import os
import json
import datetime
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

from data.custom_dataset_class import CustomDataset
from sup_contrastive import (
    TwoViewDataset,
    SupConModel,
    SupConLoss
)

from sklearn.model_selection import StratifiedShuffleSplit


"""
To train the encoder part
"""


# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"checkpoints_supcon/{timestamp}"
os.makedirs(save_dir, exist_ok=True)

print(f"Saving encoder to: {save_dir}")


# Data preparation
DATASET_DIR = os.path.join("data", "dataset")
image_dir = Path(DATASET_DIR) / "final_patched_BTXRD"
json_dir = Path(DATASET_DIR) / "BTXRD" / "Annotations"

# Create base dataset (no transform)
base_dataset = CustomDataset(image_dir=str(image_dir), json_dir=str(json_dir), transform=None)


# Stratified Split

targets = [label for _, label in base_dataset.samples]
targets = [base_dataset.class_to_idx[label] for label in targets]
targets = torch.tensor(targets)
indices = torch.arange(len(base_dataset))
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(indices, targets))

# Save split
split_dict = {
    "train": train_idx.tolist(),
    "test":  test_idx.tolist()
}

# Save split
with open(os.path.join(save_dir, "split.json"), "w") as f:
    json.dump(split_dict, f, indent=4)


# Contrastive augmentations
contrastive_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.GaussianBlur(kernel_size=3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# Create dataset & dataloader
train_full = CustomDataset(
    image_dir=str(image_dir),
    json_dir=str(json_dir),
    transform=None,
)

train_dataset = Subset(train_full, train_idx)

# Wrap with TwoViewDataset (2 augmentations per sample)
contrastive_dataset = TwoViewDataset(train_dataset, contrastive_transform)

train_loader = DataLoader(
    contrastive_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
)


# Model: Encoder + MLP projection head
resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()  # remove classifier

model = SupConModel(resnet, feature_dim=128).to(device)

# Loss + Optimizer
criterion = SupConLoss(temperature=0.1)
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

# Training loop (only contrastive loss)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)  # shape [B, 2, C, H, W]
        labels = labels.to(device)
        B = images.size(0)

        # reshape: [B*2, C, H, W]
        images = images.view(-1, *images.shape[2:])

        features = model(images)  # → [B*2, 128]
        features = features.view(B, 2, -1)

        loss = criterion(features, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Contrastive Loss = {avg_loss:.4f}")



# Save encoder
encoder_path = os.path.join(save_dir, "encoder_supcon.pth")
torch.save(model.encoder.state_dict(), encoder_path)
print(f"Saved encoder to: {encoder_path}")
