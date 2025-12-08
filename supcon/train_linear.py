import os
import json
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


# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ask for SupCon encoder path
encoder_base_dir = "checkpoints_supcon"

encoder_path = Path(encoder_base_dir) / "2025-12-07_15-01-11" / "encoder_supcon.pth"
split_path   = Path(encoder_base_dir) / "2025-12-07_15-01-11" / "split.json"

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"checkpoints_linear/{timestamp}"
os.makedirs(save_dir, exist_ok=True)

print(f"Saving classifier to: {save_dir}")

# Load split
with open(split_path, "r") as f:
    split = json.load(f)

train_idx = split["train"]   # same train indices as train_supcon

# Normal transforms
normal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])


# Dataset
DATASET_DIR = "data/dataset"
image_dir = Path(DATASET_DIR) / "final_patched_BTXRD"
json_dir  = Path(DATASET_DIR) / "BTXRD" / "Annotations"

full = CustomDataset(image_dir=str(image_dir), json_dir=str(json_dir), transform=normal_transform)
train_subset = Subset(full, train_idx)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

# Load encoder & freeze it

encoder = models.resnet34(weights=None)
encoder.fc = nn.Identity()  # remove classification head

encoder.load_state_dict(torch.load(encoder_path, map_location=device))
encoder.to(device)
encoder.eval()

# freeze encoder parameters
for p in encoder.parameters():
    p.requires_grad = False

# Linear classifier (trainable)
num_classes = 7
classifier = nn.Linear(512, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-3)


# Training loop
num_epochs = 20

for epoch in range(num_epochs):
    classifier.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)

        # Forward through frozen encoder
        with torch.no_grad():
            feats = encoder(images)  # 512-d vector

        # Train classifier
        logits = classifier(feats)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Acc={acc:.2f}%")


# Save classifier

clf_path = os.path.join(save_dir, "classifier.pth")
torch.save(classifier.state_dict(), clf_path)

print(f"Saved classifier to: {clf_path}")
