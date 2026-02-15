"""
Evaluation Script for SupCon Encoder with Linear Classifier.

This script evaluates a pretrained Supervised Contrastive (SupCon) encoder
combined with a trained linear classifier on the BTXRD dataset test split.
"""


import os
import json
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



# Pretrained encoder & classifier

encoder_base_dir = "checkpoints_supcon"
classifier_base_dir = "checkpoints_linear"

encoder_path = Path(encoder_base_dir) / "2025-12-07_15-01-11" / "encoder_supcon.pth"
classifier_path = Path(classifier_base_dir) / "2025-12-07_17-40-30" / "classifier.pth"

split_path   = Path(encoder_base_dir) / "2025-12-07_15-01-11" / "split.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load split
with open(split_path, "r") as f:
    split = json.load(f)

test_idx = split.get("test")


# Create test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

DATASET_DIR = "data/dataset"
image_dir = Path(DATASET_DIR) / "final_patched_BTXRD"
json_dir  = Path(DATASET_DIR) / "BTXRD" / "Annotations"

full_data = CustomDataset(str(image_dir), str(json_dir), transform=transform)
test_dataset = Subset(full_data, test_idx)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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

# Evaluation
correct = 0
total = 0

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing SupCon"):
        images = images.to(device)
        labels = labels.to(device)

        feats = encoder(images)
        logits = classifier(feats)
        _, preds = logits.max(1)

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())



display_confusion_matrix(all_labels, all_preds)
precision_weighted = precision_score(all_labels, all_preds, average="weighted")
recall_weighted = recall_score(all_labels, all_preds, average="weighted")
f1_weighted = f1_score(all_labels, all_preds, average="weighted")
accuracy = accuracy_score(all_labels, all_preds)
balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

# Init wandb
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=f"supcon-eval-{encoder_path.parent.name}"
)

wandb.log({
    "weighted_precision": precision_weighted,
    "weighted_recall": recall_weighted,
    "weighted_f1": f1_weighted,
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy
})

wandb.finish()