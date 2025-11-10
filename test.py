import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from data.custom_dataset_class import CustomDataset
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    accuracy_score,
    recall_score,
    balanced_accuracy_score,
) 
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import json 
import numpy as np
from metrics import confusionMatrix
from pathlib import Path
import wandb

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Paths
base_dir = "checkpoints"
# TODO: Replace hardcoding with CLI argument
run_name = "resnet_2025-11-10_13-45-11" #run_2025-11-10_13-08-10 checkpoints/
run_dir = os.path.join(base_dir, run_name)  
best_model_path = os.path.join(run_dir, "best_model.pth")

# Transformations (no augmentation, only normalization) 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset (dynamic, repo-relative)
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" / "dataset"
image_dir = DATASET_DIR / "patched_BTXRD_merged"  #--------------------Folder might have to be changed----------------------------
json_dir = DATASET_DIR / "BTXRD" / "Annotations"

# Dataset
dataset_folder_path = os.path.join("data", "dataset")
dataset = CustomDataset(
    image_dir=str(image_dir),
    json_dir=str(json_dir),
    transform=transform
)

# Load split indices from training 
with open("data_split.json", "r") as f:
    split_indices = json.load(f)

test_indices = split_indices["test"]

# Create test subset
test_dataset = Subset(dataset, test_indices)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model
model = models.resnet34(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 7),
)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()


# Loss and metrics
criterion = nn.CrossEntropyLoss()
test_loss = 0.0
correct = 0
total = 0

all_preds = []
all_labels = []

api = wandb.Api()
runs = api.runs(
    "faris-demirovic-tum-technical-university-of-munich/bone-tumor-classification",
    filters={"display_name": run_name},
)
if len(runs) == 0:
    print("Error: No wandb runs found with given name")
    exit()
if len(runs) > 1:
    print("Warning: More than one wandb run found with given name")
run_id = runs[0].id
test_run = wandb.init(
    entity="faris-demirovic-tum-technical-university-of-munich",
    project="bone-tumor-classification",
    id=run_id,
    resume="must",
    job_type="test",
)

with torch.no_grad():
    for images, labels in tqdm(test_dataloader, desc="[Testing]"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

avg_test_loss = test_loss / len(test_dataloader)
test_acc = 100 * correct / total

print(f"\nTest Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")


confusionMatrix(all_labels, all_preds, run_dir)

precision_weighted = precision_score(all_labels, all_preds, average="weighted")
recall_weighted = recall_score(all_labels, all_preds, average="weighted")
f1_weighted = f1_score(all_labels, all_preds, average="weighted")
accuracy = accuracy_score(all_labels, all_preds)
balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

print("Weighted Precision:", precision_weighted)
print("Weighted Recall:", recall_weighted)
print("Weighted F1:", f1_weighted)
print("Accuracy:", accuracy)
print("Balanced Accuracy:", balanced_accuracy)

test_run.log(
    {
        "Weighted Precision": precision_weighted,
        "Weighted Recall": recall_weighted,
        "Weighted F1": f1_weighted,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
    }
)

test_run.finish()