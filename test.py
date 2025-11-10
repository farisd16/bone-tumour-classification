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
import wandb


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Paths
base_dir = "checkpoints"
# TODO: Replace hardcoding with CLI argument
run_name = "resnet_2025-11-10_11-02-01"
run_dir = os.path.join(base_dir, run_name)
best_model_path = os.path.join(run_dir, "best_model.pth")

# Transformations (no augmentation, only normalization)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # TODO: Adjust the mean and std to match pretrained dataset
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

# Dataset
dataset_folder_path = os.path.join("data", "dataset")
dataset = CustomDataset(
    image_dir=os.path.join(dataset_folder_path, "patched_BTXRD"),
    json_dir=os.path.join(dataset_folder_path, "BTXRD", "Annotations"),
    transform=transform,
)

# Load split indices from training
with open(os.path.join(run_dir, "data_split.json"), "r") as f:
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


cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

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
