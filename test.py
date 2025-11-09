import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.custom_dataset_class import CustomDataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import json 
import numpy as np
from metrics import confusionMatrix

# Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Paths 
base_dir = "checkpoints"
run_dir = os.path.join(base_dir, "run_2025-11-08_19-58-56")  
best_model_path = os.path.join(run_dir, "best_model.pth")

# Transformations (no augmentation, only normalization) 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Dataset 
dataset = CustomDataset(
    image_dir="/Users/bartu/Desktop/Bartu/RCI/3.Semester/ADLM/bone-tumour-classification/data/dataset/patched_BTXRD",
    json_dir="/Users/bartu/Desktop/Bartu/RCI/3.Semester/ADLM/bone-tumour-classification/data/dataset/BTXRD/Annotations",
    transform=transform
)

# Load split indices from training 
with open(os.path.join(run_dir,"data_split.json"), "r") as f:
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
model = model.to(device)
model.eval()


# Loss and metrics 
criterion = nn.CrossEntropyLoss()
test_loss = 0.0
correct = 0
total = 0

all_preds = []
all_labels = []


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