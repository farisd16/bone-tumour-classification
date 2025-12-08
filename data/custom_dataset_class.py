import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
import torch
import numpy as np


class CustomDataset(Dataset):
    def __init__(
        self,
        image_dir,
        json_dir,
        transform=None,
        minority_transform=None,
        minority_classes=None,
    ):
        """
        Args:
            image_dir (str): Folder containing patched X-ray images
            json_dir (str): Folder containing JSON annotations
            classes (list): List of unhealthy class names
            transform (callable, optional): torchvision transforms
        """
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.classes = [
            "osteochondroma",
            "osteosarcoma",
            "multiple osteochondromas",
            "simple bone cyst",
            "giant cell tumor",
            "synovial osteochondroma",
            "osteofibroma",
        ]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.transform = transform
        # Optional: apply different transform to specified minority classes
        self.minority_transform = minority_transform
        # store lowercased class names for comparison; default to empty set
        self.minority_classes = set([c.lower() for c in (minority_classes or [])])
        self.samples = []

        for fname in os.listdir(image_dir):
            if not fname.lower().endswith(".jpeg"):
                continue

            json_name = fname.replace(".jpeg", ".json")
            json_path = os.path.join(json_dir, json_name)

            with open(json_path, "r") as f:
                data = json.load(f)

            label = data["shapes"][0]["label"].lower()

            if label not in self.classes:
                continue

            img_path = os.path.join(image_dir, fname)

            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        # If a minority transform is provided and the sample's label is in the
        # configured minority set, apply that; else fall back to the default transform.
        if self.minority_transform is not None and label in self.minority_classes:
            image = self.minority_transform(image)
        elif self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )(image)

        label_idx = self.class_to_idx[label]
        return image, label_idx
