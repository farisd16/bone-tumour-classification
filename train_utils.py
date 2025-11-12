import os
import json
from typing import Dict, Tuple, Optional, List

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from data.custom_dataset_class import CustomDataset


def make_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return training and validation transforms matching current pipeline."""
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
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize to mean=0 (and std=1 by default)
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return train_transform, val_transform


def make_minority_transform() -> transforms.Compose:
    """
    Stronger augmentation used only for minority classes.
    Keeps resize and normalization consistent with make_transforms().
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.7),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def default_minority_classes() -> List[str]:
    return [
        "giant cell tumor",
        "synovial osteochondroma",
        "osteofibroma",
    ]


def _stratified_indices(dataset_base: CustomDataset, test_size: float = 0.2, random_state: int = 42) -> Dict[str, np.ndarray]:
    """Create stratified train/val/test indices (80/10/10 split)."""
    targets = np.array([dataset_base.class_to_idx[label] for _, label in dataset_base.samples])
    indices = np.arange(len(dataset_base))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, temp_idx = next(sss.split(indices, targets))

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_rel, test_rel = next(sss_val.split(temp_idx, np.array(targets)[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_splits_and_loaders(
    image_dir: str,
    json_dir: str,
    run_dir: str,
    batch_size: int = 16,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_minority_aug: bool = False,
    minority_classes: Optional[List[str]] = None,
    minority_transform: Optional[transforms.Compose] = None,
):
    """Create stratified splits, save to run_dir, and return datasets, dataloaders, and split metadata."""
    # Base dataset for splitting (no transform)
    dataset_base = CustomDataset(image_dir=str(image_dir), json_dir=str(json_dir), transform=None)

    # Indices
    split_indices = _stratified_indices(dataset_base, test_size=test_size, random_state=random_state)

    # Save split
    split_save_path = os.path.join(run_dir, "data_split.json")
    with open(split_save_path, "w") as f:
        json.dump({k: v.tolist() for k, v in split_indices.items()}, f)

    # Transforms
    train_t, val_t = make_transforms()

    # Full datasets with transforms
    if apply_minority_aug:
        if minority_classes is None:
            minority_classes = default_minority_classes()
        if minority_transform is None:
            minority_transform = make_minority_transform()
        train_ds_full = CustomDataset(
            image_dir=str(image_dir),
            json_dir=str(json_dir),
            transform=train_t,
            minority_transform=minority_transform,
            minority_classes=minority_classes,
        )
    else:
        train_ds_full = CustomDataset(image_dir=str(image_dir), json_dir=str(json_dir), transform=train_t)
    val_ds_full = CustomDataset(image_dir=str(image_dir), json_dir=str(json_dir), transform=val_t)

    # Subsets
    train_dataset = Subset(train_ds_full, split_indices["train"])
    val_dataset = Subset(val_ds_full, split_indices["val"])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader, split_save_path, split_indices
