from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPOCHS = 10


class BTXRDDataset(Dataset):
    def __init__(self, csv_file, images_root, transform=None):
        self.df = pd.read_csv(csv_file)
        self.images_root = Path(images_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_root / row["image_id"]
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)


def build_dataloaders(
    csv_root, images_root, batch_sizes=({"train": 32, "validation": 32, "test": 32})
):
    train_tfm = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
        ]
    )
    eval_tfm = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )

    datasets = {
        "train": BTXRDDataset(csv_root / "train.csv", images_root, train_tfm),
        "validation": BTXRDDataset(csv_root / "val.csv", images_root, eval_tfm),
        "test": BTXRDDataset(csv_root / "test.csv", images_root, eval_tfm),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_sizes["train"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_sizes["validation"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_sizes["test"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        ),
    }
    return datasets, loaders


def training_resnet(model_name):
    csv_root = PROJECT_ROOT / "data" / "BTXRD" / "splits"
    images_root = PROJECT_ROOT / "data" / "patched_BTXRD"
    datasets, loaders = build_dataloaders(csv_root, images_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match model_name:
        case "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, 7),  # 7 classes (0..6) according to create_csv.py
            )
        case "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 7),  # 7 classes (0..6) according to create_csv.py
            )
        case "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 7),  # 7 classes (0..6) according to create_csv.py
            )

    model.to(device)
    print(f"Using model {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=5e-3, weight_decay=2e-2, momentum=0.9)
    best_acc = 0.0
    output_dir = PROJECT_ROOT / "checkpoints" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/100")
        for phase in ("train", "validation"):
            model.train(phase == "train")
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, _ in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                preds = outputs.argmax(dim=1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum()

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])
            print(f"{phase:>11s} loss {epoch_loss:.4f}  acc {epoch_acc:.4f}")

            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), output_dir / "best.pt")

    torch.save(model.state_dict(), output_dir / "final.pt")

    # Testing
    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device))
    model.eval()

    softmax = nn.Softmax(dim=1)
    all_labels, all_preds, all_probs, all_paths = [], [], [], []

    with torch.no_grad():
        for inputs, labels, paths in loaders["test"]:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = softmax(outputs)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(
                probs[:, 1].cpu().numpy()
            )  # ROC for grade 1, adjust for 7 grades
            all_paths.extend(paths)

    acc = (np.array(all_labels) == np.array(all_preds)).mean()
    print(f"Test accuracy: {acc:.4f}")

    report = list(zip(all_paths, all_labels, all_preds))
    np.save(output_dir / "test_predictions.npy", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training pipeline with a variant of the ResNet model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50, resnet34", "resnet18"],
        help="ResNet model variant to train with",
    )
    args = parser.parse_args()
    training_resnet(args.model)
