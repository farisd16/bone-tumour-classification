from pathlib import Path
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ensure project root is on sys.path when running as a script (python src/..)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data import build_dataloaders
EPOCHS = 100


# Dataset/split/dataloader utilities moved to data/data_utils.py


def training_resnet(
    model_name,
    early_stop: bool = False,
    patience: int = 10,
    min_delta: float = 0.0,
):
    images_root = PROJECT_ROOT / "data" / "patched_BTXRD"
    annotations_root = PROJECT_ROOT / "data" / "BTXRD" / "Annotations"
    datasets, loaders = build_dataloaders(images_root, annotations_root)

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
    best_val_loss = float("inf")
    no_improve_count = 0
    use_early_stopping = early_stop and len(datasets["validation"]) > 0 and patience > 0
    output_dir = PROJECT_ROOT / "checkpoints" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        if use_early_stopping and epoch == 0:
            print(
                f"Early stopping enabled (patience={patience}, min_delta={min_delta:.6f})"
            )
        stop_training = False
        for phase in ("train", "validation"):
            if len(datasets[phase]) == 0:
                print(f"Skipping {phase} phase: no samples available.")
                continue
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
            # Early stopping check on validation loss
            if phase == "validation" and use_early_stopping:
                if epoch_loss + min_delta < best_val_loss:
                    best_val_loss = epoch_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(
                            f"Early stopping triggered after {patience} epochs without validation loss improvement."
                        )
                        stop_training = True
                        break  # break out of phase loop immediately
        if stop_training:
            break

    final_path = output_dir / "final.pt"
    torch.save(model.state_dict(), final_path)

    # Testen wurde nach src/testing_ResNet.py ausgelagert


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training pipeline with a variant of the ResNet model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet34", "resnet18"],
        help="ResNet model variant to train with",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping based on validation loss",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Epochs without val loss improvement before stopping",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum required improvement in val loss to reset patience",
    )
    args = parser.parse_args()
    training_resnet(
        args.model,
        early_stop=args.early_stop,
        patience=args.patience,
        min_delta=args.min_delta,
    )
