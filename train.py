import argparse
import datetime
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
import wandb

from config import WANDB_ENTITY, WANDB_PROJECT
from train_utils import EarlyStopper, build_splits_and_loaders, FocalLoss


DEFAULT_CONFIG: Dict[str, Any] = {
    "architecture": "ResNet34",
    "learning_rate": 5e-5,
    "weight_decay": 1e-5,
    "batch_size": 16,
    "epochs": 30,
    "dropout": 0.5,
    "loss_fn": "ce",
    "focal_gamma": 2.0,
    "apply_minority_aug": False,
    "early_stop": False,
    "early_stop_patience": 5,
    "early_stop_min_delta": 0.0,
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "test_size": 0.2,
    "random_state": 42,
    "num_classes": 7,
    "run_name_prefix": "resnet",
}


def parse_cli_args() -> argparse.Namespace:
    """Collect command-line options for training script."""
    parser = argparse.ArgumentParser()

    # Optimization hyperparameters
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Learning rate for the optimizer")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"],
                        help="Weight decay for the optimizer")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Batch size for training and validation")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"],
                        help="Dropout probability before the final classification head")
    parser.add_argument("--scheduler-factor", type=float, default=DEFAULT_CONFIG["scheduler_factor"],
                        help="Multiplicative factor for ReduceLROnPlateau")
    parser.add_argument("--scheduler-patience", type=int, default=DEFAULT_CONFIG["scheduler_patience"],
                        help="Epochs to wait before reducing LR")

    # Dataset and splits
    parser.add_argument("--test-size", type=float, default=DEFAULT_CONFIG["test_size"],
                        help="Hold-out ratio for validation/test split")
    parser.add_argument("--random-state", type=int, default=DEFAULT_CONFIG["random_state"],
                        help="Random seed for data splits")

    # Loss configuration
    parser.add_argument(
        "--loss-fn", # ce = CrossEntropy Loss | wce = WeightedCrossEntropy Loss | focal = Focal Loss | wfocal = WeightedFocal Loss
        choices=["ce", "wce", "focal", "wfocal"],
        default=DEFAULT_CONFIG["loss_fn"],
        help="Loss to optimize: cross entropy variants or focal loss",
    )
    parser.add_argument("--focal-gamma", type=float, default=DEFAULT_CONFIG["focal_gamma"],
                        help="Gamma focusing parameter when using focal loss")

    # Regularization / augmentation toggles
    parser.add_argument("--apply-minority-aug", action="store_true",
                        help="Apply stronger augmentation only to minority classes")

    # Early stopping
    parser.add_argument("--early-stop", action="store_true",
                        help="Enable early stopping on validation loss")
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_CONFIG["early_stop_patience"],
                        help="Epochs without improvement before stopping")
    parser.add_argument("--early-stop-min-delta", type=float, default=DEFAULT_CONFIG["early_stop_min_delta"],
                        help="Minimum improvement in val loss to reset patience")

    # Bookkeeping
    parser.add_argument("--run-name-prefix", type=str, default=DEFAULT_CONFIG["run_name_prefix"],
                        help="Prefix for checkpoint/run folders")
    parser.add_argument("--num-classes", type=int, default=DEFAULT_CONFIG["num_classes"],
                        help="Number of output classes")

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert CLI namespace into a config dict for wandb sweeps."""
    return {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "loss_fn": args.loss_fn,
        "focal_gamma": args.focal_gamma,
        "apply_minority_aug": args.apply_minority_aug,
        "early_stop": args.early_stop,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "num_classes": args.num_classes,
        "run_name_prefix": args.run_name_prefix,
        "architecture": DEFAULT_CONFIG["architecture"],
    }


def train(config: Optional[Dict[str, Any]] = None) -> float:
    base_config = DEFAULT_CONFIG.copy()
    if config:
        base_config.update(config)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name_prefix = base_config.get("run_name_prefix", DEFAULT_CONFIG["run_name_prefix"])
    run_name = f"{run_name_prefix}_{timestamp}"

    checkpoints_base_dir = "checkpoints"
    os.makedirs(checkpoints_base_dir, exist_ok=True)
    run_dir = os.path.join(checkpoints_base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    dataset_dir = os.path.join("data", "dataset")
    image_dir = Path(dataset_dir) / "patched_BTXRD_merged"
    json_dir = Path(dataset_dir) / "BTXRD" / "Annotations"

    with wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        config=base_config,
        name=run_name,
    ) as run:
        cfg = wandb.config
        writer = SummaryWriter(log_dir=run_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            (
                train_dataset,
                val_dataset,
                train_dataloader,
                val_dataloader,
                split_save_path,
                _,
            ) = build_splits_and_loaders(
                image_dir=str(image_dir),
                json_dir=str(json_dir),
                run_dir=run_dir,
                batch_size=int(cfg.batch_size),
                test_size=float(cfg.test_size),
                random_state=int(cfg.random_state),
                apply_minority_aug=bool(cfg.apply_minority_aug),
            )

            architecture = cfg.architecture
            if architecture != "ResNet34":
                raise ValueError(f"Unsupported architecture: {architecture}")
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            model.fc = nn.Sequential(
                nn.Dropout(float(cfg.dropout)),
                nn.Linear(512, int(cfg.num_classes)),
            )
            model.to(device)

            loss_choice = cfg.loss_fn
            use_class_weights = loss_choice in {"wce", "wfocal"}
            class_weights = None

            if use_class_weights:
                targets = [label for _, label in train_dataset]
                class_counts = Counter(targets)
                weights = 1.0 / np.array([class_counts[i] for i in range(int(cfg.num_classes))])
                weights = weights / weights.sum() * int(cfg.num_classes)
                class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

            if loss_choice in {"focal", "wfocal"}:
                criterion = FocalLoss(gamma=float(cfg.focal_gamma), alpha=class_weights)
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)

            optimizer = optim.Adam(
                model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay)
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(cfg.scheduler_factor),
                patience=int(cfg.scheduler_patience),
            )

            num_epochs = int(cfg.epochs)
            best_val_acc = 0.0
            best_epoch = 0
            best_model_path = os.path.join(run_dir, "best_model.pth")
            best_model_saved = False

            if bool(cfg.early_stop):
                best_val_loss = float("inf")
                early_stopper = EarlyStopper(
                    patience=int(cfg.early_stop_patience), min_delta=float(cfg.early_stop_min_delta)
                )
            else:
                best_val_loss = None
                early_stopper = None

            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0

                for images, labels in tqdm(
                    train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"
                ):
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                train_acc = 100 * correct / total if total > 0 else 0.0
                avg_train_loss = train_loss / max(len(train_dataloader), 1)

                model.eval()
                val_loss = 0.0
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for images, labels in tqdm(
                        val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"
                    ):
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()

                val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
                avg_val_loss = val_loss / max(len(val_dataloader), 1)

                scheduler.step(avg_val_loss)

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                    f"| Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
                )

                writer.add_scalar("Loss/Train", avg_train_loss, epoch)
                writer.add_scalar("Loss/Val", avg_val_loss, epoch)
                writer.add_scalar("Accuracy/Train", train_acc, epoch)
                writer.add_scalar("Accuracy/Val", val_acc, epoch)

                run.log(
                    {
                        "Loss/Train": avg_train_loss,
                        "Loss/Val": avg_val_loss,
                        "Accuracy/Train": train_acc,
                        "Accuracy/Val": val_acc,
                    }
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), best_model_path)
                    best_model_saved = True
                    print("Saved new best model")

                if early_stopper:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                    should_stop, improved = early_stopper.step(avg_val_loss)
                    if improved:
                        run.summary["best_val_loss"] = early_stopper.best
                    if should_stop:
                        message = (
                            f"Early stopping: no val loss improvement in {cfg.early_stop_patience} epochs. "
                            f"Best val loss: {early_stopper.best:.4f}"
                        )
                        print(message)
                        run.summary["early_stopped"] = True
                        run.summary["early_stop_epoch"] = epoch + 1
                        break

            if best_model_saved:
                artifact = wandb.Artifact(name=f"{run_name}_best", type="model")
                artifact.add_file(best_model_path)
                artifact.add_file(split_save_path)
                run.log_artifact(artifact)

            run.summary["best_val_acc"] = best_val_acc
            run.summary["best_val_epoch"] = best_epoch

            print("Training complete")
            return best_val_acc
        finally:
            writer.close()


def main() -> None:
    args = parse_cli_args()
    cli_config = args_to_config(args)
    train(cli_config)


if __name__ == "__main__":
    main()
