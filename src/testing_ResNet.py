from pathlib import Path
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ensure project root is on sys.path when running as a script (python src/..)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data import build_dataloaders, CLASS_NAMES


def evaluate_resnet(model_name: str):
    # Prefer merged patched images; fallback to original patched folder if missing
    images_root = PROJECT_ROOT / "data" / "dataset" / "patched_BTXRD_merged"
    if not images_root.exists():
        fallback = PROJECT_ROOT / "data" / "dataset" / "patched_BTXRD"
        if fallback.exists():
            print(
                f"[WARN] {images_root} not found. Falling back to {fallback}"
            )
            images_root = fallback
    annotations_root = PROJECT_ROOT / "data" / "dataset" / "BTXRD" / "Annotations"
    datasets, loaders = build_dataloaders(images_root, annotations_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match model_name:
        case "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, 7),
            )
        case "resnet34":
            model = models.resnet34(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 7),
            )
        case "resnet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 7),
            )

    model.to(device)
    output_dir = PROJECT_ROOT / "checkpoints" / model_name
    best_path = output_dir / "best.pt"
    final_path = output_dir / "final.pt"
    load_path = best_path if best_path.exists() else final_path
    if not load_path.exists():
        raise FileNotFoundError(f"No weights found at {load_path}. Train first.")

    try:
        state_dict = torch.load(load_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if len(datasets["test"]) == 0:
        print("Skipping test evaluation: no samples available.")
        return

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
            all_probs.extend(probs.max(dim=1).values.cpu().numpy()) # all_probs.extend(probs[:, 1].cpu().numpy())
            all_paths.extend(paths)

    acc = (np.array(all_labels) == np.array(all_preds)).mean()
    print(f"Test accuracy: {acc:.4f}")

    report = list(zip(all_paths, all_labels, all_preds))
    np.save(output_dir / "test_predictions.npy", report)

    # Confusion matrix
    num_classes = len(CLASS_NAMES)
    labels_array = np.array(all_labels, dtype=int)
    preds_array = np.array(all_preds, dtype=int)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels_array, preds_array):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    lines = ["Per-class recall (true positive rate):"]
    for i, name in enumerate(CLASS_NAMES):
        total = cm[i].sum()
        correct = cm[i, i]
        if total > 0:
            lines.append(f"  {name}: {correct/total:.4%} ({correct}/{total})")
        else:
            lines.append(f"  {name}: n/a (no samples)")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(1.2 * num_classes, 1.0 * num_classes))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Test)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        perc = np.where(row_sums > 0, cm / row_sums, 0.0)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm[i, j]
            pct = perc[i, j] * 100.0
            text_color = "white" if value > thresh else "black"
            ax.text(
                j,
                i,
                f"{value}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained ResNet model")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet34", "resnet18"],
        help="ResNet model variant to evaluate",
    )
    args = parser.parse_args()
    evaluate_resnet(args.model)
