from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.num_bad = 0

    def step(self, current):
        improved = current < (self.best - self.min_delta)
        if improved:
            self.best = current
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience, improved


def display_confusion_matrix(all_labels, all_preds):
    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    classes = [
        "osteochondroma",
        "osteosarcoma",
        "multiple osteochondromas",
        "simple bone cyst",
        "giant cell tumor",
        "synovial osteochondroma",
        "osteofibroma",
    ]

    # --- Normalize by row (true label) ---
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    # --- Always show both count and percentage (no empty cells) ---
    annot_text = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_text[i, j] = f"{cm[i, j]}\n{cm_percent[i, j]:.1f}%"

    # --- Plot ---
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=annot_text,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (Test)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
