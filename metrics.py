import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def confusionMatrix(all_labels, all_preds, path_to_save):

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    classes = [
        "osteochondroma",
        "osteosarcoma",
        "multiple osteochondromas",
        "simple bone cyst",
        "giant cell tumor",
        "synovial osteochondroma",
        "osteofibroma"
    ]

    # --- Normalize by row (true label) ---
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    report = classification_report(all_labels, all_preds, target_names=classes, digits=3)
    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(path_to_save, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n")
        f.write(report)
    print(f"\nSaved classification report to: {report_path}")


    # --- Always show both count and percentage (no empty cells) ---
    annot_text = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_text[i, j] = f"{cm[i,j]}\n{cm_percent[i,j]:.1f}%"

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
        linecolor='white'
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (Test)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()