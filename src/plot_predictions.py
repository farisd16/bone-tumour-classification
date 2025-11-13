import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

report = np.load("checkpoints/resnet50/test_predictions.npy", allow_pickle=True)

TUMOR_COLS = [
    "osteochondroma",
    "osteosarcoma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "synovial osteochondroma",
    "osteofibroma",
]
# Class order corresponds to numerical labels 0..6
class_names = TUMOR_COLS

correct = 0
for _, gt, pred in report:
    correct += int(gt) == int(pred)
acc = correct / len(report)
print(f"Test Accuracy: {acc:.2%}  ({correct}/{len(report)} korrekt)")

n_show = min(8, len(report))
rows = 2
cols = (n_show + 1) // 2
fig, axes = plt.subplots(rows, cols, figsize=(14, 6))
axes = np.array(axes).ravel()

for i, (path, gt, pred) in enumerate(report[:n_show]):
    gt = int(gt)
    pred = int(pred)
    img = Image.open(path).convert("RGB")
    axes[i].imshow(img)
    color = "green" if gt == pred else "red"
    axes[i].set_title(f"GT: {class_names[gt]}\nPred: {class_names[pred]}", color=color)
    axes[i].axis("off")

# hide excess axes (if n_show is odd)
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()