import os
from pathlib import Path

import numpy as np

METRIC_TO_MONITOR = {
    # "VAE": "val_lpips",
    "VAE": None,
    "LDM": "ema_loss",
}
METRIC_MODE = {
    "VAE": "min",
    "LDM": "min",
}

SEED = int(np.random.randint(2147483647))
MAX_EPOCH = 800
IMAGE_SIZE = 256
BATCH_SIZE = {
    "train": 16,
    "val": 4,
    "test": 4,
}
NUM_CLASSES = 21
TEST_SPLIT_RATIO = 0.1
DATASET_DIR = os.path.join("/","vol", "miltank", "projects", "practical_wise2526", "bone-tumor-classification-gen-models", "bone-tumour-classification", "data", "dataset")
IMAGE_DIR = Path(DATASET_DIR) / "final_patched_BTXRD"
JSON_DIR = Path(DATASET_DIR) / "BTXRD" / "Annotations"
XLSX_PATH = Path(DATASET_DIR) / "BTXRD" / "dataset.xlsx"

LEARNING_RATE = 1e-4
DIFFUSION_STEP = 1000
BETA_START = 1e-4
BETA_END = 2e-2
SCALE_DOWN = 5
USE_EMA = "True"
CLASSIFIER_FREE_GUIDANCE_SCALE = (
    4  # @param {type:"slider", min:0, max:3, step:0.001}
) #original: 1.069
