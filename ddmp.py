from torch.utils.data import DataLoader
from data.custom_dataset_class import CustomBoneDataset
import os, shutil
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from pathlib import Path

# === Paths ===
base_dir = os.path.dirname(__file__)
json_folder = os.path.join(base_dir, "dataset", "BTXRD", "Annotations")
image_folder = os.path.join(base_dir, "dataset", "BTXRD", "images")


source_dir = Path("./dataset/images")
target_dir = Path("./filtered_data/train")

target_classes = [
    "osteochondroma",
    "osteosarcoma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "synovial osteochondroma",
    "osteofibroma",
]

# create filtered folder structure
for cls in target_classes:
    (target_dir / cls).mkdir(parents=True, exist_ok=True)
    src_class_dir = source_dir / cls
    if not src_class_dir.exists():
        continue
    for img_file in src_class_dir.glob("*.png"):
        shutil.copy(img_file, target_dir / cls / img_file.name)



# ==== Diffusion model ====
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1  # grayscale
)


diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

# ==== Training loop ====

trainer = Trainer(
    diffusion,
    dataloader,
    train_batch_size=8,
    gradient_accumulate_every=2,
    train_lr=2e-4,
    train_num_steps=70000,
    save_and_sample_every=5000,
    results_folder='./results',
    amp=True
)


trainer.train()
