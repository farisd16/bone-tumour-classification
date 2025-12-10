import argparse

import torch
from torchvision.transforms.functional import to_pil_image

from latent_diffusion.vae.model import VAEWrapper
from latent_diffusion.diffusion.model import LatentDiffusionWrapper
from latent_diffusion.config import USE_EMA, CLASSIFIER_FREE_GUIDANCE_SCALE
from latent_diffusion.utils import animate_result

classes = [
    "osteochondroma",
    "osteosarcoma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "synovial osteochondroma",
    "osteofibroma",
]
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

parser = argparse.ArgumentParser(description="Test autoencoder")
parser.add_argument(
    "--class-name",
    required=True,
    choices=classes + ["None"],
    help="The name of the folder inside of the runs folder to load the model from",
)
parser.add_argument(
    "--vae-run-name",
    required=True,
    help="The name of the folder inside of the runs folder to load the VAE model from",
)
parser.add_argument(
    "--ldm-run-name",
    required=True,
    help="The name of the folder inside of the runs folder to load the diffusion model from",
)
args = parser.parse_args()
class_name = args.class_name
vae_run_name = args.vae_run_name
ldm_run_name = args.ldm_run_name

VAE_RUN_DIR = f"latent_diffusion/vae/runs/{vae_run_name}"
VAE_CKPT_PATH = f"{VAE_RUN_DIR}/model/VAE_best.ckpt"

LDM_RUN_DIR = f"latent_diffusion/diffusion/runs/{ldm_run_name}"
LDM_CKPT_PATH = f"{LDM_RUN_DIR}/model/LDM_best.ckpt"

vae_model = VAEWrapper.load_from_checkpoint(
    VAE_CKPT_PATH,
    map_location=("cuda" if torch.cuda.is_available() else "cpu"),
).model

model = LatentDiffusionWrapper.load_from_checkpoint(
    LDM_CKPT_PATH,
    map_location=("cuda" if torch.cuda.is_available() else "cpu"),
    vae_model=vae_model,
)

context = None if class_name == "None" else class_to_idx[class_name]
USE_EMA = True if USE_EMA == "True" else False

progress, progress_decoded = model(
    context,
    inference=True,
    use_ema=USE_EMA,
    cfg_scale=CLASSIFIER_FREE_GUIDANCE_SCALE,
)

animate_result(
    progress_decoded,
    caption=f"LDM Sampling Progress for class: {class_name}",
    run_name=ldm_run_name,
    class_name=class_name,
)

# Save the final decoded sample as an image (CHW tensor -> PNG)
last_img = progress_decoded[-1].squeeze(0).cpu()
last_img = torch.clamp(last_img, -1.0, 1.0)
last_img = (last_img + 1.0) / 2.0
pil_img = to_pil_image(last_img)
save_path = f"{LDM_RUN_DIR}/sample_final_{class_name}.png"
pil_img.save(save_path)
print(f"Saved final {class_name} sample to {save_path}")
