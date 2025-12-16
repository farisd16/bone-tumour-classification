import torch
from diffusers import AutoencoderKL

from new_latent_diffusion.config import NUM_LATENT_CHANNELS

device = "cuda" if torch.cuda.is_available() else "cpu"

vae = AutoencoderKL(
    in_channels=1,
    out_channels=1,
    latent_channels=NUM_LATENT_CHANNELS,
    sample_size=256,
    block_out_channels=(64, 128, 256, 512),
    down_block_types=(
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ),
    up_block_types=(
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ),
    norm_num_groups=32,
).to(device)
