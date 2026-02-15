import torch
from diffusers import UNet2DConditionModel

from new_latent_diffusion.config import NUM_LATENT_CHANNELS, NUM_CLASSES

device = "cuda" if torch.cuda.is_available() else "cpu"

unet = UNet2DConditionModel(
    in_channels=NUM_LATENT_CHANNELS,
    out_channels=NUM_LATENT_CHANNELS,
    sample_size=32,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 768),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    num_class_embeds=NUM_CLASSES,
    class_embeddings_concat=False,
    norm_num_groups=32,
    time_embedding_act_fn="silu",
).to(device)
