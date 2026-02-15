import argparse

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from latent_diffusion.datamodule import BTXRDDataModule
from latent_diffusion.vae.model import VAEWrapper
from latent_diffusion.config import IMAGE_SIZE

parser = argparse.ArgumentParser(description="Sample from autoencoder")
parser.add_argument(
    "--run-name",
    required=True,
    help="The name of the folder inside of the runs folder to load the model from",
)
args = parser.parse_args()
run_name = args.run_name

RUN_DIR = f"latent_diffusion/vae/runs/{run_name}"
CKPT_PATH = f"{RUN_DIR}/model/VAE_best.ckpt"


# Load the best model
model = VAEWrapper.load_from_checkpoint(CKPT_PATH).model
model.eval()

# Get a sample image from the dataset
datamodule = BTXRDDataModule(run_dir=RUN_DIR)
datamodule.setup(stage="test")
test_loader = datamodule.test_dataloader()
image, cls = next(iter(test_loader))
random_idx = np.random.randint(0, len(image))
image = image[random_idx : random_idx + 1].to(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Label:", cls)

scale_factor = 8

noise = torch.randn_like(
    torch.randn(1, 3, IMAGE_SIZE // scale_factor, IMAGE_SIZE // scale_factor)
).to("cuda" if torch.cuda.is_available() else "cpu")
torch.nn.init.uniform_(noise, -1.0, 1.0)
print("Noise Range:", noise.min().cpu().numpy(), "to", noise.max().cpu().numpy())

for scale in np.linspace(0, 1, 10):
    x_min = 0.0
    x_max = 0.0
    x_hat_min = 0.0
    x_hat_max = 0.0
    z_min = 0.0
    z_max = 0.0

    # Reconstruct the image using the loaded model
    with torch.no_grad():
        print(f"\nNoise Level: {round(scale * 100.0, 2)}%")
        z = model.encode(image, inference=True)
        z = (1.0 - scale) * z + scale * noise
        z_min = round(float(z.min().cpu().numpy()), 3)
        z_max = round(float(z.max().cpu().numpy()), 3)
        print(f"Latent Image Range: {z_min} to {z_max}")
        x = model.decode(z)

    # Process the output image
    z = F.interpolate(z, scale_factor=4, mode="nearest")
    z = torch.clamp(z, -1.0, 1.0)
    latent_image = z.squeeze(0).permute(1, 2, 0).cpu().numpy()
    latent_image = (latent_image + 1) / 2
    latent_image = (latent_image * 255).astype("uint8")

    x_hat_min = round(float(x.min().cpu().numpy()), 3)
    x_hat_max = round(float(x.max().cpu().numpy()), 3)
    x = torch.clamp(x, -1.0, 1.0)
    reconstructed_image = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstructed_image = (reconstructed_image + 1) / 2
    reconstructed_image = (reconstructed_image * 255).astype("uint8")

    # Process the original image for display
    x_min = round(float(image.min().cpu().numpy()), 3)
    x_max = round(float(image.max().cpu().numpy()), 3)
    original_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    original_image = (original_image + 1) / 2
    original_image = (original_image * 255).astype("uint8")

    plt.clf()

    # Display images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title(f"Original Image\nRange = [{x_min}, {x_max}]")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(latent_image, cmap="gray")
    plt.title(
        f"Latent Image with the {round(scale * 100.0, 2)}% Noise Level\n"
        f"Range = [{z_min}, {z_max}]"
    )
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_image, cmap="gray")
    plt.title(f"Reconstructed Image\nRange = [{x_hat_min}, {x_hat_max}]")
    plt.axis("off")

    plt.tight_layout()

    plt.show()
