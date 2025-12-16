import os
import datetime

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np


from train_utils import build_splits_and_loaders
from new_latent_diffusion.utils import plot_side_by_side
from new_latent_diffusion.config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    VAE_LR,
    VAE_EPOCHS,
    VAE_IMAGE_DIR,
    JSON_DIR,
    TEST_SPLIT_RATIO,
    XLSX_PATH,
)
from new_latent_diffusion.vae.model import vae


def train():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"new_latent_diffusion/vae/runs/train_vae_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, _, train_loader, _, test_loader, _, _ = build_splits_and_loaders(
        image_dir=str(VAE_IMAGE_DIR),
        json_dir=str(JSON_DIR),
        run_dir=run_dir,
        test_size=TEST_SPLIT_RATIO,
        transform=transform,
        xlsx_path=str(XLSX_PATH),
        exclude_val=True,
    )

    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=VAE_LR)
    for epoch in range(VAE_EPOCHS):
        losses = []
        for step, (images, _) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            vae_optimizer.zero_grad()

            # VAE forward pass
            posterior = vae.encode(images)
            outputs = vae.decode(posterior["latent_dist"].sample())
            # Compute VAE loss
            recon_loss = F.mse_loss(outputs.sample, images, reduction="mean")
            kl_loss = (
                posterior.latent_dist.kl() / (BATCH_SIZE * IMAGE_SIZE * IMAGE_SIZE)
            ).mean()
            vae_loss = recon_loss + 0.5 * kl_loss

            losses.append(vae_loss.item())
            vae_loss.backward()
            vae_optimizer.step()

        print(f"VAE Epoch {epoch + 1}, Step {step + 1}, Loss: {np.mean(losses):.4f}")
        plot_side_by_side(
            images, outputs.sample, posterior.latent_dist.sample(), epoch + 1, run_dir
        )

        with torch.no_grad():
            losses = []
            for step, (images, _) in enumerate(tqdm(test_loader)):
                images = images.to(device)

                # VAE forward pass
                posterior = vae.encode(images)
                outputs = vae.decode(posterior["latent_dist"].sample())
                # Compute VAE loss
                recon_loss = F.mse_loss(outputs.sample, images, reduction="mean")
                kl_loss = (posterior.latent_dist.kl() / (64 * 28 * 28)).mean()
                vae_loss = recon_loss + 0.5 * kl_loss

                losses.append(vae_loss.item())
        print(
            f"VAE Epoch {epoch + 1}, Step {step + 1}, Test Loss: {np.mean(losses):.4f}"
        )

    torch.save(vae.state_dict(), f"{run_dir}/model/vae_final.pth")


if __name__ == "__main__":
    train()
