import os
import datetime

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm
import lpips

from config import WANDB_ENTITY, WANDB_PROJECT
from train_utils import build_splits_and_loaders
from new_latent_diffusion.config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    VAE_LR,
    VAE_EPOCHS,
    VAE_IMAGE_DIR,
    JSON_DIR,
    TEST_SPLIT_RATIO,
    XLSX_PATH,
    VAE_KL_BETA_MAX,
    VAE_LPIPS_WEIGHT,
    VAE_KL_WARMUP_EPOCHS,
)
from new_latent_diffusion.vae.model import vae


def train():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"vae_{timestamp}"
    run_dir = f"new_latent_diffusion/vae/runs/train_vae_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/model", exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
        ]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae.to(device)
    vae.train()

    lpips_loss = lpips.LPIPS(net="alex").to(device)
    lpips_loss.eval()

    _, _, _, train_loader, _, test_loader, _, _ = build_splits_and_loaders(
        image_dir=str(VAE_IMAGE_DIR),
        json_dir=str(JSON_DIR),
        run_dir=run_dir,
        test_size=TEST_SPLIT_RATIO,
        transform=transform,
        xlsx_path=str(XLSX_PATH),
        exclude_val=True,
    )

    with wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        config={
            "model": "VAE",
            "learning_rate": VAE_LR,
            "epochs": VAE_EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "kl_beta_max": VAE_KL_BETA_MAX,
            "kl_warmup_epochs": VAE_KL_WARMUP_EPOCHS,
            "lpips_weight": VAE_LPIPS_WEIGHT,
            "latent_channels": vae.config.latent_channels,
            "block_out_channels": vae.config.block_out_channels,
        },
        name=run_name,
    ) as run:
        vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=VAE_LR)

        for epoch in range(VAE_EPOCHS):
            vae.train()
            epoch_losses = []

            # KL warm-up
            beta = min(
                VAE_KL_BETA_MAX,
                (epoch + 1) / VAE_KL_WARMUP_EPOCHS * VAE_KL_BETA_MAX,
            )

            for images, _ in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{VAE_EPOCHS} [Train]"
            ):
                images = images.to(device)
                vae_optimizer.zero_grad()

                # Encode
                posterior = vae.encode(images)
                latents = posterior.latent_dist.mean

                # Decode
                recon = vae.decode(latents).sample

                # Losses
                l1_loss = F.l1_loss(recon, images)
                perceptual = lpips_loss(recon, images).mean()
                kl_loss = posterior.latent_dist.kl().mean()

                vae_loss = l1_loss + VAE_LPIPS_WEIGHT * perceptual + beta * kl_loss

                vae_loss.backward()
                vae_optimizer.step()

                epoch_losses.append(vae_loss.item())

            avg_train_loss = float(np.mean(epoch_losses))

            # -----------------------
            # Validation
            # -----------------------
            vae.eval()
            test_losses = []

            with torch.no_grad():
                for images, _ in tqdm(
                    test_loader, desc=f"Epoch {epoch + 1}/{VAE_EPOCHS} [Test]"
                ):
                    images = images.to(device)

                    posterior = vae.encode(images)
                    latents = posterior.latent_dist.mean
                    recon = vae.decode(latents).sample

                    l1_loss = F.l1_loss(recon, images)
                    perceptual = lpips_loss(recon, images).mean()
                    kl_loss = posterior.latent_dist.kl().mean()

                    vae_loss = l1_loss + VAE_LPIPS_WEIGHT * perceptual + beta * kl_loss

                    test_losses.append(vae_loss.item())

            avg_test_loss = float(np.mean(test_losses))

            run.log(
                {
                    "VAE Loss/Train": avg_train_loss,
                    "VAE Loss/Test": avg_test_loss,
                    "VAE KL Beta": beta,
                    "VAE L1 Loss": l1_loss.item(),
                    "VAE Perceptual Loss": perceptual.item(),
                    "VAE KL Loss": kl_loss.item(),
                    "VAE Epoch": epoch + 1,
                }
            )

            # -----------------------
            # Checkpointing
            # -----------------------
            if (epoch + 1) % 50 == 0:
                checkpoint_path = f"{run_dir}/model/vae_epoch_{epoch + 1}.pth"
                torch.save(vae.state_dict(), checkpoint_path)
                artifact = wandb.Artifact(
                    name=f"vae_checkpoint_epoch_{epoch + 1}",
                    type="model",
                )
                artifact.add_file(checkpoint_path)
                run.log_artifact(artifact)

        # Final model
        final_model_path = f"{run_dir}/model/vae_final.pth"
        torch.save(vae.state_dict(), final_model_path)
        final_artifact = wandb.Artifact(name="vae_final", type="model")
        final_artifact.add_file(final_model_path)
        run.log_artifact(final_artifact)


if __name__ == "__main__":
    train()
