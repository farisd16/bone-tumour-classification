import os
import datetime

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm

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

    with wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        config={
            "model": "VAE",
            "learning_rate": VAE_LR,
            "epochs": VAE_EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
        },
        name=run_name,
    ) as run:
        train_losses = []
        test_losses = []

        vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=VAE_LR)
        for epoch in range(VAE_EPOCHS):
            epoch_losses = []
            for step, (images, _) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{VAE_EPOCHS} [Train]")
            ):
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

                epoch_losses.append(vae_loss.item())
                vae_loss.backward()
                vae_optimizer.step()

            avg_train_loss = np.mean(epoch_losses)
            train_losses.append(avg_train_loss)

            with torch.no_grad():
                test_epoch_losses = []
                for step, (images, _) in enumerate(
                    tqdm(test_loader, desc=f"Epoch {epoch + 1}/{VAE_EPOCHS} [Test]")
                ):
                    images = images.to(device)

                    # VAE forward pass
                    posterior = vae.encode(images)
                    outputs = vae.decode(posterior["latent_dist"].sample())
                    # Compute VAE loss
                    recon_loss = F.mse_loss(outputs.sample, images, reduction="mean")
                    kl_loss = (
                        posterior.latent_dist.kl()
                        / (BATCH_SIZE * IMAGE_SIZE * IMAGE_SIZE)
                    ).mean()
                    vae_loss = recon_loss + 0.5 * kl_loss

                    test_epoch_losses.append(vae_loss.item())

            avg_test_loss = np.mean(test_epoch_losses)
            test_losses.append(avg_test_loss)

            # Log to wandb
            run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "test_loss": avg_test_loss,
                }
            )

            # Save model weights every 50 epochs
            if (epoch + 1) % 50 == 0:
                checkpoint_path = f"{run_dir}/model/vae_epoch_{epoch + 1}.pth"
                torch.save(vae.state_dict(), checkpoint_path)
                artifact = wandb.Artifact(
                    name=f"vae_checkpoint_epoch_{epoch + 1}",
                    type="model",
                )
                artifact.add_file(checkpoint_path)
                run.log_artifact(artifact)

        # Save final model
        final_model_path = f"{run_dir}/model/vae_final.pth"
        torch.save(vae.state_dict(), final_model_path)
        final_artifact = wandb.Artifact(name="vae_final", type="model")
        final_artifact.add_file(final_model_path)
        run.log_artifact(final_artifact)

        # Save loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, VAE_EPOCHS + 1), train_losses, label="Train Loss")
        plt.plot(range(1, VAE_EPOCHS + 1), test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("VAE Training and Test Loss")
        plt.legend()
        plt.grid(True)
        loss_plot_path = f"{run_dir}/loss_plot.png"
        plt.savefig(loss_plot_path)
        plt.close()

        # Log loss plot to wandb
        run.log({"loss_plot": wandb.Image(loss_plot_path)})


if __name__ == "__main__":
    train()
