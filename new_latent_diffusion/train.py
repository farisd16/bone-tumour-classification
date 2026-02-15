import os
import datetime
import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from accelerate import Accelerator


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
    DENOISING_STEPS,
    UNET_LR,
    WARMUP_STEPS,
)
from new_latent_diffusion.model import unet
from new_latent_diffusion.vae.model import vae


def train(vae_model_path, unet_run_dir):
    vae.load_state_dict(torch.load(vae_model_path))
    vae.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    _, _, _, train_loader, _, test_loader, _, _ = build_splits_and_loaders(
        image_dir=str(VAE_IMAGE_DIR),
        json_dir=str(JSON_DIR),
        run_dir=unet_run_dir,
        test_size=TEST_SPLIT_RATIO,
        transform=transform,
        xlsx_path=str(XLSX_PATH),
        exclude_val=True,
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=DENOISING_STEPS)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=UNET_LR)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=(len(train_loader) * 100),
    )

    accelerator = Accelerator()
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )
    ema_model = EMAModel(model.parameters(), decay=0.9999, use_ema_warmup=True)
    for epoch in range(100):
        losses = []
        for step, (images, labels) in enumerate(tqdm(train_dataloader)):
            images = images.to(device)
            labels = labels.to(device)
            labels_raw = labels
            latents = vae.encode(images).latent_dist.sample()
            noise = torch.randn(latents.shape).to(device)
            bs = images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict the noise residual
            optimizer.zero_grad()
            noise_pred = unet(
                sample=noisy_images,
                timestep=timesteps,
                encoder_hidden_states=None,
                class_labels=labels,
            )
            noise_pred = noise_pred.sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            ema_model.step(model.parameters())

            losses.append(loss.item())

        print(f"Epoch: {epoch + 1}, Train loss: {np.mean(losses)}")
        # generate(vae, unet, noise_scheduler, epoch)

        losses = []
        with torch.no_grad():
            for step, (images, labels) in enumerate(tqdm(test_loader)):
                images = images.to(device)
                labels = labels.to(device)
                latents = vae.encode(images).latent_dist.sample()
                noise = torch.randn(latents.shape).to(device)
                bs = images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = noise_scheduler.add_noise(latents, noise, timesteps)

                noise_pred = unet(
                    sample=noisy_images,
                    timestep=timesteps,
                    encoder_hidden_states=None,
                    class_labels=labels,
                )
                noise_pred = noise_pred.sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                losses.append(loss.item())

        print(f"Epoch: {epoch + 1}, Test loss: {np.mean(losses)}")
    torch.save(unet.state_dict(), f"{unet_run_dir}/model/vae_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet model")
    parser.add_argument(
        "--run-name",
        required=True,
        help="The name of the folder inside of the runs folder to load the vae model from",
    )
    args = parser.parse_args()
    run_name = args.run_name

    vae_run_dir = f"new_latent_diffusion/vae/runs/{run_name}"
    vae_model_path = f"{vae_run_dir}/model/vae_final.pth"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unet_run_dir = f"new_latent_diffusion/runs/train_unet_{timestamp}"
    os.makedirs(unet_run_dir, exist_ok=True)

    train(vae_model_path, unet_run_dir)
