import argparse
import os
import datetime

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from latent_diffusion.config import (
    MAX_EPOCH,
    SEED,
    METRIC_TO_MONITOR,
    METRIC_MODE,
)
from latent_diffusion.datamodule import BTXRDDataModule
from latent_diffusion.vae.model import VAEWrapper
from latent_diffusion.diffusion.model import LatentDiffusionWrapper

parser = argparse.ArgumentParser(description="Train diffusion model")
parser.add_argument(
    "--run-name",
    required=True,
    help="The name of the folder inside of the runs folder to load the autoencoder model from",
)
args = parser.parse_args()
run_name = args.run_name
VAE_RUN_DIR = f"latent_diffusion/vae/runs/{run_name}"
VAE_CKPT_PATH = f"{VAE_RUN_DIR}/model/VAE_best.ckpt"

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DIFFUSION_RUN_DIR = f"latent_diffusion/diffusion/runs/train_ldm_{timestamp}"
os.makedirs(DIFFUSION_RUN_DIR, exist_ok=True)
DIFFUSION_CKPT_PATH = None


def _train_loop():
    seed_everything(SEED, workers=True)

    vae_model = VAEWrapper.load_from_checkpoint(
        VAE_CKPT_PATH,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    ).model
    diffusion_model = LatentDiffusionWrapper(vae_model, DIFFUSION_RUN_DIR)

    callbacks = list()

    checkpoint = ModelCheckpoint(
        monitor=METRIC_TO_MONITOR["LDM"],
        dirpath=os.path.join(DIFFUSION_RUN_DIR, "model"),
        mode=METRIC_MODE["LDM"],
        filename="LDM_best",
    )
    callbacks.append(checkpoint)

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCH,
        logger=False,
        callbacks=callbacks,
        log_every_n_steps=5,
        # Add more Trainer arguments
        # precision="16-mixed",           # recommended for diffusion
        # deterministic=True,             # fixes nondeterminism
    )
    trainer.fit(
        diffusion_model,
        ckpt_path=DIFFUSION_CKPT_PATH,
        datamodule=BTXRDDataModule(DIFFUSION_RUN_DIR),
    )


_train_loop()
