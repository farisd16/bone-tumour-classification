import os
import datetime

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

CKPT_PATH = None


def _train_loop():
    seed_everything(SEED, workers=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"latent_diffusion/vae/runs/train_vae_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    model = VAEWrapper(run_dir)

    callbacks = list()
    checkpoint = ModelCheckpoint(
        monitor=METRIC_TO_MONITOR["VAE"],
        dirpath=os.path.join(run_dir, "model"),
        mode=METRIC_MODE["VAE"],
        filename="VAE_best",
    )
    callbacks.append(checkpoint)

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCH,
        logger=False,
        callbacks=callbacks,
        log_every_n_steps=5,
    )
    trainer.fit(model, ckpt_path=CKPT_PATH, datamodule=BTXRDDataModule(run_dir))


_train_loop()
