import argparse

from lightning.pytorch import Trainer

from latent_diffusion.datamodule import BTXRDDataModule
from latent_diffusion.vae.model import VAEWrapper

parser = argparse.ArgumentParser(description="Test autoencoder")
parser.add_argument(
    "--run-name",
    required=True,
    help="The name of the folder inside of the runs folder to load the model from",
)
args = parser.parse_args()
run_name = args.run_name

RUN_DIR = f"latent_diffusion/vae/runs/{run_name}"
CKPT_PATH = f"{RUN_DIR}/model/VAE_best.ckpt"


def _test_loop():
    trainer = Trainer(accelerator="auto", logger=False)
    model = VAEWrapper(run_dir=RUN_DIR)
    trainer.test(
        model=model,
        ckpt_path=CKPT_PATH,
        datamodule=BTXRDDataModule(run_dir=RUN_DIR),
    )


_test_loop()
