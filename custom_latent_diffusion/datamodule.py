import lightning as L
import torch.utils.data as data
from torchvision.transforms.v2 import Compose, Lambda, ToTensor, Resize

from data.custom_dataset_class import CustomDataset
from train_utils import build_splits_and_loaders
from latent_diffusion.config import (
    IMAGE_DIR,
    JSON_DIR,
    BATCH_SIZE,
    TEST_SPLIT_RATIO,
    IMAGE_SIZE,
    XLSX_PATH,
)

transform = Compose(
    [
        Resize(IMAGE_SIZE),
        ToTensor(),
        Lambda(lambda x: (x * 2) - 1),
    ]
)


class BTXRDDataModule(L.LightningDataModule):
    def __init__(self, run_dir):
        super().__init__()
        self.split = ["train", "test"]
        self.train_dataset = None
        self.test_dataset = None
        self.run_dir = run_dir
        self.dataset = CustomDataset

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_dataset, _, test_dataset, _, _, _, _, _ = build_splits_and_loaders(
            image_dir=str(IMAGE_DIR),
            json_dir=str(JSON_DIR),
            run_dir=self.run_dir,
            test_size=TEST_SPLIT_RATIO,
            transform=transform,
            xlsx_path=str(XLSX_PATH),
            exclude_val=True,
        )

        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset

        if stage == "test":
            self.test_dataset = test_dataset

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE["train"],
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=BATCH_SIZE["test"],
            num_workers=2,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=BATCH_SIZE["test"],
            num_workers=2,
        )
