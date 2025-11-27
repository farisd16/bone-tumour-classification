import argparse
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils as vutils
from tqdm import tqdm


TUMOR_TYPE_COLUMNS: Dict[str, str] = {
    "osteochondroma": "osteochondroma",
    "multiple_osteochondromas": "multiple osteochondromas",
    "simple_bone_cyst": "simple bone cyst",
    "giant_cell_tumor": "giant cell tumor",
    "osteofibroma": "osteofibroma",
    "synovial_osteochondroma": "synovial osteochondroma",
    "osteosarcoma": "osteosarcoma",
}

ANATOMY_PRIORITY: List[Tuple[str, str]] = [
    ("hand", "hand"),
    ("foot", "foot"),
    ("upper limb", "upper_limb"),
    ("lower limb", "lower_limb"),
    ("pelvis", "pelvis"),
]

VIEW_COLUMNS: List[str] = ["frontal", "lateral", "oblique"]
IMAGE_EXTS = {".jpeg", ".jpg", ".png"}


def _read_metadata_table(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    suffix = metadata_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(metadata_path)
    if suffix == ".csv":
        return pd.read_csv(metadata_path)
    raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")


def _column_value(row: pd.Series, column: str) -> float:
    value = row.get(column, 0)
    if pd.isna(value):
        return 0.0
    return float(value)


def _extract_tumor_type(row: pd.Series) -> Optional[str]:
    for key, column in TUMOR_TYPE_COLUMNS.items():
        if _column_value(row, column) >= 0.5:
            return key
    return None


def _extract_anatomy(row: pd.Series) -> Optional[str]:
    for column, label in ANATOMY_PRIORITY:
        if _column_value(row, column) >= 0.5:
            return label
    return None


def _extract_view(row: pd.Series) -> Optional[str]:
    for column in VIEW_COLUMNS:
        if _column_value(row, column) >= 0.5:
            return column
    return None


def build_samples_from_metadata(
    metadata_path: Path,
    image_root: Path,
    image_column: str = "image_id",
) -> List[Dict[str, str]]:
    df = _read_metadata_table(metadata_path)
    if image_column not in df.columns:
        raise KeyError(f"Column '{image_column}' missing from {metadata_path}")

    df = df[df[image_column].notna()].copy()
    df[image_column] = df[image_column].astype(str).str.strip()

    available_images = {
        p.name for p in image_root.iterdir() if p.suffix.lower() in IMAGE_EXTS
    }

    samples: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        filename = row[image_column]
        if filename not in available_images:
            continue

        tumor_type = _extract_tumor_type(row)
        anatomy = _extract_anatomy(row)
        view = _extract_view(row)
        if tumor_type is None or anatomy is None or view is None:
            continue

        samples.append(
            {"filename": filename, "tumor_type": tumor_type, "anatomy": anatomy, "view": view}
        )

    if not samples:
        raise RuntimeError("No valid samples found after filtering metadata and images.")

    return samples


def summarize_samples(samples: List[Dict[str, str]]) -> None:
    tumor_counter = Counter(s["tumor_type"] for s in samples)
    anatomy_counter = Counter(s["anatomy"] for s in samples)
    view_counter = Counter(s["view"] for s in samples)

    print("Sample counts (tumor type):")
    for key, count in tumor_counter.most_common():
        print(f"  {key:>25s}: {count}")

    print("\nSample counts (anatomy):")
    for key, count in anatomy_counter.most_common():
        print(f"  {key:>25s}: {count}")

    print("\nSample counts (view):")
    for key, count in view_counter.most_common():
        print(f"  {key:>25s}: {count}")


def _one_hot(index: int, num_classes: int) -> torch.Tensor:
    vec = torch.zeros(num_classes, dtype=torch.float32)
    vec[index] = 1.0
    return vec


class TumorGanDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, str]],
        image_root: Path,
        img_size: int,
        tumor_type_to_idx: Dict[str, int],
        anatomy_to_idx: Dict[str, int],
        view_to_idx: Dict[str, int],
    ):
        self.samples = samples
        self.image_root = image_root
        self.img_size = img_size

        self.tumor_type_to_idx = tumor_type_to_idx
        self.anatomy_to_idx = anatomy_to_idx
        self.view_to_idx = view_to_idx

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.cond_vectors = torch.stack(
            [self.encode_condition(sample) for sample in self.samples], dim=0
        )

    def __len__(self) -> int:
        return len(self.samples)

    def encode_condition(self, sample: Dict[str, str]) -> torch.Tensor:
        tumor_vec = _one_hot(self.tumor_type_to_idx[sample["tumor_type"]], len(self.tumor_type_to_idx))
        anatomy_vec = _one_hot(self.anatomy_to_idx[sample["anatomy"]], len(self.anatomy_to_idx))
        view_vec = _one_hot(self.view_to_idx[sample["view"]], len(self.view_to_idx))
        return torch.cat([tumor_vec, anatomy_vec, view_vec], dim=0)

    def get_condition_batch(self, indices: Sequence[int]) -> torch.Tensor:
        return torch.stack([self.cond_vectors[int(i)] for i in indices], dim=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img_path = self.image_root / sample["filename"]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        return image, self.cond_vectors[idx]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_ch))
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if use_bn:
            layers.insert(-1, nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class GeneratorUNetCond(nn.Module):
    def __init__(
        self,
        noise_channels: int = 1,
        img_channels: int = 1,
        cond_dim: int = 16,
        base_ch: int = 64,
    ):
        super().__init__()
        self.noise_channels = noise_channels
        self.img_channels = img_channels
        self.cond_dim = cond_dim

        in_ch = noise_channels + cond_dim

        self.down1 = DownBlock(in_ch, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        self.up4 = UpBlock(base_ch * 16, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch)

        self.final = nn.Conv2d(base_ch, img_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, noise_img: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = noise_img.shape
        cond_map = cond_vec.view(bsz, self.cond_dim, 1, 1).expand(-1, -1, height, width)
        x = torch.cat([noise_img, cond_map], dim=1)

        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)

        x = self.bottleneck(x)

        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        x = self.final(x)
        return self.tanh(x)


class PatchDiscriminatorCond(nn.Module):
    def __init__(self, img_channels: int = 1, cond_dim: int = 16, base_ch: int = 64):
        super().__init__()
        in_ch = img_channels + cond_dim

        def disc_block(in_f: int, out_f: int, normalize: bool = True) -> List[nn.Module]:
            layers: List[nn.Module] = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers: List[nn.Module] = []
        layers += disc_block(in_ch, base_ch, normalize=False)
        layers += disc_block(base_ch, base_ch * 2)
        layers += disc_block(base_ch * 2, base_ch * 4)
        layers += disc_block(base_ch * 4, base_ch * 8)
        self.model = nn.Sequential(*layers)
        self.final = nn.Conv2d(base_ch * 8, 1, 4, padding=1)

    def forward(self, img: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = img.shape
        cond_map = cond_vec.view(bsz, -1, 1, 1).expand(-1, -1, height, width)
        x = torch.cat([img, cond_map], dim=1)
        x = self.model(x)
        x = self.final(x)
        return x


def save_sample_grid(
    generator: GeneratorUNetCond,
    noise: torch.Tensor,
    cond_vec: torch.Tensor,
    sample_dir: Path,
    epoch: int,
    nrow: int,
) -> None:
    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(noise, cond_vec)
    sample_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_dir / f"epoch_{epoch:04d}.png"
    vutils.save_image(fake_imgs, out_path, nrow=nrow, normalize=True, value_range=(-1, 1))
    generator.train()


def save_checkpoint(
    generator: GeneratorUNetCond,
    discriminator: PatchDiscriminatorCond,
    output_dir: Path,
    epoch: int,
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), ckpt_dir / f"generator_epoch_{epoch:04d}.pth")
    torch.save(discriminator.state_dict(), ckpt_dir / f"discriminator_epoch_{epoch:04d}.pth")


def train_gan(
    generator: GeneratorUNetCond,
    discriminator: PatchDiscriminatorCond,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr_g: float,
    lr_d: float,
    noise_channels: int,
    sample_dir: Optional[Path],
    sample_interval: int,
    fixed_noise: Optional[torch.Tensor],
    fixed_conditions: Optional[torch.Tensor],
    nrow: int,
    checkpoint_interval: int,
    output_dir: Path,
) -> Dict[str, List[float]]:
    generator.to(device)
    discriminator.to(device)

    optim_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    adv_criterion = nn.BCEWithLogitsLoss()

    history = {"generator": [], "discriminator": []}
    g_updates_per_step = 2
    dataset_size = len(dataloader.dataset)

    for epoch in range(num_epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for real_imgs, cond_vec in pbar:
            real_imgs = real_imgs.to(device)
            cond_vec = cond_vec.to(device)
            bsz = real_imgs.size(0)
            height, width = real_imgs.shape[2], real_imgs.shape[3]

            discriminator.zero_grad(set_to_none=True)
            real_validity = discriminator(real_imgs, cond_vec)
            d_real_loss = adv_criterion(real_validity, torch.ones_like(real_validity))

            noise_imgs = torch.randn(bsz, noise_channels, height, width, device=device)
            fake_imgs = generator(noise_imgs, cond_vec).detach()
            fake_validity = discriminator(fake_imgs, cond_vec)
            d_fake_loss = adv_criterion(fake_validity, torch.zeros_like(fake_validity))

            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            d_loss.backward()
            optim_d.step()

            # Two generator updates per discriminator step for better balance
            g_loss_batch = 0.0
            for _ in range(g_updates_per_step):
                generator.zero_grad(set_to_none=True)
                noise_imgs = torch.randn(bsz, noise_channels, height, width, device=device)
                gen_imgs = generator(noise_imgs, cond_vec)
                validity = discriminator(gen_imgs, cond_vec)
                g_loss = adv_criterion(validity, torch.ones_like(validity))
                g_loss.backward()
                optim_g.step()
                g_loss_batch += g_loss.item()

            # Average the two generator losses for logging
            g_loss_batch /= g_updates_per_step

            g_loss_epoch += g_loss_batch * bsz
            d_loss_epoch += d_loss.item() * bsz

            pbar.set_postfix({"D": f"{d_loss.item():.4f}", "G": f"{g_loss_batch:.4f}"})

        g_loss_epoch /= dataset_size
        d_loss_epoch /= dataset_size
        history["generator"].append(g_loss_epoch)
        history["discriminator"].append(d_loss_epoch)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"D_loss: {d_loss_epoch:.4f} "
            f"G_loss: {g_loss_epoch:.4f}"
        )

        if (
            sample_dir is not None
            and fixed_noise is not None
            and fixed_conditions is not None
            and (epoch + 1) % sample_interval == 0
        ):
            save_sample_grid(generator, fixed_noise, fixed_conditions, sample_dir, epoch + 1, nrow)

        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(generator, discriminator, output_dir, epoch + 1)

    return history


def prepare_fixed_conditions(
    dataset: TumorGanDataset,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    if num_samples > len(dataset):
        raise ValueError("Requested more fixed samples than dataset size.")
    step = max(len(dataset) // num_samples, 1)
    indices = list(range(0, len(dataset), step))[:num_samples]
    cond_vec = dataset.get_condition_batch(indices).to(device)
    return cond_vec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional GAN to generate BTXRD patches.")
    parser.add_argument(
        "--image-root",
        type=str,
        default="data/dataset/final_patched_BTXRD",
        help="Directory containing patched/cropped training images.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/dataset/BTXRD/dataset.xlsx",
        help="Spreadsheet or CSV file providing tumor/anatomy/view annotations.",
    )
    parser.add_argument("--image-column", type=str, default="image_id", help="Column holding image filenames.")
    parser.add_argument("--img-size", type=int, default=256, help="Resolution used for GAN training.")
    parser.add_argument("--batch-size", type=int, default=8, help="GAN batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr-g", type=float, default=2e-4, help="Generator learning rate.")
    parser.add_argument("--lr-d", type=float, default=1e-4, help="Discriminator learning rate.")
    parser.add_argument("--noise-channels", type=int, default=1, help="Channels in Gaussian noise input.")
    parser.add_argument("--base-channels", type=int, default=64, help="Base feature width for U-Net.")
    parser.add_argument("--num-workers", type=int, default=2, help="PyTorch DataLoader workers.")
    parser.add_argument("--sample-interval", type=int, default=5, help="Epoch interval for saving sample grids.")
    parser.add_argument("--checkpoint-interval", type=int, default=25, help="Epoch interval for saving checkpoints.")
    parser.add_argument("--sample-grid-rows", type=int, default=4, help="Rows in saved sample grid.")
    parser.add_argument("--sample-grid-cols", type=int, default=4, help="Columns in saved sample grid.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/gan",
        help="Directory used for checkpoints and sample grids.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU execution.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    image_root = Path(args.image_root)
    metadata_path = Path(args.metadata_path)
    base_output_dir = Path(args.output_dir)
    run_output_dir = base_output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = run_output_dir / "samples"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved to {run_output_dir.resolve()}")

    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    samples = build_samples_from_metadata(metadata_path, image_root, image_column=args.image_column)
    print(f"Loaded {len(samples)} samples for GAN training.")
    summarize_samples(samples)

    tumor_type_to_idx = {key: idx for idx, key in enumerate(TUMOR_TYPE_COLUMNS.keys())}
    anatomy_to_idx = {label: idx for idx, (_, label) in enumerate(ANATOMY_PRIORITY)}
    view_to_idx = {view: idx for idx, view in enumerate(VIEW_COLUMNS)}

    dataset = TumorGanDataset(
        samples=samples,
        image_root=image_root,
        img_size=args.img_size,
        tumor_type_to_idx=tumor_type_to_idx,
        anatomy_to_idx=anatomy_to_idx,
        view_to_idx=view_to_idx,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    cond_dim = len(tumor_type_to_idx) + len(anatomy_to_idx) + len(view_to_idx)
    generator = GeneratorUNetCond(
        noise_channels=args.noise_channels,
        img_channels=1,
        cond_dim=cond_dim,
        base_ch=args.base_channels,
    )
    discriminator = PatchDiscriminatorCond(img_channels=1, cond_dim=cond_dim, base_ch=args.base_channels)

    num_grid_items = args.sample_grid_rows * args.sample_grid_cols
    fixed_conditions = prepare_fixed_conditions(dataset, num_grid_items, device)
    fixed_noise = torch.randn(num_grid_items, args.noise_channels, args.img_size, args.img_size, device=device)

    history = train_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        num_epochs=args.epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        noise_channels=args.noise_channels,
        sample_dir=sample_dir,
        sample_interval=args.sample_interval,
        fixed_noise=fixed_noise,
        fixed_conditions=fixed_conditions,
        nrow=args.sample_grid_cols,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=run_output_dir,
    )

    history_path = run_output_dir / "training_history.pt"
    torch.save(history, history_path)
    print(f"Training history saved to {history_path.resolve()}")


if __name__ == "__main__":
    main()
