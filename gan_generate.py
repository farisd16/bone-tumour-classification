import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torchvision.utils import save_image

from gan_training import (
    ANATOMY_PRIORITY,
    VIEW_COLUMNS,
    TUMOR_TYPE_COLUMNS,
    GeneratorUNetCond,
    TumorGanDataset,
    build_samples_from_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic BTXRD patches using a trained GAN.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained generator checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dataset/gan_synthetic",
        help="Directory where generated PNGs will be stored.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to synthesize.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size when running the generator.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="data/dataset/final_patched_BTXRD",
        help="Directory containing real patched images (used for metadata alignment).",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/dataset/BTXRD/dataset.xlsx",
        help="Metadata spreadsheet/CSV used to define conditioning labels.",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="image_id",
        help="Column name of the image filename inside the metadata file.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Image resolution expected by the generator.",
    )
    parser.add_argument(
        "--noise-channels",
        type=int,
        default=1,
        help="Noise channels used during training.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=64,
        help="Base channel width of the generator backbone.",
    )
    parser.add_argument(
        "--tumor-type",
        type=str,
        choices=list(TUMOR_TYPE_COLUMNS.keys()),
        help="Force generation for a specific tumor type.",
    )
    parser.add_argument(
        "--anatomy",
        type=str,
        choices=[label for _, label in ANATOMY_PRIORITY],
        help="Force generation for a specific anatomy cluster.",
    )
    parser.add_argument(
        "--view",
        type=str,
        choices=VIEW_COLUMNS,
        help="Force generation for a specific X-ray projection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible condition sampling.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _filtered_indices(
    samples: Sequence[Dict[str, str]],
    tumor_type: Optional[str],
    anatomy: Optional[str],
    view: Optional[str],
) -> List[int]:
    indices: List[int] = []
    for idx, sample in enumerate(samples):
        if tumor_type and sample["tumor_type"] != tumor_type:
            continue
        if anatomy and sample["anatomy"] != anatomy:
            continue
        if view and sample["view"] != view:
            continue
        indices.append(idx)
    if not indices:
        raise RuntimeError(
            "No samples match the requested condition filters. "
            "Loosen or remove --tumor-type/--anatomy/--view."
        )
    return indices


def _sample_condition_vectors(
    dataset: TumorGanDataset,
    indices: Sequence[int],
    num_images: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    chosen_indices: List[int] = []
    for _ in range(num_images):
        chosen_indices.append(random.choice(indices))
    cond = dataset.get_condition_batch(chosen_indices).to(device)
    return cond, chosen_indices


def _format_sample_name(sample_meta: Dict[str, str], counter: int) -> str:
    tumor = sample_meta["tumor_type"]
    anatomy = sample_meta["anatomy"]
    view = sample_meta["view"]
    return f"{counter:05d}_{tumor}_{anatomy}_{view}.png"


def generate_images(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image_root = Path(args.image_root)
    metadata_path = Path(args.metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, skipped = build_samples_from_metadata(metadata_path, image_root, image_column=args.image_column)
    if skipped:
        print("Metadata filtering skipped entries:", dict(skipped))

    tumor_type_to_idx = {key: idx for idx, key in enumerate(TUMOR_TYPE_COLUMNS.keys())}
    anatomy_to_idx = {label: idx for idx, (_, label) in enumerate(ANATOMY_PRIORITY)}
    view_to_idx = {view_name: idx for idx, view_name in enumerate(VIEW_COLUMNS)}

    dataset = TumorGanDataset(
        samples=samples,
        image_root=image_root,
        img_size=args.img_size,
        tumor_type_to_idx=tumor_type_to_idx,
        anatomy_to_idx=anatomy_to_idx,
        view_to_idx=view_to_idx,
    )

    available_indices = _filtered_indices(samples, args.tumor_type, args.anatomy, args.view)

    set_seed(args.seed)
    cond_vectors, chosen_indices = _sample_condition_vectors(
        dataset=dataset,
        indices=available_indices,
        num_images=args.num_images,
        device=device,
    )

    generator = GeneratorUNetCond(
        noise_channels=args.noise_channels,
        img_channels=1,
        cond_dim=cond_vectors.size(1),
        base_ch=args.base_channels,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state)
    generator.eval()

    batch_size = args.batch_size
    total = args.num_images
    counter = 0
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            current_batch = end - start
            noise = torch.randn(
                current_batch,
                args.noise_channels,
                args.img_size,
                args.img_size,
                device=device,
            )
            cond_batch = cond_vectors[start:end]
            generated = generator(noise, cond_batch)
            for idx in range(current_batch):
                sample_idx = chosen_indices[start + idx]
                sample_meta = samples[sample_idx]
                filename = _format_sample_name(sample_meta, counter)
                save_path = output_dir / sample_meta["tumor_type"] / sample_meta["anatomy"] / sample_meta["view"]
                save_path.mkdir(parents=True, exist_ok=True)
                save_image(
                    generated[idx],
                    save_path / filename,
                    normalize=True,
                    value_range=(-1, 1),
                )
                counter += 1

    print(f"Saved {counter} synthetic images to {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    generate_images(args)


if __name__ == "__main__":
    main()
