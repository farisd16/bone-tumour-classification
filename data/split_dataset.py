import argparse
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = PROJECT_ROOT / "data" / "BTXRD" / "dataset_singlelabel.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "data" / "BTXRD" / "splits"


def split_group(group: pd.DataFrame, ratios, seed: int):
    """Return train/val/test splits for a single label group."""
    shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    desired = [ratio * n for ratio in ratios]
    counts = [math.floor(x) for x in desired]
    remainder = n - sum(counts)

    # Distribute leftover samples to splits with largest fractional parts
    fractional = [d - c for d, c in zip(desired, counts)]
    order = sorted(range(3), key=lambda i: fractional[i], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1

    train_end = counts[0]
    val_end = counts[0] + counts[1]

    train_split = shuffled.iloc[:train_end]
    val_split = shuffled.iloc[counts[0]:val_end]
    test_split = shuffled.iloc[val_end:]
    return train_split, val_split, test_split


def stratified_split(df: pd.DataFrame, ratios, seed: int):
    """Split the dataframe per label using the provided ratios (train, val, test)."""
    train_parts = []
    val_parts = []
    test_parts = []

    for label, group in df.groupby("label", sort=False):
        train_split, val_split, test_split = split_group(group, ratios, seed)
        train_parts.append(train_split)
        val_parts.append(val_split)
        test_parts.append(test_split)

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Split dataset_singlelabel.csv into train/val/test splits.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="Path to the source CSV containing image_id and label.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="Output directory for train/val/test CSV files.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of data for training set.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of data for validation set.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Fraction of data for test set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if not math.isclose(sum(ratios), 1.0, rel_tol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {ratios}")

    if not args.src.exists():
        raise FileNotFoundError(f"Source CSV not found: {args.src}")

    df = pd.read_csv(args.src)
    if "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Source CSV must contain 'image_id' and 'label' columns.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = stratified_split(df, ratios, args.seed)
    train_path = args.outdir / "train.csv"
    val_path = args.outdir / "val.csv"
    test_path = args.outdir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved {len(train_df)} rows to {train_path}")
    print(f"Saved {len(val_df)} rows to {val_path}")
    print(f"Saved {len(test_df)} rows to {test_path}")


if __name__ == "__main__":
    main()
