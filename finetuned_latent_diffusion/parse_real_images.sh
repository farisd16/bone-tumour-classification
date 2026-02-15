#!/bin/bash
#SBATCH --job-name=organize-real-images
#SBATCH --output=./organize-real-images-%A.out
#SBATCH --error=./organize-real-images-%A.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

# -------- ENV SETUP --------
ENV_NAME=bone-tumour-classification

ml python/anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# -------- REPO --------
REPODIR="/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned"
cd "$REPODIR"

# -------- PATHS --------
export IMAGE_DIR="./real_samples/real_512"
export OUTPUT_DIR="./real_samples/organized_real_512"
export EXCEL_PATH="/vol/miltank/projects/practical_wise2526/bone-tumor-classification-gen-models/dataset/BTXRD/dataset.xlsx"

# -------- RUN --------
python parse_real_images.py

echo "Real image organization complete."
