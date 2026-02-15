#!/bin/bash
#SBATCH --job-name=prep-real-fid
#SBATCH --output=prep-real-fid-%j.out
#SBATCH --error=prep-real-fid-%j.err
#SBATCH --partition=asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

ENV_NAME=bone-tumour-classification
PROJECT_DIR=/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned

# ---- FOLDERS TO PASS ----
REAL_SRC=/vol/miltank/projects/practical_wise2526/bone-tumor-classification-gen-models/dataset/final_patched_BTXRD
SPLIT_JSON=/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/data_split.json
REAL_DST=/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/real_samples/real_512
IMAGE_SIZE=512
# --------------------------

cd "$PROJECT_DIR"

ml python/anaconda3
eval "$(conda shell.bash hook)"
conda deactivate
conda activate "$ENV_NAME"

python prepare_real_samples.py \
    --real_src "$REAL_SRC" \
    --split_json "$SPLIT_JSON" \
    --real_dst "$REAL_DST" \
    --size "$IMAGE_SIZE"
