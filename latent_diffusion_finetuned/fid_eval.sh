#!/bin/bash
#SBATCH --job-name=fid-eval
#SBATCH --output=fid-eval-%A.out
#SBATCH --error=fid-eval-%A.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# 1) Load Python / Anaconda module
ml python/anaconda3

# 2) Activate your conda environment
eval "$(conda shell.bash hook)"
conda deactivate
conda activate bone-tumour-classification

# 3) Define folders
# REAL_DIR: your real train images (already exist)
REAL_DIR="/vol/miltank/projects/practical_wise2526/bone-tumor-classification-gen-models/dataset/hf_entire_final_patched_BTXRD/train"

# FAKE_DIR: folder where generate_for_fid.py saved fake images
FAKE_DIR="/vol/miltank/projects/practical_wise2526/bone-tumor-classification-gen-models/btxrd_fid/fake_ldm_train"

# 4) Run FID computation
python -m pytorch_fid "$REAL_DIR" "$FAKE_DIR" --device cuda:0