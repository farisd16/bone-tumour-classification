#!/bin/bash
#SBATCH --job-name=gen-fid
#SBATCH --output=gen-fid-%A.out
#SBATCH --error=gen-fid-%A.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

ml python/anaconda3
eval "$(conda shell.bash hook)"
conda deactivate
conda activate bone-tumour-classification

python generate_for_fid.py
