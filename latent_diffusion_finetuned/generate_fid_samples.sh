#!/bin/bash
#SBATCH --job-name=fid-samples
#SBATCH --output=fid-samples-%j.out
#SBATCH --error=fid-samples-%j.err
#SBATCH --partition=asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

PROJECT_DIR=/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned
ENV_NAME=bone-tumour-classification

cd "$PROJECT_DIR"

ml python/anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python - << 'EOF'
import torch
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
EOF

python generate_fid_samples.py
