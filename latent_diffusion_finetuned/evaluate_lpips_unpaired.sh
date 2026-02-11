#!/bin/bash
#SBATCH --job-name=lpips-unpaired
#SBATCH --output=lpips-unpaired-%j.out
#SBATCH --error=lpips-unpaired-%j.err
#SBATCH --partition=asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

ENV_NAME=bone-tumour-classification
PROJECT_DIR=/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned

cd "$PROJECT_DIR"

ml python/anaconda3
eval "$(conda shell.bash hook)"
conda deactivate
conda activate "$ENV_NAME"

echo "CUDA check:"
python - << 'EOF'
import torch
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
EOF

srun python compute_lpips_unpaired.py
