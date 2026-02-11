#!/bin/bash
#SBATCH --job-name=lpips-by-subtype
#SBATCH --output=lpips-by-subtype-%j.out
#SBATCH --error=lpips-by-subtype-%j.err
#SBATCH --partition=asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=04:00:00  # Longer for subtype parsing/more pairs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G  # More RAM for pandas/XLSX

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

# Ensure compute_lpips_by_subtype.py is saved
echo "Starting LPIPS by subtype..."

srun python compute_lpips_by_subtype.py

echo "Done. Check lpips_by_subtype.csv for results."
