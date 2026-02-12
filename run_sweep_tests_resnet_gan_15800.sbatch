#!/bin/bash
#SBATCH --job-name=resnet_gan_15800_tests
#SBATCH --output=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/resnet_gan_15800_tests-%A.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/resnet_gan_15800_tests-%A.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load python module
ml python/anaconda3

# Activate corresponding environment
eval "$(conda shell.bash hook)"
conda deactivate
conda activate ADLM

# Optional limit and dry-run via env vars
LIMIT=${LIMIT:-}
DRY_RUN=${DRY_RUN:-}

ARGS=("--pattern" "resnet_gan_15800")

if [ -n "$LIMIT" ]; then
    ARGS+=("--limit" "$LIMIT")
fi

if [ -n "$DRY_RUN" ]; then
    ARGS+=("--dry-run")
fi

python run_sweep_tests.py "${ARGS[@]}"
