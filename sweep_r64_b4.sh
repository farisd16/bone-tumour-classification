#!/bin/bash
#SBATCH --job-name=sweep-r64-b4
#SBATCH --output=./sweep-r64-b4-%A.out
#SBATCH --error=./sweep-r64-b4-%A.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load python module
ml python/anaconda3

# Activate corresponding environment
eval "$(conda shell.bash hook)"
conda deactivate
conda activate bone-tumour-classification

# Number of sweep runs to execute (default: 10)
COUNT=${1:-10}

# Create the sweep and capture the sweep ID
SWEEP_ID=$(wandb sweep sweeps/resnet_diffusion_r64_b4.yaml 2>&1 | grep -oP 'wandb agent \K[^ ]+')

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Failed to create sweep or extract sweep ID"
    exit 1
fi

echo "Created sweep with ID: $SWEEP_ID"
echo "Running $COUNT agents..."

# Run the sweep agent
srun wandb agent --count $COUNT $SWEEP_ID
