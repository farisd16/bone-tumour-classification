#!/bin/bash
#SBATCH --job-name=sweep-r64-b8
#SBATCH --output=./sweep-r64-b8-%A.out
#SBATCH --error=./sweep-r64-b8-%A.err
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

# Number of sweep runs to execute (optional)
COUNT=${1:-}

# Create the sweep and capture the sweep ID
SWEEP_ID=$(wandb sweep sweeps/resnet_diffusion_r64_b8.yaml 2>&1 | grep -oP 'wandb agent \K[^ ]+')

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Failed to create sweep or extract sweep ID"
    exit 1
fi

echo "Created sweep with ID: $SWEEP_ID"

# Run the sweep agent
if [ -n "$COUNT" ]; then
    echo "Running $COUNT agents..."
    srun wandb agent --count $COUNT $SWEEP_ID
else
    echo "Running agents until sweep completes..."
    srun wandb agent $SWEEP_ID
fi
