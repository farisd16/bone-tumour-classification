#!/bin/bash
#SBATCH --job-name=sweep-r64-b8
#SBATCH --output=./logs/resnet34-%A.out
#SBATCH --error=./logs/resnet34-%A.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load python module
ml python/anaconda3

# Activate corresponding environment
eval "$(conda shell.bash hook)"
conda deactivate
conda activate /meta/users/cahu/my_env ########################
# Number of sweep runs to execute (optional)
COUNT=${1:-}

# Create the sweep and capture the sweep ID
SWEEP_ID=$(wandb sweep sweeps/resnet34.yaml 2>&1 | grep -oP 'wandb agent \K[^ ]+')

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Failed to create sweep or extract sweep ID"
    exit 1
fi

echo "Created sweep with ID: $SWEEP_ID"

# Run the sweep agent
if [ -n "$COUNT" ]; then
    echo "Running $COUNT agents..."
    wandb agent --count $COUNT $SWEEP_ID
else
    echo "Running agents until sweep completes..."
    wandb agent $SWEEP_ID
fi