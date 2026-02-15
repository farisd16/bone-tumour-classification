#!/bin/bash
#SBATCH --job-name=resnet34-sweep-tests
#SBATCH --output=./logs/sweep-tests-%A.out
#SBATCH --error=./logs/sweep-tests-%A.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load python module
ml python/anaconda3

# Activate corresponding environment (same as sweep file)
eval "$(conda shell.bash hook)"
conda deactivate
conda activate /meta/users/cahu/my_env


LIMIT=${1:-}
EXTRA_ARGS=("${@:2}")

ARGS=(--pattern "resnet_short_notice")

if [ -n "$LIMIT" ]; then
  ARGS+=(--limit "$LIMIT")
fi

# Pass through any extra args (e.g. --dry-run)
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  ARGS+=("${EXTRA_ARGS[@]}")
fi

python run_sweep_tests.py "${ARGS[@]}"