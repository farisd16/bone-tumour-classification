#!/bin/bash
#SBATCH --job-name=cuda-env
#SBATCH --output=cuda-env-%j.out
#SBATCH --error=cuda-env-%j.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

ENV_NAME=bone-tumour-classification

ml purge
ml python/anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "===== nvidia-smi ====="
nvidia-smi | head -n 5

echo
echo "===== torch.utils.collect_env ====="
python -m torch.utils.collect_env | head -n 40
