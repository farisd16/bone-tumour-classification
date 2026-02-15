#!/bin/bash
#SBATCH --job-name=lpips-eval
#SBATCH --output=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/lpips_eval-%A.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/lpips_eval-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail

# Load Python module
source /meta/opt/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

# Activate conda env (override via: sbatch --export=ALL,CONDA_ENV=...)
# Avoid nounset because conda activate/deactivate scripts use unset vars.
CONDA_ENV="${CONDA_ENV:-stylegan5}"
conda activate "${CONDA_ENV}"

REPO_DIR="/vol/miltank/users/wiep/Documents/bone-tumour-classification"
cd "${REPO_DIR}"

# Ensure required Python packages are available
pip install -q lpips pillow torchvision

# Required inputs (override via: sbatch --export=ALL,REAL_ROOT=...,GEN_ROOT=...)
REAL_ROOT="${REAL_ROOT:-/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/data/dataset/BTXRD_anatomical_resized_sorted}"
GEN_ROOT="${GEN_ROOT:-/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/generated_images}"
CLASSES="${CLASSES:-}"

# Optional inputs
PAIRS="${PAIRS:-10000}"
IMG_SIZE="${IMG_SIZE:-256}"
BACKBONE="${BACKBONE:-alex}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

if [[ -z "${REAL_ROOT}" || -z "${GEN_ROOT}" ]]; then
  echo "Missing required vars. Provide REAL_ROOT and GEN_ROOT."
  echo "Example: sbatch --export=ALL,REAL_ROOT=/path/real,GEN_ROOT=/path/generated lpips_eval.sbatch"
  exit 1
fi

mkdir -p "${REPO_DIR}/logs"

python lpips_eval.py \
  --real_root "${REAL_ROOT}" \
  --gen_root "${GEN_ROOT}" \
  ${CLASSES:+--classes ${CLASSES}} \
  --pairs "${PAIRS}" \
  --img_size "${IMG_SIZE}" \
  --backbone "${BACKBONE}" \
  --device "${DEVICE}" \
  --seed "${SEED}"

# Optional cleanup
conda deactivate || true
