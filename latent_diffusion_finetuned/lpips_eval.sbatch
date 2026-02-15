#!/bin/bash
#SBATCH --job-name=lpips-eval
#SBATCH --output=/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/lpips_eval-%A.out
#SBATCH --error=/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/lpips_eval-%A.err
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
ENV_NAME=bone-tumour-classification
ml python/anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

REPO_DIR="/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned"
cd "${REPO_DIR}"

# Ensure required Python packages are available
pip install -q lpips pillow torchvision

# Required inputs (override via: sbatch --export=ALL,REAL_ROOT=...,GEN_ROOT=...)
REAL_ROOT="${REAL_ROOT:-/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/real_samples/organized_real_512}"
GEN_ROOT="${GEN_ROOT:-/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/organized_evaluation_samples/rep_rep_stable_diffusion_rank_64_batch_8_v1_5_lora_1_ckp_20000}"
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
