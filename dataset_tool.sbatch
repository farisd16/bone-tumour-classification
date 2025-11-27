#!/bin/bash
#SBATCH --job-name=dataset_tool
#SBATCH --output=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/dataset_tool-%A.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/dataset_tool-%A.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

# Load python module if available.
if command -v ml >/dev/null 2>&1; then
  ml python/anaconda3
fi

# Activate environment (matches gan_training.sbatch).
if [ -f "/meta/opt/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/meta/opt/anaconda3/etc/profile.d/conda.sh"
fi
conda deactivate || true
conda activate /meta/users/wiep/envs/stylegan

REPO_DIR="/vol/miltank/users/wiep/Documents/bone-tumour-classification"
cd "${REPO_DIR}/stylegan2-ada-pytorch"

SOURCE_DIR="${SOURCE_DIR:-BTXRD_resized_sorted}"
DEST_PATH="${DEST_PATH:-${REPO_DIR}/data/btxrd_dataset.zip}"
MAX_IMAGES="${MAX_IMAGES:-}"
TRANSFORM="${TRANSFORM:-}"
RESIZE_FILTER="${RESIZE_FILTER:-}"
WIDTH="${WIDTH:-}"
HEIGHT="${HEIGHT:-}"

mkdir -p "$(dirname "${DEST_PATH}")"

python dataset_tool.py \
  --source "${SOURCE_DIR}" \
  --dest "${DEST_PATH}" \
  ${MAX_IMAGES:+--max-images "${MAX_IMAGES}"} \
  ${TRANSFORM:+--transform "${TRANSFORM}"} \
  ${RESIZE_FILTER:+--resize-filter "${RESIZE_FILTER}"} \
  ${WIDTH:+--width "${WIDTH}"} \
  ${HEIGHT:+--height "${HEIGHT}"}
