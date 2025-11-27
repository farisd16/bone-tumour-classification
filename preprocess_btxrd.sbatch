#!/bin/bash
#SBATCH --job-name=preprocess_btxrd
#SBATCH --output=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/preprocess-%A.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/preprocess-%A.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

# Load python module if available.
if command -v ml >/dev/null 2>&1; then
  ml python/anaconda3
fi

# Activate environment (same as other scripts).
if [ -f "/meta/opt/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/meta/opt/anaconda3/etc/profile.d/conda.sh"
fi
conda deactivate || true
conda activate /meta/users/wiep/envs/stylegan

REPO_DIR="/vol/miltank/users/wiep/Documents/bone-tumour-classification"
SCRIPT="${REPO_DIR}/data/style_gan_preprocessing.py"

IMAGE_DIR="${IMAGE_DIR:-${REPO_DIR}/data/dataset/final_patched_BTXRD}"
JSON_DIR="${JSON_DIR:-${REPO_DIR}/data/dataset/BTXRD/Annotations}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/stylegan2-ada-pytorch/BTXRD_resized_sorted}"
TARGET_SIZE="${TARGET_SIZE:-256}"
CENTER_CROP="${CENTER_CROP:-0}" # set to 1 to enable
NO_DATASET_JSON="${NO_DATASET_JSON:-0}" # set to 1 to skip dataset.json

mkdir -p "${OUTPUT_DIR}" "${REPO_DIR}/logs"
cd "${REPO_DIR}"

python "${SCRIPT}" \
  --image-dir "${IMAGE_DIR}" \
  --json-dir "${JSON_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --target-size "${TARGET_SIZE}" \
  $( [ "${CENTER_CROP}" = "1" ] && echo "--center-crop" ) \
  $( [ "${NO_DATASET_JSON}" = "1" ] && echo "--no-dataset-json" )
