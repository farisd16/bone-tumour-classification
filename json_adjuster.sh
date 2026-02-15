#!/bin/bash
#SBATCH --job-name=json-adjuster
#SBATCH --output=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/json_adjuster-%A.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/json_adjuster-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -eo pipefail

# Load Python module
source /meta/opt/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

# Activate conda env (override via: sbatch --export=ALL,CONDA_ENV=...)
# Avoid nounset because conda activate/deactivate scripts use unset vars.
CONDA_ENV="${CONDA_ENV:-ADLM}"
conda activate "${CONDA_ENV}"

REPO_DIR="/vol/miltank/users/wiep/Documents/bone-tumour-classification"
cd "${REPO_DIR}"

mkdir -p "${REPO_DIR}/logs"

# Required inputs (override via: sbatch --export=ALL,INPUT_SPLIT=...,SYNTHETIC_IMAGES=...)
INPUT_SPLIT="${INPUT_SPLIT:-data/dataset/splits/dataset_split_final.json}"
SYNTHETIC_IMAGES="${SYNTHETIC_IMAGES:-generated_images_15800}"

# Optional inputs
OUTPUT_SPLIT="${OUTPUT_SPLIT:-data/dataset/splits}"
INPUT_IMAGES="${INPUT_IMAGES:-data/dataset/final_patched_BTXRD}"
OUTPUT_IMAGES="${OUTPUT_IMAGES:-data/dataset/final_patched_BTXRD}"
INPUT_ANNOTATIONS="${INPUT_ANNOTATIONS:-data/dataset/BTXRD/Annotations}"
OUTPUT_ANNOTATIONS="${OUTPUT_ANNOTATIONS:-data/dataset/BTXRD/Annotations}"
SEED="${SEED:-42}"
START_IDX="${START_IDX:-1707}"
START_IMG="${START_IMG:-1868}"
EXT="${EXT:-jpeg}"

if [[ -z "${INPUT_SPLIT}" || -z "${SYNTHETIC_IMAGES}" ]]; then
  echo "Missing required vars. Provide INPUT_SPLIT and SYNTHETIC_IMAGES."
  echo "Example: sbatch --export=ALL,INPUT_SPLIT=/path/split.json,SYNTHETIC_IMAGES=/path/generated json_adjuster.sbatch"
  exit 1
fi

python json_adjuster.py \
  --input_split "${INPUT_SPLIT}" \
  --output_split "${OUTPUT_SPLIT}" \
  --input_images "${INPUT_IMAGES}" \
  --synthetic_images "${SYNTHETIC_IMAGES}" \
  --output_images "${OUTPUT_IMAGES}" \
  --input_annotations "${INPUT_ANNOTATIONS}" \
  --output_annotations "${OUTPUT_ANNOTATIONS}" \
  --seed "${SEED}" \
  --start-idx "${START_IDX}" \
  --start-img "${START_IMG}" \
  --ext "${EXT}"

# Optional cleanup
conda deactivate || true
