#!/bin/bash
#SBATCH --job-name=stylegan_cond
#SBATCH --output=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/stylegan_cond-%A.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/stylegan_cond-%A.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=0-48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

set -euo pipefail

# Load python module if available.
if command -v ml >/dev/null 2>&1; then
  ml python/anaconda3
fi

# Activate StyleGAN2-ADA environment.
if [ -f "/meta/opt/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/meta/opt/anaconda3/etc/profile.d/conda.sh"
fi
conda deactivate || true
conda activate /meta/users/wiep/envs/stylegan

# Ensure libtinfo.so.5 is available (Torch sometimes needs it on this cluster).
TINFO6="/usr/lib/x86_64-linux-gnu/libtinfo.so.6"
TINFO5="${CONDA_PREFIX}/lib/libtinfo.so.5"
if [ -f "${TINFO6}" ] && [ ! -e "${TINFO5}" ]; then
  ln -s "${TINFO6}" "${TINFO5}"
fi
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

REPO_DIR="/vol/miltank/users/wiep/Documents/bone-tumour-classification"
cd "${REPO_DIR}/stylegan2-ada-pytorch"

# Override via env vars when submitting (e.g. `sbatch --export=ALL,BATCH=8 ...`).
DATASET_ZIP="${DATASET_ZIP:-${REPO_DIR}/data/btxrd_dataset.zip}"
OUTDIR="${OUTDIR:-${REPO_DIR}/checkpoints/stylegan2ada_cond}"
GPUS="${GPUS:-1}"
BATCH="${BATCH:-16}"
CFG="${CFG:-auto}"
SNAP="${SNAP:-10}"
MIRROR="${MIRROR:-1}"
AUG="${AUG:-ada}"
METRICS="${METRICS:-none}"
KIMG="${KIMG:-}"
RESUME="${RESUME:-}"
SEED="${SEED:-42}"

mkdir -p "${OUTDIR}" "${REPO_DIR}/logs"

python train.py \
  --outdir="${OUTDIR}" \
  --data="${DATASET_ZIP}" \
  --gpus="${GPUS}" \
  --batch="${BATCH}" \
  --cfg="${CFG}" \
  --snap="${SNAP}" \
  --mirror="${MIRROR}" \
  --cond=1 \
  --seed="${SEED}" \
  --aug="${AUG}" \
  --metrics="${METRICS}" \
  ${KIMG:+--kimg="${KIMG}"} \
  ${RESUME:+--resume="${RESUME}"}
