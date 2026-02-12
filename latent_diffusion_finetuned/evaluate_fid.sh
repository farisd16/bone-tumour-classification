#!/bin/bash
#SBATCH --job-name=fid-pytorch
#SBATCH --output=fid-pytorch-bootstrap-%j.out
#SBATCH --error=fid-pytorch-bootstrap-%j.err
#SBATCH --partition=asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

REAL_DIR=/vol/miltank/users/wiep/Documents/tmp/fid_real_flat
FAKE_DIR=/vol/miltank/users/wiep/Documents/tmp/fid_fake_flat_020000
ENV_NAME=ADLM

FAKE_SUFFIX="${FAKE_DIR##*_}"
OUT_FILE="fid-pytorch-${FAKE_SUFFIX}.out"
ERR_FILE="fid-pytorch-${FAKE_SUFFIX}.err"
exec > >(tee -a "$OUT_FILE") 2> >(tee -a "$ERR_FILE" >&2)

cd /vol/miltank/users/wiep/Documents

ml purge
ml python/anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi


python -m pytorch_fid \
  "$REAL_DIR" \
  "$FAKE_DIR" \
  --device cuda:0 \
  --batch-size 32 \
  --num-workers 0
