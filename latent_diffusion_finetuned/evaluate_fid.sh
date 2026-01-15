#!/bin/bash
#SBATCH --job-name=fid-pytorch
#SBATCH --output=fid-pytorch-%j.out
#SBATCH --error=fid-pytorch-%j.err
#SBATCH --partition=asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

REAL_DIR=/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/real_samples/real_512
FAKE_DIR=/vol/miltank/users/carre/bone-tumour-classification/latent_diffusion_finetuned/fid_evaluation_samples/fid_1000_samples_0_7
ENV_NAME=bone-tumour-classification

cd /vol/miltank/users/carre/bone-tumour-classification

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