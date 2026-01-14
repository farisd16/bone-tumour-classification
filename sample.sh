#!/bin/bash
#SBATCH --job-name=sample_ldm
#SBATCH --output=./sample_ldm-%A.out
#SBATCH --error=./sample_ldm-%A.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Load python module
ml python/anaconda3
# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that.
# Not necessary if you always run this script from a clean terminal
eval "$(conda shell.bash hook)"
conda deactivate
# If the following does not work, try 'source activate <env-name>'
conda activate bone-tumour-classification
# Run the program
srun python -m latent_diffusion.sample --vae-run-name train_vae_2025-12-11_17-45-28 --ldm-run-name train_ldm_2025-12-11_20-48-31 --class-name osteochondroma