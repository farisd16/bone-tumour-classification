#!/bin/bash
#SBATCH --job-name=train_vae
#SBATCH --output=./train_vae-%A.out
#SBATCH --error=./train_vae-%A.err
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Load python module
ml python/anaconda3
# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that.
# Not necessary if you always run this script from a clean terminal
conda deactivate
# If the following does not work, try 'source activate <env-name>'
conda activate bone-tumour-classification
# Run the program
python -m latent_diffusion.vae.train