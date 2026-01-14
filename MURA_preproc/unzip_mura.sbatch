#!/bin/bash

#SBATCH --job-name=unzip_mura
#SBATCH --output=unzip_mura-%A.out
#SBATCH --error=unzip_mura-%A.err
#SBATCH --qos=master
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

echo "Starting safe unzip of MURA dataset..."

ZIP_PATH="/vol/miltank/projects/practical_wise2526/bone-tumor-classification/dataset/MURA.zip"
DEST_PATH="/vol/miltank/projects/practical_wise2526/bone-tumor-classification/dataset/"

unzip -q $ZIP_PATH -d $DEST_PATH

echo "Unzip completed."
