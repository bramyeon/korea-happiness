#!/bin/bash
#SBATCH --job-name=bryan
#SBATCH --ntasks=1
#SBATCH --mem=125000
#SBATCH --cpus-per-task=18
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o ./train-korea-clean.log.%j
#SBATCH -e ./train-korea-clean.log.%j

source activate /u/brywi/anaconda3/envs/gamma
python train.py