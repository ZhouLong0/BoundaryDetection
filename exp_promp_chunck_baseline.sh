#!/bin/bash
#SBATCH --job-name=openxlab_download
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=baseline_exp_%j.out

module load gcc

source ~/miniconda3/etc/profile.d/conda.sh
conda activate EfficientGEBD

python -u test_best_prompt_baseline.py