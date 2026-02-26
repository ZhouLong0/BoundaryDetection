#!/bin/bash
#SBATCH --job-name=exp_baseline_wprob
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=baseline_exp_wprob_%j.out

module load gcc

source ~/miniconda3/etc/profile.d/conda.sh
conda activate EfficientGEBD

python -u test_best_prompt_baseline_withprob.py