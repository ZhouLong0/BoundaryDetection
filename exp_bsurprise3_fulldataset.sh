#!/bin/bash
#SBATCH --job-name=exp_qwen_old
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=exp/bsurprise/tapos_exp3_%j.out

# Load necessary modules
module load gcc

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate EfficientGEBD

SAMPLES=8

python -u test_baseline.py \
    --model_name "Qwen/Qwen3-VL-8B-Instruct" \
    --durations 0.5 \
    --samples ${SAMPLES} \
    --method bsurprise \
    --split_file split.txt \
    --det_threshold 0.5


