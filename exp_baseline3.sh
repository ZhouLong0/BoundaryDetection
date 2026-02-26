#!/bin/bash
#SBATCH --job-name=exp_qwen_old
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=exp/tapos_exp3_%j.out

# Load necessary modules
module load gcc

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate EfficientGEBD

SAMPLES=8

python -u test_baseline.py \
    --model_name "Qwen/Qwen3-VL-8B-Instruct" \
    --durations 0.5 1.0 \
    --samples ${SAMPLES} \
    --prompt_file "prompts_prob.json" \
    --det_threshold 0.5

python -u test_baseline.py \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --durations 0.5 1.0 \
    --samples ${SAMPLES} \
    --prompt_file "prompts_prob.json" \
    --det_threshold 0.5
