#!/bin/bash
#SBATCH --job-name=exp_qwen_tprior2
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=exp/tapos_exp_%j.out

# Load necessary modules
module load gcc

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate EfficientGEBD

# Run the experiment with your new arguments
# Note: durations are passed as a space-separated list


python -u test_baseline.py \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --durations 0.5 1.0 \
    --samples 8 \
    --prompt_file "prompts_tprior.json" \
    --det_threshold 0.5

python -u test_baseline.py \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --durations 0.5 1.0 \
    --samples 4 \
    --prompt_file "prompts_tprior.json" \
    --det_threshold 0.5

python -u test_baseline.py \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --durations 0.5 1.0 \
    --samples 16 \
    --prompt_file "prompts_tprior.json" \
    --det_threshold 0.5
