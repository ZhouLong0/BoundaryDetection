#!/bin/bash
#SBATCH --job-name=bsurprise_tas_breakfast
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=exp/breakfast/hcond_f_tas_%j.out

set -euo pipefail

# Usage:
#   sbatch 01_bsurprise_breakfast_v1.sh [EXP_ID]

# "test.split2.bundle,test.split3.bundle,test.split4.bundle"
EXP_ID="${1:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p exp

echo "===== Running Breakfast BSurprise job ====="
echo "Experiment ID: $EXP_ID"

module load gcc
module load cuda

source ~/miniconda3/etc/profile.d/conda.sh
conda activate EfficientGEBD

python -u 01_bsurprise_egoper.py \
    --dataset breakfast \
    --split_files "test.split1.bundle" \
    --exp_id "$EXP_ID"  \
    --verbose \
    --full_steps

echo "✅ Breakfast run completed."

