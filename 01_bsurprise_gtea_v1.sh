#!/bin/bash
#SBATCH --job-name=bsurprise_tas_gtea
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=exp/gtea_tas_%j.out

set -euo pipefail

# Usage:
#   sbatch 01_bsurprise_gtea_v1.sh [EXP_ID]
EXP_ID="${1:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p exp

echo "===== Running GTEA BSurprise job ====="
echo "Experiment ID: $EXP_ID"

module load gcc
module load cuda

source ~/miniconda3/etc/profile.d/conda.sh
conda activate EfficientGEBD

python -u 01_bsurprise_egoper.py \
    --dataset gtea \
    --split_files "test.split1.bundle,test.split2.bundle,test.split3.bundle,test.split4.bundle" \
    --exp_id "$EXP_ID"
echo "✅ GTEA run completed."
