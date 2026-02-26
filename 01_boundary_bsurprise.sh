#!/bin/bash
#SBATCH --job-name=bsurprise_boundary_tapos
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=exp/boundary/%j.out

set -euo pipefail

EXP_ID="${1:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p exp/boundary/tapos

echo "===== Running TAPOS Boundary BSurprise job ====="
echo "Experiment ID: $EXP_ID"

module load gcc
module load cuda

source ~/miniconda3/etc/profile.d/conda.sh
conda activate EfficientGEBD

python -u 01_boundary.py \
    --dataset tapos \
    --method bsurprise \
    --exp_id "$EXP_ID" \
    --verbose \

echo "✅ TAPOS boundary run completed."

