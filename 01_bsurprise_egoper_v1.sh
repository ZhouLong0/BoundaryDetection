#!/bin/bash
#SBATCH --job-name=egoper_launcher
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --output=exp/launcher_%j.out

# This script submits one SLURM job per recipe.

set -euo pipefail

# Usage:
#   sbatch egoper_launcher.sh [EXP_ID]
EXP_ID="${1:-$(date +%Y%m%d_%H%M%S)}"

RECIPES=("coffee" "tea" "oatmeal" "pinwheels" "quesadilla")

# Ensure output directory exists
mkdir -p exp
mkdir -p exp/egoper

echo "===== Launching parallel jobs for all recipes ====="
echo "Experiment ID: $EXP_ID"
echo "Recipes: ${RECIPES[*]}"
echo

# Loop through all recipes
for recipe in "${RECIPES[@]}"; do

    JOB_NAME="bsp_as_egoper_${recipe}_${EXP_ID}"
    OUT_FILE="exp/egoper/hcond_tas_${recipe}_${EXP_ID}_%j.out"

    echo "Submitting job: $JOB_NAME"

    # The heredoc dynamically generates and submits the SLURM script
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --output=$OUT_FILE
#SBATCH --mem=64G

# Load necessary modules
module load gcc
module load cuda

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate EfficientGEBD

python -u 01_bsurprise_egoper.py \
    --recipe "$recipe" \
    --exp_id "$EXP_ID" \
    --verbose \
    --history_conditioning \
EOT

done

echo "✅ All jobs submitted to SLURM."