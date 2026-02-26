#!/bin/bash

# --- Resource Request ---
#SBATCH --job-name=crosseffgedb_kintapos        # Job name
#SBATCH --nodes=1                         # Request one node
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=crosseffgedb_kintapos_%j.out
# ******************testing BasicGEBD-ResNet50-L3**************************

# ******************testing EfficientGEBD-CSN(R50)-L2L3L4******************
torchrun --nproc_per_node 2 \
--master_port 1111 train.py \
--expname test \
--test-only \
--resume output/Kinetics-GEBD/x2x3x4_csn_r50_eff/model_best.pth \
MODEL.BACKBONE.NAME 'csn' \
MODEL.CAT_PREV True \
MODEL.FPN_START_IDX 1 \
MODEL.HEAD_CHOICE [3] \
MODEL.IS_BASIC False
#**************************************************************************