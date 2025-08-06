#!/bin/bash
#SBATCH --job-name=rubiks_supervised
#SBATCH --output=rubiks_supervised_%j.out
#SBATCH --error=rubiks_supervised_%j.err
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

# This script pretrains an MLP with the exact same architecture as run_reinforce.slurm,
# so weights can be loaded 1:1. It mirrors HIDDEN_LAYERS/ACTIVATION/NORM/DROPOUT and
# uses AdamW with the same LR/WEIGHT_DECAY defaults.

# Mirror REINFORCE network/hparams (defaults copied from run_reinforce.slurm)
export HIDDEN_LAYERS=${HIDDEN_LAYERS:-"1024,1024,1024,1024"}
export ACTIVATION=${ACTIVATION:-"silu"}
export NORM=${NORM:-"layernorm"}
export DROPOUT=${DROPOUT:-0.05}

export LR=${LR:-3e-4}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}

# Supervised dataset generation controls
export SUP_NUM_SAMPLES=${SUP_NUM_SAMPLES:-1000000}
export SUP_SCRAMBLE_MIN=${SUP_SCRAMBLE_MIN:-1}
export SUP_SCRAMBLE_MAX=${SUP_SCRAMBLE_MAX:-10}
export SUP_BATCH_SIZE=${SUP_BATCH_SIZE:-1024}
export SUP_EPOCHS=${SUP_EPOCHS:-20}
export SUP_VAL_RATIO=${SUP_VAL_RATIO:-0.05}
export SUP_GRAD_CLIP=${SUP_GRAD_CLIP:-1.0}
export SUP_SEED=${SUP_SEED:-42}
export SUP_DEVICE=${SUP_DEVICE:-"cuda"}
export SUP_AMP=${SUP_AMP:-"1"}

# Map REINFORCE network envs into supervised script envs
export SUP_HIDDEN="${HIDDEN_LAYERS}"
export SUP_ACTIVATION="${ACTIVATION}"
export SUP_NORM="${NORM}"
export SUP_DROPOUT="${DROPOUT}"
export SUP_LR="${LR}"
export SUP_WEIGHT_DECAY="${WEIGHT_DECAY}"

# W&B  (disable by setting SUP_WANDB_MODE=disabled)
export SUP_WANDB_PROJECT=${SUP_WANDB_PROJECT:-"RubiksRLSup"}
export SUP_WANDB_MODE=${SUP_WANDB_MODE:-"online"}
# export SUP_WANDB_ENTITY=your_entity
export SUP_RUN_NAME=${SUP_RUN_NAME:-"supervised_pretrain_${SLURM_JOB_ID}"}

# Output checkpoint path to be consumed by REINFORCE
export SUP_SAVE_PATH=${SUP_SAVE_PATH:-"checkpoints/supervised_mlp_xlong_large.pt"}

# Conda env
source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate torch

echo "Starting supervised pretraining..."
python train_supervised.py

echo "Supervised pretraining completed. Checkpoint at: ${SUP_SAVE_PATH}"