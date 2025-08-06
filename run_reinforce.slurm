#!/bin/bash
#SBATCH --job-name=rubiks_reinforce
#SBATCH --output=rubiks_reinforce_%j.out
#SBATCH --error=rubiks_reinforce_%j.err
#SBATCH --mem=64G
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1

# Optional: expose config via env vars (override as needed)
# Core env params
export SCRAMBLE_DISTANCE=${SCRAMBLE_DISTANCE:-5}
export MAX_STEPS=${MAX_STEPS:-10}
export STEP_PENALTY=${STEP_PENALTY:-0.0}
export SEED=${SEED:-42}
export TOTAL_EPISODES=${TOTAL_EPISODES:-200000}
export EVAL_EVERY_EPISODES=${EVAL_EVERY_EPISODES:-250}
export EVAL_EPISODES_PER_DISTANCE=${EVAL_EPISODES_PER_DISTANCE:-30}
export EVAL_DISTANCES=${EVAL_DISTANCES:-"5,10,15,20"}

# Curriculum params
export CURRICULUM=${CURRICULUM:-"false"}
export CUR_START_DISTANCE=${CUR_START_DISTANCE:-1}
export CUR_MAX_DISTANCE=${CUR_MAX_DISTANCE:-20}
export CUR_THRESHOLD=${CUR_THRESHOLD:-0.8}
export CUR_WINDOW_EPISODES=${CUR_WINDOW_EPISODES:-500}
export CUR_MIN_EPISODES_PER_LEVEL=${CUR_MIN_EPISODES_PER_LEVEL:-2000}
export CUR_INCREASE_STEP=${CUR_INCREASE_STEP:-1}

# Policy Gradient specific params (defaults per your request)
export GAMMA=${GAMMA:-0.995}
export LR=${LR:-3e-4}
export HIDDEN_LAYERS=${HIDDEN_LAYERS:-"1024,1024,1024,1024"}
export ACTIVATION=${ACTIVATION:-"silu"}
export NORM=${NORM:-"layernorm"}
export DROPOUT=${DROPOUT:-0.05}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
export ENTROPY_COEF=${ENTROPY_COEF:-0.003}
export VALUE_HEAD=${VALUE_HEAD:-"true"}
export VALUE_COEF=${VALUE_COEF:-0.5}
export REWARD_TO_GO=${REWARD_TO_GO:-"true"}
export NORMALIZE_RETURNS=${NORMALIZE_RETURNS:-"true"}
export MAX_EPISODE_LEN=${MAX_EPISODE_LEN:-100}

# Pretrained checkpoint integration
# Point this to the checkpoint produced by run_supervised.slurm
export PRETRAINED_PATH=${PRETRAINED_PATH:-"checkpoints/supervised_mlp_xlong_large.pt"}
# Optionally freeze the backbone for the first N episodes to stabilize fine-tuning
export FREEZE_BACKBONE_EPOCHS=${FREEZE_BACKBONE_EPOCHS:-0}

# W&B params
export WANDB_PROJECT=${WANDB_PROJECT:-"RubiksRL"}
# Set WANDB_MODE to "disabled" if you don't want to log
export WANDB_MODE=${WANDB_MODE:-"online"}
# export WANDB_ENTITY=your_entity
# export RUN_NAME="rubiks_reinforce_${SLURM_JOB_ID}"

source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate torch

echo "Using PRETRAINED_PATH=${PRETRAINED_PATH}"
echo "Using FREEZE_BACKBONE_EPOCHS=${FREEZE_BACKBONE_EPOCHS}"

# Launch training
python train_reinforce.py