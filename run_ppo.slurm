#!/bin/bash
#SBATCH --job-name=rubiks_ppo
#SBATCH --output=rubiks_ppo_%j.out
#SBATCH --error=rubiks_ppo_%j.err
#SBATCH --mem=64G
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1

# Core env params
export SCRAMBLE_DISTANCE=${SCRAMBLE_DISTANCE:-5}
export MAX_STEPS=${MAX_STEPS:-10}
export STEP_PENALTY=${STEP_PENALTY:-0.0}
export SEED=${SEED:-42}

# Curriculum params
export CURRICULUM=${CURRICULUM:-"true"}
export CUR_START_DISTANCE=${CUR_START_DISTANCE:-1}
export CUR_MAX_DISTANCE=${CUR_MAX_DISTANCE:-20}
export CUR_THRESHOLD=${CUR_THRESHOLD:-0.8}
export CUR_WINDOW_EPISODES=${CUR_WINDOW_EPISODES:-500}
export CUR_MIN_EPISODES_PER_LEVEL=${CUR_MIN_EPISODES_PER_LEVEL:-2000}
export CUR_INCREASE_STEP=${CUR_INCREASE_STEP:-1}

# PPO hyperparameters
export GAMMA=${GAMMA:-0.995}
export GAE_LAMBDA=${GAE_LAMBDA:-0.95}
export LR=${LR:-3e-4}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-6}
export CLIP_COEF=${CLIP_COEF:-0.2}
export VF_CLIP_COEF=${VF_CLIP_COEF:-0.2}
export ENTROPY_COEF=${ENTROPY_COEF:-0.003}
export VALUE_COEF=${VALUE_COEF:-0.5}
export GRAD_CLIP=${GRAD_CLIP:-1.0}
export NORMALIZE_ADVANTAGES=${NORMALIZE_ADVANTAGES:-"true"}

# Network
export HIDDEN_LAYERS=${HIDDEN_LAYERS:-"512,512"}
export ACTIVATION=${ACTIVATION:-"silu"}
export NORM=${NORM:-"layernorm"}
export DROPOUT=${DROPOUT:-0.05}

# Rollout/Optimization
export ROLLOUT_STEPS=${ROLLOUT_STEPS:-2048}
export PPO_EPOCHS=${PPO_EPOCHS:-10}
export MINIBATCH_SIZE=${MINIBATCH_SIZE:-256}
export MAX_EPISODE_LEN=${MAX_EPISODE_LEN:-100}

# Evaluation cadence
export EVAL_EVERY_UPDATES=${EVAL_EVERY_UPDATES:-20}
export EVAL_EPISODES_PER_DISTANCE=${EVAL_EPISODES_PER_DISTANCE:-30}
export EVAL_DISTANCES=${EVAL_DISTANCES:-"5,10,15,20"}

# Training budget
export TOTAL_UPDATES=${TOTAL_UPDATES:-4000}

# W&B params
export WANDB_PROJECT=${WANDB_PROJECT:-"RubiksRL"}
# Set WANDB_MODE to "disabled" if you don't want to log
export WANDB_MODE=${WANDB_MODE:-"online"}
# export WANDB_ENTITY=your_entity
# export RUN_NAME="rubiks_ppo_${SLURM_JOB_ID}"

source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate torch

python train_ppo.py