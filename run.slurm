#!/bin/bash
#SBATCH --job-name=rubiks_dqn
#SBATCH --output=rubiks_dqn_%j.out
#SBATCH --error=rubiks_dqn_%j.err
#SBATCH --mem=64G
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1

# Optional: expose config via env vars (override as needed)
# Core env params
export SCRAMBLE_DISTANCE=${SCRAMBLE_DISTANCE:-5}
export MAX_STEPS=${MAX_STEPS:-40}
export STEP_PENALTY=${STEP_PENALTY:-0.0}
export SEED=${SEED:-42}
export TOTAL_EPISODES=${TOTAL_EPISODES:-50000}
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

# W&B params
export WANDB_PROJECT=${WANDB_PROJECT:-"RubiksRL"}
# Set WANDB_MODE to "disabled" if you don't want to log
export WANDB_MODE=${WANDB_MODE:-"online"}
# export WANDB_ENTITY=your_entity
# export RUN_NAME="rubiks_dqn_${SLURM_JOB_ID}"

source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate torch

# Launch training
python train_dqn.py