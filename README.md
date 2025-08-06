📝 [**[Experiments Report]**](Report.md)

📊 [**[Raw Experimental Results]**]([Report.md](https://api.wandb.ai/links/lukaskuhn-lku/n9tmf9bs))

# RubiksRL - PuzzleCube Solvers with DQN, REINFORCE, PPO and Supervised Pretraining


This repository trains agents to solve a Rubik's Cube (via the PuzzleCube implementation) using multiple approaches:
- Deep Q-Network (DQN) for value-based learning
- REINFORCE (policy gradient) with optional value baseline and curriculum
- Proximal Policy Optimization (PPO) with GAE and minibatch updates
- Supervised pretraining of an MLP policy from synthetic inverse-move labels, compatible with REINFORCE fine-tuning

It logs metrics to Weights & Biases (W&B) and evaluates performance across multiple scramble distances.

Components:
- [`envs/rubiks_env.py`](envs/rubiks_env.py): Minimal Gym-like environment over PuzzleCube with reward shaping, immediate inverse-move masking, and evaluator utilities.
- [`utils/puzzle_cube.py`](utils/puzzle_cube.py): Thin wrapper around BatchCube; exposes move set and solved checks.
- [`utils/batch_cube.py`](utils/batch_cube.py): Numpy implementation of batched cube representation and moves.
- [`train_dqn.py`](train_dqn.py): DQN agent with replay buffer, target network, epsilon-greedy with inverse-move masking, curriculum, W&B logging, periodic evaluation.
- [`train_reinforce.py`](train_reinforce.py): REINFORCE with optional value head (baseline), entropy regularization, returns normalization, pretraining warm-start, curriculum, W&B logging, periodic evaluation.
- [`train_ppo.py`](train_ppo.py): PPO with GAE, clipping, entropy, cosine LR schedule, curriculum, W&B logging, periodic evaluation.
- [`train_supervised.py`](train_supervised.py): Supervised pretraining that generates pairs (state, next-inverse-move) from random scrambles. Produces checkpoint that loads into REINFORCE policy.
- SLURM job scripts:
  - [`run.slurm`](run.slurm): DQN job.
  - [`run_reinforce.slurm`](run_reinforce.slurm): REINFORCE job with pretraining support.
  - [`run_ppo.slurm`](run_ppo.slurm): PPO job.
  - [`run_supervised.slurm`](run_supervised.slurm): Supervised pretraining job.

Requirements:
- Python 3.9+
- PyTorch
- numpy
- wandb (optional; can be disabled)

Install:
- pip install "torch<3" numpy wandb

Environment and Observation/Action Spaces
- Observation: 324-dim float32 vector representing a (54, 6) one-hot per sticker-color via [`envs.rubiks_env.py:cube_to_obs()`](envs/rubiks_env.py:52).
- Action space: 12 discrete moves in the order defined by [`utils/puzzle_cube.py:valid_moves`](utils/puzzle_cube.py:13) — ["L","L'","R","R'","U","U'","D","D'","F","F'","B","B'"].
- Environment wrapper: [`envs/rubiks_env.py:RubiksCubeEnv`](envs/rubiks_env.py:58)
  - Reward = terminal bonus (+1 if solved) - step_penalty + shaping_alpha * (prev_manhattan - cur_manhattan).
  - Manhattan distance heuristic available via [`utils/puzzle_cube.py:PuzzleCube.manhattan_distance`](utils/puzzle_cube.py:74).
  - Immediate inverse-move masking to avoid back-and-forth moves at both env level and some agents.
  - Evaluator: [`envs/rubiks_env.py:evaluate_policy`](envs/rubiks_env.py:167) runs multiple episodes across configured distances and reports success rate, average steps (when solved), and average return.

Implemented Algorithms

1) DQN
File: [`train_dqn.py`](train_dqn.py)
- Network: MLP with ReLU layers [`train_dqn.py:MLP`](train_dqn.py:64)
- Replay Buffer: [`train_dqn.py:ReplayBuffer`](train_dqn.py:33)
- Double DQN + Huber loss: [`train_dqn.py:dqn_loss`](train_dqn.py:170)
- Epsilon-greedy with immediate inverse masking (exclude only the inverse of previous action): [`train_dqn.py:select_action`](train_dqn.py:146)
- Target net updates: soft (Polyak, tau) each step or optional hard update cadence
- Curriculum learning across scramble distance levels
- Periodic evaluation across distances via evaluator

Key configurable parameters (via env; defaults shown):
- SCRAMBLE_DISTANCE=5
- MAX_STEPS=40
- STEP_PENALTY=0.0
- SEED=42
- GAMMA=0.995
- LR=1e-4
- BATCH_SIZE=256
- BUFFER_SIZE=200000
- TRAIN_START_SIZE=10000
- TARGET_UPDATE_STEPS=2000
- TAU=0.005
- EPS_START=1.0
- EPS_END=0.1
- EPS_DECAY_STEPS=300000
- HIDDEN1=512, HIDDEN2=512
- TOTAL_EPISODES=50000
- MAX_EPISODE_LEN=100
- EVAL_EVERY_EPISODES=250
- EVAL_EPISODES_PER_DISTANCE=30
- EVAL_DISTANCES="5,10,15,20"
- Curriculum toggles: CURRICULUM=true|false, CUR_START_DISTANCE=1, CUR_MAX_DISTANCE=20, CUR_THRESHOLD=0.8, CUR_WINDOW_EPISODES=500, CUR_MIN_EPISODES_PER_LEVEL=2000, CUR_INCREASE_STEP=1
- W&B: WANDB_PROJECT="RubiksRL", WANDB_ENTITY, WANDB_MODE="online|offline|disabled", RUN_NAME

Run locally:
- WANDB_MODE=disabled python train_dqn.py

2) REINFORCE (Policy Gradient)
File: [`train_reinforce.py`](train_reinforce.py)
- Policy: flexible MLP with optional LayerNorm and SiLU/ReLU activation, optional dropout, and optional value head baseline [`train_reinforce.py:MLPPolicy`](train_reinforce.py:34)
- Advantage = returns - value (when baseline used); supports reward-to-go and returns normalization
- Entropy regularization
- Gradient clipping; AdamW with weight decay
- Optional supervised pretraining warm start and backbone freezing for initial episodes
- Curriculum learning over scramble distance
- Periodic evaluation logging per distance

Key configurable parameters (via env; defaults shown):
- SCRAMBLE_DISTANCE=5, MAX_STEPS=40, STEP_PENALTY=0.0, SEED=42
- GAMMA=0.995
- LR=1e-4
- HIDDEN1=512, HIDDEN2=512 or HIDDEN_LAYERS (comma-separated, overrides HIDDEN1/2)
- ACTIVATION=relu|silu
- NORM=none|layernorm
- DROPOUT=0.0
- ENTROPY_COEF=0.0
- VALUE_HEAD=true|false
- VALUE_COEF=0.5
- REWARD_TO_GO=true|false
- NORMALIZE_RETURNS=true|false
- WEIGHT_DECAY=0.0
- GRAD_CLIP=10.0
- TOTAL_EPISODES=50000
- MAX_EPISODE_LEN=100
- EVAL_EVERY_EPISODES=250
- EVAL_EPISODES_PER_DISTANCE=30
- EVAL_DISTANCES="5,10,15,20"
- Curriculum: CURRICULUM, CUR_START_DISTANCE, CUR_MAX_DISTANCE, CUR_THRESHOLD, CUR_WINDOW_EPISODES, CUR_MIN_EPISODES_PER_LEVEL, CUR_INCREASE_STEP
- Pretraining: PRETRAINED_PATH="checkpoints/supervised_mlp.pt", FREEZE_BACKBONE_EPOCHS=0
- W&B: WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE, RUN_NAME

Run locally (no logging):
- WANDB_MODE=disabled python train_reinforce.py

3) PPO
File: [`train_ppo.py`](train_ppo.py)
- Actor-Critic network with SiLU/LayerNorm by default [`train_ppo.py:ActorCritic`](train_ppo.py:33)
- GAE advantages, clipped policy loss, clipped value loss, entropy regularization
- AdamW optimizer + cosine LR schedule with warmup
- Rollout buffer with vectorized storage
- Curriculum learning, periodic evaluation

Key configurable parameters (via env; defaults shown):
- SCRAMBLE_DISTANCE=5, MAX_STEPS=10, STEP_PENALTY=0.0, SEED=42
- CURRICULUM=true, CUR_START_DISTANCE=1, CUR_MAX_DISTANCE=20, CUR_THRESHOLD=0.8, CUR_WINDOW_EPISODES=500, CUR_MIN_EPISODES_PER_LEVEL=2000, CUR_INCREASE_STEP=1
- GAMMA=0.995, GAE_LAMBDA=0.95
- LR=3e-4, WEIGHT_DECAY=1e-6
- CLIP_COEF=0.2, VF_CLIP_COEF=0.2
- ENTROPY_COEF=0.003, VALUE_COEF=0.5, GRAD_CLIP=1.0
- NORMALIZE_ADVANTAGES=true
- Network: HIDDEN_LAYERS="1024,1024,1024" (default in code), ACTIVATION=silu, NORM=layernorm, DROPOUT=0.05
- ROLLOUT_STEPS=2048, PPO_EPOCHS=10, MINIBATCH_SIZE=256, MAX_EPISODE_LEN=100
- EVAL_EVERY_UPDATES=20, EVAL_EPISODES_PER_DISTANCE=30, EVAL_DISTANCES="5,10,15,20"
- TOTAL_UPDATES=4000
- W&B: WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE, RUN_NAME

Run locally (no logging):
- WANDB_MODE=disabled python train_ppo.py

4) Supervised Pretraining
File: [`train_supervised.py`](train_supervised.py)

Goal
- Train an MLP to predict the immediate inverse of the last scramble action from the current cube state. This provides a strong local policy prior that can be loaded into REINFORCE for fine-tuning.

Dataset Generation
- See [`train_supervised.py:generate_dataset`](train_supervised.py:116).
- Sample a scramble length L in [SUP_SCRAMBLE_MIN, SUP_SCRAMBLE_MAX].
- From a solved cube, apply random scramble actions a_1..a_L.
- For each intermediate state s_t (after a_1..a_t), label y_t = inverse(a_t).
- Aggregate approximately SUP_NUM_SAMPLES pairs.

Model
- Same architecture class as REINFORCE policy backbone+pi for compatibility: [`train_supervised.py:MLPPolicy`](train_supervised.py:23)
- Optimizer: AdamW; AMP optional; gradient clipping.
- Best checkpoint is saved to SUP_SAVE_PATH containing state_dict and architecture metadata.
- Utility to load into REINFORCE policy: [`train_supervised.py:load_pretrained_into_reinforce_policy`](train_supervised.py:292)

Key configurable parameters (via env; defaults shown in script or SLURM):
- SUP_NUM_SAMPLES=200000
- SUP_SCRAMBLE_MIN=1
- SUP_SCRAMBLE_MAX=10
- SUP_BATCH_SIZE=1024
- SUP_EPOCHS=10
- SUP_LR=3e-4
- SUP_WEIGHT_DECAY=0.01
- SUP_GRAD_CLIP=1.0
- SUP_ACTIVATION=relu|silu
- SUP_NORM=none|layernorm
- SUP_DROPOUT=0.0
- SUP_HIDDEN="512,512"
- SUP_SEED=42
- SUP_VAL_RATIO=0.05
- SUP_DEVICE=cuda|cpu
- SUP_AMP=1
- SUP_WANDB_PROJECT, SUP_WANDB_ENTITY, SUP_WANDB_MODE, SUP_RUN_NAME
- SUP_SAVE_PATH="checkpoints/supervised_mlp.pt"

Run locally (no logging):
- SUP_WANDB_MODE=disabled python train_supervised.py

How to Run with SLURM

Common SLURM Directives
All provided job scripts specify these directives at the top (examples below). Override as needed per your cluster:

- --job-name: Name of the job.
- --output: Stdout file path with %j for job ID.
- --error: Stderr file path with %j for job ID.
- --mem: Memory allocation, e.g., 32G or 64G.
- --time: Walltime limit, e.g., 48:00:00 for 48 hours or 240:00:00 for 10 days.
- --gres=gpu:1: Request 1 GPU. Adjust if multi-GPU is available.

The scripts also assume a conda environment "torch":
- source /home/$USER/miniconda3/etc/profile.d/conda.sh
- conda activate torch

Weights & Biases
- Set WANDB_MODE=disabled to turn off logging.
- Use WANDB_PROJECT to group runs, and optionally WANDB_ENTITY and RUN_NAME.

A) DQN Job
Script: [`run.slurm`](run.slurm)

Important environment variables it exports:
- Core env: SCRAMBLE_DISTANCE, MAX_STEPS, STEP_PENALTY, SEED, TOTAL_EPISODES, EVAL_EVERY_EPISODES, EVAL_EPISODES_PER_DISTANCE, EVAL_DISTANCES
- Curriculum: CURRICULUM, CUR_START_DISTANCE, CUR_MAX_DISTANCE, CUR_THRESHOLD, CUR_WINDOW_EPISODES, CUR_MIN_EPISODES_PER_LEVEL, CUR_INCREASE_STEP
- W&B: WANDB_PROJECT, WANDB_MODE, WANDB_ENTITY, RUN_NAME
- Launch: python train_dqn.py

Example submit:
- sbatch run.slurm

B) REINFORCE Job
Script: [`run_reinforce.slurm`](run_reinforce.slurm)

Highlights:
- Core env, curriculum env similar to DQN.
- Policy hyperparameters: GAMMA, LR, HIDDEN_LAYERS, ACTIVATION, NORM, DROPOUT, WEIGHT_DECAY, ENTROPY_COEF, VALUE_HEAD, VALUE_COEF, REWARD_TO_GO, NORMALIZE_RETURNS, MAX_EPISODE_LEN.
- Pretraining: PRETRAINED_PATH (point to supervised checkpoint), FREEZE_BACKBONE_EPOCHS to freeze backbone for first N episodes.
- W&B: WANDB_PROJECT, WANDB_MODE, WANDB_ENTITY, RUN_NAME.
- Launch: python train_reinforce.py

Example submit:
- sbatch run_reinforce.slurm

C) PPO Job
Script: [`run_ppo.slurm`](run_ppo.slurm)

Highlights:
- Core env and curriculum env variables.
- PPO hyperparameters: GAMMA, GAE_LAMBDA, LR, WEIGHT_DECAY, CLIP_COEF, VF_CLIP_COEF, ENTROPY_COEF, VALUE_COEF, GRAD_CLIP, NORMALIZE_ADVANTAGES.
- Network: HIDDEN_LAYERS, ACTIVATION, NORM, DROPOUT.
- Rollout/Optimization: ROLLOUT_STEPS, PPO_EPOCHS, MINIBATCH_SIZE, MAX_EPISODE_LEN.
- Evaluation cadence: EVAL_EVERY_UPDATES, EVAL_EPISODES_PER_DISTANCE, EVAL_DISTANCES.
- Training budget: TOTAL_UPDATES.
- W&B: WANDB_PROJECT, WANDB_MODE, WANDB_ENTITY, RUN_NAME.
- Launch: python train_ppo.py

Example submit:
- sbatch run_ppo.slurm

D) Supervised Pretraining Job
Script: [`run_supervised.slurm`](run_supervised.slurm)

Highlights:
- Mirrors REINFORCE network defaults via envs HIDDEN_LAYERS/ACTIVATION/NORM/DROPOUT so checkpoints are compatible.
- Training params: LR, WEIGHT_DECAY, SUP_NUM_SAMPLES, SUP_SCRAMBLE_MIN, SUP_SCRAMBLE_MAX, SUP_BATCH_SIZE, SUP_EPOCHS, SUP_VAL_RATIO, SUP_GRAD_CLIP, SUP_SEED, SUP_DEVICE, SUP_AMP.
- W&B for supervised: SUP_WANDB_PROJECT, SUP_WANDB_MODE, SUP_WANDB_ENTITY, SUP_RUN_NAME.
- Output checkpoint path: SUP_SAVE_PATH (ensure directory exists; script will create directories).
- Launch: python train_supervised.py

Example submit:
- sbatch run_supervised.slurm

Tips
- To disable W&B logging on cluster, set WANDB_MODE=disabled (or SUP_WANDB_MODE for supervised).
- To use curriculum, set CURRICULUM=true and adjust thresholds and windows. Both DQN and REINFORCE progress after the window is full and recent success exceeds CUR_THRESHOLD; PPO uses a similar window condition adapted per updates.
- step_penalty can help discourage long trajectories; shaping_alpha inside the env provides small dense shaping from Manhattan distance improvements.

Logging and Evaluation

Key logging (W&B when enabled):
- DQN: train loss, TD error, Q mean, epsilon, buffer size, eval success rates and returns per distance, curriculum progress stats.
- REINFORCE: total loss, policy loss, value loss, entropy, episode return/length/solved, evaluation metrics per distance, curriculum stats, optional pretraining info.
- PPO: avg pg/value losses and entropy across minibatches, learning rate schedule progress, solved counts per rollout, evaluation metrics per distance, curriculum stats.

Evaluation
- Evaluates periodically across distances EVAL_DISTANCES with EVAL_EPISODES_PER_DISTANCE episodes each using greedy action selection.
- Metrics: success_rate, avg_steps_when_solved, avg_return.

Notes and Design Choices
- Observation is a flat 324 vector built from a one-hot sticker encoding.
- Action order matches valid_moves; immediate inverse-move masking reduces oscillation.
- Environment reward: sparse terminal + optional small step penalty + small dense shaping from Manhattan distance improvements (alpha times decrease in distance).
- Curriculum starts at CUR_START_DISTANCE and increases when the recent window meets threshold conditions until CUR_MAX_DISTANCE.

Local Quickstarts

Disable logging (recommended for quick checks):
- WANDB_MODE=disabled python train_dqn.py
- WANDB_MODE=disabled python train_reinforce.py
- WANDB_MODE=disabled python train_ppo.py
- SUP_WANDB_MODE=disabled python train_supervised.py

Reproducibility
- All scripts accept SEED; GPU determinism may still vary. Replay-based DQN results depend on buffer stochasticity.

Acknowledgements
- PuzzleCube and BatchCube implementations adapted from @jasonrute (see file headers for references).