import os
import math
import time
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

from envs.rubiks_env import RubiksCubeEnv, evaluate_policy


# =========================
# Utilities
# =========================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.from_numpy(self.obs[idxs]).to(self.device)
        actions = torch.from_numpy(self.actions[idxs]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idxs]).to(self.device)
        next_obs = torch.from_numpy(self.next_obs[idxs]).to(self.device)
        dones = torch.from_numpy(self.dones[idxs]).to(self.device)
        return obs, actions, rewards, next_obs, dones


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, ...] = (512, 512)):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

        # Kaiming initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Config:
    # Env
    scramble_distance: int = 5
    max_steps: int = 15  # cap per episode
    step_penalty: float = 0.0
    seed: int = 42

    # Curriculum (enabled by default per user request)
    curriculum: bool = True
    cur_start_distance: int = 1
    cur_max_distance: int = 20
    cur_threshold: float = 0.8
    cur_window_episodes: int = 500
    cur_min_episodes_per_level: int = 2000
    cur_increase_step: int = 1

    # DQN
    gamma: float = 0.995
    lr: float = 1e-3
    batch_size: int = 256
    buffer_size: int = 200_000
    train_start_size: int = 10_000
    target_update_steps: int = 2_000
    # Soft target update coefficient (Polyak). If >0, apply every gradient step; if 0, fall back to hard update.
    tau: float = 0.005
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000

    # Network
    hidden1: int = 512
    hidden2: int = 512

    # Training loop
    total_episodes: int = 50_000
    max_episode_len: int = 100  # hard safety cap if env max_steps is higher
    eval_every_episodes: int = 250
    eval_episodes_per_distance: int = 30
    eval_distances: Tuple[int, ...] = (5, 10, 15, 20)

    # Logging
    wandb_project: str = "RubiksRL"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # "offline" or "disabled"
    run_name: Optional[str] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def epsilon_by_step(step: int, cfg: Config) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    slope = (cfg.eps_end - cfg.eps_start) / cfg.eps_decay_steps
    return cfg.eps_start + slope * step


def select_action(qnet: MLP, obs: np.ndarray, epsilon: float, n_actions: int, device: torch.device, prev_action: Optional[int] = None) -> int:
    """
    Epsilon-greedy with immediate inverse-move masking if prev_action is provided.
    Masking is applied only to the inverse of the immediately previous action.
    """
    # Mapping consistent with envs.rubiks_env.INVERSE_ACTION
    inverse_map = {0:1,1:0,2:3,3:2,4:5,5:4,6:7,7:6,8:9,9:8,10:11,11:10}
    if np.random.rand() < epsilon:
        # sample uniformly excluding the inverse of prev_action if present
        if prev_action is not None and prev_action in inverse_map:
            inv = inverse_map[prev_action]
            choices = [a for a in range(n_actions) if a != inv]
            return int(np.random.choice(choices))
        return int(np.random.randint(n_actions))
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
        q = qnet(obs_t)  # [1, n_actions]
        if prev_action is not None and prev_action in inverse_map:
            inv = inverse_map[prev_action]
            q = q.clone()
            q[0, inv] = -float("inf")
        return int(q.argmax(dim=1).item())


def dqn_loss(qnet: MLP, tgt: MLP, batch, gamma: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Double DQN with Huber loss.
    Uses online net to select argmax action on next state, target net to evaluate it.
    """
    obs, actions, rewards, next_obs, dones = batch
    # Current Q(s,a)
    q = qnet(obs).gather(1, actions.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        # Online net selects next action
        next_actions = qnet(next_obs).argmax(dim=1, keepdim=True)  # [B,1]
        # Target net evaluates selected action
        next_q_eval = tgt(next_obs).gather(1, next_actions).squeeze(1)
        target = rewards + (1.0 - dones) * gamma * next_q_eval

    loss = F.smooth_l1_loss(q, target)

    with torch.no_grad():
        td_err = (q - target).abs().mean().item()
        q_mean = q.mean().item()
        target_mean = target.mean().item()
    return loss, {"td_error": td_err, "q_mean": q_mean, "target_mean": target_mean}


def evaluate_agent(qnet: MLP, device: torch.device, distances: Tuple[int, ...], episodes_per_distance: int, step_penalty: float) -> Dict[int, Dict[str, float]]:
    def policy_fn(obs: np.ndarray) -> int:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            q = qnet(obs_t)
            return int(q.argmax(dim=1).item())
    return evaluate_policy(policy_fn, distances=list(distances), episodes_per_distance=episodes_per_distance, step_penalty=step_penalty)


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Env probe to get dims and actions (probe uses starting distance or curriculum start)
    probe_dist = cfg.cur_start_distance if getattr(cfg, "curriculum", False) else cfg.scramble_distance
    probe_env = RubiksCubeEnv(scramble_distance=probe_dist, max_steps=cfg.max_steps, step_penalty=cfg.step_penalty, seed=cfg.seed)
    obs, _ = probe_env.reset()
    obs_dim = obs.shape[0]
    n_actions = probe_env.action_space.n

    # Networks
    qnet = MLP(obs_dim, n_actions, hidden=(cfg.hidden1, cfg.hidden2)).to(device)
    tgt = MLP(obs_dim, n_actions, hidden=(cfg.hidden1, cfg.hidden2)).to(device)
    tgt.load_state_dict(qnet.state_dict())
    tgt.eval()

    opt = optim.Adam(qnet.parameters(), lr=cfg.lr, eps=1e-5)

    # Buffer
    rb = ReplayBuffer(cfg.buffer_size, obs_dim, device)

    # W&B
    if WANDB_AVAILABLE and cfg.wandb_mode != "disabled":
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=asdict(cfg),
            name=cfg.run_name,
            mode=cfg.wandb_mode,
        )
        wandb.watch(qnet, log="all", log_freq=500)

    global_step = 0
    best_eval_success = -1.0

    start_time = time.time()

    # Curriculum state
    current_distance = probe_dist
    level_start_ep = 1
    recent_solved: List[float] = []  # 1.0 if episode solved else 0.0

    for ep in range(1, cfg.total_episodes + 1):
        # If curriculum is enabled, use current_distance; else use fixed cfg.scramble_distance
        dist_for_episode = current_distance if getattr(cfg, "curriculum", False) else cfg.scramble_distance
        env = RubiksCubeEnv(scramble_distance=dist_for_episode, max_steps=cfg.max_steps, step_penalty=cfg.step_penalty, seed=cfg.seed + ep)
        obs, info = env.reset()
        total_reward = 0.0
        solved = False

        prev_action = None
        for t in range(cfg.max_episode_len):
            epsilon = epsilon_by_step(global_step, cfg)
            action = select_action(qnet, obs, epsilon, n_actions, device, prev_action=prev_action)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            rb.add(obs, action, reward, next_obs, done)
            total_reward += reward
            obs = next_obs
            prev_action = action
            global_step += 1

            # Update
            if rb.size >= cfg.train_start_size:
                batch = rb.sample(cfg.batch_size)
                loss, stats = dqn_loss(qnet, tgt, batch, cfg.gamma)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # Gradient clipping to stabilize training
                nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=10.0)
                opt.step()

                # Soft target update (Polyak)
                if cfg.tau > 0.0:
                    with torch.no_grad():
                        for p_t, p in zip(tgt.parameters(), qnet.parameters()):
                            p_t.data.lerp_(p.data, cfg.tau)

                if WANDB_AVAILABLE and cfg.wandb_mode != "disabled":
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/td_error": stats["td_error"],
                        "train/q_mean": stats["q_mean"],
                        "train/target_mean": stats.get("target_mean", 0.0),
                        "train/epsilon": epsilon,
                        "train/buffer_size": rb.size,
                        "train/global_step": global_step,
                    }, step=global_step)

            # Optional periodic hard target update if tau == 0 (or in addition to soft for safety but less frequent)
            if cfg.tau == 0.0 and global_step % cfg.target_update_steps == 0 and rb.size >= cfg.train_start_size:
                tgt.load_state_dict(qnet.state_dict())

            if done:
                solved = terminated
                break

        # Track solve for curriculum window
        recent_solved.append(1.0 if solved else 0.0)
        if len(recent_solved) > cfg.cur_window_episodes:
            recent_solved.pop(0)

        # Episode logging
        if WANDB_AVAILABLE and cfg.wandb_mode != "disabled":
            log_data = {
                "episode/return": total_reward,
                "episode/length": t + 1,
                "episode/solved": float(solved),
                "episode/ep": ep,
                "time/elapsed_min": (time.time() - start_time) / 60.0,
            }
            if getattr(cfg, "curriculum", False):
                log_data.update({
                    "curriculum/current_distance": dist_for_episode,
                    "curriculum/level_start_ep": level_start_ep,
                    "curriculum/recent_window": len(recent_solved),
                    "curriculum/recent_success": float(np.mean(recent_solved)) if recent_solved else 0.0,
                })
            wandb.log(log_data, step=global_step)

        # Periodic evaluation
        if ep % cfg.eval_every_episodes == 0:
            eval_metrics = evaluate_agent(
                qnet, device, distances=cfg.eval_distances,
                episodes_per_distance=cfg.eval_episodes_per_distance,
                step_penalty=cfg.step_penalty
            )
            # Log per-distance
            if WANDB_AVAILABLE and cfg.wandb_mode != "disabled":
                flat = {}
                for d, m in eval_metrics.items():
                    flat[f"eval/d{d}_success_rate"] = m["success_rate"]
                    flat[f"eval/d{d}_avg_steps_when_solved"] = m["avg_steps_when_solved"]
                    flat[f"eval/d{d}_avg_return"] = m["avg_return"]
                if getattr(cfg, "curriculum", False):
                    flat["curriculum/current_distance"] = dist_for_episode
                    flat["curriculum/recent_success"] = float(np.mean(recent_solved)) if recent_solved else 0.0
                    flat["curriculum/level_start_ep"] = level_start_ep
                wandb.log(flat, step=global_step)

            # Track best on distance=5
            sr5 = eval_metrics.get(5, {}).get("success_rate", 0.0)
            best_eval_success = max(best_eval_success, sr5)

            # Curriculum progression check: only if curriculum enabled
            if getattr(cfg, "curriculum", False):
                # Conditions: at least min episodes at this level, window full, success above threshold
                episodes_on_level = ep - level_start_ep + 1
                window_full = len(recent_solved) >= cfg.cur_window_episodes
                recent_success = float(np.mean(recent_solved)) if recent_solved else 0.0
                can_increase = (
                    episodes_on_level >= cfg.cur_min_episodes_per_level and
                    window_full and
                    recent_success >= cfg.cur_threshold and
                    current_distance < cfg.cur_max_distance
                )
                if can_increase:
                    current_distance = min(cfg.cur_max_distance, current_distance + cfg.cur_increase_step)
                    level_start_ep = ep + 1
                    recent_solved.clear()

    if WANDB_AVAILABLE and cfg.wandb_mode != "disabled":
        wandb.summary["best_success_rate_d5"] = best_eval_success
        wandb.finish()


def build_cfg_from_env() -> Config:
    # Simple environment overrides to make Slurm usage easy
    def _get(name: str, default):
        return os.getenv(name, default)

    cfg = Config(
        scramble_distance=int(_get("SCRAMBLE_DISTANCE", 5)),
        max_steps=int(_get("MAX_STEPS", 40)),
        step_penalty=float(_get("STEP_PENALTY", 0.0)),
        seed=int(_get("SEED", 42)),
        gamma=float(_get("GAMMA", 0.995)),
        lr=float(_get("LR", 1e-4)),  # safer default
        batch_size=int(_get("BATCH_SIZE", 256)),
        buffer_size=int(_get("BUFFER_SIZE", 200_000)),
        train_start_size=int(_get("TRAIN_START_SIZE", 10_000)),
        target_update_steps=int(_get("TARGET_UPDATE_STEPS", 2_000)),
        eps_start=float(_get("EPS_START", 1.0)),
        eps_end=float(_get("EPS_END", 0.1)),  # slightly higher floor for sparse rewards
        eps_decay_steps=int(_get("EPS_DECAY_STEPS", 300_000)),  # slower decay
        hidden1=int(_get("HIDDEN1", 512)),
        hidden2=int(_get("HIDDEN2", 512)),
        total_episodes=int(_get("TOTAL_EPISODES", 50_000)),
        max_episode_len=int(_get("MAX_EPISODE_LEN", 100)),
        eval_every_episodes=int(_get("EVAL_EVERY_EPISODES", 250)),
        eval_episodes_per_distance=int(_get("EVAL_EPISODES_PER_DISTANCE", 30)),
        wandb_project=str(_get("WANDB_PROJECT", "RubiksRL")),
        wandb_entity=os.getenv("WANDB_ENTITY", None),
        wandb_mode=str(_get("WANDB_MODE", "online")),
        run_name=os.getenv("RUN_NAME", None),
        device=str(_get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")),
    )

    # Distances
    distances_str = os.getenv("EVAL_DISTANCES", "5,10,15,20")
    try:
        cfg.eval_distances = tuple(int(x.strip()) for x in distances_str.split(",") if x.strip())
    except Exception:
        cfg.eval_distances = (5, 10, 15, 20)

    # Curriculum env overrides via environment
    cfg.curriculum = str(os.getenv("CURRICULUM", "true")).lower() in ("1", "true", "yes", "y")
    cfg.cur_start_distance = int(os.getenv("CUR_START_DISTANCE", 1))
    cfg.cur_max_distance = int(os.getenv("CUR_MAX_DISTANCE", 20))
    cfg.cur_threshold = float(os.getenv("CUR_THRESHOLD", 0.8))
    cfg.cur_window_episodes = int(os.getenv("CUR_WINDOW_EPISODES", 500))
    cfg.cur_min_episodes_per_level = int(os.getenv("CUR_MIN_EPISODES_PER_LEVEL", 2000))
    cfg.cur_increase_step = int(os.getenv("CUR_INCREASE_STEP", 1))

    return cfg


def main():
    cfg = build_cfg_from_env()
    train(cfg)


if __name__ == "__main__":
    main()