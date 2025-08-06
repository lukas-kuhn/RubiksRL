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
from functools import partial


# =========================
# Utilities
# =========================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPPolicy(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Tuple[int, ...] = (512, 512),
        value_head: bool = False,
        activation: str = "relu",
        norm: str = "none",
        dropout: float = 0.0,
    ):
        super().__init__()
        act_layer = nn.ReLU if activation.lower() == "relu" else nn.SiLU
        use_layernorm = norm.lower() == "layernorm"
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_layer())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            last = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.pi = nn.Linear(last, out_dim)
        self.use_value = value_head
        if value_head:
            self.v = nn.Linear(last, 1)

        # Kaiming initialization suitable for ReLU/SiLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        logits = self.pi(h)
        if self.use_value:
            value = self.v(h).squeeze(-1)
            return logits, value
        return logits, None

    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value


@dataclass
class Config:
    # Env
    scramble_distance: int = 5
    max_steps: int = 15  # cap per episode
    step_penalty: float = 0.0
    seed: int = 42

    # Curriculum (match DQN defaults)
    curriculum: bool = True
    cur_start_distance: int = 1
    cur_max_distance: int = 20
    cur_threshold: float = 0.8
    cur_window_episodes: int = 500
    cur_min_episodes_per_level: int = 2000
    cur_increase_step: int = 1

    # REINFORCE / Policy Gradient
    gamma: float = 0.995
    lr: float = 1e-3
    hidden1: int = 512
    hidden2: int = 512
    # Flexible hidden layers list (overrides hidden1/hidden2 if provided via env)
    hidden_layers: Tuple[int, ...] = ()
    activation: str = "relu"           # "relu" | "silu"
    norm: str = "none"                 # "none" | "layernorm"
    dropout: float = 0.0
    entropy_coef: float = 0.0          # optional entropy bonus
    value_head: bool = True            # use baseline via value head
    value_coef: float = 0.5            # coefficient for value loss if baseline enabled
    reward_to_go: bool = True          # use reward-to-go returns
    normalize_returns: bool = True     # normalize returns (advantages) per episode
    grad_clip: float = 10.0
    weight_decay: float = 0.0          # AdamW weight decay

    # Training loop
    total_episodes: int = 50_000
    max_episode_len: int = 100
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

    # Supervised pretraining
    pretrained_path: Optional[str] = None  # path to checkpoints/supervised_mlp.pt
    freeze_backbone_epochs: int = 0        # optionally freeze backbone for first N episodes


def compute_returns(rewards: List[float], gamma: float, reward_to_go: bool) -> List[float]:
    if reward_to_go:
        G = 0.0
        returns = [0.0] * len(rewards)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G
        return returns
    else:
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
        return [G] * len(rewards)


def evaluate_agent(policy: MLPPolicy, device: torch.device, distances: Tuple[int, ...], episodes_per_distance: int, step_penalty: float) -> Dict[int, Dict[str, float]]:
    def policy_fn(obs: np.ndarray) -> int:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            logits, _ = policy(obs_t)
            action = torch.argmax(logits, dim=-1).item()
            return int(action)
    return evaluate_policy(policy_fn, distances=list(distances), episodes_per_distance=episodes_per_distance, step_penalty=step_penalty)


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Env probe to get dims and actions
    probe_dist = cfg.cur_start_distance if getattr(cfg, "curriculum", False) else cfg.scramble_distance
    probe_env = RubiksCubeEnv(scramble_distance=probe_dist, max_steps=cfg.max_steps, step_penalty=cfg.step_penalty, seed=cfg.seed)
    obs, _ = probe_env.reset()
    obs_dim = obs.shape[0]
    n_actions = probe_env.action_space.n

    # Policy network
    # Determine hidden config
    if cfg.hidden_layers and len(cfg.hidden_layers) > 0:
        hidden = cfg.hidden_layers
    else:
        hidden = (cfg.hidden1, cfg.hidden2)
    policy = MLPPolicy(
        obs_dim,
        n_actions,
        hidden=hidden,
        value_head=cfg.value_head,
        activation=cfg.activation,
        norm=cfg.norm,
        dropout=cfg.dropout,
    ).to(device)

    # Optionally load supervised-pretrained weights
    if getattr(cfg, "pretrained_path", None):
        try:
            ckpt = torch.load(cfg.pretrained_path, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)
            missing, unexpected = policy.load_state_dict(sd, strict=False)
            print(f"[Pretrain] Loaded {cfg.pretrained_path}. Missing: {missing}, Unexpected: {unexpected}")
        except Exception as e:
            print(f"[Pretrain] Failed to load {cfg.pretrained_path}: {e}")

        # Optionally freeze backbone for warmup
        if cfg.freeze_backbone_epochs and cfg.freeze_backbone_epochs > 0:
            for p in policy.backbone.parameters():
                p.requires_grad = False

    opt = optim.AdamW(policy.parameters(), lr=cfg.lr, eps=1e-5, weight_decay=cfg.weight_decay)

    # W&B
    if WANDB_AVAILABLE and cfg.wandb_mode != "disabled":
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=asdict(cfg),
            name=cfg.run_name,
            mode=cfg.wandb_mode,
        )
        wandb.watch(policy, log="all", log_freq=500)

    global_step = 0
    best_eval_success = -1.0

    start_time = time.time()

    # Curriculum state
    current_distance = probe_dist
    level_start_ep = 1
    recent_solved: List[float] = []

    for ep in range(1, cfg.total_episodes + 1):
        dist_for_episode = current_distance if getattr(cfg, "curriculum", False) else cfg.scramble_distance
        env = RubiksCubeEnv(scramble_distance=dist_for_episode, max_steps=cfg.max_steps, step_penalty=cfg.step_penalty, seed=cfg.seed + ep)
        obs, info = env.reset()

        ep_obs: List[np.ndarray] = []
        ep_actions: List[int] = []
        ep_logps: List[float] = []
        ep_values: List[float] = []
        ep_rewards: List[float] = []

        total_reward = 0.0
        solved = False

        for t in range(cfg.max_episode_len):
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            with torch.no_grad():
                logits, value = policy(obs_t)
                # Immediate inverse-move masking without walrus operator to avoid SyntaxError
                step_info_local = locals().get("step_info", {})
                last_a = None
                if isinstance(step_info_local, dict) and ("last_action" in step_info_local) and (step_info_local.get("last_action") is not None):
                    last_a = int(step_info_local["last_action"])
                elif len(ep_actions) > 0:
                    last_a = int(ep_actions[-1])
                inv = None
                if last_a is not None:
                    inv = {0:1,1:0,2:3,3:2,4:5,5:4,6:7,7:6,8:9,9:8,10:11,11:10}.get(last_a, None)
                logits_eff = logits.clone()
                if inv is not None:
                    logits_eff[0, inv] = -float("inf")
                probs = F.softmax(logits_eff, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
                logp = dist.log_prob(action)

            next_obs, reward, terminated, truncated, step_info = env.step(int(action.item()))
            done = terminated or truncated

            # Store trajectory
            ep_obs.append(obs.copy())
            ep_actions.append(int(action.item()))
            ep_logps.append(float(logp.item()))
            if cfg.value_head and value is not None:
                ep_values.append(float(value.squeeze().item()))
            ep_rewards.append(float(reward))

            total_reward += reward
            obs = next_obs
            global_step += 1

            if done:
                solved = terminated
                break

        # Track solve for curriculum window
        recent_solved.append(1.0 if solved else 0.0)
        if len(recent_solved) > cfg.cur_window_episodes:
            recent_solved.pop(0)

        # Compute returns (and advantages if baseline)
        returns = compute_returns(ep_rewards, cfg.gamma, cfg.reward_to_go)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        if cfg.normalize_returns and len(returns) > 1:
            mean, std = returns_t.mean(), returns_t.std().clamp_min(1e-8)
            returns_t = (returns_t - mean) / std

        logps_t = torch.tensor(ep_logps, dtype=torch.float32, device=device)
        actions_t = torch.tensor(ep_actions, dtype=torch.int64, device=device)

        if cfg.value_head:
            values_t = torch.tensor(ep_values if len(ep_values) == len(returns) else [0.0]*len(returns), dtype=torch.float32, device=device)
            advantages = returns_t - values_t
        else:
            advantages = returns_t

        # Recompute log-probs (on-policy, but ensure graph)
        obs_t = torch.tensor(np.array(ep_obs, dtype=np.float32), device=device)
        logits, values_now = policy(obs_t)
        log_probs_all = F.log_softmax(logits, dim=-1)
        chosen_logp = log_probs_all.gather(1, actions_t.view(-1, 1)).squeeze(1)

        # Policy loss (maximize advantage * logpi)
        pg_loss = -(chosen_logp * advantages.detach()).mean()

        # Entropy bonus
        entropy = -(F.softmax(logits, dim=-1) * log_probs_all).sum(dim=1).mean()
        entropy_loss = -cfg.entropy_coef * entropy if cfg.entropy_coef > 0.0 else 0.0

        # Value loss
        if cfg.value_head:
            value_loss = F.mse_loss(values_now.squeeze(-1), returns_t)
        else:
            value_loss = torch.tensor(0.0, device=device)

        loss = pg_loss + cfg.value_coef * value_loss + (entropy_loss if isinstance(entropy_loss, torch.Tensor) else 0.0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip)
        opt.step()

        if WANDB_AVAILABLE and cfg.wandb_mode != "disabled":
            log_data = {
                "train/loss": float(loss.item()),
                "train/pg_loss": float(pg_loss.item()),
                "train/value_loss": float(value_loss.item()) if cfg.value_head else 0.0,
                "train/entropy": float(entropy.item()),
                "episode/return": total_reward,
                "episode/length": len(ep_rewards),
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
        # Unfreeze backbone after warmup episodes if requested
        if getattr(cfg, "freeze_backbone_epochs", 0) and ep == cfg.freeze_backbone_epochs:
            for p in policy.backbone.parameters():
                p.requires_grad = True

        if ep % cfg.eval_every_episodes == 0:
            eval_metrics = evaluate_agent(
                policy, device, distances=cfg.eval_distances,
                episodes_per_distance=cfg.eval_episodes_per_distance,
                step_penalty=cfg.step_penalty
            )
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

            sr5 = eval_metrics.get(5, {}).get("success_rate", 0.0)
            best_eval_success = max(best_eval_success, sr5)

            # Curriculum progression check
            if getattr(cfg, "curriculum", False):
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


def _parse_hidden_layers(env_val: Optional[str]) -> Tuple[int, ...]:
    if not env_val:
        return ()
    try:
        parts = [int(x.strip()) for x in env_val.split(",") if x.strip()]
        return tuple(parts)
    except Exception:
        return ()

def build_cfg_from_env() -> Config:
    def _get(name: str, default):
        return os.getenv(name, default)

    cfg = Config(
        scramble_distance=int(_get("SCRAMBLE_DISTANCE", 5)),
        max_steps=int(_get("MAX_STEPS", 40)),
        step_penalty=float(_get("STEP_PENALTY", 0.0)),
        seed=int(_get("SEED", 42)),
        gamma=float(_get("GAMMA", 0.995)),
        lr=float(_get("LR", 1e-4)),
        hidden1=int(_get("HIDDEN1", 512)),
        hidden2=int(_get("HIDDEN2", 512)),
        hidden_layers=_parse_hidden_layers(os.getenv("HIDDEN_LAYERS", "")),
        activation=str(_get("ACTIVATION", "relu")),
        norm=str(_get("NORM", "none")),
        dropout=float(_get("DROPOUT", 0.0)),
        entropy_coef=float(_get("ENTROPY_COEF", 0.0)),
        value_head=str(os.getenv("VALUE_HEAD", "true")).lower() in ("1", "true", "yes", "y"),
        value_coef=float(_get("VALUE_COEF", 0.5)),
        reward_to_go=str(os.getenv("REWARD_TO_GO", "true")).lower() in ("1", "true", "yes", "y"),
        normalize_returns=str(os.getenv("NORMALIZE_RETURNS", "true")).lower() in ("1", "true", "yes", "y"),
        weight_decay=float(_get("WEIGHT_DECAY", 0.0)),
        total_episodes=int(_get("TOTAL_EPISODES", 50_000)),
        max_episode_len=int(_get("MAX_EPISODE_LEN", 100)),
        eval_every_episodes=int(_get("EVAL_EVERY_EPISODES", 250)),
        eval_episodes_per_distance=int(_get("EVAL_EPISODES_PER_DISTANCE", 30)),
        wandb_project=str(_get("WANDB_PROJECT", "RubiksRL")),
        wandb_entity=os.getenv("WANDB_ENTITY", None),
        wandb_mode=str(_get("WANDB_MODE", "online")),
        run_name=os.getenv("RUN_NAME", None),
        device=str(_get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")),

        # Pretraining env overrides
        pretrained_path=os.getenv("PRETRAINED_PATH", None),
        freeze_backbone_epochs=int(os.getenv("FREEZE_BACKBONE_EPOCHS", "0")),
    )

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