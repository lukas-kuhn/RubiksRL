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


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_actions: int,
        hidden: Tuple[int, ...] = (1024, 1024, 1024),
        activation: str = "silu",
        norm: str = "layernorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        act_layer = nn.SiLU if activation.lower() == "silu" else nn.ReLU
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

        self.policy = nn.Linear(last, n_actions)
        self.value = nn.Linear(last, 1)

        # Kaiming init for ReLU/SiLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    @torch.no_grad()
    def get_logp(self, logits: torch.Tensor, actions: torch.Tensor):
        logp_all = F.log_softmax(logits, dim=-1)
        return logp_all.gather(1, actions.view(-1, 1)).squeeze(1)

    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray, device: torch.device) -> int:
        t = torch.from_numpy(obs).to(device).unsqueeze(0)
        logits, _ = self.forward(t)
        return int(torch.argmax(logits, dim=-1).item())


@dataclass
class Config:
    # Env
    scramble_distance: int = 5
    max_steps: int = 15
    step_penalty: float = 0.0
    seed: int = 42

    # Curriculum
    curriculum: bool = True
    cur_start_distance: int = 1
    cur_max_distance: int = 20
    cur_threshold: float = 0.8
    cur_window_episodes: int = 500
    cur_min_episodes_per_level: int = 2000
    cur_increase_step: int = 1

    # PPO core
    gamma: float = 0.995
    gae_lambda: float = 0.95
    lr: float = 3e-4
    weight_decay: float = 1e-6
    clip_coef: float = 0.2
    vf_clip_coef: float = 0.2
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    grad_clip: float = 1.0
    normalize_advantages: bool = True

    # Network
    hidden_layers: Tuple[int, ...] = (1024, 1024, 1024)
    activation: str = "silu"      # "relu" | "silu"
    norm: str = "layernorm"       # "none" | "layernorm"
    dropout: float = 0.05

    # Rollout/Optimization
    rollout_steps: int = 2048      # total timesteps collected before each update
    ppo_epochs: int = 10
    minibatch_size: int = 256
    max_episode_len: int = 100

    # Evaluation
    eval_every_updates: int = 20   # evaluate every N PPO updates
    eval_episodes_per_distance: int = 30
    eval_distances: Tuple[int, ...] = (5, 10, 15, 20)

    # Training budget (in updates)
    total_updates: int = 4000      # total PPO updates

    # Logging
    wandb_project: str = "RubiksRL"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"     # "offline" | "disabled"
    run_name: Optional[str] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_agent(ac: ActorCritic, device: torch.device, distances: Tuple[int, ...], episodes_per_distance: int, step_penalty: float) -> Dict[int, Dict[str, float]]:
    def policy_fn(obs: np.ndarray) -> int:
        return ac.greedy_action(obs, device)
    return evaluate_policy(policy_fn, distances=list(distances), episodes_per_distance=episodes_per_distance, step_penalty=step_penalty)


class RolloutBuffer:
    def __init__(self, obs_dim: int, rollout_steps: int, device: torch.device):
        self.obs = torch.zeros((rollout_steps, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((rollout_steps,), dtype=torch.long, device=device)
        self.logp = torch.zeros((rollout_steps,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((rollout_steps,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((rollout_steps,), dtype=torch.float32, device=device)
        self.values = torch.zeros((rollout_steps,), dtype=torch.float32, device=device)
        self.ptr = 0
        self.capacity = rollout_steps
        self.device = device

    def add(self, obs, action, logp, reward, done, value):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.logp[idx] = logp
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.capacity

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float):
        T = self.ptr
        advantages = torch.zeros_like(self.rewards[:T])
        last_gae = 0.0
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - (self.dones[t])
            next_value = last_value if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values[:T]
        return advantages, returns

    def get(self):
        T = self.ptr
        return (
            self.obs[:T],
            self.actions[:T],
            self.logp[:T],
            self.rewards[:T],
            self.dones[:T],
            self.values[:T],
        )

    def clear(self):
        self.ptr = 0


def cosine_lr_scheduler(optimizer, base_lr: float, total_steps: int, warmup_steps: int = 0):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / float(max(1, warmup_steps)))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Probe env to get dims and actions
    probe_dist = cfg.cur_start_distance if getattr(cfg, "curriculum", False) else cfg.scramble_distance
    probe_env = RubiksCubeEnv(scramble_distance=probe_dist, max_steps=cfg.max_steps, step_penalty=cfg.step_penalty, seed=cfg.seed)
    obs_np, _ = probe_env.reset()
    obs_dim = obs_np.shape[0]
    n_actions = probe_env.action_space.n

    # Model and optimizer
    ac = ActorCritic(
        obs_dim,
        n_actions,
        hidden=cfg.hidden_layers if cfg.hidden_layers else (1024, 1024, 1024),
        activation=cfg.activation,
        norm=cfg.norm,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = optim.AdamW(ac.parameters(), lr=cfg.lr, eps=1e-5, weight_decay=cfg.weight_decay)

    # Scheduler - total optimizer steps roughly total_updates * (rollout_steps / minibatch_size) * ppo_epochs
    opt_total_steps = cfg.total_updates * (cfg.rollout_steps // max(1, cfg.minibatch_size)) * cfg.ppo_epochs
    scheduler = cosine_lr_scheduler(optimizer, cfg.lr, total_steps=max(1, opt_total_steps), warmup_steps=max(1, int(0.03 * opt_total_steps)))

    # W&B
    if WANDB_AVAILABLE and cfg.wandb_mode != "disabled":
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=asdict(cfg),
            name=cfg.run_name,
            mode=cfg.wandb_mode,
        )
        wandb.watch(ac, log="all", log_freq=500)

    rollout = RolloutBuffer(obs_dim, cfg.rollout_steps, device)

    global_step = 0
    update_idx = 0
    best_eval_success = -1.0
    start_time = time.time()

    # Curriculum state
    current_distance = probe_dist
    level_start_update = 1
    recent_solved: List[float] = []

    # Storage for environment to maintain across steps
    env = RubiksCubeEnv(scramble_distance=current_distance if cfg.curriculum else cfg.scramble_distance,
                        max_steps=cfg.max_steps, step_penalty=cfg.step_penalty, seed=cfg.seed)
    obs, _ = env.reset()
    ep_steps = 0
    ep_solved_flag = 0.0

    for update_idx in range(1, cfg.total_updates + 1):
        # potentially update env scramble distance at episode boundaries via curriculum
        if cfg.curriculum:
            # distance is assigned when a new episode starts; handled when resetting
            pass

        rollout.clear()
        ep_in_update = 0
        solved_in_update = 0
        sum_return_in_update = 0.0
        sum_len_in_update = 0

        # Collect rollout
        while not rollout.full():
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            with torch.no_grad():
                logits, value = ac(obs_t)
                # Immediate inverse-move masking based on previously taken action in the current episode
                # We approximate last action from the last rollout entry if there is one for this episode.
                # As episodes can span across collection, we rely on env-level safety as well.
                if rollout.ptr > 0 and ep_steps > 0:
                    prev_action = int(rollout.actions[rollout.ptr - 1].item())
                    inv = {0:1,1:0,2:3,3:2,4:5,5:4,6:7,7:6,8:9,9:8,10:11,11:10}.get(prev_action, None)
                else:
                    inv = None
                logits_eff = logits.clone()
                if inv is not None:
                    logits_eff[0, inv] = -float("inf")
                probs = F.softmax(logits_eff, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
                logp = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            # add to buffer
            rollout.add(
                obs=torch.from_numpy(obs).to(device),
                action=int(action.item()),
                logp=float(logp.item()),
                reward=float(reward),
                done=float(done),
                value=float(value.squeeze().item()),
            )

            global_step += 1
            ep_steps += 1
            sum_return_in_update += reward
            sum_len_in_update += 1

            if done or ep_steps >= cfg.max_episode_len:
                if terminated:
                    solved_in_update += 1
                    ep_solved_flag = 1.0
                else:
                    ep_solved_flag = 0.0

                recent_solved.append(ep_solved_flag)
                if len(recent_solved) > cfg.cur_window_episodes:
                    recent_solved.pop(0)

                # reset new episode, possibly with updated curriculum distance
                dist_for_episode = current_distance if cfg.curriculum else cfg.scramble_distance
                env = RubiksCubeEnv(scramble_distance=dist_for_episode, max_steps=cfg.max_steps, step_penalty=cfg.step_penalty, seed=cfg.seed + update_idx + ep_in_update)
                obs, _ = env.reset()
                ep_steps = 0
                ep_in_update += 1
            else:
                obs = next_obs

        # Bootstrap value for GAE
        with torch.no_grad():
            last_obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)
            _, last_value = ac(last_obs_t)

        advantages, returns = rollout.compute_gae(last_value.squeeze(), cfg.gamma, cfg.gae_lambda)
        obs_batch, act_batch, logp_old_batch, _, _, val_batch = rollout.get()

        if cfg.normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std().clamp_min(1e-8)
            advantages = (advantages - adv_mean) / adv_std

        # PPO update
        batch_size = rollout.ptr
        inds = np.arange(batch_size)
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0

        for epoch in range(cfg.ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = min(start + cfg.minibatch_size, batch_size)
                mb_inds = inds[start:end]

                mb_obs = obs_batch[mb_inds]
                mb_act = act_batch[mb_inds]
                mb_logp_old = logp_old_batch[mb_inds]
                mb_adv = advantages[mb_inds]
                mb_ret = returns[mb_inds]
                mb_val_old = val_batch[mb_inds]

                logits, value = ac(mb_obs)
                logp = ac.get_logp(logits, mb_act)
                ratio = torch.exp(logp - mb_logp_old)

                # Policy loss with clipping
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
                pg_loss = -torch.min(unclipped, clipped).mean()

                # Value loss with optional clipping
                if cfg.vf_clip_coef > 0:
                    v_clipped = mb_val_old + torch.clamp(value - mb_val_old, -cfg.vf_clip_coef, cfg.vf_clip_coef)
                    v_loss_unclipped = (value - mb_ret).pow(2)
                    v_loss_clipped = (v_clipped - mb_ret).pow(2)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * (value - mb_ret).pow(2).mean()

                # Entropy bonus
                probs = F.softmax(logits, dim=-1)
                logp_all = F.log_softmax(logits, dim=-1)
                entropy = -(probs * logp_all).sum(dim=1).mean()

                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_norm=cfg.grad_clip)
                optimizer.step()
                scheduler.step()

                total_pg_loss += float(pg_loss.item())
                total_v_loss += float(v_loss.item())
                total_entropy += float(entropy.item())

        denom_updates = max(1, (cfg.ppo_epochs * math.ceil(batch_size / max(1, cfg.minibatch_size))))
        avg_pg_loss = total_pg_loss / denom_updates
        avg_v_loss = total_v_loss / denom_updates
        avg_entropy = total_entropy / denom_updates

        # Logging
        if WANDB_AVAILABLE and cfg.wandb_mode != "disabled":
            log_data = {
                "train/avg_pg_loss": avg_pg_loss,
                "train/avg_value_loss": avg_v_loss,
                "train/avg_entropy": avg_entropy,
                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                "rollout/episodes_in_update": ep_in_update,
                "rollout/solved_in_update": solved_in_update,
                "time/elapsed_min": (time.time() - start_time) / 60.0,
                "progress/update": update_idx,
                "progress/global_steps": global_step,
            }
            if getattr(cfg, "curriculum", False):
                log_data.update({
                    "curriculum/current_distance": current_distance if cfg.curriculum else cfg.scramble_distance,
                    "curriculum/level_start_update": level_start_update,
                    "curriculum/recent_window": len(recent_solved),
                    "curriculum/recent_success": float(np.mean(recent_solved)) if recent_solved else 0.0,
                })
            wandb.log(log_data, step=global_step)

        # Evaluation
        if update_idx % cfg.eval_every_updates == 0:
            eval_metrics = evaluate_agent(
                ac, device, distances=cfg.eval_distances,
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
                    flat["curriculum/current_distance"] = current_distance if cfg.curriculum else cfg.scramble_distance
                    flat["curriculum/recent_success"] = float(np.mean(recent_solved)) if recent_solved else 0.0
                    flat["curriculum/level_start_update"] = level_start_update
                wandb.log(flat, step=global_step)

            sr5 = eval_metrics.get(5, {}).get("success_rate", 0.0)
            best_eval_success = max(best_eval_success, sr5)

        # Curriculum progression check based on recent_solved window, every update
        if getattr(cfg, "curriculum", False):
            updates_on_level = update_idx - level_start_update + 1
            window_full = len(recent_solved) >= cfg.cur_window_episodes
            recent_success = float(np.mean(recent_solved)) if recent_solved else 0.0
            can_increase = (
                updates_on_level * 1.0 >= max(1, cfg.cur_min_episodes_per_level // max(1, (cfg.rollout_steps // cfg.max_episode_len)))
                and window_full
                and recent_success >= cfg.cur_threshold
                and current_distance < cfg.cur_max_distance
            )
            if can_increase:
                current_distance = min(cfg.cur_max_distance, current_distance + cfg.cur_increase_step)
                level_start_update = update_idx + 1
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

    hidden_layers = _parse_hidden_layers(os.getenv("HIDDEN_LAYERS", "1024,1024,1024"))
    eval_distances_str = os.getenv("EVAL_DISTANCES", "5,10,15,20")
    try:
        eval_distances = tuple(int(x.strip()) for x in eval_distances_str.split(",") if x.strip())
    except Exception:
        eval_distances = (5, 10, 15, 20)

    cfg = Config(
        scramble_distance=int(_get("SCRAMBLE_DISTANCE", 5)),
        max_steps=int(_get("MAX_STEPS", 40)),
        step_penalty=float(_get("STEP_PENALTY", 0.0)),
        seed=int(_get("SEED", 42)),
        curriculum=str(os.getenv("CURRICULUM", "true")).lower() in ("1", "true", "yes", "y"),
        cur_start_distance=int(os.getenv("CUR_START_DISTANCE", 1)),
        cur_max_distance=int(os.getenv("CUR_MAX_DISTANCE", 20)),
        cur_threshold=float(os.getenv("CUR_THRESHOLD", 0.8)),
        cur_window_episodes=int(os.getenv("CUR_WINDOW_EPISODES", 500)),
        cur_min_episodes_per_level=int(os.getenv("CUR_MIN_EPISODES_PER_LEVEL", 2000)),
        cur_increase_step=int(os.getenv("CUR_INCREASE_STEP", 1)),
        gamma=float(_get("GAMMA", 0.995)),
        gae_lambda=float(_get("GAE_LAMBDA", 0.95)),
        lr=float(_get("LR", 3e-4)),
        weight_decay=float(_get("WEIGHT_DECAY", 1e-6)),
        clip_coef=float(_get("CLIP_COEF", 0.2)),
        vf_clip_coef=float(_get("VF_CLIP_COEF", 0.2)),
        entropy_coef=float(_get("ENTROPY_COEF", 0.003)),
        value_coef=float(_get("VALUE_COEF", 0.5)),
        grad_clip=float(_get("GRAD_CLIP", 1.0)),
        normalize_advantages=str(os.getenv("NORMALIZE_ADVANTAGES", "true")).lower() in ("1", "true", "yes", "y"),
        hidden_layers=hidden_layers if len(hidden_layers) > 0 else (1024, 1024, 1024),
        activation=str(_get("ACTIVATION", "silu")),
        norm=str(_get("NORM", "layernorm")),
        dropout=float(_get("DROPOUT", 0.05)),
        rollout_steps=int(_get("ROLLOUT_STEPS", 2048)),
        ppo_epochs=int(_get("PPO_EPOCHS", 10)),
        minibatch_size=int(_get("MINIBATCH_SIZE", 256)),
        max_episode_len=int(_get("MAX_EPISODE_LEN", 100)),
        eval_every_updates=int(_get("EVAL_EVERY_UPDATES", 20)),
        eval_episodes_per_distance=int(_get("EVAL_EPISODES_PER_DISTANCE", 30)),
        eval_distances=eval_distances,
        total_updates=int(_get("TOTAL_UPDATES", 4000)),
        wandb_project=str(_get("WANDB_PROJECT", "RubiksRL")),
        wandb_entity=os.getenv("WANDB_ENTITY", None),
        wandb_mode=str(_get("WANDB_MODE", "online")),
        run_name=os.getenv("RUN_NAME", None),
        device=str(_get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")),
    )
    return cfg


def main():
    cfg = build_cfg_from_env()
    train(cfg)


if __name__ == "__main__":
    main()