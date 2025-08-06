import os
import math
import time
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

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

from envs.rubiks_env import RubiksCubeEnv, cube_to_obs
from utils.puzzle_cube import PuzzleCube, valid_moves


# We mirror the REINFORCE MLPPolicy so checkpoints are compatible
class MLPPolicy(nn.Module):  # train_supervised.py:line
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Tuple[int, ...] = (512, 512),
        activation: str = "relu",
        norm: str = "none",
        dropout: float = 0.0,
    ):
        super().__init__()
        act_layer = nn.ReLU if activation.lower() == "relu" else nn.SiLU
        use_layernorm = norm.lower() == "layernorm"
        layers = []
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

        # Kaiming init
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
        return logits


# Build inverse action mapping from index order of valid_moves
# valid_moves: ["L","L'","R","R'","U","U'","D","D'","F","F'","B","B'"]
def _build_inverse_idx():  # train_supervised.py:line
    inv = {}
    name_to_idx = {m: i for i, m in enumerate(valid_moves)}
    pairs = [("L", "L'"), ("R", "R'"), ("U", "U'"), ("D", "D'"), ("F", "F'"), ("B", "B'")]
    for a, b in pairs:
        ia, ib = name_to_idx[a], name_to_idx[b]
        inv[ia] = ib
        inv[ib] = ia
    return inv


INVERSE_IDX = _build_inverse_idx()


@dataclass
class Config:  # train_supervised.py:line
    # Data generation
    num_samples: int = 200_000
    scramble_min: int = 1
    scramble_max: int = 10
    # Training
    batch_size: int = 1024
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    activation: str = "relu"        # "relu" or "silu"
    norm: str = "none"              # "none" or "layernorm"
    dropout: float = 0.0
    hidden_layers: Tuple[int, ...] = (512, 512)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Logging / checkpoints
    wandb_project: str = "RubiksRL"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # "offline" or "disabled"
    run_name: Optional[str] = "supervised_pretrain"
    save_path: str = "checkpoints/supervised_mlp.pt"
    val_ratio: float = 0.05
    amp: bool = True


def set_seed(seed: int):  # train_supervised.py:line
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_dataset(cfg: Config, obs_dim: int, n_actions: int):  # train_supervised.py:line
    """
    Generate supervised pairs (state, next_action) by:
      1) Sampling scramble length L in [scramble_min, scramble_max]
      2) Start from solved, apply a random scramble sequence a_1..a_L
      3) At each intermediate state s_t (after applying a_1..a_t), label y_t = inverse(a_t)
         which is the next move to undo the last scramble action immediately.
      4) Return all (s_t, y_t) pairs aggregated across all generated scrambles until we reach num_samples.
    """
    rng = np.random.default_rng(cfg.seed)
    states = np.zeros((cfg.num_samples, obs_dim), dtype=np.float32)
    labels = np.zeros((cfg.num_samples,), dtype=np.int64)

    filled = 0
    while filled < cfg.num_samples:
        L = int(rng.integers(cfg.scramble_min, cfg.scramble_max + 1))
        # Build scramble sequence indices
        scramble_idxs = [int(rng.integers(n_actions)) for _ in range(L)]

        cube = PuzzleCube()
        # Apply scramble step-by-step; after each step, record state and label inverse of that step
        for t, a_idx in enumerate(scramble_idxs, start=1):
            cube = cube.move(valid_moves[a_idx])
            if filled >= cfg.num_samples:
                break
            obs = cube_to_obs(cube)  # 324-dim float32
            states[filled] = obs
            labels[filled] = INVERSE_IDX[a_idx]
            filled += 1

    return states, labels


def train_supervised(cfg: Config):  # train_supervised.py:line
    set_seed(cfg.seed)
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)

    # Probe env to get dims
    probe_env = RubiksCubeEnv(scramble_distance=1, max_steps=10, seed=cfg.seed)
    obs, _ = probe_env.reset()
    obs_dim = obs.shape[0]
    n_actions = probe_env.action_space.n

    states, labels = generate_dataset(cfg, obs_dim, n_actions)

    # Shuffle and split
    idx = np.arange(len(states))
    np.random.shuffle(idx)
    states = states[idx]
    labels = labels[idx]

    val_size = int(len(states) * cfg.val_ratio)
    train_states = states[val_size:]
    train_labels = labels[val_size:]
    val_states = states[:val_size]
    val_labels = labels[:val_size]

    # Tensors
    x_train = torch.from_numpy(train_states)
    y_train = torch.from_numpy(train_labels)
    x_val = torch.from_numpy(val_states)
    y_val = torch.from_numpy(val_labels)

    device = torch.device(cfg.device)
    model = MLPPolicy(
        in_dim=obs_dim,
        out_dim=n_actions,
        hidden=cfg.hidden_layers,
        activation=cfg.activation,
        norm=cfg.norm,
        dropout=cfg.dropout,
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    if WANDB_AVAILABLE and cfg.wandb_mode != "disabled":
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=asdict(cfg),
            name=cfg.run_name,
            mode=cfg.wandb_mode,
        )
        wandb.watch(model, log="all", log_freq=100)

    def iterate_minibatches(X: torch.Tensor, Y: torch.Tensor, batch_size: int):
        n = X.shape[0]
        for i in range(0, n, batch_size):
            yield X[i:i + batch_size], Y[i:i + batch_size]

    best_val_acc = -1.0
    start = time.time()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0

        # Shuffle each epoch
        perm = torch.randperm(x_train.shape[0])
        x_train_sh = x_train[perm]
        y_train_sh = y_train[perm]

        for xb_cpu, yb_cpu in iterate_minibatches(x_train_sh, y_train_sh, cfg.batch_size):
            xb = xb_cpu.to(device)
            yb = yb_cpu.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == yb).sum().item()
            train_count += xb.size(0)

        train_loss /= max(1, train_count)
        train_acc = train_correct / max(1, train_count)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for xb_cpu, yb_cpu in iterate_minibatches(x_val, y_val, cfg.batch_size):
                xb = xb_cpu.to(device)
                yb = yb_cpu.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == yb).sum().item()
        val_loss /= max(1, x_val.shape[0])
        val_acc = val_correct / max(1, x_val.shape[0])

        if WANDB_AVAILABLE and cfg.wandb_mode != "disabled":
            wandb.log(
                {
                    "supervised/train_loss": float(train_loss),
                    "supervised/train_acc": float(train_acc),
                    "supervised/val_loss": float(val_loss),
                    "supervised/val_acc": float(val_acc),
                    "supervised/epoch": ep,
                    "time/elapsed_min": (time.time() - start) / 60.0,
                },
                step=ep,
            )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "obs_dim": obs_dim,
                    "n_actions": n_actions,
                    "hidden": cfg.hidden_layers,
                    "activation": cfg.activation,
                    "norm": cfg.norm,
                    "dropout": cfg.dropout,
                },
                cfg.save_path,
            )

    if WANDB_AVAILABLE and cfg.wandb_mode != "disabled":
        wandb.summary["supervised_best_val_acc"] = best_val_acc
        wandb.finish()


def load_pretrained_into_reinforce_policy(policy: nn.Module, ckpt_path: str):  # train_supervised.py:line
    """
    Utility to load pretrained weights into the REINFORCE policy model.
    It assumes the REINFORCE model uses the same naming: .backbone and .pi
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    return missing, unexpected


def main():  # train_supervised.py:line
    # Allow configuration via environment variables (simple)
    cfg = Config(
        num_samples=int(os.getenv("SUP_NUM_SAMPLES", "200000")),
        scramble_min=int(os.getenv("SUP_SCRAMBLE_MIN", "1")),
        scramble_max=int(os.getenv("SUP_SCRAMBLE_MAX", "10")),
        batch_size=int(os.getenv("SUP_BATCH_SIZE", "1024")),
        epochs=int(os.getenv("SUP_EPOCHS", "10")),
        lr=float(os.getenv("SUP_LR", "3e-4")),
        weight_decay=float(os.getenv("SUP_WEIGHT_DECAY", "0.01")),
        grad_clip=float(os.getenv("SUP_GRAD_CLIP", "1.0")),
        activation=os.getenv("SUP_ACTIVATION", "relu"),
        norm=os.getenv("SUP_NORM", "none"),
        dropout=float(os.getenv("SUP_DROPOUT", "0.0")),
        hidden_layers=tuple(int(x.strip()) for x in os.getenv("SUP_HIDDEN", "512,512").split(",") if x.strip()),
        seed=int(os.getenv("SUP_SEED", "42")),
        device=os.getenv("SUP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        wandb_project=os.getenv("SUP_WANDB_PROJECT", "RubiksRL"),
        wandb_entity=os.getenv("SUP_WANDB_ENTITY", None),
        wandb_mode=os.getenv("SUP_WANDB_MODE", "online"),
        run_name=os.getenv("SUP_RUN_NAME", "supervised_pretrain"),
        save_path=os.getenv("SUP_SAVE_PATH", "checkpoints/supervised_mlp.pt"),
        val_ratio=float(os.getenv("SUP_VAL_RATIO", "0.05")),
        amp=(os.getenv("SUP_AMP", "1").lower() in ("1", "true", "yes", "y")),
    )
    train_supervised(cfg)


if __name__ == "__main__":  # train_supervised.py:line
    main()