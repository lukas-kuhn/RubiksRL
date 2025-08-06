import numpy as np
from typing import Tuple, Optional, Dict, Any, List

# Gymnasium-lite interface without dependency
class Space:
    def sample(self):
        raise NotImplementedError

    @property
    def shape(self):
        return None

class Discrete(Space):
    def __init__(self, n: int):
        self.n = n

    def sample(self) -> int:
        return int(np.random.randint(self.n))

    def contains(self, x: int) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n

class Box(Space):
    def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype=np.float32):
        self.low = low
        self.high = high
        self._shape = shape
        self.dtype = dtype

    @property
    def shape(self):
        return self._shape

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=self._shape).astype(self.dtype)


# Rubik environment using PuzzleCube
from utils.puzzle_cube import PuzzleCube, valid_moves
from utils.batch_cube import BatchCube, solved_cube_bit_array
# Inverse mapping for immediate inverse-move masking
INVERSE_ACTION = {
    0: 1, 1: 0,  # L <-> L'
    2: 3, 3: 2,  # R <-> R'
    4: 5, 5: 4,  # U <-> U'
    6: 7, 7: 6,  # D <-> D'
    8: 9, 9: 8,  # F <-> F'
    10: 11, 11: 10,  # B <-> B'
}


def cube_to_obs(cube: PuzzleCube) -> np.ndarray:
    # Convert cube to one-hot bit array: (54, 6) -> flatten to 324 vector
    bit = cube._inner_cube.bit_array()[0]  # (54, 6) bool
    return bit.astype(np.float32).reshape(-1)


class RubiksCubeEnv:
    """
    Minimal RL env wrapper over PuzzleCube.
    Observation: 324 float32 vector (one-hot per sticker).
    Action space: 12 discrete moves, ordered like valid_moves in utils.puzzle_cube.
    Reward scheme:
      +1 for solving on the step that reaches solved (and episode ends),
      0 otherwise (sparse).
    Optionally a small step penalty can be applied.
    """
    def __init__(
        self,
        scramble_distance: int = 5,
        max_steps: Optional[int] = None,
        step_penalty: float = 0.0,
        seed: Optional[int] = None,
        shaping_alpha: float = 0.1,  # alpha for Manhattan-distance shaping
        mask_inverse: bool = True,   # enable immediate inverse-move masking
    ):
        assert scramble_distance >= 0
        self.scramble_distance = int(scramble_distance)
        self.action_space = Discrete(len(valid_moves))
        self.observation_space = Box(low=0.0, high=1.0, shape=(54 * 6,), dtype=np.float32)
        self.step_penalty = float(step_penalty)
        self._rng = np.random.default_rng(seed)
        self._cube: Optional[PuzzleCube] = None
        # heuristic reasonable cap
        self.max_steps = max_steps if max_steps is not None else max(1, 2 * scramble_distance + 20)
        self._steps = 0
        # shaping and masking state
        self.shaping_alpha = float(shaping_alpha)
        self.mask_inverse = bool(mask_inverse)
        self._last_action: Optional[int] = None
        self._prev_manhattan: Optional[int] = None

    def seed(self, seed: Optional[int]):
        self._rng = np.random.default_rng(seed)

    def reset(self, *, scramble_distance: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        dist = self.scramble_distance if scramble_distance is None else int(scramble_distance)
        base = PuzzleCube()
        self._cube = base.scramble(dist)
        self._steps = 0
        self._last_action = None
        # initialize previous manhattan distance for shaping
        try:
            self._prev_manhattan = int(self._cube.manhattan_distance())
        except Exception:
            self._prev_manhattan = None
        obs = cube_to_obs(self._cube)
        info = {"scramble_distance": dist}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self._cube is not None, "Call reset() before step()"
        assert self.action_space.contains(action), f"Action out of range 0..{self.action_space.n-1}"
        # enforce immediate inverse masking at env-level (safety)
        if self.mask_inverse and self._last_action is not None:
            inv = INVERSE_ACTION.get(self._last_action, None)
            if inv is not None and action == inv:
                # if masked action chosen, fall back to a random legal non-inverse action
                candidates = [a for a in range(self.action_space.n) if a != inv]
                action = int(candidates[np.random.randint(len(candidates))])

        self._steps += 1
        move = valid_moves[action]
        self._cube = self._cube.move(move)

        solved = self._cube.is_solved()
        terminated = bool(solved)
        truncated = bool(self._steps >= self.max_steps and not terminated)

        # Manhattan-distance shaping: r = terminal_bonus + alpha * (d_{t-1} - d_t) - step_penalty
        reward = 0.0
        # compute current distance
        cur_dist = None
        try:
            cur_dist = int(self._cube.manhattan_distance())
        except Exception:
            cur_dist = None
        # shaping delta
        if self._prev_manhattan is not None and cur_dist is not None:
            reward += self.shaping_alpha * float(self._prev_manhattan - cur_dist)
        # terminal bonus
        if terminated:
            reward += 1.0
        # step penalty
        reward -= self.step_penalty
        # update prev distance and last action
        self._prev_manhattan = cur_dist if cur_dist is not None else self._prev_manhattan
        self._last_action = int(action)

        obs = cube_to_obs(self._cube)
        info: Dict[str, Any] = {
            "steps": self._steps,
            "solved": solved,
            "manhattan": cur_dist if cur_dist is not None else -1,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> str:
        assert self._cube is not None, "Call reset() first"
        return str(self._cube)

    def sample_random_action(self) -> int:
        return self.action_space.sample()


# Batched evaluator
def evaluate_policy(
    policy_fn,
    distances: List[int],
    episodes_per_distance: int = 25,
    max_steps_factor: float = 2.0,
    step_penalty: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """
    policy_fn: function(obs: np.ndarray) -> int (action index)
    Returns dictionary keyed by distance with metrics:
      success_rate, avg_steps_when_solved, avg_return
    """
    rng = np.random.default_rng(seed)
    results: Dict[int, Dict[str, float]] = {}
    for d in distances:
        successes = 0
        steps_solved: List[int] = []
        returns: List[float] = []
        for _ in range(episodes_per_distance):
            # Use high=1_000_000 for integers to satisfy typing (low defaults to 0)
            env_seed = int(rng.integers(1_000_000))
            env = RubiksCubeEnv(
                scramble_distance=d,
                max_steps=int(max_steps_factor * d + 20),
                step_penalty=step_penalty,
                seed=env_seed,
            )
            obs, _ = env.reset()
            ret = 0.0
            done = False
            while not done:
                a = policy_fn(obs)
                obs, r, term, trunc, _ = env.step(a)
                ret += r
                done = term or trunc
                if term:
                    successes += 1
                    steps_solved.append(env._steps)
            returns.append(ret)
        denom = max(1, episodes_per_distance)
        results[d] = {
            "success_rate": successes / denom,
            "avg_steps_when_solved": float(np.mean(steps_solved)) if steps_solved else float("nan"),
            "avg_return": float(np.mean(returns)) if returns else 0.0,
        }
    return results