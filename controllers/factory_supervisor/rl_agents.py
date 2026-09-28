"""Dependency-light SARSA(0) and Double-DQN agents for scheduling."""

import json
import pickle
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from rl_environment import ENVIRONMENT_VERSION
from schedulers import ModelValidationError


def _legal_actions(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return np.flatnonzero(mask)


@dataclass
class SarsaConfig:
    learning_rate: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995


class SarsaAgent:
    """Tabular on-policy SARSA(0), with fixed masked action space."""

    def __init__(self, action_dim: int, no_op_action: int,
                 config: Optional[SarsaConfig] = None, seed: int = 42):
        self.action_dim = action_dim
        self.no_op_action = no_op_action
        self.config = config or SarsaConfig()
        self.epsilon = self.config.epsilon_start
        self.episode = 0
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    @staticmethod
    def discretize(observation: np.ndarray) -> Tuple[int, ...]:
        """Compact, stable buckets derived from the global observation prefix."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.size < 6 or not np.isfinite(obs).all():
            raise ValueError("invalid observation")
        return (
            int(np.clip(obs[1] * 5, 0, 5)),  # pending ratio
            int(np.clip(obs[2] * 5, 0, 5)),  # idle ratio
            int(np.clip(obs[3] * 4, 0, 4)),  # congestion
            int(np.clip(obs[4] * 10, 0, 10)),  # feasible pair density
            int(np.clip(obs[0] * 5, 0, 20)),  # time
        )

    def values(self, state: Tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim, dtype=np.float32)
        return self.q_table[state]

    def select_action(self, state: Tuple[int, ...], action_mask: np.ndarray,
                      training: bool = True) -> int:
        legal = _legal_actions(action_mask)
        if not legal.size:
            return self.no_op_action
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))
        q = self.values(state)
        return int(legal[np.argmax(q[legal])])

    def update(self, state: Tuple[int, ...], action: int, reward: float,
               next_state: Tuple[int, ...], next_action: int,
               done: bool) -> float:
        q = self.values(state)
        target = float(reward)
        if not done:
            target += self.config.gamma * float(
                self.values(next_state)[next_action])
        td_error = target - float(q[action])
        q[action] += self.config.learning_rate * td_error
        return td_error

    def end_episode(self) -> None:
        self.episode += 1
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay)

    def save(self, path: str) -> None:
        payload = {
            "algorithm": "SARSA", "environment_version": ENVIRONMENT_VERSION,
            "action_dim": self.action_dim, "no_op_action": self.no_op_action,
            "epsilon": self.epsilon, "episode": self.episode,
            "config": asdict(self.config), "seed": self.seed,
            "q_table": {"|".join(map(str, key)): value.tolist()
                        for key, value in self.q_table.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: str) -> None:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelValidationError(f"SARSA checkpoint unreadable: {exc}") from exc
        if payload.get("algorithm") != "SARSA":
            raise ModelValidationError("checkpoint algorithm is not SARSA")
        if payload.get("environment_version") != ENVIRONMENT_VERSION:
            raise ModelValidationError("SARSA environment version mismatch")
        if payload.get("action_dim") != self.action_dim:
            raise ModelValidationError("SARSA action dimension mismatch")
        self.epsilon = float(payload["epsilon"])
        self.episode = int(payload["episode"])
        self.q_table = {}
        for key, value in payload.get("q_table", {}).items():
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (self.action_dim,) or not np.isfinite(array).all():
                raise ModelValidationError("invalid SARSA Q table")
            self.q_table[tuple(map(int, key.split("|")))] = array


@dataclass
class DQNConfig:
    hidden_size: int = 64
    learning_rate: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 20000
    warmup_steps: int = 256
    target_update_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000
    max_grad_norm: float = 5.0
    double_dqn: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


def dqn_config_from_checkpoint(path: str) -> DQNConfig:
    """Read architecture/training metadata before constructing a DQNAgent."""
    try:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ModelValidationError(f"DQN checkpoint unreadable: {exc}") from exc
    if payload.get("algorithm") != "DQN":
        raise ModelValidationError("checkpoint algorithm is not DQN")
    raw = payload.get("config", {})
    allowed = DQNConfig.__dataclass_fields__
    try:
        return DQNConfig(**{key: value for key, value in raw.items()
                           if key in allowed})
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(f"invalid DQN checkpoint config: {exc}") from exc


class QNetwork:
    """Two-layer ReLU MLP with explicit NumPy backpropagation."""

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 rng: np.random.Generator):
        self.input_dim, self.output_dim, self.hidden_size = (
            input_dim, output_dim, hidden_size)
        self.params = {
            "w1": rng.normal(0, np.sqrt(2 / input_dim),
                             (input_dim, hidden_size)).astype(np.float32),
            "b1": np.zeros(hidden_size, np.float32),
            "w2": rng.normal(0, np.sqrt(2 / hidden_size),
                             (hidden_size, hidden_size)).astype(np.float32),
            "b2": np.zeros(hidden_size, np.float32),
            "w3": rng.normal(0, np.sqrt(2 / hidden_size),
                             (hidden_size, output_dim)).astype(np.float32),
            "b3": np.zeros(output_dim, np.float32),
        }

    def forward(self, x: np.ndarray, cache: bool = False):
        x = np.asarray(x, dtype=np.float32)
        one = x.ndim == 1
        if one:
            x = x[None, :]
        z1 = x @ self.params["w1"] + self.params["b1"]
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.params["w2"] + self.params["b2"]
        h2 = np.maximum(z2, 0)
        out = h2 @ self.params["w3"] + self.params["b3"]
        if cache:
            return out, (x, z1, h1, z2, h2)
        return out[0] if one else out

    def copy_from(self, other: "QNetwork") -> None:
        for key in self.params:
            self.params[key][...] = other.params[key]

    def backward(self, cache, grad_out: np.ndarray):
        x, z1, h1, z2, h2 = cache
        grads = {}
        grads["w3"] = h2.T @ grad_out
        grads["b3"] = grad_out.sum(axis=0)
        gh2 = grad_out @ self.params["w3"].T
        gz2 = gh2 * (z2 > 0)
        grads["w2"] = h1.T @ gz2
        grads["b2"] = gz2.sum(axis=0)
        gh1 = gz2 @ self.params["w2"].T
        gz1 = gh1 * (z1 > 0)
        grads["w1"] = x.T @ gz1
        grads["b1"] = gz1.sum(axis=0)
        return grads


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42):
        self.data = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def add(self, state, action, reward, next_state, done, next_action_mask):
        self.data.append((
            np.asarray(state, np.float32).copy(), int(action), float(reward),
            np.asarray(next_state, np.float32).copy(), bool(done),
            np.asarray(next_action_mask, bool).copy()))

    def sample(self, size: int):
        indices = self.rng.choice(len(self.data), size=size, replace=False)
        rows = [self.data[int(index)] for index in indices]
        return (
            np.stack([r[0] for r in rows]), np.asarray([r[1] for r in rows]),
            np.asarray([r[2] for r in rows], np.float32),
            np.stack([r[3] for r in rows]),
            np.asarray([r[4] for r in rows], np.float32),
            np.stack([r[5] for r in rows]),
        )


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, no_op_action: int,
                 config: Optional[DQNConfig] = None, seed: int = 42):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.no_op_action = no_op_action
        self.config = config or DQNConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(
            state_dim, action_dim, self.config.hidden_size, self.rng)
        self.target = QNetwork(
            state_dim, action_dim, self.config.hidden_size, self.rng)
        self.target.copy_from(self.online)
        self.replay = ReplayBuffer(self.config.replay_capacity, seed)
        self.training_step = 0
        self.episode = 0
        self.epsilon = self.config.epsilon_start
        self.optimizer_m = {
            key: np.zeros_like(value) for key, value in self.online.params.items()}
        self.optimizer_v = {
            key: np.zeros_like(value) for key, value in self.online.params.items()}

    def select_action(self, state: np.ndarray, action_mask: np.ndarray,
                      training: bool = True) -> int:
        legal = _legal_actions(action_mask)
        if not legal.size:
            return self.no_op_action
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))
        q = self.online.forward(state)
        if not np.isfinite(q).all():
            raise ValueError("non-finite DQN output")
        return int(legal[np.argmax(q[legal])])

    def remember(self, *transition) -> None:
        self.replay.add(*transition)

    def _targets(self, rewards, next_states, dones, next_masks):
        online_q = self.online.forward(next_states)
        target_q = self.target.forward(next_states)
        future = np.zeros(len(rewards), np.float32)
        for index, mask in enumerate(next_masks):
            legal = _legal_actions(mask)
            if dones[index] or not legal.size:
                continue
            if self.config.double_dqn:
                action = legal[np.argmax(online_q[index, legal])]
                future[index] = target_q[index, action]
            else:
                future[index] = np.max(target_q[index, legal])
        targets = rewards + self.config.gamma * (1.0 - dones) * future
        if not np.isfinite(targets).all():
            raise ValueError("non-finite DQN target")
        return targets

    def train_step(self) -> Optional[float]:
        minimum = max(self.config.warmup_steps, self.config.batch_size)
        if len(self.replay) < minimum:
            return None
        states, actions, rewards, next_states, dones, masks = (
            self.replay.sample(self.config.batch_size))
        targets = self._targets(rewards, next_states, dones, masks)
        q, cache = self.online.forward(states, cache=True)
        chosen = q[np.arange(len(actions)), actions]
        error = chosen - targets
        abs_error = np.abs(error)
        loss = np.where(abs_error <= 1, 0.5 * error ** 2,
                        abs_error - 0.5).mean()
        grad_chosen = np.where(abs_error <= 1, error, np.sign(error))
        grad_out = np.zeros_like(q)
        grad_out[np.arange(len(actions)), actions] = grad_chosen / len(actions)
        grads = self.online.backward(cache, grad_out)
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = min(1.0, self.config.max_grad_norm / max(norm, 1e-12))
        self.training_step += 1
        for key in self.online.params:
            grad = grads[key] * scale
            self.optimizer_m[key] = (
                self.config.adam_beta1 * self.optimizer_m[key]
                + (1 - self.config.adam_beta1) * grad)
            self.optimizer_v[key] = (
                self.config.adam_beta2 * self.optimizer_v[key]
                + (1 - self.config.adam_beta2) * grad * grad)
            m_hat = self.optimizer_m[key] / (
                1 - self.config.adam_beta1 ** self.training_step)
            v_hat = self.optimizer_v[key] / (
                1 - self.config.adam_beta2 ** self.training_step)
            self.online.params[key] -= self.config.learning_rate * m_hat / (
                np.sqrt(v_hat) + self.config.adam_epsilon)
        fraction = min(1.0, self.training_step /
                       max(1, self.config.epsilon_decay_steps))
        self.epsilon = (
            self.config.epsilon_start
            + fraction * (self.config.epsilon_end
                          - self.config.epsilon_start))
        if self.training_step % self.config.target_update_interval == 0:
            self.target.copy_from(self.online)
        return float(loss)

    def save(self, path: str) -> None:
        payload = {
            "algorithm": "DQN", "environment_version": ENVIRONMENT_VERSION,
            "state_dim": self.state_dim, "action_dim": self.action_dim,
            "no_op_action": self.no_op_action, "config": asdict(self.config),
            "training_step": self.training_step, "episode": self.episode,
            "epsilon": self.epsilon, "seed": self.seed,
            "online": self.online.params, "target": self.target.params,
            "optimizer_m": self.optimizer_m,
            "optimizer_v": self.optimizer_v,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        try:
            with Path(path).open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            raise ModelValidationError(f"DQN checkpoint unreadable: {exc}") from exc
        expected = ("DQN", ENVIRONMENT_VERSION, self.state_dim, self.action_dim)
        actual = (payload.get("algorithm"), payload.get("environment_version"),
                  payload.get("state_dim"), payload.get("action_dim"))
        if actual != expected:
            raise ModelValidationError("DQN checkpoint metadata mismatch")
        for network_name, network in (("online", self.online),
                                      ("target", self.target)):
            params = payload.get(network_name, {})
            for key, target in network.params.items():
                value = np.asarray(params.get(key), np.float32)
                if value.shape != target.shape or not np.isfinite(value).all():
                    raise ModelValidationError(
                        f"invalid DQN parameter {network_name}.{key}")
                target[...] = value
        self.training_step = int(payload["training_step"])
        self.episode = int(payload["episode"])
        self.epsilon = float(payload["epsilon"])
        for state_name, target_state in (
                ("optimizer_m", self.optimizer_m),
                ("optimizer_v", self.optimizer_v)):
            state = payload.get(state_name, {})
            for key, target in target_state.items():
                value = np.asarray(state.get(key), np.float32)
                if value.shape != target.shape or not np.isfinite(value).all():
                    raise ModelValidationError(
                        f"invalid DQN optimizer state {state_name}.{key}")
                target[...] = value
