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


SARSA_STATE_VERSION = "sarsa-state-v2-priority-robot-task-slots"
SARSA_STATE_BUCKET_COUNT = 49
DQN_MODEL_VERSION = "dqn-v2-action-reward-priority-slots"
DQN_TRAINING_CONTRACT = {
    "candidate_selection": "priority_waiting_feasibility_v7",
    "reward_attribution": "task_lifecycle_action_level",
    "return_gamma": 1.0,
    "replay_resume": "checkpoint_exact",
}


def _legal_actions(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return np.flatnonzero(mask)


@dataclass
class SarsaConfig:
    learning_rate: float = 0.1
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.97
    physical_epsilon_cap: float = 0.02

    def __post_init__(self):
        numeric = (
            self.learning_rate, self.gamma, self.epsilon_start,
            self.epsilon_end, self.epsilon_decay,
            self.physical_epsilon_cap,
        )
        if (any(isinstance(value, bool) or not isinstance(
                    value, (int, float)) or not np.isfinite(value)
                for value in numeric) or
                not 0 < self.learning_rate <= 1 or self.gamma != 1.0 or
                not 0 <= self.epsilon_end <= self.epsilon_start <= 1 or
                not 0 < self.epsilon_decay < 1 or
                not 0 <= self.physical_epsilon_cap <= self.epsilon_end):
            raise ValueError("invalid SARSA configuration")


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
        """Bucket global load plus the leading robot/task decision slots."""
        obs = np.asarray(observation, dtype=np.float32)
        # RL v7 fixed layout: 6 global + 8*8 robot + 20*9 task + masks.
        if obs.size < 278 or not np.isfinite(obs).all():
            raise ValueError("invalid observation")
        buckets = [
            int(np.clip(obs[1] * 5, 0, 5)),  # pending ratio
            int(np.clip(obs[2] * 5, 0, 5)),  # idle ratio
            int(np.clip(obs[3] * 4, 0, 4)),  # congestion
            int(np.clip(obs[4] * 10, 0, 10)),  # feasible pair density
            int(np.clip(obs[0] * 5, 0, 20)),  # time
        ]
        robot_start = 6
        task_start = robot_start + 8 * 8
        robot_mask_start = task_start + 20 * 9
        task_mask_start = robot_mask_start + 8
        for slot in range(4):
            offset = robot_start + slot * 8
            buckets.extend((
                int(obs[robot_mask_start + slot] > 0.5),
                int(np.clip((obs[offset] + 1.0) * 4, 0, 8)),
                int(np.clip((obs[offset + 1] + 1.0) * 4, 0, 8)),
                int(obs[offset + 2] > 0.5),
                int(np.clip(obs[offset + 3] * 5, 0, 5)),
            ))
        for slot in range(4):
            offset = task_start + slot * 9
            buckets.extend((
                int(obs[task_mask_start + slot] > 0.5),
                int(np.clip(round(obs[offset + 4] * 2), 0, 3)),
                int(np.clip(obs[offset + 5] * 2, 0, 20)),
                int(obs[offset + 6] > 0.5),
                int(np.clip(obs[offset + 7] * 4, 0, 4)),
                int(np.clip(obs[offset + 8] * 4, 0, 20)),
            ))
        result = tuple(buckets)
        if len(result) != SARSA_STATE_BUCKET_COUNT:
            raise RuntimeError("SARSA state contract length mismatch")
        return result

    def values(self, state: Tuple[int, ...]) -> np.ndarray:
        if (not isinstance(state, tuple) or
                len(state) != SARSA_STATE_BUCKET_COUNT or
                any(isinstance(value, bool) or not isinstance(
                    value, (int, np.integer)) for value in state)):
            raise ValueError("invalid SARSA discrete state")
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_dim, dtype=np.float32)
        return self.q_table[state]

    def select_action(self, state: Tuple[int, ...], action_mask: np.ndarray,
                      training: bool = True,
                      exploration_cap: Optional[float] = None) -> int:
        legal = _legal_actions(action_mask)
        if not legal.size:
            return self.no_op_action
        epsilon = self.epsilon
        if exploration_cap is not None:
            if (isinstance(exploration_cap, bool) or
                    not isinstance(exploration_cap, (int, float)) or
                    not np.isfinite(exploration_cap) or
                    not 0 <= exploration_cap <= 1):
                raise ValueError("exploration cap must lie in [0, 1]")
            epsilon = min(epsilon, float(exploration_cap))
        if training and self.rng.random() < epsilon:
            return int(self.rng.choice(legal))
        q = self.values(state)
        return int(legal[np.argmax(q[legal])])

    def update(self, state: Tuple[int, ...], action: int, reward: float,
               next_state: Tuple[int, ...], next_action: int,
               done: bool) -> float:
        if (isinstance(action, bool) or not isinstance(
                action, (int, np.integer)) or
                isinstance(next_action, bool) or not isinstance(
                    next_action, (int, np.integer)) or
                not 0 <= int(action) < self.action_dim or
                not 0 <= int(next_action) < self.action_dim or
                isinstance(reward, bool) or not isinstance(
                    reward, (int, float, np.integer, np.floating)) or
                not np.isfinite(reward) or not isinstance(done, bool)):
            raise ValueError("invalid SARSA update")
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
            "state_version": SARSA_STATE_VERSION,
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
        if payload.get("state_version") != SARSA_STATE_VERSION:
            raise ModelValidationError("SARSA state version mismatch")
        if payload.get("action_dim") != self.action_dim:
            raise ModelValidationError("SARSA action dimension mismatch")
        if payload.get("no_op_action") != self.no_op_action:
            raise ModelValidationError("SARSA no-op action mismatch")
        try:
            raw_config = payload.get("config")
            if not isinstance(raw_config, dict):
                raise ValueError("config is missing")
            config = SarsaConfig(**raw_config)
            epsilon = float(payload["epsilon"])
            episode = int(payload["episode"])
            if (not np.isfinite(epsilon) or
                    not config.epsilon_end <= epsilon <=
                    config.epsilon_start or episode < 0):
                raise ValueError("epsilon or episode is invalid")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelValidationError(
                f"invalid SARSA training state: {exc}") from exc
        self.config = config
        self.epsilon = epsilon
        self.episode = episode
        self.q_table = {}
        for key, value in payload.get("q_table", {}).items():
            array = np.asarray(value, dtype=np.float32)
            state = tuple(map(int, key.split("|")))
            if (len(state) != SARSA_STATE_BUCKET_COUNT or
                    array.shape != (self.action_dim,) or
                    not np.isfinite(array).all()):
                raise ModelValidationError("invalid SARSA Q table")
            self.q_table[state] = array


@dataclass
class DQNConfig:
    hidden_size: int = 64
    learning_rate: float = 5e-4
    gamma: float = 1.0
    batch_size: int = 64
    replay_capacity: int = 20000
    warmup_steps: int = 256
    target_update_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000
    physical_epsilon_cap: float = 0.02
    max_grad_norm: float = 5.0
    double_dqn: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    def __post_init__(self):
        integers = (
            self.hidden_size, self.batch_size, self.replay_capacity,
            self.warmup_steps, self.target_update_interval,
            self.epsilon_decay_steps,
        )
        numeric = (
            self.learning_rate, self.gamma, self.epsilon_start,
            self.epsilon_end, self.physical_epsilon_cap,
            self.max_grad_norm, self.adam_beta1, self.adam_beta2,
            self.adam_epsilon,
        )
        if (any(isinstance(value, bool) or not isinstance(
                    value, (int, np.integer)) or int(value) <= 0
                for value in integers) or
                any(isinstance(value, bool) or not isinstance(
                    value, (int, float, np.integer, np.floating)) or
                    not np.isfinite(value) for value in numeric) or
                not isinstance(self.double_dqn, bool) or
                not 0 < self.learning_rate <= 1 or self.gamma != 1.0 or
                self.replay_capacity < self.batch_size or
                self.warmup_steps < self.batch_size or
                not 0 <= self.epsilon_end <= self.epsilon_start <= 1 or
                not 0 <= self.physical_epsilon_cap <= self.epsilon_end or
                self.max_grad_norm <= 0 or
                not 0 <= self.adam_beta1 < 1 or
                not 0 <= self.adam_beta2 < 1 or self.adam_epsilon <= 0):
            raise ValueError("invalid DQN configuration")


def dqn_config_from_checkpoint(path: str) -> DQNConfig:
    """Read architecture/training metadata before constructing a DQNAgent."""
    try:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ModelValidationError(f"DQN checkpoint unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelValidationError("DQN checkpoint payload must be an object")
    if payload.get("algorithm") != "DQN":
        raise ModelValidationError("checkpoint algorithm is not DQN")
    if payload.get("model_version") != DQN_MODEL_VERSION:
        raise ModelValidationError("DQN model version mismatch")
    if payload.get("environment_version") != ENVIRONMENT_VERSION:
        raise ModelValidationError("DQN environment version mismatch")
    if payload.get("training_contract") != DQN_TRAINING_CONTRACT:
        raise ModelValidationError("DQN training contract mismatch")
    raw = payload.get("config")
    expected = set(DQNConfig.__dataclass_fields__)
    try:
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("DQN config fields are incomplete or unknown")
        return DQNConfig(**raw)
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
        if (isinstance(state_dim, bool) or not isinstance(
                state_dim, (int, np.integer)) or int(state_dim) <= 0 or
                isinstance(action_dim, bool) or not isinstance(
                    action_dim, (int, np.integer)) or int(action_dim) <= 0 or
                isinstance(no_op_action, bool) or not isinstance(
                    no_op_action, (int, np.integer)) or
                not 0 <= int(no_op_action) < int(action_dim)):
            raise ValueError("invalid DQN dimensions")
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
                      training: bool = True,
                      exploration_cap: Optional[float] = None) -> int:
        legal = _legal_actions(action_mask)
        if not legal.size:
            return self.no_op_action
        epsilon = self.epsilon
        if exploration_cap is not None:
            if (isinstance(exploration_cap, bool) or
                    not isinstance(exploration_cap, (int, float)) or
                    not np.isfinite(exploration_cap) or
                    not 0 <= exploration_cap <= 1):
                raise ValueError("exploration cap must lie in [0, 1]")
            epsilon = min(epsilon, float(exploration_cap))
        if training and self.rng.random() < epsilon:
            return int(self.rng.choice(legal))
        q = self.online.forward(state)
        if not np.isfinite(q).all():
            raise ValueError("non-finite DQN output")
        return int(legal[np.argmax(q[legal])])

    def remember(self, state, action, reward, next_state, done,
                 next_action_mask) -> None:
        state = np.asarray(state, np.float32)
        next_state = np.asarray(next_state, np.float32)
        next_action_mask = np.asarray(next_action_mask, bool)
        if (state.shape != (self.state_dim,) or
                next_state.shape != (self.state_dim,) or
                not np.isfinite(state).all() or
                not np.isfinite(next_state).all() or
                isinstance(action, bool) or not isinstance(
                    action, (int, np.integer)) or
                not 0 <= int(action) < self.action_dim or
                isinstance(reward, bool) or not isinstance(
                    reward, (int, float, np.integer, np.floating)) or
                not np.isfinite(reward) or not isinstance(done, bool) or
                next_action_mask.shape != (self.action_dim,) or
                (not done and not next_action_mask.any())):
            raise ValueError("invalid DQN replay transition")
        self.replay.add(
            state, int(action), float(reward), next_state, done,
            next_action_mask)

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
            "algorithm": "DQN", "model_version": DQN_MODEL_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "training_contract": dict(DQN_TRAINING_CONTRACT),
            "state_dim": self.state_dim, "action_dim": self.action_dim,
            "no_op_action": self.no_op_action, "config": asdict(self.config),
            "training_step": self.training_step, "episode": self.episode,
            "epsilon": self.epsilon, "seed": self.seed,
            "online": self.online.params, "target": self.target.params,
            "optimizer_m": self.optimizer_m,
            "optimizer_v": self.optimizer_v,
            "agent_rng_state": self.rng.bit_generator.state,
            "replay_rng_state": self.replay.rng.bit_generator.state,
            "replay": list(self.replay.data),
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
        if not isinstance(payload, dict):
            raise ModelValidationError(
                "DQN checkpoint payload must be an object")
        expected = (
            "DQN", DQN_MODEL_VERSION, ENVIRONMENT_VERSION, self.state_dim,
            self.action_dim, self.no_op_action, DQN_TRAINING_CONTRACT,
            asdict(self.config))
        actual = (
            payload.get("algorithm"), payload.get("model_version"),
            payload.get("environment_version"), payload.get("state_dim"),
            payload.get("action_dim"), payload.get("no_op_action"),
            payload.get("training_contract"), payload.get("config"))
        if actual != expected:
            raise ModelValidationError("DQN checkpoint metadata mismatch")
        for network_name, network in (("online", self.online),
                                      ("target", self.target)):
            params = payload.get(network_name, {})
            if not isinstance(params, dict) or set(params) != set(network.params):
                raise ModelValidationError(
                    f"invalid DQN parameter set {network_name}")
            for key, target in network.params.items():
                value = np.asarray(params.get(key), np.float32)
                if value.shape != target.shape or not np.isfinite(value).all():
                    raise ModelValidationError(
                        f"invalid DQN parameter {network_name}.{key}")
                target[...] = value
        try:
            raw_training_step = payload["training_step"]
            raw_episode = payload["episode"]
            raw_seed = payload["seed"]
            if (isinstance(raw_training_step, bool) or
                    not isinstance(raw_training_step, (int, np.integer)) or
                    isinstance(raw_episode, bool) or
                    not isinstance(raw_episode, (int, np.integer)) or
                    isinstance(raw_seed, bool) or
                    not isinstance(raw_seed, (int, np.integer))):
                raise ValueError("DQN counters and seed must be integers")
            training_step = int(raw_training_step)
            episode = int(raw_episode)
            seed = int(raw_seed)
            epsilon = float(payload["epsilon"])
            if (training_step < 0 or episode < 0 or seed < 0 or
                    not np.isfinite(epsilon) or
                    not self.config.epsilon_end <= epsilon <=
                    self.config.epsilon_start):
                raise ValueError("invalid DQN training counters")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ModelValidationError(
                f"invalid DQN training state: {exc}") from exc
        self.training_step = training_step
        self.episode = episode
        self.epsilon = epsilon
        self.seed = seed
        for state_name, target_state in (
                ("optimizer_m", self.optimizer_m),
                ("optimizer_v", self.optimizer_v)):
            state = payload.get(state_name, {})
            if (not isinstance(state, dict) or
                    set(state) != set(target_state)):
                raise ModelValidationError(
                    f"invalid DQN optimizer state set {state_name}")
            for key, target in target_state.items():
                value = np.asarray(state.get(key), np.float32)
                if value.shape != target.shape or not np.isfinite(value).all():
                    raise ModelValidationError(
                        f"invalid DQN optimizer state {state_name}.{key}")
                target[...] = value
        replay = payload.get("replay")
        if (not isinstance(replay, list) or
                len(replay) > self.config.replay_capacity):
            raise ModelValidationError("invalid DQN replay state")
        self.replay = ReplayBuffer(self.config.replay_capacity, self.seed)
        try:
            for transition in replay:
                if not isinstance(transition, tuple) or len(transition) != 6:
                    raise ValueError("malformed replay row")
                self.remember(*transition)
            self.rng.bit_generator.state = payload["agent_rng_state"]
            self.replay.rng.bit_generator.state = payload["replay_rng_state"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelValidationError(
                f"invalid DQN stochastic resume state: {exc}") from exc
