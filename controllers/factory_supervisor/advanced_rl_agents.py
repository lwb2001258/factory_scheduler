"""NumPy implementations of advanced discrete RL scheduling agents.

The agents share the versioned 161-action environment but are deliberately
independent algorithms:

* RainbowDQN uses a dueling C51 network, Double-DQN action selection,
  prioritised replay and n-step returns.
* QRDQN learns fixed return quantiles with quantile-Huber regression.
* CQL learns from a fixed transition data set with a discrete conservative
  log-sum-exp regulariser.

None of these classes owns path planning or robot control.
"""

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from advanced_ai_common import (
    load_checkpoint, masked_argmax, save_checkpoint, validate_action_mask,
)
from config import RL_ENVIRONMENT_VERSION
from schedulers import ModelValidationError


def _config_from_metadata(metadata: dict, cls):
    raw = metadata.get("config")
    if not isinstance(raw, dict):
        raise ModelValidationError("checkpoint config is missing")
    try:
        return cls(**raw)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(
            f"invalid {metadata.get('algorithm')} config: {exc}") from exc


def _check_probability(name: str, value: float, *, closed: bool = True):
    lower = 0.0 <= value if closed else 0.0 < value
    if not lower or value > 1.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be in {'[' if closed else '('}0, 1]")


def _check_integer(name: str, value, *, minimum: int = 1) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or
            int(value) < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _check_finite(name: str, value, *, positive: bool = False,
                  nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if (not math.isfinite(number) or
            (positive and number <= 0) or
            (nonnegative and number < 0)):
        raise ValueError(f"{name} has an invalid numeric value")


def _validate_agent_dimensions(state_dim: int, action_dim: int,
                               no_op_action: int) -> None:
    if (isinstance(state_dim, bool) or isinstance(action_dim, bool) or
            int(state_dim) != state_dim or int(action_dim) != action_dim or
            int(state_dim) <= 0 or int(action_dim) <= 0 or
            isinstance(no_op_action, bool) or
            int(no_op_action) != no_op_action or
            not 0 <= int(no_op_action) < int(action_dim)):
        raise ValueError("invalid agent dimensions or no-op action")


def _restore_training_step(metadata: dict, optimizer: "Adam") -> int:
    value = metadata.get("training_step", 0)
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or
            int(value) < 0):
        raise ModelValidationError("checkpoint training step is invalid")
    optimizer.step_count = int(value)
    return int(value)


def _network_arrays(prefix: str, params: Dict[str, np.ndarray]) -> dict:
    return {f"{prefix}_{name}": value for name, value in params.items()}


def _restore_params(arrays: dict, prefix: str,
                    params: Dict[str, np.ndarray]) -> None:
    for name, target in params.items():
        key = f"{prefix}_{name}"
        value = np.asarray(arrays.get(key), dtype=np.float32)
        if value.shape != target.shape or not np.all(np.isfinite(value)):
            raise ModelValidationError(f"invalid checkpoint parameter {key}")
        target[...] = value


def _global_norm(grads: Dict[str, np.ndarray]) -> float:
    return math.sqrt(sum(float(np.sum(value * value))
                         for value in grads.values()))


class Adam:
    def __init__(self, params: Dict[str, np.ndarray], learning_rate: float,
                 max_grad_norm: float = 10.0):
        if learning_rate <= 0 or max_grad_norm <= 0:
            raise ValueError("Adam rates and gradient norm must be positive")
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = float(max_grad_norm)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.step_count = 0
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}

    def step(self, params: Dict[str, np.ndarray],
             grads: Dict[str, np.ndarray]) -> None:
        norm = _global_norm(grads)
        scale = min(1.0, self.max_grad_norm / max(norm, 1e-12))
        self.step_count += 1
        for name, parameter in params.items():
            gradient = np.asarray(grads[name], np.float32) * scale
            self.m[name] = self.beta1 * self.m[name] + (1-self.beta1)*gradient
            self.v[name] = self.beta2 * self.v[name] + (1-self.beta2)*gradient**2
            m_hat = self.m[name] / (1-self.beta1**self.step_count)
            v_hat = self.v[name] / (1-self.beta2**self.step_count)
            parameter -= self.learning_rate * m_hat / (
                np.sqrt(v_hat) + self.epsilon)

    def arrays(self, prefix: str) -> dict:
        result = _network_arrays(f"{prefix}_m", self.m)
        result.update(_network_arrays(f"{prefix}_v", self.v))
        return result

    def restore(self, arrays: dict, prefix: str) -> None:
        _restore_params(arrays, f"{prefix}_m", self.m)
        _restore_params(arrays, f"{prefix}_v", self.v)


@dataclass(frozen=True)
class ReplayTransition:
    state: np.ndarray
    action_mask: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_action_mask: np.ndarray
    done: bool
    discount: float


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, *, alpha: float = 0.6,
                 priority_epsilon: float = 1e-5, seed: int = 42):
        if capacity <= 0 or alpha < 0 or priority_epsilon <= 0:
            raise ValueError("invalid prioritised replay configuration")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self.rng = np.random.default_rng(seed)
        self.data: List[ReplayTransition] = []
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        self.cursor = 0

    def __len__(self):
        return len(self.data)

    def add(self, transition: ReplayTransition,
            priority: Optional[float] = None) -> None:
        if priority is None:
            priority = (float(np.max(self.priorities[:len(self.data)]))
                        if self.data else 1.0)
        if not math.isfinite(priority) or priority <= 0:
            raise ValueError("replay priority must be finite and positive")
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.cursor] = transition
        self.priorities[self.cursor] = max(
            float(priority), self.priority_epsilon)
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        if batch_size <= 0 or batch_size > len(self.data):
            raise ValueError("invalid replay batch size")
        _check_probability("beta", beta)
        scaled = self.priorities[:len(self.data)] ** self.alpha
        total = float(np.sum(scaled))
        if not math.isfinite(total) or total <= 0:
            raise ValueError("replay priorities cannot be normalised")
        probabilities = scaled / total
        indices = self.rng.choice(
            len(self.data), size=batch_size, replace=False, p=probabilities)
        weights = (len(self.data) * probabilities[indices]) ** (-beta)
        weights /= max(float(np.max(weights)), 1e-12)
        rows = [self.data[int(index)] for index in indices]
        return (
            np.stack([row.state for row in rows]),
            np.stack([row.action_mask for row in rows]),
            np.asarray([row.action for row in rows], dtype=np.int64),
            np.asarray([row.reward for row in rows], dtype=np.float32),
            np.stack([row.next_state for row in rows]),
            np.stack([row.next_action_mask for row in rows]),
            np.asarray([row.done for row in rows], dtype=np.float32),
            np.asarray([row.discount for row in rows], dtype=np.float32),
            np.asarray(indices, dtype=np.int64),
            np.asarray(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, priorities) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float64)
        if indices.shape != priorities.shape:
            raise ValueError("priority indices and values must align")
        if (np.any(indices < 0) or np.any(indices >= len(self.data)) or
                not np.all(np.isfinite(priorities)) or
                np.any(priorities <= 0)):
            raise ValueError("invalid replay priority update")
        self.priorities[indices] = np.maximum(
            priorities, self.priority_epsilon)


class NStepAccumulator:
    def __init__(self, n_step: int, gamma: float):
        if n_step <= 0:
            raise ValueError("n_step must be positive")
        _check_probability("gamma", gamma)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.queue = deque()

    def _aggregate(self) -> ReplayTransition:
        reward = 0.0
        count = 0
        last = None
        for item in list(self.queue)[:self.n_step]:
            reward += (self.gamma ** count) * item.reward
            count += 1
            last = item
            if item.done:
                break
        first = self.queue[0]
        return ReplayTransition(
            first.state, first.action_mask, first.action, reward,
            last.next_state, last.next_action_mask, last.done,
            self.gamma ** count)

    def add(self, state, action_mask, action: int, reward: float,
            next_state, next_action_mask, done: bool
            ) -> List[ReplayTransition]:
        state = np.asarray(state, dtype=np.float32).copy()
        next_state = np.asarray(next_state, dtype=np.float32).copy()
        mask = validate_action_mask(action_mask).copy()
        next_mask = validate_action_mask(next_action_mask).copy()
        if (state.ndim != 1 or next_state.shape != state.shape or
                not np.all(np.isfinite(state)) or
                not np.all(np.isfinite(next_state))):
            raise ValueError("transition states are invalid")
        if not 0 <= int(action) < mask.size or not mask[int(action)]:
            raise ValueError("transition action is not legal")
        if not math.isfinite(reward):
            raise ValueError("transition reward must be finite")
        self.queue.append(ReplayTransition(
            state, mask, int(action), float(reward), next_state, next_mask,
            bool(done), self.gamma))
        emitted = []
        if done:
            while self.queue:
                emitted.append(self._aggregate())
                self.queue.popleft()
        elif len(self.queue) >= self.n_step:
            emitted.append(self._aggregate())
            self.queue.popleft()
        return emitted


class DuelingC51Network:
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int,
                 atoms: int, rng: np.random.Generator):
        scale1 = math.sqrt(2.0 / max(1, state_dim))
        scale2 = math.sqrt(2.0 / max(1, hidden_size))
        self.action_dim = int(action_dim)
        self.atoms = int(atoms)
        self.params = {
            "w1": rng.normal(0, scale1, (state_dim, hidden_size)).astype(np.float32),
            "b1": np.zeros(hidden_size, np.float32),
            "wv": rng.normal(0, scale2, (hidden_size, atoms)).astype(np.float32),
            "bv": np.zeros(atoms, np.float32),
            "wa": rng.normal(
                0, scale2, (hidden_size, action_dim * atoms)).astype(np.float32),
            "ba": np.zeros(action_dim * atoms, np.float32),
        }

    def copy_from(self, other) -> None:
        for name in self.params:
            self.params[name][...] = other.params[name]

    def forward(self, states, *, cache: bool = False):
        x = np.asarray(states, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        z1 = x @ self.params["w1"] + self.params["b1"]
        hidden = np.maximum(z1, 0.0)
        value = hidden @ self.params["wv"] + self.params["bv"]
        advantage = (
            hidden @ self.params["wa"] + self.params["ba"]
        ).reshape(len(x), self.action_dim, self.atoms)
        logits = value[:, None, :] + advantage - advantage.mean(
            axis=1, keepdims=True)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("C51 network produced non-finite probabilities")
        output = probabilities[0] if single else probabilities
        if not cache:
            return output
        return output, (x, z1, hidden, probabilities)

    def backward(self, cache, grad_logits):
        x, z1, hidden, _ = cache
        gradient = np.asarray(grad_logits, np.float32)
        value_gradient = gradient.sum(axis=1)
        advantage_gradient = gradient - gradient.mean(axis=1, keepdims=True)
        flat_advantage = advantage_gradient.reshape(len(x), -1)
        grads = {
            "wv": hidden.T @ value_gradient,
            "bv": value_gradient.sum(axis=0),
            "wa": hidden.T @ flat_advantage,
            "ba": flat_advantage.sum(axis=0),
        }
        hidden_gradient = (
            value_gradient @ self.params["wv"].T
            + flat_advantage @ self.params["wa"].T)
        z_gradient = hidden_gradient * (z1 > 0)
        grads["w1"] = x.T @ z_gradient
        grads["b1"] = z_gradient.sum(axis=0)
        return grads


class QuantileNetwork:
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int,
                 quantiles: int, rng: np.random.Generator):
        scale1 = math.sqrt(2.0 / max(1, state_dim))
        scale2 = math.sqrt(2.0 / max(1, hidden_size))
        self.action_dim = int(action_dim)
        self.quantiles = int(quantiles)
        self.params = {
            "w1": rng.normal(0, scale1, (state_dim, hidden_size)).astype(np.float32),
            "b1": np.zeros(hidden_size, np.float32),
            "wq": rng.normal(
                0, scale2, (hidden_size, action_dim * quantiles)).astype(np.float32),
            "bq": np.zeros(action_dim * quantiles, np.float32),
        }

    def copy_from(self, other):
        for name in self.params:
            self.params[name][...] = other.params[name]

    def forward(self, states, *, cache=False):
        x = np.asarray(states, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        z1 = x @ self.params["w1"] + self.params["b1"]
        hidden = np.maximum(z1, 0.0)
        quantiles = (
            hidden @ self.params["wq"] + self.params["bq"]
        ).reshape(len(x), self.action_dim, self.quantiles)
        if not np.all(np.isfinite(quantiles)):
            raise ValueError("quantile network produced non-finite values")
        output = quantiles[0] if single else quantiles
        if not cache:
            return output
        return output, (x, z1, hidden)

    def backward(self, cache, grad_output):
        x, z1, hidden = cache
        flat = np.asarray(grad_output, np.float32).reshape(len(x), -1)
        grads = {
            "wq": hidden.T @ flat,
            "bq": flat.sum(axis=0),
        }
        hidden_gradient = flat @ self.params["wq"].T
        z_gradient = hidden_gradient * (z1 > 0)
        grads["w1"] = x.T @ z_gradient
        grads["b1"] = z_gradient.sum(axis=0)
        return grads


class DenseQNetwork:
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int,
                 rng: np.random.Generator):
        scale1 = math.sqrt(2.0 / max(1, state_dim))
        scale2 = math.sqrt(2.0 / max(1, hidden_size))
        self.params = {
            "w1": rng.normal(0, scale1, (state_dim, hidden_size)).astype(np.float32),
            "b1": np.zeros(hidden_size, np.float32),
            "w2": rng.normal(0, scale2, (hidden_size, action_dim)).astype(np.float32),
            "b2": np.zeros(action_dim, np.float32),
        }

    def copy_from(self, other):
        for name in self.params:
            self.params[name][...] = other.params[name]

    def forward(self, states, *, cache=False):
        x = np.asarray(states, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        z1 = x @ self.params["w1"] + self.params["b1"]
        hidden = np.maximum(z1, 0.0)
        values = hidden @ self.params["w2"] + self.params["b2"]
        if not np.all(np.isfinite(values)):
            raise ValueError("Q network produced non-finite values")
        output = values[0] if single else values
        if not cache:
            return output
        return output, (x, z1, hidden)

    def backward(self, cache, grad_output):
        x, z1, hidden = cache
        gradient = np.asarray(grad_output, np.float32)
        grads = {
            "w2": hidden.T @ gradient,
            "b2": gradient.sum(axis=0),
        }
        hidden_gradient = gradient @ self.params["w2"].T
        z_gradient = hidden_gradient * (z1 > 0)
        grads["w1"] = x.T @ z_gradient
        grads["b1"] = z_gradient.sum(axis=0)
        return grads


@dataclass(frozen=True)
class RainbowConfig:
    hidden_size: int = 64
    atoms: int = 51
    value_min: float = -100.0
    value_max: float = 100.0
    gamma: float = 0.99
    n_step: int = 3
    replay_capacity: int = 20000
    batch_size: int = 64
    warmup_steps: int = 256
    learning_rate: float = 0.0005
    target_update_interval: int = 250
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    epsilon: float = 0.05

    def __post_init__(self):
        for name, value, minimum in (
                ("hidden_size", self.hidden_size, 1),
                ("atoms", self.atoms, 2),
                ("n_step", self.n_step, 1),
                ("replay_capacity", self.replay_capacity, 1),
                ("batch_size", self.batch_size, 1),
                ("warmup_steps", self.warmup_steps, 0),
                ("target_update_interval", self.target_update_interval, 1)):
            _check_integer(name, value, minimum=minimum)
        for name, value in (
                ("value_min", self.value_min),
                ("value_max", self.value_max)):
            _check_finite(name, value)
        _check_finite("learning_rate", self.learning_rate, positive=True)
        _check_finite("priority_alpha", self.priority_alpha,
                      nonnegative=True)
        if self.value_max <= self.value_min:
            raise ValueError("invalid Rainbow configuration")
        _check_probability("gamma", self.gamma)
        _check_probability("priority_beta", self.priority_beta)
        _check_probability("epsilon", self.epsilon)


class RainbowDQNAgent:
    ALGORITHM = "RainbowDQN"

    def __init__(self, state_dim: int, action_dim: int, no_op_action: int,
                 config: Optional[RainbowConfig] = None, seed: int = 42):
        _validate_agent_dimensions(state_dim, action_dim, no_op_action)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.no_op_action = int(no_op_action)
        self.config = config or RainbowConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.support = np.linspace(
            self.config.value_min, self.config.value_max,
            self.config.atoms, dtype=np.float32)
        self.online = DuelingC51Network(
            state_dim, action_dim, self.config.hidden_size,
            self.config.atoms, self.rng)
        self.target = DuelingC51Network(
            state_dim, action_dim, self.config.hidden_size,
            self.config.atoms, self.rng)
        self.target.copy_from(self.online)
        self.optimizer = Adam(self.online.params, self.config.learning_rate)
        self.replay = PrioritizedReplayBuffer(
            self.config.replay_capacity, alpha=self.config.priority_alpha,
            seed=seed)
        self.n_step = NStepAccumulator(
            self.config.n_step, self.config.gamma)
        self.training_step = 0

    def _expected(self, probabilities):
        return np.sum(probabilities * self.support, axis=-1)

    def action_values(self, state) -> np.ndarray:
        return np.asarray(
            self._expected(self.online.forward(state)), dtype=np.float64)

    def select_action(self, state, action_mask, *, training=False) -> int:
        mask = validate_action_mask(action_mask, self.action_dim)
        legal = np.flatnonzero(mask)
        if training and self.rng.random() < self.config.epsilon:
            return int(self.rng.choice(legal))
        return masked_argmax(self.action_values(state), mask)

    def remember(self, state, action_mask, action, reward,
                 next_state, next_action_mask, done) -> int:
        emitted = self.n_step.add(
            state, action_mask, action, reward,
            next_state, next_action_mask, done)
        for transition in emitted:
            self.replay.add(transition)
        return len(emitted)

    def _project_distribution(self, rewards, dones, discounts,
                              next_probabilities):
        batch = len(rewards)
        target = np.zeros((batch, self.config.atoms), dtype=np.float32)
        transformed = rewards[:, None] + (
            (1.0-dones) * discounts)[:, None] * self.support[None, :]
        transformed = np.clip(
            transformed, self.config.value_min, self.config.value_max)
        delta = ((self.config.value_max-self.config.value_min) /
                 (self.config.atoms-1))
        positions = (transformed-self.config.value_min) / delta
        lower = np.floor(positions).astype(np.int64)
        upper = np.ceil(positions).astype(np.int64)
        for row in range(batch):
            for atom in range(self.config.atoms):
                probability = next_probabilities[row, atom]
                lo, hi = lower[row, atom], upper[row, atom]
                if lo == hi:
                    target[row, lo] += probability
                else:
                    target[row, lo] += probability * (hi-positions[row, atom])
                    target[row, hi] += probability * (positions[row, atom]-lo)
        return target

    def train_step(self) -> Optional[float]:
        minimum = max(self.config.batch_size, self.config.warmup_steps)
        if len(self.replay) < minimum:
            return None
        batch = self.replay.sample(
            self.config.batch_size, self.config.priority_beta)
        (states, masks, actions, rewards, next_states, next_masks, dones,
         discounts, indices, importance) = batch
        online_next = self._expected(self.online.forward(next_states))
        target_next = self.target.forward(next_states)
        next_actions = np.empty(len(states), dtype=np.int64)
        for row in range(len(states)):
            next_actions[row] = masked_argmax(
                online_next[row], next_masks[row])
        next_probabilities = target_next[
            np.arange(len(states)), next_actions]
        target_distribution = self._project_distribution(
            rewards, dones, discounts, next_probabilities)
        probabilities, cache = self.online.forward(states, cache=True)
        chosen = probabilities[np.arange(len(states)), actions]
        loss_rows = -np.sum(
            target_distribution * np.log(np.maximum(chosen, 1e-8)), axis=1)
        loss = float(np.mean(importance * loss_rows))
        grad_logits = np.zeros_like(probabilities, dtype=np.float32)
        grad_logits[np.arange(len(states)), actions] = (
            (chosen-target_distribution)
            * (importance / len(states))[:, None])
        grads = self.online.backward(cache, grad_logits)
        self.optimizer.step(self.online.params, grads)
        predicted = self._expected(chosen)
        expected_target = self._expected(target_distribution)
        self.replay.update_priorities(
            indices, np.abs(predicted-expected_target) + 1e-5)
        self.training_step += 1
        if self.training_step % self.config.target_update_interval == 0:
            self.target.copy_from(self.online)
        if not math.isfinite(loss):
            raise ValueError("Rainbow loss is non-finite")
        return loss

    def save(self, path) -> None:
        metadata = {
            "algorithm": self.ALGORITHM,
            "environment_version": RL_ENVIRONMENT_VERSION,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "no_op_action": self.no_op_action,
            "config": asdict(self.config),
            "training_step": self.training_step,
            "seed": self.seed,
        }
        arrays = _network_arrays("online", self.online.params)
        arrays.update(_network_arrays("target", self.target.params))
        arrays.update(self.optimizer.arrays("adam"))
        save_checkpoint(path, metadata, arrays)

    @classmethod
    def load(cls, path, state_dim: int, action_dim: int,
             no_op_action: int, seed: int = 42):
        metadata, arrays = load_checkpoint(
            path, expected_algorithm=cls.ALGORITHM,
            expected_state_dim=state_dim, expected_action_dim=action_dim)
        config = _config_from_metadata(metadata, RainbowConfig)
        agent = cls(state_dim, action_dim, no_op_action, config, seed)
        if int(metadata.get("no_op_action", -1)) != int(no_op_action):
            raise ModelValidationError("checkpoint no-op action mismatch")
        _restore_params(arrays, "online", agent.online.params)
        _restore_params(arrays, "target", agent.target.params)
        agent.optimizer.restore(arrays, "adam")
        agent.training_step = _restore_training_step(
            metadata, agent.optimizer)
        return agent


@dataclass(frozen=True)
class QRDQNConfig:
    hidden_size: int = 64
    quantiles: int = 32
    risk_fraction: float = 1.0
    huber_kappa: float = 1.0
    gamma: float = 0.99
    n_step: int = 3
    replay_capacity: int = 20000
    batch_size: int = 64
    warmup_steps: int = 256
    learning_rate: float = 0.0005
    target_update_interval: int = 250
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    epsilon: float = 0.05

    def __post_init__(self):
        for name, value, minimum in (
                ("hidden_size", self.hidden_size, 1),
                ("quantiles", self.quantiles, 2),
                ("n_step", self.n_step, 1),
                ("replay_capacity", self.replay_capacity, 1),
                ("batch_size", self.batch_size, 1),
                ("warmup_steps", self.warmup_steps, 0),
                ("target_update_interval", self.target_update_interval, 1)):
            _check_integer(name, value, minimum=minimum)
        _check_finite("huber_kappa", self.huber_kappa, positive=True)
        _check_finite("learning_rate", self.learning_rate, positive=True)
        _check_finite("priority_alpha", self.priority_alpha,
                      nonnegative=True)
        _check_probability("risk_fraction", self.risk_fraction, closed=False)
        _check_probability("gamma", self.gamma)
        _check_probability("priority_beta", self.priority_beta)
        _check_probability("epsilon", self.epsilon)


class QRDQNAgent:
    ALGORITHM = "QRDQN"

    def __init__(self, state_dim: int, action_dim: int, no_op_action: int,
                 config: Optional[QRDQNConfig] = None, seed: int = 42):
        _validate_agent_dimensions(state_dim, action_dim, no_op_action)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.no_op_action = int(no_op_action)
        self.config = config or QRDQNConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.online = QuantileNetwork(
            state_dim, action_dim, self.config.hidden_size,
            self.config.quantiles, self.rng)
        self.target = QuantileNetwork(
            state_dim, action_dim, self.config.hidden_size,
            self.config.quantiles, self.rng)
        self.target.copy_from(self.online)
        self.optimizer = Adam(self.online.params, self.config.learning_rate)
        self.replay = PrioritizedReplayBuffer(
            self.config.replay_capacity, alpha=self.config.priority_alpha,
            seed=seed)
        self.n_step = NStepAccumulator(self.config.n_step, self.config.gamma)
        self.training_step = 0
        self.taus = ((np.arange(self.config.quantiles, dtype=np.float32)+0.5)
                     / self.config.quantiles)

    def action_values(self, state):
        quantiles = self.online.forward(state)
        count = max(1, int(math.ceil(
            self.config.quantiles * self.config.risk_fraction)))
        return np.asarray(np.mean(quantiles[..., :count], axis=-1),
                          dtype=np.float64)

    def select_action(self, state, action_mask, *, training=False):
        mask = validate_action_mask(action_mask, self.action_dim)
        legal = np.flatnonzero(mask)
        if training and self.rng.random() < self.config.epsilon:
            return int(self.rng.choice(legal))
        return masked_argmax(self.action_values(state), mask)

    def remember(self, state, action_mask, action, reward,
                 next_state, next_action_mask, done):
        emitted = self.n_step.add(
            state, action_mask, action, reward,
            next_state, next_action_mask, done)
        for transition in emitted:
            self.replay.add(transition)
        return len(emitted)

    def train_step(self) -> Optional[float]:
        minimum = max(self.config.batch_size, self.config.warmup_steps)
        if len(self.replay) < minimum:
            return None
        (states, masks, actions, rewards, next_states, next_masks, dones,
         discounts, indices, importance) = self.replay.sample(
             self.config.batch_size, self.config.priority_beta)
        online_next = self.online.forward(next_states).mean(axis=-1)
        target_next = self.target.forward(next_states)
        next_actions = np.asarray([
            masked_argmax(online_next[row], next_masks[row])
            for row in range(len(states))], dtype=np.int64)
        target_quantiles = target_next[
            np.arange(len(states)), next_actions]
        target_values = rewards[:, None] + (
            (1.0-dones)*discounts)[:, None] * target_quantiles
        quantiles, cache = self.online.forward(states, cache=True)
        chosen = quantiles[np.arange(len(states)), actions]
        delta = target_values[:, None, :] - chosen[:, :, None]
        absolute = np.abs(delta)
        kappa = self.config.huber_kappa
        huber = np.where(
            absolute <= kappa, 0.5*delta**2,
            kappa*(absolute-0.5*kappa))
        quantile_weights = np.abs(
            self.taus[None, :, None] - (delta < 0).astype(np.float32))
        row_loss = np.mean(
            quantile_weights*huber/kappa, axis=(1, 2))
        loss = float(np.mean(importance*row_loss))
        huber_gradient = np.clip(delta, -kappa, kappa)
        chosen_gradient = -np.mean(
            quantile_weights*huber_gradient/kappa, axis=2)
        chosen_gradient /= self.config.quantiles
        chosen_gradient *= (importance/len(states))[:, None]
        grad_output = np.zeros_like(quantiles, dtype=np.float32)
        grad_output[np.arange(len(states)), actions] = chosen_gradient
        self.optimizer.step(
            self.online.params,
            self.online.backward(cache, grad_output))
        priorities = np.abs(
            chosen.mean(axis=1)-target_values.mean(axis=1)) + 1e-5
        self.replay.update_priorities(indices, priorities)
        self.training_step += 1
        if self.training_step % self.config.target_update_interval == 0:
            self.target.copy_from(self.online)
        if not math.isfinite(loss):
            raise ValueError("QRDQN loss is non-finite")
        return loss

    def save(self, path):
        metadata = {
            "algorithm": self.ALGORITHM,
            "environment_version": RL_ENVIRONMENT_VERSION,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "no_op_action": self.no_op_action,
            "config": asdict(self.config),
            "training_step": self.training_step,
            "seed": self.seed,
        }
        arrays = _network_arrays("online", self.online.params)
        arrays.update(_network_arrays("target", self.target.params))
        arrays.update(self.optimizer.arrays("adam"))
        save_checkpoint(path, metadata, arrays)

    @classmethod
    def load(cls, path, state_dim, action_dim, no_op_action, seed=42):
        metadata, arrays = load_checkpoint(
            path, expected_algorithm=cls.ALGORITHM,
            expected_state_dim=state_dim, expected_action_dim=action_dim)
        config = _config_from_metadata(metadata, QRDQNConfig)
        agent = cls(state_dim, action_dim, no_op_action, config, seed)
        if int(metadata.get("no_op_action", -1)) != int(no_op_action):
            raise ModelValidationError("checkpoint no-op action mismatch")
        _restore_params(arrays, "online", agent.online.params)
        _restore_params(arrays, "target", agent.target.params)
        agent.optimizer.restore(arrays, "adam")
        agent.training_step = _restore_training_step(
            metadata, agent.optimizer)
        return agent


@dataclass(frozen=True)
class CQLConfig:
    hidden_size: int = 64
    gamma: float = 0.99
    conservative_weight: float = 1.0
    replay_capacity: int = 50000
    batch_size: int = 64
    warmup_steps: int = 64
    learning_rate: float = 0.0005
    target_update_interval: int = 250
    priority_alpha: float = 0.0
    priority_beta: float = 0.0

    def __post_init__(self):
        for name, value, minimum in (
                ("hidden_size", self.hidden_size, 1),
                ("replay_capacity", self.replay_capacity, 1),
                ("batch_size", self.batch_size, 1),
                ("warmup_steps", self.warmup_steps, 0),
                ("target_update_interval", self.target_update_interval, 1)):
            _check_integer(name, value, minimum=minimum)
        _check_finite("conservative_weight", self.conservative_weight,
                      nonnegative=True)
        _check_finite("learning_rate", self.learning_rate, positive=True)
        _check_finite("priority_alpha", self.priority_alpha,
                      nonnegative=True)
        _check_probability("gamma", self.gamma)
        _check_probability("priority_beta", self.priority_beta)


class CQLAgent:
    ALGORITHM = "CQL"

    def __init__(self, state_dim: int, action_dim: int, no_op_action: int,
                 config: Optional[CQLConfig] = None, seed: int = 42):
        _validate_agent_dimensions(state_dim, action_dim, no_op_action)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.no_op_action = int(no_op_action)
        self.config = config or CQLConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.online = DenseQNetwork(
            state_dim, action_dim, self.config.hidden_size, self.rng)
        self.target = DenseQNetwork(
            state_dim, action_dim, self.config.hidden_size, self.rng)
        self.target.copy_from(self.online)
        self.optimizer = Adam(self.online.params, self.config.learning_rate)
        self.replay = PrioritizedReplayBuffer(
            self.config.replay_capacity, alpha=self.config.priority_alpha,
            seed=seed)
        self.training_step = 0

    def action_values(self, state):
        return np.asarray(self.online.forward(state), dtype=np.float64)

    def select_action(self, state, action_mask):
        return masked_argmax(self.action_values(state), action_mask)

    def remember(self, state, action_mask, action, reward,
                 next_state, next_action_mask, done):
        state = np.asarray(state, np.float32)
        next_state = np.asarray(next_state, np.float32)
        mask = validate_action_mask(action_mask, self.action_dim).copy()
        next_mask = validate_action_mask(
            next_action_mask, self.action_dim).copy()
        if (state.shape != (self.state_dim,) or
                next_state.shape != (self.state_dim,) or
                not np.all(np.isfinite(state)) or
                not np.all(np.isfinite(next_state)) or
                not math.isfinite(reward)):
            raise ValueError("invalid CQL transition")
        if not 0 <= int(action) < self.action_dim or not mask[int(action)]:
            raise ValueError("CQL behaviour action is not legal")
        self.replay.add(ReplayTransition(
            state.copy(), mask, int(action), float(reward),
            next_state.copy(), next_mask, bool(done), self.config.gamma))

    def train_step(self) -> Optional[float]:
        minimum = max(self.config.batch_size, self.config.warmup_steps)
        if len(self.replay) < minimum:
            return None
        (states, masks, actions, rewards, next_states, next_masks, dones,
         discounts, indices, importance) = self.replay.sample(
             self.config.batch_size, self.config.priority_beta)
        online_next = self.online.forward(next_states)
        target_next = self.target.forward(next_states)
        next_actions = np.asarray([
            masked_argmax(online_next[row], next_masks[row])
            for row in range(len(states))], dtype=np.int64)
        targets = rewards + (1.0-dones)*discounts*target_next[
            np.arange(len(states)), next_actions]
        values, cache = self.online.forward(states, cache=True)
        chosen = values[np.arange(len(states)), actions]
        td_error = chosen-targets
        absolute = np.abs(td_error)
        td_loss_rows = np.where(
            absolute <= 1.0, 0.5*td_error**2, absolute-0.5)
        gradient = np.zeros_like(values, dtype=np.float32)
        td_gradient = np.where(
            absolute <= 1.0, td_error, np.sign(td_error))
        gradient[np.arange(len(states)), actions] = (
            importance*td_gradient/len(states))
        conservative_rows = np.zeros(len(states), dtype=np.float32)
        for row in range(len(states)):
            legal = np.flatnonzero(masks[row])
            legal_values = values[row, legal]
            maximum = float(np.max(legal_values))
            weights = np.exp(legal_values-maximum)
            weights /= np.sum(weights)
            logsumexp = maximum + math.log(float(np.sum(
                np.exp(legal_values-maximum))))
            conservative_rows[row] = logsumexp-chosen[row]
            gradient[row, legal] += (
                self.config.conservative_weight
                * importance[row] * weights / len(states))
            gradient[row, actions[row]] -= (
                self.config.conservative_weight
                * importance[row] / len(states))
        loss = float(np.mean(importance*(
            td_loss_rows + self.config.conservative_weight*conservative_rows)))
        self.optimizer.step(
            self.online.params, self.online.backward(cache, gradient))
        self.replay.update_priorities(
            indices, np.abs(td_error)+1e-5)
        self.training_step += 1
        if self.training_step % self.config.target_update_interval == 0:
            self.target.copy_from(self.online)
        if not math.isfinite(loss):
            raise ValueError("CQL loss is non-finite")
        return loss

    def save(self, path):
        metadata = {
            "algorithm": self.ALGORITHM,
            "environment_version": RL_ENVIRONMENT_VERSION,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "no_op_action": self.no_op_action,
            "config": asdict(self.config),
            "training_step": self.training_step,
            "seed": self.seed,
        }
        arrays = _network_arrays("online", self.online.params)
        arrays.update(_network_arrays("target", self.target.params))
        arrays.update(self.optimizer.arrays("adam"))
        save_checkpoint(path, metadata, arrays)

    @classmethod
    def load(cls, path, state_dim, action_dim, no_op_action, seed=42):
        metadata, arrays = load_checkpoint(
            path, expected_algorithm=cls.ALGORITHM,
            expected_state_dim=state_dim, expected_action_dim=action_dim)
        config = _config_from_metadata(metadata, CQLConfig)
        agent = cls(state_dim, action_dim, no_op_action, config, seed)
        if int(metadata.get("no_op_action", -1)) != int(no_op_action):
            raise ModelValidationError("checkpoint no-op action mismatch")
        _restore_params(arrays, "online", agent.online.params)
        _restore_params(arrays, "target", agent.target.params)
        agent.optimizer.restore(arrays, "adam")
        agent.training_step = _restore_training_step(
            metadata, agent.optimizer)
        return agent


@dataclass(frozen=True)
class OfflineTransitionDataset:
    states: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    next_action_masks: np.ndarray
    dones: np.ndarray
    behavior_policy: str
    seeds: Tuple[int, ...]

    def __post_init__(self):
        states = np.asarray(self.states, dtype=np.float32)
        next_states = np.asarray(self.next_states, dtype=np.float32)
        raw_masks = np.asarray(self.action_masks)
        raw_next_masks = np.asarray(self.next_action_masks)
        raw_actions = np.asarray(self.actions)
        raw_dones = np.asarray(self.dones)
        if (raw_masks.dtype.kind not in "biuf" or
                raw_next_masks.dtype.kind not in "biuf" or
                raw_actions.dtype.kind not in "iu" or
                raw_dones.dtype.kind not in "biuf" or
                not np.all(np.isfinite(raw_masks)) or
                not np.all(np.isfinite(raw_next_masks)) or
                not np.all(np.isin(raw_masks, (0, 1))) or
                not np.all(np.isin(raw_next_masks, (0, 1))) or
                not np.all(np.isfinite(raw_dones)) or
                not np.all(np.isin(raw_dones, (0, 1)))):
            raise ValueError("offline masks, actions or dones are invalid")
        masks = raw_masks.astype(bool, copy=False)
        next_masks = raw_next_masks.astype(bool, copy=False)
        actions = raw_actions.astype(np.int64, copy=False)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = raw_dones.astype(bool, copy=False)
        raw_seeds = tuple(self.seeds)
        if (not raw_seeds or any(
                isinstance(seed, bool) or
                not isinstance(seed, (int, np.integer))
                for seed in raw_seeds)):
            raise ValueError("offline dataset seeds must be integers")
        seeds = tuple(int(seed) for seed in raw_seeds)
        count = len(states)
        if (states.ndim != 2 or next_states.shape != states.shape or
                masks.ndim != 2 or next_masks.shape != masks.shape or
                masks.shape[0] != count or actions.shape != (count,) or
                rewards.shape != (count,) or dones.shape != (count,) or
                not isinstance(self.behavior_policy, str) or
                not self.behavior_policy.strip() or count == 0 or
                not np.all(np.isfinite(states)) or
                not np.all(np.isfinite(next_states)) or
                not np.all(np.isfinite(rewards)) or
                np.any(~masks.any(axis=1)) or
                np.any(~next_masks.any(axis=1)) or
                np.any(actions < 0) or np.any(actions >= masks.shape[1]) or
                np.any(~masks[np.arange(count), actions])):
            raise ValueError("invalid offline transition dataset")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "next_states", next_states)
        object.__setattr__(self, "action_masks", masks)
        object.__setattr__(self, "next_action_masks", next_masks)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "dones", dones)
        object.__setattr__(self, "behavior_policy", self.behavior_policy.strip())
        object.__setattr__(self, "seeds", seeds)

    @property
    def state_dim(self):
        return self.states.shape[1]

    @property
    def action_dim(self):
        return self.action_masks.shape[1]

    def add_to(self, agent: CQLAgent) -> None:
        if (agent.state_dim != self.state_dim or
                agent.action_dim != self.action_dim):
            raise ValueError("dataset dimensions do not match CQL agent")
        for row in range(len(self.states)):
            agent.remember(
                self.states[row], self.action_masks[row], self.actions[row],
                self.rewards[row], self.next_states[row],
                self.next_action_masks[row], self.dones[row])

    def save(self, path) -> None:
        metadata = {
            "algorithm": "CQLDataset",
            "environment_version": RL_ENVIRONMENT_VERSION,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "behavior_policy": self.behavior_policy,
            "seeds": list(self.seeds),
        }
        save_checkpoint(path, metadata, {
            "states": self.states,
            "action_masks": self.action_masks,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "next_action_masks": self.next_action_masks,
            "dones": self.dones,
        })

    @classmethod
    def load(cls, path):
        metadata, arrays = load_checkpoint(
            path, expected_algorithm="CQLDataset")
        try:
            dataset = cls(
                arrays["states"], arrays["action_masks"], arrays["actions"],
                arrays["rewards"], arrays["next_states"],
                arrays["next_action_masks"], arrays["dones"],
                str(metadata["behavior_policy"]),
                tuple(metadata["seeds"]))
            if (dataset.state_dim != metadata["state_dim"] or
                    dataset.action_dim != metadata["action_dim"]):
                raise ValueError("dataset metadata dimensions do not match")
            return dataset
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelValidationError(
                f"invalid CQL dataset: {exc}") from exc
