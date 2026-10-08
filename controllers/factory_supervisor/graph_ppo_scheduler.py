"""Graph-PPO task assignment over variable feasible bipartite edges.

Physical fine tuning is opt-in.  Normal deployment remains deterministic and
selects the highest-probability edge from the same action space used during
training; a Webots collection run samples one edge and emits the complete
on-policy observation in scheduler diagnostics.  The supervisor persists that
observation only after the physical dispatch commits.
"""

import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np

from advanced_ai_common import load_checkpoint, masked_softmax, save_checkpoint
from advanced_rl_agents import Adam
from config import RL_ENVIRONMENT_VERSION
from learning_scheduler import (
    GRAPH_FEATURE_NAMES, graph_edge_feature_tensor, pair_feature_tensor,
)
from rl_environment import RLEnvironmentConfig, SchedulingEnvironment
from headless_training_runtime import summarize_runtime_telemetry
from schedulers import (
    BaseScheduler, CostMatrix, ModelValidationError, SchedulerResult,
    SchedulingContext, _result_from_matching, build_cost_matrix,
)
from training_scenarios import factory_scenario


GRAPH_PPO_ALGORITHM = "GraphPPO"
GRAPH_PPO_ACTION_SEMANTICS = "variable_bipartite_edge"


def _check_integer(name, value, minimum=1):
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or
            int(value) < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _check_probability(name, value, *, positive=False):
    if (isinstance(value, bool) or
            not isinstance(value, (int, float, np.integer, np.floating)) or
            not math.isfinite(float(value)) or
            (float(value) <= 0 if positive else float(value) < 0) or
            float(value) > 1):
        raise ValueError(f"{name} must be within the unit interval")


@dataclass(frozen=True)
class GraphPPOConfig:
    hidden_size: int = 48
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 0.0005
    update_epochs: int = 4
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 5.0
    value_huber_delta: float = 10.0

    def __post_init__(self):
        _check_integer("hidden_size", self.hidden_size)
        _check_integer("update_epochs", self.update_epochs)
        _check_probability("gamma", self.gamma)
        _check_probability("gae_lambda", self.gae_lambda)
        _check_probability("clip_epsilon", self.clip_epsilon, positive=True)
        for name, value, allow_zero in (
                ("learning_rate", self.learning_rate, False),
                ("value_coefficient", self.value_coefficient, True),
                ("entropy_coefficient", self.entropy_coefficient, True),
                ("max_grad_norm", self.max_grad_norm, False),
                ("value_huber_delta", self.value_huber_delta, False)):
            if (isinstance(value, bool) or
                    not isinstance(value, (int, float, np.integer, np.floating)) or
                    not math.isfinite(float(value)) or
                    (float(value) < 0 if allow_zero else float(value) <= 0)):
                raise ValueError(f"{name} is invalid")


def _huber_loss_and_gradient(error: float, delta: float):
    """Return scalar Huber loss and its bounded derivative."""
    error = float(error)
    delta = float(delta)
    if not math.isfinite(error) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("Huber inputs must be finite and delta positive")
    magnitude = abs(error)
    if magnitude <= delta:
        return 0.5 * error * error, error
    return delta * (magnitude - 0.5 * delta), math.copysign(delta, error)


def graph_policy_inputs(robot_states, pending_tasks,
                        context: Optional[SchedulingContext] = None):
    """Return the shared cost matrix, feasible coordinates and edge features."""
    context = context or SchedulingContext()
    matrix = build_cost_matrix(pending_tasks, robot_states, context)
    coordinates = np.argwhere(matrix.feasible)
    if coordinates.size == 0:
        return matrix, coordinates.reshape(0, 2), np.empty(
            (0, len(GRAPH_FEATURE_NAMES)), dtype=np.float32)
    pair_features = pair_feature_tensor(
        matrix, robot_states, pending_tasks, context)
    graph_features = graph_edge_feature_tensor(matrix, pair_features)
    features = np.asarray(
        graph_features[matrix.feasible], dtype=np.float32)
    if (features.shape != (len(coordinates), len(GRAPH_FEATURE_NAMES)) or
            not np.all(np.isfinite(features))):
        raise ValueError("graph policy features are invalid")
    return matrix, coordinates, features


def _transform_features(features):
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(GRAPH_FEATURE_NAMES):
        raise ValueError("GraphPPO edge features have an invalid shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("GraphPPO edge features contain non-finite values")
    return np.sign(values) * np.log1p(np.abs(values))


class GraphPPONetwork:
    def __init__(self, config: GraphPPOConfig, rng: np.random.Generator):
        self.config = config
        input_dim = len(GRAPH_FEATURE_NAMES)
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / config.hidden_size)
        self.params = {
            "w1": rng.normal(
                0, scale1, (input_dim, config.hidden_size)).astype(np.float32),
            "b1": np.zeros(config.hidden_size, dtype=np.float32),
            "wa": rng.normal(
                0, scale2, config.hidden_size).astype(np.float32),
            "ba": np.zeros(1, dtype=np.float32),
            "wv": rng.normal(
                0, scale2, config.hidden_size).astype(np.float32),
            "bv": np.zeros(1, dtype=np.float32),
        }

    def forward(self, features, *, cache=False):
        x = _transform_features(features)
        if len(x) == 0:
            raise ValueError("GraphPPO requires at least one feasible edge")
        z1 = x @ self.params["w1"] + self.params["b1"]
        hidden = np.tanh(z1)
        logits = hidden @ self.params["wa"] + self.params["ba"][0]
        probabilities = masked_softmax(logits, np.ones(len(logits), dtype=bool))
        pooled = hidden.mean(axis=0)
        value = float(pooled @ self.params["wv"] + self.params["bv"][0])
        if not math.isfinite(value):
            raise ValueError("GraphPPO value is non-finite")
        if not cache:
            return probabilities, value, logits
        return probabilities, value, logits, (x, hidden, pooled)

    def backward(self, cache, grad_logits, grad_value):
        x, hidden, pooled = cache
        grad_logits = np.asarray(grad_logits, dtype=np.float32)
        if grad_logits.shape != (len(x),) or not math.isfinite(grad_value):
            raise ValueError("GraphPPO gradients are invalid")
        grads = {
            "wa": hidden.T @ grad_logits,
            "ba": np.asarray([grad_logits.sum()], dtype=np.float32),
            "wv": pooled * float(grad_value),
            "bv": np.asarray([grad_value], dtype=np.float32),
        }
        hidden_gradient = (
            grad_logits[:, None] * self.params["wa"][None, :]
            + (float(grad_value) / len(x)) * self.params["wv"][None, :])
        z_gradient = hidden_gradient * (1.0-hidden**2)
        grads["w1"] = x.T @ z_gradient
        grads["b1"] = z_gradient.sum(axis=0)
        return grads


@dataclass
class GraphRolloutStep:
    features: np.ndarray
    action_index: int
    old_log_probability: float
    value: float
    reward: float
    done: bool
    advantage: float = 0.0
    return_value: float = 0.0


class GraphPPOModel:
    def __init__(self, config: Optional[GraphPPOConfig] = None,
                 seed: int = 42):
        self.config = config or GraphPPOConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.network = GraphPPONetwork(self.config, self.rng)
        self.optimizer = Adam(
            self.network.params, self.config.learning_rate,
            self.config.max_grad_norm)
        self.training_step = 0
        self.training_episodes = 0

    def edge_policy(self, features):
        return self.network.forward(features)

    def sample_edge(self, features):
        probabilities, value, _ = self.network.forward(features)
        index = int(self.rng.choice(len(probabilities), p=probabilities))
        return index, float(math.log(max(probabilities[index], 1e-12))), value

    def _prepare_advantages(self, rollout):
        gae = 0.0
        next_value = 0.0
        for step in reversed(rollout):
            nonterminal = 0.0 if step.done else 1.0
            delta = (step.reward + self.config.gamma*next_value*nonterminal
                     - step.value)
            gae = (delta + self.config.gamma*self.config.gae_lambda
                   * nonterminal*gae)
            step.advantage = float(gae)
            step.return_value = float(gae+step.value)
            next_value = step.value
        advantages = np.asarray(
            [step.advantage for step in rollout], dtype=np.float64)
        if len(advantages) > 1 and float(np.std(advantages)) > 1e-8:
            advantages = (advantages-np.mean(advantages))/np.std(advantages)
            for step, advantage in zip(rollout, advantages):
                step.advantage = float(advantage)

    def update(self, rollout) -> dict:
        if not rollout:
            return {"updates": 0, "mean_loss": None}
        self._prepare_advantages(rollout)
        losses = []
        actor_losses = []
        value_losses = []
        entropies = []
        maximum_value_error = 0.0
        for _ in range(self.config.update_epochs):
            for step in rollout:
                probabilities, value, _, cache = self.network.forward(
                    step.features, cache=True)
                probability = max(
                    float(probabilities[step.action_index]), 1e-12)
                ratio = math.exp(math.log(probability)-step.old_log_probability)
                active = (
                    (step.advantage >= 0 and
                     ratio <= 1.0+self.config.clip_epsilon) or
                    (step.advantage < 0 and
                     ratio >= 1.0-self.config.clip_epsilon))
                actor_factor = (-step.advantage*ratio) if active else 0.0
                grad_logits = -actor_factor*probabilities
                grad_logits[step.action_index] += actor_factor
                entropy = -float(np.sum(
                    probabilities*np.log(np.maximum(probabilities, 1e-12))))
                grad_logits += self.config.entropy_coefficient*probabilities*(
                    np.log(np.maximum(probabilities, 1e-12))+entropy)
                value_error = value-step.return_value
                value_loss, value_gradient = _huber_loss_and_gradient(
                    value_error, self.config.value_huber_delta)
                grad_value = self.config.value_coefficient*value_gradient
                grads = self.network.backward(
                    cache, grad_logits, grad_value)
                self.optimizer.step(self.network.params, grads)
                clipped_ratio = np.clip(
                    ratio, 1.0-self.config.clip_epsilon,
                    1.0+self.config.clip_epsilon)
                actor_loss = -min(
                    ratio*step.advantage,
                    clipped_ratio*step.advantage)
                loss = (actor_loss
                        + self.config.value_coefficient*value_loss
                        - self.config.entropy_coefficient*entropy)
                if not math.isfinite(loss):
                    raise ValueError("GraphPPO loss is non-finite")
                losses.append(float(loss))
                actor_losses.append(float(actor_loss))
                value_losses.append(float(value_loss))
                entropies.append(float(entropy))
                maximum_value_error = max(
                    maximum_value_error, abs(float(value_error)))
                self.training_step += 1
        self.training_episodes += 1
        return {
            "updates": len(losses),
            "mean_loss": float(np.mean(losses)),
            "mean_actor_loss": float(np.mean(actor_losses)),
            "mean_huber_value_loss": float(np.mean(value_losses)),
            "mean_entropy": float(np.mean(entropies)),
            "max_abs_value_error": maximum_value_error,
            "value_huber_delta": self.config.value_huber_delta,
        }

    def save(self, path):
        metadata = {
            "algorithm": GRAPH_PPO_ALGORITHM,
            "environment_version": RL_ENVIRONMENT_VERSION,
            "state_dim": len(GRAPH_FEATURE_NAMES),
            "action_dim": 1,
            "action_semantics": GRAPH_PPO_ACTION_SEMANTICS,
            "config": asdict(self.config),
            "training_step": self.training_step,
            "training_episodes": self.training_episodes,
            "seed": self.seed,
        }
        arrays = {
            f"network_{name}": value
            for name, value in self.network.params.items()
        }
        arrays.update(self.optimizer.arrays("adam"))
        save_checkpoint(path, metadata, arrays)

    @classmethod
    def load(cls, path, seed=42):
        metadata, arrays = load_checkpoint(
            path, expected_algorithm=GRAPH_PPO_ALGORITHM,
            expected_state_dim=len(GRAPH_FEATURE_NAMES),
            expected_action_dim=1)
        if metadata.get("action_semantics") != GRAPH_PPO_ACTION_SEMANTICS:
            raise ModelValidationError("GraphPPO action semantics mismatch")
        try:
            config = GraphPPOConfig(**metadata["config"])
            model = cls(config, seed)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelValidationError(f"invalid GraphPPO config: {exc}") from exc
        for name, target in model.network.params.items():
            value = np.asarray(
                arrays.get(f"network_{name}"), dtype=np.float32)
            if value.shape != target.shape or not np.all(np.isfinite(value)):
                raise ModelValidationError(
                    f"invalid GraphPPO parameter {name}")
            target[...] = value
        try:
            model.optimizer.restore(arrays, "adam")
            training_step = metadata.get("training_step", 0)
            training_episodes = metadata.get("training_episodes", 0)
            if (isinstance(training_step, bool) or
                    not isinstance(training_step, int) or training_step < 0 or
                    isinstance(training_episodes, bool) or
                    not isinstance(training_episodes, int) or
                    training_episodes < 0):
                raise ValueError("invalid training counters")
            model.training_step = training_step
            model.training_episodes = training_episodes
            model.optimizer.step_count = training_step
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelValidationError(
                f"invalid GraphPPO optimiser state: {exc}") from exc
        return model


class GraphPPOScheduler(BaseScheduler):
    def __init__(self, model_path, seed=42):
        super().__init__(GRAPH_PPO_ALGORITHM)
        self.model = GraphPPOModel.load(model_path, seed)

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        result = self.assign(
            pending_tasks, robot_states,
            SchedulingContext(congestion_map=congestion_map))
        if not result.assignments:
            return None
        assignment = result.assignments[0]
        return assignment.robot_id, assignment.task

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None):
        started = time.perf_counter()
        context = context or SchedulingContext()
        try:
            matrix, coordinates, features = graph_policy_inputs(
                robot_states, pending_tasks, context)
            if not len(coordinates):
                return _result_from_matching(
                    self.name, matrix, [], pending_tasks, robot_states,
                    context, started, {"model": GRAPH_PPO_ALGORITHM})
            physical_fine_tune = any(os.environ.get(
                name, "0").strip().lower() in {"1", "true", "yes", "on"}
                for name in (
                    "SMART_FACTORY_PHYSICAL_FINE_TUNE",
                    "SMART_FACTORY_GRAPH_PPO_FINE_TUNE"))
            if physical_fine_tune:
                edge_index, old_log_probability, value = (
                    self.model.sample_edge(features))
                row, column = coordinates[edge_index]
                pairs = [(int(row), int(column))]
                return _result_from_matching(
                    self.name, matrix, pairs, pending_tasks, robot_states,
                    context, started, {
                        "model": GRAPH_PPO_ALGORITHM,
                        "decoder": "on_policy_sample",
                        "physical_fine_tune": True,
                        "physical_rollout_step": {
                            "kind": "graph_ppo",
                            "features": features.tolist(),
                            "action_index": int(edge_index),
                            "old_log_probability": float(
                                old_log_probability),
                            "value": float(value),
                            "selected_robot_id": int(
                                matrix.robot_ids[int(row)]),
                            "selected_task_id": int(
                                matrix.tasks[int(column)].task_id),
                        },
                    })
            probabilities, _, _ = self.model.edge_policy(features)
            edge_index = int(np.argmax(probabilities))
            row, column = coordinates[edge_index]
            pairs = [(int(row), int(column))]
            return _result_from_matching(
                self.name, matrix, pairs, pending_tasks, robot_states,
                context, started, {
                    "model": GRAPH_PPO_ALGORITHM,
                    "decoder": "policy_argmax_edge",
                    "action_semantics": GRAPH_PPO_ACTION_SEMANTICS,
                    "selected_edge_index": edge_index,
                    "selected_probability": float(
                        probabilities[edge_index]),
                })
        except Exception as exc:
            return SchedulerResult(
                [], None, time.perf_counter()-started, False, self.name,
                {"reason": f"graph_ppo_inference_error:{type(exc).__name__}"})


def _strict_seeds(values: Iterable[int]):
    seeds = tuple(values)
    if (not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            for seed in seeds)):
        raise ValueError("GraphPPO seeds must be integers")
    return tuple(int(seed) for seed in seeds)


def train_graph_ppo(model: GraphPPOModel, episode_seeds: Iterable[int], *,
                    env_config: Optional[RLEnvironmentConfig] = None,
                    max_steps: Optional[int] = None) -> dict:
    seeds = _strict_seeds(episode_seeds)
    environment = SchedulingEnvironment(
        env_config, simulation_mode="headless")
    step_limit = (environment.config.max_steps_per_episode
                  if max_steps is None else int(max_steps))
    if step_limit <= 0:
        raise ValueError("GraphPPO max_steps must be positive")
    losses = []
    returns = []
    decision_count = 0
    runtime_rows = []
    for seed in seeds:
        robots, tasks, context = factory_scenario(seed)
        environment.reset(robots, tasks, context, seed=seed)
        rollout = []
        episode_return = 0.0
        for step_index in range(step_limit):
            live_robots, live_tasks, live_context = environment.policy_snapshot()
            matrix, coordinates, features = graph_policy_inputs(
                live_robots, live_tasks, live_context)
            if not len(coordinates):
                mask = environment.get_action_mask()
                if not mask[environment.no_op_action]:
                    raise ValueError("GraphPPO found no edge while no-op is masked")
                next_state, reward, terminated, truncated, _ = environment.step(
                    environment.no_op_action)
                del next_state
                done = bool(
                    terminated or truncated or step_index+1 >= step_limit)
                if rollout:
                    rollout[-1].reward += float(reward)
                    rollout[-1].done = done
                episode_return += float(reward)
                if done:
                    break
                continue
            edge_index, log_probability, value = model.sample_edge(features)
            row, column = coordinates[edge_index]
            robot_id = matrix.robot_ids[int(row)]
            task_id = matrix.tasks[int(column)].task_id
            action = environment.action_for_pair(robot_id, task_id)
            _, reward, terminated, truncated, _ = environment.step(action)
            done = bool(terminated or truncated or step_index+1 >= step_limit)
            rollout.append(GraphRolloutStep(
                features.copy(), edge_index, log_probability, value,
                float(reward), done))
            decision_count += 1
            episode_return += float(reward)
            if done:
                break
        report = model.update(rollout)
        if report["mean_loss"] is not None:
            losses.append(report["mean_loss"])
        returns.append(episode_return)
        runtime_rows.append(environment.runtime_telemetry())
    if (not np.all(np.isfinite(returns)) or
            (losses and not np.all(np.isfinite(losses)))):
        raise ValueError("GraphPPO training produced non-finite metrics")
    return {
        "episode_seeds": list(seeds),
        "episodes": len(seeds),
        "decisions": decision_count,
        "updates": model.training_step,
        "mean_return": float(np.mean(returns)),
        "mean_loss": float(np.mean(losses)) if losses else None,
        "runtime": summarize_runtime_telemetry(runtime_rows),
    }
