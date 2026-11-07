"""PER, n-step Double-DQN and standalone training for GraphRainbow."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from advanced_ai_common import load_checkpoint, masked_argmax, save_checkpoint
from config import RL_ENVIRONMENT_VERSION
from cost_guided_graph_observation import (
    COST_GUIDED_GRAPH_FEATURE_DIM,
    COST_GUIDED_GRAPH_SCHEMA_SHA256,
    CostGuidedGraphObservation,
    build_cost_guided_graph_observation,
)
from dual_objective import CountRewardProfile, UtilityV2RewardProfile
from graph_rainbow_model import GraphRainbowModel
from headless_training_runtime import (
    HeadlessRuntimeConfig, summarize_runtime_telemetry,
)
from rl_environment import RLEnvironmentConfig, SchedulingEnvironment
from schedulers import CostMatrix, ModelValidationError
from training_scenarios import SCENARIOS, factory_scenario


GRAPH_RAINBOW_REPLAY_ALGORITHM = "GraphRainbowReplay"
GRAPH_RAINBOW_REPLAY_VERSION = "graph-rainbow-ragged-replay-v1"
MAX_REPLAY_ROBOTS = 8
# Formal Webots manifests can contain roughly 250 horizon tasks and the live
# pending-task matrix may therefore exceed the standalone environment's
# 20-task snapshot cap.  Keep a finite corruption/allocation bound while
# covering the frozen physical protocol (whose current maximum is 244).
MAX_REPLAY_TASKS = 512
MAX_REPLAY_CAPACITY = 1_000_000


def graph_rainbow_replay_identity(model: GraphRainbowModel) -> str:
    if not isinstance(model, GraphRainbowModel):
        raise ValueError("GraphRainbow model is required for replay identity")
    value = {
        "replay_version": GRAPH_RAINBOW_REPLAY_VERSION,
        "feature_schema_sha256": COST_GUIDED_GRAPH_SCHEMA_SHA256,
        "prior_identity_sha256": model.prior_identity.sha256,
        "provenance": model.provenance.to_metadata(),
        "model_config": asdict(model.config),
    }
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GraphRainbowTransition:
    observation: CostGuidedGraphObservation
    action_index: int
    reward: float
    next_observation: CostGuidedGraphObservation
    done: bool
    discount: float

    def __post_init__(self):
        if (not isinstance(self.observation, CostGuidedGraphObservation) or
                not isinstance(
                    self.next_observation, CostGuidedGraphObservation) or
                self.observation.prior_identity_sha256 !=
                self.next_observation.prior_identity_sha256 or
                isinstance(self.action_index, bool) or
                not isinstance(self.action_index, (int, np.integer)) or
                not 0 <= int(self.action_index) <
                self.observation.action_count or
                not self.observation.action_mask[int(self.action_index)] or
                not math.isfinite(float(self.reward)) or
                not isinstance(self.done, bool) or
                not math.isfinite(float(self.discount)) or
                not 0.0 <= float(self.discount) <= 1.0):
            raise ValueError("GraphRainbow transition is invalid")
        object.__setattr__(self, "action_index", int(self.action_index))
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "discount", float(self.discount))


@dataclass(frozen=True)
class _RawStep:
    observation: CostGuidedGraphObservation
    action_index: int
    reward: float
    next_observation: CostGuidedGraphObservation
    terminated: bool


class GraphRainbowNStepAccumulator:
    def __init__(self, n_step: int, gamma: float):
        if (isinstance(n_step, bool) or not isinstance(n_step, int) or
                n_step <= 0 or not math.isfinite(float(gamma)) or
                not 0.0 <= float(gamma) <= 1.0):
            raise ValueError("GraphRainbow n-step configuration is invalid")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.queue = deque()

    def _transition(self) -> GraphRainbowTransition:
        if not self.queue:
            raise ValueError("GraphRainbow n-step queue is empty")
        reward = 0.0
        last = None
        count = 0
        done = False
        for index, step in enumerate(self.queue):
            if index >= self.n_step:
                break
            reward += (self.gamma ** index) * float(step.reward)
            last = step
            count += 1
            if step.terminated:
                done = True
                break
        first = self.queue[0]
        return GraphRainbowTransition(
            first.observation, first.action_index, reward,
            last.next_observation, done, self.gamma ** count)

    def append(self, observation, action_index, reward, next_observation, *,
               terminated: bool, episode_boundary: bool):
        if (not isinstance(terminated, bool) or
                not isinstance(episode_boundary, bool) or
                (terminated and not episode_boundary)):
            raise ValueError("GraphRainbow n-step boundary is invalid")
        raw = _RawStep(
            observation, int(action_index), float(reward),
            next_observation, terminated)
        # Reuse transition validation for the one-step action/identity fields.
        GraphRainbowTransition(
            raw.observation, raw.action_index, raw.reward,
            raw.next_observation, raw.terminated, self.gamma)
        self.queue.append(raw)
        emitted = []
        if len(self.queue) >= self.n_step:
            emitted.append(self._transition())
            self.queue.popleft()
        if episode_boundary:
            while self.queue:
                emitted.append(self._transition())
                self.queue.popleft()
        return emitted

    def clear(self):
        self.queue.clear()


class GraphRainbowReplayBuffer:
    def __init__(self, capacity: int, *, alpha: float,
                 priority_epsilon: float, prior_identity_sha256: str,
                 training_identity_sha256: str,
                 seed: int = 42):
        if (isinstance(capacity, bool) or not isinstance(capacity, int) or
                not 0 < capacity <= MAX_REPLAY_CAPACITY or
                not math.isfinite(float(alpha)) or
                not 0.0 <= float(alpha) <= 1.0 or
                not math.isfinite(float(priority_epsilon)) or
                float(priority_epsilon) <= 0.0 or
                not isinstance(prior_identity_sha256, str) or
                len(prior_identity_sha256) != 64 or
                not isinstance(training_identity_sha256, str) or
                len(training_identity_sha256) != 64 or
                any(character not in "0123456789abcdef"
                    for character in (
                        prior_identity_sha256 + training_identity_sha256))):
            raise ValueError("GraphRainbow replay configuration is invalid")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self.prior_identity_sha256 = prior_identity_sha256
        self.training_identity_sha256 = training_identity_sha256
        self.rng = np.random.default_rng(int(seed))
        self.data = []
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        self.cursor = 0

    def __len__(self):
        return len(self.data)

    def add(self, transition: GraphRainbowTransition) -> None:
        if (not isinstance(transition, GraphRainbowTransition) or
                transition.observation.prior_identity_sha256 !=
                self.prior_identity_sha256):
            raise ValueError("GraphRainbow replay transition identity differs")
        maximum = (
            float(np.max(self.priorities[:len(self.data)]))
            if self.data else 1.0)
        maximum = max(maximum, self.priority_epsilon)
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.cursor] = transition
        self.priorities[self.cursor] = maximum
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or
                batch_size <= 0 or batch_size > len(self.data) or
                not math.isfinite(float(beta)) or
                not 0.0 < float(beta) <= 1.0):
            raise ValueError("GraphRainbow replay sample request is invalid")
        raw = np.maximum(
            self.priorities[:len(self.data)], self.priority_epsilon)
        scaled = raw ** self.alpha
        probabilities = scaled / float(np.sum(scaled))
        indices = self.rng.choice(
            len(self.data), size=batch_size, replace=False,
            p=probabilities)
        weights = (
            len(self.data) * probabilities[indices]) ** (-float(beta))
        weights /= float(np.max(weights))
        if not np.all(np.isfinite(weights)):
            raise ValueError("GraphRainbow importance weights are invalid")
        return (
            np.asarray(indices, dtype=np.int64),
            [self.data[int(index)] for index in indices],
            weights.astype(np.float32))

    def update_priorities(self, indices, values) -> None:
        indices = np.asarray(indices)
        values = np.asarray(values, dtype=np.float64)
        if (indices.ndim != 1 or values.shape != indices.shape or
                any(isinstance(value, (bool, np.bool_))
                    for value in indices.tolist()) or
                np.any(indices < 0) or np.any(indices >= len(self.data)) or
                not np.all(np.isfinite(values)) or np.any(values < 0.0)):
            raise ValueError("GraphRainbow replay priorities are invalid")
        self.priorities[indices.astype(np.int64)] = (
            values + self.priority_epsilon)

    @staticmethod
    def _pack_observations(observations):
        feature_offsets = [0]
        coordinate_offsets = [0]
        features = []
        coordinates = []
        current_times = []
        for observation in observations:
            features.append(observation.action_features)
            coordinates.append(observation.coordinates)
            feature_offsets.append(
                feature_offsets[-1] + observation.action_count)
            coordinate_offsets.append(
                coordinate_offsets[-1] + observation.edge_count)
            current_times.append(observation.current_time)
        return {
            "observation_features": np.concatenate(
                features, axis=0).astype(np.float32),
            "observation_feature_offsets": np.asarray(
                feature_offsets, dtype=np.int64),
            "observation_coordinates": np.concatenate(
                coordinates, axis=0).astype(np.int64),
            "observation_coordinate_offsets": np.asarray(
                coordinate_offsets, dtype=np.int64),
            "observation_current_times": np.asarray(
                current_times, dtype=np.float64),
        }

    @staticmethod
    def _unpack_observations(arrays, count, prior_identity_sha256):
        features = np.asarray(
            arrays["observation_features"], dtype=np.float32)
        feature_offsets = np.asarray(
            arrays["observation_feature_offsets"], dtype=np.int64)
        coordinates = np.asarray(
            arrays["observation_coordinates"], dtype=np.int64)
        coordinate_offsets = np.asarray(
            arrays["observation_coordinate_offsets"], dtype=np.int64)
        current_times = np.asarray(
            arrays["observation_current_times"], dtype=np.float64)
        if (features.ndim != 2 or
                features.shape[1] != COST_GUIDED_GRAPH_FEATURE_DIM or
                feature_offsets.shape != (count + 1,) or
                coordinates.ndim != 2 or coordinates.shape[1] != 2 or
                coordinate_offsets.shape != (count + 1,) or
                current_times.shape != (count,) or
                feature_offsets[0] != 0 or
                coordinate_offsets[0] != 0 or
                feature_offsets[-1] != len(features) or
                coordinate_offsets[-1] != len(coordinates) or
                np.any(np.diff(feature_offsets) <= 0) or
                np.any(np.diff(coordinate_offsets) < 0)):
            raise ValueError("GraphRainbow packed observations are invalid")
        result = []
        for index in range(count):
            action_features = features[
                feature_offsets[index]:feature_offsets[index + 1]]
            edge_coordinates = coordinates[
                coordinate_offsets[index]:coordinate_offsets[index + 1]]
            edge_count = len(edge_coordinates)
            if len(action_features) != edge_count + 1:
                raise ValueError("GraphRainbow packed action count differs")
            if (edge_count and (
                    np.any(edge_coordinates < 0) or
                    np.any(edge_coordinates[:, 0] >= MAX_REPLAY_ROBOTS) or
                    np.any(edge_coordinates[:, 1] >= MAX_REPLAY_TASKS))):
                raise ValueError("GraphRainbow replay coordinates are invalid")
            rows = (
                int(np.max(edge_coordinates[:, 0])) + 1
                if edge_count else 0)
            columns = (
                int(np.max(edge_coordinates[:, 1])) + 1
                if edge_count else 0)
            feasible = np.zeros((rows, columns), dtype=bool)
            if edge_count:
                feasible[
                    edge_coordinates[:, 0],
                    edge_coordinates[:, 1]] = True
            matrix = CostMatrix(
                tuple(range(rows)), tuple(range(columns)),
                np.zeros((rows, columns), dtype=np.float64),
                feasible,
                np.zeros((rows, columns), dtype=np.float64))
            mask = np.ones(edge_count + 1, dtype=bool)
            mask[-1] = edge_count == 0
            result.append(CostGuidedGraphObservation(
                matrix, edge_coordinates, action_features, mask,
                edge_count, prior_identity_sha256,
                COST_GUIDED_GRAPH_SCHEMA_SHA256,
                float(current_times[index])))
        return result

    def save(self, path) -> None:
        observations = []
        for transition in self.data:
            observations.extend((
                transition.observation,
                transition.next_observation))
        packed = self._pack_observations(observations) if observations else {
            "observation_features": np.empty(
                (0, COST_GUIDED_GRAPH_FEATURE_DIM), dtype=np.float32),
            "observation_feature_offsets": np.asarray([0], dtype=np.int64),
            "observation_coordinates": np.empty((0, 2), dtype=np.int64),
            "observation_coordinate_offsets": np.asarray([0], dtype=np.int64),
            "observation_current_times": np.empty(0, dtype=np.float64),
        }
        metadata = {
            "algorithm": GRAPH_RAINBOW_REPLAY_ALGORITHM,
            "environment_version": RL_ENVIRONMENT_VERSION,
            "state_dim": COST_GUIDED_GRAPH_FEATURE_DIM,
            "action_dim": 1,
            "replay_version": GRAPH_RAINBOW_REPLAY_VERSION,
            "feature_schema_sha256": COST_GUIDED_GRAPH_SCHEMA_SHA256,
            "prior_identity_sha256": self.prior_identity_sha256,
            "training_identity_sha256": self.training_identity_sha256,
            "capacity": self.capacity,
            "alpha": self.alpha,
            "priority_epsilon": self.priority_epsilon,
            "cursor": self.cursor,
            "count": len(self.data),
            "rng_state": self.rng.bit_generator.state,
        }
        arrays = {
            **packed,
            "actions": np.asarray(
                [row.action_index for row in self.data], dtype=np.int64),
            "rewards": np.asarray(
                [row.reward for row in self.data], dtype=np.float64),
            "done": np.asarray(
                [row.done for row in self.data], dtype=np.uint8),
            "discounts": np.asarray(
                [row.discount for row in self.data], dtype=np.float64),
            "priorities": self.priorities[:len(self.data)].copy(),
        }
        save_checkpoint(path, metadata, arrays)

    @classmethod
    def load(cls, path, *, expected_prior_identity_sha256,
             expected_training_identity_sha256):
        metadata, arrays = load_checkpoint(
            path, expected_algorithm=GRAPH_RAINBOW_REPLAY_ALGORITHM,
            expected_state_dim=COST_GUIDED_GRAPH_FEATURE_DIM,
            expected_action_dim=1)
        try:
            required = {
                "replay_version", "feature_schema_sha256",
                "prior_identity_sha256", "training_identity_sha256",
                "capacity", "alpha",
                "priority_epsilon", "cursor", "count", "rng_state",
            }
            if (not required.issubset(metadata) or
                    metadata["replay_version"] !=
                    GRAPH_RAINBOW_REPLAY_VERSION or
                    metadata["feature_schema_sha256"] !=
                    COST_GUIDED_GRAPH_SCHEMA_SHA256 or
                    metadata["prior_identity_sha256"] !=
                    expected_prior_identity_sha256 or
                    metadata["training_identity_sha256"] !=
                    expected_training_identity_sha256):
                raise ValueError("GraphRainbow replay identity differs")
            capacity = int(metadata["capacity"])
            count = int(metadata["count"])
            cursor = int(metadata["cursor"])
            if (capacity <= 0 or not 0 <= count <= capacity or
                    not 0 <= cursor < capacity or
                    (count < capacity and cursor != count)):
                raise ValueError("GraphRainbow replay counters are invalid")
            replay = cls(
                capacity, alpha=float(metadata["alpha"]),
                priority_epsilon=float(metadata["priority_epsilon"]),
                prior_identity_sha256=expected_prior_identity_sha256,
                training_identity_sha256=
                expected_training_identity_sha256)
            observations = cls._unpack_observations(
                arrays, 2 * count, expected_prior_identity_sha256)
            actions = np.asarray(arrays["actions"], dtype=np.int64)
            rewards = np.asarray(arrays["rewards"], dtype=np.float64)
            done = np.asarray(arrays["done"], dtype=np.uint8)
            discounts = np.asarray(arrays["discounts"], dtype=np.float64)
            priorities = np.asarray(
                arrays["priorities"], dtype=np.float64)
            if (any(value.shape != (count,) for value in (
                    actions, rewards, done, discounts, priorities)) or
                    np.any(done > 1) or
                    not np.all(np.isfinite(rewards)) or
                    not np.all(np.isfinite(discounts)) or
                    np.any(discounts < 0.0) or np.any(discounts > 1.0) or
                    not np.all(np.isfinite(priorities)) or
                    np.any(priorities <= 0.0)):
                raise ValueError("GraphRainbow replay arrays are invalid")
            replay.data = [
                GraphRainbowTransition(
                    observations[2 * index], int(actions[index]),
                    float(rewards[index]), observations[2 * index + 1],
                    bool(done[index]), float(discounts[index]))
                for index in range(count)]
            replay.priorities[:count] = priorities
            replay.cursor = cursor
            replay.rng.bit_generator.state = metadata["rng_state"]
            return replay
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelValidationError(
                f"invalid GraphRainbow replay: {exc}") from exc


def c51_project(model: GraphRainbowModel,
                transition: GraphRainbowTransition) -> np.ndarray:
    """Double-DQN selection with target-network C51 evaluation."""
    support = model.config.support.astype(np.float64)
    if transition.done:
        target_atoms = np.full_like(support, transition.reward)
        next_distribution = np.ones_like(support)
    else:
        online_values = model.q_values(
            transition.next_observation, target=False, noisy=True)
        next_action = masked_argmax(
            online_values, transition.next_observation.action_mask)
        target_distributions, _ = model.distributions(
            transition.next_observation, target=True, noisy=True)
        next_distribution = np.asarray(
            target_distributions[next_action], dtype=np.float64)
        target_atoms = (
            transition.reward + transition.discount * support)
    target_atoms = np.clip(
        target_atoms, model.config.v_min, model.config.v_max)
    delta = (
        model.config.v_max - model.config.v_min) / (
        model.config.atom_count - 1)
    positions = (target_atoms - model.config.v_min) / delta
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    projected = np.zeros(model.config.atom_count, dtype=np.float64)
    for index, probability in enumerate(next_distribution):
        low = int(lower[index])
        high = int(upper[index])
        if low == high:
            projected[low] += probability
        else:
            projected[low] += probability * (high - positions[index])
            projected[high] += probability * (positions[index] - low)
    total = float(np.sum(projected))
    if (not np.all(np.isfinite(projected)) or total <= 0.0):
        raise ValueError("GraphRainbow C51 projection is invalid")
    return (projected / total).astype(np.float32)


def graph_rainbow_beta(model: GraphRainbowModel) -> float:
    fraction = min(
        1.0, model.training_step / model.config.per_beta_steps)
    return float(
        model.config.per_beta_start
        + fraction * (1.0 - model.config.per_beta_start))


def update_graph_rainbow(model: GraphRainbowModel,
                         replay: GraphRainbowReplayBuffer) -> dict:
    if (replay.prior_identity_sha256 != model.prior_identity.sha256 or
            replay.training_identity_sha256 !=
            graph_rainbow_replay_identity(model) or
            replay.capacity != model.config.replay_capacity or
            not math.isclose(
                replay.alpha, model.config.per_alpha,
                rel_tol=0.0, abs_tol=0.0) or
            not math.isclose(
                replay.priority_epsilon, model.config.priority_epsilon,
                rel_tol=0.0, abs_tol=0.0)):
        raise ValueError("GraphRainbow replay training identity differs")
    if len(replay) < model.config.batch_size:
        return {"updates": 0, "mean_loss": None, "mean_abs_td_error": None}
    beta = graph_rainbow_beta(model)
    indices, transitions, weights = replay.sample(
        model.config.batch_size, beta)
    accumulated = {
        name: np.zeros_like(value)
        for name, value in model.online.params.items()}
    losses = []
    errors = []
    support = model.config.support
    for transition, weight in zip(transitions, weights):
        distributions, _, cache = model.distributions(
            transition.observation, target=False, noisy=True, cache=True)
        predicted = np.asarray(
            distributions[transition.action_index], dtype=np.float64)
        target = c51_project(model, transition)
        loss = -float(np.sum(
            target * np.log(np.maximum(predicted, 1e-12))))
        grad_logits = np.zeros_like(distributions, dtype=np.float32)
        grad_logits[transition.action_index] = (
            float(weight) * (predicted - target)
            / model.config.batch_size)
        grads = model.online.backward(cache, grad_logits)
        for name in accumulated:
            accumulated[name] += grads[name]
        predicted_value = float(predicted @ support)
        target_value = float(np.asarray(target) @ support)
        errors.append(abs(target_value - predicted_value))
        losses.append(float(weight) * loss)
    if (not np.all(np.isfinite(losses)) or
            not np.all(np.isfinite(errors))):
        raise ValueError("GraphRainbow update metrics are non-finite")
    model.optimizer.step(model.online.params, accumulated)
    model.training_step = model.optimizer.step_count
    replay.update_priorities(indices, errors)
    target_synced = False
    if model.training_step % model.config.target_update_interval == 0:
        model.sync_target()
        target_synced = True
    return {
        "updates": 1,
        "mean_loss": float(np.mean(losses)),
        "mean_abs_td_error": float(np.mean(errors)),
        "beta": beta,
        "target_synced": target_synced,
    }


def _validate_profile(model, profile):
    if not isinstance(profile, (CountRewardProfile, UtilityV2RewardProfile)):
        raise ValueError("GraphRainbow requires a formal reward profile")
    if (profile.canonical().get("objective") !=
            model.prior_identity.objective or
            profile.version != model.provenance.reward_profile_version or
            profile.sha256 != model.provenance.reward_profile_sha256):
        raise ValueError("GraphRainbow reward profile identity differs")


def _training_seeds(model, values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(values)
    if (not result or len(result) != len(set(result)) or
            any(isinstance(value, bool) or
                not isinstance(value, (int, np.integer))
                for value in result)):
        raise ValueError("GraphRainbow training seeds are invalid")
    result = tuple(int(value) for value in result)
    if not set(result).issubset(model.provenance.standalone_seeds):
        raise ValueError("training seed is outside standalone provenance")
    return result


def _scenarios(values):
    result = tuple(str(value) for value in values)
    if (not result or len(result) != len(set(result)) or
            any(value not in SCENARIOS for value in result)):
        raise ValueError("GraphRainbow scenarios are invalid")
    return result


def _environment_action(environment, observation, action_index):
    coordinate = observation.coordinate_for_action(action_index)
    if coordinate is None:
        return environment.no_op_action
    row, column = coordinate
    return environment.action_for_pair(
        observation.matrix.robot_ids[row],
        observation.matrix.tasks[column].task_id)


def train_graph_rainbow(
        model: GraphRainbowModel, replay: GraphRainbowReplayBuffer,
        episode_seeds: Iterable[int], *, reward_profile,
        scenario_ids=("A", "B", "C"),
        env_config: RLEnvironmentConfig | None = None,
        runtime_config: HeadlessRuntimeConfig | None = None,
        scenario_factory=factory_scenario) -> dict:
    if (not isinstance(model, GraphRainbowModel) or
            model.provenance.training_phase != "standalone" or
            not isinstance(replay, GraphRainbowReplayBuffer) or
            replay.prior_identity_sha256 != model.prior_identity.sha256 or
            replay.training_identity_sha256 !=
            graph_rainbow_replay_identity(model) or
            replay.capacity != model.config.replay_capacity or
            replay.alpha != model.config.per_alpha or
            replay.priority_epsilon != model.config.priority_epsilon):
        raise ValueError("GraphRainbow standalone model/replay is required")
    seeds = _training_seeds(model, episode_seeds)
    scenarios = _scenarios(scenario_ids)
    _validate_profile(model, reward_profile)
    horizon = float(reward_profile.horizon_seconds)
    env_config = env_config or RLEnvironmentConfig(
        max_robots=8, max_tasks=20, max_steps_per_episode=10000)
    runtime_config = runtime_config or HeadlessRuntimeConfig(
        episode_end_time=horizon, max_advance_seconds=horizon,
        fixed_horizon=True)
    if (not runtime_config.fixed_horizon or
            not math.isclose(
                runtime_config.episode_end_time, horizon,
                rel_tol=0.0, abs_tol=1e-9)):
        raise ValueError("GraphRainbow runtime horizon differs from reward")
    environment = SchedulingEnvironment(
        env_config, reward_profile=reward_profile,
        simulation_mode="headless", runtime_config=runtime_config)
    episode_rows = []
    runtime_rows = []
    update_rows = []
    for seed in seeds:
        for scenario in scenarios:
            robots, tasks, context = scenario_factory(
                seed, scenario_id=scenario, duration_seconds=horizon)
            configuration = dict(context.configuration)
            configuration["graph_traffic_enabled"] = True
            environment.reset(
                robots, tasks,
                replace(context, configuration=configuration), seed=seed)
            raw_steps = []
            episode_return = 0.0
            terminated = truncated = False
            last_info = {}
            for _step in range(env_config.max_steps_per_episode):
                live = environment.policy_snapshot()
                observation = build_cost_guided_graph_observation(
                    *live, model.prior_identity)
                action_index = model.select_action(
                    observation, deterministic=False)
                environment_action = _environment_action(
                    environment, observation, action_index)
                _, reward, terminated, truncated, info = environment.step(
                    environment_action)
                if (info.get("invalid_action") or
                        info.get("assignment_rejected") or
                        info.get("deferred_feasible_assignment") or
                        info.get("invalid_truncation")):
                    raise RuntimeError(
                        "GraphRainbow standalone action contract failed: "
                        f"{info}")
                components = info.get("reward_components")
                if (not isinstance(components, dict) or
                        not math.isclose(
                            float(components.get("reward_total", math.nan)),
                            float(reward), rel_tol=0.0, abs_tol=1e-12)):
                    raise RuntimeError(
                        "GraphRainbow step reward audit failed")
                next_live = environment.policy_snapshot()
                next_observation = build_cost_guided_graph_observation(
                    *next_live, model.prior_identity)
                raw_steps.append((
                    observation, action_index, float(reward),
                    next_observation, bool(terminated),
                    bool(terminated or truncated)))
                episode_return += float(reward)
                last_info = info
                if terminated or truncated:
                    break
            if not (terminated or truncated):
                raise RuntimeError("GraphRainbow episode exceeded step budget")
            telemetry = environment.runtime_telemetry()
            accounted = telemetry.get(
                "reward_totals", {}).get("reward_total")
            error = (
                float("inf") if accounted is None else
                abs(episode_return - float(accounted)))
            if not math.isfinite(episode_return) or error > 1e-9:
                raise RuntimeError(
                    "GraphRainbow formal reward conservation failed")
            # Commit audited episode data to replay only after all checks pass.
            accumulator = GraphRainbowNStepAccumulator(
                model.config.n_step, model.config.gamma)
            updates_before = model.training_step
            losses = []
            for raw in raw_steps:
                transitions = accumulator.append(
                    raw[0], raw[1], raw[2], raw[3],
                    terminated=raw[4], episode_boundary=raw[5])
                for transition in transitions:
                    replay.add(transition)
                    update = update_graph_rainbow(model, replay)
                    if update["mean_loss"] is not None:
                        losses.append(update["mean_loss"])
                        update_rows.append(update)
            model.training_episodes += 1
            episode_rows.append({
                "seed": seed,
                "scenario": scenario,
                "steps": len(raw_steps),
                "return": float(episode_return),
                "reward_total": float(accounted),
                "reward_conservation_error": error,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "completed_count": int(
                    last_info.get("completed_count", 0)),
                "replay_size": len(replay),
                "updates": model.training_step - updates_before,
                "mean_loss": (
                    float(np.mean(losses)) if losses else None),
            })
            runtime_rows.append(telemetry)
    return {
        "algorithm": "GraphRainbow",
        "policy_id": model.prior_identity.policy_id,
        "objective": model.prior_identity.objective,
        "prior_identity_sha256": model.prior_identity.sha256,
        "reward_profile_version": reward_profile.version,
        "reward_profile_sha256": reward_profile.sha256,
        "episode_seeds": list(seeds),
        "scenarios": list(scenarios),
        "episode_count": len(episode_rows),
        "episodes": episode_rows,
        "training_step": model.training_step,
        "training_episodes": model.training_episodes,
        "replay_size": len(replay),
        "mean_return": float(np.mean([
            row["return"] for row in episode_rows])),
        "mean_update_loss": (
            float(np.mean([
                row["mean_loss"] for row in update_rows]))
            if update_rows else None),
        "max_reward_conservation_error": float(max(
            row["reward_conservation_error"] for row in episode_rows)),
        "runtime": summarize_runtime_telemetry(runtime_rows),
    }
