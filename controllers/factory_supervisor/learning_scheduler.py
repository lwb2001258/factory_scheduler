"""Data-driven schedulers with deterministic safety boundaries.

The learned components only estimate edge costs/scores. Feasibility is still
owned by :mod:`schedulers`, and the final one-to-one assignment is solved by
Hungarian matching and validated through the same runtime path oracle used by
the conventional algorithms.
"""

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import RL_ENVIRONMENT_VERSION, RobotState, TaskStatus
from schedulers import (
    Assignment,
    BaseScheduler,
    CostMatrix,
    ModelValidationError,
    SchedulerResult,
    SchedulingContext,
    _pair_cost,
    _result_from_matching,
    build_cost_matrix,
)
from task_generator import TransportTask
from traffic_features import TRAFFIC_FEATURE_NAMES, TRAFFIC_VERSION, TrafficSnapshot


FEATURE_VERSION = "factory-pair-v2-priority-horizon"
RIDGE_MODEL_VERSION = "learned-cost-ridge-v5-unfloored-objective-expert"
GRAPH_MODEL_VERSION = "graph-edge-imitation-v3-unfloored-objective-expert"
WEBOTS_SHARED_SUPERVISION_VERSION = (
    "webots-shared-supervision-v3-unfloored-dual-objective-expert")

PAIR_FEATURE_NAMES = (
    "robot_x",
    "robot_y",
    "robot_battery_pct",
    "robot_tasks_completed",
    "robot_total_distance",
    "pickup_x",
    "pickup_y",
    "delivery_x",
    "delivery_y",
    "euclidean_empty_distance",
    "euclidean_loaded_distance",
    "baseline_pair_cost",
    "task_priority",
    "task_waiting_seconds",
    "pickup_congestion",
    "delivery_congestion",
    "idle_robot_count",
    "pending_task_count",
    "task_horizon_limited",
    "seconds_to_episode_end",
)

GRAPH_CONTEXT_FEATURE_NAMES = (
    "cost_minus_row_min",
    "cost_minus_column_min",
    "cost_minus_global_min",
    "cost_over_row_mean",
    "cost_over_column_mean",
    "cost_over_global_mean",
    "row_feasible_fraction",
    "column_feasible_fraction",
    "row_cost_rank",
    "column_cost_rank",
)

GRAPH_FEATURE_NAMES = PAIR_FEATURE_NAMES + GRAPH_CONTEXT_FEATURE_NAMES
TRAFFIC_GRAPH_FEATURE_NAMES = GRAPH_FEATURE_NAMES + TRAFFIC_FEATURE_NAMES
TRAFFIC_GRAPH_MODEL_VERSION = "graph-edge-imitation-v4-route-traffic"
TRAFFIC_GRAPH_FEATURE_VERSION = "factory-graph-v3-route-traffic"
TRAFFIC_PAIR_FEATURE_NAMES = PAIR_FEATURE_NAMES + TRAFFIC_FEATURE_NAMES + (
    "unfloored_pair_cost",
    "task_deadline_seconds",
    "task_weighted_deadline_pressure",
    "traffic_expert_penalty",
)
TRAFFIC_PAIR_FEATURE_VERSION = "factory-pair-v3-route-traffic-sla"
TRAFFIC_RIDGE_MODEL_VERSION = "learned-cost-ridge-v6-route-traffic-sla"


def _finite_number(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _position(value) -> Tuple[float, float]:
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) < 2:
        return 0.0, 0.0
    return _finite_number(value[0]), _finite_number(value[1])


def _congestion_at(position: Tuple[float, float],
                   context: SchedulingContext) -> float:
    grid = context.congestion_map
    if grid is None:
        return 0.0
    try:
        height = len(grid)
        width = len(grid[0]) if height else 0
    except (TypeError, IndexError):
        return 0.0
    if not height or not width:
        return 0.0
    resolution = max(1e-9, _finite_number(
        context.configuration.get("congestion_grid_resolution", 1.0), 1.0))
    x, y = position
    column = int((x + width * resolution / 2.0) / resolution)
    row = int((y + height * resolution / 2.0) / resolution)
    if not (0 <= row < height and 0 <= column < width):
        return 0.0
    try:
        return _finite_number(grid[row][column])
    except (TypeError, IndexError):
        return 0.0


def pair_feature_vector(robot_id: int, task: TransportTask,
                        robot_states: Dict[int, dict],
                        pending_tasks: Sequence[TransportTask],
                        context: Optional[SchedulingContext] = None,
                        *, base_cost: Optional[float] = None,
                        pending_count_override: Optional[int] = None,
                        idle_count_override: Optional[int] = None
                        ) -> np.ndarray:
    """Build the versioned, future-information-free pair feature vector."""
    context = context or SchedulingContext()
    if robot_id not in robot_states:
        raise ValueError(f"unknown robot ID: {robot_id}")
    state = robot_states[robot_id]
    robot_x, robot_y = _position(state.get("position"))
    pickup_x, pickup_y = _position(task.pickup_position)
    delivery_x, delivery_y = _position(task.delivery_position)
    empty_distance = math.hypot(robot_x - pickup_x, robot_y - pickup_y)
    loaded_distance = math.hypot(pickup_x - delivery_x,
                                 pickup_y - delivery_y)
    if base_cost is None:
        base_cost = _pair_cost(robot_id, task, robot_states, context)
    base_cost = float(base_cost)
    if not math.isfinite(base_cost) or base_cost < 0:
        raise ValueError("pair has no finite non-negative baseline cost")
    idle_count = (sum(
        1 for robot in robot_states.values()
        if robot.get("state") == RobotState.IDLE and
        robot.get("current_task") is None)
        if idle_count_override is None else max(0, int(idle_count_override)))
    pending_count = (sum(
        1 for item in pending_tasks if item.status == TaskStatus.PENDING)
        if pending_count_override is None else
        max(0, int(pending_count_override)))
    episode_end_time = _finite_number(
        context.configuration.get(
            "episode_end_time",
            context.configuration.get("duration_seconds", 1800.0)),
        1800.0,
    )
    if episode_end_time <= 0:
        episode_end_time = 1800.0
    horizon_limited = float(task.is_horizon_limited(episode_end_time))
    seconds_to_episode_end = max(
        0.0, episode_end_time - _finite_number(context.current_time))
    result = np.asarray([
        robot_x,
        robot_y,
        _finite_number(state.get("battery", 100.0), 100.0),
        _finite_number(state.get("tasks_completed", 0.0)),
        _finite_number(state.get("total_distance", 0.0)),
        pickup_x,
        pickup_y,
        delivery_x,
        delivery_y,
        empty_distance,
        loaded_distance,
        base_cost,
        _finite_number(task.priority, 1.0),
        max(0.0, _finite_number(context.current_time) -
            _finite_number(task.arrival_time)),
        _congestion_at((pickup_x, pickup_y), context),
        _congestion_at((delivery_x, delivery_y), context),
        float(idle_count),
        float(pending_count),
        horizon_limited,
        seconds_to_episode_end,
    ], dtype=np.float64)
    if result.shape != (len(PAIR_FEATURE_NAMES),) or not np.all(
            np.isfinite(result)):
        raise ValueError("pair feature vector is invalid")
    return result


def pair_feature_tensor(matrix: CostMatrix, robot_states: Dict[int, dict],
                        pending_tasks: Sequence[TransportTask],
                        context: SchedulingContext) -> np.ndarray:
    """Return ``robots x tasks x features``; infeasible edges remain zero."""
    tensor = np.zeros(
        matrix.values.shape + (len(PAIR_FEATURE_NAMES),), dtype=np.float64)
    idle_count = sum(1 for state in robot_states.values()
                     if state.get("state") == RobotState.IDLE and
                     state.get("current_task") is None)
    pending_count = sum(task.status == TaskStatus.PENDING for task in pending_tasks)
    for row, robot_id in enumerate(matrix.robot_ids):
        for column, task in enumerate(matrix.tasks):
            if matrix.feasible[row, column]:
                tensor[row, column] = pair_feature_vector(
                    robot_id, task, robot_states, pending_tasks, context,
                    base_cost=float(matrix.values[row, column]),
                    pending_count_override=pending_count,
                    idle_count_override=idle_count)
    return tensor


def _rank_fraction(values: np.ndarray, value: float) -> float:
    if values.size <= 1:
        return 0.0
    less = int(np.count_nonzero(values < value))
    equal = int(np.count_nonzero(values == value))
    return (less + max(0, equal - 1) * 0.5) / (values.size - 1)


def graph_edge_feature_tensor(matrix: CostMatrix,
                              pair_features: np.ndarray) -> np.ndarray:
    """Add permutation-invariant bipartite neighbourhood statistics."""
    expected = matrix.values.shape + (len(PAIR_FEATURE_NAMES),)
    if pair_features.shape != expected:
        raise ValueError(f"pair feature tensor must have shape {expected}")
    output = np.zeros(
        matrix.values.shape + (len(GRAPH_FEATURE_NAMES),), dtype=np.float64)
    finite_values = matrix.values[matrix.feasible]
    if finite_values.size == 0:
        return output
    global_min = float(np.min(finite_values))
    global_mean = max(1e-9, float(np.mean(finite_values)))
    row_count, column_count = matrix.values.shape
    for row in range(row_count):
        row_values = matrix.values[row, matrix.feasible[row]]
        if row_values.size == 0:
            continue
        row_min = float(np.min(row_values))
        row_mean = max(1e-9, float(np.mean(row_values)))
        for column in range(column_count):
            if not matrix.feasible[row, column]:
                continue
            column_values = matrix.values[
                matrix.feasible[:, column], column]
            cost = float(matrix.values[row, column])
            column_min = float(np.min(column_values))
            column_mean = max(1e-9, float(np.mean(column_values)))
            context_features = np.asarray([
                cost - row_min,
                cost - column_min,
                cost - global_min,
                cost / row_mean,
                cost / column_mean,
                cost / global_mean,
                row_values.size / max(1, column_count),
                column_values.size / max(1, row_count),
                _rank_fraction(row_values, cost),
                _rank_fraction(column_values, cost),
            ], dtype=np.float64)
            output[row, column] = np.concatenate((
                pair_features[row, column], context_features))
    if not np.all(np.isfinite(output)):
        raise ValueError("graph feature tensor contains non-finite values")
    return output


def _traffic_edge_values(matrix, context):
    snapshot = context.traffic_context
    if not isinstance(snapshot, TrafficSnapshot) or snapshot.now != context.current_time:
        raise ValueError("traffic model requires a current, frozen traffic snapshot")
    features = np.zeros(matrix.values.shape + (len(TRAFFIC_FEATURE_NAMES),))
    penalties = np.zeros(matrix.values.shape)
    for row, rid in enumerate(matrix.robot_ids):
        for column, task in enumerate(matrix.tasks):
            if matrix.feasible[row, column]:
                vector = snapshot.features(rid, task.pickup_position, task.delivery_position)
                features[row, column] = vector
                penalties[row, column] = snapshot.expert_penalty(vector)
    return features, penalties


def _traffic_adjusted_pairs(matrix, pair_features, context):
    """Share endpoint and route semantics between both learned matchers."""
    if pair_features.shape != matrix.values.shape + (len(PAIR_FEATURE_NAMES),):
        raise ValueError("traffic pair feature tensor has an invalid shape")
    traffic, penalties = _traffic_edge_values(matrix, context)
    raw = (matrix.unfloored_values if matrix.unfloored_values is not None
           else matrix.values)
    values = np.where(matrix.feasible, np.maximum(0.0, raw + penalties), np.inf)
    adjusted = CostMatrix(matrix.robot_ids, matrix.tasks, values,
                          matrix.feasible, raw + penalties)
    pairs = pair_features.copy()
    for column, task in enumerate(matrix.tasks):
        pickup = context.traffic_context.endpoint_load(task.pickup_position)
        delivery = context.traffic_context.endpoint_load(task.delivery_position)
        for row in range(len(matrix.robot_ids)):
            if matrix.feasible[row, column]:
                pairs[row, column, PAIR_FEATURE_NAMES.index("baseline_pair_cost")] = values[row, column]
                pairs[row, column, PAIR_FEATURE_NAMES.index("pickup_congestion")] = pickup
                pairs[row, column, PAIR_FEATURE_NAMES.index("delivery_congestion")] = delivery
    return adjusted, pairs, traffic, penalties


def traffic_graph_feature_tensor(matrix, pair_features, context):
    """39-feature graph contract; legacy pair features remain unchanged."""
    adjusted, pairs, traffic, _penalties = _traffic_adjusted_pairs(
        matrix, pair_features, context)
    result = np.concatenate((graph_edge_feature_tensor(adjusted, pairs), traffic), axis=2)
    if not np.all(np.isfinite(result)):
        raise ValueError("traffic graph features must be finite")
    return result


def _weighted_deadline_pressure(task, now):
    waiting = max(0.0, float(now) - float(task.arrival_time))
    deadline = task.max_completion_time_seconds
    pressure = min(1.0, waiting / float(deadline)) if deadline else 0.0
    return {1: 1.0, 2: 2.0, 3: 4.0}[int(task.priority)] * pressure


def traffic_pair_feature_tensor(matrix, pair_features, context):
    """Ridge inputs retain raw cost and expose nonlinear expert terms explicitly."""
    raw = matrix.unfloored_values
    if (raw is None or raw.shape != matrix.values.shape or
            not np.all(np.isfinite(raw[matrix.feasible]))):
        raise ValueError("traffic Ridge features require valid unfloored costs")
    _adjusted, pairs, traffic, penalties = _traffic_adjusted_pairs(
        matrix, pair_features, context)
    extra = np.zeros(matrix.values.shape + (4,), dtype=np.float64)
    for column, task in enumerate(matrix.tasks):
        mask = matrix.feasible[:, column]
        extra[mask, column, 0] = raw[mask, column]
        extra[mask, column, 1] = task.max_completion_time_seconds or 0.0
        extra[mask, column, 2] = _weighted_deadline_pressure(task, context.current_time)
        extra[mask, column, 3] = penalties[mask, column]
    result = np.concatenate((pairs, traffic, extra), axis=2)
    if not np.all(np.isfinite(result)):
        raise ValueError("traffic Ridge features must be finite")
    return result


def traffic_objective_expert_values(matrix, context, objective):
    """Preserve Count/Utility objectives, adding route-specific delay costs."""
    values = objective_expert_values(matrix, context, objective)
    _features, penalties = _traffic_edge_values(matrix, context)
    values[matrix.feasible] += penalties[matrix.feasible]
    return values


def traffic_ridge_supervision(matrix, features, context, objective):
    """Dense traffic cost labels from exactly the same snapshot as inference."""
    expected = matrix.values.shape + (len(TRAFFIC_PAIR_FEATURE_NAMES),)
    if features.shape != expected or not np.all(np.isfinite(features)):
        raise ValueError("traffic Ridge supervision feature contract mismatch")
    values = traffic_objective_expert_values(matrix, context, objective)
    return {
        "feature_version": TRAFFIC_PAIR_FEATURE_VERSION,
        "traffic_version": TRAFFIC_VERSION,
        "features": features[matrix.feasible].tolist(),
        "targets": values[matrix.feasible].tolist(),
        "sample_count": int(np.count_nonzero(matrix.feasible)),
    }


def traffic_graph_supervision(matrix, graph_features, context, objective,
                              behavior_coordinate=None):
    expected = matrix.values.shape + (len(TRAFFIC_GRAPH_FEATURE_NAMES),)
    if graph_features.shape != expected:
        raise ValueError("traffic graph supervision feature contract mismatch")
    costs = traffic_objective_expert_values(matrix, context, objective)
    positives = sorted(solve_hungarian(costs, matrix.feasible))
    positive_set = set(positives)
    negatives = [tuple(map(int, value)) for value in np.argwhere(matrix.feasible)
                 if tuple(map(int, value)) not in positive_set]
    negatives.sort(key=lambda value: (float(costs[value]), value))
    if behavior_coordinate in negatives:
        negatives.remove(behavior_coordinate)
        negatives.insert(0, behavior_coordinate)
    coordinates = positives + negatives[:max(4, 3 * len(positives))]
    return {
        "feature_version": TRAFFIC_GRAPH_FEATURE_VERSION,
        "traffic_version": TRAFFIC_VERSION,
        "features": [graph_features[value].tolist() for value in coordinates],
        "labels": [float(value in positive_set) for value in coordinates],
        "positive_count": len(positives),
        "negative_count": len(coordinates) - len(positives),
    }


def _validate_training_arrays(features, targets, feature_count: int):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != feature_count:
        raise ValueError(
            f"features must have shape (n, {feature_count})")
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("at least two aligned training samples are required")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("training data must be finite")
    return x, y


def _atomic_save_npz(path, **arrays) -> None:
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("model path must end with .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metadata_array(metadata: dict) -> np.ndarray:
    return np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)


def _load_archive(path) -> dict:
    if not path:
        raise ModelValidationError("model checkpoint path is required")
    source = Path(path)
    if not source.is_file():
        raise ModelValidationError(f"model checkpoint not found: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True)
                    for name in archive.files}
    except ModelValidationError:
        raise
    except Exception as exc:
        raise ModelValidationError(
            f"cannot read model checkpoint: {type(exc).__name__}") from exc


def _read_metadata(archive: dict) -> dict:
    if "metadata" not in archive:
        raise ModelValidationError("checkpoint metadata is missing")
    try:
        metadata = json.loads(str(archive["metadata"].item()))
    except Exception as exc:
        raise ModelValidationError("checkpoint metadata is invalid") from exc
    if not isinstance(metadata, dict):
        raise ModelValidationError("checkpoint metadata must be an object")
    return metadata


@dataclass(frozen=True)
class RidgeCostModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    training_samples: int
    target_name: str = "execution_time"

    @property
    def uses_traffic(self):
        return len(self.coefficients) == len(TRAFFIC_PAIR_FEATURE_NAMES)

    @property
    def feature_names(self):
        return TRAFFIC_PAIR_FEATURE_NAMES if self.uses_traffic else PAIR_FEATURE_NAMES

    @property
    def feature_version(self):
        return TRAFFIC_PAIR_FEATURE_VERSION if self.uses_traffic else FEATURE_VERSION

    @property
    def model_version(self):
        return TRAFFIC_RIDGE_MODEL_VERSION if self.uses_traffic else RIDGE_MODEL_VERSION

    @classmethod
    def fit(cls, features, targets, *, l2: float = 1.0,
            target_name: str = "execution_time"):
        supplied = np.asarray(features)
        if supplied.ndim != 2 or supplied.shape[1] not in (
                len(PAIR_FEATURE_NAMES), len(TRAFFIC_PAIR_FEATURE_NAMES)):
            raise ValueError("unsupported learned-cost training feature contract")
        x, y = _validate_training_arrays(features, targets, supplied.shape[1])
        if (supplied.shape[1] == len(TRAFFIC_PAIR_FEATURE_NAMES) and
                target_name not in {"count_expert_cost", "utility_v2_expert_cost"}):
            raise ValueError("traffic Ridge requires objective expert cost targets")
        if not math.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be finite and non-negative")
        supported_targets = {
            "execution_time", "baseline_pair_cost",
            "count_expert_cost", "utility_v2_expert_cost",
        }
        if target_name not in supported_targets:
            raise ValueError("unsupported learned-cost target")
        if (target_name in {"execution_time", "baseline_pair_cost"} and
                np.any(y < 0)):
            raise ValueError("distance/time cost targets must be non-negative")
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        design = np.column_stack((np.ones(x.shape[0]), (x - mean) / scale))
        regularizer = np.eye(design.shape[1], dtype=np.float64) * float(l2)
        regularizer[0, 0] = 0.0
        try:
            parameters = np.linalg.solve(
                design.T @ design + regularizer, design.T @ y)
        except np.linalg.LinAlgError:
            parameters = np.linalg.lstsq(
                design.T @ design + regularizer,
                design.T @ y, rcond=None)[0]
        if not np.all(np.isfinite(parameters)):
            raise ValueError("ridge fitting produced invalid parameters")
        return cls(mean, scale, parameters[1:], float(parameters[0]),
                   int(x.shape[0]), target_name)

    def predict(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("prediction feature dimension is incompatible")
        if not np.all(np.isfinite(x)):
            raise ValueError("prediction features must be finite")
        values = self.intercept + ((x - self.mean) / self.scale) @ self.coefficients
        if not np.all(np.isfinite(values)):
            raise ValueError("model produced non-finite predictions")
        if self.target_name in {"execution_time", "baseline_pair_cost"}:
            values = np.maximum(values, 0.0)
        return values[0] if single else values

    def save(self, path) -> None:
        metadata = {
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "environment_version": RL_ENVIRONMENT_VERSION,
            "feature_names": list(self.feature_names),
            "training_samples": int(self.training_samples),
            "target_name": self.target_name,
        }
        if self.uses_traffic:
            metadata["traffic_version"] = TRAFFIC_VERSION
        _atomic_save_npz(
            path, metadata=_metadata_array(metadata), mean=self.mean,
            scale=self.scale, coefficients=self.coefficients,
            intercept=np.asarray(self.intercept, dtype=np.float64))

    @classmethod
    def load(cls, path):
        archive = _load_archive(path)
        metadata = _read_metadata(archive)
        traffic = metadata.get("model_version") == TRAFFIC_RIDGE_MODEL_VERSION
        if metadata.get("model_version") not in (RIDGE_MODEL_VERSION, TRAFFIC_RIDGE_MODEL_VERSION):
            raise ModelValidationError("unsupported learned-cost model version")
        names = TRAFFIC_PAIR_FEATURE_NAMES if traffic else PAIR_FEATURE_NAMES
        version = TRAFFIC_PAIR_FEATURE_VERSION if traffic else FEATURE_VERSION
        if (metadata.get("environment_version") != RL_ENVIRONMENT_VERSION or
                metadata.get("feature_version") != version or
                tuple(metadata.get("feature_names", ())) != names or
                (traffic and metadata.get("traffic_version") != TRAFFIC_VERSION)):
            raise ModelValidationError("learned-cost feature contract mismatch")
        required = {"mean", "scale", "coefficients", "intercept"}
        if not required.issubset(archive):
            raise ModelValidationError("learned-cost parameters are incomplete")
        mean = np.asarray(archive["mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(archive["scale"], dtype=np.float64).reshape(-1)
        coefficients = np.asarray(
            archive["coefficients"], dtype=np.float64).reshape(-1)
        size = len(names)
        if mean.shape != (size,) or scale.shape != (size,) or \
                coefficients.shape != (size,):
            raise ModelValidationError("learned-cost parameter shape mismatch")
        intercept = _finite_number(archive["intercept"].item(), math.nan)
        if (not np.all(np.isfinite(mean)) or
                not np.all(np.isfinite(scale)) or np.any(scale <= 0) or
                not np.all(np.isfinite(coefficients)) or
                not math.isfinite(intercept)):
            raise ModelValidationError("learned-cost parameters are invalid")
        target_name = metadata.get("target_name", "execution_time")
        if target_name not in {
                "execution_time", "baseline_pair_cost",
                "count_expert_cost", "utility_v2_expert_cost"}:
            raise ModelValidationError("unsupported learned-cost target")
        if traffic and target_name not in {"count_expert_cost", "utility_v2_expert_cost"}:
            raise ModelValidationError("traffic Ridge requires objective expert cost targets")
        return cls(mean, scale, coefficients, intercept,
                   int(metadata.get("training_samples", 0)), target_name)


@dataclass(frozen=True)
class GraphEdgeImitationModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    training_samples: int

    @property
    def uses_traffic(self):
        return len(self.weights) == len(TRAFFIC_GRAPH_FEATURE_NAMES)

    @property
    def feature_names(self):
        return TRAFFIC_GRAPH_FEATURE_NAMES if self.uses_traffic else GRAPH_FEATURE_NAMES

    @property
    def model_version(self):
        return TRAFFIC_GRAPH_MODEL_VERSION if self.uses_traffic else GRAPH_MODEL_VERSION

    @classmethod
    def fit(cls, features, labels, *, epochs: int = 500,
            learning_rate: float = 0.05, l2: float = 1e-4):
        supplied = np.asarray(features)
        if supplied.ndim != 2 or supplied.shape[1] not in (
                len(GRAPH_FEATURE_NAMES), len(TRAFFIC_GRAPH_FEATURE_NAMES)):
            raise ValueError("unsupported graph training feature contract")
        x, y = _validate_training_arrays(
            features, labels, supplied.shape[1])
        if np.any((y != 0.0) & (y != 1.0)):
            raise ValueError("imitation labels must be binary")
        if not np.any(y == 1.0) or not np.any(y == 0.0):
            raise ValueError("imitation data needs positive and negative edges")
        if epochs < 1 or learning_rate <= 0 or l2 < 0:
            raise ValueError("invalid graph model training parameters")
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        normalized = (x - mean) / scale
        weights = np.zeros(normalized.shape[1], dtype=np.float64)
        positive_weight = max(1.0, float(np.count_nonzero(y == 0.0)) /
                              max(1, np.count_nonzero(y == 1.0)))
        sample_weights = np.where(y == 1.0, positive_weight, 1.0)
        intercept = 0.0
        for step in range(int(epochs)):
            logits = np.clip(normalized @ weights + intercept, -40.0, 40.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            error = (probabilities - y) * sample_weights
            rate = float(learning_rate) / math.sqrt(1.0 + step * 0.01)
            weights -= rate * (
                normalized.T @ error / np.sum(sample_weights) + l2 * weights)
            intercept -= rate * float(np.sum(error) / np.sum(sample_weights))
        if not np.all(np.isfinite(weights)) or not math.isfinite(intercept):
            raise ValueError("graph imitation fitting produced invalid parameters")
        return cls(mean, scale, weights, intercept, int(x.shape[0]))

    def predict_proba(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("graph prediction feature dimension is incompatible")
        if not np.all(np.isfinite(x)):
            raise ValueError("graph prediction features must be finite")
        logits = np.clip(
            ((x - self.mean) / self.scale) @ self.weights + self.intercept,
            -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("graph model produced non-finite probabilities")
        return probabilities[0] if single else probabilities

    def save(self, path) -> None:
        metadata = {
            "model_version": self.model_version,
            "feature_version": (TRAFFIC_GRAPH_FEATURE_VERSION
                                if self.uses_traffic else FEATURE_VERSION),
            "environment_version": RL_ENVIRONMENT_VERSION,
            "feature_names": list(self.feature_names),
            "training_samples": int(self.training_samples),
        }
        _atomic_save_npz(
            path, metadata=_metadata_array(metadata), mean=self.mean,
            scale=self.scale, weights=self.weights,
            intercept=np.asarray(self.intercept, dtype=np.float64))

    @classmethod
    def load(cls, path):
        archive = _load_archive(path)
        metadata = _read_metadata(archive)
        traffic = metadata.get("model_version") == TRAFFIC_GRAPH_MODEL_VERSION
        if metadata.get("model_version") not in (
                GRAPH_MODEL_VERSION, TRAFFIC_GRAPH_MODEL_VERSION):
            raise ModelValidationError("unsupported graph imitation model version")
        names = TRAFFIC_GRAPH_FEATURE_NAMES if traffic else GRAPH_FEATURE_NAMES
        version = TRAFFIC_GRAPH_FEATURE_VERSION if traffic else FEATURE_VERSION
        if (metadata.get("environment_version") != RL_ENVIRONMENT_VERSION or
                metadata.get("feature_version") != version or
                tuple(metadata.get("feature_names", ())) != names):
            raise ModelValidationError("graph feature contract mismatch")
        required = {"mean", "scale", "weights", "intercept"}
        if not required.issubset(archive):
            raise ModelValidationError("graph model parameters are incomplete")
        mean = np.asarray(archive["mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(archive["scale"], dtype=np.float64).reshape(-1)
        weights = np.asarray(archive["weights"], dtype=np.float64).reshape(-1)
        size = len(names)
        if mean.shape != (size,) or scale.shape != (size,) or \
                weights.shape != (size,):
            raise ModelValidationError("graph parameter shape mismatch")
        intercept = _finite_number(archive["intercept"].item(), math.nan)
        if (not np.all(np.isfinite(mean)) or
                not np.all(np.isfinite(scale)) or np.any(scale <= 0) or
                not np.all(np.isfinite(weights)) or
                not math.isfinite(intercept)):
            raise ModelValidationError("graph model parameters are invalid")
        return cls(mean, scale, weights, intercept,
                   int(metadata.get("training_samples", 0)))


def solve_hungarian(values: np.ndarray,
                    feasible: np.ndarray) -> List[Tuple[int, int]]:
    """Solve a rectangular matching without allowing infeasible edges."""
    values = np.asarray(values, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    if values.shape != feasible.shape or values.ndim != 2:
        raise ValueError("values and feasible mask must be aligned matrices")
    if values.size == 0 or not np.any(feasible):
        return []
    if not np.all(np.isfinite(values[feasible])):
        raise ValueError("feasible matching costs must be finite")
    finite = values[feasible]
    penalty = max(1.0, float(np.max(np.abs(finite)))) * (
        max(values.shape, default=1) + 1) * 1_000.0
    solver_values = np.where(feasible, values, penalty)
    rows, columns = solver_values.shape
    tie = (np.arange(rows)[:, None] * max(columns, 1) +
           np.arange(columns)[None, :]) * np.finfo(float).eps
    row_indices, column_indices = linear_sum_assignment(solver_values + tie)
    return [(int(row), int(column))
            for row, column in zip(row_indices, column_indices)
            if feasible[row, column]]


def objective_expert_values(matrix: CostMatrix,
                            context: SchedulingContext,
                            objective: str) -> np.ndarray:
    """Return a deterministic objective-specific imitation expert matrix.

    Count removes the priority/waiting bonuses from the shared operational
    cost so the expert focuses on short service cycles and throughput.
    Utility retains those bonuses and adds deadline pressure using the formal
    1/2/4 priority weights.  Feasibility always remains owned by the common
    path-cost matrix.
    """
    if objective not in {"count", "utility_v2"}:
        raise ValueError("expert objective must be count or utility_v2")
    values = np.asarray(matrix.values, dtype=np.float64).copy()
    if values.shape != matrix.feasible.shape:
        raise ValueError("expert matrix shape differs from feasibility mask")
    weights = {
        "priority": 1.0,
        "waiting": 0.01,
    }
    weights.update(context.configuration.get("cost_weights", {}))
    now = max(0.0, _finite_number(context.current_time))
    unfloored = matrix.unfloored_values
    if unfloored is None:
        raise ValueError(
            "objective expert requires the unfloored operational cost matrix")
    unfloored = np.asarray(unfloored, dtype=np.float64)
    if (unfloored.shape != values.shape or
            not np.all(np.isfinite(unfloored[matrix.feasible]))):
        raise ValueError("objective expert unfloored costs are invalid")
    # Both experts must start before the operational zero floor.  Utility then
    # retains the priority/waiting bonuses, while Count removes them below.
    values = unfloored.copy()
    for column, task in enumerate(matrix.tasks):
        waiting = max(0.0, now - float(task.arrival_time))
        priority_bonus = (
            float(weights["priority"]) *
            max(0.0, float(task.priority) - 1.0))
        waiting_bonus = float(weights["waiting"]) * waiting
        if objective == "count":
            # Undo priority/waiting terms on the pre-floor value.  Doing this
            # from matrix.values would make a long-waiting task artificially
            # expensive after the operational max(0, cost) lost information.
            values[:, column] += priority_bonus + waiting_bonus
            continue
        values[:, column] -= 2.0 * _weighted_deadline_pressure(task, now)
    if not np.all(np.isfinite(values[matrix.feasible])):
        raise ValueError("objective expert produced non-finite feasible costs")
    return values


def objective_supervision_batches(matrix: CostMatrix,
                                  pair_features: np.ndarray,
                                  graph_features: np.ndarray,
                                  context: SchedulingContext, *,
                                  behavior_coordinate: tuple[int, int] | None =
                                  None) -> dict:
    """Label one scheduler snapshot for both formal objectives.

    LearnedHungarian receives the complete feasible expert cost surface, not
    only the edge selected by the expert.  GraphImitation receives every
    expert-matching edge plus deterministic low-cost hard negatives.  Keeping
    this logic shared by standalone collection and Webots telemetry prevents
    the two training paths from silently learning different tasks.
    """
    expected_pair_shape = matrix.values.shape + (len(PAIR_FEATURE_NAMES),)
    expected_graph_shape = matrix.values.shape + (len(GRAPH_FEATURE_NAMES),)
    if pair_features.shape != expected_pair_shape:
        raise ValueError("pair supervision tensor has an invalid shape")
    if graph_features.shape != expected_graph_shape:
        raise ValueError("graph supervision tensor has an invalid shape")
    coordinates = [
        (int(row), int(column))
        for row, column in np.argwhere(matrix.feasible)]
    if not coordinates:
        raise ValueError("objective supervision has no feasible edges")

    ridge_by_objective = {}
    graph_by_objective = {}
    for objective in ("count", "utility_v2"):
        expert_values = objective_expert_values(matrix, context, objective)
        expert = set(solve_hungarian(expert_values, matrix.feasible))
        positives = sorted(expert)
        negatives = [
            coordinate for coordinate in coordinates
            if coordinate not in expert]
        negative_limit = max(4, 3 * len(positives))
        chosen_negatives = []
        if (behavior_coordinate is not None and
                behavior_coordinate in negatives):
            chosen_negatives.append(behavior_coordinate)
        for coordinate in sorted(
                negatives,
                key=lambda item: (
                    float(expert_values[item[0], item[1]]), item)):
            if coordinate not in chosen_negatives:
                chosen_negatives.append(coordinate)
            if len(chosen_negatives) >= negative_limit:
                break
        graph_samples = positives + chosen_negatives
        if not positives or not graph_samples:
            raise ValueError("objective expert produced no supervision samples")

        ridge_by_objective[objective] = {
            "features": [
                pair_features[row, column].tolist()
                for row, column in coordinates],
            "targets": [
                float(expert_values[row, column])
                for row, column in coordinates],
            "sample_count": len(coordinates),
        }
        graph_by_objective[objective] = {
            "features": [
                graph_features[row, column].tolist()
                for row, column in graph_samples],
            "labels": [
                float((row, column) in expert)
                for row, column in graph_samples],
            "positive_count": len(positives),
            "negative_count": len(chosen_negatives),
        }
    return {
        "ridge_by_objective": ridge_by_objective,
        "graph_by_objective": graph_by_objective,
    }


def learned_cost_values(model: RidgeCostModel, matrix: CostMatrix,
                        pair_features: np.ndarray,
                        context: SchedulingContext) -> np.ndarray:
    """Build the exact cost matrix consumed by LearnedHungarian."""
    learned_values = np.full(matrix.values.shape, np.inf, dtype=np.float64)
    predicted = model.predict(pair_features[matrix.feasible])
    priority_weight = _finite_number(
        context.configuration.get("learned_priority_weight", 1.0), 1.0)
    waiting_weight = _finite_number(
        context.configuration.get("learned_waiting_weight", 0.01), 0.01)
    for index, (row, column) in enumerate(np.argwhere(matrix.feasible)):
        task = matrix.tasks[int(column)]
        urgency = 0.0
        if model.target_name == "execution_time":
            urgency = (
                priority_weight * max(0.0, float(task.priority) - 1.0) +
                waiting_weight * max(
                    0.0, context.current_time - float(task.arrival_time)))
        value = float(predicted[index]) - urgency
        learned_values[row, column] = (
            max(0.0, value)
            if model.target_name in {"execution_time", "baseline_pair_cost"}
            else value)
    return learned_values


def graph_imitation_values(model: "GraphEdgeImitationModel",
                           matrix: CostMatrix,
                           graph_features: np.ndarray) -> tuple[np.ndarray,
                                                               np.ndarray]:
    """Build GraphImitation matching costs and its probability matrix."""
    probabilities = model.predict_proba(graph_features[matrix.feasible])
    learned_values = np.full(matrix.values.shape, np.inf, dtype=np.float64)
    probability_matrix = np.zeros(matrix.values.shape, dtype=np.float64)
    base_scale = max(1.0, float(np.max(matrix.values[matrix.feasible])))
    for index, (row, column) in enumerate(np.argwhere(matrix.feasible)):
        probability = min(
            1.0 - 1e-9, max(1e-9, float(probabilities[index])))
        probability_matrix[row, column] = probability
        learned_values[row, column] = (
            -math.log(probability) +
            1e-6 * float(matrix.values[row, column]) / base_scale)
    return learned_values, probability_matrix


def physical_supervision_snapshot(
        assignment: Assignment, robot_states: Dict[int, dict],
        pending_tasks: Sequence[TransportTask],
        context: SchedulingContext) -> dict:
    """Capture objective-isolated supervision from one physical decision.

    The snapshot is telemetry only: it is computed after a route was found but
    from the immutable pre-commit scheduler inputs.  It never changes the
    assignment or Webots motion command.  Each objective receives labels from
    its own deterministic expert, so behavior generated by another scheduler
    is useful state coverage without being mistaken for an expert action.
    """
    matrix = build_cost_matrix(list(pending_tasks), robot_states, context)
    pair_features = pair_feature_tensor(
        matrix, robot_states, pending_tasks, context)
    graph_features = graph_edge_feature_tensor(matrix, pair_features)
    try:
        selected_row = matrix.robot_ids.index(int(assignment.robot_id))
        selected_column = next(
            index for index, task in enumerate(matrix.tasks)
            if int(task.task_id) == int(assignment.task.task_id))
    except (ValueError, StopIteration) as exc:
        raise ValueError(
            "committed pair is absent from supervision snapshot") from exc
    if not matrix.feasible[selected_row, selected_column]:
        raise ValueError("committed pair is infeasible in supervision snapshot")

    batches = objective_supervision_batches(
        matrix, pair_features, graph_features, context,
        behavior_coordinate=(selected_row, selected_column))
    if isinstance(context.traffic_context, TrafficSnapshot):
        ridge_features = traffic_pair_feature_tensor(matrix, pair_features, context)
        batches["traffic_ridge_by_objective"] = {
            objective: traffic_ridge_supervision(matrix, ridge_features, context, objective)
            for objective in ("count", "utility_v2")}
        traffic_features = traffic_graph_feature_tensor(matrix, pair_features, context)
        batches["traffic_graph_by_objective"] = {
            objective: traffic_graph_supervision(
                matrix, traffic_features, context, objective,
                (selected_row, selected_column))
            for objective in ("count", "utility_v2")}
    return {
        "schema_version": 2,
        "dataset_version": WEBOTS_SHARED_SUPERVISION_VERSION,
        "environment_version": RL_ENVIRONMENT_VERSION,
        "feature_version": FEATURE_VERSION,
        "selected_pair_features":
            pair_features[selected_row, selected_column].tolist(),
        **batches,
    }


class _LearnedMatchingScheduler(BaseScheduler):
    """Shared structured adapter for learned full-matching schedulers."""

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        result = self.assign(
            pending_tasks, robot_states,
            SchedulingContext(congestion_map=congestion_map))
        if not result.assignments:
            return None
        assignment = result.assignments[0]
        return assignment.robot_id, assignment.task


class LearnedCostHungarianScheduler(_LearnedMatchingScheduler):
    """Predict pair execution cost, then solve an exact legal matching."""

    def __init__(self, model_path):
        super().__init__("LearnedHungarian")
        self.model_path = str(model_path) if model_path is not None else ""
        self.model = RidgeCostModel.load(model_path)

    @property
    def uses_traffic(self):
        return self.model.uses_traffic

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        context = context or SchedulingContext()
        started = time.perf_counter()
        try:
            base = build_cost_matrix(pending_tasks, robot_states, context)
        except ValueError as exc:
            return SchedulerResult(
                computation_time=time.perf_counter() - started,
                algorithm_name=self.name,
                diagnostics={"reason": str(exc)})
        if base.values.size == 0 or not np.any(base.feasible):
            return _result_from_matching(
                self.name, base, [], pending_tasks, robot_states, context,
                started, {"model_version": self.model.model_version})
        features = pair_feature_tensor(
            base, robot_states, pending_tasks, context)
        if self.uses_traffic:
            features = traffic_pair_feature_tensor(base, features, context)
        learned_values = learned_cost_values(
            self.model, base, features, context)
        learned = CostMatrix(
            base.robot_ids, base.tasks, learned_values, base.feasible.copy())
        pairs = solve_hungarian(learned.values, learned.feasible)
        physical_fine_tune = os.environ.get(
            "SMART_FACTORY_PHYSICAL_FINE_TUNE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if physical_fine_tune:
            pairs = pairs[:1]
        diagnostics = {
            "solver": "scipy.linear_sum_assignment",
            "model_version": self.model.model_version,
            "feature_version": self.model.feature_version,
            "traffic_version": TRAFFIC_VERSION if self.uses_traffic else None,
            "training_samples": self.model.training_samples,
            "target_name": self.model.target_name,
        }
        if physical_fine_tune and pairs:
            row, column = pairs[0]
            diagnostics.update({
                "physical_fine_tune": True,
                "physical_rollout_step": {
                    "kind": "learned_hungarian",
                    "features": features[row, column].tolist(),
                    "selected_robot_id": int(base.robot_ids[row]),
                    "selected_task_id": int(base.tasks[column].task_id),
                },
            })
        return _result_from_matching(
            self.name, learned, pairs, pending_tasks, robot_states, context,
            started, diagnostics)


class GraphImitationScheduler(_LearnedMatchingScheduler):
    """Score bipartite edges from graph context, then enforce exact matching."""

    def __init__(self, model_path):
        super().__init__("GraphImitation")
        self.model_path = str(model_path) if model_path is not None else ""
        self.model = GraphEdgeImitationModel.load(model_path)

    @property
    def uses_traffic(self):
        return self.model.uses_traffic

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        context = context or SchedulingContext()
        started = time.perf_counter()
        try:
            base = build_cost_matrix(pending_tasks, robot_states, context)
        except ValueError as exc:
            return SchedulerResult(
                computation_time=time.perf_counter() - started,
                algorithm_name=self.name,
                diagnostics={"reason": str(exc)})
        if base.values.size == 0 or not np.any(base.feasible):
            return _result_from_matching(
                self.name, base, [], pending_tasks, robot_states, context,
                started, {"model_version": self.model.model_version})
        pair_features = pair_feature_tensor(
            base, robot_states, pending_tasks, context)
        graph_features = (traffic_graph_feature_tensor(base, pair_features, context)
                          if self.uses_traffic else
                          graph_edge_feature_tensor(base, pair_features))
        learned_values, probability_matrix = graph_imitation_values(
            self.model, base, graph_features)
        learned = CostMatrix(
            base.robot_ids, base.tasks, learned_values, base.feasible.copy())
        pairs = solve_hungarian(learned.values, learned.feasible)
        physical_fine_tune = os.environ.get(
            "SMART_FACTORY_PHYSICAL_FINE_TUNE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if physical_fine_tune:
            pairs = pairs[:1]
        diagnostics = {
            "solver": "scipy.linear_sum_assignment",
            "model_version": self.model.model_version,
            "training_samples": self.model.training_samples,
            "mean_selected_probability": float(np.mean([
                probability_matrix[row, column]
                for row, column in pairs])) if pairs else 0.0,
        }
        if physical_fine_tune and pairs:
            selected_row, selected_column = pairs[0]
            objective = os.environ.get(
                "SMART_FACTORY_FINE_TUNE_OBJECTIVE", "utility_v2").strip()
            expert_values = (traffic_objective_expert_values(base, context, objective)
                             if self.uses_traffic else
                             objective_expert_values(base, context, objective))
            expert = set(solve_hungarian(expert_values, base.feasible))
            coordinates = [tuple(map(int, value)) for value in
                           np.argwhere(base.feasible)]
            positives = sorted(expert)[:2]
            negatives = sorted(
                value for value in coordinates if value not in expert)[:2]
            samples = positives + negatives
            diagnostics.update({
                "physical_fine_tune": True,
                "physical_rollout_step": {
                    "kind": "graph_imitation",
                    "training_features": [
                        graph_features[row, column].tolist()
                        for row, column in samples],
                    "training_labels": [
                        float((row, column) in expert)
                        for row, column in samples],
                    "selected_robot_id": int(
                        base.robot_ids[selected_row]),
                    "selected_task_id": int(
                        base.tasks[selected_column].task_id),
                },
            })
        return _result_from_matching(
            self.name, learned, pairs, pending_tasks, robot_states, context,
            started, diagnostics)


def attach_learning_trace(task: TransportTask, assignment: Assignment,
                          robot_states: Dict[int, dict],
                          pending_tasks: Sequence[TransportTask],
                          context: SchedulingContext,
                          algorithm_name: str) -> bool:
    """Attach an immutable-at-completion training snapshot after commit.

    Returning ``False`` is intentionally non-fatal: telemetry must never make
    an otherwise valid physical dispatch fail.
    """
    try:
        baseline_cost = _pair_cost(
            assignment.robot_id, task, robot_states, context)
        features = pair_feature_vector(
            assignment.robot_id, task, robot_states, pending_tasks, context,
            base_cost=baseline_cost,
            pending_count_override=len(pending_tasks))
        task.learning_trace = {
            "feature_version": FEATURE_VERSION,
            "environment_version": RL_ENVIRONMENT_VERSION,
            "feature_names": list(PAIR_FEATURE_NAMES),
            "features": [float(value) for value in features],
            "algorithm": str(algorithm_name),
            "robot_id": int(assignment.robot_id),
            "task_id": int(task.task_id),
            "baseline_pair_cost": float(baseline_cost),
            "scheduler_estimated_cost": (
                float(assignment.estimated_cost)
                if assignment.estimated_cost is not None else None),
            "captured_at": float(context.current_time),
        }
        return True
    except Exception:
        task.learning_trace = None
        return False


def completion_samples(documents: Iterable[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract compatible real execution-time samples from result documents."""
    features: List[List[float]] = []
    targets: List[float] = []
    for document in documents:
        for completion in document.get("task_completions", []):
            trace = completion.get("learning_trace") or {}
            target = completion.get("execution_time")
            if (trace.get("environment_version") != RL_ENVIRONMENT_VERSION or
                    trace.get("feature_version") != FEATURE_VERSION or
                    tuple(trace.get("feature_names", ())) != PAIR_FEATURE_NAMES):
                continue
            vector = trace.get("features")
            try:
                vector_array = np.asarray(vector, dtype=np.float64)
                target_value = float(target)
            except (TypeError, ValueError):
                continue
            if (vector_array.shape != (len(PAIR_FEATURE_NAMES),) or
                    not np.all(np.isfinite(vector_array)) or
                    not math.isfinite(target_value) or target_value < 0):
                continue
            features.append(vector_array.tolist())
            targets.append(target_value)
    return (np.asarray(features, dtype=np.float64).reshape(
                -1, len(PAIR_FEATURE_NAMES)),
            np.asarray(targets, dtype=np.float64))
