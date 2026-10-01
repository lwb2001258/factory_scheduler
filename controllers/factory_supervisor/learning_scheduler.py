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

from config import RobotState, TaskStatus
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


FEATURE_VERSION = "factory-pair-v1"
RIDGE_MODEL_VERSION = "learned-cost-ridge-v1"
GRAPH_MODEL_VERSION = "graph-edge-imitation-v1"

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
                        pending_count_override: Optional[int] = None
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
    idle_count = sum(
        1 for robot in robot_states.values()
        if robot.get("state") == RobotState.IDLE and
        robot.get("current_task") is None)
    pending_count = (sum(
        1 for item in pending_tasks if item.status == TaskStatus.PENDING)
        if pending_count_override is None else
        max(0, int(pending_count_override)))
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
    for row, robot_id in enumerate(matrix.robot_ids):
        for column, task in enumerate(matrix.tasks):
            if matrix.feasible[row, column]:
                tensor[row, column] = pair_feature_vector(
                    robot_id, task, robot_states, pending_tasks, context,
                    base_cost=float(matrix.values[row, column]))
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

    @classmethod
    def fit(cls, features, targets, *, l2: float = 1.0,
            target_name: str = "execution_time"):
        x, y = _validate_training_arrays(
            features, targets, len(PAIR_FEATURE_NAMES))
        if np.any(y < 0):
            raise ValueError("cost targets must be non-negative")
        if not math.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be finite and non-negative")
        if target_name not in {"execution_time", "baseline_pair_cost"}:
            raise ValueError("unsupported learned-cost target")
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
        if x.ndim != 2 or x.shape[1] != len(PAIR_FEATURE_NAMES):
            raise ValueError("prediction feature dimension is incompatible")
        if not np.all(np.isfinite(x)):
            raise ValueError("prediction features must be finite")
        values = self.intercept + ((x - self.mean) / self.scale) @ self.coefficients
        if not np.all(np.isfinite(values)):
            raise ValueError("model produced non-finite predictions")
        values = np.maximum(values, 0.0)
        return values[0] if single else values

    def save(self, path) -> None:
        metadata = {
            "model_version": RIDGE_MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_names": list(PAIR_FEATURE_NAMES),
            "training_samples": int(self.training_samples),
            "target_name": self.target_name,
        }
        _atomic_save_npz(
            path, metadata=_metadata_array(metadata), mean=self.mean,
            scale=self.scale, coefficients=self.coefficients,
            intercept=np.asarray(self.intercept, dtype=np.float64))

    @classmethod
    def load(cls, path):
        archive = _load_archive(path)
        metadata = _read_metadata(archive)
        if metadata.get("model_version") != RIDGE_MODEL_VERSION:
            raise ModelValidationError("unsupported learned-cost model version")
        if (metadata.get("feature_version") != FEATURE_VERSION or
                tuple(metadata.get("feature_names", ())) != PAIR_FEATURE_NAMES):
            raise ModelValidationError("learned-cost feature contract mismatch")
        required = {"mean", "scale", "coefficients", "intercept"}
        if not required.issubset(archive):
            raise ModelValidationError("learned-cost parameters are incomplete")
        mean = np.asarray(archive["mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(archive["scale"], dtype=np.float64).reshape(-1)
        coefficients = np.asarray(
            archive["coefficients"], dtype=np.float64).reshape(-1)
        size = len(PAIR_FEATURE_NAMES)
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
        if target_name not in {"execution_time", "baseline_pair_cost"}:
            raise ModelValidationError("unsupported learned-cost target")
        return cls(mean, scale, coefficients, intercept,
                   int(metadata.get("training_samples", 0)), target_name)


@dataclass(frozen=True)
class GraphEdgeImitationModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    training_samples: int

    @classmethod
    def fit(cls, features, labels, *, epochs: int = 500,
            learning_rate: float = 0.05, l2: float = 1e-4):
        x, y = _validate_training_arrays(
            features, labels, len(GRAPH_FEATURE_NAMES))
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
        if x.ndim != 2 or x.shape[1] != len(GRAPH_FEATURE_NAMES):
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
            "model_version": GRAPH_MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "feature_names": list(GRAPH_FEATURE_NAMES),
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
        if metadata.get("model_version") != GRAPH_MODEL_VERSION:
            raise ModelValidationError("unsupported graph imitation model version")
        if (metadata.get("feature_version") != FEATURE_VERSION or
                tuple(metadata.get("feature_names", ())) != GRAPH_FEATURE_NAMES):
            raise ModelValidationError("graph feature contract mismatch")
        required = {"mean", "scale", "weights", "intercept"}
        if not required.issubset(archive):
            raise ModelValidationError("graph model parameters are incomplete")
        mean = np.asarray(archive["mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(archive["scale"], dtype=np.float64).reshape(-1)
        weights = np.asarray(archive["weights"], dtype=np.float64).reshape(-1)
        size = len(GRAPH_FEATURE_NAMES)
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

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        context = context or SchedulingContext()
        started = time.perf_counter()
        base = build_cost_matrix(pending_tasks, robot_states, context)
        if base.values.size == 0 or not np.any(base.feasible):
            return _result_from_matching(
                self.name, base, [], pending_tasks, robot_states, context,
                started, {"model_version": RIDGE_MODEL_VERSION})
        features = pair_feature_tensor(
            base, robot_states, pending_tasks, context)
        learned_values = np.full(base.values.shape, np.inf, dtype=np.float64)
        predicted = self.model.predict(features[base.feasible])
        priority_weight = _finite_number(
            context.configuration.get("learned_priority_weight", 1.0), 1.0)
        waiting_weight = _finite_number(
            context.configuration.get("learned_waiting_weight", 0.01), 0.01)
        for index, (row, column) in enumerate(np.argwhere(base.feasible)):
            task = base.tasks[int(column)]
            urgency = 0.0
            if self.model.target_name == "execution_time":
                urgency = (
                    priority_weight * max(0.0, float(task.priority) - 1.0) +
                    waiting_weight * max(
                        0.0, context.current_time - float(task.arrival_time)))
            learned_values[row, column] = max(0.0, float(predicted[index]) - urgency)
        learned = CostMatrix(
            base.robot_ids, base.tasks, learned_values, base.feasible.copy())
        pairs = solve_hungarian(learned.values, learned.feasible)
        return _result_from_matching(
            self.name, learned, pairs, pending_tasks, robot_states, context,
            started, {
                "solver": "scipy.linear_sum_assignment",
                "model_version": RIDGE_MODEL_VERSION,
                "training_samples": self.model.training_samples,
                "target_name": self.model.target_name,
            })


class GraphImitationScheduler(_LearnedMatchingScheduler):
    """Score bipartite edges from graph context, then enforce exact matching."""

    def __init__(self, model_path):
        super().__init__("GraphImitation")
        self.model_path = str(model_path) if model_path is not None else ""
        self.model = GraphEdgeImitationModel.load(model_path)

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None) -> SchedulerResult:
        context = context or SchedulingContext()
        started = time.perf_counter()
        base = build_cost_matrix(pending_tasks, robot_states, context)
        if base.values.size == 0 or not np.any(base.feasible):
            return _result_from_matching(
                self.name, base, [], pending_tasks, robot_states, context,
                started, {"model_version": GRAPH_MODEL_VERSION})
        pair_features = pair_feature_tensor(
            base, robot_states, pending_tasks, context)
        graph_features = graph_edge_feature_tensor(base, pair_features)
        probabilities = self.model.predict_proba(
            graph_features[base.feasible])
        learned_values = np.full(base.values.shape, np.inf, dtype=np.float64)
        probability_matrix = np.zeros(base.values.shape, dtype=np.float64)
        base_scale = max(1.0, float(np.max(base.values[base.feasible])))
        for index, (row, column) in enumerate(np.argwhere(base.feasible)):
            probability = min(1.0 - 1e-9, max(1e-9,
                float(probabilities[index])))
            probability_matrix[row, column] = probability
            # The base-cost term is only a deterministic tie-breaker. The
            # learned expert probability owns the primary edge ordering.
            learned_values[row, column] = (
                -math.log(probability) +
                1e-6 * float(base.values[row, column]) / base_scale)
        learned = CostMatrix(
            base.robot_ids, base.tasks, learned_values, base.feasible.copy())
        pairs = solve_hungarian(learned.values, learned.feasible)
        return _result_from_matching(
            self.name, learned, pairs, pending_tasks, robot_states, context,
            started, {
                "solver": "scipy.linear_sum_assignment",
                "model_version": GRAPH_MODEL_VERSION,
                "training_samples": self.model.training_samples,
                "mean_selected_probability": float(np.mean([
                    probability_matrix[row, column]
                    for row, column in pairs])) if pairs else 0.0,
            })


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
            if (trace.get("feature_version") != FEATURE_VERSION or
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
