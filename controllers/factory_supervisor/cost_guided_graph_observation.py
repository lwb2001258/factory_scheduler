"""Variable-edge observation contract for cost-guided graph RL."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from cost_guided_graph import (
    CostPriorIdentity, CostPriorMatrices, cost_prior_matrices,
)
from learning_scheduler import (
    TRAFFIC_GRAPH_FEATURE_NAMES, TRAFFIC_GRAPH_FEATURE_VERSION,
    pair_feature_tensor, solve_hungarian, traffic_graph_feature_tensor,
)
from schedulers import CostMatrix, SchedulingContext, build_cost_matrix


COST_GUIDED_GRAPH_FEATURE_VERSION = (
    "cost-guided-graph-observation-v1-variable-edge-noop")
COST_PRIOR_EDGE_FEATURE_NAMES = (
    "primary_cost_raw",
    "secondary_cost_raw",
    "primary_cost_normalized",
    "secondary_cost_normalized",
    "normalized_secondary_minus_primary",
    "raw_secondary_minus_primary",
    "primary_minus_row_min",
    "primary_minus_column_min",
    "secondary_minus_row_min",
    "secondary_minus_column_min",
    "primary_global_rank",
    "secondary_global_rank",
    "primary_row_rank",
    "primary_column_rank",
    "secondary_row_rank",
    "secondary_column_rank",
    "primary_teacher_matching",
    "secondary_teacher_matching",
)
COST_GUIDED_GRAPH_FEATURE_NAMES = (
    tuple(TRAFFIC_GRAPH_FEATURE_NAMES)
    + COST_PRIOR_EDGE_FEATURE_NAMES
    + ("action_is_noop",))
COST_GUIDED_GRAPH_FEATURE_DIM = len(COST_GUIDED_GRAPH_FEATURE_NAMES)


def _schema_sha256() -> str:
    value = {
        "version": COST_GUIDED_GRAPH_FEATURE_VERSION,
        "traffic_feature_version": TRAFFIC_GRAPH_FEATURE_VERSION,
        "feature_names": COST_GUIDED_GRAPH_FEATURE_NAMES,
    }
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


COST_GUIDED_GRAPH_SCHEMA_SHA256 = _schema_sha256()


def _rank_fraction(values: np.ndarray, value: float) -> float:
    if values.size <= 1:
        return 0.0
    less = int(np.count_nonzero(values < value))
    equal = int(np.count_nonzero(values == value))
    return (less + max(0, equal - 1) * 0.5) / (values.size - 1)


def cost_prior_edge_feature_tensor(
        matrix: CostMatrix, priors: CostPriorMatrices) -> np.ndarray:
    """Build deterministic neighbourhood features from both raw priors."""
    shape = matrix.values.shape
    mask = np.asarray(matrix.feasible, dtype=bool)
    if (priors.feasible.shape != shape or
            not np.array_equal(priors.feasible, mask)):
        raise ValueError("cost-prior features use a different feasibility mask")
    output = np.zeros(
        shape + (len(COST_PRIOR_EDGE_FEATURE_NAMES),), dtype=np.float64)
    if not np.any(mask):
        return output
    primary = priors.primary_cost
    secondary = priors.secondary_cost
    primary_global = primary[mask]
    secondary_global = secondary[mask]
    primary_matching = set(solve_hungarian(primary, mask))
    secondary_matching = set(solve_hungarian(secondary, mask))
    for row, column in np.argwhere(mask):
        row = int(row)
        column = int(column)
        primary_row = primary[row, mask[row]]
        secondary_row = secondary[row, mask[row]]
        primary_column = primary[mask[:, column], column]
        secondary_column = secondary[mask[:, column], column]
        p_value = float(primary[row, column])
        s_value = float(secondary[row, column])
        p_normalized = float(priors.primary_normalized[row, column])
        s_normalized = float(priors.secondary_normalized[row, column])
        output[row, column] = np.asarray((
            p_value,
            s_value,
            p_normalized,
            s_normalized,
            s_normalized - p_normalized,
            s_value - p_value,
            p_value - float(np.min(primary_row)),
            p_value - float(np.min(primary_column)),
            s_value - float(np.min(secondary_row)),
            s_value - float(np.min(secondary_column)),
            _rank_fraction(primary_global, p_value),
            _rank_fraction(secondary_global, s_value),
            _rank_fraction(primary_row, p_value),
            _rank_fraction(primary_column, p_value),
            _rank_fraction(secondary_row, s_value),
            _rank_fraction(secondary_column, s_value),
            float((row, column) in primary_matching),
            float((row, column) in secondary_matching),
        ), dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise ValueError("cost-prior edge features contain non-finite values")
    return output


@dataclass(frozen=True)
class CostGuidedGraphObservation:
    """One immutable decision snapshot and its variable action rows."""

    matrix: CostMatrix
    coordinates: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray
    no_op_index: int
    prior_identity_sha256: str
    feature_schema_sha256: str
    current_time: float

    def __post_init__(self):
        values = np.asarray(self.matrix.values, dtype=np.float64).copy()
        feasible = np.asarray(self.matrix.feasible, dtype=bool).copy()
        unfloored = (None if self.matrix.unfloored_values is None else
                     np.asarray(
                         self.matrix.unfloored_values,
                         dtype=np.float64).copy())
        if (values.shape != feasible.shape or
                (unfloored is not None and unfloored.shape != values.shape)):
            raise ValueError("graph cost matrix is invalid")
        values.setflags(write=False)
        feasible.setflags(write=False)
        if unfloored is not None:
            unfloored.setflags(write=False)
        matrix = CostMatrix(
            tuple(self.matrix.robot_ids), tuple(self.matrix.tasks),
            values, feasible, unfloored)
        coordinates = np.asarray(self.coordinates, dtype=np.int64).copy()
        features = np.asarray(self.action_features, dtype=np.float32).copy()
        mask = np.asarray(self.action_mask, dtype=bool).copy()
        edge_count = len(coordinates)
        if (coordinates.shape != (edge_count, 2) or
                features.shape != (
                    edge_count + 1, COST_GUIDED_GRAPH_FEATURE_DIM) or
                mask.shape != (edge_count + 1,) or
                self.no_op_index != edge_count or
                not np.all(np.isfinite(features)) or
                not np.all(mask[:edge_count]) or
                bool(mask[self.no_op_index]) != (edge_count == 0) or
                np.any(features[:edge_count, -1] != 0.0) or
                features[self.no_op_index, -1] != 1.0):
            raise ValueError("cost-guided graph observation is invalid")
        matrix_coordinates = np.argwhere(matrix.feasible)
        if not np.array_equal(coordinates, matrix_coordinates):
            raise ValueError("graph coordinates differ from matrix feasibility")
        try:
            valid_prior_hash = (
                isinstance(self.prior_identity_sha256, str) and
                len(self.prior_identity_sha256) == 64 and
                int(self.prior_identity_sha256, 16) >= 0)
        except ValueError:
            valid_prior_hash = False
        if (self.feature_schema_sha256 != COST_GUIDED_GRAPH_SCHEMA_SHA256 or
                not valid_prior_hash or
                not np.isfinite(float(self.current_time)) or
                float(self.current_time) < 0.0):
            raise ValueError("graph observation identity is invalid")
        coordinates.setflags(write=False)
        features.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "action_features", features)
        object.__setattr__(self, "action_mask", mask)

    @property
    def edge_count(self) -> int:
        return len(self.coordinates)

    @property
    def action_count(self) -> int:
        return len(self.action_mask)

    def coordinate_for_action(self, action_index: int):
        if (isinstance(action_index, bool) or
                not isinstance(action_index, (int, np.integer)) or
                not 0 <= int(action_index) < self.action_count):
            raise ValueError("graph action index is out of range")
        index = int(action_index)
        if not self.action_mask[index]:
            raise ValueError("graph action is masked")
        if index == self.no_op_index:
            return None
        row, column = self.coordinates[index]
        return int(row), int(column)


def build_cost_guided_graph_observation(
        robot_states, pending_tasks, context: SchedulingContext,
        identity: CostPriorIdentity) -> CostGuidedGraphObservation:
    """Build the shared matrix once and reuse it for every observation field."""
    if not isinstance(context, SchedulingContext):
        raise ValueError("graph observation requires a SchedulingContext")
    matrix = build_cost_matrix(pending_tasks, robot_states, context)
    pairs = pair_feature_tensor(
        matrix, robot_states, pending_tasks, context)
    traffic_graph = traffic_graph_feature_tensor(matrix, pairs, context)
    priors = cost_prior_matrices(matrix, context, identity)
    prior_features = cost_prior_edge_feature_tensor(matrix, priors)
    expected = matrix.values.shape + (len(TRAFFIC_GRAPH_FEATURE_NAMES),)
    if traffic_graph.shape != expected:
        raise ValueError("traffic graph feature shape is incompatible")
    edge_tensor = np.concatenate(
        (traffic_graph, prior_features), axis=2)
    coordinates = np.argwhere(matrix.feasible)
    edge_features = np.asarray(
        edge_tensor[matrix.feasible], dtype=np.float32)
    action_features = np.zeros(
        (len(coordinates) + 1, COST_GUIDED_GRAPH_FEATURE_DIM),
        dtype=np.float32)
    if len(coordinates):
        action_features[:-1, :-1] = edge_features
    action_features[-1, -1] = 1.0
    action_mask = np.ones(len(coordinates) + 1, dtype=bool)
    action_mask[-1] = len(coordinates) == 0
    return CostGuidedGraphObservation(
        matrix, coordinates, action_features, action_mask,
        len(coordinates), identity.sha256,
        COST_GUIDED_GRAPH_SCHEMA_SHA256, float(context.current_time))
