"""Neural SARSA observations with canonical GraphImitation traffic costs."""
from __future__ import annotations

import numpy as np

from schedulers import build_cost_matrix
from sarsa_features import (
    ACTION_DIM, GLOBAL_FIELDS, INDEX as V3_INDEX, MAX_ROBOTS, MAX_TASKS,
    ROBOT_FIELDS, TASK_FIELDS, TRAFFIC_SCALE, WAIT_ACTION, ActionSnapshot,
    FEATURE_DIM as V3_FEATURE_DIM, FEATURE_NAMES as V3_FEATURE_NAMES,
    build_snapshot as build_v3_snapshot, select_candidates)
from traffic_features import TRAFFIC_FEATURE_NAMES, evaluate_traffic_edge


FEATURE_VERSION = "sarsa-pair-context-v4-graph-traffic"
CONTEXT_DIM = (len(GLOBAL_FIELDS) + MAX_ROBOTS * len(ROBOT_FIELDS) +
               MAX_TASKS * len(TASK_FIELDS))
TRAFFIC_ALIGNMENT_FIELDS = (
    "traffic_expert_penalty_distance",
    "traffic_adjusted_baseline_cost",
    "traffic_pickup_endpoint_load",
    "traffic_delivery_endpoint_load",
)
TRAFFIC_ALIGNMENT_SCALE = np.asarray([60.0, 60.0, 8.0, 8.0])
FEATURE_NAMES = V3_FEATURE_NAMES + TRAFFIC_ALIGNMENT_FIELDS
FEATURE_DIM = len(FEATURE_NAMES)
ACTION_FEATURE_DIM = FEATURE_DIM - CONTEXT_DIM
INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def _matrix_coordinates(matrix):
    robots = {robot_id: row for row, robot_id in enumerate(matrix.robot_ids)}
    tasks = {task.task_id: column for column, task in enumerate(matrix.tasks)}
    return robots, tasks


def build_snapshot(robots, tasks, context, *, allow_wait=False):
    """Extend the stable v3 observation without changing its checkpoint ABI."""
    base = build_v3_snapshot(
        robots, tasks, context, allow_wait=allow_wait)
    matrix = build_cost_matrix(list(base.tasks), robots, context)
    robot_rows, task_columns = _matrix_coordinates(matrix)
    raw = (matrix.unfloored_values if matrix.unfloored_values is not None
           else matrix.values)
    features = np.zeros((ACTION_DIM, FEATURE_DIM), dtype=np.float64)
    features[:, :V3_FEATURE_DIM] = base.features
    for action in np.flatnonzero(base.mask[:WAIT_ACTION]):
        robot_slot, task_slot = divmod(int(action), MAX_TASKS)
        robot_id = base.robot_ids[robot_slot]
        task = base.tasks[task_slot]
        row = robot_rows[robot_id]
        column = task_columns[task.task_id]
        edge = evaluate_traffic_edge(
            context.traffic_context, robot_id, task.pickup_position,
            task.delivery_position, raw[row, column])
        existing = np.asarray([
            base.features[action, V3_INDEX[name]]
            for name in TRAFFIC_FEATURE_NAMES])
        expected = np.clip(
            np.asarray(edge.features, dtype=np.float64) / TRAFFIC_SCALE,
            -5.0, 5.0)
        if not np.allclose(existing, expected, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "SARSA and GraphImitation traffic features diverged")
        aligned = np.asarray([
            edge.expert_penalty, edge.adjusted_cost,
            edge.pickup_load, edge.delivery_load], dtype=np.float64)
        features[action, V3_FEATURE_DIM:] = aligned / TRAFFIC_ALIGNMENT_SCALE
    if not np.isfinite(features).all():
        raise ValueError("nonfinite neural SARSA feature tensor")
    features = np.clip(features, -5.0, 5.0)
    features.setflags(write=False)
    return ActionSnapshot(
        base.time, base.robot_ids, base.tasks, features, base.mask,
        base.route_costs, base.pending_count)


__all__ = (
    "ACTION_DIM", "ACTION_FEATURE_DIM", "CONTEXT_DIM", "FEATURE_DIM",
    "FEATURE_NAMES", "FEATURE_VERSION", "INDEX", "MAX_ROBOTS",
    "MAX_TASKS", "TRAFFIC_ALIGNMENT_FIELDS", "WAIT_ACTION",
    "build_snapshot", "select_candidates")
