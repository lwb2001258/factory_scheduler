"""Cost-prior identity and numerical semantics for graph RL."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/factory_supervisor"))
sys.path.insert(0, str(ROOT / "scripts"))

from config import RobotState
from cost_guided_graph import (
    COST_GUIDED_POLICY_SPECS,
    EXPECTED_BUNDLE_SHA256,
    CostPriorIdentity,
    cost_prior_matrices,
    robust_cost_normalize,
)
from learning_scheduler import traffic_objective_expert_values
from schedulers import SchedulingContext, build_cost_matrix
from task_generator import TransportTask
from traffic_features import MotionHistory, TrafficRobot, TrafficSnapshot
import cost_guided_graph_rl_workflow_contract as workflow


def fixture(now=120.0):
    robots = {
        0: dict(position=(0.2, 0.2), state=RobotState.IDLE,
                current_task=None, battery=100.0),
        1: dict(position=(1.2, 0.2), state=RobotState.IDLE,
                current_task=None, battery=100.0),
        2: dict(position=(2.2, 0.2), state=RobotState.EN_ROUTE_PICKUP,
                current_task=object(), battery=100.0),
    }
    tasks = [
        TransportTask(10, "S1", "WS1", (2.2, 0.2), (2.2, 1.2), 0,
                      priority=3, max_completion_time_seconds=250),
        TransportTask(11, "S2", "WS2", (0.2, 3.2), (1.2, 3.2), 0,
                      priority=1),
        TransportTask(12, "S3", "WS3", (3.2, 1.2), (3.2, 2.2), 0,
                      priority=2, max_completion_time_seconds=425),
    ]
    peer_history = MotionHistory(wait_seconds=7.0, stop_events=2)
    snapshot = TrafficSnapshot(now, [
        TrafficRobot(0, robots[0]["position"], (), now, 0.22,
                     MotionHistory()),
        TrafficRobot(1, robots[1]["position"], (), now, 0.22,
                     MotionHistory()),
        TrafficRobot(2, robots[2]["position"],
                     ((2.2, 0.2), (2.2, 1.2)), now, 0.22,
                     peer_history),
    ], lambda start, end: (start, end))
    context = SchedulingContext(
        current_time=now, traffic_context=snapshot,
        congestion_map=[[2.0] * 8 for _ in range(8)],
        configuration={"congestion_grid_resolution": 1.0,
                       "graph_traffic_enabled": True})
    return robots, tasks, context


def selected_bundles():
    value = json.loads(workflow.SOURCE_SELECTION.read_text(encoding="utf-8"))
    return value["policies"]


class CostGuidedGraphTests(unittest.TestCase):
    def test_policy_specs_match_frozen_workflow_contract(self):
        core = {
            policy: {
                "objective": row["objective"],
                "primary": row["primary"],
                "secondary": row["secondary"],
            }
            for policy, row in COST_GUIDED_POLICY_SPECS.items()}
        self.assertEqual(core, workflow.TEACHER_MAPPING)
        self.assertEqual(
            EXPECTED_BUNDLE_SHA256, workflow.EXPECTED_BUNDLE_SHA256)

    def test_identity_round_trip_is_canonical_and_tamper_evident(self):
        rows = selected_bundles()
        for policy_id in COST_GUIDED_POLICY_SPECS:
            identity = CostPriorIdentity.from_selected_bundles(policy_id, rows)
            restored = CostPriorIdentity.from_metadata(identity.to_metadata())
            self.assertEqual(identity, restored)
            self.assertEqual(identity.sha256, restored.sha256)
            tampered = identity.to_metadata()
            tampered["objective"] = (
                "utility_v2" if identity.objective == "count" else "count")
            with self.assertRaises(ValueError):
                CostPriorIdentity.from_metadata(tampered)
            tampered = identity.to_metadata()
            tampered["normalization"] = "different-normalization"
            with self.assertRaises(ValueError):
                CostPriorIdentity.from_metadata(tampered)

    def test_cost_matrices_exactly_reuse_both_canonical_experts(self):
        robots, tasks, context = fixture()
        matrix = build_cost_matrix(tasks, robots, context)
        rows = selected_bundles()
        for policy_id in COST_GUIDED_POLICY_SPECS:
            identity = CostPriorIdentity.from_selected_bundles(policy_id, rows)
            priors = cost_prior_matrices(matrix, context, identity)
            expected_primary = traffic_objective_expert_values(
                matrix, context, identity.objective,
                coefficient_bundle=identity.primary_bundle)
            expected_secondary = traffic_objective_expert_values(
                matrix, context, identity.objective,
                coefficient_bundle=identity.secondary_bundle)
            np.testing.assert_allclose(
                priors.primary_cost[matrix.feasible],
                expected_primary[matrix.feasible], rtol=0, atol=0)
            np.testing.assert_allclose(
                priors.secondary_cost[matrix.feasible],
                expected_secondary[matrix.feasible], rtol=0, atol=0)
            self.assertEqual(priors.identity_sha256, identity.sha256)
            self.assertTrue(np.all(np.isfinite(
                priors.primary_normalized[matrix.feasible])))
            with self.assertRaises(ValueError):
                priors.primary_cost[0, 0] = 123.0

    def test_stale_or_missing_traffic_snapshot_fails_closed(self):
        robots, tasks, context = fixture()
        matrix = build_cost_matrix(tasks, robots, context)
        identity = CostPriorIdentity.from_selected_bundles(
            "CostGraphPPO_count", selected_bundles())
        for invalid in (
                replace(context, traffic_context=None),
                replace(context, current_time=context.current_time + 1.0)):
            with self.assertRaises(ValueError):
                cost_prior_matrices(matrix, invalid, identity)

    def test_robust_normalization_is_order_preserving_and_mask_neutral(self):
        values = np.asarray([[1.0, 1.0, 1000.0], [5.0, -4.0, 2.0]])
        mask = np.asarray([[True, True, True], [False, False, True]])
        normalized = robust_cost_normalize(values, mask)
        legal = normalized[mask]
        raw = values[mask]
        for left in range(len(raw)):
            for right in range(len(raw)):
                self.assertEqual(
                    int(np.sign(raw[left] - raw[right])),
                    int(np.sign(legal[left] - legal[right])))
        self.assertTrue(np.all(normalized[~mask] == 0.0))
        self.assertLessEqual(float(np.max(np.abs(legal))), 8.0)
        with self.assertRaisesRegex(ValueError, "statistics are non-finite"):
            robust_cost_normalize(
                np.asarray([[1e308, -1e308]]),
                np.asarray([[True, True]]))


if __name__ == "__main__":
    unittest.main()
