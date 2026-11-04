"""Contracts for learned traffic-expert coefficients and their provenance."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/factory_supervisor"))
sys.path.insert(0, str(ROOT / "scripts"))

from config import RobotState
from expert_coefficients import (
    ExpertCostCoefficients,
    bundle_sha256,
    canonical_bundle,
    default_bundle,
    row_from_baseline_log_multipliers,
    row_from_log_multipliers,
    with_optimized_rows,
)
from graph_expert_scheduler import (
    OPTIMIZED_EXPERT_VERSION,
    TrafficHungarianScheduler,
    expert_descriptor,
)
from learning_scheduler import (
    GraphEdgeImitationModel,
    RidgeCostModel,
    physical_supervision_snapshot,
    traffic_objective_expert_values,
)
from schedulers import (
    Assignment,
    ModelValidationError,
    SchedulingContext,
    build_cost_matrix,
)
from task_generator import TransportTask
from traffic_features import MotionHistory, TrafficRobot, TrafficSnapshot
import run_strict_1800_workflow as strict


def coefficient_case():
    robots = {
        1: dict(position=(0.2, 0.2), state=RobotState.IDLE,
                current_task=None, battery=100.0),
        2: dict(position=(1.2, 0.2), state=RobotState.EN_ROUTE_PICKUP,
                current_task=object(), battery=100.0),
    }
    tasks = [
        TransportTask(1, "S1", "WS1", (2.2, 0.2), (2.2, 1.2), 0,
                      priority=3, max_completion_time_seconds=250),
        TransportTask(2, "S2", "WS2", (0.2, 3.2), (1.2, 3.2), 0,
                      priority=1),
    ]
    peer_history = MotionHistory(wait_seconds=7.0, stop_events=2)
    snapshot = TrafficSnapshot(120.0, [
        TrafficRobot(1, robots[1]["position"], (), 120.0, 0.22,
                     MotionHistory()),
        TrafficRobot(2, robots[2]["position"],
                     ((1.2, 0.2), (2.2, 0.2), (2.2, 1.2)),
                     120.0, 0.22, peer_history),
    ], lambda start, end: (start, end))
    context = SchedulingContext(
        current_time=120.0,
        traffic_context=snapshot,
        congestion_map=[[2.0] * 8 for _ in range(8)],
        configuration={"congestion_grid_resolution": 1.0},
    )
    return robots, tasks, context, build_cost_matrix(tasks, robots, context)


def optimized_bundle():
    base = default_bundle()
    rows = dict(base["objectives"])
    rows["count"] = row_from_log_multipliers(
        "count", [0.2, 0.4, -0.1, 0.3])
    rows["utility_v2"] = row_from_log_multipliers(
        "utility_v2", [0.1, -0.2, 0.3, 0.2, 0.4, -0.1, 0.3])
    return with_optimized_rows(rows, method="separable-cma-es-v1")


class ExpertCoefficientTests(unittest.TestCase):
    def test_default_parameterization_exactly_reproduces_legacy_expert(self):
        _, _, context, matrix = coefficient_case()
        for objective in ("count", "utility_v2"):
            legacy = traffic_objective_expert_values(
                matrix, context, objective)
            parameterized = traffic_objective_expert_values(
                matrix, context, objective,
                coefficient_bundle=default_bundle())
            np.testing.assert_allclose(parameterized, legacy, rtol=0, atol=1e-12)

    def test_optimized_coefficients_change_ranking_costs_not_feasibility(self):
        _, _, context, matrix = coefficient_case()
        before = traffic_objective_expert_values(matrix, context, "count")
        after = traffic_objective_expert_values(
            matrix, context, "count",
            coefficient_bundle=optimized_bundle())
        self.assertFalse(np.allclose(before[matrix.feasible], after[matrix.feasible]))
        np.testing.assert_array_equal(matrix.feasible, np.isfinite(after))

    def test_fixed_travel_anchor_is_independent_of_operational_multiplier(self):
        robots, tasks, context, baseline_matrix = coefficient_case()
        doubled = replace(context, configuration={
            **context.configuration,
            "cost_weights": {"travel": 2.0},
        })
        doubled_matrix = build_cost_matrix(tasks, robots, doubled)
        for objective in ("count", "utility_v2"):
            baseline = traffic_objective_expert_values(
                baseline_matrix, context, objective,
                coefficient_bundle=default_bundle())
            anchored = traffic_objective_expert_values(
                doubled_matrix, doubled, objective,
                coefficient_bundle=default_bundle())
            np.testing.assert_allclose(anchored, baseline, rtol=0, atol=1e-12)

    def test_scale_anchor_objective_isolation_and_schema_are_fail_closed(self):
        with self.assertRaises(ValueError):
            ExpertCostCoefficients(objective="count", travel=1.01)
        with self.assertRaises(ValueError):
            ExpertCostCoefficients(objective="count", priority_bonus=0.1)
        with self.assertRaises(ValueError):
            ExpertCostCoefficients(objective="utility_v2", waiting_bonus=-0.1)
        bad = default_bundle()
        bad["objectives"]["count"]["unknown"] = 1.0
        with self.assertRaises(ValueError):
            canonical_bundle(bad)

    def test_bundle_hash_is_canonical_and_changes_with_coefficients(self):
        baseline = default_bundle()
        reordered = json.loads(json.dumps(baseline, sort_keys=True))
        self.assertEqual(bundle_sha256(baseline), bundle_sha256(reordered))
        self.assertNotEqual(bundle_sha256(baseline),
                            bundle_sha256(optimized_bundle()))

    def test_warm_start_multipliers_preserve_identity_and_custom_bounds(self):
        baseline = optimized_bundle()["objectives"]["utility_v2"]
        unchanged = row_from_baseline_log_multipliers(
            "utility_v2", baseline, np.zeros(7),
            minimum_multiplier=0.25, maximum_multiplier=4.0)
        self.assertEqual(baseline, unchanged)
        bounded = row_from_baseline_log_multipliers(
            "utility_v2", baseline, np.full(7, 100.0),
            minimum_multiplier=0.25, maximum_multiplier=4.0)
        for name in (
                "priority_bonus", "waiting_bonus", "deadline_pressure_bonus",
                "congestion_penalty", "traffic_expected_wait_penalty",
                "traffic_recent_wait_penalty", "traffic_stop_event_penalty"):
            self.assertAlmostEqual(4.0 * baseline[name], bounded[name])
        wrong = dict(baseline, objective="count")
        with self.assertRaises(ValueError):
            row_from_baseline_log_multipliers(
                "utility_v2", wrong, np.zeros(7))


class CoefficientProvenanceTests(unittest.TestCase):
    def _tamper_metadata(self, path):
        with np.load(path, allow_pickle=False) as archive:
            data = {name: np.array(archive[name], copy=True)
                    for name in archive.files}
        metadata = json.loads(str(data["metadata"].item()))
        metadata["expert_coefficient_sha256"] = "0" * 64
        data["metadata"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(path, **data)

    def test_ridge_and_graph_checkpoints_bind_exact_bundle(self):
        bundle = optimized_bundle()
        models = (
            RidgeCostModel.fit(
                np.zeros((2, 33)), [1.0, 2.0],
                target_name="count_expert_cost",
                expert_coefficient_bundle=bundle),
            GraphEdgeImitationModel.fit(
                np.vstack((np.zeros(39), np.ones(39))), [0, 1], epochs=2,
                expert_coefficient_bundle=bundle),
        )
        for model, model_type in zip(
                models, (RidgeCostModel, GraphEdgeImitationModel)):
            with self.subTest(model=model_type.__name__), \
                    tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.npz"
                model.save(path)
                loaded = model_type.load(path)
                self.assertEqual(bundle_sha256(bundle), bundle_sha256(
                    loaded.expert_coefficient_bundle))
                self._tamper_metadata(path)
                with self.assertRaises(ModelValidationError):
                    model_type.load(path)

    def test_bundle_cannot_be_attached_to_legacy_features_or_downgraded(self):
        bundle = optimized_bundle()
        with self.assertRaises(ValueError):
            RidgeCostModel.fit(
                np.zeros((2, 20)), [1.0, 2.0],
                expert_coefficient_bundle=bundle)
        with self.assertRaises(ValueError):
            GraphEdgeImitationModel.fit(
                np.vstack((np.zeros(30), np.ones(30))), [0, 1], epochs=2,
                expert_coefficient_bundle=bundle)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model = RidgeCostModel.fit(
                np.zeros((2, 33)), [1.0, 2.0],
                target_name="count_expert_cost",
                expert_coefficient_bundle=bundle)
            model.save(path)
            with np.load(path, allow_pickle=False) as archive:
                data = {name: np.array(archive[name], copy=True)
                        for name in archive.files}
            metadata = json.loads(str(data["metadata"].item()))
            del metadata["expert_coefficient_bundle"]
            data["metadata"] = np.asarray(json.dumps(metadata))
            np.savez_compressed(path, **data)
            with self.assertRaises(ModelValidationError):
                RidgeCostModel.load(path)

    def test_optimized_hungarian_descriptor_is_canonical_and_immutable(self):
        bundle = optimized_bundle()
        descriptor = expert_descriptor("count", bundle)
        self.assertEqual(OPTIMIZED_EXPERT_VERSION, descriptor["version"])
        self.assertEqual(bundle_sha256(bundle), descriptor["coefficient_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expert.json"
            path.write_text(json.dumps(descriptor), encoding="utf-8")
            scheduler = TrafficHungarianScheduler(path)
            self.assertEqual(bundle_sha256(bundle),
                             scheduler.evaluation_metadata()["coefficient_sha256"])
            bad = dict(descriptor, coefficient_sha256="0" * 64)
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ModelValidationError):
                TrafficHungarianScheduler(path)

    def test_physical_supervision_carries_same_bundle_for_both_students(self):
        robots, tasks, context, _ = coefficient_case()
        bundle = optimized_bundle()
        snapshot = physical_supervision_snapshot(
            Assignment(1, tasks[0]), robots, tasks, context,
            expert_coefficient_bundle=bundle)
        expected = bundle_sha256(bundle)
        for group in ("traffic_ridge_by_objective",
                      "traffic_graph_by_objective"):
            self.assertEqual({"count", "utility_v2"},
                             set(snapshot[group]))
            for objective in ("count", "utility_v2"):
                self.assertEqual(
                    expected,
                    snapshot[group][objective]["expert_coefficient_sha256"])

    def test_physical_refit_preserves_exact_coefficient_bundle(self):
        robots, tasks, context, _ = coefficient_case()
        bundle = optimized_bundle()
        snapshot = physical_supervision_snapshot(
            Assignment(1, tasks[0]), robots, tasks, context,
            expert_coefficient_bundle=bundle)
        envelope = {"result": {"formal_evaluation": {"physical_fine_tune": {
            "rollout": [{"shared_supervision": snapshot, "done": True}]
        }}}}
        for policy_id, model in (
                ("LearnedHungarian_count", RidgeCostModel.fit(
                    np.zeros((2, 33)), [1.0, 2.0],
                    target_name="count_expert_cost",
                    expert_coefficient_bundle=bundle)),
                ("GraphImitation_count", GraphEdgeImitationModel.fit(
                    np.vstack((np.zeros(39), np.ones(39))), [0, 1], epochs=2,
                    expert_coefficient_bundle=bundle))):
            with self.subTest(policy=policy_id), \
                    tempfile.TemporaryDirectory() as directory:
                spec = dict(next(item for item in strict.AI_SPECS
                                 if item["policy_id"] == policy_id))
                spec.update(
                    traffic_version="route-traffic-v1",
                    expert_coefficient_bundle=bundle,
                    training_parameters=strict.training_parameters(spec))
                source = Path(directory) / "source.npz"
                target = Path(directory) / "target.npz"
                strict.atomic_save_model(spec, model, source)
                strict.update_physical_model(
                    spec, source, target, [envelope], [], None)
                loaded = strict.load_model(spec, target, 0)
                self.assertEqual(
                    bundle_sha256(bundle),
                    bundle_sha256(loaded.expert_coefficient_bundle))


if __name__ == "__main__":
    unittest.main()
