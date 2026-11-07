"""Variable-edge cost-guided observation contract."""

from dataclasses import replace
import sys
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/factory_supervisor"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import cost_guided_graph_observation as observation_module
from cost_guided_graph import CostPriorIdentity
from cost_guided_graph_observation import (
    COST_GUIDED_GRAPH_FEATURE_DIM,
    COST_GUIDED_GRAPH_FEATURE_NAMES,
    COST_GUIDED_GRAPH_SCHEMA_SHA256,
    COST_PRIOR_EDGE_FEATURE_NAMES,
    build_cost_guided_graph_observation,
)
from test_cost_guided_graph import fixture, selected_bundles


class CostGuidedGraphObservationTests(unittest.TestCase):
    def identity(self, policy="CostGraphPPO_count"):
        return CostPriorIdentity.from_selected_bundles(
            policy, selected_bundles())

    def test_builds_matrix_once_and_exposes_edges_plus_explicit_noop(self):
        robots, tasks, context = fixture()
        original = observation_module.build_cost_matrix
        with mock.patch.object(
                observation_module, "build_cost_matrix",
                wraps=original) as build:
            snapshot = build_cost_guided_graph_observation(
                robots, tasks, context, self.identity())
        self.assertEqual(build.call_count, 1)
        self.assertEqual(snapshot.edge_count, 6)
        self.assertEqual(snapshot.action_count, 7)
        self.assertEqual(
            snapshot.action_features.shape,
            (7, COST_GUIDED_GRAPH_FEATURE_DIM))
        self.assertTrue(np.all(snapshot.action_mask[:-1]))
        self.assertFalse(snapshot.action_mask[-1])
        self.assertEqual(snapshot.no_op_index, 6)
        self.assertTrue(np.all(snapshot.action_features[:-1, -1] == 0.0))
        self.assertEqual(snapshot.action_features[-1, -1], 1.0)
        self.assertEqual(
            snapshot.feature_schema_sha256,
            COST_GUIDED_GRAPH_SCHEMA_SHA256)

    def test_prior_features_align_with_coordinates_and_teacher_costs(self):
        robots, tasks, context = fixture()
        snapshot = build_cost_guided_graph_observation(
            robots, tasks, context, self.identity())
        start = len(COST_GUIDED_GRAPH_FEATURE_NAMES) - 1 - len(
            COST_PRIOR_EDGE_FEATURE_NAMES)
        primary_raw = start + COST_PRIOR_EDGE_FEATURE_NAMES.index(
            "primary_cost_raw")
        primary_normalized = start + COST_PRIOR_EDGE_FEATURE_NAMES.index(
            "primary_cost_normalized")
        primary_matching = start + COST_PRIOR_EDGE_FEATURE_NAMES.index(
            "primary_teacher_matching")
        self.assertTrue(np.all(np.isfinite(
            snapshot.action_features[:-1, primary_raw])))
        self.assertTrue(np.all(np.isfinite(
            snapshot.action_features[:-1, primary_normalized])))
        self.assertGreater(
            int(np.count_nonzero(
                snapshot.action_features[:-1, primary_matching])), 0)
        for index, (row, column) in enumerate(snapshot.coordinates):
            self.assertTrue(snapshot.matrix.feasible[row, column])
            self.assertNotEqual(
                snapshot.action_features[index, primary_raw], 0.0)

    def test_input_permutations_produce_identical_sorted_observation(self):
        robots, tasks, context = fixture()
        first = build_cost_guided_graph_observation(
            robots, tasks, context, self.identity("GraphRainbow_utility_v2"))
        reversed_robots = dict(reversed(list(robots.items())))
        second = build_cost_guided_graph_observation(
            reversed_robots, list(reversed(tasks)), context,
            self.identity("GraphRainbow_utility_v2"))
        self.assertEqual(first.matrix.robot_ids, second.matrix.robot_ids)
        self.assertEqual(
            tuple(task.task_id for task in first.matrix.tasks),
            tuple(task.task_id for task in second.matrix.tasks))
        np.testing.assert_array_equal(
            first.coordinates, second.coordinates)
        np.testing.assert_allclose(
            first.action_features, second.action_features, rtol=0, atol=0)

    def test_no_legal_edge_has_one_legal_noop_action(self):
        robots, tasks, context = fixture()
        blocked = replace(
            context, path_cost_provider=lambda *_: float("inf"))
        snapshot = build_cost_guided_graph_observation(
            robots, tasks, blocked, self.identity())
        self.assertEqual(snapshot.edge_count, 0)
        self.assertEqual(snapshot.action_count, 1)
        np.testing.assert_array_equal(snapshot.action_mask, [True])
        self.assertIsNone(snapshot.coordinate_for_action(0))

    def test_stale_traffic_and_mutation_fail_closed(self):
        robots, tasks, context = fixture()
        with self.assertRaises(ValueError):
            build_cost_guided_graph_observation(
                robots, tasks,
                replace(context, current_time=context.current_time + 1.0),
                self.identity())
        snapshot = build_cost_guided_graph_observation(
            robots, tasks, context, self.identity())
        for array in (
                snapshot.coordinates, snapshot.action_features,
                snapshot.action_mask, snapshot.matrix.values,
                snapshot.matrix.feasible,
                snapshot.matrix.unfloored_values):
            with self.assertRaises(ValueError):
                array.flat[0] = 0
        with self.assertRaisesRegex(ValueError, "masked"):
            snapshot.coordinate_for_action(snapshot.no_op_index)


if __name__ == "__main__":
    unittest.main()
