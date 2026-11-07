"""GraphRainbow n-step, PER, C51 and formal training contracts."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/factory_supervisor"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from cost_guided_graph import CostPriorIdentity
from cost_guided_graph_observation import (
    COST_GUIDED_GRAPH_FEATURE_DIM, COST_GUIDED_GRAPH_SCHEMA_SHA256,
    CostGuidedGraphObservation,
    build_cost_guided_graph_observation,
)
from dual_objective import CountRewardProfile
from graph_rainbow_model import (
    GraphRainbowConfig, GraphRainbowModel, GraphRainbowProvenance,
)
from graph_rainbow_training import (
    MAX_REPLAY_TASKS,
    GraphRainbowNStepAccumulator,
    GraphRainbowReplayBuffer,
    GraphRainbowTransition,
    c51_project,
    graph_rainbow_replay_identity,
    train_graph_rainbow,
    update_graph_rainbow,
)
from headless_training_runtime import HeadlessRuntimeConfig
from rl_environment import RLEnvironmentConfig
from schedulers import CostMatrix
from test_cost_guided_graph import fixture, selected_bundles


def make_model(profile, *, seed=47, batch_size=2, n_step=3):
    identity = CostPriorIdentity.from_selected_bundles(
        "GraphRainbow_count", selected_bundles())
    provenance = GraphRainbowProvenance(
        "count", profile.version, profile.sha256, "standalone", (72000,))
    config = GraphRainbowConfig(
        hidden_size=16, atom_count=51, v_min=0.0, v_max=100.0,
        n_step=n_step, replay_capacity=64, batch_size=batch_size,
        target_update_interval=2)
    model = GraphRainbowModel(identity, provenance, config, seed)
    replay = GraphRainbowReplayBuffer(
        config.replay_capacity, alpha=config.per_alpha,
        priority_epsilon=config.priority_epsilon,
        prior_identity_sha256=identity.sha256,
        training_identity_sha256=graph_rainbow_replay_identity(model),
        seed=seed)
    return model, replay


def observation(model):
    robots, tasks, context = fixture()
    return build_cost_guided_graph_observation(
        robots, tasks, context, model.prior_identity)


class GraphRainbowTrainingTests(unittest.TestCase):
    def test_n_step_accumulates_discounted_rewards_and_flushes_boundary(self):
        profile = CountRewardProfile(horizon_seconds=60.0)
        model, _ = make_model(profile, n_step=3)
        obs = observation(model)
        accumulator = GraphRainbowNStepAccumulator(3, 0.9)
        self.assertEqual(accumulator.append(
            obs, 0, 1.0, obs, terminated=False,
            episode_boundary=False), [])
        self.assertEqual(accumulator.append(
            obs, 0, 2.0, obs, terminated=False,
            episode_boundary=False), [])
        emitted = accumulator.append(
            obs, 0, 3.0, obs, terminated=True,
            episode_boundary=True)
        self.assertEqual(len(emitted), 3)
        self.assertAlmostEqual(
            emitted[0].reward, 1.0 + 0.9 * 2.0 + 0.9 ** 2 * 3.0)
        self.assertAlmostEqual(emitted[0].discount, 0.9 ** 3)
        self.assertTrue(all(row.done for row in emitted))
        self.assertEqual(len(accumulator.queue), 0)

    def test_prioritized_replay_round_trip_restores_ragged_rows_and_rng(self):
        profile = CountRewardProfile(horizon_seconds=60.0)
        model, replay = make_model(profile)
        obs = observation(model)
        for reward in (1.0, 2.0, 3.0):
            replay.add(GraphRainbowTransition(
                obs, 0, reward, obs, False, 0.99))
        replay.update_priorities([0, 1, 2], [1.0, 4.0, 2.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            replay.save(path)
            expected = replay.sample(2, 0.5)
            restored = GraphRainbowReplayBuffer.load(
                path,
                expected_prior_identity_sha256=model.prior_identity.sha256,
                expected_training_identity_sha256=
                graph_rainbow_replay_identity(model))
            actual = restored.sample(2, 0.5)
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_allclose(expected[2], actual[2], rtol=0, atol=0)
        self.assertEqual(len(restored), 3)
        for left, right in zip(replay.data, restored.data):
            self.assertEqual(left.action_index, right.action_index)
            self.assertEqual(left.reward, right.reward)
            np.testing.assert_array_equal(
                left.observation.action_features,
                right.observation.action_features)

    def test_replay_round_trip_covers_formal_webots_task_columns(self):
        profile = CountRewardProfile(horizon_seconds=60.0)
        model, replay = make_model(profile, batch_size=1, n_step=1)
        column = 243
        self.assertLess(column, MAX_REPLAY_TASKS)
        feasible = np.zeros((1, column + 1), dtype=bool)
        feasible[0, column] = True
        features = np.zeros(
            (2, COST_GUIDED_GRAPH_FEATURE_DIM), dtype=np.float32)
        features[-1, -1] = 1.0
        wide = CostGuidedGraphObservation(
            CostMatrix(
                (0,), tuple(range(column + 1)),
                np.zeros(feasible.shape), feasible,
                np.zeros(feasible.shape)),
            np.asarray([[0, column]], dtype=np.int64), features,
            np.asarray([True, False]), 1,
            model.prior_identity.sha256,
            COST_GUIDED_GRAPH_SCHEMA_SHA256, 1000.0)
        replay.add(GraphRainbowTransition(
            wide, 0, 1.0, wide, True, model.config.gamma))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide_replay.npz"
            replay.save(path)
            restored = GraphRainbowReplayBuffer.load(
                path,
                expected_prior_identity_sha256=model.prior_identity.sha256,
                expected_training_identity_sha256=
                    graph_rainbow_replay_identity(model))
        self.assertEqual(
            int(restored.data[0].observation.coordinates[0, 1]), column)

    def test_c51_projection_is_normalized_for_terminal_and_bootstrap(self):
        profile = CountRewardProfile(horizon_seconds=60.0)
        model, _ = make_model(profile)
        obs = observation(model)
        terminal = GraphRainbowTransition(
            obs, 0, 7.25, obs, True, 0.99)
        continuing = GraphRainbowTransition(
            obs, 0, 1.0, obs, False, 0.99)
        for transition in (terminal, continuing):
            projected = c51_project(model, transition)
            self.assertEqual(projected.shape, (model.config.atom_count,))
            self.assertAlmostEqual(float(np.sum(projected)), 1.0, places=6)
            self.assertTrue(np.all(projected >= 0.0))
        terminal_mean = float(c51_project(
            model, terminal) @ model.config.support)
        self.assertAlmostEqual(terminal_mean, 7.25, places=5)

    def test_double_distributional_per_update_changes_online_and_priorities(self):
        profile = CountRewardProfile(horizon_seconds=60.0)
        model, replay = make_model(profile, batch_size=2)
        obs = observation(model)
        replay.add(GraphRainbowTransition(
            obs, 0, 1.0, obs, False, model.config.gamma))
        replay.add(GraphRainbowTransition(
            obs, 1, 2.0, obs, True, model.config.gamma))
        before = {
            name: value.copy() for name, value in model.online.params.items()}
        priorities_before = replay.priorities[:2].copy()
        report = update_graph_rainbow(model, replay)
        self.assertEqual(report["updates"], 1)
        self.assertTrue(np.isfinite(report["mean_loss"]))
        self.assertTrue(any(
            not np.array_equal(before[name], value)
            for name, value in model.online.params.items()))
        self.assertFalse(np.array_equal(
            priorities_before, replay.priorities[:2]))
        self.assertEqual(model.training_step, 1)

    def test_short_formal_headless_training_conserves_reward(self):
        profile = CountRewardProfile(horizon_seconds=60.0)
        model, replay = make_model(
            profile, batch_size=1, n_step=2)
        report = train_graph_rainbow(
            model, replay, (72000,), reward_profile=profile,
            scenario_ids=("A",),
            env_config=RLEnvironmentConfig(
                max_robots=8, max_tasks=20,
                max_steps_per_episode=1000),
            runtime_config=HeadlessRuntimeConfig(
                episode_end_time=60.0, max_advance_seconds=60.0,
                fixed_horizon=True))
        self.assertEqual(report["episode_count"], 1)
        self.assertLessEqual(
            report["max_reward_conservation_error"], 1e-9)
        self.assertGreater(report["replay_size"], 0)
        self.assertGreater(report["training_step"], 0)

    def test_identity_and_seed_boundaries_fail_closed(self):
        profile = CountRewardProfile(horizon_seconds=60.0)
        model, replay = make_model(profile)
        with self.assertRaisesRegex(ValueError, "outside standalone"):
            train_graph_rainbow(
                model, replay, (99999,), reward_profile=profile,
                scenario_ids=("A",))
        wrong = CountRewardProfile(horizon_seconds=61.0)
        with self.assertRaisesRegex(ValueError, "reward profile"):
            train_graph_rainbow(
                model, replay, (72000,), reward_profile=wrong,
                scenario_ids=("A",))
        with self.assertRaises(ValueError):
            replay.update_priorities([0], [float("nan")])
        model.mark_webots_fine_tuned((73000,))
        with self.assertRaisesRegex(ValueError, "training identity differs"):
            update_graph_rainbow(model, replay)


if __name__ == "__main__":
    unittest.main()
