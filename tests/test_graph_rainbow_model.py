"""Variable-edge dueling C51 GraphRainbow model contracts."""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/factory_supervisor"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from advanced_ai_common import load_checkpoint, save_checkpoint
from cost_guided_graph import CostPriorIdentity
from cost_guided_graph_observation import (
    COST_GUIDED_GRAPH_FEATURE_DIM,
    build_cost_guided_graph_observation,
)
from graph_rainbow_model import (
    GRAPH_RAINBOW_ALGORITHM,
    PRIMARY_NORMALIZED_INDEX,
    GraphRainbowConfig,
    GraphRainbowModel,
    GraphRainbowProvenance,
)
from schedulers import ModelValidationError
from test_cost_guided_graph import fixture, selected_bundles


REWARD_HASH = "c" * 64


def model_and_observation(policy="GraphRainbow_count", seed=41):
    identity = CostPriorIdentity.from_selected_bundles(
        policy, selected_bundles())
    provenance = GraphRainbowProvenance(
        identity.objective, "test-formal-reward-v1", REWARD_HASH,
        "standalone", (72000, 72001, 72002, 72003, 72004, 72005))
    config = GraphRainbowConfig(
        hidden_size=32, atom_count=51,
        v_min=0.0 if identity.objective == "count" else -200.0,
        v_max=100.0 if identity.objective == "count" else 200.0,
        replay_capacity=128, batch_size=8)
    model = GraphRainbowModel(identity, provenance, config, seed)
    robots, tasks, context = fixture()
    observation = build_cost_guided_graph_observation(
        robots, tasks, context, identity)
    return model, observation


class GraphRainbowModelTests(unittest.TestCase):
    def test_dueling_c51_shapes_normalization_and_initial_cost_prior(self):
        model, observation = model_and_observation()
        distributions, logits = model.distributions(
            observation, noisy=False)
        self.assertEqual(
            distributions.shape,
            (observation.action_count, model.config.atom_count))
        self.assertEqual(logits.shape, distributions.shape)
        np.testing.assert_allclose(
            np.sum(distributions, axis=1), 1.0, rtol=0, atol=1e-6)
        values = model.q_values(observation, noisy=False)
        expected = (
            float(np.mean(model.config.support))
            - model.config.prior_q_scale
            * observation.action_features[:, PRIMARY_NORMALIZED_INDEX])
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-5)
        action = model.select_action(observation, deterministic=True)
        self.assertEqual(
            observation.action_features[
                action, PRIMARY_NORMALIZED_INDEX],
            np.min(observation.action_features[
                :observation.edge_count, PRIMARY_NORMALIZED_INDEX]))

    def test_noisy_training_scores_change_but_deterministic_scores_do_not(self):
        model, observation = model_and_observation()
        first = model.q_values(observation, noisy=False)
        second = model.q_values(observation, noisy=False)
        np.testing.assert_array_equal(first, second)
        noisy_first = model.q_values(observation, noisy=True)
        noisy_second = model.q_values(observation, noisy=True)
        self.assertFalse(np.array_equal(noisy_first, noisy_second))
        self.assertTrue(np.all(np.isfinite(noisy_first)))

    def test_distribution_backward_updates_online_and_target_sync_is_exact(self):
        model, observation = model_and_observation()
        distributions, _, cache = model.distributions(
            observation, noisy=True, cache=True)
        action = model.select_action(observation, deterministic=True)
        target = np.zeros(model.config.atom_count, dtype=np.float32)
        target[model.config.atom_count // 2] = 1.0
        grad_logits = np.zeros_like(distributions, dtype=np.float32)
        grad_logits[action] = distributions[action] - target
        grads = model.online.backward(cache, grad_logits)
        self.assertEqual(set(grads), set(model.online.params))
        before = {
            name: value.copy() for name, value in model.online.params.items()}
        model.optimizer.step(model.online.params, grads)
        self.assertTrue(any(
            not np.array_equal(before[name], value)
            for name, value in model.online.params.items()))
        self.assertTrue(any(
            not np.array_equal(
                model.online.params[name], model.target.params[name])
            for name in model.online.params))
        model.sync_target()
        for name in model.online.params:
            np.testing.assert_array_equal(
                model.online.params[name], model.target.params[name])

    def test_no_edge_snapshot_selects_only_noop(self):
        model, _ = model_and_observation()
        robots, tasks, context = fixture()
        observation = build_cost_guided_graph_observation(
            robots, tasks,
            replace(context, path_cost_provider=lambda *_: float("inf")),
            model.prior_identity)
        action = model.select_action(observation, deterministic=True)
        self.assertEqual(action, observation.no_op_index)
        np.testing.assert_array_equal(observation.action_mask, [True])

    def test_checkpoint_restores_online_target_optimizer_and_rng(self):
        model, observation = model_and_observation(seed=43)
        distributions, _, cache = model.distributions(
            observation, noisy=True, cache=True)
        action = 0
        target = np.zeros(model.config.atom_count, dtype=np.float32)
        target[3] = 1.0
        gradient = np.zeros_like(distributions, dtype=np.float32)
        gradient[action] = distributions[action] - target
        model.optimizer.step(
            model.online.params,
            model.online.backward(cache, gradient))
        model.training_step = 1
        model.training_episodes = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rainbow.npz"
            model.save(path)
            expected_noisy = model.q_values(observation, noisy=True)
            restored = GraphRainbowModel.load(
                path, expected_policy_id=model.prior_identity.policy_id,
                expected_reward_profile_sha256=REWARD_HASH)
            actual_noisy = restored.q_values(observation, noisy=True)
        np.testing.assert_allclose(
            expected_noisy, actual_noisy, rtol=0, atol=0)
        for name in model.online.params:
            np.testing.assert_array_equal(
                model.online.params[name], restored.online.params[name])
            np.testing.assert_array_equal(
                model.target.params[name], restored.target.params[name])
        self.assertEqual(restored.optimizer.step_count, 1)

    def test_checkpoint_and_provenance_tampering_fail_closed(self):
        model, _ = model_and_observation()
        with self.assertRaises(ValueError):
            GraphRainbowConfig(prior_q_scale=0.0)
        with self.assertRaises(ValueError):
            GraphRainbowProvenance(
                "count", "reward", REWARD_HASH, "standalone",
                (72000,), (73000,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rainbow.npz"
            bad = Path(directory) / "bad.npz"
            model.save(path)
            metadata, arrays = load_checkpoint(
                path, expected_algorithm=GRAPH_RAINBOW_ALGORITHM,
                expected_state_dim=COST_GUIDED_GRAPH_FEATURE_DIM,
                expected_action_dim=1)
            metadata["prior_identity"]["objective"] = "utility_v2"
            save_checkpoint(bad, metadata, arrays)
            with self.assertRaises(ModelValidationError):
                GraphRainbowModel.load(bad)
            with self.assertRaises(ModelValidationError):
                GraphRainbowModel.load(
                    path, expected_reward_profile_sha256="d" * 64)


if __name__ == "__main__":
    unittest.main()
