"""CostGraphPPO message passing, prior logits, and checkpoint identity."""

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
from cost_graph_ppo_model import (
    COST_GRAPH_PPO_ALGORITHM,
    PRIMARY_NORMALIZED_INDEX,
    CostGraphPPOConfig,
    CostGraphPPOModel,
    CostGraphPPOProvenance,
)
from cost_guided_graph import CostPriorIdentity
from cost_guided_graph_observation import (
    COST_GUIDED_GRAPH_FEATURE_DIM,
    build_cost_guided_graph_observation,
)
from schedulers import ModelValidationError
from test_cost_guided_graph import fixture, selected_bundles


REWARD_HASH = "a" * 64


def model_and_observation(policy="CostGraphPPO_count", seed=17):
    identity = CostPriorIdentity.from_selected_bundles(
        policy, selected_bundles())
    provenance = CostGraphPPOProvenance(
        identity.objective, "test-formal-reward-v1", REWARD_HASH, "standalone",
        (72000, 72001, 72002, 72003, 72004, 72005))
    model = CostGraphPPOModel(identity, provenance, seed=seed)
    robots, tasks, context = fixture()
    observation = build_cost_guided_graph_observation(
        robots, tasks, context, identity)
    return model, observation


class CostGraphPPOModelTests(unittest.TestCase):
    def test_initial_actor_is_exact_primary_cost_prior(self):
        model, observation = model_and_observation()
        probabilities, value, logits = model.policy(observation)
        expected = (
            -model.config.prior_logit_scale
            * observation.action_features[:, PRIMARY_NORMALIZED_INDEX])
        np.testing.assert_allclose(logits, expected, rtol=0, atol=0)
        self.assertEqual(value, 0.0)
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=12)
        self.assertEqual(probabilities[observation.no_op_index], 0.0)
        action, _, _ = model.select_action(
            observation, deterministic=True)
        self.assertEqual(
            action, int(np.argmin(expected[:observation.edge_count] * -1.0)))
        selected_primary = observation.action_features[
            action, PRIMARY_NORMALIZED_INDEX]
        self.assertEqual(
            selected_primary,
            np.min(observation.action_features[
                :observation.edge_count, PRIMARY_NORMALIZED_INDEX]))

    def test_message_network_and_backward_have_finite_expected_shapes(self):
        model, observation = model_and_observation()
        probabilities, _, _, cache = model.network.forward(
            observation, cache=True)
        grad_logits = -probabilities.astype(np.float32)
        action = int(np.argmax(probabilities))
        grad_logits[action] += 1.0
        grads = model.network.backward(cache, grad_logits, 0.5)
        self.assertEqual(set(grads), set(model.network.params))
        for name, value in grads.items():
            self.assertEqual(value.shape, model.network.params[name].shape)
            self.assertTrue(np.all(np.isfinite(value)))
        before = {
            name: value.copy() for name, value in model.network.params.items()}
        model.optimizer.step(model.network.params, grads)
        self.assertTrue(any(
            not np.array_equal(before[name], value)
            for name, value in model.network.params.items()))

    def test_no_edge_observation_selects_explicit_noop(self):
        model, _ = model_and_observation()
        robots, tasks, context = fixture()
        blocked = replace(
            context, path_cost_provider=lambda *_: float("inf"))
        observation = build_cost_guided_graph_observation(
            robots, tasks, blocked, model.prior_identity)
        probabilities, value, logits = model.policy(observation)
        np.testing.assert_array_equal(probabilities, [1.0])
        self.assertTrue(np.isfinite(value))
        self.assertTrue(np.isfinite(logits[0]))
        action, _, _ = model.select_action(
            observation, deterministic=True)
        self.assertEqual(action, observation.no_op_index)
        self.assertIsNone(observation.coordinate_for_action(action))

    def test_checkpoint_round_trip_restores_outputs_optimizer_and_rng(self):
        model, observation = model_and_observation(seed=23)
        _, _, _, cache = model.network.forward(observation, cache=True)
        probabilities, _, _ = model.policy(observation)
        gradient = -probabilities.astype(np.float32)
        gradient[0] += 1.0
        grads = model.network.backward(cache, gradient, 0.25)
        model.optimizer.step(model.network.params, grads)
        model.training_step = 1
        model.training_episodes = 2
        sampled_before_save = model.select_action(
            observation, deterministic=False)[0]
        del sampled_before_save
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            expected_next = model.select_action(
                observation, deterministic=False)[0]
            restored = CostGraphPPOModel.load(
                path, expected_policy_id=model.prior_identity.policy_id,
                expected_reward_profile_sha256=REWARD_HASH)
            actual_next = restored.select_action(
                observation, deterministic=False)[0]
        for left, right in zip(
                model.policy(observation), restored.policy(observation)):
            np.testing.assert_allclose(left, right, rtol=0, atol=0)
        self.assertEqual(expected_next, actual_next)
        self.assertEqual(restored.training_step, 1)
        self.assertEqual(restored.training_episodes, 2)
        self.assertEqual(restored.optimizer.step_count, 1)

    def test_checkpoint_rejects_prior_and_reward_identity_tampering(self):
        model, _ = model_and_observation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            tampered_path = Path(directory) / "tampered.npz"
            model.save(path)
            metadata, arrays = load_checkpoint(
                path, expected_algorithm=COST_GRAPH_PPO_ALGORITHM,
                expected_state_dim=COST_GUIDED_GRAPH_FEATURE_DIM,
                expected_action_dim=1)
            metadata["prior_identity"]["objective"] = "utility_v2"
            save_checkpoint(tampered_path, metadata, arrays)
            with self.assertRaises(ModelValidationError):
                CostGraphPPOModel.load(tampered_path)
            with self.assertRaises(ModelValidationError):
                CostGraphPPOModel.load(
                    path, expected_reward_profile_sha256="b" * 64)

    def test_provenance_phase_and_seed_boundaries_fail_closed(self):
        with self.assertRaises(ValueError):
            CostGraphPPOConfig(prior_logit_scale=0.0)
        with self.assertRaises(ValueError):
            CostGraphPPOProvenance(
                "count", "reward", REWARD_HASH, "standalone",
                (72000,), (73000,))
        with self.assertRaises(ValueError):
            CostGraphPPOProvenance(
                "count", "reward", REWARD_HASH, "webots_fine_tuned",
                (72000,), ())
        identity = CostPriorIdentity.from_selected_bundles(
            "CostGraphPPO_utility_v2", selected_bundles())
        mismatched = CostGraphPPOProvenance(
            "count", "reward", REWARD_HASH, "standalone", (72000,))
        with self.assertRaises(ValueError):
            CostGraphPPOModel(identity, mismatched)
        model, _ = model_and_observation()
        model.mark_webots_fine_tuned((73000, 73001))
        self.assertEqual(
            model.provenance.training_phase, "webots_fine_tuned")
        with self.assertRaises(ValueError):
            model.mark_webots_fine_tuned((73000, 73001))


if __name__ == "__main__":
    unittest.main()
