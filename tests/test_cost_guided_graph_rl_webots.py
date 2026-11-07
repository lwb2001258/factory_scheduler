"""Webots phase contracts for cost-guided GraphPPO and GraphRainbow."""

from pathlib import Path
import json
import shutil
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/factory_supervisor"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import cost_guided_graph_rl_workflow_contract as contract
import run_cost_guided_graph_rl_webots as webots_runner
from cost_graph_ppo_model import (
    CostGraphPPOConfig, CostGraphPPOModel, CostGraphPPOProvenance,
)
from cost_guided_graph import CostPriorIdentity
from cost_guided_graph_observation import build_cost_guided_graph_observation
from dual_objective import CountRewardProfile
from graph_rainbow_model import (
    GraphRainbowConfig, GraphRainbowModel, GraphRainbowProvenance,
)
from test_cost_guided_graph import fixture, selected_bundles


def payload(observation):
    return {
        "feature_schema_sha256": observation.feature_schema_sha256,
        "prior_identity_sha256": observation.prior_identity_sha256,
        "current_time": float(observation.current_time),
        "matrix_shape": list(observation.matrix.values.shape),
        "coordinates": observation.coordinates.tolist(),
        "action_features": observation.action_features.tolist(),
        "action_mask": observation.action_mask.astype(int).tolist(),
        "no_op_index": observation.no_op_index,
    }


def physical_envelope(rows, reward_total):
    return {
        "result": {
            "formal_evaluation": {
                "physical_fine_tune": {
                    "reward_attribution_version":
                        "physical_action_reward_v1",
                    "rollout_step_count": len(rows),
                    "rollout": rows,
                    "reward_audit": {"reward_total": reward_total},
                    "episode_return": reward_total,
                }
            }
        }
    }


class CostGuidedGraphRLWebotsTests(unittest.TestCase):
    def setUp(self):
        self.root = (
            contract.ROOT / "results" /
            "test_cost_guided_graph_rl_webots")
        self._remove()
        contract.initialize(self.root)

    def tearDown(self):
        self._remove()

    def _remove(self):
        resolved = self.root.resolve()
        self.assertEqual(
            resolved.parent, (contract.ROOT / "results").resolve())
        self.assertTrue(resolved.name.startswith(
            "test_cost_guided_graph_rl_"))
        if resolved.exists():
            shutil.rmtree(resolved)

    def _observation(self, policy_id):
        robots, tasks, context = fixture()
        identity = CostPriorIdentity.from_selected_bundles(
            policy_id, selected_bundles())
        return identity, build_cost_guided_graph_observation(
            robots, tasks, context, identity)

    def test_canonical_manifests_are_partitioned_and_idempotent(self):
        first = webots_runner.prepare_manifests(
            self.root, "fine_tune", contract.WEBOTS_FINE_TUNE_SEEDS)
        second = webots_runner.prepare_manifests(
            self.root, "fine_tune", contract.WEBOTS_FINE_TUNE_SEEDS)
        self.assertEqual(first, second)
        self.assertEqual(
            len(first), len(contract.WEBOTS_FINE_TUNE_SEEDS) * 3)
        self.assertTrue(all(path.is_file() for path in first))
        with self.assertRaisesRegex(ValueError, "partition differs"):
            webots_runner.prepare_manifests(
                self.root, "fine_tune", contract.EVALUATION_SEEDS)

    def test_physical_reward_profiles_keep_count_and_upgrade_utility(self):
        count = webots_runner.physical_reward_profile("count")
        utility = webots_runner.physical_reward_profile("utility_v2")
        self.assertEqual(
            count.sha256,
            CountRewardProfile(horizon_seconds=1800.0).sha256)
        self.assertEqual(utility.safety_observation, "webots_physical")
        self.assertEqual(
            utility.sha256,
            webots_runner.webots_workflow._physical_utility_reward_hash())

    def test_observation_payload_round_trip_is_identity_bound(self):
        identity, observation = self._observation("CostGraphPPO_count")
        restored = webots_runner.observation_from_payload(
            payload(observation), expected_prior_sha256=identity.sha256)
        self.assertEqual(restored.prior_identity_sha256, identity.sha256)
        self.assertTrue((restored.coordinates == observation.coordinates).all())
        self.assertTrue((restored.action_features ==
                         observation.action_features).all())
        wrong = payload(observation)
        wrong["prior_identity_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity differs"):
            webots_runner.observation_from_payload(
                wrong, expected_prior_sha256=identity.sha256)

    def test_physical_rollouts_update_both_algorithms_after_audit(self):
        profile = CountRewardProfile(horizon_seconds=1800.0)
        seeds = tuple(contract.STANDALONE_SEEDS)

        ppo_identity, ppo_observation = self._observation(
            "CostGraphPPO_count")
        ppo = CostGraphPPOModel(
            ppo_identity,
            CostGraphPPOProvenance(
                "count", profile.version, profile.sha256,
                "standalone", seeds),
            CostGraphPPOConfig(hidden_size=16, update_epochs=1), seed=71)
        webots_runner._transition_to_physical(ppo, "count")
        action, log_probability, value = ppo.select_action(
            ppo_observation, deterministic=True)
        ppo_rows = [{
            "kind": "cost_graph_ppo",
            "observation": payload(ppo_observation),
            "action_index": action,
            "old_log_probability": log_probability,
            "value": value,
            "reward": 1.0,
            "task_reward": 1.0,
            "shared_reward": 0.0,
            "done": True,
        }]
        with mock.patch.object(
                webots_runner.webots_workflow,
                "valid_action_reward_rollout", return_value=True):
            ppo_metrics = webots_runner.update_ppo_from_envelope(
                ppo, physical_envelope(ppo_rows, 1.0))
        self.assertEqual(ppo.training_episodes, 1)
        self.assertGreater(ppo_metrics["updates"], 0)
        self.assertEqual(ppo_metrics["reward_conservation_error"], 0.0)

        rainbow_identity, rainbow_observation = self._observation(
            "GraphRainbow_count")
        rainbow = GraphRainbowModel(
            rainbow_identity,
            GraphRainbowProvenance(
                "count", profile.version, profile.sha256,
                "standalone", seeds),
            GraphRainbowConfig(
                hidden_size=16, atom_count=51, v_min=0.0, v_max=100.0,
                n_step=1, replay_capacity=32, batch_size=1), seed=73)
        webots_runner._transition_to_physical(rainbow, "count")
        replay = webots_runner._new_rainbow_replay(rainbow)
        rainbow_action = rainbow.select_action(
            rainbow_observation, deterministic=True)
        rainbow_rows = [{
            "kind": "graph_rainbow",
            "observation": payload(rainbow_observation),
            "action_index": rainbow_action,
            "reward": 1.0,
            "task_reward": 1.0,
            "shared_reward": 0.0,
            "done": True,
        }]
        with mock.patch.object(
                webots_runner.webots_workflow,
                "valid_action_reward_rollout", return_value=True):
            rainbow_metrics = webots_runner.update_rainbow_from_envelope(
                rainbow, replay, physical_envelope(rainbow_rows, 1.0))
        self.assertEqual(rainbow.training_episodes, 1)
        self.assertEqual(len(replay), 1)
        self.assertEqual(rainbow_metrics["updates"], 1)
        self.assertEqual(
            rainbow_metrics["reward_conservation_error"], 0.0)

    def test_paired_comparisons_require_the_same_eighteen_keys(self):
        rows = []
        for policy_id in contract.POLICIES:
            for seed in contract.EVALUATION_SEEDS:
                for scenario_index, scenario in enumerate(contract.SCENARIOS):
                    rows.append({
                        "policy_id": policy_id,
                        "seed": seed,
                        "scenario": scenario,
                        "score": float(seed + scenario_index + (
                            1 if policy_id.startswith("CostGraphPPO_") else 0)),
                    })
        comparisons = webots_runner._paired_comparisons(rows)
        self.assertEqual(set(comparisons), {"count", "utility_v2"})
        self.assertTrue(all(
            row["paired_count"] == 18 and
            row["mean_difference"] == 1.0
            for row in comparisons.values()))

    def test_webots_resume_distinguishes_factory_and_result_scheduler(self):
        directory = self.root / "resume_test"
        directory.mkdir(parents=True)
        canonical = directory / "result.json"
        canonical.write_text(json.dumps({"cached": True}),
                             encoding="utf-8")
        with mock.patch.object(
                webots_runner.webots_workflow, "valid_episode",
                return_value=True) as valid:
            envelope, cached = (
                webots_runner.webots_workflow.run_or_resume_episode(
                    canonical_path=canonical,
                    webots=webots_runner.DEFAULT_WEBOTS,
                    scenario="A", seed=73000,
                    manifest=directory / "manifest.json",
                    scheduler="CostGraphPPO", model_path=None,
                    fine_tune_objective="count",
                    fine_tune_algorithm="CostGraphPPO",
                    expected_result_scheduler="CostGraphPPO_count"))
        self.assertTrue(cached)
        self.assertEqual(envelope, {"cached": True})
        self.assertEqual(
            valid.call_args.kwargs["expected_scheduler"],
            "CostGraphPPO_count")


if __name__ == "__main__":
    unittest.main()
