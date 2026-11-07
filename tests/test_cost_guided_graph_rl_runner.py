"""Standalone workflow planning, resume, and execution-review contracts."""

from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/factory_supervisor"))
sys.path.insert(0, str(ROOT / "scripts"))

import cost_guided_graph_rl_workflow_contract as contract
import run_cost_guided_graph_rl_workflow as runner
from cost_graph_ppo_model import CostGraphPPOModel
from graph_rainbow_model import GraphRainbowModel
from graph_rainbow_training import GraphRainbowReplayBuffer


class CostGuidedGraphRLRunnerTests(unittest.TestCase):
    def setUp(self):
        self.root = (
            contract.ROOT / "results" /
            "test_cost_guided_graph_rl_runner")
        self._remove()
        contract.initialize(self.root)

    def tearDown(self):
        self._remove()

    def _remove(self):
        resolved = self.root.resolve()
        self.assertEqual(
            resolved.parent, (contract.ROOT / "results").resolve())
        self.assertTrue(
            resolved.name.startswith("test_cost_guided_graph_rl_"))
        if resolved.exists():
            shutil.rmtree(resolved)

    def test_plan_freezes_four_policy_profiles_configs_and_partitions(self):
        workflow = contract.validate_root(self.root)[1]
        plan = runner.standalone_plan(workflow)
        self.assertEqual(
            set(plan["policies"]), set(contract.POLICIES))
        self.assertEqual(
            plan["standalone_seeds"], list(contract.STANDALONE_SEEDS))
        self.assertEqual(plan["scenarios"], list(contract.SCENARIOS))
        self.assertEqual(
            plan["calibration_sha256"], runner.CALIBRATION_SHA256)
        for policy_id, row in plan["policies"].items():
            self.assertEqual(
                row["objective"],
                contract.TEACHER_MAPPING[policy_id]["objective"])
            self.assertEqual(len(row["reward_profile_sha256"]), 64)
            self.assertGreater(row["model_config"]["hidden_size"], 0)

    def test_prepare_is_idempotent_and_checkpoints_are_strictly_loadable(self):
        with mock.patch.object(contract, "require_reviews"):
            first = runner.prepare_standalone(self.root)
            second = runner.prepare_standalone(self.root)
        self.assertEqual(first["plan"], second["plan"])
        for policy_id, state in first["states"].items():
            self.assertEqual(
                state["checkpoint_sha256"],
                contract.sha256(state["checkpoint"]))
            if policy_id.startswith("CostGraphPPO_"):
                model = CostGraphPPOModel.load(
                    state["checkpoint"], expected_policy_id=policy_id)
                self.assertIsNone(state["replay"])
            else:
                model = GraphRainbowModel.load(
                    state["checkpoint"], expected_policy_id=policy_id)
                replay = GraphRainbowReplayBuffer.load(
                    state["replay"],
                    expected_prior_identity_sha256=
                        model.prior_identity.sha256,
                    expected_training_identity_sha256=
                        runner.graph_rainbow_replay_identity(model))
                self.assertEqual(len(replay), 0)

    def test_execution_reviews_are_three_hash_bound_passes(self):
        sentinel = self.root / "standalone/review_sentinel.json"
        contract.write(self.root, sentinel, {"revision": 1})
        for perspective in contract.REVIEW_PERSPECTIVES:
            runner.record_execution_review(
                self.root, "standalone",
                perspective=perspective, files=[sentinel],
                findings=["reviewed"], checks=["hash"])
        runner.require_execution_reviews(self.root, "standalone")
        contract.write(self.root, sentinel, {"revision": 2})
        with self.assertRaisesRegex(
                RuntimeError, "reviewed execution file changed"):
            runner.require_execution_reviews(
                self.root, "standalone")


if __name__ == "__main__":
    unittest.main()
