"""Runtime integration for cost-guided graph RL schedulers."""

from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/factory_supervisor"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from cost_graph_ppo_model import (
    CostGraphPPOModel, CostGraphPPOProvenance,
)
from cost_guided_graph import CostPriorIdentity
from cost_guided_rl_schedulers import (
    CostGraphPPOScheduler, GraphRainbowScheduler,
)
from dual_objective import CountRewardProfile
from graph_rainbow_model import (
    GraphRainbowConfig, GraphRainbowModel, GraphRainbowProvenance,
)
from rl_schedulers import RLSchedulerSafetyWrapper
from schedulers import create_scheduler
from test_cost_guided_graph import fixture, selected_bundles


def write_models(directory):
    profile = CountRewardProfile(horizon_seconds=60.0)
    seeds = (72000, 72001, 72002, 72003, 72004, 72005)
    ppo_identity = CostPriorIdentity.from_selected_bundles(
        "CostGraphPPO_count", selected_bundles())
    rainbow_identity = CostPriorIdentity.from_selected_bundles(
        "GraphRainbow_count", selected_bundles())
    ppo = CostGraphPPOModel(
        ppo_identity,
        CostGraphPPOProvenance(
            "count", profile.version, profile.sha256, "standalone", seeds),
        seed=53)
    rainbow = GraphRainbowModel(
        rainbow_identity,
        GraphRainbowProvenance(
            "count", profile.version, profile.sha256, "standalone", seeds),
        GraphRainbowConfig(
            hidden_size=16, atom_count=51, v_min=0.0, v_max=100.0,
            replay_capacity=64, batch_size=4),
        seed=59)
    ppo_path = Path(directory) / "cost_graph_ppo.npz"
    rainbow_path = Path(directory) / "graph_rainbow.npz"
    ppo.save(ppo_path)
    rainbow.save(rainbow_path)
    return profile, ppo_path, rainbow_path


class CostGuidedRLSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.profile, self.ppo_path, self.rainbow_path = write_models(
            self.directory.name)
        self.robots, self.tasks, self.context = fixture()

    def test_deterministic_schedulers_return_native_feasible_assignments(self):
        for scheduler in (
                CostGraphPPOScheduler(self.ppo_path),
                GraphRainbowScheduler(self.rainbow_path)):
            first = scheduler.assign(
                self.tasks, self.robots, self.context)
            second = scheduler.assign(
                self.tasks, self.robots, self.context)
            self.assertTrue(first.is_feasible)
            self.assertEqual(len(first.assignments), 1)
            self.assertEqual(
                (first.assignments[0].robot_id,
                 first.assignments[0].task.task_id),
                (second.assignments[0].robot_id,
                 second.assignments[0].task.task_id))
            self.assertFalse(first.diagnostics.get("fallback", False))
            self.assertEqual(
                first.diagnostics["objective"], "count")
            self.assertEqual(
                first.diagnostics["reward_profile_sha256"],
                self.profile.sha256)

    def test_factory_wrapper_proxies_traffic_bundle_and_pure_policy(self):
        environment = {
            "SMART_FACTORY_PURE_POLICY_REQUIRED": "1",
            "SMART_FACTORY_COST_GUIDED_POLICY_ID": "CostGraphPPO_count",
            "SMART_FACTORY_EXPECTED_REWARD_PROFILE_SHA256":
                self.profile.sha256,
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            scheduler = create_scheduler(
                "CostGraphPPO", self.ppo_path,
                allow_safe_fallback=False)
        self.assertIsInstance(scheduler, RLSchedulerSafetyWrapper)
        self.assertTrue(scheduler.uses_traffic)
        self.assertTrue(scheduler.pure_policy_required)
        self.assertEqual(scheduler.objective, "count")
        self.assertEqual(
            scheduler.expert_coefficient_sha256,
            scheduler.cost_prior_identity.primary_bundle_sha256)
        decision = scheduler.assign(
            self.tasks, self.robots, self.context)
        self.assertTrue(decision.is_feasible)
        self.assertFalse(decision.diagnostics["fallback"])

    def test_no_feasible_edge_is_native_wait_not_fallback(self):
        scheduler = RLSchedulerSafetyWrapper(
            CostGraphPPOScheduler(self.ppo_path))
        blocked = replace(
            self.context, path_cost_provider=lambda *_: float("inf"))
        decision = scheduler.assign(
            self.tasks, self.robots, blocked)
        self.assertTrue(decision.is_feasible)
        self.assertEqual(decision.assignments, [])
        self.assertTrue(decision.diagnostics["intentional_wait"])
        self.assertFalse(decision.diagnostics["fallback"])
        self.assertEqual(scheduler.fallback_decisions, 0)

    def test_physical_collection_emits_committed_action_payload(self):
        environment = {
            "SMART_FACTORY_PHYSICAL_FINE_TUNE": "1",
            "SMART_FACTORY_PURE_POLICY_REQUIRED": "1",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            for scheduler, expected_kind in (
                    (CostGraphPPOScheduler(self.ppo_path),
                     "cost_graph_ppo"),
                    (GraphRainbowScheduler(self.rainbow_path),
                     "graph_rainbow")):
                wrapped = RLSchedulerSafetyWrapper(scheduler)
                decision = wrapped.assign(
                    self.tasks, self.robots, self.context)
                self.assertTrue(decision.is_feasible)
                self.assertFalse(decision.diagnostics["fallback"])
                self.assertTrue(
                    decision.diagnostics["physical_fine_tune"])
                rollout = decision.diagnostics[
                    "physical_rollout_step"]
                self.assertEqual(rollout["kind"], expected_kind)
                payload = rollout["observation"]
                self.assertEqual(
                    len(payload["action_features"]),
                    len(payload["action_mask"]))
                self.assertEqual(
                    rollout["action_index"],
                    decision.diagnostics["selected_edge_index"])
                assignment = decision.assignments[0]
                self.assertEqual(
                    rollout["selected_robot_id"], assignment.robot_id)
                self.assertEqual(
                    rollout["selected_task_id"], assignment.task.task_id)

    def test_physical_evaluation_audit_collects_without_exploration(self):
        environment = {
            "SMART_FACTORY_PHYSICAL_FINE_TUNE": "1",
            "SMART_FACTORY_COST_GRAPH_RL_DETERMINISTIC": "1",
            "SMART_FACTORY_PURE_POLICY_REQUIRED": "1",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            ppo = CostGraphPPOScheduler(self.ppo_path)
            with mock.patch.object(
                    ppo.model, "select_action",
                    wraps=ppo.model.select_action) as select_action:
                ppo_decision = ppo.assign(
                    self.tasks, self.robots, self.context)
            rainbow = GraphRainbowScheduler(self.rainbow_path)
            with mock.patch.object(
                    rainbow.model, "q_values",
                    wraps=rainbow.model.q_values) as q_values:
                rainbow_decision = rainbow.assign(
                    self.tasks, self.robots, self.context)

        self.assertTrue(ppo_decision.diagnostics["physical_fine_tune"])
        self.assertFalse(ppo_decision.diagnostics["exploration_enabled"])
        self.assertEqual(
            ppo_decision.diagnostics["decoder"],
            "deterministic_policy_argmax")
        self.assertTrue(select_action.call_args.kwargs["deterministic"])
        self.assertTrue(rainbow_decision.diagnostics["physical_fine_tune"])
        self.assertFalse(
            rainbow_decision.diagnostics["exploration_enabled"])
        self.assertEqual(
            rainbow_decision.diagnostics["decoder"],
            "deterministic_policy_argmax")
        self.assertFalse(q_values.call_args.kwargs["noisy"])

    def test_pure_policy_inference_error_never_uses_hungarian(self):
        environment = {"SMART_FACTORY_PURE_POLICY_REQUIRED": "1"}
        with mock.patch.dict(os.environ, environment, clear=False):
            scheduler = RLSchedulerSafetyWrapper(
                CostGraphPPOScheduler(self.ppo_path))
        stale = replace(
            self.context, current_time=self.context.current_time + 1.0)
        decision = scheduler.assign(
            self.tasks, self.robots, stale)
        self.assertFalse(decision.is_feasible)
        self.assertEqual(decision.assignments, [])
        self.assertTrue(decision.diagnostics["pure_policy_rejected"])
        self.assertFalse(decision.diagnostics["fallback"])
        self.assertEqual(scheduler.fallback_decisions, 0)


if __name__ == "__main__":
    unittest.main()
