"""PPO rollout math and standalone formal training guardrails."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/factory_supervisor"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from cost_graph_ppo_model import (
    CostGraphPPOConfig, CostGraphPPOModel, CostGraphPPOProvenance,
)
from cost_graph_ppo_training import (
    CostGraphPPORolloutStep,
    prepare_cost_graph_ppo_advantages,
    train_cost_graph_ppo,
    update_cost_graph_ppo,
)
from cost_guided_graph import CostPriorIdentity
from cost_guided_graph_observation import (
    build_cost_guided_graph_observation,
)
from dual_objective import CountRewardProfile
from headless_training_runtime import HeadlessRuntimeConfig
from rl_environment import RLEnvironmentConfig
from test_cost_guided_graph import fixture, selected_bundles


def make_model(profile, *, seeds=(72000,), config=None):
    identity = CostPriorIdentity.from_selected_bundles(
        "CostGraphPPO_count", selected_bundles())
    provenance = CostGraphPPOProvenance(
        "count", profile.version, profile.sha256, "standalone", seeds)
    return CostGraphPPOModel(
        identity, provenance, config=config, seed=31)


class CostGraphPPOTrainingTests(unittest.TestCase):
    def test_terminal_and_truncated_advantages_use_correct_bootstrap(self):
        profile = CountRewardProfile(horizon_seconds=60.0)
        model = make_model(profile)
        robots, tasks, context = fixture()
        observation = build_cost_guided_graph_observation(
            robots, tasks, context, model.prior_identity)
        action, log_probability, value = model.select_action(
            observation, deterministic=False)
        terminal = [CostGraphPPORolloutStep(
            observation, action, log_probability, value,
            1.0, True, False)]
        truncated = [CostGraphPPORolloutStep(
            observation, action, log_probability, value,
            1.0, False, True)]
        prepare_cost_graph_ppo_advantages(
            model, terminal, bootstrap_value=10.0)
        prepare_cost_graph_ppo_advantages(
            model, truncated, bootstrap_value=10.0)
        self.assertAlmostEqual(
            terminal[0].return_value, 1.0, places=12)
        self.assertAlmostEqual(
            truncated[0].return_value,
            1.0 + model.config.gamma * 10.0, places=12)
        with self.assertRaisesRegex(ValueError, "requires bootstrap"):
            update_cost_graph_ppo(model, truncated)
        terminal_and_limit = [CostGraphPPORolloutStep(
            observation, action, log_probability, value,
            1.0, True, True)]
        prepare_cost_graph_ppo_advantages(
            model, terminal_and_limit, bootstrap_value=10.0)
        self.assertAlmostEqual(
            terminal_and_limit[0].return_value, 1.0, places=12)

    def test_clipped_ppo_update_changes_parameters_and_counters(self):
        profile = CountRewardProfile(horizon_seconds=60.0)
        model = make_model(
            profile, config=CostGraphPPOConfig(update_epochs=2))
        robots, tasks, context = fixture()
        observation = build_cost_guided_graph_observation(
            robots, tasks, context, model.prior_identity)
        rollout = []
        for index, reward in enumerate((1.0, -0.25, 2.0)):
            action, log_probability, value = model.select_action(
                observation, deterministic=False)
            rollout.append(CostGraphPPORolloutStep(
                observation, action, log_probability, value, reward,
                index == 2, False))
        before = {
            name: value.copy() for name, value in model.network.params.items()}
        report = update_cost_graph_ppo(
            model, rollout, bootstrap_value=0.0)
        self.assertEqual(report["updates"], 6)
        self.assertEqual(model.training_step, 6)
        self.assertEqual(model.optimizer.step_count, 6)
        self.assertEqual(model.training_episodes, 1)
        self.assertTrue(np.isfinite(report["mean_loss"]))
        self.assertTrue(any(
            not np.array_equal(before[name], value)
            for name, value in model.network.params.items()))

    def test_rollout_rejects_masked_action_and_identity_mismatch(self):
        profile = CountRewardProfile(horizon_seconds=60.0)
        model = make_model(profile)
        robots, tasks, context = fixture()
        observation = build_cost_guided_graph_observation(
            robots, tasks, context, model.prior_identity)
        with self.assertRaises(ValueError):
            CostGraphPPORolloutStep(
                observation, observation.no_op_index, 0.0, 0.0,
                0.0, False, False)
        legal_action = int(np.flatnonzero(observation.action_mask)[0])
        with self.assertRaises(ValueError):
            CostGraphPPORolloutStep(
                observation, legal_action, 0.1, 0.0,
                0.0, False, False)
        other_identity = CostPriorIdentity.from_selected_bundles(
            "CostGraphPPO_utility_v2", selected_bundles())
        other_observation = build_cost_guided_graph_observation(
            robots, tasks, context, other_identity)
        action = int(np.flatnonzero(other_observation.action_mask)[0])
        step = CostGraphPPORolloutStep(
            other_observation, action, 0.0, 0.0, 0.0, True, False)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            update_cost_graph_ppo(model, [step])

    def test_standalone_training_checks_seed_and_reward_identity(self):
        profile = CountRewardProfile(horizon_seconds=60.0)
        model = make_model(profile)
        with self.assertRaisesRegex(ValueError, "outside standalone"):
            train_cost_graph_ppo(
                model, (99999,), reward_profile=profile,
                scenario_ids=("A",))
        wrong = CountRewardProfile(horizon_seconds=61.0)
        with self.assertRaisesRegex(ValueError, "reward profile"):
            train_cost_graph_ppo(
                model, (72000,), reward_profile=wrong,
                scenario_ids=("A",))

    def test_short_formal_headless_smoke_conserves_reward(self):
        profile = CountRewardProfile(horizon_seconds=60.0)
        model = make_model(
            profile,
            config=CostGraphPPOConfig(
                hidden_size=16, update_epochs=1))
        report = train_cost_graph_ppo(
            model, (72000,), reward_profile=profile,
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
        self.assertEqual(report["episode_seeds"], [72000])
        self.assertEqual(report["scenarios"], ["A"])
        self.assertGreater(report["training_step"], 0)


if __name__ == "__main__":
    unittest.main()
