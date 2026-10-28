import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "controllers/factory_supervisor"), str(ROOT / "scripts")]

import neural_sarsa_workflow_contract as audit
from config import RobotState
from learning_scheduler import traffic_edge_values
from schedulers import ModelValidationError, build_cost_matrix, create_scheduler
from sarsa_v4_agent import NeuralDecision, SarsaV4Agent, SarsaV4Config
from sarsa_v4_features import (
    ACTION_DIM, ACTION_FEATURE_DIM, CONTEXT_DIM, FEATURE_DIM, INDEX,
    TRAFFIC_ALIGNMENT_SCALE, build_snapshot)
from sarsa_v4_runtime import SarsaV4Scheduler
from task_generator import TaskStatus
from test_sarsa_features import context, fixture


class NeuralFeatureTests(unittest.TestCase):
    def test_graph_and_sarsa_share_exact_traffic_penalty(self):
        robots, tasks, ctx = fixture(480.0, 2)
        snapshot = build_snapshot(robots, tasks, ctx)
        matrix = build_cost_matrix(list(snapshot.tasks), robots, ctx)
        traffic, penalties = traffic_edge_values(matrix, ctx)
        self.assertEqual(snapshot.features.shape, (ACTION_DIM, FEATURE_DIM))
        self.assertEqual(CONTEXT_DIM, 385)
        self.assertEqual(ACTION_FEATURE_DIM, 72)
        robot_row = matrix.robot_ids.index(snapshot.robot_ids[0])
        task_column = tuple(
            task.task_id for task in matrix.tasks).index(
                snapshot.tasks[0].task_id)
        self.assertAlmostEqual(
            snapshot.features[0, INDEX["traffic_expert_penalty_distance"]] *
            TRAFFIC_ALIGNMENT_SCALE[0], penalties[robot_row, task_column])
        expected = np.asarray(ctx.traffic_context.features(
            snapshot.robot_ids[0], snapshot.tasks[0].pickup_position,
            snapshot.tasks[0].delivery_position))
        np.testing.assert_allclose(
            traffic[robot_row, task_column], expected, rtol=0.0, atol=0.0)


class NeuralAgentTests(unittest.TestCase):
    @staticmethod
    def decision(first=0.0, second=0.0):
        state = np.zeros(CONTEXT_DIM, np.float32)
        actions = np.zeros((ACTION_DIM, ACTION_FEATURE_DIM), np.float32)
        actions[0, 0] = first
        actions[1, 0] = second
        mask = np.zeros(ACTION_DIM, bool)
        mask[:2] = True
        for value in (state, actions, mask):
            value.setflags(write=False)
        return NeuralDecision(state, actions, mask)

    def test_exact_masked_expected_target_not_selected_action(self):
        config = SarsaV4Config(
            n_step=1, batch_size=1, target_sync_transitions=1,
            epsilon_start=0.2, epsilon_end=0.2)
        agent = SarsaV4Agent("count", config=config, seed=1)
        with torch.no_grad():
            for network in (agent.online, agent.target):
                network.linear_prior.zero_()
                network.linear_prior[CONTEXT_DIM] = 1.0
        next_decision = self.decision(1.0, 3.0)
        probabilities = agent.policy_probabilities(
            next_decision, training=True)
        np.testing.assert_allclose(
            probabilities[:2], [0.1, 0.9], rtol=0.0, atol=1e-12)
        self.assertTrue(np.all(probabilities[~next_decision.mask] == 0.0))
        agent.observe_transition(
            self.decision(), 0, 0.0, next_decision)
        self.assertAlmostEqual(agent.last_mean_absolute_td, 2.8, places=5)

    def test_n_step_terminal_flush_and_checkpoint_boundary(self):
        config = SarsaV4Config(
            n_step=5, batch_size=32, target_sync_transitions=64)
        agent = SarsaV4Agent("utility_v2", config=config, seed=8)
        decision = self.decision()
        for _ in range(10):
            agent.observe_transition(decision, 0, 0.0, decision)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.pt"
            with self.assertRaises(RuntimeError):
                agent.save(path)
            agent.observe_transition(decision, 0, 1.0, done=True)
            agent.end_episode()
            agent.save(path)
            restored = SarsaV4Agent.load(
                path, objective="utility_v2", device="cpu")
            self.assertEqual(agent.transition_updates, 11)
            self.assertEqual(
                agent.document()["state_sha256"],
                restored.document()["state_sha256"])
            self.assertEqual(agent.rng.random(), restored.rng.random())


class NeuralRuntimeTests(unittest.TestCase):
    def test_actual_outcomes_train_and_evaluation_is_frozen(self):
        robots, tasks, ctx = fixture(600.0, 2)
        config = SarsaV4Config(
            n_step=1, batch_size=1, target_sync_transitions=2,
            epsilon_start=0.0, epsilon_end=0.0)
        agent = SarsaV4Agent("count", config=config)
        scheduler = SarsaV4Scheduler(
            agent=agent, training=True, physical=False)
        scheduler.observe(tasks, 600.0)
        first = scheduler.assign(tasks, robots, ctx).assignments[0]
        first.task.status = TaskStatus.ASSIGNED
        first.task.assignment_time = 600.0
        first.task.ideal_distance = first.estimated_cost
        robots[first.robot_id]["current_task"] = first.task
        robots[first.robot_id]["state"] = RobotState.EN_ROUTE_PICKUP
        scheduler.on_assignment_committed(first)
        scheduler.observe(tasks, 600.0)
        second = scheduler.assign(
            tasks, robots, context(robots, 600.0)).assignments[0]
        scheduler.on_assignment_rejected(second, "unreachable")
        report = scheduler.finalize(tasks, 1800.0)
        self.assertEqual(report["transition_updates"], 2)
        self.assertEqual(report["rejected_count"], 1)
        frozen = copy.deepcopy(agent.document())
        evaluator = SarsaV4Scheduler(agent=agent, training=False)
        evaluator.observe(tasks, 600.0)
        evaluator.assign(tasks, robots, context(robots, 600.0))
        self.assertEqual(
            evaluator.finalize(tasks, 1800.0)["transition_updates"], 0)
        self.assertEqual(agent.document(), frozen)

    def test_factory_opt_in_is_versioned_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            SarsaV4Agent("count").save(checkpoint)
            with patch.dict(os.environ, {
                    "SMART_FACTORY_SARSA_V4": "1",
                    "SMART_FACTORY_SARSA_V3": "0"}, clear=False):
                scheduler = create_scheduler(
                    "SARSA", str(checkpoint), allow_safe_fallback=False)
                self.assertIsInstance(scheduler, SarsaV4Scheduler)
            with patch.dict(os.environ, {
                    "SMART_FACTORY_SARSA_V4": "1",
                    "SMART_FACTORY_SARSA_V3": "1"}, clear=False):
                with self.assertRaises(ModelValidationError):
                    create_scheduler(
                        "SARSA", str(checkpoint), allow_safe_fallback=True)


class NeuralWorkflowContractTests(unittest.TestCase):
    def test_three_reviews_bind_identical_sources(self):
        holder = tempfile.TemporaryDirectory(
            prefix="sarsa_v4_contract_", dir=ROOT / "results")
        root = Path(holder.name)
        root.rmdir()
        try:
            protocol = audit.initialize(root)
            self.assertFalse(protocol["nonregression_validation"])
            files = ["scripts/neural_sarsa_workflow_contract.py"]
            for index in range(3):
                audit.record_review(
                    root, 1, f"finding {index}", f"check {index}", files)
            audit.require_reviews(root, 1)
            with self.assertRaises(RuntimeError):
                audit.require_reviews(root, 2)
        finally:
            holder.cleanup()


if __name__ == "__main__":
    unittest.main()
