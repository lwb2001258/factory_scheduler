import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "controllers" / "factory_supervisor"
sys.path.insert(0, os.fspath(SUPERVISOR))

from config import (  # noqa: E402
    CHARGING_STATIONS, FULL_BATTERY_THRESHOLD, PARKING_SPOTS, TIMESTEP,
    RobotState, TaskStatus,
)
from headless_training_runtime import (  # noqa: E402
    HEADLESS_DYNAMICS_VERSION, HeadlessFactoryRuntime, HeadlessRuntimeConfig,
)
from ai_training_integration import (  # noqa: E402
    AI_ALGORITHM_NAMES, AI_TRAINING_CAPABILITIES, capability_report,
    run_scheduler_episode, train_dqn_agent, train_pairwise_ppo,
    train_sarsa_agent,
)
from rl_environment import RLEnvironmentConfig, SchedulingEnvironment  # noqa: E402
from rl_agents import DQNAgent, DQNConfig, SarsaAgent  # noqa: E402
from schedulers import (  # noqa: E402
    Assignment, BaseScheduler, PPONetwork, SchedulerResult, SchedulingContext,
)
from task_generator import TransportTask  # noqa: E402
from training_scenarios import FactoryAStarCostOracle  # noqa: E402
from config import STORAGE_AREAS, WORKSTATIONS  # noqa: E402


def _robot(position=None, battery=100.0):
    return {
        1: {
            "position": tuple(position or PARKING_SPOTS[1]),
            "heading": 0.0,
            "state": RobotState.IDLE,
            "battery": battery,
            "current_task": None,
            "has_task": False,
            "goal_location": None,
            "tasks_completed": 0,
            "total_distance": 0.0,
        }
    }


def _task(task_id=1):
    return TransportTask(
        task_id, "S1", "WS1", STORAGE_AREAS["S1"],
        WORKSTATIONS["WS1"], 0.0, priority=2)


def _runtime(robots, tasks, seed=10, config=None):
    oracle = FactoryAStarCostOracle(robots, len(robots))
    return HeadlessFactoryRuntime(
        robots, tasks,
        SchedulingContext(current_time=0.0, path_cost_provider=oracle),
        seed=seed, config=config)


class HeadlessRuntimeContractTests(unittest.TestCase):
    def test_runtime_uses_webots_timestep_and_declares_fidelity(self):
        runtime = _runtime(_robot(), [_task()])
        self.assertAlmostEqual(
            runtime.config.timestep_seconds, TIMESTEP / 1000.0)
        self.assertEqual(
            runtime.context.configuration["runtime_mode"],
            "headless_webots_logic")
        self.assertEqual(
            runtime.context.configuration["dynamics_version"],
            HEADLESS_DYNAMICS_VERSION)
        self.assertEqual(
            runtime.context.configuration["physics_fidelity"],
            "business_logic_only")

    def test_assignment_runs_pickup_delivery_state_machine(self):
        robots = _robot()
        task = _task()
        runtime = _runtime(
            robots, [task], config=HeadlessRuntimeConfig(linear_speed=4.0))
        cost = runtime.context.path_cost_provider(1, task)
        committed = runtime.dispatch(Assignment(1, task, cost))
        self.assertEqual(committed["type"], "assignment_committed")
        self.assertEqual(task.status, TaskStatus.ASSIGNED)
        self.assertEqual(robots[1]["state"], RobotState.EN_ROUTE_PICKUP)

        seen = []
        for _ in range(20):
            seen.extend(runtime.advance_until_event(max_seconds=30.0))
            if task.status == TaskStatus.COMPLETED:
                break
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(task.pickup_time)
        self.assertIsNotNone(task.completion_time)
        self.assertGreater(task.completion_time, task.pickup_time)
        self.assertEqual(robots[1]["state"], RobotState.IDLE)
        self.assertIsNone(robots[1]["current_task"])
        self.assertEqual(robots[1]["tasks_completed"], 1)
        self.assertTrue(any(event["type"] == "task_picked_up"
                            for event in seen))
        self.assertTrue(any(event["type"] == "task_completed"
                            for event in seen))
        self.assertGreater(runtime.telemetry["distance_travelled"], 0.0)

    def test_dynamic_pickup_failure_does_not_commit_domain_state(self):
        robots = _robot()
        task = _task()
        runtime = _runtime(robots, [task])
        with mock.patch.object(runtime, "_plan", return_value=None):
            result = runtime.dispatch(Assignment(1, task, 1.0))
        self.assertEqual(result["type"], "assignment_rejected")
        self.assertEqual(result["reason"], "pickup_path_unreachable")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertIsNone(task.assigned_robot)
        self.assertEqual(robots[1]["state"], RobotState.IDLE)
        self.assertIn((1, task.task_id), runtime.context.failed_pairs)

    def test_low_battery_station_arrival_uses_five_second_seeded_swap(self):
        station = next(iter(CHARGING_STATIONS.values()))
        first = _runtime(_robot(station, battery=100.0), [], seed=91)
        second = _runtime(_robot(station, battery=100.0), [], seed=91)
        first.robots[1]["battery"] = 24.0
        second.robots[1]["battery"] = 24.0
        with mock.patch.object(first, "_plan", return_value=None):
            first._send_to_charging(1)
        with mock.patch.object(second, "_plan", return_value=None):
            second._send_to_charging(1)
        self.assertEqual(first.robots[1]["state"], RobotState.CHARGING)
        self.assertEqual(second.robots[1]["state"], RobotState.CHARGING)
        for _ in range(math.ceil(5.0 / first.config.timestep_seconds) + 1):
            first.tick()
            second.tick()
        self.assertEqual(first.robots[1]["state"], RobotState.IDLE)
        self.assertEqual(first.robots[1]["battery"],
                         second.robots[1]["battery"])
        self.assertGreaterEqual(first.robots[1]["battery"],
                                FULL_BATTERY_THRESHOLD)
        self.assertLessEqual(first.robots[1]["battery"], 100.0)
        self.assertEqual(first.telemetry["charge_swaps"], 1)

    def test_return_to_charge_does_not_drain_under_current_supervisor_rule(self):
        robots = _robot(battery=24.0)
        runtime = _runtime(robots, [], seed=92)
        if robots[1]["state"] != RobotState.RETURNING_TO_CHARGE:
            self.skipTest("fixture starts within charging arrival tolerance")
        before = robots[1]["battery"]
        runtime.tick()
        self.assertEqual(robots[1]["battery"], before)

    def test_invalid_runtime_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            HeadlessRuntimeConfig(timestep_seconds=float("nan"))
        duplicate = [_task(1), _task(1)]
        with self.assertRaises(ValueError):
            _runtime(_robot(), duplicate)
        broken = _robot()
        broken[1]["position"] = (float("nan"), 0.0)
        with self.assertRaises(ValueError):
            _runtime(broken, [])


class SchedulingEnvironmentRuntimeTests(unittest.TestCase):
    def test_uninitialised_headless_telemetry_is_not_labelled_webots(self):
        environment = SchedulingEnvironment(simulation_mode="headless")
        telemetry = environment.runtime_telemetry()
        self.assertEqual(telemetry["runtime_mode"],
                         "headless_webots_logic")
        self.assertEqual(telemetry["physics_fidelity"],
                         "business_logic_only")
        self.assertFalse(telemetry["initialized"])

    def test_abstract_alias_uses_headless_runtime_and_real_distance(self):
        robots = _robot()
        task = _task()
        oracle = FactoryAStarCostOracle(robots, 1)
        environment = SchedulingEnvironment(
            RLEnvironmentConfig(max_robots=1, max_tasks=1),
            simulation_mode="abstract",
            runtime_config=HeadlessRuntimeConfig(linear_speed=4.0))
        _, metadata = environment.reset(
            robots, [task], SchedulingContext(path_cost_provider=oracle), seed=7)
        self.assertEqual(metadata["simulation_mode"], "headless")
        self.assertEqual(metadata["dynamics_version"], HEADLESS_DYNAMICS_VERSION)
        action = int(environment.get_action_mask().nonzero()[0][0])
        _, reward, terminated, truncated, info = environment.step(action)
        self.assertFalse(truncated)
        self.assertTrue(terminated)
        self.assertTrue(math.isfinite(reward))
        self.assertGreater(info["distance_travelled"], 0.0)
        self.assertEqual(info["completed_this_step"], 1)
        self.assertFalse(any("_abstract_execution" in state
                             for state in environment._robots.values()))
        telemetry = environment.runtime_telemetry()
        self.assertEqual(telemetry["runtime_mode"], "headless_webots_logic")
        self.assertTrue(telemetry["initialized"])
        self.assertEqual(telemetry["tasks_completed"], 1)

    def test_webots_snapshot_mode_never_mutates_low_battery_state(self):
        robots = _robot(battery=10.0)
        before = dict(robots[1])
        environment = SchedulingEnvironment(
            RLEnvironmentConfig(max_robots=1, max_tasks=1),
            simulation_mode="webots")
        environment.reset(robots, [_task()], SchedulingContext(), seed=8)
        self.assertEqual(robots[1], before)
        self.assertIsNone(environment._runtime)
        self.assertEqual(
            environment.runtime_telemetry()["runtime_mode"],
            "webots_snapshot")
        with self.assertRaises(RuntimeError):
            environment.step(environment.no_op_action)

    def test_headless_snapshot_copy_preserves_active_task_identity(self):
        task = _task()
        task.status = TaskStatus.ASSIGNED
        task.assigned_robot = 1
        task.assignment_time = 0.0
        robots = _robot()
        robots[1]["state"] = RobotState.EN_ROUTE_PICKUP
        robots[1]["current_task"] = task
        robots[1]["has_task"] = True
        robots[1]["goal_location"] = task.pickup_location
        oracle = FactoryAStarCostOracle(robots, 1)
        environment = SchedulingEnvironment(
            RLEnvironmentConfig(max_robots=1, max_tasks=1),
            simulation_mode="headless")
        environment.reset(
            robots, [task], SchedulingContext(path_cost_provider=oracle), seed=9)
        self.assertIs(
            environment._robots[1]["current_task"], environment._tasks[0])
        self.assertIsNot(environment._tasks[0], task)


class AITrainingIntegrationTests(unittest.TestCase):
    def test_capability_registry_covers_exactly_ten_ai_algorithms(self):
        self.assertEqual(tuple(AI_TRAINING_CAPABILITIES), AI_ALGORITHM_NAMES)
        self.assertEqual(len(AI_ALGORITHM_NAMES), 10)
        report = capability_report()
        self.assertEqual(tuple(report), AI_ALGORITHM_NAMES)
        self.assertTrue(all(item["headless_execution"]
                            for item in report.values()))
        self.assertEqual(
            report["LearnedHungarian"]["learning_method"],
            "supervised_regression")
        self.assertEqual(report["CQL"]["learning_method"],
                         "offline_conservative_value_learning")

    def test_scheduler_adapter_rejects_duplicate_matching_without_commit(self):
        class DuplicateScheduler(BaseScheduler):
            def __init__(self):
                super().__init__("Duplicate")

            def assign_task(self, pending_tasks, robot_states,
                            congestion_map=None):
                return None

            def assign(self, pending_tasks, robot_states, context=None):
                task = pending_tasks[0]
                duplicate = [Assignment(1, task, 1.0),
                             Assignment(1, task, 1.0)]
                return SchedulerResult(
                    assignments=duplicate, is_feasible=True,
                    algorithm_name=self.name)

        robots = _robot()
        task = _task()
        oracle = FactoryAStarCostOracle(robots, 1)
        environment = SchedulingEnvironment(
            RLEnvironmentConfig(max_robots=1, max_tasks=1),
            simulation_mode="headless")
        environment.reset(
            robots, [task], SchedulingContext(path_cost_provider=oracle), seed=12)
        _, reward, _, _, info = environment.step_scheduler(DuplicateScheduler())
        self.assertEqual(reward, environment.reward_config.invalid_action)
        self.assertTrue(info["scheduler_output_rejected"])
        self.assertEqual(info["reason"], "duplicate_robot_assignment")
        self.assertEqual(environment._tasks[0].status, TaskStatus.PENDING)
        self.assertEqual(
            environment.runtime_telemetry()["assignments_committed"], 0)

    def test_legacy_sarsa_dqn_and_pairwise_ppo_train_headlessly(self):
        environment = SchedulingEnvironment(simulation_mode="headless")
        sarsa = SarsaAgent(
            environment.action_dim, environment.no_op_action, seed=21)
        sarsa_report = train_sarsa_agent(
            sarsa, (9201,), max_steps=3)
        self.assertGreater(sarsa_report["updates"], 0)

        dqn = DQNAgent(
            environment.observation_dim, environment.action_dim,
            environment.no_op_action,
            DQNConfig(hidden_size=8, batch_size=2, warmup_steps=2,
                      replay_capacity=16, target_update_interval=2),
            seed=22)
        dqn_report = train_dqn_agent(dqn, (9202,), max_steps=3)
        self.assertGreater(dqn_report["updates"], 0)

        numpy_state = np.random.get_state()
        try:
            np.random.seed(23)
            ppo = PPONetwork(
                environment.observation_dim, environment.action_dim, 8)
        finally:
            np.random.set_state(numpy_state)
        before = ppo.W_policy.copy()
        ppo_report = train_pairwise_ppo(
            ppo, (9203,), max_steps=3, update_epochs=1)
        self.assertGreater(ppo_report["updates"], 0)
        self.assertFalse(np.array_equal(before, ppo.W_policy))
        for report in (sarsa_report, dqn_report, ppo_report):
            self.assertEqual(
                report["runtime"]["dynamics_version"],
                HEADLESS_DYNAMICS_VERSION)
            self.assertTrue(math.isfinite(report["mean_return"]))

    def test_pairwise_ppo_training_is_seeded_and_fails_closed(self):
        environment = SchedulingEnvironment(simulation_mode="headless")
        numpy_state = np.random.get_state()
        try:
            np.random.seed(24)
            first = PPONetwork(
                environment.observation_dim, environment.action_dim, 8)
            np.random.seed(24)
            second = PPONetwork(
                environment.observation_dim, environment.action_dim, 8)
        finally:
            np.random.set_state(numpy_state)
        first_report = train_pairwise_ppo(
            first, (9210, 9211), max_steps=3, update_epochs=2)
        second_report = train_pairwise_ppo(
            second, (9210, 9211), max_steps=3, update_epochs=2)
        self.assertEqual(first_report, second_report)
        for name in ("W1", "b1", "W2", "b2", "W_policy", "b_policy",
                     "W_value", "b_value"):
            np.testing.assert_array_equal(
                getattr(first, name), getattr(second, name))

        broken = PPONetwork(
            environment.observation_dim, environment.action_dim, 8)
        broken.W_policy[0, 0] = np.nan
        with self.assertRaises(ValueError):
            train_pairwise_ppo(
                broken, (9212,), max_steps=1, update_epochs=1)

    def test_all_ten_ai_scheduler_artifacts_execute_in_one_runtime(self):
        from advanced_rl_agents import (
            CQLAgent, CQLConfig, QRDQNAgent, QRDQNConfig, RainbowConfig,
            RainbowDQNAgent,
        )
        from bandit_scheduler import LinUCBModel
        from graph_ppo_scheduler import GraphPPOConfig, GraphPPOModel
        from learning_scheduler import (
            GRAPH_FEATURE_NAMES, PAIR_FEATURE_NAMES, GraphEdgeImitationModel,
            RidgeCostModel,
        )
        from schedulers import create_scheduler

        environment = SchedulingEnvironment(simulation_mode="headless")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                name: root / f"{name}.npz" for name in AI_ALGORITHM_NAMES
            }
            paths["SARSA"] = root / "sarsa.json"
            paths["DQN"] = root / "dqn.pkl"

            RidgeCostModel(
                np.zeros(len(PAIR_FEATURE_NAMES)),
                np.ones(len(PAIR_FEATURE_NAMES)),
                np.zeros(len(PAIR_FEATURE_NAMES)), 1.0, 2,
                "baseline_pair_cost").save(paths["LearnedHungarian"])
            GraphEdgeImitationModel(
                np.zeros(len(GRAPH_FEATURE_NAMES)),
                np.ones(len(GRAPH_FEATURE_NAMES)),
                np.zeros(len(GRAPH_FEATURE_NAMES)), 0.0, 2).save(
                    paths["GraphImitation"])

            numpy_state = np.random.get_state()
            try:
                np.random.seed(31)
                ppo = PPONetwork(
                    environment.observation_dim, environment.action_dim, 8)
            finally:
                np.random.set_state(numpy_state)
            ppo.save(paths["PPO_RL"])
            SarsaAgent(
                environment.action_dim, environment.no_op_action,
                seed=32).save(paths["SARSA"])
            DQNAgent(
                environment.observation_dim, environment.action_dim,
                environment.no_op_action,
                DQNConfig(hidden_size=8, replay_capacity=8,
                          batch_size=2, warmup_steps=2), seed=33).save(
                    paths["DQN"])
            GraphPPOModel(
                GraphPPOConfig(hidden_size=8, update_epochs=1), seed=34).save(
                    paths["GraphPPO"])
            RainbowDQNAgent(
                environment.observation_dim, environment.action_dim,
                environment.no_op_action,
                RainbowConfig(hidden_size=8, atoms=7, replay_capacity=8,
                              batch_size=2, warmup_steps=2), seed=35).save(
                    paths["RainbowDQN"])
            QRDQNAgent(
                environment.observation_dim, environment.action_dim,
                environment.no_op_action,
                QRDQNConfig(hidden_size=8, quantiles=4, replay_capacity=8,
                            batch_size=2, warmup_steps=2), seed=36).save(
                    paths["QRDQN"])
            CQLAgent(
                environment.observation_dim, environment.action_dim,
                environment.no_op_action,
                CQLConfig(hidden_size=8, replay_capacity=8,
                          batch_size=2, warmup_steps=2), seed=37).save(
                    paths["CQL"])
            LinUCBModel.create(("FCFS", "Hungarian"), alpha=0.1).save(
                paths["LinUCB"])

            results = {}
            for index, name in enumerate(AI_ALGORITHM_NAMES):
                scheduler = create_scheduler(
                    name, str(paths[name]), seed=40 + index,
                    allow_safe_fallback=False)
                if hasattr(scheduler, "timeout_seconds"):
                    scheduler.timeout_seconds = 10.0
                result = run_scheduler_episode(
                    scheduler, 9300 + index, max_decisions=2)
                self.assertEqual(result["rejected_outputs"], 0, name)
                self.assertGreater(
                    result["runtime"]["assignments_committed"], 0, name)
                results[name] = result
            self.assertEqual(tuple(results), AI_ALGORITHM_NAMES)


if __name__ == "__main__":
    unittest.main()
