import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "controllers" / "factory_supervisor"
if str(SUPERVISOR) not in sys.path:
    sys.path.insert(0, str(SUPERVISOR))

from advanced_ai_common import (  # noqa: E402
    load_checkpoint, masked_argmax, masked_softmax, save_checkpoint,
    validate_action_mask,
)
from advanced_rl_agents import (  # noqa: E402
    CQLAgent, CQLConfig, NStepAccumulator, OfflineTransitionDataset,
    PrioritizedReplayBuffer, QRDQNAgent, QRDQNConfig, RainbowConfig,
    RainbowDQNAgent, ReplayTransition,
)
from advanced_rl_schedulers import (  # noqa: E402
    CQLScheduler, QRDQNScheduler, RainbowDQNScheduler, ValuePolicyScheduler,
)
from advanced_rl_training import (  # noqa: E402
    collect_cql_dataset, train_online_value_agent,
)
from config import RL_ENVIRONMENT_VERSION, RobotState  # noqa: E402
from rl_environment import RLEnvironmentConfig, SchedulingEnvironment  # noqa: E402
from graph_ppo_scheduler import (  # noqa: E402
    GRAPH_PPO_ALGORITHM, GraphPPOConfig, GraphPPOModel, GraphPPOScheduler,
    graph_policy_inputs, train_graph_ppo,
)
from bandit_scheduler import (  # noqa: E402
    BANDIT_CONTEXT_FEATURES, DEFAULT_BANDIT_ARMS, LinUCBModel,
    LinUCBScheduler, bandit_context_vector, train_linucb,
)
from rl_schedulers import RLSchedulerSafetyWrapper  # noqa: E402
from schedulers import ModelValidationError, SchedulingContext  # noqa: E402
from schedulers import create_scheduler  # noqa: E402
from task_generator import TransportTask  # noqa: E402


def _metadata(algorithm="TestPolicy", state_dim=3, action_dim=4):
    return {
        "algorithm": algorithm,
        "environment_version": RL_ENVIRONMENT_VERSION,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }


def _task(task_id=1):
    return TransportTask(
        task_id=task_id,
        pickup_location="Storage_A",
        delivery_location="Assembly_1",
        pickup_position=(-3.0, 1.5),
        delivery_position=(0.0, 0.0),
        priority=1,
        arrival_time=0.0,
    )


def _robots():
    return {
        0: {
            "position": (-4.0, 0.0),
            "state": RobotState.IDLE,
            "battery": 100.0,
            "current_task": None,
        }
    }


class _FinitePolicy:
    def __init__(self, action_dim, bad=False):
        self.action_dim = action_dim
        self.bad = bad

    def action_values(self, state):
        values = np.arange(self.action_dim, dtype=np.float64)
        if self.bad:
            values[0] = np.nan
        return values


class _SlowPolicy(_FinitePolicy):
    def action_values(self, state):
        time.sleep(0.01)
        return super().action_values(state)


class AdvancedContractTests(unittest.TestCase):
    def test_npz_checkpoint_round_trip_and_algorithm_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_checkpoint(path, _metadata(), {"weights": np.eye(3)})
            metadata, arrays = load_checkpoint(
                path, expected_algorithm="TestPolicy",
                expected_state_dim=3, expected_action_dim=4)
            self.assertEqual(metadata["algorithm"], "TestPolicy")
            np.testing.assert_array_equal(arrays["weights"], np.eye(3))
            with self.assertRaises(ModelValidationError):
                load_checkpoint(path, expected_algorithm="WrongPolicy")

    def test_checkpoint_rejects_unsafe_arrays_and_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            with self.assertRaises(ValueError):
                save_checkpoint(path, _metadata(), {
                    "weights": np.asarray([np.nan])})
            with self.assertRaises(ValueError):
                save_checkpoint(path, _metadata(), {
                    "weights": np.asarray([{"unsafe": True}], dtype=object)})
            with self.assertRaises(ValueError):
                save_checkpoint(path, _metadata(), {
                    "weights": np.asarray([1 + 2j])})
            metadata = _metadata()
            metadata["metric"] = float("nan")
            with self.assertRaises(ValueError):
                save_checkpoint(path, metadata, {"weights": np.ones(1)})
            save_checkpoint(path, _metadata(), {"weights": np.ones(1)})
            with self.assertRaises(ModelValidationError):
                load_checkpoint(
                    path, expected_algorithm="TestPolicy",
                    expected_state_dim=99)

    def test_missing_and_corrupt_checkpoint_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.npz"
            with self.assertRaises(ModelValidationError):
                load_checkpoint(missing, expected_algorithm="TestPolicy")
            corrupt = Path(directory) / "corrupt.npz"
            corrupt.write_bytes(b"not-an-npz")
            with self.assertRaises(ModelValidationError):
                load_checkpoint(corrupt, expected_algorithm="TestPolicy")

    def test_mask_helpers_reject_empty_nonfinite_and_wrong_shape(self):
        np.testing.assert_allclose(
            masked_softmax([1000.0, 999.0, -1000.0], [1, 1, 0]),
            [0.7310585786, 0.2689414214, 0.0])
        self.assertEqual(masked_argmax([1.0, 20.0, 3.0], [1, 0, 1]), 2)
        with self.assertRaises(ValueError):
            validate_action_mask([0, 0, 0])
        with self.assertRaises(ValueError):
            masked_argmax([1.0, np.nan], [1, 1])
        with self.assertRaises(ValueError):
            masked_softmax([[1.0, 2.0]], [1, 1])

    def test_value_policy_uses_environment_mask_and_shared_validation(self):
        scheduler = ValuePolicyScheduler(
            "TestPolicy", _FinitePolicy(161))
        result = scheduler.assign(
            [_task()], _robots(), SchedulingContext(current_time=0.0))
        self.assertTrue(result.is_feasible)
        self.assertEqual(result.assignments[0].robot_id, 0)
        self.assertEqual(result.assignments[0].task.task_id, 1)

    def test_nonfinite_policy_fails_closed_and_wrapper_falls_back(self):
        policy = ValuePolicyScheduler(
            "TestPolicy", _FinitePolicy(161, bad=True))
        result = policy.assign([_task()], _robots())
        self.assertFalse(result.is_feasible)
        self.assertEqual(
            result.diagnostics["reason"], "policy_inference_error:ValueError")
        wrapped = RLSchedulerSafetyWrapper(policy)
        fallback = wrapped.assign([_task()], _robots())
        self.assertTrue(fallback.is_feasible)
        self.assertEqual(fallback.algorithm_name,
                         "TestPolicy_FALLBACK_HUNGARIAN")

    def test_low_battery_and_duplicate_tasks_are_rejected(self):
        scheduler = ValuePolicyScheduler(
            "TestPolicy", _FinitePolicy(161))
        robots = _robots()
        robots[0]["battery"] = 5.0
        low_battery = scheduler.assign([_task()], robots)
        self.assertFalse(low_battery.is_feasible)
        duplicate = scheduler.assign([_task(7), _task(7)], _robots())
        self.assertFalse(duplicate.is_feasible)
        self.assertEqual(
            duplicate.diagnostics["reason"],
            "policy_inference_error:ValueError")

    def test_policy_timeout_uses_hungarian_fallback(self):
        policy = ValuePolicyScheduler(
            "SlowPolicy", _SlowPolicy(161))
        wrapped = RLSchedulerSafetyWrapper(
            policy, timeout_seconds=0.001)
        result = wrapped.assign([_task()], _robots())
        self.assertTrue(result.is_feasible)
        self.assertEqual(result.algorithm_name,
                         "SlowPolicy_FALLBACK_HUNGARIAN")
        self.assertEqual(result.diagnostics["rl_failure"],
                         "inference_timeout")


class AdvancedValueAgentTests(unittest.TestCase):
    STATE_DIM = 6
    ACTION_DIM = 4
    NO_OP = 3

    @staticmethod
    def _transition(index, done=False):
        state = np.asarray([
            index, index % 2, 0.1, -0.1, 0.5, 1.0], dtype=np.float32)
        next_state = state + 0.01
        mask = np.asarray([1, 1, 0, 0], dtype=bool)
        next_mask = np.asarray([1, 0, 1, 0], dtype=bool)
        action = index % 2
        reward = float((index % 5)-2)
        return state, mask, action, reward, next_state, next_mask, done

    def test_n_step_return_and_terminal_flush(self):
        accumulator = NStepAccumulator(3, 0.5)
        self.assertEqual(accumulator.add(*self._transition(0)), [])
        self.assertEqual(accumulator.add(*self._transition(1)), [])
        emitted = accumulator.add(*self._transition(2))
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0].reward, -2-0.5+0.0)
        flushed = accumulator.add(*self._transition(3, done=True))
        self.assertEqual(len(flushed), 3)
        self.assertTrue(flushed[-1].done)

    def test_prioritized_replay_sampling_and_priority_update(self):
        replay = PrioritizedReplayBuffer(8, seed=7)
        for index in range(4):
            row = self._transition(index)
            replay.add(ReplayTransition(
                row[0], row[1], row[2], row[3], row[4], row[5],
                row[6], 0.99), priority=index+1)
        batch = replay.sample(3, beta=0.4)
        self.assertEqual(batch[0].shape, (3, self.STATE_DIM))
        self.assertEqual(batch[1].shape, (3, self.ACTION_DIM))
        replay.update_priorities(batch[-2], np.ones(3))
        with self.assertRaises(ValueError):
            replay.update_priorities([99], [1.0])

    def _fill_and_train(self, agent, count=12):
        for index in range(count):
            row = self._transition(index, done=(index % 6 == 5))
            agent.remember(*row)
        losses = [agent.train_step() for _ in range(3)]
        finite = [loss for loss in losses if loss is not None]
        self.assertTrue(finite)
        self.assertTrue(np.all(np.isfinite(finite)))
        return finite

    def test_rainbow_c51_training_mask_and_checkpoint_round_trip(self):
        config = RainbowConfig(
            hidden_size=12, atoms=11, value_min=-10, value_max=10,
            n_step=2, replay_capacity=64, batch_size=4,
            warmup_steps=4, target_update_interval=2)
        agent = RainbowDQNAgent(
            self.STATE_DIM, self.ACTION_DIM, self.NO_OP, config, seed=11)
        self._fill_and_train(agent)
        state, mask, *_ = self._transition(0)
        self.assertIn(agent.select_action(state, mask), (0, 1))
        probabilities = agent.online.forward(state)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rainbow.npz"
            before = agent.action_values(state)
            agent.save(path)
            loaded = RainbowDQNAgent.load(
                path, self.STATE_DIM, self.ACTION_DIM, self.NO_OP, seed=99)
            np.testing.assert_allclose(loaded.action_values(state), before)
            self.assertEqual(loaded.optimizer.step_count,
                             loaded.training_step)

    def test_qrdqn_quantile_training_risk_and_checkpoint_round_trip(self):
        config = QRDQNConfig(
            hidden_size=12, quantiles=8, risk_fraction=0.5,
            n_step=2, replay_capacity=64, batch_size=4,
            warmup_steps=4, target_update_interval=2)
        agent = QRDQNAgent(
            self.STATE_DIM, self.ACTION_DIM, self.NO_OP, config, seed=12)
        self._fill_and_train(agent)
        state, mask, *_ = self._transition(0)
        self.assertIn(agent.select_action(state, mask), (0, 1))
        quantiles = agent.online.forward(state)
        self.assertEqual(quantiles.shape, (self.ACTION_DIM, 8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qrdqn.npz"
            before = agent.action_values(state)
            agent.save(path)
            loaded = QRDQNAgent.load(
                path, self.STATE_DIM, self.ACTION_DIM, self.NO_OP, seed=99)
            np.testing.assert_allclose(loaded.action_values(state), before)
            self.assertEqual(loaded.optimizer.step_count,
                             loaded.training_step)

    def test_cql_training_uses_only_legal_actions_and_round_trips(self):
        config = CQLConfig(
            hidden_size=12, conservative_weight=0.5,
            replay_capacity=64, batch_size=4, warmup_steps=4,
            target_update_interval=2)
        agent = CQLAgent(
            self.STATE_DIM, self.ACTION_DIM, self.NO_OP, config, seed=13)
        self._fill_and_train(agent)
        state, mask, *_ = self._transition(0)
        self.assertIn(agent.select_action(state, mask), (0, 1))
        bad = list(self._transition(0))
        bad[2] = 2
        with self.assertRaises(ValueError):
            agent.remember(*bad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cql.npz"
            before = agent.action_values(state)
            agent.save(path)
            loaded = CQLAgent.load(
                path, self.STATE_DIM, self.ACTION_DIM, self.NO_OP, seed=99)
            np.testing.assert_allclose(loaded.action_values(state), before)
            self.assertEqual(loaded.optimizer.step_count,
                             loaded.training_step)

    def test_offline_dataset_contract_and_round_trip(self):
        rows = [self._transition(index, done=(index == 3))
                for index in range(4)]
        dataset = OfflineTransitionDataset(
            np.stack([row[0] for row in rows]),
            np.stack([row[1] for row in rows]),
            np.asarray([row[2] for row in rows]),
            np.asarray([row[3] for row in rows]),
            np.stack([row[4] for row in rows]),
            np.stack([row[5] for row in rows]),
            np.asarray([row[6] for row in rows]),
            "Hungarian", (100, 101))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            dataset.save(path)
            loaded = OfflineTransitionDataset.load(path)
            np.testing.assert_array_equal(loaded.actions, dataset.actions)
            self.assertEqual(loaded.seeds, (100, 101))
        with self.assertRaises(ValueError):
            OfflineTransitionDataset(
                dataset.states, dataset.action_masks,
                np.asarray([2, 2, 2, 2]), dataset.rewards,
                dataset.next_states, dataset.next_action_masks,
                dataset.dones, "Hungarian", (100,))
        with self.assertRaises(ValueError):
            OfflineTransitionDataset(
                dataset.states, np.full_like(dataset.action_masks, np.nan,
                                             dtype=float),
                dataset.actions.astype(float), dataset.rewards,
                dataset.next_states, dataset.next_action_masks,
                dataset.dones, "Hungarian", (100.5,))

    def test_seeded_agents_are_reproducible_and_dimensions_are_guarded(self):
        rainbow_config = RainbowConfig(
            hidden_size=8, atoms=7, replay_capacity=16,
            batch_size=2, warmup_steps=2)
        first = RainbowDQNAgent(
            self.STATE_DIM, self.ACTION_DIM, self.NO_OP,
            rainbow_config, seed=91)
        second = RainbowDQNAgent(
            self.STATE_DIM, self.ACTION_DIM, self.NO_OP,
            rainbow_config, seed=91)
        state, *_ = self._transition(0)
        np.testing.assert_array_equal(
            first.action_values(state), second.action_values(state))
        with self.assertRaises(ValueError):
            CQLAgent(self.STATE_DIM, self.ACTION_DIM, self.ACTION_DIM)
        with self.assertRaises(ValueError):
            RainbowConfig(learning_rate=float("nan"))
        with self.assertRaises(ValueError):
            QRDQNConfig(quantiles=4.5)
        with self.assertRaises(ValueError):
            CQLConfig(conservative_weight=float("inf"))

    def test_value_scheduler_adapters_load_and_keep_assignment_legal(self):
        env_config = RLEnvironmentConfig(max_robots=1, max_tasks=1)
        environment = SchedulingEnvironment(
            env_config, simulation_mode="abstract")
        cases = (
            (RainbowDQNAgent, RainbowConfig(
                hidden_size=8, atoms=7, replay_capacity=8,
                batch_size=2, warmup_steps=2), RainbowDQNScheduler,
             "rainbow.npz"),
            (QRDQNAgent, QRDQNConfig(
                hidden_size=8, quantiles=4, replay_capacity=8,
                batch_size=2, warmup_steps=2), QRDQNScheduler,
             "qrdqn.npz"),
            (CQLAgent, CQLConfig(
                hidden_size=8, replay_capacity=8,
                batch_size=2, warmup_steps=2), CQLScheduler,
             "cql.npz"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (agent_type, config, scheduler_type, name) in enumerate(cases):
                agent = agent_type(
                    environment.observation_dim, environment.action_dim,
                    environment.no_op_action, config, seed=200+index)
                path = Path(directory) / name
                agent.save(path)
                scheduler = scheduler_type(
                    str(path), seed=300+index, env_config=env_config)
                result = scheduler.assign([_task()], _robots())
                self.assertTrue(result.is_feasible, scheduler.name)
                self.assertEqual(result.assignments[0].task.task_id, 1)

    def test_project_abstract_training_and_offline_collection_are_seeded(self):
        env = SchedulingEnvironment(simulation_mode="abstract")
        config = RainbowConfig(
            hidden_size=8, atoms=7, value_min=-20, value_max=20,
            n_step=2, replay_capacity=64, batch_size=2,
            warmup_steps=2, target_update_interval=2, epsilon=0.0)
        first = RainbowDQNAgent(
            env.observation_dim, env.action_dim, env.no_op_action,
            config, seed=501)
        second = RainbowDQNAgent(
            env.observation_dim, env.action_dim, env.no_op_action,
            config, seed=501)
        first_report = train_online_value_agent(
            first, (7001,), max_steps=6)
        second_report = train_online_value_agent(
            second, (7001,), max_steps=6)
        self.assertEqual(first_report, second_report)
        for name in first.online.params:
            np.testing.assert_array_equal(
                first.online.params[name], second.online.params[name])
        dataset = collect_cql_dataset((7101, 7102), max_steps=4)
        self.assertEqual(dataset.behavior_policy, "MaskedCostGreedy")
        self.assertEqual(dataset.seeds, (7101, 7102))
        cql = CQLAgent(
            dataset.state_dim, dataset.action_dim,
            dataset.action_dim-1,
            CQLConfig(hidden_size=8, replay_capacity=32,
                      batch_size=2, warmup_steps=2), seed=502)
        dataset.add_to(cql)
        self.assertTrue(math.isfinite(cql.train_step()))
        with self.assertRaises(ValueError):
            collect_cql_dataset((7101.5,), max_steps=1)


class GraphPPOTests(unittest.TestCase):
    def test_network_is_edge_permutation_equivariant_and_round_trips(self):
        model = GraphPPOModel(
            GraphPPOConfig(hidden_size=8, update_epochs=1), seed=801)
        rng = np.random.default_rng(802)
        features = rng.normal(size=(5, 28)).astype(np.float32)
        permutation = np.asarray([3, 0, 4, 1, 2])
        first_probabilities, first_value, _ = model.edge_policy(features)
        second_probabilities, second_value, _ = model.edge_policy(
            features[permutation])
        np.testing.assert_allclose(
            second_probabilities, first_probabilities[permutation], atol=1e-7)
        self.assertAlmostEqual(first_value, second_value, places=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph_ppo.npz"
            model.save(path)
            loaded = GraphPPOModel.load(path, seed=999)
            np.testing.assert_allclose(
                loaded.edge_policy(features)[0], first_probabilities)

    def test_scheduler_decodes_unique_feasible_hungarian_matching(self):
        from training_scenarios import factory_scenario

        robots, tasks, context = factory_scenario(803)
        model = GraphPPOModel(
            GraphPPOConfig(hidden_size=8, update_epochs=1), seed=804)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph_ppo.npz"
            model.save(path)
            result = GraphPPOScheduler(path).assign(tasks, robots, context)
        self.assertTrue(result.is_feasible)
        robot_ids = [item.robot_id for item in result.assignments]
        task_ids = [item.task.task_id for item in result.assignments]
        self.assertEqual(len(robot_ids), len(set(robot_ids)))
        self.assertEqual(len(task_ids), len(set(task_ids)))
        self.assertEqual(result.diagnostics["decoder"], "hungarian")

    def test_training_is_seeded_finite_and_uses_project_environment(self):
        config = GraphPPOConfig(
            hidden_size=8, update_epochs=1, learning_rate=0.001)
        first = GraphPPOModel(config, seed=805)
        second = GraphPPOModel(config, seed=805)
        first_report = train_graph_ppo(first, (806,), max_steps=5)
        second_report = train_graph_ppo(second, (806,), max_steps=5)
        self.assertEqual(first_report, second_report)
        self.assertGreater(first_report["decisions"], 0)
        self.assertTrue(math.isfinite(first_report["mean_loss"]))
        for name in first.network.params:
            np.testing.assert_array_equal(
                first.network.params[name], second.network.params[name])

    def test_action_mapping_handles_busy_lower_robot_slot(self):
        environment = SchedulingEnvironment(
            RLEnvironmentConfig(max_robots=2, max_tasks=2),
            simulation_mode="abstract")
        robots = _robots()
        robots[1] = dict(robots[0])
        robots[0]["state"] = RobotState.EN_ROUTE_PICKUP
        robots[0]["current_task"] = _task(99)
        state, _ = environment.reset(robots, [_task(1)])
        del state
        action = environment.action_for_pair(1, 1)
        self.assertEqual(action, 2)
        assignment = environment.assignment_for_action(action)
        self.assertEqual(assignment.robot_id, 1)

    def test_future_tasks_are_not_exposed_to_graph_policy(self):
        environment = SchedulingEnvironment(
            RLEnvironmentConfig(max_robots=1, max_tasks=2),
            simulation_mode="abstract")
        future = _task(2)
        future.arrival_time = 10.0
        environment.reset(
            _robots(), [future], SchedulingContext(current_time=0.0))
        _, visible_tasks, _ = environment.policy_snapshot()
        self.assertEqual(visible_tasks, ())
        mask = environment.get_action_mask()
        self.assertTrue(mask[environment.no_op_action])
        self.assertEqual(np.count_nonzero(mask), 1)

    def test_missing_or_wrong_semantics_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.npz"
            with self.assertRaises(ModelValidationError):
                GraphPPOModel.load(missing)
            wrong = Path(directory) / "wrong.npz"
            save_checkpoint(wrong, {
                "algorithm": GRAPH_PPO_ALGORITHM,
                "environment_version": RL_ENVIRONMENT_VERSION,
                "state_dim": 28,
                "action_dim": 1,
                "action_semantics": "flat_action",
            }, {"placeholder": np.ones(1)})
            with self.assertRaises(ModelValidationError):
                GraphPPOModel.load(wrong)


class LinUCBTests(unittest.TestCase):
    def test_context_is_deterministic_order_invariant_and_hides_future_tasks(self):
        context = SchedulingContext(current_time=5.0)
        robots = _robots()
        robots[1] = dict(robots[0])
        robots[1]["position"] = (2.0, 1.0)
        arrived = _task(1)
        future = _task(2)
        future.arrival_time = 50.0
        first = bandit_context_vector(
            robots, [future, arrived], context)
        second = bandit_context_vector(
            dict(reversed(list(robots.items()))), [arrived], context)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (len(BANDIT_CONTEXT_FEATURES),))

    def test_model_select_update_and_checkpoint_round_trip(self):
        model = LinUCBModel.create(("FCFS", "Hungarian"), alpha=0.0)
        vector = np.ones(len(BANDIT_CONTEXT_FEATURES))
        for _ in range(4):
            model.update(vector, "FCFS", -2.0)
            model.update(vector, "Hungarian", 1.0)
        arm, scores = model.select(vector)
        self.assertEqual(arm, "Hungarian")
        self.assertTrue(np.all(np.isfinite(scores)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linucb.npz"
            model.save(path)
            loaded = LinUCBModel.load(path)
            self.assertEqual(loaded.select(vector)[0], "Hungarian")
            self.assertEqual(loaded.training_samples, 8)

    def test_project_training_is_seeded_and_scheduler_preserves_arm_result(self):
        first, first_report = train_linucb((901, 902, 903), alpha=0.2)
        second, second_report = train_linucb((901, 902, 903), alpha=0.2)
        self.assertEqual(first_report, second_report)
        np.testing.assert_array_equal(first.covariance, second.covariance)
        self.assertEqual(first.arms, DEFAULT_BANDIT_ARMS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linucb.npz"
            first.save(path)
            scheduler = LinUCBScheduler(path)
            from training_scenarios import factory_scenario
            robots, tasks, context = factory_scenario(904)
            result = scheduler.assign(tasks, robots, context)
        self.assertTrue(result.is_feasible)
        self.assertIn(result.diagnostics["selected_arm"], DEFAULT_BANDIT_ARMS)
        self.assertEqual(result.algorithm_name, "LinUCB")

    def test_bad_covariance_and_arm_failure_fail_closed(self):
        model = LinUCBModel.create(("Hungarian",), alpha=0.1)
        model.covariance[0, 0, 0] = -1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            model.save(path)
            with self.assertRaises(ModelValidationError):
                LinUCBModel.load(path)

        class ExplodingArm:
            def assign(self, pending_tasks, robot_states, context):
                raise RuntimeError("arm failed")

            def on_assignment_committed(self, assignment):
                return None

            def on_assignment_rejected(self, assignment, reason):
                return None

            def reset(self):
                return None

        valid = LinUCBModel.create(("Hungarian",), alpha=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.npz"
            valid.save(path)
            scheduler = LinUCBScheduler(path)
            scheduler.arms["Hungarian"] = ExplodingArm()
            failed = scheduler.assign([_task()], _robots())
            self.assertFalse(failed.is_feasible)
            self.assertEqual(
                failed.diagnostics["reason"],
                "linucb_inference_error:RuntimeError")
            fallback = RLSchedulerSafetyWrapper(scheduler).assign(
                [_task()], _robots())
            self.assertTrue(fallback.is_feasible)
            self.assertEqual(fallback.algorithm_name,
                             "LinUCB_FALLBACK_HUNGARIAN")


class AdvancedWorkflowIntegrationTests(unittest.TestCase):
    def test_factory_has_explicit_fallback_for_every_new_algorithm(self):
        for name in ("GraphPPO", "RainbowDQN", "QRDQN", "CQL", "LinUCB"):
            scheduler = create_scheduler(name, model_path=None)
            self.assertEqual(
                scheduler.name, f"{name}_FALLBACK_HUNGARIAN")
            with self.assertRaises(ModelValidationError):
                create_scheduler(
                    name, model_path=None, allow_safe_fallback=False)

    def test_workflow_smoke_writes_loadable_candidates_and_strict_report(self):
        scripts = ROOT/"scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from run_advanced_ai_workflow import run_workflow
        from run_experiments import SCHEDULER_TYPES

        for name in ("GraphPPO", "RainbowDQN", "QRDQN", "CQL", "LinUCB"):
            self.assertIn(name, SCHEDULER_TYPES)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"workflow"
            report = run_workflow(
                output_dir=output,
                train_seeds=(9301, 9302),
                validation_seeds=(109301,),
                max_steps=6,
                offline_updates=3)
            self.assertTrue(report["research_candidates_ready"])
            self.assertFalse(report["production_promotion"])
            self.assertTrue(all(report["gates"].values()))
            document = json.loads(
                (output/"workflow_report.json").read_text(encoding="utf-8"))
            self.assertEqual(document["status"], "research_candidates_ready")
            self.assertFalse(set(document["data"]["training_seeds"]) &
                             set(document["data"]["validation_seeds"]))
            for path in document["artifacts"].values():
                self.assertTrue(Path(path).is_file())
            artifact_by_scheduler = {
                name: document["artifacts"][name]
                for name in (
                    "GraphPPO", "RainbowDQN", "QRDQN", "CQL", "LinUCB")}
            from training_scenarios import factory_scenario
            robots, tasks, context = factory_scenario(109302)
            visible = [task for task in tasks
                       if task.arrival_time <= context.current_time+1e-9]
            for name, path in artifact_by_scheduler.items():
                scheduler = create_scheduler(
                    name, path, allow_safe_fallback=False)
                result = scheduler.assign(visible, robots, context)
                self.assertTrue(result.is_feasible, name)


if __name__ == "__main__":
    unittest.main()
