import json
import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_DIR = ROOT / "controllers" / "factory_supervisor"
sys.path.insert(0, str(SUPERVISOR_DIR))
sys.path.insert(0, str(ROOT / "scripts"))

from config import RobotState
from learning_scheduler import (
    FEATURE_VERSION,
    GRAPH_CONTEXT_FEATURE_NAMES,
    GRAPH_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    GraphEdgeImitationModel,
    GraphImitationScheduler,
    LearnedCostHungarianScheduler,
    RidgeCostModel,
    attach_learning_trace,
    completion_samples,
    graph_edge_feature_tensor,
    pair_feature_tensor,
    pair_feature_vector,
    solve_hungarian,
)
from metrics_collector import MetricsCollector
from schedulers import (Assignment, CostMatrix, HungarianScheduler,
                        ModelValidationError, SchedulingContext,
                        create_scheduler)
from task_generator import TaskGenerator, TransportTask
from run_ml_scheduler_workflow import (filter_documents_by_runtime,
                                       run_workflow, split_real_samples)
from evaluate_scheduler_results import (evaluate_candidate, evaluate_results,
                                        load_runs)


def make_task(task_id=1, pickup=(1.0, 0.0), delivery=(3.0, 0.0),
              priority=1):
    return TransportTask(
        task_id, "S1", "WS1", pickup, delivery, 0.0,
        priority=priority)


class TaskGenerationContractTests(unittest.TestCase):
    @staticmethod
    def task_stream(seed, count=250):
        generator = TaskGenerator(
            mean_interval=8.0, seed=seed,
            initial_task_immediately=True)
        stream = []
        while len(stream) < count:
            task = generator.update(generator.next_arrival_time)
            if task is not None:
                stream.append((
                    task.task_id, task.pickup_location,
                    task.delivery_location, task.arrival_time,
                    task.priority))
        return stream

    def test_same_seed_produces_identical_task_ids_and_attributes(self):
        self.assertEqual(self.task_stream(7201), self.task_stream(7201))

    def test_priority_roll_boundaries_are_5_20_75_percent(self):
        cases = ((0.0, 3), (0.049999, 3), (0.05, 2),
                 (0.249999, 2), (0.25, 1), (0.999999, 1))
        for priority_roll, expected in cases:
            with self.subTest(priority_roll=priority_roll):
                generator = TaskGenerator(
                    mean_interval=8.0, seed=1,
                    initial_task_immediately=True)
                generator._generate_task_pair = lambda: ("S1", "WS1")
                generator.rng.random = lambda: priority_roll
                generator._schedule_next_arrival = lambda now: setattr(
                    generator, "next_arrival_time", now + 1.0)
                self.assertEqual(expected, generator.update(0.0).priority)

    def test_priority_distribution_matches_configured_probabilities(self):
        stream = self.task_stream(18_731, count=50_000)
        counts = {priority: 0 for priority in (1, 2, 3)}
        for task in stream:
            counts[task[-1]] += 1
        self.assertAlmostEqual(0.05, counts[3] / len(stream), delta=0.004)
        self.assertAlmostEqual(0.20, counts[2] / len(stream), delta=0.006)
        self.assertAlmostEqual(0.75, counts[1] / len(stream), delta=0.007)

    def test_result_filename_contains_seed_for_parallel_run_isolation(self):
        metrics = MetricsCollector(
            "C", "GraphImitation", 8, seed=7201,
            runtime_mode="webots")
        self.assertIn("_seed7201_", Path(metrics.output_path).name)


def make_robots():
    return {
        1: {"position": (0.0, 0.0), "state": RobotState.IDLE,
            "battery": 80.0, "current_task": None,
            "tasks_completed": 2, "total_distance": 9.0},
        2: {"position": (4.0, 0.0), "state": RobotState.IDLE,
            "battery": 60.0, "current_task": None,
            "tasks_completed": 4, "total_distance": 15.0},
    }


class PairFeatureTests(unittest.TestCase):
    def test_pair_features_are_finite_deterministic_and_versioned(self):
        robots = make_robots()
        task = make_task(priority=3)
        context = SchedulingContext(current_time=7.0)
        first = pair_feature_vector(1, task, robots, [task], context)
        second = pair_feature_vector(1, task, robots, [task], context)
        np.testing.assert_array_equal(first, second)
        self.assertEqual((len(PAIR_FEATURE_NAMES),), first.shape)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertEqual(3.0, first[PAIR_FEATURE_NAMES.index("task_priority")])

    def test_unreachable_pair_is_rejected(self):
        robots = make_robots()
        task = make_task()
        context = SchedulingContext(
            path_cost_provider=lambda _robot, _task: float("inf"))
        with self.assertRaises(ValueError):
            pair_feature_vector(1, task, robots, [task], context)

    def test_graph_context_is_permutation_equivariant(self):
        robots = make_robots()
        tasks = (make_task(1), make_task(2, (2.0, 1.0), (4.0, 1.0)))
        values = np.asarray([[1.0, 4.0], [3.0, 2.0]])
        mask = np.ones_like(values, dtype=bool)
        matrix = CostMatrix((1, 2), tasks, values, mask)
        context = SchedulingContext()
        pair = pair_feature_tensor(matrix, robots, list(tasks), context)
        graph = graph_edge_feature_tensor(matrix, pair)

        permuted = CostMatrix(
            (2, 1), (tasks[1], tasks[0]),
            values[[1, 0]][:, [1, 0]], mask[[1, 0]][:, [1, 0]])
        perm_pair = pair_feature_tensor(
            permuted, robots, list(tasks), context)
        perm_graph = graph_edge_feature_tensor(permuted, perm_pair)
        np.testing.assert_allclose(graph, perm_graph[[1, 0]][:, [1, 0]])

    def test_committed_trace_flows_into_completion_metrics(self):
        robots = make_robots()
        task = make_task(priority=2)
        assignment = Assignment(1, task, estimated_cost=3.5)
        context = SchedulingContext(current_time=5.0)
        task.status = "assigned"
        task.assigned_robot = 1
        task.assignment_time = 5.0
        self.assertTrue(attach_learning_trace(
            task, assignment, robots, [task], context, "Hungarian"))
        pending_index = PAIR_FEATURE_NAMES.index("pending_task_count")
        self.assertEqual(1.0, task.learning_trace["features"][pending_index])
        task.completion_time = 14.0
        metrics = MetricsCollector("test", "Hungarian", 2)
        metrics.record_task_completion(task, 1, 14.0)
        record = metrics.task_completions[0]
        self.assertEqual(2, record["priority"])
        self.assertEqual(9.0, record["execution_time"])
        self.assertEqual(FEATURE_VERSION,
                         record["learning_trace"]["feature_version"])

    def test_failed_trace_capture_is_nonfatal_and_clears_trace(self):
        robots = make_robots()
        task = make_task()
        task.learning_trace = {"stale": True}
        context = SchedulingContext(
            path_cost_provider=lambda _robot, _task: float("inf"))
        self.assertFalse(attach_learning_trace(
            task, Assignment(1, task), robots, [task], context, "test"))
        self.assertIsNone(task.learning_trace)


class ModelContractTests(unittest.TestCase):
    def model_path(self, name):
        path = ROOT / "tests" / f"_{os.getpid()}_{name}.npz"
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_ridge_round_trip_and_invalid_version(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(40, len(PAIR_FEATURE_NAMES)))
        targets = np.maximum(0.0, 5.0 + 2.0 * features[:, 0])
        model = RidgeCostModel.fit(features, targets, l2=0.01)
        path = self.model_path("cost")
        model.save(path)
        loaded = RidgeCostModel.load(path)
        np.testing.assert_allclose(
            model.predict(features), loaded.predict(features))
        self.assertEqual("execution_time", loaded.target_name)

        bad_path = self.model_path("bad")
        metadata = {
            "model_version": "wrong",
            "feature_version": FEATURE_VERSION,
            "feature_names": list(PAIR_FEATURE_NAMES),
        }
        np.savez_compressed(
            bad_path, metadata=np.asarray(json.dumps(metadata)),
            mean=model.mean, scale=model.scale,
            coefficients=model.coefficients,
            intercept=np.asarray(model.intercept))
        with self.assertRaises(ModelValidationError):
            RidgeCostModel.load(bad_path)

    def test_missing_model_and_nonfinite_training_data_are_rejected(self):
        missing = ROOT / "tests" / "_definitely_missing_model.npz"
        with self.assertRaises(ModelValidationError):
            RidgeCostModel.load(missing)
        features = np.zeros((2, len(PAIR_FEATURE_NAMES)))
        features[0, 0] = np.nan
        with self.assertRaises(ValueError):
            RidgeCostModel.fit(features, [1.0, 2.0])

    def test_graph_model_learns_separable_edges_and_round_trips(self):
        rng = np.random.default_rng(11)
        features = rng.normal(size=(200, len(GRAPH_FEATURE_NAMES)))
        labels = (features[:, 0] - features[:, 1] > 0).astype(float)
        model = GraphEdgeImitationModel.fit(
            features, labels, epochs=800, learning_rate=0.08)
        probabilities = model.predict_proba(features)
        accuracy = np.mean((probabilities >= 0.5) == labels)
        self.assertGreater(accuracy, 0.9)
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        path = self.model_path("graph")
        model.save(path)
        loaded = GraphEdgeImitationModel.load(path)
        np.testing.assert_allclose(
            probabilities, loaded.predict_proba(features))

    def test_hungarian_never_selects_masked_edge(self):
        values = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        feasible = np.asarray([[True, False], [False, True]])
        self.assertEqual([(0, 0), (1, 1)],
                         solve_hungarian(values, feasible))

    def test_completion_sample_filter_rejects_bad_contracts(self):
        vector = [float(index) for index in range(len(PAIR_FEATURE_NAMES))]
        valid = {"task_completions": [{
            "execution_time": 12.0,
            "learning_trace": {
                "feature_version": FEATURE_VERSION,
                "feature_names": list(PAIR_FEATURE_NAMES),
                "features": vector,
            },
        }]}
        invalid = {"task_completions": [{
            "execution_time": 2.0,
            "learning_trace": {"feature_version": "old", "features": []},
        }]}
        features, targets = completion_samples([valid, invalid])
        self.assertEqual((1, len(PAIR_FEATURE_NAMES)), features.shape)
        np.testing.assert_array_equal(targets, [12.0])


class LearnedSchedulerIntegrationTests(unittest.TestCase):
    def model_path(self, name):
        path = ROOT / "tests" / f"_{os.getpid()}_{name}.npz"
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def scheduling_case(self):
        robots = make_robots()
        robots[2]["position"] = (10.0, 0.0)
        tasks = [
            make_task(1, pickup=(1.0, 0.0), delivery=(2.0, 0.0)),
            make_task(2, pickup=(9.0, 0.0), delivery=(8.0, 0.0)),
        ]
        return robots, tasks, SchedulingContext()

    def test_learned_cost_scheduler_returns_unique_feasible_matching(self):
        coefficients = np.zeros(len(PAIR_FEATURE_NAMES))
        coefficients[PAIR_FEATURE_NAMES.index(
            "euclidean_empty_distance")] = 1.0
        model = RidgeCostModel(
            np.zeros(len(PAIR_FEATURE_NAMES)),
            np.ones(len(PAIR_FEATURE_NAMES)), coefficients, 0.0, 20)
        path = self.model_path("scheduler_cost")
        model.save(path)
        scheduler = LearnedCostHungarianScheduler(path)
        robots, tasks, context = self.scheduling_case()
        result = scheduler.assign(tasks, robots, context)
        self.assertTrue(result.is_feasible)
        self.assertEqual({(1, 1), (2, 2)}, {
            (item.robot_id, item.task.task_id)
            for item in result.assignments})

    def test_graph_scheduler_returns_unique_feasible_matching(self):
        weights = np.zeros(len(GRAPH_FEATURE_NAMES))
        weights[len(PAIR_FEATURE_NAMES) +
                GRAPH_CONTEXT_FEATURE_NAMES.index(
                    "cost_minus_row_min")] = -8.0
        model = GraphEdgeImitationModel(
            np.zeros(len(GRAPH_FEATURE_NAMES)),
            np.ones(len(GRAPH_FEATURE_NAMES)), weights, 3.0, 20)
        path = self.model_path("scheduler_graph")
        model.save(path)
        scheduler = GraphImitationScheduler(path)
        robots, tasks, context = self.scheduling_case()
        result = scheduler.assign(tasks, robots, context)
        self.assertTrue(result.is_feasible)
        self.assertEqual(2, len(result.assignments))
        self.assertEqual(2, len({a.robot_id for a in result.assignments}))
        self.assertEqual(2, len({a.task.task_id for a in result.assignments}))

    def test_proxy_cost_model_does_not_double_apply_urgency(self):
        coefficients = np.zeros(len(PAIR_FEATURE_NAMES))
        coefficients[PAIR_FEATURE_NAMES.index("baseline_pair_cost")] = 1.0
        model = RidgeCostModel(
            np.zeros(len(PAIR_FEATURE_NAMES)),
            np.ones(len(PAIR_FEATURE_NAMES)), coefficients, 0.0, 20,
            target_name="baseline_pair_cost")
        path = self.model_path("scheduler_proxy_cost")
        model.save(path)
        scheduler = LearnedCostHungarianScheduler(path)
        robots = {1: make_robots()[1]}
        task = make_task(priority=3)
        result = scheduler.assign([task], robots, SchedulingContext())
        self.assertTrue(result.is_feasible)
        # Distance is 3 and baseline urgency is 2, so the checkpoint target
        # is 1. It must not be reduced by the same urgency a second time.
        self.assertAlmostEqual(1.0, result.assignments[0].estimated_cost)
        self.assertEqual("baseline_pair_cost",
                         result.diagnostics["target_name"])

    def test_factory_uses_explicit_safe_fallback_for_missing_models(self):
        scheduler = create_scheduler("LearnedHungarian", model_path=None)
        self.assertIsInstance(scheduler, HungarianScheduler)
        self.assertEqual("LearnedHungarian_FALLBACK_HUNGARIAN",
                         scheduler.name)
        with self.assertRaises(ModelValidationError):
            create_scheduler(
                "GraphImitation", model_path=None,
                allow_safe_fallback=False)


class WorkflowTests(unittest.TestCase):
    def workflow_directory(self):
        path = ROOT / "results" / f"_ml_workflow_test_{os.getpid()}"

        def cleanup():
            for directory_name in ("candidates", "promoted"):
                directory = path / directory_name
                if directory.is_dir():
                    for file_path in directory.iterdir():
                        if file_path.is_file():
                            file_path.unlink()
                    directory.rmdir()
            report = path / "workflow_report.json"
            report.unlink(missing_ok=True)
            if path.is_dir():
                path.rmdir()

        self.addCleanup(cleanup)
        return path

    def test_real_results_are_split_by_seed_group(self):
        vector = [float(index) for index in range(len(PAIR_FEATURE_NAMES))]

        def document(seed):
            return {
                "experiment_info": {"seed": seed},
                "task_completions": [{
                    "execution_time": 4.0,
                    "learning_trace": {
                        "feature_version": FEATURE_VERSION,
                        "feature_names": list(PAIR_FEATURE_NAMES),
                        "features": vector,
                    },
                }],
            }

        documents = []
        for seed in range(5):
            documents.append((Path(f"seed-{seed}-a.json"), document(seed)))
            documents.append((Path(f"seed-{seed}-b.json"), document(seed)))
        _, _, _, _, train_files, validation_files = split_real_samples(
            documents)
        train_seeds = {name.split("-")[1] for name in train_files}
        validation_seeds = {name.split("-")[1]
                            for name in validation_files}
        self.assertFalse(train_seeds & validation_seeds)

    def test_workflow_smoke_promotes_loadable_models(self):
        output = self.workflow_directory()
        report = run_workflow(
            output_dir=output, train_snapshots=4,
            validation_snapshots=2, seed=12_000,
            max_bootstrap_mae=10.0, min_graph_overlap=0.0,
            max_graph_objective_gap=100.0,
            allow_proxy_bootstrap=True)
        self.assertTrue(report["promoted"])
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual("factory_astar_proxy",
                         report["data"]["ridge_source"])
        RidgeCostModel.load(output / "promoted" / "learned_cost.npz")
        GraphEdgeImitationModel.load(
            output / "promoted" / "graph_imitation.npz")
        saved = json.loads((output / "workflow_report.json").read_text(
            encoding="utf-8"))
        self.assertEqual("promoted", saved["status"])

    def test_production_training_filters_out_standalone_results(self):
        documents = [
            (Path("webots.json"), {
                "experiment_info": {"runtime_mode": "webots"}}),
            (Path("standalone.json"), {
                "experiment_info": {"runtime_mode": "standalone"}}),
            (Path("legacy.json"), {"experiment_info": {}}),
        ]
        selected = filter_documents_by_runtime(documents, "webots")
        self.assertEqual([Path("webots.json")],
                         [path for path, _ in selected])


class OnlineEvaluationTests(unittest.TestCase):
    @staticmethod
    def run_metrics(*, throughput=5.0, completion=70.0, p95=100.0,
                    waiting=10.0, safety=2, distance=2, latency=1.0):
        return {
            "throughput": throughput,
            "completion_mean": completion,
            "completion_p95": p95,
            "waiting_mean": waiting,
            "high_priority_wait_mean": None,
            "high_priority_completed": 0,
            "safety_events": safety,
            "distance_violations": distance,
            "deadlocks": 0,
            "invalid_outputs": 0,
            "fallback_commits": 0,
            "latency_p95_ms": latency,
        }

    def paired_runs(self, candidate_metrics):
        runs = {}
        for seed in range(5):
            runs[("Hungarian", seed)] = {
                "metrics": self.run_metrics(), "path": "", "scenario": "C"}
            runs[("Candidate", seed)] = {
                "metrics": dict(candidate_metrics), "path": "",
                "scenario": "C"}
        return runs

    def test_result_loader_excludes_non_webots_runtime(self):
        def document(seed, runtime_mode):
            return {
                "experiment_info": {
                    "scheduler": "Hungarian", "seed": seed,
                    "scenario": "C", "runtime_mode": runtime_mode},
                "summary_metrics": {
                    "throughput_per_minute": 1.0,
                    "avg_task_completion_time": 1.0,
                    "avg_waiting_time": 1.0},
                "task_completions": [],
            }

        documents = {
            "webots.json": document(1, "webots"),
            "standalone.json": document(2, "standalone"),
        }
        with patch("evaluate_scheduler_results.glob.glob",
                   return_value=list(documents)), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(
                    Path, "read_text", autospec=True,
                    side_effect=lambda path, **_: json.dumps(
                        documents[str(path)])):
            runs = load_runs(
                "unused", scenario="C", required_runtime_mode="webots")
        self.assertEqual([("Hungarian", 1)], list(runs))

    def test_paired_evaluation_recognizes_real_improvement(self):
        candidate = self.run_metrics(
            throughput=5.2, completion=65.0, p95=94.0, waiting=8.0)
        result = evaluate_candidate(
            self.paired_runs(candidate), "Hungarian", "Candidate")
        self.assertEqual("outperforms_baseline", result["status"])
        self.assertTrue(all(result["gates"].values()))

    def test_safety_regression_rejects_candidate(self):
        candidate = self.run_metrics(safety=3, distance=3)
        result = evaluate_candidate(
            self.paired_runs(candidate), "Hungarian", "Candidate")
        self.assertEqual("rejected", result["status"])
        self.assertFalse(result["gates"]["safety_events_noninferior"])

    def test_latency_above_absolute_budget_passes_when_close_to_baseline(self):
        runs = self.paired_runs(self.run_metrics(latency=58.0))
        for scheduler, seed in list(runs):
            if scheduler == "Hungarian":
                runs[(scheduler, seed)]["metrics"]["latency_p95_ms"] = 50.0
        result = evaluate_candidate(
            runs, "Hungarian", "Candidate",
            max_latency_p95_ms=20.0, latency_relative_tolerance=0.20)
        self.assertTrue(result["gates"]["latency_within_budget"])
        self.assertEqual(
            60.0, result["aggregate_safety"]["allowed_max_latency_p95_ms"])

    def test_latency_regression_relative_to_baseline_is_rejected(self):
        runs = self.paired_runs(self.run_metrics(latency=61.0))
        for scheduler, seed in list(runs):
            if scheduler == "Hungarian":
                runs[(scheduler, seed)]["metrics"]["latency_p95_ms"] = 50.0
        result = evaluate_candidate(
            runs, "Hungarian", "Candidate",
            max_latency_p95_ms=20.0, latency_relative_tolerance=0.20)
        self.assertEqual("rejected", result["status"])
        self.assertFalse(result["gates"]["latency_within_budget"])

    def test_negative_latency_tolerance_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "thresholds"):
            evaluate_results(
                "unused", "Hungarian", ["Candidate"],
                latency_relative_tolerance=-0.01)

    def test_no_paired_results_is_rejected_and_strict_json_safe(self):
        result = evaluate_candidate({}, "Hungarian", "Candidate")
        self.assertEqual("rejected", result["status"])
        self.assertFalse(result["gates"]["complete_seed_coverage"])
        self.assertFalse(result["gates"]["latency_within_budget"])
        self.assertIsNone(
            result["aggregate_safety"]["candidate_max_latency_p95_ms"])
        json.dumps(result, allow_nan=False)

    def test_missing_seed_and_zero_baseline_are_rejected_cleanly(self):
        runs = self.paired_runs(self.run_metrics())
        del runs[("Candidate", 4)]
        for scheduler, seed in list(runs):
            runs[(scheduler, seed)]["metrics"]["waiting_mean"] = 0.0
        runs[("Candidate", 0)]["metrics"]["waiting_mean"] = 1.0
        result = evaluate_candidate(
            runs, "Hungarian", "Candidate", min_paired_seeds=4)
        self.assertEqual("rejected", result["status"])
        self.assertFalse(result["gates"]["complete_seed_coverage"])
        json.dumps(result, allow_nan=False)

    def test_deployment_decision_retains_baseline_when_candidate_fails(self):
        runs = self.paired_runs(self.run_metrics(safety=3, distance=3))
        with patch("evaluate_scheduler_results.load_runs", return_value=runs):
            report = evaluate_results(
                "unused", "Hungarian", ["Candidate"], scenario="C")
        decision = report["deployment_decision"]
        self.assertEqual("baseline_retained", decision["status"])
        self.assertEqual("Hungarian", decision["active_scheduler"])


if __name__ == "__main__":
    unittest.main()
