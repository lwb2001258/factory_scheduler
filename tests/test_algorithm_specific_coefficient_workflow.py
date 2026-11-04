"""Contracts for the per-algorithm coefficient workflow."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/factory_supervisor"))
sys.path.insert(0, str(ROOT / "scripts"))

from expert_coefficients import bundle_sha256, canonical_bundle, search_fields
import run_algorithm_specific_coefficient_workflow as workflow


class AlgorithmSpecificWorkflowTests(unittest.TestCase):
    def _root_with_baseline(self, directory):
        root = Path(directory)
        target = root / "baseline/shared_selected_bundle.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        source = (ROOT / "results/expert_coefficients_v1_20261030/coefficients/"
                  "selected_bundle.json")
        target.write_bytes(source.read_bytes())
        return root

    def test_policy_matrix_and_seed_partitions_are_complete_and_disjoint(self):
        self.assertEqual(6, len(workflow.POLICY_IDS))
        self.assertEqual(6, len(set(workflow.POLICY_IDS)))
        for policy_id in workflow.POLICY_IDS:
            algorithm, objective = workflow._policy_parts(policy_id)
            self.assertIn(algorithm, workflow.ALGORITHMS)
            self.assertIn(objective, workflow.OBJECTIVES)
        partitions = (
            workflow.INNER_TRAINING_SEEDS,
            workflow.OUTER_VALIDATION_SEEDS,
            workflow.TRAINING_SEEDS,
            workflow.FINE_TUNE_SEEDS,
            workflow.EVALUATION_SEEDS,
        )
        for index, left in enumerate(partitions):
            for right in partitions[index + 1:]:
                self.assertFalse(set(left) & set(right))

    def test_zero_vector_reproduces_warm_start_row_per_policy(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = self._root_with_baseline(directory)
            warm = workflow.warm_start_bundle(root)
            hashes = {}
            for policy_id in workflow.POLICY_IDS:
                _algorithm, objective = workflow._policy_parts(policy_id)
                vector = [0.0] * len(search_fields(objective))
                bundle = workflow.coefficient_bundle_for(
                    root, policy_id, vector)
                self.assertEqual(
                    warm["objectives"][objective],
                    bundle["objectives"][objective])
                self.assertFalse(bundle["optimization"]["training_enabled"])
                hashes[policy_id] = bundle_sha256(bundle)
            # Count policies may start numerically equal, but identities remain
            # policy-indexed by selected_bundles rather than a shared key.
            self.assertEqual(set(workflow.POLICY_IDS), set(hashes))

    def test_search_bounds_are_exactly_quarter_to_four_times_warm_start(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = self._root_with_baseline(directory)
            base = workflow.warm_start_bundle(root)
            for multiplier, vector_value in (
                    (0.25, math.log(0.25)), (4.0, math.log(4.0))):
                bundle = workflow.coefficient_bundle_for(
                    root, "GraphImitation_utility_v2",
                    [vector_value] * len(search_fields("utility_v2")))
                for field in search_fields("utility_v2"):
                    self.assertAlmostEqual(
                        multiplier * base["objectives"]["utility_v2"][field],
                        bundle["objectives"]["utility_v2"][field])

    def test_selected_bundle_collection_rejects_policy_or_hash_mismatch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = self._root_with_baseline(directory)
            base = workflow.warm_start_bundle(root)
            policies = {
                policy_id: {
                    "bundle": base,
                    "bundle_sha256": bundle_sha256(base),
                } for policy_id in workflow.POLICY_IDS}
            document = {
                "status": "complete",
                "policies": policies,
                "selection_sha256": workflow.digest({
                    policy_id: bundle_sha256(base)
                    for policy_id in sorted(policies)}),
            }
            path = root / "coefficients/selected_bundles.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = workflow.selected_bundles(root)
            self.assertEqual(set(workflow.POLICY_IDS), set(loaded))
            document["policies"][workflow.POLICY_IDS[0]][
                "bundle_sha256"] = "0" * 64
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                workflow.selected_bundles(root)

    def test_paired_comparison_uses_main_metric_and_six_seed_clusters(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = Path(directory)
            baseline_episodes = []
            current_episodes = []
            baseline_aggregates = []
            current_aggregates = []
            for policy_id in workflow.POLICY_IDS:
                algorithm, objective = workflow._policy_parts(policy_id)
                baseline_aggregates.append({
                    "policy_id": policy_id,
                    "scheduler": algorithm,
                    "objective": objective,
                    "mean_completed_tasks": 10.0,
                    "mean_utility_score_deadline_v2": 20.0,
                })
                current_aggregates.append({
                    "policy_id": policy_id,
                    "scheduler": algorithm,
                    "objective": objective,
                    "mean_completed_tasks": 11.0,
                    "mean_utility_score_deadline_v2": 21.0,
                })
                for scenario in workflow.SCENARIOS:
                    for seed in workflow.EVALUATION_SEEDS:
                        common = {
                            "policy_id": policy_id,
                            "scheduler": algorithm,
                            "objective": objective,
                            "scenario": scenario,
                            "seed": seed,
                            "manifest_sha256": f"{seed:064x}",
                        }
                        baseline_episodes.append({
                            **common, "completed_tasks": 10,
                            "utility_score_deadline_v2": 20.0})
                        current_episodes.append({
                            **common, "completed_tasks": 11,
                            "utility_score_deadline_v2": 21.0})
            baseline = {
                "status": "complete",
                "seeds": list(workflow.EVALUATION_SEEDS),
                "aggregates": baseline_aggregates,
                "episodes": baseline_episodes,
            }
            current = {
                "status": "complete",
                "seeds": list(workflow.EVALUATION_SEEDS),
                "aggregates": current_aggregates,
                "episodes": current_episodes,
            }
            baseline_path = root / "baseline/shared_evaluation_summary.json"
            baseline_path.parent.mkdir(parents=True)
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            summary_path = root / "evaluation_summary.json"
            summary_path.write_text(json.dumps(current), encoding="utf-8")
            comparison = workflow.compare_to_shared_baseline(root, current)
            self.assertEqual(6, len(comparison["comparisons"]))
            for row in comparison["comparisons"]:
                self.assertAlmostEqual(1.0, row["delta"])
                self.assertEqual([18, 0, 0], row["wins_ties_losses"])
                self.assertEqual(6, len(row["seed_cluster_delta_means"]))
                self.assertEqual([1.0, 1.0], row["delta_95_percent_interval"])

    def test_physical_handoff_has_no_command_boundary(self):
        source = (ROOT / "scripts/run_algorithm_specific_coefficient_workflow.py"
                  ).read_text(encoding="utf-8")
        start = source.index("def physical_then_evaluate")
        end = source.index("\ndef run_all", start)
        body = source[start:end]
        self.assertLess(body.index("fine_tune_all"), body.index("evaluate_all"))
        self.assertIn("review_runtime_step(root, 7)", body)
        self.assertIn("review_runtime_step(root, 8)", body)


if __name__ == "__main__":
    unittest.main()
