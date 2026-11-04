"""Audit and identity tests for the expert-coefficient workflow."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "controllers/factory_supervisor"))

import expert_coefficient_workflow_contract as audit
import run_expert_coefficient_workflow as workflow
import run_strict_1800_workflow as strict
from expert_coefficients import default_bundle, with_optimized_rows


class WorkflowAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        # TemporaryDirectory creates the directory, whereas initialize owns
        # creation. Remove the empty leaf without touching any broad path.
        self.root.rmdir()
        audit.initialize(self.root)
        self.reviewed = ROOT / "tests/test_expert_coefficient_workflow.py"

    def test_three_distinct_perspectives_and_exact_hash_are_required(self):
        for perspective in ("correctness", "safety", "compatibility"):
            audit.record_review(
                self.root, 1, perspective=perspective,
                findings=["checked"], checks=["unit"],
                files=[self.reviewed])
        audit.require_reviews(self.root, 1)
        receipt = audit.read(self.root / "audit/step_01_reviews.json")
        receipt["reviews"][2]["perspective"] = "safety"
        audit.write(self.root, self.root / "audit/step_01_reviews.json", receipt)
        with self.assertRaises(RuntimeError):
            audit.require_reviews(self.root, 1)

    def test_seed_partitions_are_disjoint_and_frozen(self):
        groups = [set(values[1]) for values in workflow.SEED_PARTITIONS.values()]
        for index, left in enumerate(groups):
            for right in groups[index + 1:]:
                self.assertFalse(left & right)
        document = audit.read(self.root / "workflow.json")
        self.assertEqual(list(workflow.EVALUATION_SEEDS),
                         document["evaluation_seeds"])
        self.assertEqual("same uninterrupted command",
                         document["webots"]["fine_tune_to_evaluation_handoff"])

    def test_training_identity_changes_with_expert_coefficients(self):
        base = default_bundle()
        optimized = with_optimized_rows(
            base["objectives"], method="separable-cma-es-v1")
        spec = dict(next(item for item in strict.AI_SPECS
                         if item["policy_id"] == "GraphImitation_count"))
        spec["traffic_version"] = "route-traffic-v1"
        before = strict.training_config_sha256(spec)
        spec["expert_coefficient_bundle"] = optimized
        after = strict.training_config_sha256(spec)
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
