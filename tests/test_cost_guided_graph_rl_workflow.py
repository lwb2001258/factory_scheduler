import shutil
import unittest
from pathlib import Path
from unittest import mock

from controllers.factory_supervisor.expert_coefficients import bundle_sha256
from scripts import cost_guided_graph_rl_workflow_contract as contract


class CostGuidedGraphRLWorkflowContractTests(unittest.TestCase):
    def setUp(self):
        self.root = (
            contract.ROOT / "results" /
            "test_cost_guided_graph_rl_workflow_contract")
        self._safe_remove()

    def tearDown(self):
        self._safe_remove()

    def _safe_remove(self):
        resolved = self.root.resolve()
        expected_parent = (contract.ROOT / "results").resolve()
        self.assertEqual(resolved.parent, expected_parent)
        self.assertTrue(resolved.name.startswith("test_cost_guided_graph_rl_"))
        if resolved.exists():
            shutil.rmtree(resolved)

    def test_seed_partitions_are_disjoint_and_final_batch_has_six_seeds(self):
        partitions = list(contract.SEED_PARTITIONS.values())
        self.assertEqual(len(contract.EVALUATION_SEEDS), 6)
        for index, left in enumerate(partitions):
            self.assertEqual(len(left), len(set(left)))
            for right in partitions[index + 1:]:
                self.assertFalse(set(left).intersection(right))

    def test_policy_mapping_is_complete_and_has_distinct_priors(self):
        self.assertEqual(
            set(contract.TEACHER_MAPPING), set(contract.POLICIES))
        for policy, mapping in contract.TEACHER_MAPPING.items():
            self.assertIn(mapping["objective"], contract.OBJECTIVES, policy)
            self.assertNotEqual(
                mapping["primary"], mapping["secondary"], policy)

    def test_initialize_freezes_canonical_bundle_hashes(self):
        workflow = contract.initialize(self.root)
        _, validated = contract.validate_root(self.root)
        self.assertEqual(workflow, validated)
        self.assertFalse(workflow["formal_reward_modified"])
        for row in workflow["teacher_bundles"].values():
            self.assertEqual(row["bundle_sha256"], bundle_sha256(row["bundle"]))
        self.assertEqual(
            set(workflow["teacher_bundles"]),
            set(contract.EXPECTED_BUNDLE_SHA256))
        self.assertEqual(
            {key: row["bundle_sha256"]
             for key, row in workflow["teacher_bundles"].items()},
            contract.EXPECTED_BUNDLE_SHA256)

    def test_three_hash_bound_reviews_are_required_and_ordered(self):
        contract.initialize(self.root)
        files = (
            contract.ROOT / "docs/cost_guided_graph_rl_workflow.md",
            Path(contract.__file__),
            Path(__file__),
        )
        with self.assertRaisesRegex(RuntimeError, "step 1 requires three"):
            contract.require_reviews(self.root, 1)
        with self.assertRaisesRegex(RuntimeError, "step 1 requires three"):
            contract.record_review(
                self.root, 2,
                perspective=contract.REVIEW_PERSPECTIVES[0], files=files,
                findings=["must not bypass step 1"], checks=["ordering"])
        for perspective in contract.REVIEW_PERSPECTIVES:
            contract.record_review(
                self.root, 1, perspective=perspective, files=files,
                findings=["reviewed"], checks=["unit-test evidence"])
        contract.require_reviews(self.root, 1)
        receipt = contract.read(contract.review_file_path(self.root, 1))
        self.assertEqual(len(receipt["reviews"]), 3)
        first = receipt["reviews"][0]
        contract.record_review(
            self.root, 1, perspective=first["perspective"], files=files,
            findings=first["findings"], checks=first["checks"])
        contract.require_reviews(self.root, 1)

    def test_review_becomes_invalid_when_a_reviewed_file_changes(self):
        contract.initialize(self.root)
        sentinel = self.root / "audit/review_sentinel.json"
        contract.write(self.root, sentinel, {"revision": 1})
        for perspective in contract.REVIEW_PERSPECTIVES:
            contract.record_review(
                self.root, 1, perspective=perspective, files=[sentinel],
                findings=["reviewed sentinel"], checks=["hash binding"])
        contract.require_reviews(self.root, 1)
        contract.write(self.root, sentinel, {"revision": 2})
        with self.assertRaisesRegex(RuntimeError, "reviewed file changed"):
            contract.require_reviews(self.root, 1)

    def test_workflow_identity_rejects_relaxed_reward_or_seed_boundary(self):
        contract.initialize(self.root)
        path = self.root / "workflow.json"
        workflow = contract.read(path)
        workflow["formal_reward_modified"] = True
        contract.write(self.root, path, workflow)
        with self.assertRaisesRegex(ValueError, "workflow identity differs"):
            contract.validate_root(self.root)

    def test_failed_initialization_does_not_publish_partial_run_root(self):
        prefix = f".{self.root.name}.init."
        before = set(self.root.parent.glob(prefix + "*"))
        with mock.patch.object(
                contract.shutil, "copyfile",
                side_effect=OSError("injected copy failure")):
            with self.assertRaisesRegex(OSError, "injected copy failure"):
                contract.initialize(self.root)
        self.assertFalse(self.root.exists())
        self.assertEqual(set(self.root.parent.glob(prefix + "*")), before)


if __name__ == "__main__":
    unittest.main()
