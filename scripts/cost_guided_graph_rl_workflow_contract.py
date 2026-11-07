"""Immutable contract and audit helpers for cost-guided graph RL."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

from controllers.factory_supervisor.expert_coefficients import (
    bundle_sha256, canonical_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results/cost_guided_graph_rl_v1_20261105"
SOURCE_SELECTION = (
    ROOT / "results/algorithm_specific_coefficients_v1_20261101/"
    "coefficients/selected_bundles.json")
SOURCE_SELECTION_SHA256 = (
    "3aa51b7e85e2c920d9d863d7c15082c076c41fe925df330e3bec2d4e5f00951d")
VERSION = "cost-guided-graph-rl-workflow-v1"
SCENARIOS = ("A", "B", "C")
OBJECTIVES = ("count", "utility_v2")
POLICIES = (
    "CostGraphPPO_count",
    "CostGraphPPO_utility_v2",
    "GraphRainbow_count",
    "GraphRainbow_utility_v2",
)
STANDALONE_SEEDS = tuple(range(72000, 72006))
WEBOTS_FINE_TUNE_SEEDS = tuple(range(73000, 73002))
EVALUATION_SEEDS = tuple(range(74000, 74006))
SEED_PARTITIONS = {
    "standalone_training": STANDALONE_SEEDS,
    "webots_fine_tune": WEBOTS_FINE_TUNE_SEEDS,
    "final_evaluation": EVALUATION_SEEDS,
}
STEPS = (
    "workflow_contract",
    "cost_prior_identity_and_math",
    "cost_prior_graph_observation",
    "cost_guided_graph_ppo_model",
    "cost_guided_graph_ppo_training",
    "variable_edge_graph_rainbow_model",
    "graph_rainbow_replay_and_training",
    "runtime_and_scheduler_integration",
    "standalone_workflow_and_training",
    "webots_fine_tune_workflow_and_run",
    "same_batch_six_seed_webots_evaluation",
    "formal_gate_report_and_reproducibility",
)
REVIEW_PERSPECTIVES = (
    "algorithm_and_numerical_semantics",
    "safety_identity_and_data_boundaries",
    "integration_regression_and_resumability",
)
TEACHER_MAPPING = {
    "CostGraphPPO_count": {
        "objective": "count",
        "primary": "GraphImitation_count",
        "secondary": "LearnedHungarian_count",
    },
    "CostGraphPPO_utility_v2": {
        "objective": "utility_v2",
        "primary": "GraphImitation_utility_v2",
        "secondary": "LearnedHungarian_utility_v2",
    },
    "GraphRainbow_count": {
        "objective": "count",
        "primary": "LearnedHungarian_count",
        "secondary": "GraphImitation_count",
    },
    "GraphRainbow_utility_v2": {
        "objective": "utility_v2",
        "primary": "LearnedHungarian_utility_v2",
        "secondary": "GraphImitation_utility_v2",
    },
}
EXPECTED_BUNDLE_SHA256 = {
    "GraphImitation_count":
        "04676a293d1710686569aac34d9acf88591db3e538a42bfb9f4d11be534f171e",
    "GraphImitation_utility_v2":
        "803c1eec98d5a0c4141922bb6955e5683f79acf00e9f4e7ab3a32f1de85e609e",
    "LearnedHungarian_count":
        "6a3c36f4c8b8cfe6ad4916ca4e4e18fcfa3f81db988268389575804633b4213d",
    "LearnedHungarian_utility_v2":
        "d1881055a0a5f26aaaf34821ca5503a0b36ccd415ce811550c93cfd5f3c54f36",
}


def _validate_static_contract() -> None:
    partitions = list(SEED_PARTITIONS.items())
    for left_index, (left_name, left_values) in enumerate(partitions):
        if not left_values or len(set(left_values)) != len(left_values):
            raise RuntimeError(f"invalid seed partition: {left_name}")
        for right_name, right_values in partitions[left_index + 1:]:
            if set(left_values).intersection(right_values):
                raise RuntimeError(
                    f"seed partitions overlap: {left_name}/{right_name}")
    if set(TEACHER_MAPPING) != set(POLICIES):
        raise RuntimeError("teacher mapping does not cover every policy")
    if len(EVALUATION_SEEDS) != 6:
        raise RuntimeError("final evaluation requires exactly six seeds")
    required_teachers = {
        mapping[key]
        for mapping in TEACHER_MAPPING.values()
        for key in ("primary", "secondary")
    }
    if required_teachers != set(EXPECTED_BUNDLE_SHA256):
        raise RuntimeError("expected coefficient hashes are incomplete")
    for policy_id, mapping in TEACHER_MAPPING.items():
        if mapping["objective"] not in OBJECTIVES:
            raise RuntimeError(f"invalid objective mapping: {policy_id}")
        if mapping["primary"] == mapping["secondary"]:
            raise RuntimeError(f"teacher priors must differ: {policy_id}")


_validate_static_contract()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def confined(root, path) -> Path:
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"workflow output escapes run root: {path}")
    return path


def _replace(source, destination) -> None:
    for attempt in range(10):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.5))


def write(root, path, value) -> None:
    path = confined(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        _replace(temporary, path)
    finally:
        if temporary.is_file():
            temporary.unlink()


def _selected_policy_bundles(selection) -> dict:
    if (not isinstance(selection, dict) or
            selection.get("status") != "complete" or
            not isinstance(selection.get("policies"), dict)):
        raise ValueError("source coefficient selection is incomplete")
    required = {
        mapping[key]
        for mapping in TEACHER_MAPPING.values()
        for key in ("primary", "secondary")
    }
    result = {}
    for policy_id in sorted(required):
        row = selection["policies"].get(policy_id)
        if (not isinstance(row, dict) or
                not isinstance(row.get("bundle"), dict) or
                not isinstance(row.get("bundle_sha256"), str)):
            raise ValueError(f"source coefficient policy is invalid: {policy_id}")
        bundle = canonical_bundle(row["bundle"])
        actual_hash = bundle_sha256(bundle)
        if row["bundle_sha256"] != actual_hash:
            raise ValueError(
                f"source coefficient hash differs: {policy_id}")
        if EXPECTED_BUNDLE_SHA256[policy_id] != actual_hash:
            raise ValueError(
                f"frozen coefficient identity differs: {policy_id}")
        result[policy_id] = {
            "bundle": bundle,
            "bundle_sha256": actual_hash,
        }
    return result


def initialize(root=DEFAULT_RUN) -> dict:
    root = Path(root).resolve()
    if root.parent != (ROOT / "results").resolve():
        raise ValueError("workflow must be a direct child of results")
    existing = read(root / "workflow.json")
    if existing is not None:
        return validate_root(root)[1]
    if root.exists():
        raise ValueError("workflow run root is partial or corrupt")
    if not SOURCE_SELECTION.is_file() or sha256(
            SOURCE_SELECTION) != SOURCE_SELECTION_SHA256:
        raise ValueError("source coefficient selection identity differs")
    selection = read(SOURCE_SELECTION)
    bundles = _selected_policy_bundles(selection)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        dir=root.parent, prefix=f".{root.name}.init."))
    try:
        staged_frozen = staging / "frozen/selected_bundles.json"
        staged_frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_SELECTION, staged_frozen)
        if sha256(staged_frozen) != SOURCE_SELECTION_SHA256:
            raise RuntimeError("frozen coefficient selection differs")
        final_frozen = root / "frozen/selected_bundles.json"
        document = {
            "version": VERSION,
            "created_at": now(),
            "run_root": str(root),
            "source_selection": str(SOURCE_SELECTION),
            "source_selection_sha256": SOURCE_SELECTION_SHA256,
            "frozen_selection": str(final_frozen),
            "frozen_selection_sha256": sha256(staged_frozen),
            "policies": list(POLICIES),
            "scenarios": list(SCENARIOS),
            "seed_partitions": {
                name: list(values) for name, values in SEED_PARTITIONS.items()},
            "common_final_seed_batch": True,
            "teacher_mapping": TEACHER_MAPPING,
            "teacher_bundles": bundles,
            "formal_reward_modified": False,
            "required_reviews_per_step": len(REVIEW_PERSPECTIVES),
            "steps": [
                {"step": index, "name": name, "status": "pending"}
                for index, name in enumerate(STEPS, 1)],
        }
        write(staging, staging / "workflow.json", document)
        staging.replace(root)
    finally:
        if staging.exists():
            resolved = staging.resolve()
            if (resolved.parent != root.parent or
                    not resolved.name.startswith(f".{root.name}.init.")):
                raise RuntimeError("unsafe workflow staging path")
            shutil.rmtree(resolved)
    return validate_root(root)[1]


def validate_root(root=DEFAULT_RUN) -> tuple[Path, dict]:
    root = Path(root).resolve()
    document = read(root / "workflow.json")
    expected_steps = [
        {"step": index, "name": name}
        for index, name in enumerate(STEPS, 1)]
    actual_steps = document.get("steps") if isinstance(document, dict) else None
    if (root.parent != (ROOT / "results").resolve() or
            not isinstance(document, dict) or
            document.get("version") != VERSION or
            document.get("run_root") != str(root) or
            document.get("source_selection_sha256") !=
            SOURCE_SELECTION_SHA256 or
            document.get("policies") != list(POLICIES) or
            document.get("scenarios") != list(SCENARIOS) or
            document.get("teacher_mapping") != TEACHER_MAPPING or
            document.get("seed_partitions") != {
                name: list(values) for name, values in SEED_PARTITIONS.items()} or
            document.get("common_final_seed_batch") is not True or
            document.get("formal_reward_modified") is not False or
            document.get("required_reviews_per_step") !=
            len(REVIEW_PERSPECTIVES) or
            not isinstance(actual_steps, list) or
            len(actual_steps) != len(expected_steps) or
            any(not isinstance(row, dict) or
                row.get("step") != expected["step"] or
                row.get("name") != expected["name"] or
                row.get("status") not in {"pending", "reviewed"}
                for row, expected in zip(actual_steps, expected_steps))):
        raise ValueError("cost-guided graph RL workflow identity differs")
    frozen_value = document.get("frozen_selection")
    if not isinstance(frozen_value, str):
        raise ValueError("frozen coefficient selection identity differs")
    frozen = Path(frozen_value).resolve()
    if (frozen != (root / "frozen/selected_bundles.json").resolve() or
            not frozen.is_file() or
            sha256(frozen) != document["frozen_selection_sha256"] or
            _selected_policy_bundles(read(frozen)) !=
            document["teacher_bundles"]):
        raise ValueError("frozen coefficient selection identity differs")
    return root, document


def review_file_path(root, step: int) -> Path:
    return Path(root) / "audit" / f"step_{step:02d}_reviews.json"


def record_review(root, step: int, *, perspective: str, files,
                  findings, checks) -> dict:
    root, workflow = validate_root(root)
    if (not 1 <= step <= len(STEPS) or
            perspective not in REVIEW_PERSPECTIVES):
        raise ValueError("review step or perspective is invalid")
    if step > 1:
        require_reviews(root, step - 1)
    paths = [Path(value).resolve() for value in files]
    if (not paths or any(
            not path.is_file() or not path.is_relative_to(ROOT)
            for path in paths)):
        raise ValueError("review files must exist inside the repository")
    hashes = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(paths)
    }
    path = review_file_path(root, step)
    receipt = read(path) or {
        "step": step, "name": STEPS[step - 1], "files": hashes,
        "reviews": []}
    if receipt.get("files") != hashes:
        receipt = {
            "step": step, "name": STEPS[step - 1], "files": hashes,
            "reviews": []}
    if not findings or not checks:
        raise ValueError("review evidence must not be empty")
    matching = [row for row in receipt["reviews"]
                if row.get("perspective") == perspective]
    if matching:
        expected = matching[0]
        if (expected.get("findings") != list(findings) or
                expected.get("checks") != list(checks) or
                expected.get("files") != hashes):
            raise ValueError("recorded review evidence differs")
    else:
        receipt["reviews"].append({
            "round": len(receipt["reviews"]) + 1,
            "reviewed_at": now(),
            "reviewer": "Codex sequential review",
            "perspective": perspective,
            "status": "pass",
            "findings": list(findings),
            "checks": list(checks),
            "files": hashes,
        })
        write(root, path, receipt)
    if len(receipt["reviews"]) == len(REVIEW_PERSPECTIVES):
        if {row["perspective"] for row in receipt["reviews"]} != set(
                REVIEW_PERSPECTIVES):
            raise RuntimeError("three review perspectives are incomplete")
        workflow["steps"][step - 1].update(
            status="reviewed", reviewed_at=now(), review_sha256=sha256(path))
        write(root, root / "workflow.json", workflow)
    return receipt


def require_reviews(root, through_step: int) -> None:
    root, workflow = validate_root(root)
    if not 1 <= through_step <= len(STEPS):
        raise ValueError("review range is invalid")
    for step in range(1, through_step + 1):
        receipt = read(review_file_path(root, step)) or {}
        reviews = receipt.get("reviews", [])
        if (len(reviews) != len(REVIEW_PERSPECTIVES) or
                {row.get("perspective") for row in reviews} !=
                set(REVIEW_PERSPECTIVES) or
                any(row.get("status") != "pass" for row in reviews) or
                workflow["steps"][step - 1].get("status") != "reviewed"):
            raise RuntimeError(f"step {step} requires three passing reviews")
        if workflow["steps"][step - 1].get("review_sha256") != sha256(
                review_file_path(root, step)):
            raise RuntimeError(f"step {step} review receipt hash differs")
        for relative, expected in receipt.get("files", {}).items():
            file_path = (ROOT / relative).resolve()
            if (not file_path.is_file() or
                    not file_path.is_relative_to(ROOT) or
                    sha256(file_path) != expected):
                raise RuntimeError(f"reviewed file changed: {relative}")
