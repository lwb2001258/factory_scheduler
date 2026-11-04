"""Audit gates for algorithm-specific expert-coefficient optimization."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results/algorithm_specific_coefficients_v1_20261101"
SHARED_BASELINE = (
    ROOT / "results/expert_coefficients_v1_20261030/coefficients/selected_bundle.json")
SHARED_EVALUATION = (
    ROOT / "results/expert_coefficients_v1_20261030/evaluation_summary.json")
VERSION = "algorithm-specific-coefficient-workflow-v1"
STEPS = (
    "two_round_design_review",
    "bounded_warm_start_coefficient_core",
    "resumable_algorithm_specific_workflow",
    "two_round_workflow_review_and_validation",
    "algorithm_specific_cma_es_search",
    "final_student_standalone_training",
    "webots_fine_tuning",
    "six_seed_evaluation_and_comparison",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def confined(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"output escapes isolated workflow: {path}")
    return path


def _replace(source, destination):
    for attempt in range(10):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.5))


def write(root, path, value):
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
    _replace(temporary, path)


def validate_root(root):
    root = Path(root).resolve()
    if root.parent != (ROOT / "results").resolve():
        raise ValueError("workflow must be a direct child of results")
    document = read(root / "workflow.json")
    if (not isinstance(document, dict) or
            document.get("version") != VERSION or
            document.get("run_root") != str(root)):
        raise ValueError("initialize the algorithm-specific workflow first")
    return root


def _copy_frozen(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"frozen copy differs: {source}")


def initialize(root=DEFAULT_RUN):
    root = Path(root).resolve()
    if root.parent != (ROOT / "results").resolve():
        raise ValueError("new workflow must be a direct child of results")
    if root.exists():
        return read(validate_root(root) / "workflow.json")
    for required in (
            SHARED_BASELINE, SHARED_EVALUATION,
            ROOT / "docs/algorithm_specific_expert_coefficients_design_review.md",
            ROOT / "docs/algorithm_specific_coefficient_workflow_review.md"):
        if not required.is_file():
            raise FileNotFoundError(required)
    root.mkdir(parents=True, exist_ok=False)
    source_files = sorted({
        path for directory in (ROOT / "controllers", ROOT / "scripts")
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts})
    source_hashes = {}
    for source in source_files:
        relative = source.relative_to(ROOT).as_posix()
        destination = confined(root, root / "baseline/source" / relative)
        _copy_frozen(source, destination)
        source_hashes[relative] = sha256(source)
    frozen_shared = root / "baseline/shared_selected_bundle.json"
    frozen_evaluation = root / "baseline/shared_evaluation_summary.json"
    _copy_frozen(SHARED_BASELINE, frozen_shared)
    _copy_frozen(SHARED_EVALUATION, frozen_evaluation)
    workflow = {
        "version": VERSION,
        "created_at": now(),
        "run_root": str(root),
        "algorithms": [
            "GraphImitation", "LearnedHungarian", "TrafficHungarian"],
        "objectives": ["count", "utility_v2"],
        "policy_ids": [
            f"{algorithm}_{objective}"
            for algorithm in (
                "GraphImitation", "LearnedHungarian", "TrafficHungarian")
            for objective in ("count", "utility_v2")],
        "scenarios": ["A", "B", "C"],
        "duration_seconds": 1800,
        "coefficient_inner_training_seeds": [11000, 11001, 11002],
        "coefficient_outer_validation_seeds": [11003, 11004],
        "student_training_seeds": list(range(21000, 21020)),
        "webots_fine_tune_seeds": list(range(41000, 41010)),
        "evaluation_seeds": list(range(61000, 61006)),
        "seed_partitions_disjoint": True,
        "coefficient_search": {
            "optimizer": "separable-cma-es-v1",
            "warm_start": str(frozen_shared),
            "warm_start_sha256": sha256(frozen_shared),
            "generations": 2,
            "population": {"count": 6, "utility_v2": 7},
            "multiplier_bounds": [0.25, 4.0],
            "fitness": "seed mean - 0.25 * seed pstdev - utility throughput shortfall",
            "utility_throughput_tolerance_tasks": 2.0,
            "deployed_policy_fitness": True,
        },
        "student_training": {
            "balanced_rounds": 20,
            "checkpoint_selection": "latest cumulative checkpoint",
            "degradation_validation": False,
            "rollback": False,
        },
        "webots": {
            "fine_tune_rounds": 10,
            "fine_tune_to_evaluation_handoff": "same uninterrupted command",
            "evaluation_learning": False,
            "evaluation_fallback": False,
        },
        "shared_baseline_evaluation": str(frozen_evaluation),
        "shared_baseline_evaluation_sha256": sha256(frozen_evaluation),
        "minimum_code_reviews_per_step": 3,
        "source_at_initialization_sha256": source_hashes,
        "steps": [
            {"step": index, "name": name, "status": "pending"}
            for index, name in enumerate(STEPS, 1)],
    }
    write(root, root / "workflow.json", workflow)
    return workflow


def _review_hashes(files):
    hashes = {}
    for value in files:
        path = ((ROOT / value).resolve()
                if not Path(value).is_absolute() else Path(value).resolve())
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"reviewed file is missing or outside repository: {value}")
        hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
    if not hashes:
        raise ValueError("a review must cover at least one file")
    return hashes


def record_review(root, step: int, *, perspective: str,
                  findings: list[str], checks: list[str], files,
                  status: str = "pass"):
    root = validate_root(root)
    if (not 1 <= step <= len(STEPS) or
            status not in {"pass", "changes_required"} or
            not perspective or not findings or not checks):
        raise ValueError("review evidence is incomplete")
    if step > 1:
        require_reviews(root, step - 1)
    hashes = _review_hashes(files)
    path = root / "audit" / f"step_{step:02d}_reviews.json"
    receipt = read(path) or {
        "step": step, "name": STEPS[step - 1], "reviews": []}
    if receipt.get("files") not in (None, hashes):
        receipt = {"step": step, "name": STEPS[step - 1], "reviews": []}
    receipt["files"] = hashes
    receipt["reviews"].append({
        "round": len(receipt["reviews"]) + 1,
        "reviewed_at": now(),
        "reviewer": "Codex sequential code/artifact review",
        "perspective": perspective,
        "status": status,
        "findings": findings,
        "checks": checks,
        "files": hashes,
    })
    write(root, path, receipt)
    return receipt


def require_reviews(root, through_step: int):
    root = validate_root(root)
    for step in range(1, through_step + 1):
        receipt = read(root / "audit" / f"step_{step:02d}_reviews.json") or {}
        reviews = receipt.get("reviews", [])
        if (len(reviews) < 3 or
                any(row.get("status") != "pass" for row in reviews[-3:]) or
                len({row.get("perspective") for row in reviews[-3:]}) != 3):
            raise RuntimeError(f"step {step} requires three passing perspectives")
        files = receipt.get("files", {})
        if any(row.get("files") != files for row in reviews[-3:]):
            raise RuntimeError(f"step {step} reviews do not cover identical hashes")
        for relative, expected in files.items():
            path = (ROOT / relative).resolve()
            if (not path.is_file() or not path.is_relative_to(ROOT) or
                    sha256(path) != expected):
                raise RuntimeError(f"reviewed file changed: {relative}")


def update_step(root, step: int, status: str, **values):
    root = validate_root(root)
    if status not in {"pending", "running", "complete", "failed"}:
        raise ValueError("workflow step status is invalid")
    workflow = read(root / "workflow.json")
    row = workflow["steps"][step - 1]
    row.update(status=status, updated_at=now(), **values)
    write(root, root / "workflow.json", workflow)
    return row
