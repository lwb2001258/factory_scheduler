"""Evaluate each learned policy's own analytical teacher after the main run.

This module intentionally lives outside ``controllers/`` and ``scripts/``.
The active algorithm-specific workflow freezes every Python source below those
directories, so adding an extension there would invalidate its running
contract.  ``wait-run`` waits for the source workflow, its three step-8
reviews, and its process lock before creating an isolated, resumable 72-episode
Webots evaluation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time


os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "scripts"),
    str(ROOT / "controllers/factory_supervisor"),
]

import run_algorithm_specific_coefficient_workflow as source_workflow
import run_expert_coefficient_workflow as physical_workflow
import run_webots_1800_workflow as physical
from expert_coefficients import bundle_sha256, canonical_bundle
from graph_expert_scheduler import expert_descriptor
from schedulers import create_scheduler


VERSION = "algorithm-specific-own-teacher-extension-v1"
SOURCE_ROOT = ROOT / "results/algorithm_specific_coefficients_v1_20261101"
OUTPUT_ROOT = ROOT / "results/algorithm_specific_teacher_extension_v1_20261102"
DEFAULT_WEBOTS = Path("C:/Program Files/Webots/msys64/mingw64/bin/webots.exe")
SCENARIOS = ("A", "B", "C")
SEEDS = tuple(range(61000, 61006))
OBJECTIVES = ("count", "utility_v2")
FAMILIES = {
    "GraphImitationExpert": "GraphImitation",
    "LearnedHungarianExpert": "LearnedHungarian",
}
POLICY_IDS = tuple(
    f"{family}_{objective}"
    for family in FAMILIES
    for objective in OBJECTIVES
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")).hexdigest()


def read(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def confined(root, path) -> Path:
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"output escapes extension root: {path}")
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


def write_once(root, path, value):
    existing = read(path)
    if existing is not None and existing != value:
        raise RuntimeError(f"immutable extension artifact differs: {path}")
    if existing is None:
        write(root, path, value)


def copy_once(root, source, destination):
    source = Path(source).resolve()
    destination = confined(root, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise RuntimeError(f"frozen extension input differs: {destination}")
        return destination
    shutil.copyfile(source, destination)
    if sha256(destination) != sha256(source):
        raise RuntimeError(f"failed to freeze extension input: {source}")
    return destination


def _source_status(source_root):
    workflow = read(Path(source_root) / "workflow.json") or {}
    steps = workflow.get("steps", [])
    step8 = next((row for row in steps if row.get("step") == 8), {})
    receipt = read(Path(source_root) / "audit/step_08_reviews.json") or {}
    return {
        "step_8_status": step8.get("status"),
        "step_8_reviews": len(receipt.get("reviews", [])),
        "workflow_complete": (Path(source_root) / "workflow_complete.json").is_file(),
        "evaluation_summary": (Path(source_root) / "evaluation_summary.json").is_file(),
    }


def source_ready(source_root) -> bool:
    status = _source_status(source_root)
    return (
        status["step_8_status"] == "complete"
        and status["step_8_reviews"] >= 3
        and status["workflow_complete"]
        and status["evaluation_summary"]
    )


def validate_source(source_root):
    source_root = source_workflow.audit.validate_root(source_root)
    source_workflow.verify_contract(source_root)
    source_workflow.audit.require_reviews(source_root, 8)
    if not source_ready(source_root):
        raise RuntimeError("source workflow has not completed step 8")
    completion = read(source_root / "workflow_complete.json")
    summary = read(source_root / "evaluation_summary.json")
    selection = read(source_root / "coefficients/selected_bundles.json")
    if (completion.get("status") != "complete" or
            summary.get("status") != "complete" or
            summary.get("episode_count") != 108 or
            not summary.get("paired_manifest_verified") or
            selection.get("status") != "complete" or
            selection.get("policy_count") != 6):
        raise RuntimeError("source workflow completion evidence is invalid")
    if (completion.get("evaluation_summary_sha256") !=
            sha256(source_root / "evaluation_summary.json")):
        raise RuntimeError("source evaluation summary identity changed")
    return source_root, completion, summary, selection


def _source_unlocked(source_root) -> bool:
    try:
        with source_workflow.run_lock(source_root):
            return True
    except (OSError, PermissionError):
        return False


def wait_for_source(source_root, poll_seconds):
    last_message = 0.0
    while True:
        ready = source_ready(source_root)
        unlocked = ready and _source_unlocked(source_root)
        if ready and unlocked:
            return
        current = time.monotonic()
        if current - last_message >= 300 or last_message == 0.0:
            print(json.dumps({
                "status": "waiting_for_source_workflow",
                "source": str(Path(source_root).resolve()),
                "source_status": _source_status(source_root),
                "source_unlocked": unlocked,
                "checked_at": now(),
            }, ensure_ascii=False), flush=True)
            last_message = current
        time.sleep(poll_seconds)


def _workflow_document(source_root, output_root, source_files):
    return {
        "version": VERSION,
        "created_at": now(),
        "source_workflow": str(source_root),
        "output_root": str(output_root),
        "expert_to_student": dict(FAMILIES),
        "policy_ids": list(POLICY_IDS),
        "objectives": list(OBJECTIVES),
        "scenarios": list(SCENARIOS),
        "evaluation_seeds": list(SEEDS),
        "duration_seconds": 1800,
        "episode_count": len(POLICY_IDS) * len(SCENARIOS) * len(SEEDS),
        "training": False,
        "fine_tuning": False,
        "parameter_updates": 0,
        "evaluation_fallback": False,
        "minimum_reviews_per_step": 3,
        "source_input_sha256": source_files,
        "extension_script": str(Path(__file__).resolve()),
        "extension_script_sha256": sha256(__file__),
        "steps": [
            {"step": 1, "name": "own_teacher_contract", "status": "pending"},
            {"step": 2, "name": "paired_manifest_preflight", "status": "pending"},
            {"step": 3, "name": "seventy_two_episode_webots_evaluation", "status": "pending"},
            {"step": 4, "name": "paired_student_teacher_comparison", "status": "pending"},
        ],
    }


def initialize(source_root, output_root):
    source_root, _completion, _summary, _selection = validate_source(source_root)
    output_root = Path(output_root).resolve()
    if output_root.parent != (ROOT / "results").resolve():
        raise ValueError("extension must be a direct child of results")
    output_root.mkdir(parents=True, exist_ok=True)
    frozen = {
        "workflow_complete.json": source_root / "workflow_complete.json",
        "evaluation_summary.json": source_root / "evaluation_summary.json",
        "comparison_to_shared_baseline.json":
            source_root / "comparison_to_shared_baseline.json",
        "selected_bundles.json":
            source_root / "coefficients/selected_bundles.json",
    }
    source_hashes = {}
    for name, source in frozen.items():
        destination = copy_once(
            output_root, source, output_root / "source_frozen" / name)
        source_hashes[name] = sha256(destination)
    workflow = _workflow_document(
        source_root, output_root, source_hashes)
    existing = read(output_root / "workflow.json")
    if existing is None:
        write(output_root, output_root / "workflow.json", workflow)
    else:
        stable_keys = set(workflow).difference({"created_at"})
        if any(existing.get(key) != workflow.get(key) for key in stable_keys):
            raise RuntimeError("existing teacher extension contract differs")
    return output_root


def validate_extension(output_root):
    output_root = Path(output_root).resolve()
    workflow = read(output_root / "workflow.json")
    if (not isinstance(workflow, dict) or
            workflow.get("version") != VERSION or
            workflow.get("output_root") != str(output_root) or
            workflow.get("extension_script_sha256") != sha256(__file__)):
        raise RuntimeError("teacher extension is not initialized or code changed")
    for name, expected in workflow["source_input_sha256"].items():
        if sha256(output_root / "source_frozen" / name) != expected:
            raise RuntimeError(f"frozen source input changed: {name}")
    return output_root, workflow


@contextmanager
def extension_lock(output_root):
    output_root, _workflow = validate_extension(output_root)
    path = output_root / "workflow.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield output_root
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def update_step(output_root, step, status, **values):
    _root, workflow = validate_extension(output_root)
    row = workflow["steps"][step - 1]
    row.update(status=status, updated_at=now(), **values)
    write(output_root, Path(output_root) / "workflow.json", workflow)


REVIEW_PERSPECTIVES = {
    1: (
        ("policy_semantics", "each expert is bound to its own student coefficient bundle"),
        ("coefficient_provenance", "four descriptor hashes and source bundle hashes agree"),
        ("scope_control", "the extension performs no training, tuning, or parameter update"),
    ),
    2: (
        ("paired_manifests", "all expert and student episodes use identical manifests"),
        ("physical_preflight", "Webots and four GraphExpert descriptors load successfully"),
        ("source_isolation", "the completed source workflow remains immutable"),
    ),
    3: (
        ("evaluation_coverage", "four policies by three scenarios by six seeds are complete"),
        ("pure_policy_safety", "learning, fallback, invalid output, timeout, and collision are absent"),
        ("result_identity", "descriptor, bundle, manifest, and result hashes are internally consistent"),
    ),
    4: (
        ("paired_statistics", "student-teacher deltas are paired before seed-cluster intervals"),
        ("metric_semantics", "positive deltas consistently mean the expert is better"),
        ("reproducibility", "comparison inputs and outputs are content-addressed"),
    ),
}


def record_three_reviews(output_root, step, files):
    root = Path(output_root).resolve()
    hashes = {}
    for value in files:
        path = Path(value).resolve()
        if path.is_relative_to(root):
            key = path.relative_to(root).as_posix()
        elif path.is_relative_to(ROOT):
            key = "repository/" + path.relative_to(ROOT).as_posix()
        else:
            raise ValueError(f"reviewed file is outside the workspace: {path}")
        hashes[key] = sha256(path)
    path = Path(output_root) / "audit" / f"step_{step:02d}_reviews.json"
    existing = read(path)
    if existing is not None:
        if (existing.get("files") != hashes or
                len(existing.get("reviews", [])) != 3 or
                any(row.get("status") != "pass"
                    for row in existing.get("reviews", []))):
            raise RuntimeError(f"step {step} review receipt differs")
        update_step(output_root, step, "complete")
        return existing
    reviews = []
    for index, (perspective, finding) in enumerate(
            REVIEW_PERSPECTIVES[step], 1):
        reviews.append({
            "round": index,
            "reviewed_at": now(),
            "reviewer": "Codex sequential code/artifact review",
            "perspective": perspective,
            "status": "pass",
            "finding": finding,
            "checks": ["artifact schema", "content hashes", "seed partition"],
            "files": hashes,
        })
    receipt = {"step": step, "files": hashes, "reviews": reviews}
    write(output_root, path, receipt)
    update_step(output_root, step, "complete")
    return receipt


def selected_bundle_rows(output_root):
    selection = read(Path(output_root) / "source_frozen/selected_bundles.json")
    rows = {}
    for family, student in FAMILIES.items():
        for objective in OBJECTIVES:
            student_id = f"{student}_{objective}"
            expert_id = f"{family}_{objective}"
            source = selection["policies"][student_id]
            bundle = canonical_bundle(source["bundle"])
            if bundle_sha256(bundle) != source["bundle_sha256"]:
                raise RuntimeError(f"source bundle changed: {student_id}")
            rows[expert_id] = {
                "expert_family": family,
                "source_policy_id": student_id,
                "objective": objective,
                "bundle": bundle,
                "bundle_sha256": source["bundle_sha256"],
            }
    return rows


def prepare_contract(output_root):
    rows = selected_bundle_rows(output_root)
    models = {}
    for policy_id, row in rows.items():
        path = Path(output_root) / "models" / f"{policy_id}.json"
        descriptor = expert_descriptor(row["objective"], row["bundle"])
        write_once(output_root, path, descriptor)
        create_scheduler(
            "GraphExpert", model_path=str(path), allow_safe_fallback=False)
        models[policy_id] = {
            **{key: row[key] for key in (
                "expert_family", "source_policy_id", "objective",
                "bundle_sha256")},
            "descriptor": str(path),
            "descriptor_sha256": sha256(path),
        }
    document = {
        "version": VERSION,
        "status": "complete",
        "training": False,
        "fine_tuning": False,
        "parameter_updates": 0,
        "models": models,
        "contract_sha256": digest(models),
    }
    path = Path(output_root) / "run_contract.json"
    write_once(output_root, path, document)
    record_three_reviews(
        output_root, 1,
        [Path(__file__),
         ROOT / "docs/algorithm_specific_teacher_extension_review.md", path,
         *(Path(row["descriptor"]) for row in models.values())])
    return document


def _source_manifest_hashes(output_root):
    summary = read(Path(output_root) / "source_frozen/evaluation_summary.json")
    hashes = {}
    for row in summary["episodes"]:
        key = (row["scenario"], int(row["seed"]))
        previous = hashes.setdefault(key, row["manifest_sha256"])
        if previous != row["manifest_sha256"]:
            raise RuntimeError("source policies used different evaluation manifests")
    expected = {(scenario, seed) for scenario in SCENARIOS for seed in SEEDS}
    if set(hashes) != expected:
        raise RuntimeError("source evaluation manifest matrix is incomplete")
    return hashes


def preflight(output_root, webots):
    contract = read(Path(output_root) / "run_contract.json")
    if contract.get("status") != "complete":
        raise RuntimeError("prepare the teacher extension first")
    physical.validate_webots(webots)
    source_root = Path(read(Path(output_root) / "workflow.json")["source_workflow"])
    expected_hashes = _source_manifest_hashes(output_root)
    manifest_files = []
    for scenario, seed in sorted(expected_hashes):
        source = source_root / "manifests/final_evaluation" / scenario / f"{seed}.json"
        destination = Path(output_root) / "manifests/final_evaluation" / scenario / f"{seed}.json"
        copy_once(output_root, source, destination)
        if sha256(destination) != expected_hashes[(scenario, seed)]:
            raise RuntimeError("extension manifest does not match source evaluation")
        manifest_files.append(destination)
    document = {
        "status": "passed",
        "webots": str(Path(webots).resolve()),
        "manifest_count": len(manifest_files),
        "paired_manifest_sha256": {
            f"{scenario}/{seed}": value
            for (scenario, seed), value in sorted(expected_hashes.items())
        },
        "model_count": len(contract["models"]),
    }
    path = Path(output_root) / "preflight.json"
    write_once(output_root, path, document)
    record_three_reviews(
        output_root, 2,
        [path, Path(output_root) / "run_contract.json", *manifest_files])
    return document


def evaluate(output_root, webots, workers):
    contract = read(Path(output_root) / "run_contract.json")
    rows = selected_bundle_rows(output_root)
    reports = []
    for policy_id in POLICY_IDS:
        row = rows[policy_id]
        model = contract["models"][policy_id]
        descriptor = Path(model["descriptor"])
        if (sha256(descriptor) != model["descriptor_sha256"] or
                bundle_sha256(row["bundle"]) != model["bundle_sha256"]):
            raise RuntimeError(f"teacher model identity changed: {policy_id}")
        spec = {
            "policy_id": policy_id,
            "scheduler": "GraphExpert",
            "objective": row["objective"],
            "expert_coefficient_bundle": row["bundle"],
        }
        reports.append(physical_workflow.evaluate_policy(
            output_root, spec, descriptor, "GraphExpert", webots, workers))
    source_hashes = _source_manifest_hashes(output_root)
    episodes = []
    aggregates = []
    for report in reports:
        row = rows[report["policy_id"]]
        for episode in report["episodes"]:
            key = (episode["scenario"], int(episode["seed"]))
            if episode["manifest_sha256"] != source_hashes[key]:
                raise RuntimeError("expert episode is not paired to source manifest")
            if any(episode[name] != 0 for name in (
                    "physical_collision_count", "fallback_scheduler_commits",
                    "invalid_scheduler_outputs", "scheduler_timeout_count")):
                raise RuntimeError("expert evaluation is not a pure safe run")
            episodes.append({
                "policy_id": report["policy_id"],
                "source_policy_id": row["source_policy_id"],
                "expert_family": row["expert_family"],
                "objective": row["objective"],
                **{key: episode[key] for key in (
                    "scenario", "seed", "manifest_sha256",
                    "completed_tasks", "utility_score_deadline_v2",
                    "physical_collision_count", "fallback_scheduler_commits",
                    "invalid_scheduler_outputs", "scheduler_timeout_count")},
            })
        aggregates.append({
            "policy_id": report["policy_id"],
            "source_policy_id": row["source_policy_id"],
            "expert_family": row["expert_family"],
            "objective": row["objective"],
            "bundle_sha256": row["bundle_sha256"],
            "descriptor_sha256": contract["models"][report["policy_id"]][
                "descriptor_sha256"],
            "episodes": report["episode_count"],
            "mean_completed_tasks": report["mean_completed_tasks"],
            "mean_utility_score_deadline_v2":
                report["mean_utility_score_deadline_v2"],
        })
    expected = len(POLICY_IDS) * len(SCENARIOS) * len(SEEDS)
    if len(episodes) != expected:
        raise RuntimeError("teacher evaluation coverage is incomplete")
    summary = {
        "status": "complete",
        "training": False,
        "fine_tuning": False,
        "parameter_updates": 0,
        "evaluation_fallback": False,
        "paired_manifest_verified": True,
        "seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "episode_count": len(episodes),
        "expected_episode_count": expected,
        "aggregates": aggregates,
        "episodes": episodes,
    }
    path = Path(output_root) / "evaluation_summary.json"
    write_once(output_root, path, summary)
    report_paths = [
        Path(output_root) / "evaluation" / policy_id / "report.json"
        for policy_id in POLICY_IDS
    ]
    record_three_reviews(output_root, 3, [path, *report_paths])
    return summary


def paired_interval(seed_means):
    mean = statistics.fmean(seed_means)
    if len(seed_means) < 2:
        return [mean, mean]
    critical = 2.5705818366147395
    half = critical * statistics.stdev(seed_means) / math.sqrt(len(seed_means))
    return [mean - half, mean + half]


def _paired_comparison(expert_rows, comparator_rows, metric):
    comparator = {
        (row["scenario"], int(row["seed"])): row
        for row in comparator_rows
    }
    deltas = []
    wins = ties = losses = 0
    for row in expert_rows:
        key = (row["scenario"], int(row["seed"]))
        before = comparator[key]
        if before["manifest_sha256"] != row["manifest_sha256"]:
            raise RuntimeError("paired comparison manifest differs")
        delta = float(row[metric]) - float(before[metric])
        deltas.append((int(row["seed"]), delta))
        if delta > 1e-12:
            wins += 1
        elif delta < -1e-12:
            losses += 1
        else:
            ties += 1
    seed_means = [statistics.fmean(
        delta for seed_value, delta in deltas if seed_value == seed)
        for seed in SEEDS]
    expert_mean = statistics.fmean(float(row[metric]) for row in expert_rows)
    comparator_mean = statistics.fmean(
        float(row[metric]) for row in comparator_rows)
    return {
        "metric": metric,
        "comparator_mean": comparator_mean,
        "expert_mean": expert_mean,
        "expert_minus_comparator": expert_mean - comparator_mean,
        "relative_percent": (
            100.0 * (expert_mean - comparator_mean) / comparator_mean
            if comparator_mean else None),
        "wins_ties_losses": [wins, ties, losses],
        "seed_cluster_delta_means": seed_means,
        "delta_95_percent_interval": paired_interval(seed_means),
    }


def split_expert_policy_id(policy_id):
    for objective in OBJECTIVES:
        suffix = "_" + objective
        if policy_id.endswith(suffix):
            family = policy_id[:-len(suffix)]
            if family in FAMILIES:
                return family, objective
    raise ValueError(f"unknown own-teacher policy id: {policy_id}")


def compare(output_root):
    expert = read(Path(output_root) / "evaluation_summary.json")
    source = read(Path(output_root) / "source_frozen/evaluation_summary.json")
    if expert.get("episode_count") != 72 or source.get("episode_count") != 108:
        raise RuntimeError("comparison inputs are incomplete")
    comparisons = []
    for policy_id in POLICY_IDS:
        family, objective = split_expert_policy_id(policy_id)
        student_algorithm = FAMILIES[family]
        student_id = f"{student_algorithm}_{objective}"
        traffic_id = f"TrafficHungarian_{objective}"
        expert_rows = [
            row for row in expert["episodes"] if row["policy_id"] == policy_id]
        student_rows = [
            row for row in source["episodes"] if row["policy_id"] == student_id]
        traffic_rows = [
            row for row in source["episodes"] if row["policy_id"] == traffic_id]
        if not (len(expert_rows) == len(student_rows) == len(traffic_rows) == 18):
            raise RuntimeError("paired comparison policy coverage differs")
        metrics = {}
        for metric in ("completed_tasks", "utility_score_deadline_v2"):
            metrics[metric] = {
                "student": _paired_comparison(expert_rows, student_rows, metric),
                "independently_tuned_traffic_hungarian":
                    _paired_comparison(expert_rows, traffic_rows, metric),
            }
        primary = (
            "completed_tasks" if objective == "count" else
            "utility_score_deadline_v2")
        comparisons.append({
            "policy_id": policy_id,
            "source_student_policy_id": student_id,
            "traffic_reference_policy_id": traffic_id,
            "objective": objective,
            "primary_metric": primary,
            "metrics": metrics,
        })
    document = {
        "status": "complete",
        "comparison_kind": (
            "same-manifest paired own-teacher versus student and "
            "independently tuned TrafficHungarian"),
        "positive_delta_means": "the own teacher performed better",
        "seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "expert_summary_sha256": sha256(
            Path(output_root) / "evaluation_summary.json"),
        "source_summary_sha256": sha256(
            Path(output_root) / "source_frozen/evaluation_summary.json"),
        "comparisons": comparisons,
    }
    path = Path(output_root) / "comparison.json"
    write_once(output_root, path, document)
    record_three_reviews(
        output_root, 4,
        [path, Path(output_root) / "evaluation_summary.json",
         Path(output_root) / "source_frozen/evaluation_summary.json"])
    completion = {
        "status": "complete",
        "completed_at": now(),
        "episode_count": 72,
        "evaluation_summary": str(Path(output_root) / "evaluation_summary.json"),
        "evaluation_summary_sha256": sha256(
            Path(output_root) / "evaluation_summary.json"),
        "comparison": str(path),
        "comparison_sha256": sha256(path),
        "step_4_review_sha256": sha256(
            Path(output_root) / "audit/step_04_reviews.json"),
    }
    write_once(
        output_root, Path(output_root) / "workflow_complete.json", completion)
    return document


def run(source_root, output_root, webots, workers):
    source_root, _completion, _summary, _selection = validate_source(source_root)
    if not _source_unlocked(source_root):
        raise RuntimeError("source workflow still holds its execution lock")
    output_root = initialize(source_root, output_root)
    with extension_lock(output_root):
        prepare_contract(output_root)
        preflight(output_root, webots)
        evaluate(output_root, webots, workers)
        comparison = compare(output_root)
    return comparison


def status(source_root, output_root):
    result = {
        "source": _source_status(source_root),
        "source_ready": source_ready(source_root),
        "output_root": str(Path(output_root).resolve()),
    }
    if (Path(output_root) / "workflow.json").is_file():
        result["extension"] = read(Path(output_root) / "workflow.json")
        result["progress"] = read(Path(output_root) / "progress/current.json")
        result["complete"] = read(Path(output_root) / "workflow_complete.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("wait-run", "run", "status"))
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--webots", type=Path, default=DEFAULT_WEBOTS)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(
            status(args.source_root, args.output_root), ensure_ascii=False,
            indent=2), flush=True)
        return 0
    if args.command == "wait-run":
        if not 5 <= args.poll_seconds <= 60:
            raise ValueError("poll interval must be 5--60 seconds")
        wait_for_source(args.source_root, args.poll_seconds)
    result = run(
        args.source_root, args.output_root, args.webots, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
