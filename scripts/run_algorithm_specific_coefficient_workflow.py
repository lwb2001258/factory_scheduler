"""Optimize per-algorithm coefficients, retrain, fine-tune, and evaluate.

GraphImitation, LearnedHungarian, and TrafficHungarian each own separate Count
and Utility-v2 bundles.  Student candidate fitness is measured only after a
fresh inner training pass.  The physical command fine-tunes all students and
then enters six-seed frozen evaluation in the same process.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "scripts"), str(ROOT / "controllers/factory_supervisor")]

import algorithm_specific_coefficient_workflow_contract as audit
import numpy as np
import run_expert_coefficient_workflow as shared
import run_strict_1800_workflow as strict
import run_webots_1800_workflow as physical
from coefficient_optimizer import SeparableCMAES, OPTIMIZER_VERSION
from expert_coefficients import (
    bundle_sha256, canonical_bundle, row_from_baseline_log_multipliers,
    search_fields, with_optimized_rows,
)
from graph_expert_scheduler import expert_descriptor
from schedulers import create_scheduler
from traffic_features import TRAFFIC_VERSION


SCENARIOS = ("A", "B", "C")
OBJECTIVES = ("count", "utility_v2")
ALGORITHMS = ("GraphImitation", "LearnedHungarian", "TrafficHungarian")
STUDENT_ALGORITHMS = ("GraphImitation", "LearnedHungarian")
POLICY_IDS = tuple(
    f"{algorithm}_{objective}"
    for algorithm in ALGORITHMS for objective in OBJECTIVES)
INNER_TRAINING_SEEDS = (11000, 11001, 11002)
OUTER_VALIDATION_SEEDS = (11003, 11004)
TRAINING_SEEDS = tuple(range(21000, 21020))
FINE_TUNE_SEEDS = tuple(range(41000, 41010))
EVALUATION_SEEDS = tuple(range(61000, 61006))
DEFAULT_WEBOTS = Path("C:/Program Files/Webots/msys64/mingw64/bin/webots.exe")
CONTRACT_VERSION = "algorithm-specific-coefficient-retraining-run-v1"
SEARCH_GENERATIONS = 2
SEARCH_POPULATION = {"count": 6, "utility_v2": 7}
SEARCH_BOUNDS = (0.25, 4.0)
SEARCH_INITIAL_SIGMA = 0.45
STABILITY_PENALTY = 0.25
UTILITY_THROUGHPUT_TOLERANCE = 2.0
UTILITY_THROUGHPUT_PENALTY = 1.0
SEARCH_OPTIMIZER_SEED = {
    "GraphImitation_count": 74101,
    "GraphImitation_utility_v2": 74102,
    "LearnedHungarian_count": 74201,
    "LearnedHungarian_utility_v2": 74202,
    "TrafficHungarian_count": 74301,
    "TrafficHungarian_utility_v2": 74302,
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ensure_ascii=True).encode("utf-8")).hexdigest()


def read(path):
    return audit.read(path)


def write(root, path, value):
    audit.write(root, path, value)


def write_once(root, path, value):
    existing = read(path)
    if existing is not None and existing != value:
        raise RuntimeError(f"immutable artifact already differs: {path}")
    if existing is None:
        write(root, path, value)


@contextmanager
def run_lock(root):
    root = audit.validate_root(root)
    path = root / "workflow.lock"
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
            yield root
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def progress(root, phase, **values):
    document = {"phase": phase, "updated_at": audit.now(), **values}
    write(root, Path(root) / "progress/current.json", document)
    print(json.dumps(document, ensure_ascii=False), flush=True)


def source_hashes():
    paths = sorted({
        path for directory in (ROOT / "controllers", ROOT / "scripts")
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts})
    result = {
        path.relative_to(ROOT).as_posix(): audit.sha256(path) for path in paths}
    for path in (
            ROOT / "config/formal_1800/calibration.json",
            ROOT / "docs/algorithm_specific_expert_coefficients_design_review.md",
            ROOT / "docs/algorithm_specific_coefficient_workflow_review.md"):
        result[path.relative_to(ROOT).as_posix()] = audit.sha256(path)
    return result


def static_contract(root):
    root = audit.validate_root(root)
    workflow = read(root / "workflow.json")
    base = root / "baseline/shared_selected_bundle.json"
    baseline_evaluation = root / "baseline/shared_evaluation_summary.json"
    document = {
        "version": CONTRACT_VERSION,
        "sources": source_hashes(),
        "calibration_sha256": audit.sha256(strict.CALIBRATION),
        "traffic_version": TRAFFIC_VERSION,
        "optimizer_version": OPTIMIZER_VERSION,
        "algorithms": list(ALGORITHMS),
        "objectives": list(OBJECTIVES),
        "policy_ids": list(POLICY_IDS),
        "scenarios": list(SCENARIOS),
        "horizon_seconds": 1800,
        "inner_training_seeds": list(INNER_TRAINING_SEEDS),
        "outer_validation_seeds": list(OUTER_VALIDATION_SEEDS),
        "student_training_seeds": list(TRAINING_SEEDS),
        "webots_fine_tune_seeds": list(FINE_TUNE_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "search_generations": SEARCH_GENERATIONS,
        "search_population": SEARCH_POPULATION,
        "search_multiplier_bounds": list(SEARCH_BOUNDS),
        "fitness_stability_penalty": STABILITY_PENALTY,
        "utility_throughput_tolerance": UTILITY_THROUGHPUT_TOLERANCE,
        "utility_throughput_penalty": UTILITY_THROUGHPUT_PENALTY,
        "warm_start_sha256": audit.sha256(base),
        "baseline_evaluation_sha256": audit.sha256(baseline_evaluation),
        "workflow_identity_sha256": digest({
            key: workflow[key] for key in (
                "algorithms", "objectives", "policy_ids",
                "coefficient_inner_training_seeds",
                "coefficient_outer_validation_seeds",
                "student_training_seeds", "webots_fine_tune_seeds",
                "evaluation_seeds")}),
        "regression_validation": False,
        "rollback": False,
        "selection": "latest cumulative checkpoint",
        "physical_handoff": "fine tune then immediate frozen evaluation",
    }
    path = root / "run_contract.json"
    existing = read(path)
    if existing is not None and existing != document:
        raise RuntimeError("source or algorithm-specific contract changed")
    if existing is None:
        write(root, path, document)
    return document


def verify_contract(root):
    document = static_contract(root)
    for relative, expected in document["sources"].items():
        if audit.sha256(ROOT / relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    return document


SEED_PARTITIONS = {
    "coefficient_inner": (
        "standalone_tuning_training", INNER_TRAINING_SEEDS),
    "coefficient_outer": (
        "standalone_tuning_training", OUTER_VALIDATION_SEEDS),
    "student_training": ("standalone_training", TRAINING_SEEDS),
    "webots_finetune": ("webots_finetune", FINE_TUNE_SEEDS),
    "final_evaluation": ("final_evaluation", EVALUATION_SEEDS),
}


def manifest(root, partition, scenario, seed):
    if partition not in SEED_PARTITIONS or scenario not in SCENARIOS:
        raise ValueError("unknown seed partition or scenario")
    directory, allowed = SEED_PARTITIONS[partition]
    if seed not in allowed:
        raise ValueError("seed is outside its registered partition")
    source = strict.manifest_path(directory, scenario, seed)
    data = strict._load_manifest(source)
    destination = audit.confined(
        root, Path(root) / "manifests" / partition / scenario / f"{seed}.json")
    if destination.exists():
        if audit.sha256(destination) != audit.sha256(source):
            raise RuntimeError("copied manifest changed")
    else:
        strict.atomic_copy(source, destination)
    if strict._load_manifest(destination) != data:
        raise RuntimeError("copied manifest content changed")
    return destination


def preflight(root, webots=None):
    verify_contract(root)
    for partition, (_directory, seeds) in SEED_PARTITIONS.items():
        for scenario in SCENARIOS:
            for seed in seeds:
                manifest(root, partition, scenario, seed)
    if webots is not None:
        physical.validate_webots(webots)
    return {"status": "passed", "manifest_count": sum(
        len(seeds) * len(SCENARIOS)
        for _directory, seeds in SEED_PARTITIONS.values())}


def _policy_parts(policy_id):
    for algorithm in ALGORITHMS:
        prefix = algorithm + "_"
        if policy_id.startswith(prefix):
            objective = policy_id[len(prefix):]
            if objective in OBJECTIVES:
                return algorithm, objective
    raise ValueError(f"unknown policy id: {policy_id}")


def warm_start_bundle(root):
    document = read(Path(root) / "baseline/shared_selected_bundle.json")
    if not isinstance(document, dict) or document.get("status") != "complete":
        raise RuntimeError("shared coefficient warm start is incomplete")
    bundle = canonical_bundle(document.get("bundle"))
    if document.get("bundle_sha256") != bundle_sha256(bundle):
        raise RuntimeError("shared coefficient warm-start identity changed")
    return bundle


def coefficient_bundle_for(root, policy_id, vector):
    _algorithm, objective = _policy_parts(policy_id)
    base = warm_start_bundle(root)
    rows = dict(base["objectives"])
    rows[objective] = row_from_baseline_log_multipliers(
        objective, rows[objective], vector,
        minimum_multiplier=SEARCH_BOUNDS[0],
        maximum_multiplier=SEARCH_BOUNDS[1])
    return with_optimized_rows(
        rows, method=f"{OPTIMIZER_VERSION}-algorithm-specific-v1")


def policy_spec(policy_id, bundle):
    algorithm, _objective = _policy_parts(policy_id)
    if algorithm not in STUDENT_ALGORITHMS:
        raise ValueError("TrafficHungarian has no trainable student spec")
    spec = dict(next(
        item for item in strict.AI_SPECS if item["policy_id"] == policy_id))
    spec.update(
        traffic_version=TRAFFIC_VERSION,
        expert_coefficient_bundle=canonical_bundle(bundle),
        training_parameters=strict.training_parameters(spec))
    strict.training_parameters(spec)
    return spec


def _candidate_student_episode(root, spec, candidate_hash, current, history,
                               scenario, seed, contract_hash):
    directory = (Path(root) / "coefficient_search" / spec["policy_id"] /
                 "candidates" / candidate_hash / "inner_training" /
                 scenario / str(seed))
    path = directory / "result.json"
    task_manifest = manifest(root, "coefficient_inner", scenario, seed)
    source_bootstrap = Path(current).resolve() == Path(spec["source"]).resolve()
    identity = {
        "contract_sha256": contract_hash,
        "policy_id": spec["policy_id"],
        "coefficient_sha256": candidate_hash,
        "input_checkpoint_sha256": (
            "fresh-expert-bootstrap" if source_bootstrap else
            audit.sha256(current)),
        "prior_sample_sha256": [audit.sha256(value) for value in history],
        "manifest_sha256": audit.sha256(task_manifest),
        "scenario": scenario,
        "seed": seed,
        "fit_model": scenario == "C" or source_bootstrap,
    }
    existing = read(path)
    if existing is not None:
        if any(existing.get(key) != value for key, value in identity.items()):
            raise RuntimeError("candidate student episode identity differs")
        checkpoint = Path(existing["checkpoint"])
        samples = Path(existing["samples"])
        if (audit.sha256(checkpoint) != existing["checkpoint_sha256"] or
                audit.sha256(samples) != existing["samples_sha256"]):
            raise RuntimeError("candidate student artifact changed")
        strict.load_model(spec, checkpoint, seed)
        return checkpoint, samples, existing
    checkpoint = directory / "model.npz"
    samples = directory / "samples.npz"
    metrics = strict.run_headless_episode(
        spec, current, checkpoint, samples,
        strict._load_manifest(task_manifest), scenario, seed, history,
        fit_supervised=identity["fit_model"],
        training_seed_partition="standalone_tuning_training")
    record = {
        **identity,
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": audit.sha256(checkpoint),
        "samples": str(samples),
        "samples_sha256": audit.sha256(samples),
        "metrics": metrics,
    }
    write(root, path, record)
    return checkpoint, samples, record


def train_candidate_student(root, spec, candidate_hash, contract_hash):
    report_path = (Path(root) / "coefficient_search" / spec["policy_id"] /
                   "candidates" / candidate_hash / "inner_training_report.json")
    existing = read(report_path)
    if existing is not None:
        if (existing.get("status") != "complete" or
                existing.get("contract_sha256") != contract_hash or
                existing.get("coefficient_sha256") != candidate_hash or
                audit.sha256(existing["checkpoint"]) !=
                existing["checkpoint_sha256"]):
            raise RuntimeError("candidate training report differs")
        strict.load_model(spec, Path(existing["checkpoint"]), 0)
        return Path(existing["checkpoint"]), existing
    current = Path(spec["source"])
    history = []
    episodes = []
    for seed in INNER_TRAINING_SEEDS:
        for scenario in SCENARIOS:
            current, samples, record = _candidate_student_episode(
                root, spec, candidate_hash, current, history,
                scenario, seed, contract_hash)
            history.append(samples)
            episodes.append(record)
    report = {
        "status": "complete",
        "contract_sha256": contract_hash,
        "policy_id": spec["policy_id"],
        "coefficient_sha256": candidate_hash,
        "checkpoint": str(current),
        "checkpoint_sha256": audit.sha256(current),
        "inner_training_seeds": list(INNER_TRAINING_SEEDS),
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    write(root, report_path, report)
    return current, report


def traffic_descriptor(root, policy_id, bundle, directory):
    _algorithm, objective = _policy_parts(policy_id)
    descriptor = expert_descriptor(objective, bundle)
    path = Path(directory) / "descriptor.json"
    existing = read(path)
    if existing is not None and existing != descriptor:
        raise RuntimeError("TrafficHungarian candidate descriptor changed")
    if existing is None:
        write(root, path, descriptor)
    return path


def _frozen_candidate_episode(root, policy_id, bundle, checkpoint,
                              scenario, seed, contract_hash):
    algorithm, objective = _policy_parts(policy_id)
    candidate_hash = bundle_sha256(bundle)
    directory = (Path(root) / "coefficient_search" / policy_id /
                 "candidates" / candidate_hash)
    path = directory / "outer_validation" / scenario / str(seed) / "result.json"
    task_manifest = manifest(root, "coefficient_outer", scenario, seed)
    checkpoint_hash = audit.sha256(checkpoint)
    identity = {
        "contract_sha256": contract_hash,
        "policy_id": policy_id,
        "scheduler": algorithm,
        "objective": objective,
        "coefficient_sha256": candidate_hash,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": audit.sha256(task_manifest),
        "scenario": scenario,
        "seed": seed,
    }
    existing = read(path)
    if existing is not None:
        metric_names = (
            "completed_tasks", "utility_score_deadline_v2",
            "fallback_scheduler_commits", "invalid_scheduler_outputs",
            "scheduler_timeout_count")
        if (existing.get("status") != "complete" or
                any(existing.get(key) != value
                    for key, value in identity.items()) or
                any(name not in existing for name in metric_names) or
                any(isinstance(existing[name], bool) or
                    not isinstance(existing[name], (int, float)) or
                    not math.isfinite(float(existing[name]))
                    for name in metric_names) or
                existing.get("fixed_horizon") is not True or
                any(existing[name] != 0 for name in (
                    "fallback_scheduler_commits",
                    "invalid_scheduler_outputs",
                    "scheduler_timeout_count"))):
            raise RuntimeError("candidate validation identity differs")
        return existing
    spec = (policy_spec(policy_id, bundle)
            if algorithm in STUDENT_ALGORITHMS else None)
    if spec is not None:
        strict.load_model(spec, checkpoint, seed)
    before_hash = audit.sha256(checkpoint)
    scheduler = create_scheduler(
        algorithm, model_path=str(checkpoint), seed=seed,
        allow_safe_fallback=False)
    data = strict._load_manifest(task_manifest)
    environment, robots, tasks, context, limit, profile = (
        strict._episode_environment(data, scenario, seed, objective))
    context = replace(context, configuration={
        **context.configuration, "graph_traffic_enabled": True})
    _state, metadata = environment.reset(robots, tasks, context, seed=seed)
    if metadata.get("reward_profile_hash") != profile.sha256:
        raise RuntimeError("candidate validation reward profile mismatch")
    fallback = invalid = timeout = 0
    terminated = False
    started = time.perf_counter()
    for _index in range(limit):
        _state, _reward, terminated, truncated, info = (
            environment.step_scheduler(scheduler))
        diagnostics = info.get("scheduler_diagnostics", {})
        fallback += int(bool(
            isinstance(diagnostics, dict) and diagnostics.get("fallback") and
            not info.get("assignment_rejected")))
        invalid += int(bool(
            info.get("invalid_action") or info.get("assignment_rejected") or
            info.get("invalid_truncation") or (truncated and not terminated)))
        timeout += int(bool(
            isinstance(diagnostics, dict) and diagnostics.get("timeout")))
        if terminated or truncated:
            break
    if not terminated or fallback or invalid or timeout:
        raise RuntimeError("candidate validation is not a pure fixed-horizon run")
    runtime = strict._validate_runtime(environment, data)
    utility = environment.formal_utility_evaluation(strict.utility_score_config())
    if audit.sha256(checkpoint) != before_hash:
        raise RuntimeError("candidate validation mutated checkpoint")
    record = {
        **identity,
        "status": "complete",
        "completed_tasks": int(runtime["tasks_completed"]),
        "utility_score_deadline_v2": float(
            utility["utility_score_deadline_v2"]),
        "fallback_scheduler_commits": fallback,
        "invalid_scheduler_outputs": invalid,
        "scheduler_timeout_count": timeout,
        "fixed_horizon": runtime["fixed_horizon"],
        "wall_seconds": time.perf_counter() - started,
    }
    write(root, path, record)
    return record


def evaluate_candidate(root_value, policy_id, vector, contract_hash,
                       throughput_floor=None):
    root = Path(root_value)
    _algorithm, objective = _policy_parts(policy_id)
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.shape != (len(search_fields(objective)),):
        raise ValueError("coefficient candidate dimension is invalid")
    bundle = coefficient_bundle_for(root, policy_id, vector)
    candidate_hash = bundle_sha256(bundle)
    directory = (root / "coefficient_search" / policy_id /
                 "candidates" / candidate_hash)
    algorithm, _objective = _policy_parts(policy_id)
    if algorithm in STUDENT_ALGORITHMS:
        checkpoint, training = train_candidate_student(
            root, policy_spec(policy_id, bundle), candidate_hash, contract_hash)
    else:
        checkpoint = traffic_descriptor(
            root, policy_id, bundle, directory)
        training = None
    rows = [
        _frozen_candidate_episode(
            root, policy_id, bundle, checkpoint, scenario, seed, contract_hash)
        for seed in OUTER_VALIDATION_SEEDS for scenario in SCENARIOS]
    metric_name = (
        "completed_tasks" if objective == "count" else
        "utility_score_deadline_v2")
    seed_means = [statistics.fmean(
        row[metric_name] for row in rows if row["seed"] == seed)
        for seed in OUTER_VALIDATION_SEEDS]
    mean_completed = statistics.fmean(row["completed_tasks"] for row in rows)
    stability = STABILITY_PENALTY * statistics.pstdev(seed_means)
    shortfall = (0.0 if objective == "count" or throughput_floor is None else
                 max(0.0, float(throughput_floor) - mean_completed))
    fitness = (
        statistics.fmean(seed_means) - stability -
        UTILITY_THROUGHPUT_PENALTY * shortfall)
    return {
        "policy_id": policy_id,
        "scheduler": algorithm,
        "objective": objective,
        "vector": vector.tolist(),
        "coefficient_sha256": candidate_hash,
        "bundle": bundle,
        "coefficients": bundle["objectives"][objective],
        "fitness": float(fitness),
        "raw_metric_mean": float(statistics.fmean(seed_means)),
        "stability_penalty": float(stability),
        "throughput_floor": throughput_floor,
        "throughput_shortfall": float(shortfall),
        "mean_completed_tasks": float(mean_completed),
        "seed_cluster_means": seed_means,
        "inner_training": training,
        "episode_count": len(rows),
        "episodes": rows,
    }


def search_policy(root, policy_id, contract_hash, workers):
    _algorithm, objective = _policy_parts(policy_id)
    directory = Path(root) / "coefficient_search" / policy_id
    report_path = directory / "report.json"
    completed = read(report_path)
    if completed is not None:
        if (completed.get("status") != "complete" or
                completed.get("contract_sha256") != contract_hash):
            raise RuntimeError("coefficient search report is incompatible")
        return completed
    dimensions = len(search_fields(objective))
    state_path = directory / "optimizer_state.json"
    checkpoint = read(state_path)
    if checkpoint is None:
        state = SeparableCMAES.create(
            dimensions, seed=SEARCH_OPTIMIZER_SEED[policy_id],
            initial_sigma=SEARCH_INITIAL_SIGMA,
            lower_bound=math.log(SEARCH_BOUNDS[0]),
            upper_bound=math.log(SEARCH_BOUNDS[1]))
        baseline = evaluate_candidate(
            str(root), policy_id, np.zeros(dimensions), contract_hash)
        throughput_floor = (
            baseline["mean_completed_tasks"] - UTILITY_THROUGHPUT_TOLERANCE
            if objective == "utility_v2" else None)
        baseline["throughput_floor"] = throughput_floor
        checkpoint = {
            "contract_sha256": contract_hash,
            "policy_id": policy_id,
            "objective": objective,
            "optimizer": state.to_dict(),
            "throughput_floor": throughput_floor,
            "baseline": baseline,
            "generations": [],
        }
        write(root, state_path, checkpoint)
    else:
        if (checkpoint.get("contract_sha256") != contract_hash or
                checkpoint.get("policy_id") != policy_id or
                checkpoint.get("objective") != objective):
            raise RuntimeError("optimizer resume identity mismatch")
        state = SeparableCMAES.from_dict(checkpoint["optimizer"])
        if len(checkpoint["generations"]) != state.generation:
            raise RuntimeError("optimizer history and state differ")
    history = list(checkpoint["generations"])
    throughput_floor = checkpoint["throughput_floor"]
    while state.generation < SEARCH_GENERATIONS:
        generation = state.generation
        candidates = state.ask(SEARCH_POPULATION[objective])
        progress(
            root, "coefficient_search", policy_id=policy_id,
            generation=generation + 1, population=len(candidates))
        if workers == 1:
            summaries = [evaluate_candidate(
                str(root), policy_id, vector, contract_hash, throughput_floor)
                for vector in candidates]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(
                    evaluate_candidate, str(root), policy_id, vector,
                    contract_hash, throughput_floor) for vector in candidates]
                summaries = [future.result() for future in futures]
        scores = [row["fitness"] for row in summaries]
        state = state.tell(candidates, scores)
        history.append({
            "generation": generation + 1,
            "candidates": summaries,
            "best_fitness": max(scores),
            "optimizer_after": state.to_dict(),
        })
        checkpoint = {
            **checkpoint,
            "optimizer": state.to_dict(),
            "generations": history,
        }
        write(root, state_path, checkpoint)
    candidates = [checkpoint["baseline"]] + [
        candidate for generation in history
        for candidate in generation["candidates"]]
    selected = max(candidates, key=lambda row: (
        row["fitness"], row["coefficient_sha256"]))
    report = {
        "status": "complete",
        "contract_sha256": contract_hash,
        "policy_id": policy_id,
        "scheduler": _policy_parts(policy_id)[0],
        "objective": objective,
        "dimensions": dimensions,
        "generations": SEARCH_GENERATIONS,
        "population": SEARCH_POPULATION[objective],
        "candidate_count": len(candidates),
        "selected": selected,
        "warm_start": checkpoint["baseline"],
        "fitness_improvement": (
            selected["fitness"] - checkpoint["baseline"]["fitness"]),
        "optimizer_final": state.to_dict(),
    }
    write(root, report_path, report)
    return report


def coefficient_search(root, workers=2):
    contract_hash = digest(verify_contract(root))
    reports = [
        search_policy(root, policy_id, contract_hash, workers)
        for policy_id in POLICY_IDS]
    policies = {
        report["policy_id"]: {
            "bundle": canonical_bundle(report["selected"]["bundle"]),
            "bundle_sha256": report["selected"]["coefficient_sha256"],
            "report": str(
                Path(root) / "coefficient_search" /
                report["policy_id"] / "report.json"),
            "fitness_improvement": report["fitness_improvement"],
        } for report in reports}
    selection = {
        "status": "complete",
        "contract_sha256": contract_hash,
        "policy_count": len(policies),
        "policies": policies,
        "selection_sha256": digest({
            policy_id: row["bundle_sha256"]
            for policy_id, row in policies.items()}),
    }
    path = Path(root) / "coefficients/selected_bundles.json"
    write_once(root, path, selection)
    audit.update_step(
        root, 5, "running", artifact=str(path),
        selection_sha256=selection["selection_sha256"])
    return selection


def selected_bundles(root):
    document = read(Path(root) / "coefficients/selected_bundles.json")
    if (not isinstance(document, dict) or document.get("status") != "complete" or
            set(document.get("policies", {})) != set(POLICY_IDS)):
        raise RuntimeError("algorithm-specific coefficient search is incomplete")
    result = {}
    for policy_id, row in document["policies"].items():
        bundle = canonical_bundle(row.get("bundle"))
        if row.get("bundle_sha256") != bundle_sha256(bundle):
            raise RuntimeError(f"selected coefficient changed: {policy_id}")
        result[policy_id] = bundle
    if document.get("selection_sha256") != digest({
            policy_id: bundle_sha256(result[policy_id])
            for policy_id in sorted(result)}):
        raise RuntimeError("selected coefficient collection identity changed")
    return result


def policy_specs(bundles):
    return [policy_spec(f"{algorithm}_{objective}", bundles[
        f"{algorithm}_{objective}"])
        for algorithm in STUDENT_ALGORITHMS for objective in OBJECTIVES]


def training_contract(root, bundles):
    specs = policy_specs(bundles)
    bundle_hashes = {
        policy_id: bundle_sha256(bundle)
        for policy_id, bundle in bundles.items()}
    document = {
        "version": "algorithm-specific-student-training-v1",
        "run_contract_sha256": audit.sha256(Path(root) / "run_contract.json"),
        "expert_coefficient_sha256_by_policy": bundle_hashes,
        "coefficient_selection_sha256": digest(bundle_hashes),
        "training_seeds": list(TRAINING_SEEDS),
        "fine_tune_seeds": list(FINE_TUNE_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "training_parameters": {
            spec["policy_id"]: strict.training_parameters(spec)
            for spec in specs},
        "regression_validation": False,
        "rollback": False,
        "final_checkpoint": "latest cumulative Webots checkpoint",
    }
    path = Path(root) / "training_contract.json"
    write_once(root, path, document)
    return document


def standalone_training(root, workers=2):
    bundles = selected_bundles(root)
    document = training_contract(root, bundles)
    specs = policy_specs(bundles)
    if workers == 1:
        reports = [
            shared._standalone_policy_worker(root, spec, document)
            for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
            futures = [pool.submit(
                shared._standalone_policy_worker, str(root), spec, document)
                for spec in specs]
            reports = [future.result() for future in futures]
    completion = {
        "status": "complete",
        "training_contract_sha256": digest(document),
        "policies": reports,
    }
    write_once(root, Path(root) / "standalone_complete.json", completion)
    audit.update_step(root, 6, "running", artifact="standalone_complete.json")
    return completion


def fine_tune_all(root, bundles, webots, workers):
    document = training_contract(root, bundles)
    reports = [
        shared._fine_tune_policy(root, spec, document, webots, workers)
        for spec in policy_specs(bundles)]
    completion = {
        "status": "complete",
        "training_contract_sha256": digest(document),
        "policies": reports,
    }
    write_once(root, Path(root) / "fine_tune_complete.json", completion)
    audit.update_step(root, 7, "running", artifact="fine_tune_complete.json")
    return completion


def _evaluation_policies(root, bundles, training_document):
    result = []
    training_hash = digest(training_document)
    for spec in policy_specs(bundles):
        report = read(Path(root) / "fine_tune" / spec["policy_id"] / "report.json")
        if (not report or report.get("training_contract_sha256") != training_hash or
                audit.sha256(report["checkpoint"]) != report["checkpoint_sha256"]):
            raise RuntimeError("fine-tuned model is missing")
        result.append((spec, Path(report["checkpoint"]), spec["scheduler"]))
    for objective in OBJECTIVES:
        policy_id = f"TrafficHungarian_{objective}"
        bundle = bundles[policy_id]
        directory = Path(root) / "evaluation/descriptors" / policy_id
        path = traffic_descriptor(root, policy_id, bundle, directory)
        spec = {
            "policy_id": policy_id,
            "scheduler": "TrafficHungarian",
            "objective": objective,
            "expert_coefficient_bundle": bundle,
        }
        result.append((spec, path, "TrafficHungarian"))
    return result


def evaluation_summary(root, reports, bundles):
    rows = []
    for report in reports:
        for episode in report["episodes"]:
            rows.append({
                "policy_id": report["policy_id"],
                "scheduler": report["scheduler"],
                "objective": report["objective"],
                **{key: episode[key] for key in (
                    "scenario", "seed", "manifest_sha256",
                    "completed_tasks", "utility_score_deadline_v2",
                    "physical_collision_count", "fallback_scheduler_commits",
                    "invalid_scheduler_outputs", "scheduler_timeout_count")},
            })
    manifest_groups = {}
    for row in rows:
        key = (row["scenario"], row["seed"])
        manifest_groups.setdefault(key, set()).add(row["manifest_sha256"])
    if any(len(values) != 1 for values in manifest_groups.values()):
        raise RuntimeError("paired evaluation policies used different manifests")
    aggregates = [{
        "policy_id": report["policy_id"],
        "scheduler": report["scheduler"],
        "objective": report["objective"],
        "expert_coefficient_sha256": bundle_sha256(
            bundles[report["policy_id"]]),
        "episodes": report["episode_count"],
        "mean_completed_tasks": report["mean_completed_tasks"],
        "mean_utility_score_deadline_v2":
            report["mean_utility_score_deadline_v2"],
        "total_physical_collisions": sum(
            row["physical_collision_count"] for row in report["episodes"]),
        "total_fallback_commits": sum(
            row["fallback_scheduler_commits"] for row in report["episodes"]),
    } for report in reports]
    document = {
        "status": "complete",
        "coefficient_sha256_by_policy": {
            policy_id: bundle_sha256(bundle)
            for policy_id, bundle in bundles.items()},
        "seeds": list(EVALUATION_SEEDS),
        "scenarios": list(SCENARIOS),
        "episode_count": len(rows),
        "expected_episode_count": len(POLICY_IDS) * len(SCENARIOS) * len(
            EVALUATION_SEEDS),
        "paired_manifest_verified": True,
        "evaluation_learning": False,
        "evaluation_fallback": False,
        "aggregates": aggregates,
        "episodes": rows,
    }
    if document["episode_count"] != document["expected_episode_count"]:
        raise RuntimeError("full algorithm-specific evaluation is incomplete")
    write_once(root, Path(root) / "evaluation_summary.json", document)
    return document


def _paired_interval(seed_means):
    mean = statistics.fmean(seed_means)
    if len(seed_means) < 2:
        return [mean, mean]
    critical = 2.5705818366147395  # t(0.975, df=5), six fixed seeds.
    half = critical * statistics.stdev(seed_means) / math.sqrt(len(seed_means))
    return [mean - half, mean + half]


def compare_to_shared_baseline(root, summary):
    baseline = read(Path(root) / "baseline/shared_evaluation_summary.json")
    if (not isinstance(baseline, dict) or baseline.get("status") != "complete" or
            baseline.get("seeds") != list(EVALUATION_SEEDS)):
        raise RuntimeError("shared baseline evaluation is incompatible")
    old = {row["policy_id"]: row for row in baseline["aggregates"]}
    new = {row["policy_id"]: row for row in summary["aggregates"]}
    old_episodes = {
        (row["policy_id"], row["scenario"], row["seed"]): row
        for row in baseline["episodes"]}
    new_episodes = {
        (row["policy_id"], row["scenario"], row["seed"]): row
        for row in summary["episodes"]}
    comparisons = []
    for policy_id in POLICY_IDS:
        _algorithm, objective = _policy_parts(policy_id)
        metric = (
            "completed_tasks" if objective == "count" else
            "utility_score_deadline_v2")
        aggregate_name = (
            "mean_completed_tasks" if objective == "count" else
            "mean_utility_score_deadline_v2")
        deltas = []
        wins = ties = losses = 0
        for scenario in SCENARIOS:
            for seed in EVALUATION_SEEDS:
                key = (policy_id, scenario, seed)
                before, after = old_episodes[key], new_episodes[key]
                if before["manifest_sha256"] != after["manifest_sha256"]:
                    raise RuntimeError("baseline comparison manifest differs")
                delta = float(after[metric]) - float(before[metric])
                deltas.append((seed, delta))
                if delta > 1e-12:
                    wins += 1
                elif delta < -1e-12:
                    losses += 1
                else:
                    ties += 1
        seed_means = [statistics.fmean(
            delta for row_seed, delta in deltas if row_seed == seed)
            for seed in EVALUATION_SEEDS]
        before = float(old[policy_id][aggregate_name])
        after = float(new[policy_id][aggregate_name])
        comparisons.append({
            "policy_id": policy_id,
            "objective": objective,
            "metric": metric,
            "shared_coefficient_mean": before,
            "algorithm_specific_mean": after,
            "delta": after - before,
            "relative_percent": (
                100.0 * (after - before) / before if before else None),
            "wins_ties_losses": [wins, ties, losses],
            "seed_cluster_delta_means": seed_means,
            "delta_95_percent_interval": _paired_interval(seed_means),
        })
    document = {
        "status": "complete",
        "baseline": str(Path(root) / "baseline/shared_evaluation_summary.json"),
        "baseline_sha256": audit.sha256(
            Path(root) / "baseline/shared_evaluation_summary.json"),
        "algorithm_specific_summary": str(Path(root) / "evaluation_summary.json"),
        "algorithm_specific_summary_sha256": audit.sha256(
            Path(root) / "evaluation_summary.json"),
        "paired_manifests_verified": True,
        "comparisons": comparisons,
    }
    write_once(root, Path(root) / "comparison_to_shared_baseline.json", document)
    return document


def evaluate_all(root, bundles, webots, workers):
    document = training_contract(root, bundles)
    reports = [shared.evaluate_policy(
        root, spec, checkpoint, scheduler, webots, workers)
        for spec, checkpoint, scheduler in _evaluation_policies(
            root, bundles, document)]
    summary = evaluation_summary(root, reports, bundles)
    comparison = compare_to_shared_baseline(root, summary)
    completion = {
        "status": "complete",
        "completed_at": audit.now(),
        "evaluation_summary": str(Path(root) / "evaluation_summary.json"),
        "evaluation_summary_sha256": audit.sha256(
            Path(root) / "evaluation_summary.json"),
        "comparison": str(Path(root) / "comparison_to_shared_baseline.json"),
        "comparison_sha256": audit.sha256(
            Path(root) / "comparison_to_shared_baseline.json"),
    }
    write_once(root, Path(root) / "workflow_complete.json", completion)
    audit.update_step(root, 8, "running", artifact="workflow_complete.json")
    return {"summary": summary, "comparison": comparison}


STATIC_REVIEW_FILES = {
    1: ["docs/algorithm_specific_expert_coefficients_design_review.md"],
    2: [
        "controllers/factory_supervisor/expert_coefficients.py",
        "controllers/factory_supervisor/coefficient_optimizer.py",
        "tests/test_expert_coefficients.py",
        "tests/test_coefficient_optimizer.py",
    ],
    3: [
        "scripts/algorithm_specific_coefficient_workflow_contract.py",
        "scripts/run_algorithm_specific_coefficient_workflow.py",
    ],
    4: [
        "docs/algorithm_specific_coefficient_workflow_review.md",
        "tests/test_algorithm_specific_coefficient_workflow.py",
        "scripts/algorithm_specific_coefficient_workflow_contract.py",
        "scripts/run_algorithm_specific_coefficient_workflow.py",
    ],
}

STATIC_REVIEW_PERSPECTIVES = {
    1: (
        ("algorithm_correctness", "deployed-policy bilevel fitness verified"),
        ("statistics_and_leakage", "four disjoint seed partitions verified"),
        ("safety_scope", "feasibility and collision rules remain nonlearned"),
    ),
    2: (
        ("coefficient_schema", "warm-start row and travel anchor verified"),
        ("optimizer_math", "bounded deterministic CMA-ES state verified"),
        ("compatibility", "default API and checkpoint compatibility preserved"),
    ),
    3: (
        ("state_machine", "candidate and episode resume identities verified"),
        ("model_provenance", "six policy bundles bind training and evaluation"),
        ("physical_handoff", "fine-tune and evaluation share one command"),
    ),
    4: (
        ("workflow_contract", "two workflow review rounds incorporated"),
        ("test_coverage", "focused tests and static compilation verified"),
        ("operational_readiness", "Webots path and manifests preflighted"),
    ),
}


def review_static_step(root, step):
    if step not in STATIC_REVIEW_FILES:
        raise ValueError("static review step must be 1--4")
    receipt = read(Path(root) / "audit" / f"step_{step:02d}_reviews.json") or {}
    completed = len(receipt.get("reviews", []))
    if completed >= 3:
        audit.require_reviews(root, step)
        return receipt
    perspective, finding = STATIC_REVIEW_PERSPECTIVES[step][completed]
    receipt = audit.record_review(
        root, step, perspective=perspective, findings=[finding],
        checks=["source inspection", "contract invariants", "test evidence"],
        files=STATIC_REVIEW_FILES[step])
    if len(receipt["reviews"]) >= 3:
        audit.require_reviews(root, step)
        audit.update_step(root, step, "complete")
    return receipt


def _runtime_review_files(root, step):
    root = Path(root)
    if step == 5:
        return [
            root / "coefficients/selected_bundles.json",
            *(root / "coefficient_search" / policy_id / "report.json"
              for policy_id in POLICY_IDS),
            *(root / "coefficient_search" / policy_id / "optimizer_state.json"
              for policy_id in POLICY_IDS),
        ]
    bundles = selected_bundles(root)
    if step == 6:
        reports = [
            root / "standalone" / spec["policy_id"] / "report.json"
            for spec in policy_specs(bundles)]
        return [
            root / "standalone_complete.json", *reports,
            *(Path(read(path)["checkpoint"]) for path in reports),
            *(strict.model_identity_path(Path(read(path)["checkpoint"]))
              for path in reports),
        ]
    if step == 7:
        reports = [
            root / "fine_tune" / spec["policy_id"] / "report.json"
            for spec in policy_specs(bundles)]
        return [
            root / "fine_tune_complete.json", *reports,
            *(Path(read(path)["checkpoint"]) for path in reports),
            *(strict.model_identity_path(Path(read(path)["checkpoint"]))
              for path in reports),
        ]
    if step == 8:
        return [
            root / "evaluation_summary.json",
            root / "comparison_to_shared_baseline.json",
            root / "workflow_complete.json",
            *(root / "evaluation" / policy_id / "report.json"
              for policy_id in POLICY_IDS),
        ]
    raise ValueError("runtime review step must be 5--8")


def validate_runtime_artifacts(root, step):
    if step == 5:
        bundles = selected_bundles(root)
        if set(bundles) != set(POLICY_IDS):
            raise RuntimeError("six coefficient bundles are incomplete")
        for policy_id in POLICY_IDS:
            report = read(Path(root) / "coefficient_search" / policy_id / "report.json")
            objective = _policy_parts(policy_id)[1]
            if (report.get("candidate_count") !=
                    1 + SEARCH_GENERATIONS * SEARCH_POPULATION[objective] or
                    report.get("selected", {}).get("episode_count") !=
                    len(OUTER_VALIDATION_SEEDS) * len(SCENARIOS)):
                raise RuntimeError("algorithm-specific search coverage is incomplete")
    elif step == 6:
        bundles = selected_bundles(root)
        document = training_contract(root, bundles)
        for spec in policy_specs(bundles):
            report = shared._standalone_report(root, spec, digest(document))
            if (report["balanced_rounds"] != len(TRAINING_SEEDS) or
                    report["training_episodes"] != 60 or
                    report["expert_coefficient_sha256"] != bundle_sha256(
                        bundles[spec["policy_id"]])):
                raise RuntimeError("student standalone coverage is incomplete")
    elif step == 7:
        bundles = selected_bundles(root)
        document = training_contract(root, bundles)
        for spec in policy_specs(bundles):
            report = read(Path(root) / "fine_tune" / spec["policy_id"] / "report.json")
            if (report.get("training_contract_sha256") != digest(document) or
                    report.get("balanced_rounds") != len(FINE_TUNE_SEEDS) or
                    report.get("episodes") != 30 or
                    report.get("expert_coefficient_sha256") != bundle_sha256(
                        bundles[spec["policy_id"]])):
                raise RuntimeError("Webots fine-tune coverage is incomplete")
            strict.load_model(spec, Path(report["checkpoint"]), 0)
    elif step == 8:
        summary = read(Path(root) / "evaluation_summary.json")
        comparison = read(Path(root) / "comparison_to_shared_baseline.json")
        if (summary.get("episode_count") != 108 or
                summary.get("seeds") != list(EVALUATION_SEEDS) or
                not summary.get("paired_manifest_verified") or
                len(comparison.get("comparisons", [])) != 6 or
                any(row["fallback_scheduler_commits"] != 0 or
                    row["invalid_scheduler_outputs"] != 0 or
                    row["scheduler_timeout_count"] != 0 or
                    row["physical_collision_count"] != 0
                    for row in summary["episodes"])):
            raise RuntimeError("frozen evaluation evidence is incomplete")
    else:
        raise ValueError("runtime review step must be 5--8")
    return _runtime_review_files(root, step)


RUNTIME_REVIEW_PERSPECTIVES = {
    5: (
        ("optimization_math", "deployed fitness and throughput guard verified"),
        ("seed_and_resume_identity", "inner/outer seed and candidate hashes verified"),
        ("policy_isolation", "six independently selected bundles verified"),
    ),
    6: (
        ("training_coverage", "four students have twenty balanced rounds"),
        ("model_provenance", "each checkpoint binds its own bundle hash"),
        ("selection_semantics", "latest checkpoints selected without rollback"),
    ),
    7: (
        ("physical_coverage", "four students have thirty Webots episodes"),
        ("supervision_alignment", "physical labels bind policy-specific bundles"),
        ("handoff_readiness", "latest checkpoints load and eval seeds stay unused"),
    ),
    8: (
        ("evaluation_matrix", "six policies by three scenes by six seeds verified"),
        ("pure_policy_safety", "collision/fallback/invalid/timeout totals are zero"),
        ("paired_comparison", "shared-baseline manifests and intervals verified"),
    ),
}


def review_runtime_step(root, step):
    files = validate_runtime_artifacts(root, step)
    receipt = read(Path(root) / "audit" / f"step_{step:02d}_reviews.json") or {}
    completed = len(receipt.get("reviews", []))
    if completed >= 3:
        audit.require_reviews(root, step)
        return receipt
    perspective, finding = RUNTIME_REVIEW_PERSPECTIVES[step][completed]
    receipt = audit.record_review(
        root, step, perspective=perspective, findings=[finding],
        checks=["artifact schema", "identity hashes", "seed partition"],
        files=files)
    if len(receipt["reviews"]) >= 3:
        audit.require_reviews(root, step)
        audit.update_step(root, step, "complete")
    return receipt


def physical_then_evaluate(root, webots, workers):
    bundles = selected_bundles(root)
    fine_tune_all(root, bundles, webots, workers)
    for _ in range(3):
        review_runtime_step(root, 7)
    audit.require_reviews(root, 7)
    result = evaluate_all(root, bundles, webots, workers)
    for _ in range(3):
        review_runtime_step(root, 8)
    return result


def run_all(root, webots, workers):
    audit.require_reviews(root, 4)
    preflight(root, webots)
    coefficient_search(root, workers)
    for _ in range(3):
        review_runtime_step(root, 5)
    audit.require_reviews(root, 5)
    standalone_training(root, workers)
    for _ in range(3):
        review_runtime_step(root, 6)
    audit.require_reviews(root, 6)
    return physical_then_evaluate(root, webots, workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=(
            "init", "review-static", "preflight", "search", "standalone",
            "physical", "run-all", "review-runtime", "status"))
    parser.add_argument("--run-root", type=Path, default=audit.DEFAULT_RUN)
    parser.add_argument("--webots", type=Path, default=DEFAULT_WEBOTS)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--step", type=int, choices=tuple(range(1, 9)))
    args = parser.parse_args()
    if args.command == "init":
        print(json.dumps(audit.initialize(args.run_root), ensure_ascii=False))
        return 0
    root = audit.validate_root(args.run_root)
    if args.command == "status":
        print(json.dumps(read(root / "workflow.json"), ensure_ascii=False, indent=2))
        return 0
    with run_lock(root):
        if args.command == "review-static":
            if args.step not in range(1, 5):
                raise ValueError("--step 1..4 is required for static review")
            result = review_static_step(root, args.step)
        elif args.command == "preflight":
            audit.require_reviews(root, 4)
            result = preflight(root, args.webots)
        elif args.command == "search":
            audit.require_reviews(root, 4)
            result = coefficient_search(root, args.workers)
        elif args.command == "standalone":
            audit.require_reviews(root, 5)
            result = standalone_training(root, args.workers)
        elif args.command == "physical":
            audit.require_reviews(root, 6)
            physical.validate_webots(args.webots)
            result = physical_then_evaluate(root, args.webots, args.workers)
        elif args.command == "run-all":
            physical.validate_webots(args.webots)
            result = run_all(root, args.webots, args.workers)
        else:
            if args.step not in range(5, 9):
                raise ValueError("--step 5..8 is required for runtime review")
            result = review_runtime_step(root, args.step)
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
