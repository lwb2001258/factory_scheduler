"""Learn expert coefficients, retrain students, then run paired Webots evaluation.

The physical command is intentionally atomic at the phase level: after all
Webots fine-tuning checkpoints pass three artifact reviews, frozen evaluation
on seeds 61000--61005 begins in the same process without a pause or selection
gate. Every episode and optimizer generation is independently resumable.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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

import expert_coefficient_workflow_contract as audit
import numpy as np
import run_strict_1800_workflow as strict
import run_webots_1800_workflow as physical
from coefficient_optimizer import SeparableCMAES, OPTIMIZER_VERSION
from expert_coefficients import (
    bundle_sha256, canonical_bundle, default_bundle, row_from_log_multipliers,
    search_fields, with_optimized_rows,
)
from graph_expert_scheduler import expert_descriptor
from schedulers import create_scheduler
from traffic_features import TRAFFIC_VERSION


SCENARIOS = ("A", "B", "C")
OBJECTIVES = ("count", "utility_v2")
COEFFICIENT_SEEDS = (11000, 11001, 11002)
TRAINING_SEEDS = tuple(range(21000, 21020))
FINE_TUNE_SEEDS = tuple(range(41000, 41010))
EVALUATION_SEEDS = tuple(range(61000, 61006))
DEFAULT_WEBOTS = Path("C:/Program Files/Webots/msys64/mingw64/bin/webots.exe")
CONTRACT_VERSION = "expert-coefficient-retraining-run-v1"
SEARCH_GENERATIONS = 4
SEARCH_POPULATION = {"count": 8, "utility_v2": 9}
SEARCH_OPTIMIZER_SEED = {"count": 73101, "utility_v2": 73102}


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
    write(root, Path(root) / "progress" / "current.json", document)
    print(json.dumps(document, ensure_ascii=False), flush=True)


def source_hashes():
    paths = sorted({
        path for directory in (ROOT / "controllers", ROOT / "scripts")
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts})
    result = {path.relative_to(ROOT).as_posix(): audit.sha256(path)
              for path in paths}
    for path in (
            ROOT / "config/formal_1800/calibration.json",
            ROOT / "docs/expert_coefficient_learning_design_review.md"):
        result[path.relative_to(ROOT).as_posix()] = audit.sha256(path)
    return result


def static_contract(root):
    root = audit.validate_root(root)
    document = {
        "version": CONTRACT_VERSION,
        "sources": source_hashes(),
        "calibration_sha256": audit.sha256(strict.CALIBRATION),
        "traffic_version": TRAFFIC_VERSION,
        "coefficient_version": default_bundle()["version"],
        "optimizer_version": OPTIMIZER_VERSION,
        "objectives": list(OBJECTIVES),
        "scenarios": list(SCENARIOS),
        "horizon_seconds": 1800,
        "coefficient_search_seeds": list(COEFFICIENT_SEEDS),
        "student_training_seeds": list(TRAINING_SEEDS),
        "webots_fine_tune_seeds": list(FINE_TUNE_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "search_generations": SEARCH_GENERATIONS,
        "search_population": SEARCH_POPULATION,
        "fitness_stability_penalty": 0.25,
        "student_balanced_rounds": len(TRAINING_SEEDS),
        "webots_fine_tune_rounds": len(FINE_TUNE_SEEDS),
        "regression_validation": False,
        "rollback": False,
        "selection": "latest cumulative checkpoint",
        "physical_handoff": "fine tune then immediate frozen evaluation",
    }
    path = root / "run_contract.json"
    existing = read(path)
    if existing is not None and existing != document:
        raise RuntimeError("source or workflow contract changed after freeze")
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
    "coefficient_search": ("standalone_tuning_training", COEFFICIENT_SEEDS),
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


def coefficient_bundle_for(objective, vector):
    rows = dict(default_bundle()["objectives"])
    rows[objective] = row_from_log_multipliers(objective, vector)
    return with_optimized_rows(rows, method=OPTIMIZER_VERSION)


def _coefficient_episode(root, objective, vector, scenario, seed,
                         contract_hash):
    bundle = coefficient_bundle_for(objective, vector)
    coefficient_hash = bundle_sha256(bundle)
    directory = (Path(root) / "coefficient_search" / objective /
                 "candidates" / coefficient_hash)
    descriptor_path = directory / "descriptor.json"
    descriptor = expert_descriptor(objective, bundle)
    if descriptor_path.exists() and read(descriptor_path) != descriptor:
        raise RuntimeError("candidate descriptor changed")
    if not descriptor_path.exists():
        write(root, descriptor_path, descriptor)
    task_manifest = manifest(root, "coefficient_search", scenario, seed)
    path = directory / "episodes" / scenario / str(seed) / "result.json"
    identity = {
        "contract_sha256": contract_hash,
        "coefficient_sha256": coefficient_hash,
        "descriptor_sha256": audit.sha256(descriptor_path),
        "manifest_sha256": audit.sha256(task_manifest),
        "objective": objective, "scenario": scenario, "seed": seed,
    }
    existing = read(path)
    if existing is not None:
        if (existing.get("status") != "complete" or
                any(existing.get(key) != value
                    for key, value in identity.items()) or
                existing.get("fixed_horizon") is not True or
                any(isinstance(existing.get(name), bool) or
                    not isinstance(existing.get(name), (int, float)) or
                    not math.isfinite(float(existing[name]))
                    for name in (
                        "metric", "completed_tasks",
                        "utility_score_deadline_v2"))):
            raise RuntimeError("coefficient episode resume identity mismatch")
        return existing

    data = strict._load_manifest(task_manifest)
    environment, robots, tasks, context, limit, _profile = (
        strict._episode_environment(data, scenario, seed, objective))
    context = replace(context, configuration={
        **context.configuration, "graph_traffic_enabled": True})
    environment.reset(robots, tasks, context, seed=seed)
    scheduler = create_scheduler(
        "TrafficHungarian", model_path=descriptor_path,
        allow_safe_fallback=False)
    terminated = False
    started = time.perf_counter()
    for _ in range(limit):
        _state, _reward, terminated, truncated, info = (
            environment.step_scheduler(scheduler))
        diagnostics = info.get("scheduler_diagnostics", {})
        if (info.get("invalid_action") or info.get("assignment_rejected") or
                info.get("invalid_truncation") or diagnostics.get("fallback") or
                (truncated and not terminated)):
            raise RuntimeError("invalid coefficient-search transition")
        if terminated:
            break
    if not terminated:
        raise RuntimeError("coefficient-search episode missed fixed horizon")
    runtime = strict._validate_runtime(environment, data)
    utility = environment.formal_utility_evaluation(
        strict.utility_score_config())
    metric = (runtime["tasks_completed"] if objective == "count" else
              utility["utility_score_deadline_v2"])
    record = {
        **identity, "status": "complete", "metric": float(metric),
        "completed_tasks": runtime["tasks_completed"],
        "utility_score_deadline_v2": utility["utility_score_deadline_v2"],
        "fixed_horizon": runtime["fixed_horizon"],
        "wall_seconds": time.perf_counter() - started,
    }
    write(root, path, record)
    return record


def evaluate_candidate(root, objective, vector, contract_hash):
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.shape != (len(search_fields(objective)),):
        raise ValueError("coefficient candidate dimension is invalid")
    rows = []
    for seed in COEFFICIENT_SEEDS:
        for scenario in SCENARIOS:
            rows.append(_coefficient_episode(
                root, objective, vector, scenario, seed, contract_hash))
    seed_means = [statistics.fmean(
        row["metric"] for row in rows if row["seed"] == seed)
        for seed in COEFFICIENT_SEEDS]
    fitness = statistics.fmean(seed_means) - 0.25 * statistics.pstdev(seed_means)
    bundle = coefficient_bundle_for(objective, vector)
    return {
        "objective": objective,
        "vector": vector.tolist(),
        "coefficient_sha256": bundle_sha256(bundle),
        "coefficients": bundle["objectives"][objective],
        "fitness": float(fitness),
        "seed_cluster_means": seed_means,
        "episode_count": len(rows),
        "episodes": rows,
    }


def search_objective(root, objective, contract_hash):
    directory = Path(root) / "coefficient_search" / objective
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
            dimensions, seed=SEARCH_OPTIMIZER_SEED[objective])
        history = []
        baseline = evaluate_candidate(
            root, objective, np.zeros(dimensions), contract_hash)
        checkpoint = {
            "contract_sha256": contract_hash,
            "objective": objective,
            "optimizer": state.to_dict(),
            "baseline": baseline,
            "generations": history,
        }
        write(root, state_path, checkpoint)
    else:
        if (checkpoint.get("contract_sha256") != contract_hash or
                checkpoint.get("objective") != objective):
            raise RuntimeError("optimizer resume identity mismatch")
        state = SeparableCMAES.from_dict(checkpoint["optimizer"])
        history = list(checkpoint["generations"])
        if len(history) != state.generation:
            raise RuntimeError("optimizer history and state differ")

    while state.generation < SEARCH_GENERATIONS:
        generation = state.generation
        candidates = state.ask(SEARCH_POPULATION[objective])
        summaries = []
        for index, vector in enumerate(candidates):
            progress(
                root, "coefficient_search", objective=objective,
                generation=generation + 1, candidate=index + 1,
                population=len(candidates))
            summaries.append(evaluate_candidate(
                root, objective, vector, contract_hash))
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
        "status": "complete", "contract_sha256": contract_hash,
        "objective": objective, "dimensions": dimensions,
        "generations": SEARCH_GENERATIONS,
        "population": SEARCH_POPULATION[objective],
        "candidate_count": len(candidates),
        "selected": selected,
        "default": checkpoint["baseline"],
        "fitness_improvement": (
            selected["fitness"] - checkpoint["baseline"]["fitness"]),
        "optimizer_final": state.to_dict(),
    }
    write(root, report_path, report)
    return report


def coefficient_search(root):
    contract_hash = digest(verify_contract(root))
    reports = [search_objective(root, objective, contract_hash)
               for objective in OBJECTIVES]
    rows = {
        report["objective"]: report["selected"]["coefficients"]
        for report in reports}
    bundle = with_optimized_rows(rows, method=OPTIMIZER_VERSION)
    selection = {
        "status": "complete", "contract_sha256": contract_hash,
        "bundle": bundle, "bundle_sha256": bundle_sha256(bundle),
        "reports": {
            report["objective"]: {
                "path": str(Path(root) / "coefficient_search" /
                            report["objective"] / "report.json"),
                "fitness_improvement": report["fitness_improvement"],
                "selected_candidate_sha256":
                    report["selected"]["coefficient_sha256"],
            } for report in reports},
    }
    path = Path(root) / "coefficients" / "selected_bundle.json"
    existing = read(path)
    if existing is not None and existing != selection:
        raise RuntimeError("selected coefficient bundle changed")
    if existing is None:
        write(root, path, selection)
    audit.update_step(
        root, 6, "running", artifact=str(path),
        bundle_sha256=selection["bundle_sha256"])
    return selection


def selected_coefficients(root):
    document = read(Path(root) / "coefficients" / "selected_bundle.json")
    if not isinstance(document, dict) or document.get("status") != "complete":
        raise RuntimeError("coefficient search is incomplete")
    bundle = canonical_bundle(document.get("bundle"))
    if document.get("bundle_sha256") != bundle_sha256(bundle):
        raise RuntimeError("selected coefficient identity changed")
    return bundle


def training_contract(root, bundle):
    document = {
        "version": "expert-coefficient-student-training-v1",
        "run_contract_sha256": audit.sha256(Path(root) / "run_contract.json"),
        "expert_coefficient_bundle": canonical_bundle(bundle),
        "expert_coefficient_sha256": bundle_sha256(bundle),
        "training_seeds": list(TRAINING_SEEDS),
        "fine_tune_seeds": list(FINE_TUNE_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "training_parameters": {
            spec["policy_id"]: strict.training_parameters(spec)
            for spec in policy_specs(bundle)},
        "regression_validation": False,
        "rollback": False,
        "final_checkpoint": "latest cumulative Webots checkpoint",
    }
    path = Path(root) / "training_contract.json"
    existing = read(path)
    if existing is not None and existing != document:
        raise RuntimeError("student training contract changed")
    if existing is None:
        write(root, path, document)
    return document


def policy_specs(bundle):
    result = []
    for scheduler in ("GraphImitation", "LearnedHungarian"):
        for objective in OBJECTIVES:
            spec = dict(next(
                item for item in strict.AI_SPECS
                if item["policy_id"] == f"{scheduler}_{objective}"))
            spec.update(
                traffic_version=TRAFFIC_VERSION,
                expert_coefficient_bundle=canonical_bundle(bundle),
                training_parameters=strict.training_parameters(spec))
            strict.training_parameters(spec)
            result.append(spec)
    return result


def _student_episode(root, spec, current, history, scenario, seed,
                     training_hash):
    directory = (Path(root) / "standalone" / spec["policy_id"] /
                 "episodes" / scenario / str(seed))
    path = directory / "result.json"
    task_manifest = manifest(root, "student_training", scenario, seed)
    source_bootstrap = Path(current).resolve() == Path(spec["source"]).resolve()
    identity = {
        "training_contract_sha256": training_hash,
        "policy_id": spec["policy_id"],
        "input_checkpoint_sha256": (
            "fresh-expert-bootstrap" if source_bootstrap else
            audit.sha256(current)),
        "prior_sample_sha256": [audit.sha256(value) for value in history],
        "manifest_sha256": audit.sha256(task_manifest),
        "scenario": scenario, "seed": seed,
        "fit_model": scenario == "C" or source_bootstrap,
    }
    existing = read(path)
    if existing is not None:
        if any(existing.get(key) != value for key, value in identity.items()):
            raise RuntimeError("student episode resume identity mismatch")
        checkpoint, samples = Path(existing["checkpoint"]), Path(existing["samples"])
        if (audit.sha256(checkpoint) != existing["checkpoint_sha256"] or
                audit.sha256(samples) != existing["samples_sha256"]):
            raise RuntimeError("student episode artifact changed")
        strict.load_model(spec, checkpoint, seed)
        return checkpoint, samples, existing
    checkpoint = directory / "model.npz"
    samples = directory / "samples.npz"
    progress(root, "student_standalone", policy_id=spec["policy_id"],
             scenario=scenario, seed=seed)
    metrics = strict.run_headless_episode(
        spec, current, checkpoint, samples,
        strict._load_manifest(task_manifest), scenario, seed, history,
        fit_supervised=identity["fit_model"],
        training_seed_partition="standalone_training")
    record = {
        **identity, "status": "complete", "checkpoint": str(checkpoint),
        "checkpoint_sha256": audit.sha256(checkpoint),
        "samples": str(samples), "samples_sha256": audit.sha256(samples),
        "metrics": metrics,
    }
    write(root, path, record)
    return checkpoint, samples, record


def _standalone_policy_worker(root_value, spec, training_document):
    root = Path(root_value)
    training_hash = digest(training_document)
    report_path = root / "standalone" / spec["policy_id"] / "report.json"
    existing = read(report_path)
    if existing is not None:
        if (existing.get("status") != "complete" or
                existing.get("training_contract_sha256") != training_hash):
            raise RuntimeError("standalone report is incompatible")
        strict.load_model(spec, Path(existing["checkpoint"]), 0)
        return existing
    current = Path(spec["source"])
    history = []
    rounds = []
    previous_round_checkpoint = None
    for index, seed in enumerate(TRAINING_SEEDS, 1):
        episodes = []
        for scenario in SCENARIOS:
            current, samples, record = _student_episode(
                root, spec, current, history, scenario, seed, training_hash)
            history.append(samples)
            episodes.append(record)
        change = None
        if previous_round_checkpoint is not None:
            change = strict._policy_parameter_change(
                spec, previous_round_checkpoint, current,
                reference_round=index - 1)
        previous_round_checkpoint = current
        rounds.append({
            "round": index, "seed": seed,
            "checkpoint": str(current),
            "checkpoint_sha256": audit.sha256(current),
            "relative_parameter_change": (
                None if change is None else change["relative_l2_change"]),
            "mean_episode_return": statistics.fmean(
                row["metrics"]["episode_return"] for row in episodes),
        })
    final = root / "standalone" / spec["policy_id"] / "model.npz"
    if final.exists():
        if audit.sha256(final) != audit.sha256(current):
            raise RuntimeError("standalone final checkpoint already differs")
    else:
        strict.atomic_copy_model(spec, current, final)
    changes = [row["relative_parameter_change"] for row in rounds[-3:]
               if row["relative_parameter_change"] is not None]
    report = {
        "status": "complete", "training_contract_sha256": training_hash,
        "policy_id": spec["policy_id"], "checkpoint": str(final),
        "checkpoint_sha256": audit.sha256(final),
        "expert_coefficient_sha256": bundle_sha256(
            spec["expert_coefficient_bundle"]),
        "balanced_rounds": len(rounds),
        "training_episodes": len(rounds) * len(SCENARIOS),
        "selection": "latest cumulative checkpoint",
        "degradation_validation": False, "rollback": False,
        "last_three_parameter_changes": changes,
        "parameter_convergence_observed": (
            len(changes) == 3 and all(value <= 0.01 for value in changes)),
        "rounds": rounds,
    }
    write(root, report_path, report)
    return report


def standalone_training(root, workers=2):
    bundle = selected_coefficients(root)
    document = training_contract(root, bundle)
    specs = policy_specs(bundle)
    if workers == 1:
        reports = [_standalone_policy_worker(root, spec, document)
                   for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
            futures = [pool.submit(
                _standalone_policy_worker, str(root), spec, document)
                for spec in specs]
            reports = [future.result() for future in futures]
    completion = {
        "status": "complete", "training_contract_sha256": digest(document),
        "policies": reports,
    }
    write_once(root, Path(root) / "standalone_complete.json", completion)
    audit.update_step(root, 7, "running", artifact="standalone_complete.json")
    return completion


def _standalone_report(root, spec, training_hash):
    report = read(Path(root) / "standalone" / spec["policy_id"] / "report.json")
    if (not isinstance(report, dict) or report.get("status") != "complete" or
            report.get("training_contract_sha256") != training_hash or
            report.get("checkpoint_sha256") != audit.sha256(report["checkpoint"])):
        raise RuntimeError(f"standalone training incomplete: {spec['policy_id']}")
    strict.load_model(spec, Path(report["checkpoint"]), 0)
    return report


def _physical_episode(root, spec, phase, checkpoint, scenario, seed,
                      webots, scheduler=None):
    if phase not in {"fine_tune", "evaluation"}:
        raise ValueError("physical phase is invalid")
    fine = phase == "fine_tune"
    partition = "webots_finetune" if fine else "final_evaluation"
    task_manifest = manifest(root, partition, scenario, seed)
    scheduler = scheduler or spec["scheduler"]
    algorithm_path = spec.get("policy_id", scheduler)
    path = (Path(root) / phase / algorithm_path / scenario /
            str(seed) / "result.json")
    checkpoint_hash = audit.sha256(checkpoint)
    valid = dict(
        scenario=scenario, seed=seed, manifest=task_manifest,
        input_model_sha256=checkpoint_hash,
        fine_tune_objective=spec["objective"] if fine else None,
        fine_tune_algorithm=scheduler if fine else None,
        expected_scheduler=scheduler)
    existing = read(path)
    if existing is not None and not physical.valid_episode(existing, **valid):
        raise RuntimeError("existing physical episode identity differs")
    progress(root, phase, policy_id=algorithm_path,
             scenario=scenario, seed=seed)
    envelope, _ = physical.run_or_resume_episode(
        canonical_path=path, webots=webots, scenario=scenario, seed=seed,
        manifest=task_manifest, scheduler=scheduler,
        model_path=checkpoint,
        fine_tune_objective=spec["objective"] if fine else None,
        fine_tune_algorithm=scheduler if fine else None)
    if audit.sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("physical episode modified its input checkpoint")
    if (not physical.valid_episode(envelope, **valid) or
            not physical.valid_pure_policy_result(envelope)):
        raise RuntimeError("physical episode is not a pure valid policy run")
    if fine:
        extractor = (strict._graph_imitation_shared_samples
                     if scheduler == "GraphImitation" else
                     strict._learned_hungarian_shared_samples)
        extractor(
            [envelope], spec["objective"], traffic=True,
            expert_coefficient_bundle=spec["expert_coefficient_bundle"])
    return path, envelope


def _fine_tune_policy(root, spec, training_document, webots, workers):
    training_hash = digest(training_document)
    report_path = Path(root) / "fine_tune" / spec["policy_id"] / "report.json"
    existing = read(report_path)
    if existing is not None:
        if (existing.get("status") != "complete" or
                existing.get("training_contract_sha256") != training_hash or
                audit.sha256(existing["checkpoint"]) !=
                existing["checkpoint_sha256"]):
            raise RuntimeError("fine-tune report is incompatible")
        strict.load_model(spec, Path(existing["checkpoint"]), 0)
        return existing
    trained = _standalone_report(root, spec, training_hash)
    current = Path(trained["checkpoint"])
    history = []
    rounds = []
    for index, seed in enumerate(FINE_TUNE_SEEDS, 1):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(
                _physical_episode, root, spec, "fine_tune", current,
                scenario, seed, webots) for scenario in SCENARIOS]
            pairs = [future.result() for future in futures]
        paths = [path for path, _envelope in pairs]
        envelopes = [envelope for _path, envelope in pairs]
        directory = Path(root) / "fine_tune" / spec["policy_id"] / "checkpoints"
        record_path = directory / f"round_{index:02d}.json"
        identity = {
            "training_contract_sha256": training_hash,
            "input_checkpoint_sha256": audit.sha256(current),
            "episode_sha256": [audit.sha256(path) for path in paths],
            "history_sha256": [audit.sha256(path) for path in history],
        }
        record = read(record_path)
        if record is None:
            checkpoint = directory / f"round_{index:02d}.npz"
            metrics = strict.update_physical_model(
                spec, current, checkpoint, envelopes,
                [read(path) for path in history], None)
            change = strict._policy_parameter_change(
                spec, current, checkpoint, reference_round=index - 1)
            record = {
                **identity, "round": index, "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": audit.sha256(checkpoint),
                "metrics": metrics, "parameter_change": change,
            }
            write(root, record_path, record)
        elif (any(record.get(key) != value
                  for key, value in identity.items()) or
              audit.sha256(record["checkpoint"]) !=
              record["checkpoint_sha256"]):
            raise RuntimeError("fine-tune round resume identity differs")
        current = Path(record["checkpoint"])
        history.extend(paths)
        rounds.append(record)
    report = {
        "status": "complete", "training_contract_sha256": training_hash,
        "policy_id": spec["policy_id"], "checkpoint": str(current),
        "checkpoint_sha256": audit.sha256(current),
        "expert_coefficient_sha256": bundle_sha256(
            spec["expert_coefficient_bundle"]),
        "balanced_rounds": len(rounds), "episodes": len(history),
        "selection": "latest cumulative checkpoint",
        "regression_validation": False, "rollback": False,
        "rounds": rounds,
    }
    write(root, report_path, report)
    return report


def fine_tune_all(root, bundle, webots, workers):
    document = training_contract(root, bundle)
    reports = [_fine_tune_policy(root, spec, document, webots, workers)
               for spec in policy_specs(bundle)]
    completion = {
        "status": "complete", "training_contract_sha256": digest(document),
        "policies": reports,
    }
    write_once(root, Path(root) / "fine_tune_complete.json", completion)
    return completion


def episode_metrics(envelope):
    result = envelope["result"]
    formal, summary = result["formal_evaluation"], result["summary_metrics"]
    values = {
        "completed_tasks":
            formal["count_evaluation"]["total_tasks_completed"],
        "utility_score_deadline_v2":
            formal["utility_v2_evaluation"]["utility_score_deadline_v2"],
        "physical_collision_count": summary["physical_collision_count"],
        "fallback_scheduler_commits":
            summary.get("fallback_scheduler_commits", 0),
        "invalid_scheduler_outputs":
            summary.get("invalid_scheduler_outputs", 0),
        "scheduler_timeout_count": summary.get("scheduler_timeout_count", 0),
    }
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(float(value)) for value in values.values()):
        raise RuntimeError("physical evaluation metric is invalid")
    return values


def _evaluation_policies(root, bundle, training_document):
    result = []
    training_hash = digest(training_document)
    for spec in policy_specs(bundle):
        report = read(Path(root) / "fine_tune" / spec["policy_id"] / "report.json")
        if (not report or report.get("training_contract_sha256") != training_hash or
                audit.sha256(report["checkpoint"]) != report["checkpoint_sha256"]):
            raise RuntimeError("fine-tuned model is missing")
        result.append((spec, Path(report["checkpoint"]), spec["scheduler"]))
    for objective in OBJECTIVES:
        descriptor = expert_descriptor(objective, bundle)
        path = Path(root) / "evaluation" / "descriptors" / f"{objective}.json"
        if path.exists() and read(path) != descriptor:
            raise RuntimeError("optimized Hungarian descriptor changed")
        if not path.exists():
            write(root, path, descriptor)
        spec = {
            "policy_id": f"TrafficHungarian_{objective}",
            "scheduler": "TrafficHungarian", "objective": objective,
            "expert_coefficient_bundle": bundle,
        }
        result.append((spec, path, "TrafficHungarian"))
    return result


def evaluate_policy(root, spec, checkpoint, scheduler, webots, workers):
    report_path = Path(root) / "evaluation" / spec["policy_id"] / "report.json"
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [(scenario, seed, pool.submit(
            _physical_episode, root, spec, "evaluation", checkpoint,
            scenario, seed, webots, scheduler))
            for scenario in SCENARIOS for seed in EVALUATION_SEEDS]
        for scenario, seed, future in jobs:
            path, envelope = future.result()
            rows.append({
                "scenario": scenario, "seed": seed,
                "manifest_sha256": envelope["manifest_sha256"],
                "result": str(path), "result_sha256": audit.sha256(path),
                **episode_metrics(envelope),
            })
    expected = {(scenario, seed) for scenario in SCENARIOS
                for seed in EVALUATION_SEEDS}
    if {(row["scenario"], row["seed"]) for row in rows} != expected:
        raise RuntimeError("evaluation matrix is incomplete")
    report = {
        "status": "complete", "policy_id": spec["policy_id"],
        "scheduler": scheduler, "objective": spec["objective"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": audit.sha256(checkpoint),
        "expert_coefficient_sha256": bundle_sha256(
            spec["expert_coefficient_bundle"]),
        "evaluation_learning": False, "evaluation_fallback": False,
        "episode_count": len(rows), "episodes": rows,
        "mean_completed_tasks": statistics.fmean(
            row["completed_tasks"] for row in rows),
        "mean_utility_score_deadline_v2": statistics.fmean(
            row["utility_score_deadline_v2"] for row in rows),
    }
    existing = read(report_path)
    if existing is not None and existing != report:
        raise RuntimeError("evaluation report changed")
    if existing is None:
        write(root, report_path, report)
    return report


def evaluation_summary(root, reports, bundle):
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
    aggregates = []
    for report in reports:
        aggregates.append({
            "policy_id": report["policy_id"],
            "scheduler": report["scheduler"],
            "objective": report["objective"],
            "episodes": report["episode_count"],
            "mean_completed_tasks": report["mean_completed_tasks"],
            "mean_utility_score_deadline_v2":
                report["mean_utility_score_deadline_v2"],
            "total_physical_collisions": sum(
                row["physical_collision_count"] for row in report["episodes"]),
            "total_fallback_commits": sum(
                row["fallback_scheduler_commits"] for row in report["episodes"]),
        })
    document = {
        "status": "complete", "bundle_sha256": bundle_sha256(bundle),
        "seeds": list(EVALUATION_SEEDS), "scenarios": list(SCENARIOS),
        "episode_count": len(rows), "expected_episode_count": 108,
        "paired_manifest_verified": True,
        "evaluation_learning": False, "evaluation_fallback": False,
        "aggregates": aggregates, "episodes": rows,
    }
    if document["episode_count"] != document["expected_episode_count"]:
        raise RuntimeError("full three-algorithm evaluation is incomplete")
    write_once(root, Path(root) / "evaluation_summary.json", document)
    return document


def evaluate_all(root, bundle, webots, workers):
    document = training_contract(root, bundle)
    reports = [evaluate_policy(
        root, spec, checkpoint, scheduler, webots, workers)
        for spec, checkpoint, scheduler in _evaluation_policies(
            root, bundle, document)]
    summary = evaluation_summary(root, reports, bundle)
    write_once(root, Path(root) / "workflow_complete.json", {
        "status": "complete", "completed_at": audit.now(),
        "bundle_sha256": bundle_sha256(bundle),
        "evaluation_summary": str(Path(root) / "evaluation_summary.json"),
        "evaluation_summary_sha256": audit.sha256(
            Path(root) / "evaluation_summary.json"),
    })
    return summary


def _review_files(root, step):
    root = Path(root)
    if step == 6:
        return [
            root / "coefficients/selected_bundle.json",
            *(root / "coefficient_search" / objective / "report.json"
              for objective in OBJECTIVES),
            *(root / "coefficient_search" / objective / "optimizer_state.json"
              for objective in OBJECTIVES),
        ]
    if step == 7:
        reports = [
            root / "standalone" / spec["policy_id"] / "report.json"
            for spec in policy_specs(selected_coefficients(root))]
        return [
            root / "standalone_complete.json",
            *reports,
            *(Path(read(path)["checkpoint"]) for path in reports),
            *(strict.model_identity_path(Path(read(path)["checkpoint"]))
              for path in reports),
        ]
    if step == 8:
        reports = [
            root / "fine_tune" / spec["policy_id"] / "report.json"
            for spec in policy_specs(selected_coefficients(root))]
        return [
            root / "fine_tune_complete.json",
            *reports,
            *(Path(read(path)["checkpoint"]) for path in reports),
            *(strict.model_identity_path(Path(read(path)["checkpoint"]))
              for path in reports),
        ]
    if step == 9:
        return [
            root / "evaluation_summary.json", root / "workflow_complete.json",
            *(root / "evaluation" / policy_id / "report.json"
              for policy_id in (
                  "GraphImitation_count", "GraphImitation_utility_v2",
                  "LearnedHungarian_count", "LearnedHungarian_utility_v2",
                  "TrafficHungarian_count", "TrafficHungarian_utility_v2")),
        ]
    raise ValueError("runtime review step must be 6--9")


def validate_runtime_artifacts(root, step):
    bundle = selected_coefficients(root)
    if step == 6:
        for objective in OBJECTIVES:
            report = read(Path(root) / "coefficient_search" / objective / "report.json")
            if (report.get("candidate_count") !=
                    1 + SEARCH_GENERATIONS * SEARCH_POPULATION[objective] or
                    report.get("selected", {}).get("episode_count") != 9):
                raise RuntimeError("coefficient search coverage is incomplete")
    elif step == 7:
        document = training_contract(root, bundle)
        for spec in policy_specs(bundle):
            report = _standalone_report(root, spec, digest(document))
            if (report["balanced_rounds"] != len(TRAINING_SEEDS) or
                    report["training_episodes"] != 60):
                raise RuntimeError("student standalone coverage is incomplete")
    elif step == 8:
        document = training_contract(root, bundle)
        for spec in policy_specs(bundle):
            report = read(Path(root) / "fine_tune" / spec["policy_id"] / "report.json")
            if (report.get("training_contract_sha256") != digest(document) or
                    report.get("balanced_rounds") != len(FINE_TUNE_SEEDS) or
                    report.get("episodes") != 30):
                raise RuntimeError("Webots fine-tune coverage is incomplete")
            strict.load_model(spec, Path(report["checkpoint"]), 0)
    elif step == 9:
        summary = read(Path(root) / "evaluation_summary.json")
        if (summary.get("episode_count") != 108 or
                summary.get("seeds") != list(EVALUATION_SEEDS) or
                not summary.get("paired_manifest_verified") or
                any(row["fallback_scheduler_commits"] != 0
                    for row in summary["episodes"])):
            raise RuntimeError("frozen evaluation evidence is incomplete")
    else:
        raise ValueError("runtime review step must be 6--9")
    return _review_files(root, step)


RUNTIME_REVIEW_PERSPECTIVES = {
    6: (
        ("optimization_math", "candidate fitness and seed clustering verified"),
        ("seed_and_resume_identity", "candidate hashes and nine paired episodes verified"),
        ("coefficient_safety", "nonnegative bounded rows and fixed travel anchor verified"),
    ),
    7: (
        ("training_coverage", "four policies have twenty balanced rounds"),
        ("model_provenance", "all checkpoints bind the selected coefficient hash"),
        ("no_regression_selection", "latest checkpoints selected without rollback gate"),
    ),
    8: (
        ("physical_coverage", "all four policies have thirty fine-tune episodes"),
        ("supervision_alignment", "Webots labels bind the selected coefficient bundle"),
        ("handoff_readiness", "latest checkpoints load and evaluation seeds remain unused"),
    ),
    9: (
        ("evaluation_matrix", "six seeds by three scenarios by six policies verified"),
        ("pure_policy_safety", "fallback-free frozen evaluation evidence verified"),
        ("paired_reporting", "shared manifests and aggregate identities verified"),
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
    bundle = selected_coefficients(root)
    # No command boundary exists between these calls. If the process is
    # interrupted, episode-level resume continues from the last valid result.
    fine_tune_all(root, bundle, webots, workers)
    for _ in range(3):
        review_runtime_step(root, 8)
    audit.require_reviews(root, 8)
    summary = evaluate_all(root, bundle, webots, workers)
    for _ in range(3):
        review_runtime_step(root, 9)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=(
            "init", "preflight", "search", "standalone", "physical",
            "review-runtime", "status"))
    parser.add_argument("--run-root", type=Path, default=audit.DEFAULT_RUN)
    parser.add_argument("--webots", type=Path, default=DEFAULT_WEBOTS)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--step", type=int, choices=(6, 7, 8, 9))
    args = parser.parse_args()
    if args.command == "init":
        print(json.dumps(audit.initialize(args.run_root), ensure_ascii=False))
        return 0
    root = audit.validate_root(args.run_root)
    if args.command == "status":
        print(json.dumps(read(root / "workflow.json"),
                         ensure_ascii=False, indent=2))
        return 0
    with run_lock(root):
        if args.command == "preflight":
            audit.require_reviews(root, 5)
            result = preflight(root, args.webots)
        elif args.command == "search":
            audit.require_reviews(root, 5)
            result = coefficient_search(root)
        elif args.command == "standalone":
            audit.require_reviews(root, 6)
            result = standalone_training(root, args.workers)
        elif args.command == "physical":
            audit.require_reviews(root, 7)
            physical.validate_webots(args.webots)
            result = physical_then_evaluate(root, args.webots, args.workers)
        else:
            if args.step is None:
                raise ValueError("--step is required for runtime review")
            result = review_runtime_step(root, args.step)
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
