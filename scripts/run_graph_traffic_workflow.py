"""Isolated, resumable GraphImitation traffic retraining and Webots evaluation.

Existing checkpoints/results are read-only inputs. No production model is
replaced. Each phase requires the preceding three-pass review receipt.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
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
for directory in (ROOT / "scripts", ROOT / "controllers/factory_supervisor"):
    sys.path.insert(0, str(directory))

import numpy as np
import run_strict_1800_workflow as strict
import run_webots_1800_workflow as physical
from traffic_features import TRAFFIC_VERSION
from webots_finetune_guard import (
    FINAL_EVALUATION_SEEDS, WEBOTS_FINETUNE_SEEDS, WEBOTS_REGRESSION_SEEDS,
    webots_finetune_regression_decision)

DEFAULT_RUN = ROOT / "results/graph_imitation_traffic_v1_20261025"
ORIGINAL_RUN = ROOT / "results/all_scheduler_model_remediation/formal_1800_dual_v6"
SCENARIOS = ("A", "B", "C")
OBJECTIVES = ("count", "utility_v2")
BOOTSTRAP_SEEDS = tuple(range(11000, 11005))
TRAINING_SEEDS = tuple(range(21000, 21200))
VALIDATION_SEEDS = (31000, 31001)
CONTRACT_VERSION = "graph-traffic-targeted-retraining-v1"
SOURCE_FILES = (
    "scripts/run_graph_traffic_workflow.py", "scripts/run_strict_1800_workflow.py",
    "scripts/run_webots_1800_workflow.py", "scripts/run_experiments.py",
    "controllers/factory_supervisor/traffic_features.py",
    "controllers/factory_supervisor/learning_scheduler.py",
    "controllers/factory_supervisor/headless_training_runtime.py",
    "controllers/factory_supervisor/factory_supervisor.py",
    "controllers/factory_supervisor/training_scenarios.py",
    "controllers/factory_supervisor/motion_coordinator.py",
    "controllers/factory_supervisor/config.py",
    "controllers/robot_controller/robot_controller.py",
    "config/formal_1800/calibration.json",
)
SEED_PARTITIONS = {
    "standalone_tuning_training": BOOTSTRAP_SEEDS, "standalone_training": TRAINING_SEEDS,
    "standalone_validation": VALIDATION_SEEDS,
    "webots_finetune": WEBOTS_FINETUNE_SEEDS,
    "webots_regression": WEBOTS_REGRESSION_SEEDS,
    "final_evaluation": FINAL_EVALUATION_SEEDS,
}


def digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def confined(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"output escapes the isolated run: {path}")
    return path


def write(root, path, value):
    strict.atomic_json(confined(root, path), value)


def read(path):
    path = Path(path)
    if not path.exists():
        return None
    # A damaged saved record must not be treated as a missing run and replaced.
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_root(root):
    root = Path(root).resolve()
    if root == ORIGINAL_RUN.resolve() or root.is_relative_to(ORIGINAL_RUN.resolve()):
        raise RuntimeError("original run is read-only")
    marker = read(root / "protocol.json")
    if not isinstance(marker, dict) or marker.get("version") != "graph-traffic-upgrade-v1":
        raise RuntimeError("isolated run must have its preservation inventory first")
    for name in ("original_artifacts.jsonl", "protected_graph_hashes.json"):
        if not (root / "audit" / name).is_file():
            raise RuntimeError("preservation inventory is missing")
    return root


def require_reviews(root, through_step):
    latest_files = {}
    for step in range(1, through_step + 1):
        receipt = read(Path(root) / "audit" / f"step_{step:02d}_reviews.json")
        rounds = receipt.get("reviews", []) if isinstance(receipt, dict) else []
        if len(rounds) != 3 or [row.get("round") for row in rounds] != [1, 2, 3] or any(
                row.get("status") != "pass" for row in rounds):
            raise RuntimeError(f"step {step} has not passed all three review rounds")
        latest_files.update(receipt.get("files", {}))
    for name, expected in latest_files.items():
        if strict.sha256_file(ROOT / name) != expected:
            raise RuntimeError(f"reviewed source changed: {name}")


@contextmanager
def run_lock(root):
    root = validate_root(root)
    path = confined(root, Path(root) / "workflow.lock")
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
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def specs():
    result = []
    for objective in OBJECTIVES:
        spec = dict(next(item for item in strict.AI_SPECS
                         if item["policy_id"] == f"GraphImitation_{objective}"))
        report = ORIGINAL_RUN / "headless" / spec["policy_id"] / "training_report.json"
        spec["training_parameters"] = read(report)["training_parameters"]
        spec["traffic_version"] = TRAFFIC_VERSION
        strict.training_parameters(spec)
        result.append(spec)
    return result


def initialize(root):
    root = validate_root(root)
    parameters = {spec["objective"]: spec["training_parameters"] for spec in specs()}
    contract = {
        "version": CONTRACT_VERSION, "traffic_version": TRAFFIC_VERSION,
        "source_sha256": {name: strict.sha256_file(ROOT / name) for name in sorted(
            set(SOURCE_FILES) | {str(path.relative_to(ROOT)).replace("\\", "/")
                for path in (ROOT / "controllers/factory_supervisor").glob("*.py")})},
        "objectives": list(OBJECTIVES), "scenarios": list(SCENARIOS),
        "duration_seconds": 1800, "training_parameters": parameters,
        "hyperparameters": "reuse_previous_selected_parameters; fresh features/weights/samples",
        "bootstrap_seeds": list(BOOTSTRAP_SEEDS), "training_seeds": list(TRAINING_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "webots_finetune_seeds": list(WEBOTS_FINETUNE_SEEDS),
        "webots_regression_seeds": list(WEBOTS_REGRESSION_SEEDS),
        "final_evaluation_seeds": list(FINAL_EVALUATION_SEEDS),
        "standalone_minimum_rounds": 20, "standalone_maximum_rounds": 200,
        "standalone_validation_interval": 5, "parameter_change_threshold": 0.01,
        "convergence_patience": 3, "maximum_physical_rounds": 10,
        "webots_supervision": "new-version own training-partition trajectories only",
        "final_evaluation_is_never_training_data": True,
    }
    destination = root / "run_contract.json"
    previous = read(destination)
    if previous is not None and previous != contract:
        raise RuntimeError("run contract/source changed; refusing to reuse existing artifacts")
    if previous is None:
        write(root, destination, contract)
    return contract


def manifest(root, partition, scenario, seed):
    if scenario not in SCENARIOS or seed not in SEED_PARTITIONS.get(partition, ()):
        raise ValueError("seed/scenario is not registered for this partition")
    source = strict.manifest_path(partition, scenario, seed)
    if not source.is_file():
        raise FileNotFoundError(f"registered manifest is missing: {source}")
    strict._load_manifest(source)
    destination = confined(root, Path(root) / "manifests" / partition / scenario / f"{seed}.json")
    if destination.exists():
        if strict.sha256_file(destination) != strict.sha256_file(source):
            raise RuntimeError("copied manifest identity changed")
    else:
        strict.atomic_copy(source, destination)
    return destination


def progress(root, spec, phase, **details):
    value = {"phase": phase, "policy_id": spec["policy_id"],
             "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **details}
    write(root, Path(root) / "status" / f"{spec['objective']}.json", value)
    print(json.dumps(value), flush=True)


def fresh_path(root, directory, name):
    directory = confined(root, directory)
    path = directory / name
    attempt = 0
    while path.exists():
        attempt += 1
        original = Path(name)
        path = directory / f"{original.stem}.attempt{attempt}{original.suffix}"
    return path


def checked_completion(root, path, contract_hash):
    report = read(path)
    if report is None:
        return None
    if report.get("contract_sha256") != contract_hash or report.get("status") != "complete":
        raise RuntimeError(f"incompatible completed phase: {path}")
    checkpoint = confined(root, report["checkpoint"])
    if strict.sha256_file(checkpoint) != report["checkpoint_sha256"]:
        raise RuntimeError("completed checkpoint changed")
    return report


def train_episode(root, spec, current, history, partition, scenario, seed,
                  fit, contract_hash):
    if partition not in {"standalone_tuning_training", "standalone_training"}:
        raise ValueError("validation/evaluation data cannot enter training")
    unit = Path(root) / "standalone" / spec["objective"] / partition / str(seed) / scenario
    record_path = unit / "result.json"
    task_manifest = manifest(root, partition, scenario, seed)
    input_hash = (strict.sha256_file(current) if current != Path(spec["source"])
                  else "fresh-expert-bootstrap")
    identity = {"contract_sha256": contract_hash, "input_checkpoint_sha256": input_hash,
                "manifest_sha256": strict.sha256_file(task_manifest),
                "fit_model": bool(fit), "prior_samples_sha256": digest([
                    strict.sha256_file(path) for path in history])}
    record = read(record_path)
    if record is not None:
        if any(record.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"training resume identity mismatch: {unit}")
        checkpoint, samples = confined(root, record["checkpoint"]), confined(root, record["samples"])
        if strict.sha256_file(checkpoint) != record["checkpoint_sha256"] or strict.sha256_file(samples) != record["samples_sha256"]:
            raise RuntimeError("training artifact changed")
        strict.load_model(spec, checkpoint, seed)
        return checkpoint, samples, record
    checkpoint = fresh_path(root, unit, "model.npz")
    samples = fresh_path(root, unit, "samples.npz")
    progress(root, spec, "standalone_training", partition=partition, scenario=scenario, seed=seed)
    started = time.perf_counter()
    metrics = strict.run_headless_episode(
        spec, current, checkpoint, samples, strict._load_manifest(task_manifest),
        scenario, seed, history, fit_supervised=fit, training_seed_partition=partition)
    record = {**identity, "checkpoint": str(checkpoint), "samples": str(samples),
              "checkpoint_sha256": strict.sha256_file(checkpoint),
              "samples_sha256": strict.sha256_file(samples), "metrics": metrics,
              "wall_seconds": time.perf_counter() - started}
    write(root, record_path, record)
    return checkpoint, samples, record


def validate_standalone(root, spec, checkpoint, round_index, contract_hash):
    directory = Path(root) / "standalone" / spec["objective"] / "validation" / f"round_{round_index:03d}"
    checkpoint_hash = strict.sha256_file(checkpoint)
    rows = []
    for scenario in SCENARIOS:
        for seed in VALIDATION_SEEDS:
            path = directory / scenario / str(seed) / "result.json"
            task_manifest = manifest(root, "standalone_validation", scenario, seed)
            data = strict._load_manifest(task_manifest)
            record = read(path)
            if record is not None:
                if record.get("contract_sha256") != contract_hash or not strict._valid_standalone_validation_episode(
                        record, spec, checkpoint_hash,
                        strict.sha256_file(strict.model_identity_path(checkpoint)), data, scenario, seed):
                    raise RuntimeError("frozen validation resume identity mismatch")
            else:
                progress(root, spec, "standalone_validation", round=round_index, scenario=scenario, seed=seed)
                references = strict._same_seed_reference_results(
                    Path(root), "standalone_validation", spec["objective"], data, scenario, seed)
                record = strict._run_standalone_validation_episode(
                    spec, checkpoint, data, scenario, seed, same_seed_references=references)
                record["contract_sha256"] = contract_hash
                write(root, path, record)
            if record["invalid_scheduler_outputs"]:
                raise RuntimeError("invalid scheduler output in standalone validation")
            rows.append(record)
    metric = "completed_tasks" if spec["objective"] == "count" else "utility_score_deadline_v2"
    gate = strict._supervised_baseline_gate(rows, spec)
    result = {"round": round_index, "metric": metric,
              "mean": statistics.fmean(row[metric] for row in rows),
              "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
              "eligible": gate["eligible"], "baseline_gate": gate,
              "episodes": rows, "contract_sha256": contract_hash}
    write(root, directory / "report.json", result)
    return result


def standalone_worker(root, spec, contract_hash):
    root = Path(root)
    report_path = root / "standalone" / spec["objective"] / "training_report.json"
    completed = checked_completion(root, report_path, contract_hash)
    if completed is not None:
        return completed
    current = Path(spec["source"])
    history = []
    for seed in BOOTSTRAP_SEEDS:
        for scenario in SCENARIOS:
            current, samples, _ = train_episode(root, spec, current, history,
                "standalone_tuning_training", scenario, seed, scenario == "C", contract_hash)
            history.append(samples)
    validation = validate_standalone(root, spec, current, 0, contract_hash)
    best = validation if validation["eligible"] else None
    validations = [validation]
    previous = current
    converged = False
    for round_index, seed in enumerate(TRAINING_SEEDS, 1):
        for scenario in SCENARIOS:
            current, samples, _ = train_episode(root, spec, current, history,
                "standalone_training", scenario, seed,
                round_index % 5 == 0 and scenario == "C", contract_hash)
            history.append(samples)
        if round_index % 5:
            continue
        validation = validate_standalone(root, spec, current, round_index, contract_hash)
        change = strict._policy_parameter_change(spec, previous, current, reference_round=round_index-5)
        validation["relative_parameter_change"] = change["relative_l2_change"]
        validations.append(validation)
        previous = current
        if validation["eligible"] and (best is None or validation["mean"] > best["mean"]):
            best = validation
        recent = validations[-3:]
        converged = bool(round_index >= 20 and best is not None and
            max(row["mean"] for row in recent) - max(row["mean"] for row in validations[:-3]) < 0.5 and
            all(row.get("relative_parameter_change", math.inf) <= 0.01 for row in recent))
        progress(root, spec, "standalone_checkpoint_validation", round=round_index,
                 validation_mean=validation["mean"], best_mean=best["mean"] if best else None,
                 relative_parameter_change=change["relative_l2_change"], converged=converged)
        if converged:
            break
    if best is None:
        raise RuntimeError("no traffic checkpoint passed the same-seed baseline gate")
    final = confined(root, root / "standalone" / spec["objective"] / "model.npz")
    if final.exists() and strict.sha256_file(final) != best["checkpoint_sha256"]:
        raise RuntimeError("refusing to replace an existing selected standalone model")
    if not final.exists():
        strict.atomic_copy_model(spec, Path(best["checkpoint"]), final)
    report = {"status": "complete", "contract_sha256": contract_hash,
              "checkpoint": str(final), "checkpoint_sha256": strict.sha256_file(final),
              "balanced_rounds": round_index, "bootstrap_episodes": 15,
              "training_episodes": round_index * 3, "parameter_converged": converged,
              "completion_reason": "converged" if converged else "maximum_rounds",
              "selected_round": best["round"], "selected_validation_mean": best["mean"],
              "validation_history": validations}
    write(root, report_path, report)
    progress(root, spec, "standalone_complete", rounds=round_index, checkpoint=str(final))
    return report


def physical_episode(root, spec, phase, checkpoint, scenario, seed, fine=False):
    if fine != (phase == "fine_tune"):
        raise ValueError("fine-tune recording is restricted to its training phase")
    partition = {"fine_tune": "webots_finetune", "regression_pre": "webots_regression",
                 "regression_post": "webots_regression", "evaluation": "final_evaluation"}[phase]
    path = confined(root, Path(root) / phase / spec["objective"] / scenario / str(seed) / "result.json")
    task_manifest = manifest(root, partition, scenario, seed)
    progress(root, spec, phase, scenario=scenario, seed=seed, checkpoint=str(checkpoint))
    before = strict.sha256_file(checkpoint)
    existing = read(path)
    if existing is not None and not physical.valid_episode(
            existing, scenario=scenario, seed=seed, manifest=task_manifest,
            input_model_sha256=before, fine_tune_objective=spec["objective"] if fine else None,
            fine_tune_algorithm="GraphImitation" if fine else None,
            expected_scheduler="GraphImitation"):
        raise RuntimeError("existing Webots result does not match this episode; refusing overwrite")
    envelope, _ = physical.run_or_resume_episode(
        canonical_path=path, webots=Path(os.environ["GRAPH_TRAFFIC_WEBOTS"]),
        scenario=scenario, seed=seed, manifest=task_manifest,
        scheduler="GraphImitation", model_path=checkpoint,
        fine_tune_objective=spec["objective"] if fine else None,
        fine_tune_algorithm="GraphImitation" if fine else None)
    if strict.sha256_file(checkpoint) != before:
        raise RuntimeError("physical episode changed its frozen input model")
    if fine:
        strict._graph_imitation_shared_samples([envelope], spec["objective"], traffic=True)
    return path, envelope


def fine_tune(root, spec, contract_hash):
    root = Path(root)
    report_path = root / "fine_tune" / spec["objective"] / "fine_tune_report.json"
    completed = checked_completion(root, report_path, contract_hash)
    if completed:
        return completed
    trained = checked_completion(root, root / "standalone" / spec["objective"] / "training_report.json", contract_hash)
    if trained is None:
        raise RuntimeError("standalone training is incomplete")
    current = Path(trained["checkpoint"])
    history, rounds = [], []
    for round_index, seed in enumerate(WEBOTS_FINETUNE_SEEDS, 1):
        paths = []
        for scenario in SCENARIOS:
            path, _ = physical_episode(root, spec, "fine_tune", current, scenario, seed, fine=True)
            paths.append(path)
        directory = root / "fine_tune" / spec["objective"] / "checkpoints"
        record_path = directory / f"round_{round_index:02d}.json"
        identity = {"input_checkpoint_sha256": strict.sha256_file(current),
                    "episode_sha256": [strict.sha256_file(path) for path in paths],
                    "contract_sha256": contract_hash}
        record = read(record_path)
        if record is None:
            checkpoint = fresh_path(root, directory, f"round_{round_index:02d}.npz")
            envelopes = [read(path) for path in paths]
            metrics = strict.update_physical_model(
                spec, current, checkpoint, envelopes, [read(path) for path in history], None)
            change = strict._policy_parameter_change(spec, current, checkpoint, reference_round=round_index-1)
            record = {**identity, "round": round_index, "seed": seed,
                      "checkpoint": str(checkpoint), "checkpoint_sha256": strict.sha256_file(checkpoint),
                      "metrics": metrics, "parameter_change": change,
                      "fit_learning_rate": strict.training_parameters(spec)["learning_rate"] * 0.1,
                      "objective_metric": strict._webots_round_objective_metric(envelopes, spec)}
            write(root, record_path, record)
        elif any(record.get(key) != value for key, value in identity.items()) or strict.sha256_file(confined(root, record["checkpoint"])) != record["checkpoint_sha256"]:
            raise RuntimeError("physical round resume identity mismatch")
        current = confined(root, record["checkpoint"])
        history.extend(paths)
        rounds.append(record)
        convergence = strict._webots_parameter_convergence(rounds)
        if convergence["converged"]:
            break
    report = {"status": "complete", "contract_sha256": contract_hash,
              "checkpoint": str(current), "checkpoint_sha256": strict.sha256_file(current),
              "balanced_rounds": len(rounds), "episodes": len(history),
              "parameter_convergence": convergence, "rounds": rounds}
    write(root, report_path, report)
    return report


def regression(root, spec, contract_hash):
    root = Path(root)
    selected_path = root / "selection" / spec["objective"] / "report.json"
    completed = checked_completion(root, selected_path, contract_hash)
    if completed:
        return completed
    before = checked_completion(root, root / "standalone" / spec["objective"] / "training_report.json", contract_hash)
    after = checked_completion(root, root / "fine_tune" / spec["objective"] / "fine_tune_report.json", contract_hash)
    if before is None or after is None:
        raise RuntimeError("both pre/post checkpoints are required for regression")
    groups = []
    for phase, report in (("regression_pre", before), ("regression_post", after)):
        checkpoint = Path(report["checkpoint"])
        rows = []
        for scenario in SCENARIOS:
            for seed in WEBOTS_REGRESSION_SEEDS:
                _, envelope = physical_episode(root, spec, phase, checkpoint, scenario, seed)
                rows.append(strict._webots_regression_row(envelope, spec, strict.sha256_file(checkpoint), scenario, seed))
        groups.append(rows)
    decision = webots_finetune_regression_decision(
        spec["objective"], Path(before["checkpoint"]), Path(after["checkpoint"]), *groups,
        policy_id=spec["policy_id"], scheduler="GraphImitation")
    report = {"status": "complete", "contract_sha256": contract_hash,
              "checkpoint": decision["selected_checkpoint"],
              "checkpoint_sha256": decision["selected_checkpoint_sha256"],
              "decision": decision}
    write(root, selected_path, report)
    return report


def evaluate(root, spec, contract_hash):
    root = Path(root)
    selected = checked_completion(root, root / "selection" / spec["objective"] / "report.json", contract_hash)
    if selected is None:
        raise RuntimeError("physical regression selection is incomplete")
    checkpoint = Path(selected["checkpoint"])
    episodes = []
    comparisons = []
    for scenario in SCENARIOS:
        for seed in FINAL_EVALUATION_SEEDS:
            path, envelope = physical_episode(root, spec, "evaluation", checkpoint, scenario, seed)
            formal = envelope["result"]["formal_evaluation"]
            summary = envelope["result"]["summary_metrics"]
            episodes.append({"scenario": scenario, "seed": seed, "result": str(path),
                "task_manifest_sha256": strict._load_manifest(Path(envelope["manifest_path"]))["task_manifest_sha256"],
                "completed_tasks": formal["count_evaluation"]["total_tasks_completed"],
                "utility_score_deadline_v2": formal["utility_v2_evaluation"]["utility_score_deadline_v2"],
                "fallback_scheduler_commits": summary.get("fallback_scheduler_commits"),
                "scheduler_timeout_count": summary.get("scheduler_timeout_count"),
                "invalid_scheduler_outputs": summary.get("invalid_scheduler_outputs")})
            original_path = (ORIGINAL_RUN / "evaluation/episodes" / spec["policy_id"] /
                             scenario / str(seed) / "result.json")
            original = read(original_path)
            if original is not None:
                original_result = original["result"]
                info = original_result["experiment_info"]
                if (info.get("scenario") != scenario or info.get("seed") != seed or
                        info.get("sim_duration") != 1800 or
                        info.get("task_manifest_sha256") != episodes[-1]["task_manifest_sha256"]):
                    raise RuntimeError("original evaluation is not a matching seed/task/horizon pair")
                original_formal = original_result["formal_evaluation"]
                old_count = original_formal["count_evaluation"]["total_tasks_completed"]
                old_utility = original_formal["utility_v2_evaluation"]["utility_score_deadline_v2"]
                comparisons.append({"scenario": scenario, "seed": seed,
                    "original_result": str(original_path),
                    "original_result_sha256": strict.sha256_file(original_path),
                    "original_checkpoint_sha256": original["input_model_sha256"],
                    "original_completed_tasks": old_count, "original_utility": old_utility,
                    "completed_tasks_difference": episodes[-1]["completed_tasks"] - old_count,
                    "utility_difference": episodes[-1]["utility_score_deadline_v2"] - old_utility})
    report = {"status": "complete", "contract_sha256": contract_hash,
              "checkpoint": str(checkpoint), "checkpoint_sha256": strict.sha256_file(checkpoint),
              "scenarios": list(SCENARIOS), "seeds": list(FINAL_EVALUATION_SEEDS),
              "episodes": episodes, "episode_count": len(episodes),
              "mean_completed_tasks": statistics.fmean(row["completed_tasks"] for row in episodes),
              "mean_utility": statistics.fmean(row["utility_score_deadline_v2"] for row in episodes),
              "original_result_comparison": {
                  "kind": "paired historical results; original runs were not rerun",
                  "pairs": comparisons, "paired_episode_count": len(comparisons),
                  "mean_completed_tasks_difference": statistics.fmean(
                      row["completed_tasks_difference"] for row in comparisons) if comparisons else None,
                  "mean_utility_difference": statistics.fmean(
                      row["utility_difference"] for row in comparisons) if comparisons else None}}
    write(root, root / "evaluation" / spec["objective"] / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--phase", choices=("check", "standalone", "fine-tune", "regression", "evaluate"), required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--webots", type=Path, default=Path(r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    with run_lock(root):
        contract = initialize(root)
        contract_hash = digest(contract)
        if args.phase == "check":
            for partition, seeds in (("standalone_tuning_training", BOOTSTRAP_SEEDS),
                ("standalone_training", TRAINING_SEEDS), ("standalone_validation", VALIDATION_SEEDS),
                ("webots_finetune", WEBOTS_FINETUNE_SEEDS), ("webots_regression", WEBOTS_REGRESSION_SEEDS),
                ("final_evaluation", FINAL_EVALUATION_SEEDS)):
                for scenario in SCENARIOS:
                    for seed in seeds:
                        manifest(root, partition, scenario, seed)
            physical.validate_webots(args.webots)
            print(json.dumps({"status": "preflight_passed", "contract_sha256": contract_hash}), flush=True)
            return 0
        gate = {"standalone": 4, "fine-tune": 5, "regression": 6, "evaluate": 7}[args.phase]
        require_reviews(root, gate)
        if args.phase == "standalone":
            if args.workers == 1:
                reports = [standalone_worker(root, spec, contract_hash) for spec in specs()]
            else:
                with ProcessPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(standalone_worker, root, spec, contract_hash) for spec in specs()]
                    reports = [future.result() for future in futures]
        else:
            physical.validate_webots(args.webots)
            os.environ["GRAPH_TRAFFIC_WEBOTS"] = str(args.webots.resolve())
            action = {"fine-tune": fine_tune, "regression": regression, "evaluate": evaluate}[args.phase]
            reports = [action(root, spec, contract_hash) for spec in specs()]
        write(root, root / f"{args.phase}_complete.json", {
            "status": "complete", "contract_sha256": contract_hash, "policies": reports})
        print(json.dumps({"phase": args.phase, "status": "complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
