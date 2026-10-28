"""Fresh traffic Ridge training, Webots refinement and paired Hungarian evaluation.

Each phase is resumable and gated by three documented reviews. Final seeds are
never training inputs. No pre/post degradation gate or historical-result reuse.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "controllers/factory_supervisor")]

import numpy as np
import run_strict_1800_workflow as strict
import run_webots_1800_workflow as physical
from run_graph_traffic_workflow import (
    digest, confined, write, read, manifest, progress, fresh_path,
    checked_completion, train_episode,
)
from graph_expert_scheduler import expert_descriptor
from learning_scheduler import TRAFFIC_PAIR_FEATURE_VERSION, TRAFFIC_RIDGE_MODEL_VERSION
from schedulers import create_scheduler
from traffic_features import TRAFFIC_VERSION
from webots_finetune_guard import WEBOTS_FINETUNE_SEEDS, FINAL_EVALUATION_SEEDS

DEFAULT_ROOT = ROOT / "results/learned_hungarian_traffic_v1_20261027"
ORIGINAL = ROOT / "results/all_scheduler_model_remediation/formal_1800_dual_v6"
WEBOTS = Path("C:/Program Files/Webots/msys64/mingw64/bin/webots.exe")
OBJECTIVES = ("count", "utility_v2")
SCENARIOS = ("A", "B", "C")
BOOTSTRAP_SEEDS = tuple(range(11000, 11005))
TRAINING_SEEDS = tuple(range(21000, 21200))
VALIDATION_SEEDS = (31000, 31001)
VERSION = "learned-traffic-retraining-paired-webots-v1"


def specs():
    result = []
    for objective in OBJECTIVES:
        spec = dict(next(s for s in strict.AI_SPECS
                         if s["policy_id"] == f"LearnedHungarian_{objective}"))
        previous = read(ORIGINAL / "headless" / spec["policy_id"] / "training_report.json")
        spec.update(training_parameters=previous["training_parameters"], traffic_version=TRAFFIC_VERSION)
        strict.training_parameters(spec)
        result.append(spec)
    return result


def validate_root(root):
    root = Path(root).resolve()
    if root.parent != (ROOT / "results").resolve():
        raise ValueError("experiment must be a direct child of results")
    protocol = read(root / "protocol.json") or {}
    expected = {"version": "learned-hungarian-traffic-upgrade-v1",
                "evaluation_seeds": list(FINAL_EVALUATION_SEEDS),
                "objectives": list(OBJECTIVES), "scenarios": list(SCENARIOS),
                "duration_seconds": 1800, "regression_validation": False,
                "minimum_review_rounds_per_step": 3, "preserve_existing_artifacts": True}
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise RuntimeError("missing or incompatible experiment protocol")
    for name in ("protected_hashes.json", "source_before_hashes.json"):
        if not read(root / "audit" / name):
            raise RuntimeError("preservation inventory is missing")
    return root


def require_reviews(root, through):
    sources = {}
    for step in range(1, through + 1):
        receipt = read(Path(root) / "audit" / f"step_{step:02d}_reviews.json") or {}
        rounds = receipt.get("reviews", [])
        if (len(rounds) < 3 or [r.get("round") for r in rounds[-3:]] != [1, 2, 3] or
                any(r.get("status") != "pass" or not r.get("checks") for r in rounds[-3:])):
            raise RuntimeError(f"step {step} requires three detailed passing reviews")
        sources.update(receipt.get("source_sha256", {}))
    for name, expected in sources.items():
        if strict.sha256_file(ROOT / name) != expected:
            raise RuntimeError(f"reviewed source changed: {name}")


@contextmanager
def run_lock(root):
    root = validate_root(root)
    with (root / "workflow.lock").open("a+b") as stream:
        if stream.tell() == 0:
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


def source_hashes():
    paths = {p for directory in ("controllers", "scripts", "tests")
             for p in (ROOT / directory).rglob("*.py") if "__pycache__" not in p.parts}
    paths.update((ROOT / "worlds").glob("*.wbt"))
    paths.update((ROOT / "protos").rglob("*.proto"))
    paths.add(ROOT / "config/formal_1800/calibration.json")
    return {p.relative_to(ROOT).as_posix(): strict.sha256_file(p) for p in sorted(paths)}


def initialize(root, workers=2):
    contract = {
        "version": VERSION, "source_sha256": source_hashes(),
        "protocol_sha256": strict.sha256_file(root / "protocol.json"),
        "traffic_version": TRAFFIC_VERSION, "feature_version": TRAFFIC_PAIR_FEATURE_VERSION,
        "model_version": TRAFFIC_RIDGE_MODEL_VERSION, "scenarios": list(SCENARIOS),
        "objectives": list(OBJECTIVES), "duration_seconds": 1800,
        "parameters": {s["objective"]: strict.training_parameters(s) for s in specs()},
        "initialization": "fresh weights and samples; reuse previously selected l2 only",
        "bootstrap_seeds": list(BOOTSTRAP_SEEDS), "training_seeds": list(TRAINING_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "fine_tune_seeds": list(WEBOTS_FINETUNE_SEEDS),
        "evaluation_seeds": list(FINAL_EVALUATION_SEEDS),
        "minimum_training_rounds": 20, "maximum_training_rounds": 200,
        "validation_interval": 5, "convergence_patience": 3,
        "parameter_change_threshold": .01, "maximum_fine_tune_rounds": 10,
        "regression_validation": False, "final_model": "last Webots-fine-tuned checkpoint",
        "baseline": "TrafficHungarian: exact shared traffic expert, zero trained parameters",
        "evaluation_order": "all learned episodes, then all nonlearning Hungarian episodes",
        "maximum_parallel_episodes": workers,
    }
    previous = read(root / "run_contract.json")
    if previous is not None and previous != contract:
        raise RuntimeError("source/protocol changed; refusing stale resume")
    if previous is None:
        write(root, root / "run_contract.json", contract)
    return digest(contract)


def validate_standalone(root, spec, checkpoint, index, contract_hash):
    """Frozen validation for model selection; does not run degradation baselines."""
    rows = []
    before = strict.sha256_file(checkpoint)
    for scenario in SCENARIOS:
        for seed in VALIDATION_SEEDS:
            task_manifest = manifest(root, "standalone_validation", scenario, seed)
            path = root / "standalone" / spec["objective"] / "validation" / str(index) / scenario / str(seed) / "result.json"
            identity = {"contract_sha256": contract_hash, "checkpoint_sha256": before,
                        "manifest_sha256": strict.sha256_file(task_manifest)}
            row = read(path)
            if row is not None:
                if any(row.get(k) != v for k, v in identity.items()):
                    raise RuntimeError("validation resume identity mismatch")
            else:
                progress(root, spec, "standalone_validation", round=index, scenario=scenario, seed=seed)
                strict.load_model(spec, checkpoint, seed)
                env, robots, tasks, context, limit, _profile = strict._episode_environment(
                    strict._load_manifest(task_manifest), scenario, seed, spec["objective"])
                context = replace(context, configuration={**context.configuration, "graph_traffic_enabled": True})
                env.reset(robots, tasks, context, seed=seed)
                scheduler = create_scheduler("LearnedHungarian", model_path=checkpoint, allow_safe_fallback=False)
                terminated = False
                for _ in range(limit):
                    _, _, terminated, truncated, info = env.step_scheduler(scheduler)
                    diagnostics = info.get("scheduler_diagnostics", {})
                    if (info.get("invalid_action") or info.get("assignment_rejected") or
                            info.get("invalid_truncation") or diagnostics.get("fallback") or
                            (truncated and not terminated)):
                        raise RuntimeError("invalid frozen standalone validation transition")
                    if terminated:
                        break
                if not terminated or strict.sha256_file(checkpoint) != before:
                    raise RuntimeError("validation horizon/model integrity failed")
                runtime = strict._validate_runtime(env, strict._load_manifest(task_manifest))
                utility = env.formal_utility_evaluation(strict.utility_score_config())
                row = {**identity, "scenario": scenario, "seed": seed,
                       "completed_tasks": runtime["tasks_completed"],
                       "utility_score_deadline_v2": utility["utility_score_deadline_v2"]}
                write(root, path, row)
            rows.append(row)
    metric = "completed_tasks" if spec["objective"] == "count" else "utility_score_deadline_v2"
    return {"round": index, "metric": metric, "mean": statistics.fmean(r[metric] for r in rows),
            "checkpoint": str(checkpoint), "checkpoint_sha256": before, "episodes": rows}


def standalone_worker(root, spec, contract_hash):
    root = Path(root)
    report_path = root / "standalone" / spec["objective"] / "training_report.json"
    completed = checked_completion(root, report_path, contract_hash)
    if completed:
        return completed
    current, history = Path(spec["source"]), []
    for seed in BOOTSTRAP_SEEDS:
        for scenario in SCENARIOS:
            current, samples, _ = train_episode(root, spec, current, history,
                "standalone_tuning_training", scenario, seed, scenario == "C", contract_hash)
            history.append(samples)
    best = validate_standalone(root, spec, current, 0, contract_hash)
    validations, previous, converged = [best], current, False
    for index, seed in enumerate(TRAINING_SEEDS, 1):
        for scenario in SCENARIOS:
            current, samples, _ = train_episode(root, spec, current, history,
                "standalone_training", scenario, seed, index % 5 == 0 and scenario == "C", contract_hash)
            history.append(samples)
        if index % 5:
            continue
        result = validate_standalone(root, spec, current, index, contract_hash)
        change = strict._policy_parameter_change(spec, previous, current, reference_round=index - 5)
        result["relative_parameter_change"] = change["relative_l2_change"]
        previous = current
        validations.append(result)
        if result["mean"] > best["mean"]:
            best = result
        recent = validations[-3:]
        converged = bool(index >= 20 and
            max(r["mean"] for r in recent) - max(r["mean"] for r in validations[:-3]) < .5 and
            all(r.get("relative_parameter_change", math.inf) <= .01 for r in recent))
        write(root, report_path.with_name("validation_history.json"), validations)
        progress(root, spec, "standalone_checkpoint_validation", round=index,
                 validation_mean=result["mean"], best_mean=best["mean"], converged=converged)
        if converged:
            break
    final = root / "standalone" / spec["objective"] / "model.npz"
    if final.exists() and strict.sha256_file(final) != best["checkpoint_sha256"]:
        raise RuntimeError("selected standalone model already differs")
    if not final.exists():
        strict.atomic_copy_model(spec, Path(best["checkpoint"]), final)
    report = {"status": "complete", "contract_sha256": contract_hash,
              "checkpoint": str(final), "checkpoint_sha256": strict.sha256_file(final),
              "balanced_rounds": index, "training_episodes": len(history),
              "selected_round": best["round"], "parameter_converged": converged,
              "completion_reason": "converged" if converged else "maximum_rounds",
              "validation_history": validations, "regression_validation": False}
    write(root, report_path, report)
    return report


def physical_episode(root, spec, phase, checkpoint, scenario, seed, webots):
    if phase not in {"fine_tune", "evaluation", "baseline"}:
        raise ValueError("unknown physical phase")
    fine = phase == "fine_tune"
    allowed = WEBOTS_FINETUNE_SEEDS if fine else FINAL_EVALUATION_SEEDS
    if seed not in allowed or scenario not in SCENARIOS:
        raise ValueError("seed/scenario does not belong to this physical phase")
    scheduler = "TrafficHungarian" if phase == "baseline" else "LearnedHungarian"
    task_manifest = manifest(root, "webots_finetune" if fine else "final_evaluation", scenario, seed)
    path = root / phase / spec["objective"] / scenario / str(seed) / "result.json"
    before = strict.sha256_file(checkpoint)
    existing = read(path)
    valid_args = dict(scenario=scenario, seed=seed, manifest=task_manifest,
                      input_model_sha256=before, expected_scheduler=scheduler,
                      fine_tune_objective=spec["objective"] if fine else None,
                      fine_tune_algorithm=scheduler if fine else None)
    if existing is not None and not physical.valid_episode(existing, **valid_args):
        raise RuntimeError("existing physical result differs; refusing overwrite")
    progress(root, spec, phase, scenario=scenario, seed=seed, scheduler=scheduler)
    # Preserve partial raw simulations too; a retry always gets a fresh folder.
    attempt = (path if existing is not None else
               fresh_path(root, path.parent / "attempts", "run") / "result.json")
    envelope, _ = physical.run_or_resume_episode(
        canonical_path=attempt, webots=webots, scenario=scenario, seed=seed,
        manifest=task_manifest, scheduler=scheduler, model_path=checkpoint,
        fine_tune_objective=spec["objective"] if fine else None,
        fine_tune_algorithm=scheduler if fine else None)
    if strict.sha256_file(checkpoint) != before:
        raise RuntimeError("physical episode modified its input checkpoint")
    if not physical.valid_pure_policy_result(envelope):
        raise RuntimeError("physical result contains non-native scheduler actions")
    if fine:
        strict._learned_hungarian_shared_samples([envelope], spec["objective"], traffic=True)
    if phase == "baseline":
        metadata = envelope["result"]["formal_evaluation"].get("graph_expert", {})
        if metadata.get("parameter_updates") != 0 or metadata.get("learning_enabled") is not False:
            raise RuntimeError("baseline lacks nonlearning evidence")
    if existing is None:
        write(root, path, envelope)
    return path, envelope


def fine_tune(root, spec, contract_hash, webots, workers=2):
    report_path = root / "fine_tune" / spec["objective"] / "fine_tune_report.json"
    completed = checked_completion(root, report_path, contract_hash)
    if completed:
        return completed
    trained = checked_completion(root, root / "standalone" / spec["objective"] / "training_report.json", contract_hash)
    if trained is None:
        raise RuntimeError("standalone training incomplete")
    current, history, rounds = Path(trained["checkpoint"]), [], []
    for index, seed in enumerate(WEBOTS_FINETUNE_SEEDS, 1):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(physical_episode, root, spec, "fine_tune", current,
                                   scenario, seed, webots) for scenario in SCENARIOS]
            paths = [future.result()[0] for future in futures]
        directory = root / "fine_tune" / spec["objective"] / "checkpoints"
        record_path = directory / f"round_{index:02d}.json"
        identity = {"contract_sha256": contract_hash, "input_sha256": strict.sha256_file(current),
                    "episodes_sha256": [strict.sha256_file(p) for p in paths],
                    "history_sha256": [strict.sha256_file(p) for p in history]}
        record = read(record_path)
        if record is None:
            checkpoint = fresh_path(root, directory, f"round_{index:02d}.npz")
            envelopes = [read(p) for p in paths]
            metrics = strict.update_physical_model(spec, current, checkpoint, envelopes,
                                                   [read(p) for p in history], None)
            change = strict._policy_parameter_change(spec, current, checkpoint, reference_round=index - 1)
            record = {**identity, "round": index, "checkpoint": str(checkpoint),
                      "checkpoint_sha256": strict.sha256_file(checkpoint), "metrics": metrics,
                      "parameter_change": change,
                      "objective_metric": strict._webots_round_objective_metric(envelopes, spec)}
            write(root, record_path, record)
        elif (any(record.get(k) != v for k, v in identity.items()) or
              strict.sha256_file(confined(root, record["checkpoint"])) != record["checkpoint_sha256"]):
            raise RuntimeError("fine-tune round resume mismatch")
        current = confined(root, record["checkpoint"])
        history.extend(paths)
        rounds.append(record)
        convergence = strict._webots_parameter_convergence(rounds)
        progress(root, spec, "fine_tune_round_complete", round=index, convergence=convergence)
        if convergence["converged"]:
            break
    report = {"status": "complete", "contract_sha256": contract_hash,
              "checkpoint": str(current), "checkpoint_sha256": strict.sha256_file(current),
              "balanced_rounds": len(rounds), "episodes": len(history), "rounds": rounds,
              "parameter_convergence": convergence, "regression_validation": False}
    write(root, report_path, report)
    return report


def episode_metrics(envelope):
    result = envelope["result"]
    formal, summary = result["formal_evaluation"], result["summary_metrics"]
    values = {"completed_tasks": formal["count_evaluation"]["total_tasks_completed"],
              "utility": formal["utility_v2_evaluation"]["utility_score_deadline_v2"],
              "collisions": summary["physical_collision_count"],
              "latency_p95_ms": summary["scheduling_latency_p95_ms"],
              "planned_wait_seconds": summary["planned_wait_total_seconds"],
              "predicted_conflicts": summary["predicted_conflicts"]}
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in values.values()):
        raise RuntimeError("missing/nonfinite comparison metric")
    return values


def evaluate(root, spec, contract_hash, webots, baseline=False, workers=2):
    phase = "baseline" if baseline else "evaluation"
    if baseline:
        learned = read(root / "evaluate_complete.json") or {}
        if learned.get("contract_sha256") != contract_hash or learned.get("status") != "complete":
            raise RuntimeError("all learned evaluations must finish before baseline")
        for objective in OBJECTIVES:
            report = checked_completion(root, root / "evaluation" / objective / "report.json", contract_hash)
            if report is None or report.get("episode_count") != 18:
                raise RuntimeError("all learned evaluations must finish before baseline")
        checkpoint = root / "baseline" / "descriptors" / f"{spec['objective']}.json"
        descriptor = expert_descriptor(spec["objective"])
        if checkpoint.exists() and read(checkpoint) != descriptor:
            raise RuntimeError("baseline descriptor changed")
        if not checkpoint.exists():
            write(root, checkpoint, descriptor)
    else:
        tuned = checked_completion(root, root / "fine_tune" / spec["objective"] / "fine_tune_report.json", contract_hash)
        if not tuned:
            raise RuntimeError("fine-tuning incomplete")
        checkpoint = Path(tuned["checkpoint"])
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [(scenario, seed, pool.submit(physical_episode, root, spec, phase,
                 checkpoint, scenario, seed, webots))
                for scenario in SCENARIOS for seed in FINAL_EVALUATION_SEEDS]
        for scenario, seed, future in jobs:
            path, envelope = future.result()
            info = envelope["result"]["experiment_info"]
            rows.append({"scenario": scenario, "seed": seed, "objective": spec["objective"],
                         "manifest_sha256": info["task_manifest_sha256"],
                         "result": str(path), "result_sha256": strict.sha256_file(path),
                         **episode_metrics(envelope)})
    report = {"status": "complete", "contract_sha256": contract_hash,
              "checkpoint": str(checkpoint), "checkpoint_sha256": strict.sha256_file(checkpoint),
              "episodes": rows, "episode_count": len(rows)}
    write(root, root / phase / spec["objective"] / "report.json", report)
    return report


def comparison(root, contract_hash):
    paired = []
    for spec in specs():
        objective = spec["objective"]
        reports = [checked_completion(root, root / phase / objective / "report.json", contract_hash)
                   for phase in ("evaluation", "baseline")]
        expected = {(scenario, seed) for scenario in SCENARIOS for seed in FINAL_EVALUATION_SEEDS}
        if any(r is None or len(r["episodes"]) != len(expected) for r in reports):
            raise RuntimeError("paired comparison is incomplete")
        maps = [{(r["scenario"], r["seed"]): r for r in report["episodes"]} for report in reports]
        if any(set(mapping) != expected for mapping in maps):
            raise RuntimeError("duplicate/missing comparison seeds")
        for key in sorted(expected):
            learned, baseline = (mapping[key] for mapping in maps)
            if learned["manifest_sha256"] != baseline["manifest_sha256"]:
                raise RuntimeError("paired task streams differ")
            for row in (learned, baseline):
                if strict.sha256_file(confined(root, row["result"])) != row["result_sha256"]:
                    raise RuntimeError("comparison result changed")
                actual = episode_metrics(read(row["result"]))
                if any(row[name] != value for name, value in actual.items()):
                    raise RuntimeError("reported metrics differ from physical result")
            metrics = episode_metrics(read(learned["result"]))
            paired.append({"objective": objective, "scenario": key[0], "seed": key[1],
                "manifest_sha256": learned["manifest_sha256"],
                **{f"learned_{m}": learned[m] for m in metrics},
                **{f"hungarian_{m}": baseline[m] for m in metrics},
                **{f"delta_{m}": learned[m] - baseline[m] for m in metrics}})
    aggregates = []
    for objective in OBJECTIVES:
        for scenario in (*SCENARIOS, "all"):
            rows = [r for r in paired if r["objective"] == objective and
                    (scenario == "all" or r["scenario"] == scenario)]
            means = {name: statistics.fmean(r[name] for r in rows)
                     for name in rows[0] if name.startswith(("learned_", "hungarian_", "delta_"))}
            aggregates.append({"objective": objective, "scenario": scenario, "pairs": len(rows), **means})
    report = {"status": "complete", "contract_sha256": contract_hash, "paired_episodes": len(paired),
              "evaluation_seeds": list(FINAL_EVALUATION_SEEDS), "regression_validation": False,
              "difference_direction": "LearnedHungarian minus nonlearning traffic Hungarian",
              "aggregates": aggregates, "pairs": paired}
    write(root, root / "comparison.json", report)
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase", required=True, choices=("check", "standalone", "fine-tune", "evaluate", "baseline", "report"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--webots", type=Path, default=WEBOTS)
    args = parser.parse_args()
    with run_lock(args.output_dir) as root:
        gate = {"check": 5, "standalone": 5, "fine-tune": 6, "evaluate": 7, "baseline": 8, "report": 9}[args.phase]
        require_reviews(root, gate)
        contract_hash = initialize(root, args.workers)
        if args.phase == "check":
            for partition, seeds in (("standalone_tuning_training", BOOTSTRAP_SEEDS),
                ("standalone_training", TRAINING_SEEDS), ("standalone_validation", VALIDATION_SEEDS),
                ("webots_finetune", WEBOTS_FINETUNE_SEEDS), ("final_evaluation", FINAL_EVALUATION_SEEDS)):
                for scenario in SCENARIOS:
                    for seed in seeds:
                        manifest(root, partition, scenario, seed)
            physical.validate_webots(args.webots)
            print(json.dumps({"status": "preflight_passed", "contract_sha256": contract_hash}), flush=True)
            return 0
        if args.phase == "standalone":
            if args.workers == 1:
                reports = [standalone_worker(root, spec, contract_hash) for spec in specs()]
            else:
                with ProcessPoolExecutor(max_workers=min(args.workers, len(OBJECTIVES))) as pool:
                    futures = [pool.submit(standalone_worker, root, spec, contract_hash) for spec in specs()]
                    reports = [future.result() for future in futures]
        elif args.phase == "report":
            comparison(root, contract_hash)
            return 0
        else:
            physical.validate_webots(args.webots)
            reports = [(fine_tune(root, spec, contract_hash, args.webots, args.workers) if args.phase == "fine-tune"
                        else evaluate(root, spec, contract_hash, args.webots,
                                      baseline=args.phase == "baseline", workers=args.workers))
                       for spec in specs()]
        write(root, root / f"{args.phase}_complete.json", {
            "status": "complete", "contract_sha256": contract_hash, "policies": reports})
        print(json.dumps({"phase": args.phase, "status": "complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
