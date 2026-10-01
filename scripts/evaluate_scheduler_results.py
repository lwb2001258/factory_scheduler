#!/usr/bin/env python3
"""Paired, seed-controlled deployment evaluation for scheduler results."""

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


EVALUATION_VERSION = "scheduler-online-evaluation-v2"


def _finite(value, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metric {field} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"metric {field} is not finite")
    return result


def _run_metrics(document: dict) -> dict:
    summary = document.get("summary_metrics") or {}
    completions = document.get("task_completions") or []
    durations = [
        _finite(item["completion_duration"], "completion_duration")
        for item in completions
        if item.get("completion_duration") is not None]
    high_priority = [item for item in completions
                     if _finite(item.get("priority", 1), "priority") >= 3]
    high_waits = [
        _finite(item["waiting_time"], "waiting_time")
        for item in high_priority if item.get("waiting_time") is not None]
    return {
        "throughput": _finite(
            summary.get("throughput_per_minute"), "throughput_per_minute"),
        "completion_mean": _finite(
            summary.get("avg_task_completion_time"),
            "avg_task_completion_time"),
        "completion_p95": float(np.percentile(durations, 95))
        if durations else 0.0,
        "waiting_mean": _finite(
            summary.get("avg_waiting_time"), "avg_waiting_time"),
        "high_priority_wait_mean": float(np.mean(high_waits))
        if high_waits else None,
        "high_priority_completed": len(high_priority),
        "safety_events": int(_finite(
            summary.get("safety_event_count", 0), "safety_event_count")),
        "distance_violations": int(_finite(
            summary.get("pair_distance_violations", 0),
            "pair_distance_violations")),
        "deadlocks": int(_finite(
            summary.get("total_deadlocks", 0), "total_deadlocks")),
        "invalid_outputs": int(_finite(
            summary.get("invalid_scheduler_outputs", 0),
            "invalid_scheduler_outputs")),
        "fallback_commits": int(_finite(
            summary.get("fallback_scheduler_commits", 0),
            "fallback_scheduler_commits")),
        "latency_p95_ms": _finite(
            summary.get("scheduling_latency_p95_ms", 0),
            "scheduling_latency_p95_ms"),
    }


def load_runs(pattern: str, *, scenario: str = "",
              required_runtime_mode: str = "") -> Dict[Tuple[str, int], dict]:
    """Load one unambiguous result for every scheduler/seed pair."""
    runs = {}
    for name in sorted(set(glob.glob(pattern, recursive=True))):
        path = Path(name)
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read result file: {path}") from exc
        info = document.get("experiment_info") or {}
        if scenario and info.get("scenario") != scenario:
            continue
        if (required_runtime_mode and
                info.get("runtime_mode") != required_runtime_mode):
            continue
        scheduler = info.get("scheduler")
        seed = info.get("seed")
        if not scheduler or seed is None:
            continue
        key = str(scheduler), int(seed)
        if key in runs:
            raise ValueError(
                f"duplicate scheduler/seed result: {key} in {path}")
        runs[key] = {
            "path": str(path),
            "scenario": info.get("scenario"),
            "metrics": _run_metrics(document),
        }
    return runs


def _relative_delta(candidate: float, baseline: float):
    if abs(baseline) < 1e-9:
        return 0.0 if abs(candidate) < 1e-9 else None
    return (candidate - baseline) / abs(baseline)


def _bootstrap_mean_ci(values: Iterable[float], seed: int = 90210,
                       repetitions: int = 10_000) -> list:
    vector = np.asarray(list(values), dtype=np.float64)
    if vector.size == 0:
        return [None, None]
    if vector.size == 1:
        return [float(vector[0]), float(vector[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, vector.size, size=(repetitions, vector.size))
    samples = np.mean(vector[indices], axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def evaluate_candidate(runs: Dict[Tuple[str, int], dict], baseline: str,
                       candidate: str, *, min_paired_seeds: int = 5,
                       throughput_tolerance: float = 0.02,
                       completion_tolerance: float = 0.05,
                       max_latency_p95_ms: float = 20.0,
                       latency_relative_tolerance: float = 0.20) -> dict:
    baseline_seeds = {seed for scheduler, seed in runs if scheduler == baseline}
    candidate_seeds = {seed for scheduler, seed in runs if scheduler == candidate}
    paired_seeds = sorted(baseline_seeds & candidate_seeds)
    rows = []
    for seed in paired_seeds:
        baseline_metrics = runs[(baseline, seed)]["metrics"]
        candidate_metrics = runs[(candidate, seed)]["metrics"]
        rows.append({
            "seed": seed,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "relative_delta": {
                key: _relative_delta(candidate_metrics[key],
                                     baseline_metrics[key])
                for key in ("throughput", "completion_mean",
                            "completion_p95", "waiting_mean")
            },
        })
    relative_summary = {}
    for metric in ("throughput", "completion_mean", "completion_p95",
                   "waiting_mean"):
        values = [row["relative_delta"][metric] for row in rows]
        finite_values = [value for value in values
                         if value is not None and math.isfinite(value)]
        relative_summary[metric] = {
            "mean": float(np.mean(finite_values)) if finite_values else None,
            "median": float(np.median(finite_values)) if finite_values else None,
            "bootstrap_mean_ci95": _bootstrap_mean_ci(finite_values),
            "candidate_win_rate": float(np.mean([
                value > 0 if metric == "throughput" else value < 0
                for value in finite_values])) if finite_values else None,
        }
    baseline_safety = sum(row["baseline"]["safety_events"] for row in rows)
    candidate_safety = sum(row["candidate"]["safety_events"] for row in rows)
    baseline_distance = sum(
        row["baseline"]["distance_violations"] for row in rows)
    candidate_distance = sum(
        row["candidate"]["distance_violations"] for row in rows)
    invalid_outputs = sum(row["candidate"]["invalid_outputs"] for row in rows)
    fallback_commits = sum(row["candidate"]["fallback_commits"] for row in rows)
    if rows:
        baseline_latency = max(
            row["baseline"]["latency_p95_ms"] for row in rows)
        candidate_latency = max(
            row["candidate"]["latency_p95_ms"] for row in rows)
        allowed_latency = max(
            max_latency_p95_ms,
            baseline_latency * (1.0 + latency_relative_tolerance))
        latency_within_budget = candidate_latency <= allowed_latency
    else:
        baseline_latency = None
        candidate_latency = None
        allowed_latency = None
        latency_within_budget = False
    throughput_delta = relative_summary["throughput"]["mean"]
    completion_delta = relative_summary["completion_mean"]["mean"]
    completion_p95_delta = relative_summary["completion_p95"]["mean"]
    waiting_delta = relative_summary["waiting_mean"]["mean"]
    gates = {
        "complete_seed_coverage": (
            len(paired_seeds) >= min_paired_seeds and
            baseline_seeds == candidate_seeds),
        "throughput_noninferior": (
            throughput_delta is not None and
            throughput_delta >= -throughput_tolerance),
        "mean_completion_noninferior": (
            completion_delta is not None and
            completion_delta <= completion_tolerance),
        "p95_completion_noninferior": (
            completion_p95_delta is not None and
            completion_p95_delta <= completion_tolerance),
        "safety_events_noninferior": candidate_safety <= baseline_safety,
        "distance_violations_noninferior": (
            candidate_distance <= baseline_distance),
        "no_invalid_outputs": invalid_outputs == 0,
        "no_fallback_commits": fallback_commits == 0,
        "latency_within_budget": latency_within_budget,
    }
    passed = all(gates.values())
    advantage = bool(passed and (
        throughput_delta >= 0.02 or completion_delta <= -0.05 or
        waiting_delta <= -0.10))
    return {
        "candidate": candidate,
        "status": ("outperforms_baseline" if advantage else
                   "validated_candidate" if passed else "rejected"),
        "paired_seeds": paired_seeds,
        "baseline_seed_set": sorted(baseline_seeds),
        "candidate_seed_set": sorted(candidate_seeds),
        "relative_metrics": relative_summary,
        "aggregate_safety": {
            "baseline_safety_events": baseline_safety,
            "candidate_safety_events": candidate_safety,
            "baseline_distance_violations": baseline_distance,
            "candidate_distance_violations": candidate_distance,
            "candidate_invalid_outputs": invalid_outputs,
            "candidate_fallback_commits": fallback_commits,
            "baseline_max_latency_p95_ms": baseline_latency,
            "candidate_max_latency_p95_ms": candidate_latency,
            "allowed_max_latency_p95_ms": allowed_latency,
        },
        "gates": gates,
        "per_seed": rows,
    }


def evaluate_results(pattern: str, baseline: str, candidates: List[str],
                     *, scenario: str = "", min_paired_seeds: int = 5,
                     throughput_tolerance: float = 0.02,
                     completion_tolerance: float = 0.05,
                     max_latency_p95_ms: float = 20.0,
                     latency_relative_tolerance: float = 0.20,
                     required_runtime_mode: str = "") -> dict:
    if (min_paired_seeds < 2 or throughput_tolerance < 0 or
            completion_tolerance < 0 or max_latency_p95_ms <= 0 or
            latency_relative_tolerance < 0):
        raise ValueError("evaluation thresholds are out of range")
    runs = load_runs(
        pattern, scenario=scenario,
        required_runtime_mode=required_runtime_mode)
    evaluations = {
        candidate: evaluate_candidate(
            runs, baseline, candidate,
            min_paired_seeds=min_paired_seeds,
            throughput_tolerance=throughput_tolerance,
            completion_tolerance=completion_tolerance,
            max_latency_p95_ms=max_latency_p95_ms,
            latency_relative_tolerance=latency_relative_tolerance)
        for candidate in candidates
    }
    outperformers = [
        name for name, result in evaluations.items()
        if result["status"] == "outperforms_baseline"]
    if outperformers:
        recommended = max(
            outperformers,
            key=lambda name: evaluations[name]["relative_metrics"]
            ["throughput"]["mean"])
        deployment_status = "candidate_selected"
    else:
        recommended = baseline
        deployment_status = "baseline_retained"
    return {
        "evaluation_version": EVALUATION_VERSION,
        "scenario": scenario or None,
        "baseline": baseline,
        "candidates": evaluations,
        "deployment_decision": {
            "status": deployment_status,
            "active_scheduler": recommended,
            "baseline": baseline,
            "scenario": scenario or None,
            "required_runtime_mode": required_runtime_mode or None,
            "eligible_outperformers": outperformers,
            "reason": ("at least one candidate passed every gate and met an "
                       "advantage threshold" if outperformers else
                       "no candidate passed every gate with a measured advantage"),
        },
        "thresholds": {
            "min_paired_seeds": min_paired_seeds,
            "throughput_noninferiority_tolerance": throughput_tolerance,
            "completion_noninferiority_tolerance": completion_tolerance,
            "max_latency_p95_ms": max_latency_p95_ms,
            "latency_relative_tolerance": latency_relative_tolerance,
            "required_runtime_mode": required_runtime_mode or None,
        },
    }


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired evaluation of scheduler result JSON files")
    parser.add_argument("--results-glob", required=True)
    parser.add_argument("--baseline", default="Hungarian")
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--scenario", default="")
    parser.add_argument("--min-paired-seeds", type=int, default=5)
    parser.add_argument("--throughput-tolerance", type=float, default=0.02)
    parser.add_argument("--completion-tolerance", type=float, default=0.05)
    parser.add_argument("--max-latency-p95-ms", type=float, default=20.0)
    parser.add_argument("--latency-relative-tolerance", type=float, default=0.20)
    parser.add_argument("--required-runtime-mode", default="webots")
    parser.add_argument("--output", type=Path,
                        default=Path("results/ml_scheduler/online_evaluation.json"))
    parser.add_argument("--deployment-manifest", type=Path)
    parser.add_argument("--fail-on-rejection", action="store_true")
    args = parser.parse_args()
    report = evaluate_results(
        args.results_glob, args.baseline, args.candidate,
        scenario=args.scenario, min_paired_seeds=args.min_paired_seeds,
        throughput_tolerance=args.throughput_tolerance,
        completion_tolerance=args.completion_tolerance,
        max_latency_p95_ms=args.max_latency_p95_ms,
        latency_relative_tolerance=args.latency_relative_tolerance,
        required_runtime_mode=args.required_runtime_mode)
    _write_json_atomic(args.output, report)
    if args.deployment_manifest is not None:
        _write_json_atomic(
            args.deployment_manifest, report["deployment_decision"])
    statuses = {name: result["status"]
                for name, result in report["candidates"].items()}
    print(json.dumps({"output": str(args.output), "statuses": statuses},
                     indent=2, sort_keys=True))
    rejected = any(status == "rejected" for status in statuses.values())
    return 2 if args.fail_on_rejection and rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
