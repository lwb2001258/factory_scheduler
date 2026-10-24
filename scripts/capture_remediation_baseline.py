#!/usr/bin/env python3
"""Capture the immutable pre-remediation evaluation and file-hash baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


PROTECTED_CONTROLLER_FILES = (
    "cbs_planner.py",
    "collision_safety.py",
    "grid_planner.py",
    "joint_grid_planner.py",
    "joint_plan_transaction.py",
    "motion_coordinator.py",
    "safety_coordination.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def load_episode(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_policy(paths: list[Path]) -> dict[str, Any]:
    utility_scores: list[float] = []
    completed_tasks: list[float] = []
    throughput: list[float] = []
    totals = {
        "native_scheduler_commits": 0,
        "fallback_scheduler_commits": 0,
        "invalid_scheduler_outputs": 0,
        "scheduler_timeout_count": 0,
    }
    model_records: dict[str, dict[str, str]] = {}
    scenarios: set[str] = set()
    seeds: set[int] = set()

    for path in paths:
        payload = load_episode(path)
        result = payload.get("result", {})
        formal = result.get("formal_evaluation", {})
        count_eval = formal.get("count_evaluation", {})
        utility_eval = formal.get("utility_v2_evaluation", {})
        summary = result.get("summary_metrics", {})
        experiment = result.get("experiment_info", {})

        if utility_eval.get("utility_score_deadline_v2") is not None:
            utility_scores.append(float(utility_eval["utility_score_deadline_v2"]))
        if count_eval.get("total_tasks_completed") is not None:
            completed_tasks.append(float(count_eval["total_tasks_completed"]))
        if count_eval.get("throughput_per_minute") is not None:
            throughput.append(float(count_eval["throughput_per_minute"]))
        for key in totals:
            totals[key] += int(summary.get(key, 0) or 0)

        scenario = experiment.get("scenario") or experiment.get("scenario_name")
        seed = experiment.get("random_seed") or experiment.get("seed")
        if scenario is not None:
            scenarios.add(str(scenario))
        if seed is not None:
            seeds.add(int(seed))

        model_path_value = payload.get("input_model_path")
        if model_path_value:
            model_path = Path(model_path_value)
            record = {
                "reported_sha256": str(payload.get("input_model_sha256", "")),
                "actual_sha256": sha256_file(model_path) if model_path.is_file() else "missing",
            }
            model_records[str(model_path.resolve())] = record

    episode_count = len(paths)
    pure_reasons = [key for key in totals if totals[key] != 0 and key != "native_scheduler_commits"]
    return {
        "episode_count": episode_count,
        "scenarios": sorted(scenarios),
        "seeds": sorted(seeds),
        "mean_completed_tasks": mean(completed_tasks),
        "mean_throughput_per_minute": mean(throughput),
        "mean_utility_score_deadline_v2": mean(utility_scores),
        **totals,
        "pure_policy_eligible_under_new_gate": not pure_reasons and episode_count > 0,
        "pure_policy_exclusion_reasons": pure_reasons,
        "input_models": model_records,
    }


def protected_hashes(repo: Path) -> dict[str, str]:
    candidates = [repo / ".gitignore"]
    controller_root = repo / "controllers" / "factory_supervisor"
    candidates.extend(controller_root / name for name in PROTECTED_CONTROLLER_FILES)
    world_root = repo / "worlds"
    if world_root.is_dir():
        candidates.extend(path for path in world_root.rglob("*") if path.is_file())
    return {
        path.relative_to(repo).as_posix(): sha256_file(path)
        for path in sorted(candidates)
        if path.is_file()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, default=Path("results/strict_1800_retrain"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/all_scheduler_model_remediation/baseline.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    source = args.source if args.source.is_absolute() else repo / args.source
    output = args.output if args.output.is_absolute() else repo / args.output
    episode_root = source / "evaluation" / "episodes"
    if not episode_root.is_dir():
        raise FileNotFoundError(f"evaluation directory not found: {episode_root}")

    policies: dict[str, Any] = {}
    for policy_dir in sorted(path for path in episode_root.iterdir() if path.is_dir()):
        paths = sorted(policy_dir.glob("*/*/result.json"))
        policies[policy_dir.name] = aggregate_policy(paths)

    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": str(source.resolve()),
        "source_workflow_state_sha256": sha256_file(source / "workflow_state.json"),
        "policy_count": len(policies),
        "policies": policies,
        "protected_file_sha256": protected_hashes(repo),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
