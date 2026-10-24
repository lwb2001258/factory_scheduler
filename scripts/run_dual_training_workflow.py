"""Execute the fail-closed Count / Utility V2 evaluation workflow."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "controllers" / "factory_supervisor"
if str(SUPERVISOR) not in sys.path:
    sys.path.insert(0, str(SUPERVISOR))

from dual_objective import (  # noqa: E402
    CountRewardProfile, UtilityScoreConfig, UtilityV2RewardProfile,
    count_evaluation, paired_count_delta, utility_v2_evaluation,
)
from headless_training_runtime import HeadlessRuntimeConfig  # noqa: E402
from rl_environment import RLEnvironmentConfig, SchedulingEnvironment  # noqa: E402
from schedulers import create_scheduler  # noqa: E402
from task_generator import (  # noqa: E402
    generate_task_manifest, task_manifest_sha256, write_task_manifest,
)
from training_scenarios import (  # noqa: E402
    factory_scenario, formal_scenario_config,
)


WORKFLOW_VERSION = "dual-training-workflow-v1"
SCENARIO_IDS = ("A", "B", "C")
HEURISTIC_SCHEDULERS = {
    "FCFS", "NearestNeighbour", "RoundRobin", "Greedy", "Random",
    "Hungarian", "Auction", "GA", "SA",
}
MODEL_BACKED_SCHEDULERS = {
    "LearnedHungarian", "GraphImitation", "GraphPPO", "RainbowDQN",
    "QRDQN", "CQL", "LinUCB", "PPO_RL", "DQN", "SARSA",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path):
    return _sha256_bytes(path.read_bytes()) if path and path.is_file() else None


def _validated_model_path(model, objective: str):
    if model is None:
        return None
    path = Path(model).resolve()
    if not path.is_file():
        raise ValueError(f"{objective} checkpoint is not a file: {path}")
    return path


def _source_fingerprint() -> dict:
    paths = [Path(__file__).resolve(), *sorted(SUPERVISOR.glob("*.py"))]
    files = {
        path.relative_to(ROOT).as_posix(): _file_sha256(path)
        for path in paths
    }
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"sha256": _sha256_bytes(canonical.encode("utf-8")),
            "files": files}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True,
                       ensure_ascii=False, allow_nan=False),
            encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _task_results(tasks, horizon_seconds: float) -> list:
    return [{
        "task_id": task.task_id,
        "status": task.status,
        "priority": task.priority,
        "arrival_time": task.arrival_time,
        "assignment_time": task.assignment_time,
        "pickup_time": task.pickup_time,
        "completion_time": task.completion_time,
        "deadline_time": task.deadline_time,
        "deadline_missed": task.deadline_missed,
        "deadline_missed_at": task.deadline_missed_at,
        "deadline_tardiness_seconds": task.deadline_tardiness_seconds,
        "completed_on_time": task.completed_on_time,
        "horizon_limited": task.is_horizon_limited(horizon_seconds),
        "ideal_distance": task.ideal_distance,
        "actual_distance": task.actual_distance,
    } for task in tasks]


def _run_episode(*, scheduler_name: str, model_path, scenario_id: str,
                 seed: int, manifest: dict, objective: str,
                 duration_seconds: float, utility_profile,
                 utility_score_config, progress_callback=None) -> dict:
    robots, tasks, context = factory_scenario(
        seed, scenario_id=scenario_id, duration_seconds=duration_seconds,
        manifest=manifest)
    profile = (CountRewardProfile(horizon_seconds=duration_seconds)
               if objective == "count" else utility_profile)
    environment = SchedulingEnvironment(
        RLEnvironmentConfig(
            max_robots=8, max_tasks=20,
            max_steps_per_episode=max(10000, len(tasks) * 30 + 100)),
        reward_profile=profile,
        runtime_config=HeadlessRuntimeConfig(
            episode_end_time=duration_seconds,
            max_advance_seconds=duration_seconds,
            fixed_horizon=True),
        simulation_mode="headless")
    _observation, reset_metadata = environment.reset(
        robots, tasks, context, seed=seed)
    scheduler = create_scheduler(
        scheduler_name, model_path, seed=seed, allow_safe_fallback=False)
    reset = getattr(scheduler, "reset", None)
    if callable(reset):
        reset()

    episode_return = 0.0
    decisions = 0
    invalid_outputs = 0
    fallback_decisions = 0
    scheduler_latencies_ms = []
    invalid_truncation = False
    terminated = False
    while decisions < environment.config.max_steps_per_episode:
        _state, reward, terminated, truncated, info = (
            environment.step_scheduler(scheduler))
        decisions += 1
        episode_return += float(reward)
        invalid_outputs += int(bool(
            info.get("scheduler_output_rejected") or
            info.get("assignment_rejected") or info.get("invalid_action")))
        diagnostics = info.get("scheduler_diagnostics", {})
        fallback_decisions += int(bool(diagnostics.get("fallback", False)))
        latency = info.get("scheduler_computation_time")
        if latency is not None:
            latency = float(latency) * 1000.0
            if not math.isfinite(latency) or latency < 0:
                raise ValueError("scheduler emitted invalid computation_time")
            scheduler_latencies_ms.append(latency)
        invalid_truncation = invalid_truncation or bool(
            info.get("invalid_truncation"))
        if progress_callback is not None:
            progress_callback(environment.runtime_telemetry(), decisions)
        if terminated or truncated:
            break

    final_tasks = environment._tasks
    actual_hash = task_manifest_sha256(
        task.generation_parameters() for task in final_tasks)
    if actual_hash != manifest["task_manifest_sha256"]:
        raise RuntimeError("runtime task manifest changed")
    route_distances = {
        task.task_id: {
            "ideal_distance": task.ideal_distance,
            "actual_distance": task.actual_distance,
        }
        for task in final_tasks if task.assignment_time is not None
    }
    count_score = count_evaluation(
        final_tasks, horizon_seconds=duration_seconds)
    utility_score = utility_v2_evaluation(
        final_tasks, utility_score_config,
        route_distances=route_distances)
    latency = sorted(scheduler_latencies_ms)

    def percentile(fraction):
        if not latency:
            return None
        index = min(len(latency) - 1,
                    max(0, math.ceil(fraction * len(latency)) - 1))
        return latency[index]

    return {
        "objective": objective,
        "scenario_id": scenario_id,
        "seed": seed,
        "scheduler": scheduler_name,
        "model_path": str(model_path) if model_path else None,
        "model_sha256": _file_sha256(Path(model_path)) if model_path else None,
        "reset_metadata": reset_metadata,
        "episode_return": episode_return,
        "decisions": decisions,
        "terminated": terminated,
        "invalid_truncation": invalid_truncation,
        "invalid_scheduler_outputs": invalid_outputs,
        "fallback_scheduler_commits": fallback_decisions,
        "scheduler_timeout_count": 0,
        "scheduler_latency_p50_ms": percentile(0.50),
        "scheduler_latency_p95_ms": percentile(0.95),
        "scheduler_latency_p99_ms": percentile(0.99),
        "physical_collision_count": None,
        "physical_collision_observation_status": "unobserved",
        "runtime": environment.runtime_telemetry(),
        "count_evaluation": count_score,
        "utility_v2_evaluation": utility_score,
        "task_manifest_sha256": actual_hash,
        "tasks": _task_results(final_tasks, duration_seconds),
    }


def run_workflow(*, output_dir: Path, mode: str, seeds, candidate: str,
                 baseline: str, count_model, utility_model,
                 duration_seconds: float, lambda_deadline: float,
                 tau_by_priority: dict, kappa_by_priority: dict,
                 provisional_calibration: bool) -> dict:
    if mode not in {"smoke", "headless_full"}:
        raise ValueError("mode must be smoke or headless_full")
    if (isinstance(duration_seconds, bool) or
            not isinstance(duration_seconds, (int, float)) or
            not math.isfinite(float(duration_seconds)) or
            duration_seconds <= 0):
        raise ValueError("duration_seconds must be finite and positive")
    duration_seconds = float(duration_seconds)
    if mode == "headless_full" and duration_seconds != 1800.0:
        raise ValueError("headless_full requires exactly 1800 seconds")
    known_schedulers = HEURISTIC_SCHEDULERS | MODEL_BACKED_SCHEDULERS
    if candidate not in known_schedulers or baseline not in known_schedulers:
        raise ValueError("candidate and baseline must be known schedulers")
    if candidate == baseline:
        raise ValueError("candidate and baseline must be different")
    if baseline not in HEURISTIC_SCHEDULERS:
        raise ValueError("baseline must be a checkpoint-free scheduler")
    count_model = _validated_model_path(count_model, "Count")
    utility_model = _validated_model_path(utility_model, "Utility V2")
    if candidate in MODEL_BACKED_SCHEDULERS:
        if count_model is None or utility_model is None:
            raise ValueError(
                "model-backed candidate requires both objective checkpoints")
    elif count_model is not None or utility_model is not None:
        raise ValueError(
            "checkpoint paths are invalid for a heuristic candidate")
    if not isinstance(provisional_calibration, bool):
        raise ValueError("provisional_calibration must be boolean")
    seeds = tuple(seeds)
    if (not seeds or len(seeds) != len(set(seeds)) or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in seeds)):
        raise ValueError("seeds must be unique integers")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "output_dir must be absent or empty to exclude stale artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_fingerprint_before = _source_fingerprint()

    utility_profile = UtilityV2RewardProfile(
        lambda_deadline=lambda_deadline,
        horizon_seconds=duration_seconds)
    utility_score_config = UtilityScoreConfig(
        tau_by_priority=tau_by_priority,
        kappa_by_priority=kappa_by_priority,
        horizon_seconds=duration_seconds)
    count_profile = CountRewardProfile(horizon_seconds=duration_seconds)
    calibration = {
        "status": "provisional" if provisional_calibration else "frozen",
        "utility_score_config": utility_score_config.canonical(),
        "utility_score_config_hash": utility_score_config.sha256,
        "count_reward_profile": count_profile.canonical(),
        "count_reward_profile_hash": count_profile.sha256,
        "utility_reward_profile": utility_profile.canonical(),
        "utility_reward_profile_hash": utility_profile.sha256,
    }
    _write_json(output_dir / "calibration" / "profiles.json", calibration)

    manifests = {}
    for scenario_id in SCENARIO_IDS:
        for seed in seeds:
            manifest = generate_task_manifest(
                scenario_id, formal_scenario_config(scenario_id), seed,
                duration_seconds=duration_seconds)
            path = output_dir / "manifests" / scenario_id / f"{seed}.json"
            write_task_manifest(path, manifest)
            manifests[(scenario_id, seed)] = manifest

    policy_descriptors = {}
    for objective, model in (("count", count_model),
                             ("utility_v2", utility_model)):
        descriptor = {
            "objective": objective,
            "scheduler": candidate,
            "checkpoint_path": str(model) if model else None,
            "checkpoint_sha256": _file_sha256(model) if model else None,
            "checkpoint_status": (
                "provided" if model else "not_trained_reference_scheduler"),
            "reward_profile_hash": (
                count_profile.sha256 if objective == "count"
                else utility_profile.sha256),
        }
        path = output_dir / objective / candidate / "policy_descriptor.json"
        _write_json(path, descriptor)
        policy_descriptors[objective] = descriptor

    runs = []
    started = time.time()
    for objective in ("count", "utility_v2"):
        for scenario_id in SCENARIO_IDS:
            for seed in seeds:
                manifest = manifests[(scenario_id, seed)]
                for scheduler_name, model in (
                        (baseline, None),
                        (candidate, count_model if objective == "count"
                         else utility_model)):
                    run = _run_episode(
                        scheduler_name=scheduler_name, model_path=model,
                        scenario_id=scenario_id, seed=seed,
                        manifest=manifest, objective=objective,
                        duration_seconds=duration_seconds,
                        utility_profile=utility_profile,
                        utility_score_config=utility_score_config)
                    path = (output_dir / objective / scheduler_name /
                            scenario_id / f"{seed}.json")
                    _write_json(path, run)
                    runs.append(run)

    source_fingerprint_after = _source_fingerprint()
    source_unchanged = source_fingerprint_before == source_fingerprint_after

    def run_is_valid(run):
        descriptor = policy_descriptors[run["objective"]]
        expected_model_hash = (
            descriptor["checkpoint_sha256"]
            if run["scheduler"] == candidate else None)
        return bool(
            run["terminated"] and not run["invalid_truncation"]
            and run["invalid_scheduler_outputs"] == 0
            and run["fallback_scheduler_commits"] == 0
            and run["scheduler_timeout_count"] == 0
            and run["model_sha256"] == expected_model_hash
            and run["runtime"]["reward_profile_hash"] ==
            descriptor["reward_profile_hash"]
            and run["runtime"]["horizon_finalized"]
            and run["runtime"]["termination_reason"] == "episode_horizon"
            and run["runtime"]["current_time"] == duration_seconds
            and run["runtime"]["episode_end_time"] == duration_seconds)

    execution_valid = source_unchanged and all(run_is_valid(run) for run in runs)
    run_by_key = {
        (run["objective"], run["scenario_id"], run["seed"],
         run["scheduler"]): run for run in runs
    }
    paired = []
    for objective in ("count", "utility_v2"):
        for scenario_id in SCENARIO_IDS:
            for seed in seeds:
                candidate_run = run_by_key[
                    (objective, scenario_id, seed, candidate)]
                baseline_run = run_by_key[
                    (objective, scenario_id, seed, baseline)]
                if (candidate_run["task_manifest_sha256"] !=
                        baseline_run["task_manifest_sha256"]):
                    raise RuntimeError("paired runs used different manifests")
                if objective == "count":
                    comparison = paired_count_delta(
                        candidate_run["count_evaluation"][
                            "total_tasks_completed"],
                        baseline_run["count_evaluation"][
                            "total_tasks_completed"])
                else:
                    candidate_score = candidate_run["utility_v2_evaluation"][
                        "utility_score_deadline_v2"]
                    baseline_score = baseline_run["utility_v2_evaluation"][
                        "utility_score_deadline_v2"]
                    comparison = {
                        "candidate_score": candidate_score,
                        "baseline_score": baseline_score,
                        "absolute_delta": (
                            None if candidate_score is None or baseline_score is None
                            else candidate_score - baseline_score),
                        "status": (
                            "ok" if candidate_score is not None
                            and baseline_score is not None else
                            "no_evaluable_tasks"),
                    }
                paired.append({
                    "objective": objective,
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "task_manifest_sha256": (
                        candidate_run["task_manifest_sha256"]),
                    "comparison": comparison,
                })
    _write_json(output_dir / "headless_validation.json", {
        "runtime_mode": "headless_webots_logic",
        "physics_fidelity": "business_logic_only",
        "pairs": paired,
    })
    physical_safety_observed = all(
        run["physical_collision_observation_status"] == "observed"
        for run in runs)
    distinct_checkpoints = bool(
        count_model and utility_model and
        _file_sha256(count_model) != _file_sha256(utility_model))
    ready = bool(
        mode == "headless_full" and execution_valid
        and not provisional_calibration and distinct_checkpoints
        and physical_safety_observed)
    deployment = {
        "status": "ready" if ready else "not_ready",
        "gates": {
            "complete_headless_matrix": execution_valid,
            "formal_1800_second_duration": (
                mode == "headless_full" and duration_seconds == 1800.0),
            "calibration_frozen": not provisional_calibration,
            "distinct_objective_checkpoints": distinct_checkpoints,
            "webots_physical_safety_observed": physical_safety_observed,
        },
        "limitations": [
            "Headless runtime cannot establish physical collision safety.",
            *([] if distinct_checkpoints else [
                "Distinct trained Count and Utility checkpoints were not provided."]),
            *([] if not provisional_calibration else [
                "Utility calibration is provisional."]),
        ],
    }
    _write_json(output_dir / "deployment_manifest.json", deployment)
    artifact_hashes = {
        str(path.relative_to(output_dir)).replace("\\", "/"):
            _file_sha256(path)
        for path in sorted(output_dir.rglob("*.json"))
        if path.name != "workflow_execution_report.json"
    }
    report = {
        "workflow_version": WORKFLOW_VERSION,
        "mode": mode,
        "duration_seconds": duration_seconds,
        "seeds": list(seeds),
        "candidate": candidate,
        "baseline": baseline,
        "calibration": calibration,
        "policy_descriptors": policy_descriptors,
        "run_count": len(runs),
        "execution_seconds": time.time() - started,
        "execution_valid": execution_valid,
        "source_fingerprint": source_fingerprint_before,
        "source_unchanged_during_run": source_unchanged,
        "training_status": (
            "external_checkpoints_evaluated" if count_model and utility_model
            else "not_trained_reference_scheduler"),
        "paired_results": paired,
        "artifact_hashes": artifact_hashes,
        "deployment": deployment,
    }
    _write_json(output_dir / "workflow_execution_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "headless_full"),
                        required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--candidate", default="Greedy")
    parser.add_argument("--baseline", default="Hungarian")
    parser.add_argument("--count-model", type=Path)
    parser.add_argument("--utility-model", type=Path)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--lambda-deadline", type=float, required=True)
    for prefix in ("tau", "kappa"):
        for priority in (1, 2, 3):
            parser.add_argument(
                f"--{prefix}-p{priority}", type=float, required=True)
    parser.add_argument("--provisional-calibration", action="store_true")
    args = parser.parse_args()
    duration = args.duration
    if duration is None:
        duration = 300.0 if args.mode == "smoke" else 1800.0
    report = run_workflow(
        output_dir=args.output_dir, mode=args.mode, seeds=args.seed,
        candidate=args.candidate, baseline=args.baseline,
        count_model=args.count_model, utility_model=args.utility_model,
        duration_seconds=duration,
        lambda_deadline=args.lambda_deadline,
        tau_by_priority={
            priority: getattr(args, f"tau_p{priority}")
            for priority in (1, 2, 3)},
        kappa_by_priority={
            priority: getattr(args, f"kappa_p{priority}")
            for priority in (1, 2, 3)},
        provisional_calibration=args.provisional_calibration)
    print(json.dumps({
        "status": report["deployment"]["status"],
        "run_count": report["run_count"],
        "execution_valid": report["execution_valid"],
        "report": str((args.output_dir / "workflow_execution_report.json").resolve()),
    }, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
