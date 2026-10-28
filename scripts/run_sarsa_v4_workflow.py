"""Neural SARSA: standalone training -> Webots fine tune -> frozen evaluation.

There is no intermediate non-regression check, rollback, or holdout-based model
selection.  Every command is episode-resumable and review-gated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback
import uuid

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "scripts", ROOT / "controllers/factory_supervisor"):
    sys.path.insert(0, str(directory))

import neural_sarsa_workflow_contract as audit
import numpy as np
import run_experiments
import run_strict_1800_workflow as strict
import run_webots_1800_workflow as physical
from config import RobotState
from dual_objective import count_evaluation, utility_v2_evaluation
from headless_training_runtime import HeadlessFactoryRuntime, HeadlessRuntimeConfig
from sarsa_v4_agent import MODEL_VERSION, SarsaV4Agent, SarsaV4Config
from sarsa_v4_features import FEATURE_DIM, FEATURE_VERSION
from sarsa_v4_runtime import SarsaV4Scheduler
from training_scenarios import factory_scenario


SCENARIOS = ("A", "B", "C")
OBJECTIVES = ("count", "utility_v2")
DEFAULT_WEBOTS = Path("C:/Program Files/Webots/msys64/mingw64/bin/webots.exe")


def digest(document):
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


@contextmanager
def lock(root):
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
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def contract(root):
    root = audit.validate_root(root)
    files = sorted({
        path for directory in (ROOT / "controllers", ROOT / "scripts")
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts})
    document = {
        "version": "sarsa-v4-neural-run-v1",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "sources": {
            path.relative_to(ROOT).as_posix(): audit.sha256(path)
            for path in files},
        "calibration_sha256": audit.sha256(strict.CALIBRATION),
        "objectives": list(OBJECTIVES),
        "scenarios": list(SCENARIOS),
        "horizon": 1800,
        "training_seeds": list(range(21000, 21060)),
        "fine_tune_seeds": list(range(41000, 41010)),
        "evaluation_seeds": list(range(61000, 61006)),
        "nonregression_validation": False,
        "rollback": False,
        "validation_model_selection": False,
        "selection": "latest cumulative checkpoint",
        "config": SarsaV4Config().__dict__,
        "physical_learning_rate_multiplier": 0.1,
        "standalone_device": "cuda",
        "webots_device": "cpu",
    }
    path = root / "run_contract.json"
    existing = audit.read(path)
    if existing is not None and existing != document:
        raise RuntimeError(
            "source/configuration changed after freeze; stale resume refused")
    if existing is None:
        audit.write(root, path, document)
    return document


def progress(root, phase, objective, **values):
    row = {
        "at": audit.now(), "phase": phase,
        "objective": objective, **values}
    audit.write(
        root, Path(root) / "progress" / f"{phase}_{objective}.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def manifest(root, partition, scenario, seed):
    allowed = {
        "standalone_training": range(21000, 21060),
        "webots_finetune": range(41000, 41010),
        "final_evaluation": range(61000, 61006),
    }
    if scenario not in SCENARIOS or seed not in allowed.get(partition, ()):
        raise ValueError("unregistered neural SARSA seed partition")
    source = strict.manifest_path(partition, scenario, seed)
    strict._load_manifest(source)
    destination = audit.confined(
        root, Path(root) / "manifests" / partition /
        scenario / f"{seed}.json")
    if destination.exists():
        if audit.sha256(destination) != audit.sha256(source):
            raise RuntimeError("copied task manifest changed")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return destination


def checked_record(path, identity):
    existing = audit.read(path)
    if existing is None:
        return None
    if any(existing.get(key) != value for key, value in identity.items()):
        raise RuntimeError(f"resume identity mismatch: {path}")
    checkpoint = existing.get("checkpoint")
    if checkpoint and (not Path(checkpoint).is_file() or
                       audit.sha256(checkpoint) !=
                       existing.get("checkpoint_sha256")):
        raise RuntimeError(f"resume checkpoint missing or changed: {path}")
    return existing


def round_summary(rows, index, seed):
    learning = [row["learning"] for row in rows]
    return {
        "round": index,
        "seed": seed,
        "mean_metric": statistics.fmean(
            row["objective_metric"] for row in rows),
        "mean_absolute_td": statistics.fmean(
            row["mean_absolute_td"] for row in learning),
        "mean_last_loss": statistics.fmean(
            row["last_loss"] for row in learning),
        "decisions": sum(row["decision_count"] for row in learning),
        "transition_updates": sum(
            row["transition_updates"] for row in learning),
        "optimizer_updates": sum(
            row["optimizer_updates"] for row in learning),
    }


def headless_episode(root, objective, input_checkpoint, scenario, seed,
                     document):
    root = Path(root)
    task_manifest = manifest(
        root, "standalone_training", scenario, seed)
    directory = root / "standalone" / objective / "episodes" / scenario / str(seed)
    result_path = directory / "result.json"
    identity = {
        "contract_sha256": digest(document),
        "input_checkpoint_sha256": audit.sha256(input_checkpoint),
        "manifest_sha256": audit.sha256(task_manifest),
        "scenario": scenario, "seed": seed, "objective": objective}
    previous = checked_record(result_path, identity)
    if previous:
        return previous
    attempt = directory / "attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    agent = SarsaV4Agent.load(
        input_checkpoint, objective=objective,
        device=document["standalone_device"])
    robots, tasks, context = factory_scenario(
        seed, scenario_id=scenario, duration_seconds=1800.0,
        manifest=strict._load_manifest(task_manifest))
    from dataclasses import replace
    context = replace(context, configuration={
        **context.configuration, "graph_traffic_enabled": True})
    runtime = HeadlessFactoryRuntime(
        robots, tasks, context, seed=seed,
        config=HeadlessRuntimeConfig(
            episode_end_time=1800.0, fixed_horizon=True))
    scheduler = SarsaV4Scheduler(
        agent=agent, seed=seed, training=True,
        physical=False, device=document["standalone_device"])
    profile = strict.utility_profile() if objective == "utility_v2" else None
    started = time.perf_counter()
    heartbeat = 0.0
    iterations = 0
    progress(root, "standalone", objective,
             scenario=scenario, seed=seed, sim_time=0.0, status="running")
    while not runtime.is_terminal():
        iterations += 1
        if iterations > 50000:
            raise RuntimeError("headless neural SARSA made no bounded progress")
        scheduler.observe(
            runtime.tasks, runtime.current_time, utility_profile=profile)
        pending = runtime.arrived_pending_tasks()
        idle = any(
            state.get("state") == RobotState.IDLE and
            state.get("current_task") is None
            for state in runtime.robots.values())
        if idle and pending:
            decision = scheduler.assign(
                pending, runtime.robots, runtime.context)
            if decision.assignments:
                selected = decision.assignments[0]
                event = runtime.dispatch(selected)
                if event.get("type") == "assignment_committed":
                    scheduler.on_assignment_committed(selected)
                else:
                    scheduler.on_assignment_rejected(
                        selected, event.get("reason", "runtime_rejected"))
            else:
                delay = (max(0.032, scheduler.wait_until - runtime.current_time)
                         if scheduler.wait_until > runtime.current_time else 2.0)
                runtime.advance_until_event(max_seconds=delay)
        else:
            runtime.advance_until_event(max_seconds=30.0)
        if runtime.current_time >= heartbeat + 300.0:
            heartbeat = runtime.current_time
            progress(
                root, "standalone", objective, scenario=scenario, seed=seed,
                sim_time=round(runtime.current_time, 3),
                transition_updates=(agent.transition_updates -
                                    scheduler.initial_transition_updates),
                status="running")
    learning = scheduler.finalize(
        runtime.tasks, runtime.current_time, utility_profile=profile)
    count = count_evaluation(runtime.tasks, horizon_seconds=1800.0)
    utility = utility_v2_evaluation(
        runtime.tasks, strict.utility_score_config(), route_distances={
            task.task_id: {
                "ideal_distance": task.ideal_distance,
                "actual_distance": task.actual_distance}
            for task in runtime.tasks if task.assignment_time is not None})
    checkpoint = attempt / "model.pt"
    agent.save(checkpoint)
    record = {
        **identity, "status": "complete", "checkpoint": str(checkpoint),
        "checkpoint_sha256": audit.sha256(checkpoint), "learning": learning,
        "completed_tasks": count["total_tasks_completed"],
        "utility": utility["utility_score_deadline_v2"],
        "objective_metric": (
            count["total_tasks_completed"] if objective == "count"
            else utility["utility_score_deadline_v2"]),
        "wall_seconds": time.perf_counter() - started,
        "physics_fidelity": "business_logic_only",
        "runtime": runtime.telemetry_snapshot(),
    }
    audit.write(root, result_path, record)
    progress(root, "standalone", objective, scenario=scenario, seed=seed,
             status="episode_complete", metric=record["objective_metric"],
             checkpoint=str(checkpoint))
    return record


def _standalone_worker(root, objective, document):
    root = Path(root)
    identity = {"contract_sha256": digest(document), "objective": objective}
    report_path = root / "standalone" / objective / "training_report.json"
    previous = checked_record(report_path, identity)
    if previous:
        return previous
    initial = root / "standalone" / objective / "initial.pt"
    expected_initial = SarsaV4Agent(
        objective, config=SarsaV4Config(**document["config"]),
        device=document["standalone_device"])
    if not initial.exists():
        expected_initial.save(initial)
    actual_initial = SarsaV4Agent.load(
        initial, objective=objective,
        device=document["standalone_device"])
    if actual_initial.document() != expected_initial.document():
        raise RuntimeError(
            "standalone initialization changed after registration")
    initial_hash = audit.sha256(initial)
    current = initial
    rounds = []
    for index, seed in enumerate(document["training_seeds"], 1):
        rows = []
        for scenario in SCENARIOS:
            row = headless_episode(
                root, objective, current, scenario, seed, document)
            rows.append(row)
            current = Path(row["checkpoint"])
        summary = round_summary(rows, index, seed)
        summary.update({
            "checkpoint": str(current),
            "checkpoint_sha256": audit.sha256(current)})
        rounds.append(summary)
        audit.write(
            root, root / "standalone" / objective / "rounds" /
            f"{index:03d}.json", summary)
        progress(root, "standalone", objective,
                 status="round_complete", **summary)
    report = {
        **identity, "status": "complete",
        "initial_checkpoint_sha256": initial_hash,
        "checkpoint": str(current),
        "checkpoint_sha256": audit.sha256(current),
        "balanced_rounds": len(rounds),
        "episodes": len(rounds) * len(SCENARIOS),
        "rounds": rounds,
        "completion_reason": "all_registered_training_seeds_complete",
        "intermediate_nonregression": "skipped_by_user_instruction",
        "selection": document["selection"],
    }
    audit.write(root, report_path, report)
    return report


def standalone_worker(root, objective, document):
    try:
        return _standalone_worker(root, objective, document)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        try:
            progress(root, "standalone", objective, status="failed",
                     error=f"{type(exc).__name__}: {exc}")
        except Exception:
            traceback.print_exc(file=sys.stderr)
        raise


def physical_episode(root, objective, checkpoint, scenario, seed,
                     phase, document, webots):
    root = Path(root)
    fine = phase == "fine_tune"
    partition = "webots_finetune" if fine else "final_evaluation"
    task_manifest = manifest(root, partition, scenario, seed)
    directory = root / phase / objective / "v4" / scenario / str(seed)
    result_path = directory / "result.json"
    identity = {
        "contract_sha256": digest(document), "objective": objective,
        "input_checkpoint_sha256": audit.sha256(checkpoint),
        "manifest_sha256": audit.sha256(task_manifest),
        "scenario": scenario, "seed": seed, "variant": "v4",
        "phase": phase}
    previous = checked_record(result_path, identity)
    if previous:
        return previous
    attempt = directory / "attempts" / uuid.uuid4().hex
    output = attempt / "updated_model.pt"
    environment = physical.common_environment(task_manifest)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
        "SMART_FACTORY_PHYSICAL_FINE_TUNE": "0",
        "SMART_FACTORY_GRAPH_PPO_FINE_TUNE": "0",
        "SMART_FACTORY_SARSA_V3": "0",
        "SMART_FACTORY_SARSA_V4": "1",
        "SMART_FACTORY_SARSA_V4_TRAINING": "1" if fine else "0",
        "SMART_FACTORY_SARSA_V4_RUN_ROOT": str(root),
        "SMART_FACTORY_SARSA_V4_OUTPUT_MODEL": str(output) if fine else "",
        "SMART_FACTORY_SARSA_V4_DEVICE": document["webots_device"],
    })
    progress(root, phase, objective, scenario=scenario, seed=seed,
             variant="v4", status="running")
    result = run_experiments.run_single_experiment(
        scenario, "SARSA", seed, webots_path=str(webots),
        model_path=str(checkpoint), controller_mode="external",
        extra_env=environment, results_dir=str(attempt / "raw"))
    envelope = {
        "schema_version": 1, "completed_at": audit.now(),
        "input_model_path": str(Path(checkpoint).resolve()),
        "input_model_sha256": identity["input_checkpoint_sha256"],
        "manifest_path": str(task_manifest),
        "manifest_sha256": identity["manifest_sha256"],
        "result": result,
    }
    if not physical.valid_episode(
            envelope, scenario=scenario, seed=seed, manifest=task_manifest,
            input_model_sha256=identity["input_checkpoint_sha256"],
            fine_tune_objective=None, expected_scheduler="SARSA"):
        raise RuntimeError(
            "physical neural SARSA result failed integrity checks")
    if audit.sha256(checkpoint) != identity["input_checkpoint_sha256"]:
        raise RuntimeError("physical run changed its input checkpoint")
    if not physical.valid_pure_policy_result(envelope):
        raise RuntimeError("neural SARSA result contains substituted decisions")
    formal = result["formal_evaluation"]
    metrics = result["summary_metrics"]
    learning = formal.get("sarsa_v4", {})
    if (learning.get("training") is not fine or
            learning.get("model_version") != MODEL_VERSION):
        raise RuntimeError("physical neural SARSA mode/model mismatch")
    if fine:
        if (learning.get("transition_updates", 0) <= 0 or
                learning.get("optimizer_updates", 0) <= 0 or
                not output.is_file()):
            raise RuntimeError("Webots neural SARSA did not update/export")
        SarsaV4Agent.load(output, objective=objective, device="cpu")
    elif (learning.get("transition_updates") != 0 or
          learning.get("optimizer_updates") != 0):
        raise RuntimeError("frozen evaluation performed learning")
    final_checkpoint = output if fine else Path(checkpoint)
    count = formal["count_evaluation"]["total_tasks_completed"]
    utility = formal["utility_v2_evaluation"]["utility_score_deadline_v2"]
    row = {
        **identity, "status": "complete",
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": audit.sha256(final_checkpoint),
        "learning": learning, "completed_tasks": count, "utility": utility,
        "objective_metric": count if objective == "count" else utility,
        "pure_policy": True,
        "fallback_commits": metrics.get("fallback_scheduler_commits"),
        "physical_collisions": metrics.get("physical_collision_count"),
        "envelope": envelope,
    }
    audit.write(root, result_path, row)
    progress(root, phase, objective, scenario=scenario, seed=seed,
             variant="v4", status="episode_complete",
             metric=row["objective_metric"])
    return row


def fine_tune(root, objective, document, webots):
    root = Path(root)
    identity = {"contract_sha256": digest(document), "objective": objective}
    path = root / "fine_tune" / objective / "fine_tune_report.json"
    trained = checked_record(
        root / "standalone" / objective / "training_report.json", identity)
    if not trained:
        raise RuntimeError("standalone training is incomplete")
    identity = {
        **identity,
        "standalone_checkpoint_sha256": trained["checkpoint_sha256"]}
    previous = checked_record(path, identity)
    if previous:
        return previous
    current = root / "fine_tune" / objective / "initial.pt"
    initial_record_path = root / "fine_tune" / objective / "initial.json"
    if not current.exists():
        model = SarsaV4Agent.load(
            trained["checkpoint"], objective=objective, device="cpu")
        model.set_learning_rate(
            document["config"]["learning_rate"] *
            document["physical_learning_rate_multiplier"])
        model.save(current)
        audit.write(root, initial_record_path, {
            "standalone_checkpoint_sha256": trained["checkpoint_sha256"],
            "checkpoint_sha256": audit.sha256(current),
            "learning_rate": (
                document["config"]["learning_rate"] *
                document["physical_learning_rate_multiplier"]),
        })
    initial_record = audit.read(initial_record_path)
    if (not initial_record or
            initial_record.get("standalone_checkpoint_sha256") !=
            trained["checkpoint_sha256"] or
            initial_record.get("checkpoint_sha256") != audit.sha256(current)):
        raise RuntimeError("fine-tune initialization identity is invalid")
    loaded_initial = SarsaV4Agent.load(
        current, objective=objective, device="cpu")
    actual_rate = {group["lr"] for group in loaded_initial.optimizer.param_groups}
    if actual_rate != {initial_record["learning_rate"]}:
        raise RuntimeError("fine-tune learning rate differs from registration")
    rounds = []
    for index, seed in enumerate(document["fine_tune_seeds"], 1):
        rows = []
        for scenario in SCENARIOS:
            row = physical_episode(
                root, objective, current, scenario, seed,
                "fine_tune", document, webots)
            rows.append(row)
            current = Path(row["checkpoint"])
        summary = round_summary(rows, index, seed)
        summary.update({
            "checkpoint": str(current),
            "checkpoint_sha256": audit.sha256(current)})
        rounds.append(summary)
        audit.write(
            root, root / "fine_tune" / objective / "rounds" /
            f"{index:02d}.json", summary)
    report = {
        **identity, "status": "complete", "checkpoint": str(current),
        "checkpoint_sha256": audit.sha256(current),
        "balanced_rounds": len(rounds),
        "episodes": len(rounds) * len(SCENARIOS), "rounds": rounds,
        "completion_reason": "all_registered_webots_fine_tune_seeds_complete",
        "intermediate_nonregression": "skipped_by_user_instruction",
    }
    audit.write(root, path, report)
    return report


def evaluate(root, objective, document, webots):
    root = Path(root)
    identity = {"contract_sha256": digest(document), "objective": objective}
    selected = checked_record(
        root / "fine_tune" / objective / "fine_tune_report.json", identity)
    if not selected:
        raise RuntimeError("Webots fine tuning is incomplete")
    model = Path(selected["checkpoint"])
    rows = []
    for scenario in SCENARIOS:
        for seed in document["evaluation_seeds"]:
            rows.append(physical_episode(
                root, objective, model, scenario, seed,
                "evaluation", document, webots))
    report = {
        **identity, "status": "complete", "checkpoint": str(model),
        "checkpoint_sha256": audit.sha256(model),
        "seeds": document["evaluation_seeds"],
        "seed_count": len(document["evaluation_seeds"]),
        "episodes": len(rows),
        "mean_objective_metric": statistics.fmean(
            row["objective_metric"] for row in rows),
        "mean_completed_tasks": statistics.fmean(
            row["completed_tasks"] for row in rows),
        "mean_utility": statistics.fmean(row["utility"] for row in rows),
        "total_collisions": sum(
            row["physical_collisions"] or 0 for row in rows),
        "total_fallback_commits": sum(
            row["fallback_commits"] or 0 for row in rows),
        "by_scenario": {
            scenario: {
                "mean_objective_metric": statistics.fmean(
                    row["objective_metric"] for row in rows
                    if row["scenario"] == scenario),
                "mean_completed_tasks": statistics.fmean(
                    row["completed_tasks"] for row in rows
                    if row["scenario"] == scenario),
                "mean_utility": statistics.fmean(
                    row["utility"] for row in rows
                    if row["scenario"] == scenario),
            } for scenario in SCENARIOS},
        "runs": [{
            "scenario": row["scenario"], "seed": row["seed"],
            "objective_metric": row["objective_metric"],
            "completed_tasks": row["completed_tasks"],
            "utility": row["utility"],
            "physical_collisions": row["physical_collisions"],
            "fallback_commits": row["fallback_commits"],
        } for row in rows],
        "comparison_gate": "none; results were not used for model selection",
    }
    audit.write(root, root / "evaluation" / objective / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=(
            "freeze", "standalone", "fine-tune", "evaluate", "status"))
    parser.add_argument("--run-root", type=Path, default=audit.DEFAULT_RUN)
    parser.add_argument("--webots", type=Path, default=DEFAULT_WEBOTS)
    args = parser.parse_args()
    root = audit.validate_root(args.run_root)
    if args.phase == "status":
        print(json.dumps(
            audit.read(root / "workflow.json"),
            ensure_ascii=False, indent=2))
        return
    required = {
        "freeze": 6, "standalone": 6,
        "fine-tune": 7, "evaluate": 8}[args.phase]
    step = {
        "freeze": 6, "standalone": 7,
        "fine-tune": 8, "evaluate": 9}[args.phase]
    with lock(root):
        audit.require_reviews(root, required)
        document = contract(root)
        if args.phase == "freeze":
            audit.update_step(root, step, "complete",
                              contract_sha256=digest(document))
            print("Neural SARSA run contract frozen:", digest(document))
            return
        audit.update_step(root, step, "in_progress")
        try:
            if args.phase == "standalone":
                with ProcessPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(
                        standalone_worker, root, objective, document)
                        for objective in OBJECTIVES]
                    for future in as_completed(futures):
                        future.result()
            elif args.phase == "fine-tune":
                for objective in OBJECTIVES:
                    fine_tune(root, objective, document, args.webots)
            else:
                for objective in OBJECTIVES:
                    evaluate(root, objective, document, args.webots)
        except Exception as exc:
            audit.update_step(
                root, step, "failed",
                error=f"{type(exc).__name__}: {exc}")
            raise
        audit.update_step(root, step, "awaiting_three_reviews")
        print(f"{args.phase} complete; review evidence is required.", flush=True)


if __name__ == "__main__":
    main()
