"""Resumable Webots-physical fine tuning and all-scheduler evaluation.

Every formal episode is exactly 1800 simulation seconds.  A completed episode
is copied to a canonical, atomically-written ``result.json``.  Resume accepts
only canonical files whose runtime, manifest, scoring contract, physical
collision observation, and input checkpoint hash still match.  An interrupted
episode is intentionally restarted; prior episodes and balanced-round PPO
checkpoints are retained.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_DIR = ROOT / "controllers" / "factory_supervisor"
for entry in (str(ROOT / "scripts"), str(SUPERVISOR_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import run_experiments  # noqa: E402
from config import RL_ENVIRONMENT_VERSION  # noqa: E402
from evaluation_purity import pure_policy_eligibility  # noqa: E402
from graph_ppo_scheduler import (  # noqa: E402
    GraphPPOModel, GraphRolloutStep,
)
from learning_scheduler import (  # noqa: E402
    FEATURE_VERSION, GRAPH_FEATURE_NAMES, PAIR_FEATURE_NAMES,
    WEBOTS_SHARED_SUPERVISION_VERSION,
)
from dual_objective import CountRewardProfile  # noqa: E402


DURATION_SECONDS = 1800.0
ATOMIC_REPLACE_ATTEMPTS = 20
ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.025
ATOMIC_REPLACE_MAX_DELAY_SECONDS = 0.25
_RETRYABLE_WINDOWS_REPLACE_ERRORS = {5, 32, 33}
# Physical fine tuning must keep committed actions on-policy.  The production
# 50 ms inference gate remains active during evaluation, but collection gets a
# generous wall-clock ceiling so a dense path-cost query cannot turn a normal
# temporary wait state into a false training failure.
PHYSICAL_FINE_TUNE_TIMEOUT_SECONDS = 5.0
SCENARIOS = ("A", "B", "C")
TRAINING_SEEDS = tuple(range(21000, 21005))
VALIDATION_SEEDS = tuple(range(31000, 31005))
CALIBRATION = (
    ROOT / "config" / "formal_1800" / "calibration.json")
MANIFEST_ROOT = (
    ROOT / "results" / "all_algorithms_formal" / "dual_auto" /
    "manifests")
COUNT_INPUT = (
    ROOT / "results" / "all_algorithms_formal" / "dual_auto" /
    "count" / "winner" / "model.npz")
UTILITY_INPUT = (
    ROOT / "results" / "all_algorithms_formal" / "dual_auto" /
    "utility_v2" / "winner" / "model.npz")

MODEL_PATHS = {
    "LearnedHungarian": ROOT / "results" / "all_algorithms_formal" /
        "ml" / "candidates" / "learned_cost.npz",
    "GraphImitation": ROOT / "results" / "all_algorithms_formal" /
        "ml" / "candidates" / "graph_imitation.npz",
    "RainbowDQN": ROOT / "results" / "all_algorithms_formal" /
        "advanced_ai" / "candidates" / "rainbow_dqn.npz",
    "QRDQN": ROOT / "results" / "all_algorithms_formal" /
        "advanced_ai" / "candidates" / "qrdqn.npz",
    "CQL": ROOT / "results" / "all_algorithms_formal" /
        "advanced_ai" / "candidates" / "cql.npz",
    "LinUCB": ROOT / "results" / "all_algorithms_formal" /
        "advanced_ai" / "candidates" / "linucb.npz",
    "PPO_RL": ROOT / "results" / "all_algorithms_formal" /
        "legacy_rl" / "candidates" / "ppo_pairwise.npz",
    "SARSA": ROOT / "results" / "all_algorithms_formal" /
        "legacy_rl" / "candidates" / "sarsa.json",
    "DQN": ROOT / "results" / "all_algorithms_formal" /
        "legacy_rl" / "candidates" / "dqn.pkl",
}
NON_GRAPH_POLICIES = tuple(
    scheduler for scheduler in run_experiments.SCHEDULER_TYPES
    if scheduler != "GraphPPO")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Replace a file despite short-lived Windows reader/AV locks.

    Windows can reject ``os.replace`` while another process has the target
    open, even when that process is only reading it.  The report generator and
    monitor both read workflow state while training updates it, so a transient
    sharing violation must not terminate a multi-hour resumable workflow.
    Non-locking failures still fail immediately.
    """
    delay = ATOMIC_REPLACE_INITIAL_DELAY_SECONDS
    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            retryable = (
                isinstance(error, PermissionError) or
                getattr(error, "winerror", None) in
                _RETRYABLE_WINDOWS_REPLACE_ERRORS)
            if not retryable or attempt + 1 >= ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, ATOMIC_REPLACE_MAX_DELAY_SECONDS)


def _discard_temporary(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # Do not mask the original commit error with a cleanup failure.  Temp
        # names are unique, so an orphan cannot corrupt a later commit.
        pass


def atomic_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", delete=False,
                dir=path.parent, prefix=path.name + ".tmp-") as handle:
            temporary = Path(handle.name)
            json.dump(
                document, handle, indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
        temporary = None
    finally:
        _discard_temporary(temporary)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with source.open("rb") as reader, tempfile.NamedTemporaryFile(
                mode="wb", delete=False, dir=destination.parent,
                prefix=destination.name + ".tmp-") as writer:
            temporary = Path(writer.name)
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        _replace_with_retry(temporary, destination)
        temporary = None
    finally:
        _discard_temporary(temporary)


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def manifest_path(partition: str, scenario: str, seed: int) -> Path:
    return MANIFEST_ROOT / partition / scenario / f"{seed}.json"


def validate_sources() -> None:
    required = [CALIBRATION, COUNT_INPUT, UTILITY_INPUT]
    required.extend(MODEL_PATHS.values())
    required.extend(
        manifest_path(partition, scenario, seed)
        for partition, seeds in (
            ("training", TRAINING_SEEDS),
            ("validation", VALIDATION_SEEDS))
        for scenario in SCENARIOS for seed in seeds)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required training/evaluation inputs are missing:\n" +
            "\n".join(missing))


def validate_webots(webots: Path) -> None:
    completed = subprocess.run(
        [str(webots), "--version"], cwd=ROOT, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=15)
    if completed.returncode != 0:
        raise RuntimeError(
            "Webots version preflight failed: " +
            (completed.stdout + completed.stderr)[-1000:])


def common_environment(manifest: Path) -> dict:
    return {
        "SMART_FACTORY_SIM_DURATION": "1800",
        "SMART_FACTORY_TASK_MANIFEST": str(manifest.resolve()),
        "SMART_FACTORY_UTILITY_SCORE_CONFIG_PATH": str(
            CALIBRATION.resolve()),
        "SMART_FACTORY_SUPERVISOR_QUIET": "1",
        # Dense scenario C can be close to real time.  This is a wall-clock
        # failure bound, not the simulated horizon.
        "SMART_FACTORY_WEBOTS_WALL_TIMEOUT_SECONDS": "3900",
    }


def _result_document(envelope):
    return envelope.get("result") if isinstance(envelope, dict) else None


def _formal_calibration_hashes() -> tuple[str, str]:
    document = load_json(CALIBRATION)
    if not isinstance(document, dict):
        raise ValueError("formal calibration is missing or unreadable")
    reward_hash = document.get("utility_reward_profile_sha256")
    score_hash = document.get("utility_score_config_sha256")
    if any(not isinstance(value, str) or len(value) != 64
           for value in (reward_hash, score_hash)):
        raise ValueError("formal calibration hashes are invalid")
    canonical = lambda value: hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")).hexdigest()
    if (not isinstance(document.get("utility_reward_profile"), dict) or
            not isinstance(document.get("utility_score_config"), dict) or
            canonical(document["utility_reward_profile"]) != reward_hash or
            canonical(document["utility_score_config"]) != score_hash):
        raise ValueError("formal calibration content hash mismatch")
    return reward_hash, score_hash


def _physical_utility_reward_hash() -> str:
    """Return the Utility reward contract actually used inside Webots.

    The registered calibration deliberately describes headless safety as
    unobserved.  The Webots supervisor upgrades only that field to physical
    contact observation before constructing ``UtilityV2RewardProfile``.  Bind
    fine-tune validation to the upgraded canonical contract instead of the
    headless calibration hash.
    """
    _formal_calibration_hashes()
    document = load_json(CALIBRATION)
    reward_profile = dict(document["utility_reward_profile"])
    reward_profile["safety_observation"] = "webots_physical"
    return hashlib.sha256(json.dumps(
        reward_profile, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def _valid_shared_supervision(document) -> bool:
    if not isinstance(document, dict) or not (
            document.get("schema_version") == 2 and
            document.get("dataset_version") ==
                WEBOTS_SHARED_SUPERVISION_VERSION and
            document.get("environment_version") == RL_ENVIRONMENT_VERSION and
            document.get("feature_version") == FEATURE_VERSION):
        return False
    try:
        selected = np.asarray(
            document.get("selected_pair_features"), dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return False
    if (selected.shape != (len(PAIR_FEATURE_NAMES),) or
            not np.all(np.isfinite(selected))):
        return False
    ridge_by_objective = document.get("ridge_by_objective")
    graph_by_objective = document.get("graph_by_objective")
    if (not isinstance(ridge_by_objective, dict) or
            not isinstance(graph_by_objective, dict)):
        return False
    for objective in ("count", "utility_v2"):
        ridge = ridge_by_objective.get(objective)
        if not isinstance(ridge, dict):
            return False
        try:
            ridge_features = np.asarray(
                ridge.get("features"), dtype=np.float64)
            targets = np.asarray(
                ridge.get("targets"), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            return False
        sample_count = ridge.get("sample_count")
        if (ridge_features.shape !=
                (len(targets), len(PAIR_FEATURE_NAMES)) or
                not len(targets) or
                not np.all(np.isfinite(ridge_features)) or
                not np.all(np.isfinite(targets)) or
                isinstance(sample_count, bool) or
                not isinstance(sample_count, int) or
                sample_count != len(targets)):
            return False
        batch = graph_by_objective.get(objective)
        if not isinstance(batch, dict):
            return False
        try:
            features = np.asarray(batch.get("features"), dtype=np.float64)
            labels = np.asarray(
                batch.get("labels"), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            return False
        positive = batch.get("positive_count")
        negative = batch.get("negative_count")
        if (features.ndim != 2 or
                features.shape != (len(labels), len(GRAPH_FEATURE_NAMES)) or
                not len(labels) or not np.all(np.isfinite(features)) or
                not np.all(np.isin(labels, (0.0, 1.0))) or
                isinstance(positive, bool) or not isinstance(positive, int) or
                isinstance(negative, bool) or not isinstance(negative, int) or
                positive < 1 or negative < 0 or
                positive + negative != len(labels) or
                int(np.count_nonzero(labels == 1.0)) != positive or
                int(np.count_nonzero(labels == 0.0)) != negative):
            return False
    return True


def valid_action_reward_rollout(
        physical: dict, *, require_shared_supervision: bool = False) -> bool:
    """Validate the dense action-reward and resume contract fail closed."""
    if (not isinstance(physical, dict) or
            physical.get("reward_attribution_version") !=
            "physical_action_reward_v1"):
        return False
    rollout = physical.get("rollout")
    if (not isinstance(rollout, list) or not rollout or
            physical.get("rollout_step_count") != len(rollout)):
        return False
    total = 0.0
    for index, row in enumerate(rollout):
        if not isinstance(row, dict):
            return False
        values = []
        for name in ("reward", "task_reward", "shared_reward"):
            value = row.get(name)
            if (isinstance(value, bool) or
                    not isinstance(value, (int, float)) or
                    not math.isfinite(float(value))):
                return False
            values.append(float(value))
        if not math.isclose(
                values[0], values[1] + values[2],
                rel_tol=0.0, abs_tol=1e-9):
            return False
        if bool(row.get("done")) != (index == len(rollout) - 1):
            return False
        if (require_shared_supervision and
                not _valid_shared_supervision(
                    row.get("shared_supervision"))):
            return False
        total += values[0]
    audit = physical.get("reward_audit")
    if not isinstance(audit, dict):
        return False
    expected = audit.get("reward_total")
    episode_return = physical.get("episode_return")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value))
           for value in (expected, episode_return)):
        return False
    return bool(
        math.isclose(total, float(expected), rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(
            total, float(episode_return), rel_tol=0.0, abs_tol=1e-9))


def valid_pure_policy_result(envelope) -> bool:
    """Require a physical episode to contain only native policy commits."""
    document = _result_document(envelope)
    if not isinstance(document, dict):
        return False
    summary = document.get("summary_metrics", {})
    if not isinstance(summary, dict):
        return False
    values = [summary.get(name) for name in (
        "fallback_scheduler_commits", "invalid_scheduler_outputs",
        "scheduler_timeout_count")]
    return bool(all(
        not isinstance(value, bool) and isinstance(value, int) and value == 0
        for value in values))


def valid_episode(
        envelope, *, scenario: str, seed: int, manifest: Path,
        input_model_sha256: str | None, fine_tune_objective: str | None,
        fine_tune_algorithm: str | None = None,
        expected_scheduler: str | None = None,
        ) -> bool:
    document = _result_document(envelope)
    if not isinstance(document, dict):
        return False
    info = document.get("experiment_info", {})
    summary = document.get("summary_metrics", {})
    formal = document.get("formal_evaluation", {})
    if not all(isinstance(value, dict) for value in (info, summary, formal)):
        return False
    manifest_document = load_json(manifest)
    if not isinstance(manifest_document, dict):
        return False
    try:
        utility_reward_hash, utility_score_hash = (
            _formal_calibration_hashes())
    except ValueError:
        return False
    utility_evaluation = formal.get("utility_v2_evaluation", {})
    count_evaluation = formal.get("count_evaluation", {})
    duration = info.get("sim_duration")
    collision_count = summary.get("physical_collision_count")
    completed_tasks = count_evaluation.get("total_tasks_completed")
    utility_score = utility_evaluation.get("utility_score_deadline_v2")
    checks = (
        envelope.get("schema_version") == 1,
        envelope.get("input_model_sha256") == input_model_sha256,
        info.get("runtime_mode") == "webots",
        (expected_scheduler is None or
         info.get("scheduler") == expected_scheduler),
        info.get("scenario") == scenario,
        info.get("seed") == seed,
        not isinstance(duration, bool) and
            isinstance(duration, (int, float)) and
            math.isfinite(float(duration)) and math.isclose(
                float(duration), DURATION_SECONDS,
                rel_tol=0.0, abs_tol=1e-9),
        info.get("task_manifest_matches_expected") is True,
        info.get("task_manifest_sha256") ==
            manifest_document.get("task_manifest_sha256"),
        summary.get("physical_collision_observation_status") == "observed",
        not isinstance(collision_count, bool) and
            isinstance(collision_count, int) and collision_count >= 0,
        isinstance(count_evaluation, dict),
        not isinstance(completed_tasks, bool) and
            isinstance(completed_tasks, int) and completed_tasks >= 0,
        isinstance(utility_evaluation, dict),
        utility_evaluation.get("utility_score_config_hash") ==
            utility_score_hash,
        not isinstance(utility_score, bool) and
            isinstance(utility_score, (int, float)) and
            math.isfinite(float(utility_score)),
    )
    if not all(checks):
        return False
    if fine_tune_objective is not None:
        physical = formal.get("physical_fine_tune", {})
        expected_reward_hash = (
            CountRewardProfile(horizon_seconds=DURATION_SECONDS).sha256
            if fine_tune_objective == "count" else
            _physical_utility_reward_hash())
        if not (
                physical.get("status") == "complete" and
                physical.get("objective") == fine_tune_objective and
                (fine_tune_algorithm is None or
                 physical.get("algorithm") == fine_tune_algorithm) and
                physical.get("reward_audit", {}).get(
                    "reward_profile_hash") == expected_reward_hash and
                valid_pure_policy_result(envelope) and
                valid_action_reward_rollout(
                    physical, require_shared_supervision=True)):
            return False
    return True


def run_or_resume_episode(
        *, canonical_path: Path, webots: Path, scenario: str, seed: int,
        manifest: Path, scheduler: str, model_path: Path | None,
        fine_tune_objective: str | None,
        fine_tune_algorithm: str | None = None) -> tuple[dict, bool]:
    model_hash = sha256_file(model_path) if model_path else None
    cached = load_json(canonical_path)
    if valid_episode(
            cached, scenario=scenario, seed=seed, manifest=manifest,
            input_model_sha256=model_hash,
            fine_tune_objective=fine_tune_objective,
            fine_tune_algorithm=fine_tune_algorithm,
            expected_scheduler=scheduler):
        print(f"[RESUME] {canonical_path.parent.relative_to(ROOT)}", flush=True)
        return cached, True

    run_dir = canonical_path.parent / "raw"
    environment = common_environment(manifest)
    if fine_tune_objective is not None:
        algorithm = fine_tune_algorithm or scheduler
        environment.update({
            "SMART_FACTORY_PHYSICAL_FINE_TUNE": "1",
            "SMART_FACTORY_GRAPH_PPO_FINE_TUNE": (
                "1" if algorithm == "GraphPPO" else "0"),
            "SMART_FACTORY_FINE_TUNE_ALGORITHM": algorithm,
            "SMART_FACTORY_FINE_TUNE_OBJECTIVE": fine_tune_objective,
            "RL_SCHEDULER_TIMEOUT_SECONDS": str(
                PHYSICAL_FINE_TUNE_TIMEOUT_SECONDS),
        })
    else:
        environment["SMART_FACTORY_PHYSICAL_FINE_TUNE"] = "0"
        environment["SMART_FACTORY_GRAPH_PPO_FINE_TUNE"] = "0"
        environment.pop("SMART_FACTORY_FINE_TUNE_ALGORITHM", None)
        environment.pop("SMART_FACTORY_FINE_TUNE_OBJECTIVE", None)
        # Evaluation deliberately uses the production default (50 ms unless
        # the caller explicitly configured another value).
        environment.pop("RL_SCHEDULER_TIMEOUT_SECONDS", None)
    print(
        f"[RUN] scheduler={scheduler} scenario={scenario} seed={seed} "
        f"duration={DURATION_SECONDS:.0f}s", flush=True)
    result = run_experiments.run_single_experiment(
        scenario, scheduler, seed, webots_path=str(webots),
        model_path=str(model_path) if model_path else None,
        controller_mode="external", extra_env=environment,
        results_dir=str(run_dir))
    envelope = {
        "schema_version": 1,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_model_path": str(model_path.resolve()) if model_path else None,
        "input_model_sha256": model_hash,
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "result": result,
    }
    if not valid_episode(
            envelope, scenario=scenario, seed=seed, manifest=manifest,
            input_model_sha256=model_hash,
            fine_tune_objective=fine_tune_objective,
            fine_tune_algorithm=fine_tune_algorithm,
            expected_scheduler=scheduler):
        raise RuntimeError(
            f"Webots episode failed formal validation: {scheduler}/"
            f"{scenario}/{seed}")
    atomic_json(canonical_path, envelope)
    return envelope, False


def rollout_from_envelopes(envelopes) -> list[GraphRolloutStep]:
    rollout = []
    for envelope in envelopes:
        physical = envelope["result"]["formal_evaluation"][
            "physical_fine_tune"]
        for item in physical["rollout"]:
            features = np.asarray(item["features"], dtype=np.float32)
            action = int(item["action_index"])
            if (features.ndim != 2 or action < 0 or
                    action >= len(features) or
                    not np.all(np.isfinite(features))):
                raise RuntimeError("saved physical rollout is invalid")
            rollout.append(GraphRolloutStep(
                features=features,
                action_index=action,
                old_log_probability=float(item["old_log_probability"]),
                value=float(item["value"]),
                reward=float(item["reward"]),
                done=bool(item["done"]),
            ))
    return rollout


def fine_tune_objective(
        objective: str, input_checkpoint: Path, output_dir: Path,
        webots: Path, update_state) -> Path:
    model = GraphPPOModel.load(input_checkpoint)
    tuned_config = replace(
        model.config, learning_rate=model.config.learning_rate * 0.1)
    model.config = tuned_config
    model.network.config = tuned_config
    model.optimizer.learning_rate = tuned_config.learning_rate
    current_checkpoint = input_checkpoint

    checkpoint_dir = output_dir / "checkpoints"
    for round_index, seed in enumerate(TRAINING_SEEDS, start=1):
        checkpoint = checkpoint_dir / f"after_round_{round_index:02d}.npz"
        round_record_path = (
            checkpoint_dir / f"after_round_{round_index:02d}.json")
        prior_record = load_json(round_record_path)
        input_hash = sha256_file(current_checkpoint)
        expected_episode_paths = [
            output_dir / "episodes" /
            f"round_{round_index:02d}_seed_{seed}" / scenario /
            "result.json"
            for scenario in SCENARIOS]
        checkpoint_reusable = bool(
            isinstance(prior_record, dict) and checkpoint.is_file() and
            prior_record.get("schema_version") == 1 and
            prior_record.get("objective") == objective and
            prior_record.get("round") == round_index and
            prior_record.get("seed") == seed and
            prior_record.get("input_checkpoint_sha256") == input_hash and
            prior_record.get("output_checkpoint_sha256") ==
                sha256_file(checkpoint) and
            prior_record.get("episode_result_sha256") == [
                sha256_file(path) if path.is_file() else None
                for path in expected_episode_paths] and
            all(valid_episode(
                load_json(path), scenario=scenario, seed=seed,
                manifest=manifest_path("training", scenario, seed),
                input_model_sha256=input_hash,
                fine_tune_objective=objective)
                for path, scenario in zip(expected_episode_paths, SCENARIOS)))
        if checkpoint_reusable:
            model = GraphPPOModel.load(checkpoint)
            if not math.isclose(
                    model.config.learning_rate, tuned_config.learning_rate,
                    rel_tol=0.0, abs_tol=1e-15):
                raise RuntimeError(
                    "saved fine-tune checkpoint has wrong learning rate")
            current_checkpoint = checkpoint
            print(
                f"[RESUME] {objective} balanced round={round_index} "
                f"checkpoint={prior_record['output_checkpoint_sha256'][:12]}",
                flush=True)
            continue
        envelopes = []
        for scenario in SCENARIOS:
            update_state(
                stage=f"fine_tune_{objective}",
                current={"round": round_index, "scenario": scenario,
                         "seed": seed})
            canonical = (
                output_dir / "episodes" / f"round_{round_index:02d}_seed_{seed}" /
                scenario / "result.json")
            envelope, _cached = run_or_resume_episode(
                canonical_path=canonical, webots=webots,
                scenario=scenario, seed=seed,
                manifest=manifest_path("training", scenario, seed),
                scheduler="GraphPPO", model_path=current_checkpoint,
                fine_tune_objective=objective)
            envelopes.append(envelope)

        rollout = rollout_from_envelopes(envelopes)
        if sum(step.done for step in rollout) != len(SCENARIOS):
            raise RuntimeError(
                "balanced fine-tune round does not contain three episodes")
        update_metrics = model.update(rollout)
        model.save(checkpoint)
        round_record = {
            "schema_version": 1,
            "objective": objective,
            "round": round_index,
            "seed": seed,
            "scenarios": list(SCENARIOS),
            "learning_rate_multiplier": 0.1,
            "rollout_steps": len(rollout),
            "input_checkpoint_sha256": sha256_file(current_checkpoint),
            "output_checkpoint_sha256": sha256_file(checkpoint),
            "episode_result_sha256": [
                sha256_file(
                    output_dir / "episodes" /
                    f"round_{round_index:02d}_seed_{seed}" / scenario /
                    "result.json")
                for scenario in SCENARIOS],
            "update_metrics": update_metrics,
        }
        atomic_json(round_record_path, round_record)
        current_checkpoint = checkpoint
        print(
            f"[CHECKPOINT] {objective} round={round_index} "
            f"sha256={round_record['output_checkpoint_sha256'][:12]}",
            flush=True)

    final_checkpoint = output_dir / "webots_finetuned_model.npz"
    # Preserve the exact last-round bytes (and therefore its stable hash) so
    # resumed evaluation never invalidates completed GraphPPO episodes merely
    # because a ZIP/NPZ timestamp changed.
    atomic_copy(current_checkpoint, final_checkpoint)
    GraphPPOModel.load(final_checkpoint)
    atomic_json(output_dir / "fine_tune_report.json", {
        "status": "complete",
        "objective": objective,
        "algorithm": "GraphPPO",
        "duration_seconds_per_episode": DURATION_SECONDS,
        "balanced_rounds": len(TRAINING_SEEDS),
        "physical_episode_count": len(TRAINING_SEEDS) * len(SCENARIOS),
        "input_checkpoint_sha256": sha256_file(input_checkpoint),
        "output_checkpoint": str(final_checkpoint.resolve()),
        "output_checkpoint_sha256": sha256_file(final_checkpoint),
        "resume_granularity": (
            "completed 1800s episode and completed balanced A/B/C round"),
    })
    return final_checkpoint


def evaluation_policies(count_model: Path, utility_model: Path):
    policies = []
    for scheduler in NON_GRAPH_POLICIES:
        policies.append({
            "policy_id": scheduler,
            "scheduler": scheduler,
            "model_path": MODEL_PATHS.get(scheduler),
        })
    policies.extend((
        {"policy_id": "GraphPPO_count", "scheduler": "GraphPPO",
         "model_path": count_model},
        {"policy_id": "GraphPPO_utility_v2", "scheduler": "GraphPPO",
         "model_path": utility_model},
    ))
    return policies


def aggregate_evaluation(
        output_dir: Path, policies,
        validation_seeds=VALIDATION_SEEDS) -> dict:
    validation_seeds = tuple(validation_seeds)
    if (not validation_seeds or len(validation_seeds) !=
            len(set(validation_seeds)) or
            any(isinstance(seed, bool) or not isinstance(seed, int) or
                seed < 0 for seed in validation_seeds)):
        raise ValueError("validation seeds must be unique nonnegative integers")
    rows = []
    for policy in policies:
        model_path = policy.get("model_path")
        expected_model_sha256 = policy.get("model_sha256")
        if "model_sha256" not in policy:
            expected_model_sha256 = (
                sha256_file(model_path) if model_path else None)
        count_values = []
        utility_values = []
        collisions = 0
        fallbacks = 0
        invalid = 0
        scheduler_timeouts = 0
        execution_totals = Counter()
        fallback_reasons = Counter()
        failure_reasons = Counter()
        for scenario in SCENARIOS:
            for seed in validation_seeds:
                canonical = (
                    output_dir / "episodes" / policy["policy_id"] /
                    scenario / str(seed) / "result.json")
                envelope = load_json(canonical)
                result = envelope["result"]
                formal = result["formal_evaluation"]
                summary = result["summary_metrics"]
                info = result["experiment_info"]
                if (info.get("scheduler") != policy["scheduler"] or
                        envelope.get("input_model_sha256") !=
                        expected_model_sha256):
                    raise ValueError(
                        "evaluation policy/model identity mismatch")
                count_values.append(float(
                    formal["count_evaluation"]["total_tasks_completed"]))
                utility_values.append(float(
                    formal["utility_v2_evaluation"]
                    ["utility_score_deadline_v2"]))
                collisions += int(summary["physical_collision_count"])
                fallbacks += int(summary.get(
                    "fallback_scheduler_commits", 0))
                invalid += int(summary.get(
                    "invalid_scheduler_outputs", 0))
                scheduler_timeouts += int(summary.get(
                    "scheduler_timeout_count", 0))
                for field in (
                        "solver_joint_solution_count",
                        "solver_proposed_pair_count",
                        "solver_unattempted_pair_count",
                        "first_assignment_attempt_count",
                        "first_assignment_commit_count",
                        "first_assignment_rejection_count"):
                    value = summary.get(field, 0)
                    if (isinstance(value, bool) or
                            not isinstance(value, int) or value < 0):
                        raise ValueError(f"{field} must be nonnegative")
                    execution_totals[field] += value
                for field, target in (
                        ("scheduler_fallback_reasons", fallback_reasons),
                        ("scheduler_failure_reasons", failure_reasons)):
                    values = summary.get(field, {})
                    if not isinstance(values, dict):
                        raise ValueError(f"{field} must be a mapping")
                    for reason, value in values.items():
                        if (isinstance(value, bool) or
                                not isinstance(value, int) or value < 0):
                            raise ValueError(
                                f"{field}[{reason}] must be nonnegative")
                        target[str(reason)] += value
        row = {
            "policy_id": policy["policy_id"],
            "scheduler": policy["scheduler"],
            "training_objective": policy.get("training_objective"),
            "model_path": (str(model_path.resolve())
                           if model_path else None),
            "model_sha256": expected_model_sha256,
            "model_identity_sha256": policy.get(
                "model_identity_sha256"),
            "evaluation_metrics": ["count", "utility_v2"],
            "run_count": len(count_values),
            "count_mean": statistics.fmean(count_values),
            "count_sum": sum(count_values),
            "utility_v2_mean": statistics.fmean(utility_values),
            "utility_v2_min": min(utility_values),
            "physical_collision_count": collisions,
            "fallback_scheduler_commits": fallbacks,
            "invalid_scheduler_outputs": invalid,
            "scheduler_timeout_count": scheduler_timeouts,
            **dict(execution_totals),
            "scheduler_fallback_reasons": dict(sorted(
                fallback_reasons.items())),
            "scheduler_failure_reasons": dict(sorted(
                failure_reasons.items())),
        }
        row.update(pure_policy_eligibility(row))
        rows.append(row)
    pure_rows = [row for row in rows if row["pure_policy_eligible"]]
    # Baselines are valid comparators in both systems.  A trainable model is
    # ranked only under the objective that produced its checkpoint, so Count
    # and Utility models can never be silently mixed while every physical
    # episode still records both metrics for auditability.
    count_rows = [
        row for row in pure_rows
        if row["training_objective"] in {None, "count"}
    ]
    utility_rows = [
        row for row in pure_rows
        if row["training_objective"] in {None, "utility_v2"}
    ]
    count_ranking = sorted(
        count_rows, key=lambda row: (-row["count_mean"],
                                     row["physical_collision_count"],
                                     row["policy_id"]))
    utility_ranking = sorted(
        utility_rows, key=lambda row: (-row["utility_v2_mean"],
                                       row["physical_collision_count"],
                                       row["policy_id"]))
    return {
        "schema_version": 2,
        "status": "complete",
        "runtime_mode": "webots",
        "duration_seconds_per_run": DURATION_SECONDS,
        "scenarios": list(SCENARIOS),
        "validation_seeds": list(validation_seeds),
        "policy_count": len(policies),
        "run_count": len(policies) * len(SCENARIOS) * len(validation_seeds),
        "policies": rows,
        "pure_policy_count": len(pure_rows),
        "objective_ranking_contract": (
            "baselines_in_both; trainable_models_in_own_objective_only"),
        "excluded_from_pure_ranking": {
            row["policy_id"]: row["pure_policy_exclusion_reasons"]
            for row in rows if not row["pure_policy_eligible"]
        },
        "count_ranking": [row["policy_id"] for row in count_ranking],
        "utility_v2_ranking": [
            row["policy_id"] for row in utility_ranking],
    }


def evaluate_all(
        output_dir: Path, webots: Path, count_model: Path,
        utility_model: Path, update_state, max_runs: int | None) -> bool:
    policies = evaluation_policies(count_model, utility_model)
    executed = 0
    for policy in policies:
        for scenario in SCENARIOS:
            for seed in VALIDATION_SEEDS:
                canonical = (
                    output_dir / "episodes" / policy["policy_id"] /
                    scenario / str(seed) / "result.json")
                update_state(
                    stage="evaluate_all",
                    current={"policy_id": policy["policy_id"],
                             "scenario": scenario, "seed": seed})
                envelope = load_json(canonical)
                model_path = policy["model_path"]
                model_hash = sha256_file(model_path) if model_path else None
                if valid_episode(
                        envelope, scenario=scenario, seed=seed,
                        manifest=manifest_path("validation", scenario, seed),
                        input_model_sha256=model_hash,
                        fine_tune_objective=None):
                    print(
                        f"[RESUME] evaluation/{policy['policy_id']}/"
                        f"{scenario}/{seed}", flush=True)
                    continue
                if max_runs is not None and executed >= max_runs:
                    return False
                run_or_resume_episode(
                    canonical_path=canonical, webots=webots,
                    scenario=scenario, seed=seed,
                    manifest=manifest_path("validation", scenario, seed),
                    scheduler=policy["scheduler"], model_path=model_path,
                    fine_tune_objective=None)
                executed += 1
    atomic_json(
        output_dir / "all_schedulers_evaluation.json",
        aggregate_evaluation(output_dir, policies))
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe GraphPPO Webots fine tuning and 1800s evaluation "
            "of every registered scheduler."))
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "webots_1800_formal")
    parser.add_argument(
        "--webots", type=Path,
        default=Path(r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"))
    parser.add_argument(
        "--phase", choices=("all", "fine-tune", "evaluate"),
        default="all")
    parser.add_argument(
        "--max-evaluation-runs", type=int,
        help="Testing aid: stop cleanly after this many new evaluation runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_evaluation_runs is not None and args.max_evaluation_runs < 0:
        raise ValueError("--max-evaluation-runs must be nonnegative")
    validate_sources()
    validate_webots(args.webots)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "workflow_state.json"
    state = load_json(state_path) or {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_seconds": DURATION_SECONDS,
        "resume_contract": {
            "completed_episode": "atomic result.json; skipped on restart",
            "incomplete_episode": "restart current 1800s episode",
            "fine_tune_checkpoint": "after every balanced A/B/C round",
        },
    }

    def update_state(**values):
        for stale_key in (
                "error", "failed_at", "interrupted_at", "completed_at"):
            state.pop(stale_key, None)
        state.update(values)
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        state["status"] = "running"
        atomic_json(state_path, state)

    count_model = (
        output_dir / "fine_tune" / "count" /
        "webots_finetuned_model.npz")
    utility_model = (
        output_dir / "fine_tune" / "utility_v2" /
        "webots_finetuned_model.npz")
    try:
        update_state(stage="preflight", current=None)
        if args.phase in {"all", "fine-tune"}:
            count_model = fine_tune_objective(
                "count", COUNT_INPUT,
                output_dir / "fine_tune" / "count", args.webots,
                update_state)
            utility_model = fine_tune_objective(
                "utility_v2", UTILITY_INPUT,
                output_dir / "fine_tune" / "utility_v2", args.webots,
                update_state)
        if args.phase in {"all", "evaluate"}:
            if not count_model.is_file() or not utility_model.is_file():
                raise FileNotFoundError(
                    "evaluation requires both Webots-fine-tuned checkpoints")
            complete = evaluate_all(
                output_dir / "evaluation", args.webots,
                count_model, utility_model, update_state,
                args.max_evaluation_runs)
            if not complete:
                update_state(stage="evaluate_all", current=None)
                state["status"] = "stopped_at_requested_run_limit"
                atomic_json(state_path, state)
                return 0
        state.update({
            "status": "complete", "stage": "complete", "current": None,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        atomic_json(state_path, state)
        print(f"[COMPLETE] report root: {output_dir}", flush=True)
        return 0
    except KeyboardInterrupt:
        state.update({
            "status": "interrupted",
            "interrupted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        atomic_json(state_path, state)
        print(
            "[INTERRUPTED] restart the same command; completed episodes and "
            "balanced-round checkpoints will be reused.", flush=True)
        return 130
    except Exception as exc:
        state.update({
            "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
