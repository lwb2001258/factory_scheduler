"""Paired Webots pre/post fine-tune regression gate with atomic rollback choice."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


WEBOTS_GUARD_VERSION = "webots-finetune-strict-improvement-v3-dual-objective"
SCENARIOS = ("A", "B", "C")
MIN_WEBOTS_FINETUNE_BALANCED_ROUNDS = 7
MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS = 21
WEBOTS_FINETUNE_CONVERGENCE_PATIENCE = 3
WEBOTS_FINETUNE_PARAMETER_CHANGE_THRESHOLD = 1e-2
WEBOTS_FINETUNE_SEEDS = tuple(
    range(41000, 41000 + MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS))
WEBOTS_REGRESSION_SEEDS = tuple(range(51000, 51007))
FINAL_EVALUATION_SEEDS = tuple(range(61000, 61010))


def _canonical_sha256(document) -> str:
    payload = json.dumps(
        document, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def _metric(objective: str, row: Mapping):
    if objective == "count":
        value = row.get("completed_tasks")
        valid = (not isinstance(value, bool) and isinstance(value, int) and
                 value >= 0)
    else:
        value = row.get("utility_score_deadline_v2")
        valid = (not isinstance(value, bool) and
                 isinstance(value, (int, float)) and math.isfinite(value))
    if not valid:
        raise ValueError("invalid Webots regression metric")
    return float(value)


def _validate_episode(row, *, objective: str, checkpoint_sha256: str,
                      expected_key, policy_id: str | None = None,
                      scheduler: str | None = None) -> None:
    if not isinstance(row, Mapping):
        raise ValueError("Webots regression episode must be a mapping")
    scenario, seed = expected_key
    duration, current_time = (
        row.get("duration_seconds"), row.get("current_time"))
    if (row.get("scenario") != scenario or row.get("seed") != seed or
            row.get("partition") != "webots_regression" or
            row.get("objective") != objective or
            (policy_id is not None and row.get("policy_id") != policy_id) or
            (scheduler is not None and row.get("scheduler") != scheduler) or
            row.get("runtime_mode") != "webots" or
            row.get("physics_fidelity") != "webots_physical" or
            isinstance(duration, bool) or
            not isinstance(duration, (int, float)) or
            not math.isfinite(duration) or duration != 1800.0 or
            isinstance(current_time, bool) or
            not isinstance(current_time, (int, float)) or
            not math.isfinite(current_time) or current_time != 1800.0 or
            row.get("fixed_horizon") is not True or
            row.get("horizon_finalized") is not True or
            row.get("termination_reason") != "episode_horizon" or
            row.get("physical_collision_observation_status") != "observed" or
            row.get("pure_policy_eligible") is not True or
            row.get("fallback_scheduler_commits") != 0 or
            row.get("invalid_scheduler_outputs") != 0 or
            row.get("scheduler_timeout_count") != 0 or
            row.get("checkpoint_sha256") != checkpoint_sha256 or
            not _is_sha256(row.get("task_manifest_sha256"))):
        raise ValueError("Webots regression episode failed strict gates")
    _metric(objective, row)


def _index_episodes(rows, *, objective, checkpoint_sha256,
                    policy_id=None, scheduler=None):
    try:
        values = list(rows)
    except TypeError as exc:
        raise ValueError("Webots regression episodes must be iterable") from exc
    expected = [(scenario, seed) for scenario in SCENARIOS
                for seed in WEBOTS_REGRESSION_SEEDS]
    indexed = {}
    for row in values:
        if not isinstance(row, Mapping):
            raise ValueError("Webots regression episode must be a mapping")
        key = (row.get("scenario"), row.get("seed"))
        if key in indexed:
            raise ValueError("duplicate Webots regression episode")
        indexed[key] = row
    if set(indexed) != set(expected):
        raise ValueError(
            "Webots regression set must cover A/B/C x seven seeds")
    for key in expected:
        _validate_episode(
            indexed[key], objective=objective,
            checkpoint_sha256=checkpoint_sha256, expected_key=key,
            policy_id=policy_id, scheduler=scheduler)
    return indexed


def _bootstrap_lower_bound(differences, *, seed: int,
                           samples: int = 20_000) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("paired differences are invalid")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.05, method="linear"))


def webots_finetune_regression_decision(
        objective: str, pre_checkpoint: Path, post_checkpoint: Path,
        pre_episodes, post_episodes, *, policy_id: str | None = None,
        scheduler: str | None = None) -> dict:
    """Accept post checkpoint only when paired physical results do not regress."""
    if objective not in {"count", "utility_v2"}:
        raise ValueError("objective must be count or utility_v2")
    if ((policy_id is None) != (scheduler is None) or
            (policy_id is not None and (
                not isinstance(policy_id, str) or not policy_id or
                not isinstance(scheduler, str) or not scheduler or
                policy_id != f"{scheduler}_{objective}"))):
        raise ValueError("policy identity does not match objective")
    pre_checkpoint, post_checkpoint = (
        Path(pre_checkpoint).resolve(), Path(post_checkpoint).resolve())
    if not pre_checkpoint.is_file() or not post_checkpoint.is_file():
        raise ValueError("pre/post checkpoint is missing")
    pre_hash, post_hash = (
        file_sha256(pre_checkpoint), file_sha256(post_checkpoint))
    pre = _index_episodes(
        pre_episodes, objective=objective, checkpoint_sha256=pre_hash,
        policy_id=policy_id, scheduler=scheduler)
    post = _index_episodes(
        post_episodes, objective=objective, checkpoint_sha256=post_hash,
        policy_id=policy_id, scheduler=scheduler)
    keys = [(scenario, seed) for scenario in SCENARIOS
            for seed in WEBOTS_REGRESSION_SEEDS]
    pairs = []
    for scenario, seed in keys:
        before, after = (
            _metric(objective, pre[(scenario, seed)]),
            _metric(objective, post[(scenario, seed)]))
        pairs.append({
            "scenario": scenario, "seed": seed,
            "pre": before, "post": after, "difference": after-before,
        })
    pre_mean = float(np.mean([row["pre"] for row in pairs]))
    differences = [row["difference"] for row in pairs]
    seed = int(_canonical_sha256({
        "pre": pre_hash, "post": post_hash, "objective": objective,
    })[:16], 16) % (2**32)
    lower_bound = _bootstrap_lower_bound(differences, seed=seed)
    scenario_differences = {
        scenario: float(np.mean([
            row["difference"] for row in pairs
            if row["scenario"] == scenario]))
        for scenario in SCENARIOS}
    mean_difference = float(np.mean(differences))
    # The formal contract is intentionally strict: fine-tuning must produce
    # an observed improvement, its one-sided 95% bootstrap bound may not be
    # negative, and no individual scenario mean may regress.  Otherwise the
    # frozen standalone checkpoint is selected atomically.
    accepted = bool(
        mean_difference > 0.0 and
        lower_bound >= 0.0 and
        all(value >= 0.0 for value in scenario_differences.values()))
    selected_path = post_checkpoint if accepted else pre_checkpoint
    selected_hash = post_hash if accepted else pre_hash
    report = {
        "version": WEBOTS_GUARD_VERSION,
        "policy_id": policy_id,
        "scheduler": scheduler,
        "objective": objective,
        "metric": ("completed_tasks" if objective == "count" else
                   "utility_score_deadline_v2"),
        "duration_seconds": 1800.0,
        "regression_seeds": list(WEBOTS_REGRESSION_SEEDS),
        "paired_run_count": len(pairs),
        "pre_checkpoint": str(pre_checkpoint),
        "pre_checkpoint_sha256": pre_hash,
        "post_checkpoint": str(post_checkpoint),
        "post_checkpoint_sha256": post_hash,
        "pre_metric_mean": pre_mean,
        "post_metric_mean": float(np.mean([row["post"] for row in pairs])),
        "mean_paired_difference": mean_difference,
        "one_sided_bootstrap_95_lower_bound": lower_bound,
        "degradation_tolerance": 0.0,
        "strict_improvement_required": True,
        "scenario_mean_differences": scenario_differences,
        "accepted": accepted,
        "decision": "accept_finetuned" if accepted else "rollback_to_pre",
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": selected_hash,
        "pairs": pairs,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report
