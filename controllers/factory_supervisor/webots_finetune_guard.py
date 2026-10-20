"""Paired Webots pre/post fine-tune regression gate with atomic rollback choice."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


PREVIOUS_WEBOTS_GUARD_VERSION = (
    "webots-finetune-bounded-regression-v8-timeout-fallback-audit-only-"
    "five-percent-relative-two-seed-ten-round-maximum")
PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE = 0.05
WEBOTS_GUARD_VERSION = (
    "webots-finetune-bounded-regression-v9-timeout-fallback-audit-only-"
    "ten-percent-relative-two-seed-ten-round-maximum")
WEBOTS_REGRESSION_RUNTIME_EVENT_POLICY = (
    "timeout_fallback_audit_only_invalid_output_blocking")
_RUNTIME_EVENT_FIELDS = (
    "fallback_scheduler_commits",
    "invalid_scheduler_outputs",
    "scheduler_timeout_count",
)
SCENARIOS = ("A", "B", "C")
# A trained checkpoint may fluctuate below its immutable pre-training
# baseline, but neither its aggregate metric nor any individual scenario may
# lose more than ten percent.  Use the same relative budget for both
# objectives so "10% of baseline" has one unambiguous meaning everywhere.
COUNT_RELATIVE_DEGRADATION_TOLERANCE = 0.10
UTILITY_RELATIVE_DEGRADATION_TOLERANCE = 0.10
# Convergence is evaluated after every balanced A/B/C round.  This is a
# protocol floor rather than a fixed training quota; the patience window
# below still requires three consecutive stable updates before stopping.
MIN_WEBOTS_FINETUNE_BALANCED_ROUNDS = 1
MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS = 10
WEBOTS_FINETUNE_CONVERGENCE_PATIENCE = 3
WEBOTS_FINETUNE_PARAMETER_CHANGE_THRESHOLD = 1e-2
WEBOTS_FINETUNE_SEEDS = tuple(
    range(41000, 41000 + MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS))
WEBOTS_REGRESSION_SEEDS = tuple(range(51000, 51002))
FINAL_EVALUATION_SEEDS = tuple(range(61000, 61006))


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


def _runtime_event_counts(row: Mapping) -> dict[str, int]:
    """Validate and return recoverable scheduler events for one episode.

    Physical rollouts can finish through the registered fallback path after a
    scheduler timeout.  Those recoverable events remain auditable, but they do
    not invalidate otherwise complete paired evidence: the bounded metric
    comparison below is the checkpoint-selection guard.  Invalid scheduler
    outputs still fail closed because they indicate malformed policy output.
    """
    counts = {}
    for field in _RUNTIME_EVENT_FIELDS:
        value = row.get(field)
        if (isinstance(value, bool) or not isinstance(value, int) or
                value < 0):
            raise ValueError(
                f"invalid Webots regression runtime event count: {field}")
        counts[field] = value
    expected_pure = not any(counts.values())
    if row.get("pure_policy_eligible") is not expected_pure:
        raise ValueError(
            "Webots regression pure-policy audit is inconsistent")
    return counts


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
            row.get("checkpoint_sha256") != checkpoint_sha256 or
            not _is_sha256(row.get("task_manifest_sha256"))):
        raise ValueError("Webots regression episode failed strict gates")
    runtime_events = _runtime_event_counts(row)
    if runtime_events["invalid_scheduler_outputs"] != 0:
        raise ValueError(
            "Webots regression episode has invalid scheduler outputs")
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
            f"Webots regression set must cover A/B/C x "
            f"{len(WEBOTS_REGRESSION_SEEDS)} seeds")
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


def _runtime_event_audit(indexed: Mapping) -> dict:
    totals = {field: 0 for field in _RUNTIME_EVENT_FIELDS}
    episodes = []
    for scenario in SCENARIOS:
        for seed in WEBOTS_REGRESSION_SEEDS:
            row = indexed[(scenario, seed)]
            counts = _runtime_event_counts(row)
            for field, value in counts.items():
                totals[field] += value
            episodes.append({
                "scenario": scenario,
                "seed": seed,
                "pure_policy_eligible": row["pure_policy_eligible"],
                **counts,
            })
    return {
        "episode_count": len(episodes),
        "pure_episode_count": sum(
            int(row["pure_policy_eligible"]) for row in episodes),
        "recovered_episode_count": sum(
            int(not row["pure_policy_eligible"]) for row in episodes),
        "totals": totals,
        "episodes": episodes,
    }


def webots_finetune_regression_decision(
        objective: str, pre_checkpoint: Path, post_checkpoint: Path,
        pre_episodes, post_episodes, *, policy_id: str | None = None,
        scheduler: str | None = None,
        degradation_tolerance: float | None = None) -> dict:
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
    pre_runtime_audit = _runtime_event_audit(pre)
    post_runtime_audit = _runtime_event_audit(post)
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
    scenario_pre_means = {
        scenario: float(np.mean([
            row["pre"] for row in pairs if row["scenario"] == scenario]))
        for scenario in SCENARIOS}
    scenario_differences = {
        scenario: float(np.mean([
            row["difference"] for row in pairs
            if row["scenario"] == scenario]))
        for scenario in SCENARIOS}
    mean_difference = float(np.mean(differences))
    configured_tolerance = (
        COUNT_RELATIVE_DEGRADATION_TOLERANCE
        if objective == "count" else
        UTILITY_RELATIVE_DEGRADATION_TOLERANCE)
    if degradation_tolerance is None:
        degradation_tolerance = configured_tolerance
    if (isinstance(degradation_tolerance, bool) or
            not isinstance(degradation_tolerance, (int, float)) or
            not math.isfinite(float(degradation_tolerance)) or
            float(degradation_tolerance) not in {
                PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE,
                configured_tolerance,
            }):
        raise ValueError("unsupported Webots degradation tolerance")
    degradation_tolerance = float(degradation_tolerance)
    degradation_tolerance_kind = "relative"

    def allowed_drop(reference: float) -> float:
        return degradation_tolerance*abs(reference)

    mean_difference_floor = -allowed_drop(pre_mean)
    scenario_difference_floors = {
        scenario: -allowed_drop(scenario_pre_means[scenario])
        for scenario in SCENARIOS}
    # Small paired fluctuations are normal in physical simulation. Accept a
    # fine-tuned model while both its observed result and conservative
    # bootstrap bound remain inside the registered degradation budget; any
    # material regression atomically selects the frozen pre-tune checkpoint.
    accepted = bool(
        mean_difference >= mean_difference_floor and
        lower_bound >= mean_difference_floor and
        all(scenario_differences[scenario] >=
            scenario_difference_floors[scenario]
            for scenario in SCENARIOS))
    selected_path = post_checkpoint if accepted else pre_checkpoint
    selected_hash = post_hash if accepted else pre_hash
    report = {
        "version": (
            WEBOTS_GUARD_VERSION
            if degradation_tolerance == configured_tolerance else
            PREVIOUS_WEBOTS_GUARD_VERSION),
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
        "degradation_tolerance": degradation_tolerance,
        "degradation_tolerance_kind": degradation_tolerance_kind,
        "strict_improvement_required": False,
        "runtime_event_policy": WEBOTS_REGRESSION_RUNTIME_EVENT_POLICY,
        "timeout_fallback_affect_checkpoint_selection": False,
        "invalid_scheduler_outputs_block_selection": True,
        "runtime_event_audit": {
            "pre": pre_runtime_audit,
            "post": post_runtime_audit,
            "post_minus_pre_totals": {
                field: (post_runtime_audit["totals"][field] -
                        pre_runtime_audit["totals"][field])
                for field in _RUNTIME_EVENT_FIELDS
            },
        },
        "mean_difference_floor": mean_difference_floor,
        "scenario_difference_floors": scenario_difference_floors,
        "scenario_mean_differences": scenario_differences,
        "accepted": accepted,
        "decision": "accept_finetuned" if accepted else "rollback_to_pre",
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": selected_hash,
        "pairs": pairs,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report
