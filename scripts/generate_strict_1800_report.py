"""Generate the final strict-1800 training and evaluation report.

The formal report is deliberately fail-closed: it is written only after the
workflow has committed a complete state, all expected training/fine-tuning
reports are present, and every registered Webots evaluation episode can be read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_DIR = ROOT / "controllers" / "factory_supervisor"
if str(SUPERVISOR_DIR) not in sys.path:
    sys.path.insert(0, str(SUPERVISOR_DIR))

from evaluation_purity import pure_policy_eligibility  # noqa: E402
from learning_scheduler import WEBOTS_SHARED_SUPERVISION_VERSION  # noqa: E402
from all_ai_auto_optimizer import (  # noqa: E402
    MAX_STANDALONE_BALANCED_ROUNDS, MIN_STANDALONE_BALANCED_ROUNDS,
    STANDALONE_TRAINING_SEEDS,
)
from webots_finetune_guard import (  # noqa: E402
    COUNT_RELATIVE_DEGRADATION_TOLERANCE,
    FINAL_EVALUATION_SEEDS, MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS,
    MIN_WEBOTS_FINETUNE_BALANCED_ROUNDS,
    PREVIOUS_WEBOTS_GUARD_VERSION,
    PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE,
    WEBOTS_GUARD_VERSION,
    WEBOTS_FINETUNE_CONVERGENCE_PATIENCE,
    WEBOTS_FINETUNE_PARAMETER_CHANGE_THRESHOLD, WEBOTS_FINETUNE_SEEDS,
    WEBOTS_REGRESSION_RUNTIME_EVENT_POLICY, WEBOTS_REGRESSION_SEEDS,
    UTILITY_RELATIVE_DEGRADATION_TOLERANCE,
)

DEFAULT_RESULTS = (
    ROOT / "results" / "all_scheduler_model_remediation" /
    "formal_1800_dual_v6")
DEFAULT_OUTPUT = (
    ROOT / "docs" /
    "all_scheduler_model_remediation_training_report_v6.md")

SCENARIOS = ("A", "B", "C")
TRAINING_SEEDS = STANDALONE_TRAINING_SEEDS
MIN_STANDALONE_ROUNDS = MIN_STANDALONE_BALANCED_ROUNDS
MAX_STANDALONE_ROUNDS = MAX_STANDALONE_BALANCED_ROUNDS
FINETUNE_SEEDS = WEBOTS_FINETUNE_SEEDS
REGRESSION_SEEDS = WEBOTS_REGRESSION_SEEDS
VALIDATION_SEEDS = FINAL_EVALUATION_SEEDS
TRAINED_POLICIES = (
    "LearnedHungarian_count",
    "LearnedHungarian_utility_v2",
    "GraphImitation_count",
    "GraphImitation_utility_v2",
    "GraphPPO_count",
    "GraphPPO_utility_v2",
    "RainbowDQN_count",
    "RainbowDQN_utility_v2",
    "QRDQN_count",
    "QRDQN_utility_v2",
    "CQL_count",
    "CQL_utility_v2",
    "LinUCB_count",
    "LinUCB_utility_v2",
    "PPO_RL_count",
    "PPO_RL_utility_v2",
    "SARSA_count",
    "SARSA_utility_v2",
    "DQN_count",
    "DQN_utility_v2",
)
STATIC_LEARNED_POLICIES = ()
TRADITIONAL_POLICIES = (
    "FCFS",
    "NearestNeighbour",
)
EXPECTED_EVALUATION_POLICIES = 22
EXPECTED_EVALUATION_RUNS = (
    EXPECTED_EVALUATION_POLICIES * len(SCENARIOS) * len(VALIDATION_SEEDS))
HORIZON_SECONDS = 1800.0
ROUND_RECORD_RE = re.compile(r"after_round_\d{2}\.json$")
EXPECTED_WORKFLOW_CONTRACT = (
    "dual-objective-training-v25-unfinetuned-ten-percent-webots-regression-"
    "six-seed-evaluation-nn-fcfs")
EXPECTED_WORKFLOW_STATE_SCHEMA = 24
EXPECTED_TRAINING_SEMANTICS = (
    "task-lifecycle-attribution-v6-convergence-bounded-regression")
EXPECTED_TRAINING_RECORD_SCHEMA = 7
EXPECTED_MODEL_IDENTITY_VERSION = (
    "dual-objective-model-identity-v8-convergence-bounded-regression")
EXPECTED_EVALUATION_IDENTITY_VERSION = (
    "dual-metric-evaluation-identity-v2-common-seeds")


class ReportDataError(RuntimeError):
    """Raised when a formal report input is missing or inconsistent."""


def trained_policy_identity(policy_id: str) -> tuple[str, str]:
    for objective in ("count", "utility_v2"):
        suffix = f"_{objective}"
        if policy_id.endswith(suffix):
            return policy_id[:-len(suffix)], objective
    raise ReportDataError(f"invalid trained policy identity: {policy_id}")


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportDataError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ReportDataError(f"JSON artifact is not an object: {path}")
    return document


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(document) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def fmt_time(value: str | None) -> str:
    parsed = parse_time(value)
    return parsed.strftime("%Y-%m-%d %H:%M:%S %z") if parsed else "—"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "—"
    seconds = max(0, int(round(seconds)))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days} 天")
    if hours or days:
        parts.append(f"{hours} 小时")
    if minutes or hours or days:
        parts.append(f"{minutes} 分钟")
    if not parts:
        parts.append(f"{seconds} 秒")
    return " ".join(parts)


def fmt_number(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "—"
    if numeric == int(numeric):
        return str(int(numeric))
    return f"{numeric:.{digits}f}"


def mean(values) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.fmean(numbers) if numbers else None


def sample_std(values) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.stdev(numbers) if len(numbers) > 1 else 0.0 if numbers else None


def md_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers: list[str], rows: list[list[object]]) -> list[str]:
    rendered = ["| " + " | ".join(md_cell(item) for item in headers) + " |"]
    rendered.append("| " + " | ".join("---" for _ in headers) + " |")
    rendered.extend(
        "| " + " | ".join(md_cell(item) for item in row) + " |"
        for row in rows
    )
    return rendered


def policy_category(policy_id: str) -> str:
    if policy_id in TRAINED_POLICIES:
        return "本轮训练 AI"
    if policy_id in STATIC_LEARNED_POLICIES:
        return "既有学习模型"
    if policy_id in TRADITIONAL_POLICIES:
        return "非 AI/优化基线"
    return "其他"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportDataError(message)


def collect_headless(results_dir: Path) -> tuple[list[dict], list[datetime]]:
    policies = []
    timestamps = []
    for policy_id in TRAINED_POLICIES:
        expected_scheduler, expected_objective = trained_policy_identity(
            policy_id)
        policy_dir = results_dir / "headless" / policy_id
        report = load_json(policy_dir / "training_report.json")
        readiness = load_json(policy_dir / "standalone_readiness.json")
        convergence_complete = bool(
            readiness.get("validation_plateau") is True and
            readiness.get("parameter_converged") is True)
        maximum_complete = bool(
            readiness.get("status") ==
                "maximum_completed_for_webots_fine_tune" and
            readiness.get("maximum_rounds_reached") is True and
            readiness.get("completed_balanced_rounds") ==
                MAX_STANDALONE_ROUNDS and
            readiness.get("standalone_completion_reason") ==
                "maximum_balanced_rounds_reached" and
            report.get("standalone_exit_mode") ==
                "maximum_balanced_rounds_reached")
        operator_complete = bool(
            readiness.get("standalone_exit_mode") ==
                "operator_requested_webots_transition" and
            report.get("standalone_exit_mode") ==
                "operator_requested_webots_transition")
        require(
            report.get("status") == "complete" and
            report.get("training_contract") ==
                "standalone_sufficient_training" and
            report.get("policy_id") == policy_id and
            report.get("scheduler") == expected_scheduler and
            report.get("objective") == expected_objective and
            readiness.get("policy_id") == policy_id and
            readiness.get("objective") == expected_objective and
            report.get("webots_fine_tune_eligible") is True and
            readiness.get("webots_fine_tune_eligible") is True and
            (convergence_complete or maximum_complete or operator_complete),
            f"standalone report is not complete/eligible: {policy_id}")
        checkpoint_value = report.get("output_checkpoint")
        require(isinstance(checkpoint_value, str) and bool(checkpoint_value),
                f"standalone checkpoint path is missing: {policy_id}")
        checkpoint = Path(checkpoint_value)
        identity_path = checkpoint.with_name(
            checkpoint.name + ".identity.json")
        identity = load_json(identity_path)
        require(
            checkpoint.is_file() and identity_path.is_file() and
            file_sha256(checkpoint) ==
                report.get("output_checkpoint_sha256") and
            file_sha256(identity_path) ==
                report.get("output_model_identity_sha256") and
            identity.get("version") == EXPECTED_MODEL_IDENTITY_VERSION and
            identity.get("training_semantics_version") ==
                EXPECTED_TRAINING_SEMANTICS and
            identity.get("policy_id") == policy_id and
            identity.get("scheduler") == expected_scheduler and
            identity.get("objective") == expected_objective,
            f"standalone checkpoint identity mismatch: {policy_id}")
        records = [load_json(path) for path in sorted(
            (policy_dir / "episodes").glob("*/result.json"))]
        rounds = report.get("balanced_rounds")
        require(
            isinstance(rounds, int) and
            MIN_STANDALONE_ROUNDS <= rounds <= MAX_STANDALONE_ROUNDS and
            len(records) == rounds*len(SCENARIOS),
            f"invalid sufficient standalone coverage for {policy_id}: "
            f"rounds={rounds}, episodes={len(records)}")
        observed = {(row.get("scenario"), row.get("seed")) for row in records}
        expected = {(scenario, seed) for seed in TRAINING_SEEDS[:rounds]
                    for scenario in SCENARIOS}
        require(observed == expected,
                f"standalone scenario/seed coverage mismatch: {policy_id}")
        for row in records:
            runtime = row.get("metrics", {}).get("runtime", {})
            require(
                row.get("schema_version") ==
                    EXPECTED_TRAINING_RECORD_SCHEMA
                and row.get("training_semantics_version") ==
                    EXPECTED_TRAINING_SEMANTICS
                and row.get("duration_seconds") == HORIZON_SECONDS
                and runtime.get("current_time") == HORIZON_SECONDS
                and runtime.get("fixed_horizon") is True
                and runtime.get("horizon_finalized") is True
                and runtime.get("termination_reason") == "episode_horizon",
                f"standalone horizon validation failed: {policy_id}/"
                f"{row.get('scenario')}/{row.get('seed')}",
            )
            parsed = parse_time(row.get("completed_at"))
            if parsed:
                timestamps.append(parsed)

        metrics = [row["metrics"] for row in records]
        returns = [row["episode_return"] for row in metrics]
        completed = [row["runtime"]["tasks_completed"] for row in metrics]
        losses = [row.get("mean_loss") for row in metrics
                  if row.get("mean_loss") is not None]
        scenario_rows = {}
        for scenario in SCENARIOS:
            selected = [row for row in records if row["scenario"] == scenario]
            scenario_rows[scenario] = {
                "return_mean": mean(row["metrics"]["episode_return"]
                                    for row in selected),
                "tasks_mean": mean(row["metrics"]["runtime"]["tasks_completed"]
                                   for row in selected),
            }
        policies.append({
            "policy_id": policy_id,
            "scheduler": report["scheduler"],
            "objective": report["objective"],
            "episode_count": len(records),
            "return_mean": mean(returns),
            "return_std": sample_std(returns),
            "tasks_mean": mean(completed),
            "decisions": sum(int(row.get("decisions", 0)) for row in metrics),
            "optimizer_updates": sum(
                int(row.get("optimizer_updates", 0)) for row in metrics),
            "mean_loss": mean(losses),
            "transition_count": sum(
                int(row.get("transition_count", 0)) for row in metrics),
            "checkpoint": report["output_checkpoint"],
            "checkpoint_sha256": report["output_checkpoint_sha256"],
            "scenario_rows": scenario_rows,
        })
    return policies, timestamps


def collect_fine_tune(results_dir: Path) -> tuple[list[dict], list[datetime]]:
    state = load_json(results_dir / "workflow_state.json")
    tolerance_registry = state.get(
        "webots_finetune_regression_tolerance_by_policy")
    legacy_policy_ids = state.get(
        "webots_finetune_legacy_tolerance_policy_ids")
    require(
        isinstance(tolerance_registry, dict) and
        set(tolerance_registry) == set(TRAINED_POLICIES) and
        isinstance(legacy_policy_ids, list) and
        len(legacy_policy_ids) == len(set(legacy_policy_ids)) and
        set(legacy_policy_ids).issubset(TRAINED_POLICIES) and
        tolerance_registry == {
            policy_id: (
                PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE
                if policy_id in legacy_policy_ids else
                COUNT_RELATIVE_DEGRADATION_TOLERANCE)
            for policy_id in TRAINED_POLICIES},
        "workflow Webots tolerance registry is invalid")
    policies = []
    timestamps = []
    for policy_id in TRAINED_POLICIES:
        expected_scheduler, expected_objective = trained_policy_identity(
            policy_id)
        policy_dir = results_dir / "webots_fine_tune" / policy_id
        report = load_json(policy_dir / "fine_tune_report.json")
        decision = load_json(policy_dir / "nonregression_decision.json")
        require(
                report.get("status") == "complete" and
                report.get("policy_id") == policy_id and
                report.get("scheduler") == expected_scheduler and
                report.get("objective") == expected_objective and
                decision.get("policy_id") == policy_id and
                decision.get("scheduler") == expected_scheduler and
                decision.get("objective") == expected_objective,
                f"fine-tune report is not complete: {policy_id}")
        completed_rounds = report.get("balanced_rounds")
        convergence = report.get("parameter_convergence", {})
        recent_changes = convergence.get("recent_relative_l2_changes", [])
        common_convergence_evidence = bool(
            isinstance(convergence, dict) and
            convergence.get("completed_balanced_rounds") ==
                completed_rounds and
            convergence.get("minimum_balanced_rounds") ==
                MIN_WEBOTS_FINETUNE_BALANCED_ROUNDS and
            convergence.get("maximum_balanced_rounds") ==
                MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS and
            convergence.get("patience_rounds") ==
                WEBOTS_FINETUNE_CONVERGENCE_PATIENCE and
            convergence.get("relative_l2_change_threshold") ==
                WEBOTS_FINETUNE_PARAMETER_CHANGE_THRESHOLD and
            isinstance(recent_changes, list) and
            len(recent_changes) == WEBOTS_FINETUNE_CONVERGENCE_PATIENCE and
            all(not isinstance(value, bool) and
                isinstance(value, (int, float)) and math.isfinite(value) and
                value >= 0 for value in recent_changes))
        convergence_completion = bool(
            common_convergence_evidence and
            convergence.get("converged") is True and
            report.get("parameter_converged") is True and
            report.get("maximum_rounds_reached") ==
                (completed_rounds == MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS) and
            report.get("completion_reason") == "parameter_converged" and
            all(value <= WEBOTS_FINETUNE_PARAMETER_CHANGE_THRESHOLD
                for value in recent_changes))
        maximum_completion = bool(
            common_convergence_evidence and
            completed_rounds == MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS and
            convergence.get("converged") is False and
            report.get("parameter_converged") is False and
            report.get("maximum_rounds_reached") is True and
            report.get("completion_reason") ==
                "maximum_balanced_rounds_reached")
        require(
            isinstance(completed_rounds, int) and
            not isinstance(completed_rounds, bool) and
            max(MIN_WEBOTS_FINETUNE_BALANCED_ROUNDS,
                WEBOTS_FINETUNE_CONVERGENCE_PATIENCE) <= completed_rounds <=
                MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS and
            report.get("physical_episode_count") ==
                len(SCENARIOS)*completed_rounds and
            report.get("fine_tune_seeds") ==
                list(FINETUNE_SEEDS[:completed_rounds]) and
            (report.get("direct_final_evaluation_eligible") is True or
             report.get("nonregression_eligible") is True) and
            report.get("final_evaluation_seed_count") ==
                len(VALIDATION_SEEDS) and
            report.get("final_evaluation_seeds") ==
                list(VALIDATION_SEEDS) and
            (convergence_completion or maximum_completion),
            f"fine-tune convergence evidence is incomplete: {policy_id}")
        selected_value = decision.get("selected_checkpoint")
        require(isinstance(selected_value, str) and bool(selected_value),
                f"selected checkpoint path is missing: {policy_id}")
        selected = Path(selected_value)
        selected_identity = selected.with_name(
            selected.name + ".identity.json")
        selected_identity_document = load_json(selected_identity)
        require(
            selected.is_file() and selected_identity.is_file() and
            file_sha256(selected) ==
                decision.get("selected_checkpoint_sha256") and
            file_sha256(selected_identity) ==
                decision.get("selected_model_identity_sha256") and
            selected_identity_document.get("version") ==
                EXPECTED_MODEL_IDENTITY_VERSION and
            selected_identity_document.get("training_semantics_version") ==
                EXPECTED_TRAINING_SEMANTICS and
            selected_identity_document.get("policy_id") == policy_id and
            selected_identity_document.get("scheduler") ==
                expected_scheduler and
            selected_identity_document.get("objective") ==
                expected_objective,
            f"selected checkpoint identity mismatch: {policy_id}")
        legacy_direct_selection = bool(
            decision.get("version") ==
                "direct-finetuned-final-evaluation-selection-v1" and
            decision.get("decision") ==
                "accept_finetuned_without_nonregression" and
            decision.get("accepted") is True and
            decision.get("selection_mode") ==
                "operator_requested_direct_finetuned_model" and
            decision.get("nonregression_skipped") is True and
            decision.get("paired_run_count") == 0 and
            decision.get("regression_seeds") == [] and
            decision.get("degradation_tolerance") is None and
            decision.get("degradation_tolerance_kind") is None and
            decision.get("existing_nonregression_artifacts_preserved") is
                True)
        current_tolerance = (
            COUNT_RELATIVE_DEGRADATION_TOLERANCE
            if expected_objective == "count" else
            UTILITY_RELATIVE_DEGRADATION_TOLERANCE)
        guard_contract = {
            WEBOTS_GUARD_VERSION: current_tolerance,
            PREVIOUS_WEBOTS_GUARD_VERSION:
                PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE,
        }
        registered_tolerance = tolerance_registry[policy_id]
        guarded_selection = bool(
            decision.get("version") in guard_contract and
            isinstance(decision.get("accepted"), bool) and
            decision.get("decision") == (
                "accept_finetuned" if decision.get("accepted") else
                "rollback_to_pre") and
            decision.get("paired_run_count") ==
                len(SCENARIOS) * len(REGRESSION_SEEDS) and
            decision.get("regression_seeds") == list(REGRESSION_SEEDS) and
            decision.get("degradation_tolerance") ==
                guard_contract.get(decision.get("version")) and
            decision.get("degradation_tolerance") ==
                registered_tolerance and
            decision.get("degradation_tolerance_kind") == "relative" and
            decision.get("strict_improvement_required") is False and
            decision.get("runtime_event_policy") ==
                WEBOTS_REGRESSION_RUNTIME_EVENT_POLICY and
            decision.get("timeout_fallback_affect_checkpoint_selection") is
                False and
            decision.get("invalid_scheduler_outputs_block_selection") is
                True and
            isinstance(decision.get("runtime_event_audit"), dict) and
            isinstance(decision.get(
                "one_sided_bootstrap_95_lower_bound"), (int, float)))
        require(
            (legacy_direct_selection or guarded_selection) and
            decision.get("source_checkpoint_sha256") ==
                decision.get("selected_checkpoint_sha256") and
            isinstance(decision.get("selected_checkpoint_sha256"), str),
            f"fine-tuned checkpoint selection is incomplete: {policy_id}")
        episode_paths = sorted((policy_dir / "episodes").glob("**/result.json"))
        expected_fine_tune_episodes = len(SCENARIOS) * completed_rounds
        require(len(episode_paths) == expected_fine_tune_episodes,
                f"expected {expected_fine_tune_episodes} fine-tune episodes "
                f"for {policy_id}, got {len(episode_paths)}")
        envelopes = [load_json(path) for path in episode_paths]
        round_paths = [path for path in (policy_dir / "checkpoints").glob("*.json")
                       if ROUND_RECORD_RE.match(path.name)]
        round_records = [load_json(path) for path in sorted(round_paths)]
        require(len(round_records) == completed_rounds,
                f"expected {completed_rounds} fine-tune round records "
                f"for {policy_id}, "
                f"got {len(round_records)}")
        require({row.get("round") for row in round_records} == set(
                    range(1, completed_rounds + 1)),
                f"fine-tune round coverage mismatch: {policy_id}")
        require(all(
                    row.get("schema_version") ==
                        EXPECTED_TRAINING_RECORD_SCHEMA and
                    row.get("training_semantics_version") ==
                        EXPECTED_TRAINING_SEMANTICS
                    for row in round_records),
                f"fine-tune training semantics mismatch: {policy_id}")

        count_values = []
        utility_values = []
        collisions = invalid = fallbacks = 0
        for envelope in envelopes:
            result = envelope.get("result", {})
            info = result.get("experiment_info", {})
            summary = result.get("summary_metrics", {})
            formal = result.get("formal_evaluation", {})
            require(
                info.get("runtime_mode") == "webots"
                and info.get("sim_duration") == HORIZON_SECONDS
                and info.get("task_manifest_matches_expected") is True
                and summary.get("physical_collision_observation_status") == "observed",
                f"fine-tune Webots validation failed: {policy_id}",
            )
            physical = formal.get("physical_fine_tune", {})
            require(physical.get("status") == "complete",
                    f"physical fine-tune rollout is incomplete: {policy_id}")
            require(all(
                isinstance(row.get("shared_supervision"), dict) and
                row["shared_supervision"].get("dataset_version") ==
                    WEBOTS_SHARED_SUPERVISION_VERSION
                for row in physical.get("rollout", [])),
                f"shared Webots supervision is missing: {policy_id}")
            count_values.append(float(
                formal["count_evaluation"]["total_tasks_completed"]))
            utility_values.append(float(
                formal["utility_v2_evaluation"]["utility_score_deadline_v2"]))
            collisions += int(summary.get("physical_collision_count", 0))
            invalid += int(summary.get("invalid_scheduler_outputs", 0))
            fallbacks += int(summary.get("fallback_scheduler_commits", 0))
            parsed = parse_time(envelope.get("completed_at"))
            if parsed:
                timestamps.append(parsed)

        updates = [row["model_update"] for row in round_records]
        if expected_scheduler in {"LearnedHungarian", "GraphImitation"}:
            minimum_sources = 16
            minimum_episodes = (
                minimum_sources * len(SCENARIOS) *
                WEBOTS_FINETUNE_CONVERGENCE_PATIENCE)
            require(all(
                row.get("shared_supervision_contract") ==
                    WEBOTS_SHARED_SUPERVISION_VERSION and
                row.get("shared_supervision_episode_count", 0) >=
                    minimum_episodes and
                row.get("shared_supervision_sample_count", 0) > 0 and
                row.get("shared_supervision_objective") ==
                    expected_objective and
                row.get("shared_supervision_excludes_evaluation") is True and
                len(row.get("shared_supervision_source_policy_ids", [])) ==
                    minimum_sources and
                isinstance(row.get("shared_supervision_corpus_sha256"), str) and
                len(row["shared_supervision_corpus_sha256"]) == 64
                for row in updates),
                f"shared supervision evidence is incomplete: {policy_id}")
        for row in round_records:
            parsed = parse_time(row.get("completed_at"))
            if parsed:
                timestamps.append(parsed)
        learning_rates = sorted({row.get("learning_rate") for row in updates
                                 if row.get("learning_rate") is not None})
        policies.append({
            "policy_id": policy_id,
            "scheduler": report["scheduler"],
            "objective": report["objective"],
            "completion_reason": report["completion_reason"],
            "parameter_converged": report["parameter_converged"],
            "rounds": report["balanced_rounds"],
            "episode_count": report["physical_episode_count"],
            "count_mean": mean(count_values),
            "utility_mean": mean(utility_values),
            "collisions": collisions,
            "invalid": invalid,
            "fallbacks": fallbacks,
            "samples": sum(int(row.get("samples", 0)) for row in updates),
            "optimizer_updates": sum(
                int(row.get("optimizer_updates", 0)) for row in updates),
            "mean_loss": mean(row.get("mean_loss") for row in updates),
            "learning_rates": learning_rates,
            "checkpoint": decision["selected_checkpoint"],
            "checkpoint_sha256": decision["selected_checkpoint_sha256"],
            "nonregression_decision": decision["decision"],
            "nonregression_lower_bound":
                decision.get("one_sided_bootstrap_95_lower_bound"),
            "nonregression_tolerance":
                decision.get("degradation_tolerance"),
            "nonregression_episode_count":
                2 * int(decision.get("paired_run_count", 0)),
        })
    return policies, timestamps


def collect_evaluation(results_dir: Path) -> tuple[dict, list[dict], list[datetime]]:
    evaluation_dir = results_dir / "evaluation"
    aggregate = load_json(evaluation_dir / "all_schedulers_evaluation.json")
    require(aggregate.get("schema_version") == 2,
            "evaluation aggregate does not use pure-ranking schema v2")
    require(aggregate.get("status") == "complete",
            "all-scheduler evaluation aggregate is not complete")
    require(aggregate.get("runtime_mode") == "webots",
            "evaluation aggregate is not a Webots result")
    require(aggregate.get("duration_seconds_per_run") == HORIZON_SECONDS,
            "evaluation aggregate does not use the 1800s horizon")
    seed_contract = aggregate.get("evaluation_seed_contract")
    seed_contract_sha256 = aggregate.get(
        "evaluation_seed_contract_sha256")
    seed_manifest_hashes = (
        seed_contract.get("manifest_file_sha256_by_scenario_seed")
        if isinstance(seed_contract, dict) else None)
    expected_seed_manifest_keys = {
        f"{scenario}:{seed}"
        for scenario in SCENARIOS for seed in VALIDATION_SEEDS}
    require(
        aggregate.get("evaluation_contract") ==
            "count-and-utility-v2-from-every-run" and
        aggregate.get("evaluation_metrics") == ["count", "utility_v2"] and
        aggregate.get("seeds_per_scenario") == len(VALIDATION_SEEDS) and
        aggregate.get("common_random_numbers") is True and
        isinstance(seed_contract, dict) and
        seed_contract.get("policy_independent") is True and
        seed_contract.get("scenarios") == list(SCENARIOS) and
        seed_contract.get("seeds") == list(VALIDATION_SEEDS) and
        isinstance(seed_manifest_hashes, dict) and
        set(seed_manifest_hashes) == expected_seed_manifest_keys and
        all(isinstance(value, str) and len(value) == 64
            for value in seed_manifest_hashes.values()) and
        isinstance(seed_contract_sha256, str) and
        canonical_sha256(seed_contract) == seed_contract_sha256 and
        aggregate.get("policy_identity_required") is True,
        "evaluation aggregate does not use the strict dual-metric contract")
    require(aggregate.get("policy_count") == EXPECTED_EVALUATION_POLICIES,
            f"expected {EXPECTED_EVALUATION_POLICIES} policies")
    require(aggregate.get("run_count") == EXPECTED_EVALUATION_RUNS,
            f"expected {EXPECTED_EVALUATION_RUNS} evaluation runs")

    policy_documents = aggregate.get("policies", [])
    require(len(policy_documents) == EXPECTED_EVALUATION_POLICIES,
            "evaluation policy table length mismatch")
    policy_ids = [row["policy_id"] for row in policy_documents]
    expected_policy_set = set(
        TRAINED_POLICIES + STATIC_LEARNED_POLICIES + TRADITIONAL_POLICIES)
    require(set(policy_ids) == expected_policy_set,
            "evaluation policy roster differs from the expected 22 policies")
    aggregate_by_id = {
        row.get("policy_id"): row for row in policy_documents
        if isinstance(row, dict)
    }

    records_by_policy: dict[str, list[dict]] = defaultdict(list)
    timestamps = []
    for policy_id in policy_ids:
        for scenario in SCENARIOS:
            for seed in VALIDATION_SEEDS:
                path = (evaluation_dir / "episodes" / policy_id / scenario /
                        str(seed) / "result.json")
                envelope = load_json(path)
                aggregate_row = aggregate_by_id[policy_id]
                identity = load_json(path.with_name("policy_identity.json"))
                require(
                    identity.get("version") ==
                        EXPECTED_EVALUATION_IDENTITY_VERSION and
                    identity.get("policy_id") == policy_id and
                    identity.get("scheduler") ==
                        aggregate_row.get("scheduler") and
                    identity.get("training_objective") ==
                        aggregate_row.get("training_objective") and
                    identity.get("evaluation_metrics") ==
                        ["count", "utility_v2"] and
                    identity.get("evaluation_seed_contract_sha256") ==
                        seed_contract_sha256 and
                    identity.get("model_sha256") ==
                        aggregate_row.get("model_sha256") and
                    identity.get("model_identity_sha256") ==
                        aggregate_row.get("model_identity_sha256") and
                    identity.get("scenario") == scenario and
                    identity.get("seed") == seed and
                    identity.get("result_sha256") == file_sha256(path),
                    f"evaluation policy identity mismatch: "
                    f"{policy_id}/{scenario}/{seed}",
                )
                result = envelope.get("result", {})
                info = result.get("experiment_info", {})
                summary = result.get("summary_metrics", {})
                formal = result.get("formal_evaluation", {})
                require(
                    info.get("runtime_mode") == "webots"
                    and info.get("scheduler") ==
                        aggregate_row.get("scheduler")
                    and info.get("scenario") == scenario
                    and info.get("seed") == seed
                    and info.get("sim_duration") == HORIZON_SECONDS
                    and info.get("task_manifest_matches_expected") is True
                    and summary.get("physical_collision_observation_status") == "observed",
                    f"evaluation validation failed: {policy_id}/{scenario}/{seed}",
                )
                row = {
                    "policy_id": policy_id,
                    "scenario": scenario,
                    "seed": seed,
                    "count": float(
                        formal["count_evaluation"]["total_tasks_completed"]),
                    "utility": float(
                        formal["utility_v2_evaluation"]
                        ["utility_score_deadline_v2"]),
                    "throughput": summary.get("throughput_per_minute"),
                    "completion_time": summary.get("avg_task_completion_time"),
                    "waiting_time": summary.get("avg_waiting_time"),
                    "distance": summary.get("total_distance_all_robots"),
                    "workload_cv": summary.get("workload_balance_cv"),
                    "latency_p95_ms": summary.get("scheduling_latency_p95_ms"),
                    "collisions": int(summary.get("physical_collision_count", 0)),
                    "safety_events": int(summary.get("safety_event_count", 0)),
                    "pair_violations": int(summary.get("pair_distance_violations", 0)),
                    "deadlocks": int(summary.get("total_deadlocks", 0)),
                    "invalid": int(summary.get("invalid_scheduler_outputs", 0)),
                    "fallbacks": int(summary.get("fallback_scheduler_commits", 0)),
                    "scheduler_timeouts": int(summary.get("scheduler_timeout_count", 0)),
                    "joint_solutions": int(summary.get(
                        "solver_joint_solution_count", 0)),
                    "proposed_pairs": int(summary.get(
                        "solver_proposed_pair_count", 0)),
                    "unattempted_pairs": int(summary.get(
                        "solver_unattempted_pair_count", 0)),
                    "first_attempts": int(summary.get(
                        "first_assignment_attempt_count", 0)),
                    "first_commits": int(summary.get(
                        "first_assignment_commit_count", 0)),
                    "first_rejections": int(summary.get(
                        "first_assignment_rejection_count", 0)),
                    "rl_timeouts": int(summary.get("rl_timeout_count", 0)),
                    "fallback_reasons": summary.get(
                        "scheduler_fallback_reasons", {}),
                    "failure_reasons": summary.get(
                        "scheduler_failure_reasons", {}),
                    "unauthorized_route_writes": int(
                        summary.get("unauthorized_route_writes", 0)),
                    "nonphysical_recoveries": int(
                        summary.get("nonphysical_recoveries", 0)),
                    "unplanned_task_stops": int(
                        summary.get("unplanned_task_stops", 0)),
                }
                records_by_policy[policy_id].append(row)
                parsed = parse_time(envelope.get("completed_at"))
                if parsed:
                    timestamps.append(parsed)

    policies = []
    expected_records_per_policy = len(SCENARIOS) * len(VALIDATION_SEEDS)
    for policy_id in policy_ids:
        records = records_by_policy[policy_id]
        require(len(records) == expected_records_per_policy,
                f"expected {expected_records_per_policy} evaluation records "
                f"for {policy_id}")
        policy = {
            "policy_id": policy_id,
            "training_objective": aggregate_by_id[policy_id].get(
                "training_objective"),
            "category": policy_category(policy_id),
            "records": records,
            "count_mean": mean(row["count"] for row in records),
            "count_std": sample_std(row["count"] for row in records),
            "count_min": min(row["count"] for row in records),
            "count_max": max(row["count"] for row in records),
            "utility_mean": mean(row["utility"] for row in records),
            "utility_std": sample_std(row["utility"] for row in records),
            "utility_min": min(row["utility"] for row in records),
            "utility_max": max(row["utility"] for row in records),
            "throughput_mean": mean(row["throughput"] for row in records),
            "completion_mean": mean(row["completion_time"] for row in records),
            "waiting_mean": mean(row["waiting_time"] for row in records),
            "distance_mean": mean(row["distance"] for row in records),
            "workload_cv_mean": mean(row["workload_cv"] for row in records),
            "latency_p95_mean": mean(row["latency_p95_ms"] for row in records),
        }
        for key in (
                "collisions", "safety_events", "pair_violations", "deadlocks",
                "invalid", "fallbacks", "scheduler_timeouts", "rl_timeouts",
                "unauthorized_route_writes", "nonphysical_recoveries",
                "unplanned_task_stops", "joint_solutions",
                "proposed_pairs", "unattempted_pairs", "first_attempts",
                "first_commits", "first_rejections"):
            policy[key] = sum(row[key] for row in records)
        for output_key, record_key in (
                ("fallback_reasons", "fallback_reasons"),
                ("failure_reasons", "failure_reasons")):
            totals = defaultdict(int)
            for record in records:
                values = record[record_key]
                require(isinstance(values, dict),
                        f"{record_key} is not a mapping: {policy_id}")
                for reason, value in values.items():
                    require(
                        not isinstance(value, bool)
                        and isinstance(value, int) and value >= 0,
                        f"invalid {record_key} count: {policy_id}/{reason}",
                    )
                    totals[str(reason)] += value
            policy[output_key] = dict(sorted(totals.items()))
        policy.update(pure_policy_eligibility({
            "fallback_scheduler_commits": policy["fallbacks"],
            "invalid_scheduler_outputs": policy["invalid"],
            "scheduler_timeout_count": policy["scheduler_timeouts"],
        }))
        policies.append(policy)

    pure_policies = [row for row in policies if row["pure_policy_eligible"]]
    for row in policies:
        aggregate_row = aggregate_by_id.get(row["policy_id"], {})
        require(
            aggregate_row.get("pure_policy_eligible")
            == row["pure_policy_eligible"]
            and aggregate_row.get("pure_policy_exclusion_reasons")
            == row["pure_policy_exclusion_reasons"],
            f"aggregate purity decision mismatch: {row['policy_id']}",
        )
    expected_exclusions = {
        row["policy_id"]: row["pure_policy_exclusion_reasons"]
        for row in policies if not row["pure_policy_eligible"]
    }
    require(aggregate.get("pure_policy_count") == len(pure_policies),
            "aggregate pure policy count mismatch")
    require(aggregate.get("excluded_from_pure_ranking") == expected_exclusions,
            "aggregate pure policy exclusions mismatch")
    count_candidates = [
        row for row in pure_policies
        if row["training_objective"] in {None, "count"}
    ]
    utility_candidates = [
        row for row in pure_policies
        if row["training_objective"] in {None, "utility_v2"}
    ]
    count_order = sorted(
        count_candidates,
        key=lambda row: (-row["count_mean"], row["collisions"], row["policy_id"]),
    )
    utility_order = sorted(
        utility_candidates,
        key=lambda row: (-row["utility_mean"], row["collisions"], row["policy_id"]),
    )
    require([row["policy_id"] for row in count_order]
            == aggregate.get("count_ranking"),
            "recomputed count ranking differs from the aggregate")
    require([row["policy_id"] for row in utility_order]
            == aggregate.get("utility_v2_ranking"),
            "recomputed utility ranking differs from the aggregate")
    count_rank = {row["policy_id"]: index
                  for index, row in enumerate(count_order, start=1)}
    utility_rank = {row["policy_id"]: index
                    for index, row in enumerate(utility_order, start=1)}
    for row in policies:
        row["count_rank"] = count_rank.get(row["policy_id"])
        row["utility_rank"] = utility_rank.get(row["policy_id"])
    return aggregate, policies, timestamps


def percent_delta(value: float, baseline: float) -> str:
    if baseline == 0:
        return "—"
    delta = (value - baseline) / abs(baseline) * 100.0
    return f"{delta:+.2f}%"


def stage_interval(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None:
        return "—"
    return fmt_duration((end - start).total_seconds())


def build_report(results_dir: Path) -> str:
    state = load_json(results_dir / "workflow_state.json")
    require(
            state.get("status") == "complete" and
            state.get("stage") == "complete" and
            state.get("schema_version") ==
                EXPECTED_WORKFLOW_STATE_SCHEMA and
            state.get("workflow_contract") ==
                EXPECTED_WORKFLOW_CONTRACT and
            state.get("training_semantics_version") ==
                EXPECTED_TRAINING_SEMANTICS,
            "workflow_state.json is not complete")

    headless, headless_times = collect_headless(results_dir)
    fine_tune, fine_times = collect_fine_tune(results_dir)
    aggregate, evaluation, evaluation_times = collect_evaluation(results_dir)

    created = parse_time(state.get("created_at"))
    completed = parse_time(state.get("completed_at"))
    headless_end = max(headless_times) if headless_times else None
    fine_end = max(fine_times) if fine_times else None
    evaluation_end = max(evaluation_times) if evaluation_times else None
    generated = datetime.now().astimezone()

    count_order = sorted(
        (row for row in evaluation
         if row["pure_policy_eligible"] and row["count_rank"] is not None),
        key=lambda row: row["count_rank"])
    utility_order = sorted(
        (row for row in evaluation
         if row["pure_policy_eligible"] and row["utility_rank"] is not None),
        key=lambda row: row["utility_rank"])
    by_id = {row["policy_id"]: row for row in evaluation}
    baseline = by_id["FCFS"]
    best_count = count_order[0]
    best_utility = utility_order[0]
    learned_set = set(TRAINED_POLICIES + STATIC_LEARNED_POLICIES)
    best_learned_count = next(row for row in count_order
                              if row["policy_id"] in learned_set)
    best_learned_utility = next(row for row in utility_order
                                if row["policy_id"] in learned_set)
    gate_fields = (
        "collisions", "safety_events", "pair_violations", "deadlocks",
        "invalid", "scheduler_timeouts", "rl_timeouts",
        "fallbacks",
        "unauthorized_route_writes", "nonphysical_recoveries",
        "unplanned_task_stops",
    )
    clean_policies = [row for row in evaluation
                      if all(row[field] == 0 for field in gate_fields)]
    failures = sorted(results_dir.glob("workflow.failed_*.log"))
    resume_count = 0
    for stdout_path in results_dir.glob("workflow*.stdout.log"):
        if stdout_path.is_file():
            resume_count += stdout_path.read_text(
                encoding="utf-8", errors="replace").count("[RESUME]")
    standalone_runs = sum(row["episode_count"] for row in headless)
    fine_tune_runs = sum(row["episode_count"] for row in fine_tune)
    regression_runs = sum(
        row["nonregression_episode_count"] for row in fine_tune)
    evaluation_runs = int(aggregate["run_count"])
    simulated_hours = (
        standalone_runs + fine_tune_runs + regression_runs + evaluation_runs
    ) * HORIZON_SECONDS / 3600.0

    lines = [
        "# 严格 1800s 调度算法训练、Webots 微调与评价报告",
        "",
        f"> 生成时间：{generated.strftime('%Y-%m-%d %H:%M:%S %z')}  ",
        "> 数据来源：`results/all_scheduler_model_remediation/formal_1800_dual_v6` "
        "下的原子化 checkpoint、"
        "episode 结果和最终汇总。  ",
        "> 生成条件：工作流状态完整、20 个 AI 目标实例的训练和微调产物完整、"
        f"22 个策略的 {EXPECTED_EVALUATION_RUNS} 次 Webots 评价全部通过结构及 "
        "1800s 时长校验。",
        "",
        "## 1. 执行结论",
        "",
        f"工作流已完成。共执行 **{standalone_runs} 次固定 1800s standalone "
        f"充足训练**、**{fine_tune_runs} 次固定 1800s Webots 物理微调**、"
        f"**{evaluation_runs} 次固定 "
        f"1800s Webots 正式评价**。累计覆盖 {simulated_hours:.0f} 个模拟小时。",
        "",
        f"- Count 指标第一名：**{best_count['policy_id']}**，"
        f"平均完成任务数 {fmt_number(best_count['count_mean'])}。",
        f"- Utility V2 指标第一名：**{best_utility['policy_id']}**，"
        f"平均得分 {fmt_number(best_utility['utility_mean'])}。",
        f"- 学习类策略中 Count 最优：**{best_learned_count['policy_id']}**，"
        f"相对 FCFS 为 {percent_delta(best_learned_count['count_mean'], baseline['count_mean'])}。",
        f"- 学习类策略中 Utility V2 最优：**{best_learned_utility['policy_id']}**，"
        f"相对 FCFS 为 {percent_delta(best_learned_utility['utility_mean'], baseline['utility_mean'])}。",
        f"- 严格运行门禁零事件策略：**{len(clean_policies)}/{len(evaluation)}**。"
        "门禁统计物理碰撞、安全事件、间距违规、死锁、无效输出、超时、"
        "越权路径写入、非物理恢复和非计划停车。",
        "",
        "正式结果只说明这些固定 manifest、场景、种子和 1800s 时域下的表现，"
        "不能单独证明面对未覆盖的工厂布局、任务分布或硬件扰动仍保持相同排名。",
        "",
        "## 2. 范围、配置与可复现性",
        "",
    ]
    lines.extend(table(
        ["阶段", "策略数", "场景", "种子", "每策略次数", "总次数", "每次时长"],
        [
            ["standalone 充足训练", len(TRAINED_POLICIES), "A/B/C",
             (f"21000–21199（最早 {MIN_STANDALONE_ROUNDS} 轮，"
              "按收敛停止）"),
             (f"{MIN_STANDALONE_ROUNDS*len(SCENARIOS)}–"
              f"{MAX_STANDALONE_ROUNDS*len(SCENARIOS)}"),
             standalone_runs, "1800s"],
            ["Webots 微调", len(TRAINED_POLICIES), "A/B/C",
             f"{min(FINETUNE_SEEDS)}–{max(FINETUNE_SEEDS)}",
             len(SCENARIOS) * MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS,
             fine_tune_runs, "1800s"],
            ["Webots 评价", EXPECTED_EVALUATION_POLICIES, "A/B/C",
             "61000–61005", len(SCENARIOS) * len(VALIDATION_SEEDS),
             evaluation_runs, "1800s"],
        ],
    ))
    lines.extend([
        "",
        "训练、验证、微调和最终评价使用隔离种子；每个阶段均对 A/B/C "
        "三种场景进行平衡覆盖。评价阶段仅包括 FCFS、NearestNeighbour 两个非 AI "
        "基线和 20 个完成 "
        "standalone 充足训练并经过 Webots 微调门禁的 AI 策略。",
        "",
        "断点协议如下：standalone 每完成一个 1800s episode 写 checkpoint；Webots "
        "微调每完成一个物理 episode 写原子结果，并在每个 A/B/C 平衡轮次后写模型 checkpoint；"
        "评价每完成一次 1800s 运行写原子结果。中断时只重跑当时未提交的最小单元。",
        "",
        f"最终启动日志中记录到 **{resume_count}** 条 `[RESUME]`，说明续跑时复用了"
        "已经通过校验的结果。历史诊断日志保留 **"
        f"{len(failures)}** 个："
        + ("、".join(f"`{path.name}`" for path in failures) if failures else "无")
        + f"。这些日志不计入最终 {EXPECTED_EVALUATION_RUNS} 次正式结果。",
        "",
        "## 3. 执行时间线",
        "",
    ])
    timeline_rows = [
        ["工作流创建", fmt_time(state.get("created_at")), "—", "状态文件时间"],
        ["standalone 充足训练完成", headless_end.strftime("%Y-%m-%d %H:%M:%S %z")
         if headless_end else "—", stage_interval(created, headless_end),
         "包含重启/停顿的墙钟时间"],
        ["Webots 微调完成", fine_end.strftime("%Y-%m-%d %H:%M:%S %z")
         if fine_end else "—", stage_interval(headless_end, fine_end),
         "包含重启/停顿的墙钟时间"],
        ["Webots 评价最后结果", evaluation_end.strftime("%Y-%m-%d %H:%M:%S %z")
         if evaluation_end else "—", stage_interval(fine_end, evaluation_end),
         "最后一个原子结果时间"],
        ["工作流提交完成", fmt_time(state.get("completed_at")),
         stage_interval(created, completed), "全流程墙钟时间"],
    ]
    lines.extend(table(["里程碑", "时间", "阶段/累计耗时", "口径"], timeline_rows))

    lines.extend([
        "",
        "## 4. Headless 训练过程与结果",
        "",
        "Headless 阶段复用 Webots 业务逻辑，但不宣称具有物理保真度。每个策略以"
        f"验证指标平台与参数收敛为停止条件（最早 {MIN_STANDALONE_ROUNDS} 个平衡轮次）；"
        "每个 episode "
        "必须推进到精确 1800s，"
        "且以 `episode_horizon` 正常结束。",
        "",
    ])
    training_rows = []
    for row in headless:
        training_rows.append([
            row["policy_id"], row["objective"], row["episode_count"],
            f"{fmt_number(row['return_mean'])} ± {fmt_number(row['return_std'])}",
            fmt_number(row["tasks_mean"]), row["decisions"],
            row["optimizer_updates"], fmt_number(row["mean_loss"]),
            row["transition_count"], row["checkpoint_sha256"][:12],
        ])
    lines.extend(table(
        ["策略", "目标", "轮数", "Return 均值±标准差", "完成数均值",
         "决策数", "优化更新", "Loss 均值", "Transition", "最终 SHA-256"],
        training_rows,
    ))
    lines.extend([
        "",
        "### 4.1 分场景训练结果",
        "",
    ])
    scenario_training_rows = []
    for row in headless:
        for scenario in SCENARIOS:
            item = row["scenario_rows"][scenario]
            scenario_training_rows.append([
                row["policy_id"], scenario,
                fmt_number(item["return_mean"]), fmt_number(item["tasks_mean"]),
            ])
    lines.extend(table(
        ["策略", "场景", "Return 均值", "完成任务数均值"],
        scenario_training_rows,
    ))
    lines.extend([
        "",
        "`Loss` 只在相应算法产生该指标时统计；不同算法的 loss 定义和尺度不同，"
        "不能直接按数值大小跨算法比较。Return 也随 Count/Utility V2 训练目标不同而不同，"
        "最终优劣以独立 Webots 评价为准。",
        "",
        "## 5. Webots 物理微调过程与结果",
        "",
        "每个 AI 策略逐轮微调至收敛：每轮依次采集 A/B/C 三个 1800s "
        "Webots episode 后更新模型并检查参数变化；连续 3 轮稳定即停止，最多 "
        f"{MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS} 轮。达到上限时保留真实的未收敛标记，"
        "完成后直接选择微调 checkpoint，并进入同批 6-seed 正式评价，不再执行额外的"
        "微调前后防退化评价。"
        "可学习率算法使用基础学习率的 0.1 倍；"
        "LinUCB 等无该参数的算法按其原生更新规则处理。",
        "",
    ])
    fine_rows = []
    for row in fine_tune:
        rates = ", ".join(fmt_number(value, 7) for value in row["learning_rates"])
        fine_rows.append([
            row["policy_id"], row["objective"], row["rounds"],
            row["episode_count"], fmt_number(row["count_mean"]),
            fmt_number(row["utility_mean"]), row["samples"],
            row["optimizer_updates"], fmt_number(row["mean_loss"]),
            rates or "—", row["collisions"], row["invalid"], row["fallbacks"],
            row["checkpoint_sha256"][:12],
        ])
    lines.extend(table(
        ["策略", "目标", "轮次", "物理轮数", "Count 均值", "Utility 均值",
         "样本", "更新", "Loss 均值", "微调学习率", "碰撞", "无效输出",
         "回退", "最终 SHA-256"],
        fine_rows,
    ))
    lines.extend([
        "",
        "微调表中的 Count/Utility 是训练 manifest 上物理 rollout 的描述性结果，"
        "用于核对采集质量，不用于最终排名；正式排名只使用未参与训练、"
        "微调的 61000–61005 种子。",
        "",
        "## 6. 全部调度算法 Webots 评价",
        "",
        "### 6.1 总排名与核心指标",
        "",
    ])
    ranking_rows = []
    for row in sorted(
            evaluation,
            key=lambda item: (
                item["count_rank"] is None,
                item["count_rank"] if item["count_rank"] is not None else 0,
                item["policy_id"])):
        gate_clean = all(row[field] == 0 for field in gate_fields)
        ranking_rows.append([
            row["count_rank"] or "未入榜", row["utility_rank"] or "未入榜",
            row["policy_id"],
            row["category"],
            f"{fmt_number(row['count_mean'])} ± {fmt_number(row['count_std'])}",
            percent_delta(row["count_mean"], baseline["count_mean"]),
            f"{fmt_number(row['utility_mean'])} ± {fmt_number(row['utility_std'])}",
            percent_delta(row["utility_mean"], baseline["utility_mean"]),
            fmt_number(row["completion_mean"]), fmt_number(row["waiting_mean"]),
            fmt_number(row["latency_p95_mean"]), row["collisions"],
            row["invalid"], row["fallbacks"],
            "通过" if row["pure_policy_eligible"] else "排除",
            "通过" if gate_clean else "未通过",
        ])
    lines.extend(table(
        ["Count 名次", "Utility 名次", "策略", "类别", "Count 均值±标准差",
         "较 FCFS", "Utility 均值±标准差", "较 FCFS", "平均完成时间(s)",
         "平均等待(s)", "调度 P95(ms)", "碰撞", "无效", "回退",
         "纯算法排名", "运行门禁"],
        ranking_rows,
    ))
    lines.extend([
        "",
        "只有 `fallback_scheduler_commits=0`、`invalid_scheduler_outputs=0` 且"
        f" `scheduler_timeout_count=0` 的策略进入纯算法排名。Count 排名先比较 "
        f"{len(SCENARIOS) * len(VALIDATION_SEEDS)} 次"
        "运行的平均完成任务数，再以碰撞数和策略名稳定破同分；Utility 排名使用"
        f" `utility_score_deadline_v2` 的 "
        f"{len(SCENARIOS) * len(VALIDATION_SEEDS)} 次平均值并采用相同破同分规则。被排除策略"
        "仍保留描述性分数，但不能用于‘最优算法’结论。",
        "",
        "### 6.2 分场景结果",
        "",
    ])
    scenario_rows = []
    for row in sorted(evaluation, key=lambda item: item["policy_id"]):
        for scenario in SCENARIOS:
            selected = [item for item in row["records"]
                        if item["scenario"] == scenario]
            scenario_rows.append([
                row["policy_id"], scenario,
                f"{fmt_number(mean(item['count'] for item in selected))} ± "
                f"{fmt_number(sample_std(item['count'] for item in selected))}",
                f"{fmt_number(mean(item['utility'] for item in selected))} ± "
                f"{fmt_number(sample_std(item['utility'] for item in selected))}",
                fmt_number(mean(item["completion_time"] for item in selected)),
                fmt_number(mean(item["waiting_time"] for item in selected)),
                sum(item["collisions"] for item in selected),
                sum(item["fallbacks"] for item in selected),
            ])
    lines.extend(table(
        ["策略", "场景", "Count 均值±标准差", "Utility 均值±标准差",
         "平均完成时间(s)", "平均等待(s)", "碰撞", "回退"],
        scenario_rows,
    ))
    lines.extend([
        "",
        "### 6.3 安全性、有效性与回退明细",
        "",
    ])
    safety_rows = []
    for row in sorted(evaluation, key=lambda item: item["policy_id"]):
        fallback_reason_text = ", ".join(
            f"{key}:{value}" for key, value in row["fallback_reasons"].items())
        failure_reason_text = ", ".join(
            f"{key}:{value}" for key, value in row["failure_reasons"].items())
        safety_rows.append([
            row["policy_id"], row["collisions"], row["safety_events"],
            row["pair_violations"], row["deadlocks"], row["invalid"],
            row["scheduler_timeouts"], row["rl_timeouts"], row["fallbacks"],
            row["unauthorized_route_writes"], row["nonphysical_recoveries"],
            row["unplanned_task_stops"], fallback_reason_text or "—",
            failure_reason_text or "—",
        ])
    lines.extend(table(
        ["策略", "物理碰撞", "安全事件", "间距违规", "死锁", "无效输出",
         "调度超时", "RL 超时", "回退提交", "越权写路径", "非物理恢复", "非计划停车",
         "回退原因", "全部调度失败原因"],
        safety_rows,
    ))
    lines.extend([
        "",
        "回退提交单独报告，不与物理碰撞等同；它表示主调度器未能直接提交结果并使用了"
        "后备逻辑。任何回退、无效输出或调度超时都会把策略排除出纯算法排名。",
        "",
        "## 7. 结果解读与建议",
        "",
        f"1. **吞吐导向选择**：本次固定条件下，Count 第一名为 "
        f"`{best_count['policy_id']}`；若限定学习类策略，则为 "
        f"`{best_learned_count['policy_id']}`。选择前应同时查看其 Utility 名次和回退次数。",
        f"2. **优先级/时限导向选择**：Utility V2 第一名为 "
        f"`{best_utility['policy_id']}`；若限定学习类策略，则为 "
        f"`{best_learned_utility['policy_id']}`。该指标综合优先级完成、及时性、"
        "排队附加收益、路线效率和 deadline 惩罚。",
        "3. **安全门禁优先于分数**：任何物理碰撞、安全事件、间距违规、死锁、"
        "无效输出、超时或回退都应先阻止部署和纯算法排名，再分析描述性分数。",
        "4. **继续验证**：建议对候选前 2–3 名增加不同布局、到达率、机器人数量、"
        "电量扰动和更长连续运行测试，并用配对置信区间检验优势是否稳定。",
        "5. **不要跨目标比较训练 loss**：本轮算法的网络结构、更新规则和目标函数不同；"
        "应使用本报告中的独立 Webots Count/Utility 与安全指标作最终比较。",
        "",
        "## 8. 产物与审计入口",
        "",
    ])
    artifact_rows = [
        ["工作流状态", "`results/all_scheduler_model_remediation/formal_1800_dual_v6/workflow_state.json`"],
        ["standalone 充足训练", "`results/all_scheduler_model_remediation/formal_1800_dual_v6/headless/<policy>/`"],
        ["Webots 微调", "`results/all_scheduler_model_remediation/formal_1800_dual_v6/webots_fine_tune/<policy>/`"],
        ["微调模型直接选择记录", "`results/all_scheduler_model_remediation/formal_1800_dual_v6/webots_fine_tune/<policy>/nonregression_decision.json`"],
        ["逐次评价", "`results/all_scheduler_model_remediation/formal_1800_dual_v6/evaluation/episodes/<policy>/<scenario>/<seed>/result.json`"],
        ["机器可读总表", "`results/all_scheduler_model_remediation/formal_1800_dual_v6/evaluation/all_schedulers_evaluation.json`"],
        ["执行标准输出", "`results/all_scheduler_model_remediation/formal_1800_dual_v6/workflow.stdout.log`"],
        ["执行标准错误", "`results/all_scheduler_model_remediation/formal_1800_dual_v6/workflow.stderr.log`"],
    ]
    lines.extend(table(["产物", "路径"], artifact_rows))
    lines.extend([
        "",
        "每个训练和微调报告均记录最终 checkpoint 的 SHA-256；每个 Webots 结果"
        "同时记录 manifest 与输入模型哈希。原始 JSON 是审计事实源，本报告中的均值、"
        "标准差、排名和事件合计均由这些文件重新计算，并与最终机器可读汇总交叉校验。",
        "",
        "---",
        "",
        "本报告由 `scripts/generate_strict_1800_report.py` 自动生成。"
        "若正式结果发生变化，应重新运行生成器，不要手工修改统计表。",
        "",
    ])
    return "\n".join(lines)


def readiness(results_dir: Path) -> tuple[bool, str]:
    state_path = results_dir / "workflow_state.json"
    if not state_path.is_file():
        return False, "workflow state is not available"
    try:
        state = load_json(state_path)
    except ReportDataError as exc:
        return False, str(exc)
    status = str(state.get("status", "unknown"))
    stage = str(state.get("stage", "unknown"))
    current = state.get("current")
    evaluation_count = len(list(
        (results_dir / "evaluation" / "episodes").glob("*/*/*/result.json")))
    ready = (
        status == "complete"
        and stage == "complete"
        and (results_dir / "evaluation" /
             "all_schedulers_evaluation.json").is_file()
    )
    detail = (f"status={status}, stage={stage}, evaluation="
              f"{evaluation_count}/{EXPECTED_EVALUATION_RUNS}, current={current}")
    return ready, detail


def wait_until_ready(results_dir: Path, poll_seconds: int) -> None:
    last_detail = None
    while True:
        ready, detail = readiness(results_dir)
        if detail != last_detail:
            print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                  f"{detail}", flush=True)
            last_detail = detail
        if ready:
            return
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(
        "Generate the final strict 1800s training/fine-tuning/evaluation report"))
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wait", action="store_true",
                        help="wait until the workflow has completed")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--check-ready", action="store_true",
                        help="print readiness and exit without writing a report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_seconds < 5:
        raise ValueError("--poll-seconds must be at least 5")
    results_dir = args.results_dir.resolve()
    output = args.output.resolve()
    if args.check_ready:
        ready, detail = readiness(results_dir)
        print(json.dumps({"ready": ready, "detail": detail}, ensure_ascii=False))
        return 0 if ready else 2
    if args.wait:
        wait_until_ready(results_dir, args.poll_seconds)
    report = build_report(results_dir)
    atomic_write_text(output, report)
    print(json.dumps({
        "status": "complete",
        "report": str(output),
        "bytes": output.stat().st_size,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
