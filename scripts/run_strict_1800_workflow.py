"""One-command, interruption-safe 1800s training and evaluation workflow.

Each objective-specific AI model runs an end-to-end pipeline before the next
model starts: automatic hyperparameter optimization, convergence-driven
standalone training, Webots fine-tuning, two-seed bounded-regression
selection, and final evaluation.  LearnedHungarian and GraphImitation run last so their
physical supervision can use the complete validated non-supervised corpus.
Atomic commit markers make every stage safely resumable at its first
incomplete unit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import (
    FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait,
)
from contextlib import contextmanager
from dataclasses import fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "controllers" / "factory_supervisor"
for entry in (ROOT / "scripts", SUPERVISOR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_experiments  # noqa: E402
import run_webots_1800_workflow as webots_workflow  # noqa: E402
from advanced_rl_agents import (  # noqa: E402
    CQLAgent, CQLConfig, OfflineTransitionDataset, QRDQNAgent, QRDQNConfig,
    RainbowConfig, RainbowDQNAgent,
)
from advanced_rl_training import _cost_greedy_action  # noqa: E402
from all_ai_auto_optimizer import (  # noqa: E402
    ACCEPTED_FORMAL_VALIDATION_SEED_CONTRACTS,
    deterministic_trial_configs, validate_trial_config,
    FORMAL_VALIDATION_SEEDS, MAX_STANDALONE_BALANCED_ROUNDS,
    MIN_STANDALONE_BALANCED_ROUNDS, STANDALONE_TRAINING_SEEDS,
    PARAMETER_CONVERGENCE_PATIENCE_CHECKS,
    PARAMETER_RELATIVE_CHANGE_THRESHOLD,
    SEARCH_SPACES, TRAINABLE_ALGORITHMS, TRAINABLE_OBJECTIVES,
    SUPERVISED_MIN_NEAREST_NEIGHBOUR_RATIO,
    SUPERVISED_MIN_OBJECTIVE_EXPERT_RATIO,
    VALIDATION_INTERVAL_ROUNDS,
    standalone_training_readiness,
)
from ai_training_integration import (  # noqa: E402
    _masked_probabilities, _pairwise_ppo_update,
)
from bandit_scheduler import (  # noqa: E402
    LinUCBModel, _arm_factory, bandit_context_vector,
)
from dual_auto_optimizer import train_objective_graph_ppo  # noqa: E402
from dual_objective import (  # noqa: E402
    CountRewardProfile, UtilityScoreConfig, UtilityV2RewardProfile,
    physical_episode_reward_attribution,
)
from graph_ppo_scheduler import (  # noqa: E402
    GraphPPOConfig, GraphPPOModel, GraphRolloutStep,
)
from headless_training_runtime import (  # noqa: E402
    HEADLESS_DYNAMICS_VERSION, HeadlessRuntimeConfig,
)
from learning_scheduler import (  # noqa: E402
    FEATURE_VERSION, GRAPH_FEATURE_NAMES, PAIR_FEATURE_NAMES,
    TRAFFIC_GRAPH_FEATURE_NAMES, TRAFFIC_GRAPH_FEATURE_VERSION,
    TRAFFIC_PAIR_FEATURE_NAMES, TRAFFIC_PAIR_FEATURE_VERSION, TRAFFIC_VERSION,
    WEBOTS_SHARED_SUPERVISION_VERSION,
    GraphEdgeImitationModel, RidgeCostModel,
    graph_edge_feature_tensor, graph_imitation_values,
    learned_cost_values, objective_expert_values,
    objective_supervision_batches, pair_feature_tensor, solve_hungarian,
    traffic_graph_feature_tensor, traffic_graph_supervision,
    traffic_objective_expert_values,
    traffic_pair_feature_tensor, traffic_ridge_supervision,
)
from rl_agents import (  # noqa: E402
    DQNAgent, DQNConfig, DQN_MODEL_VERSION, SarsaAgent, SarsaConfig,
    dqn_config_from_checkpoint,
)
from rl_environment import RLEnvironmentConfig, SchedulingEnvironment  # noqa: E402
from schedulers import (  # noqa: E402
    Assignment, ModelValidationError, PPONetwork, build_cost_matrix,
    create_scheduler,
    validate_assignments,
)
from task_generator import (  # noqa: E402
    generate_task_manifest, task_manifest_sha256, write_task_manifest,
)
from training_scenarios import (  # noqa: E402
    factory_scenario, formal_scenario_config,
)
from webots_finetune_guard import (  # noqa: E402
    COUNT_RELATIVE_DEGRADATION_TOLERANCE, FINAL_EVALUATION_SEEDS,
    MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS,
    MIN_WEBOTS_FINETUNE_BALANCED_ROUNDS,
    PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE,
    WEBOTS_FINETUNE_CONVERGENCE_PATIENCE,
    WEBOTS_FINETUNE_PARAMETER_CHANGE_THRESHOLD, WEBOTS_FINETUNE_SEEDS,
    WEBOTS_REGRESSION_RUNTIME_EVENT_POLICY, WEBOTS_REGRESSION_SEEDS,
    UTILITY_RELATIVE_DEGRADATION_TOLERANCE,
    webots_finetune_regression_decision,
)


DURATION_SECONDS = 1800.0
SCENARIOS = ("A", "B", "C")
V18_WORKFLOW_CONTRACT = (
    "dual-objective-training-v18-two-isolated-headless-lanes-"
    "nn-fcfs-evaluation")
V18_WORKFLOW_STATE_SCHEMA_VERSION = 17
V19_WORKFLOW_CONTRACT = (
    "dual-objective-training-v19-operator-webots-transition-ten-round-"
    "finetune-nn-fcfs-evaluation")
V19_WORKFLOW_STATE_SCHEMA_VERSION = 18
V20_WORKFLOW_CONTRACT = (
    "dual-objective-training-v20-ten-round-completion-six-seed-"
    "evaluation-nn-fcfs")
V20_WORKFLOW_STATE_SCHEMA_VERSION = 19
V21_WORKFLOW_CONTRACT = (
    "dual-objective-training-v21-direct-finetuned-six-seed-"
    "evaluation-nn-fcfs")
V21_WORKFLOW_STATE_SCHEMA_VERSION = 20
V22_WORKFLOW_CONTRACT = (
    "dual-objective-training-v22-two-seed-webots-regression-six-seed-"
    "evaluation-nn-fcfs")
V22_WORKFLOW_STATE_SCHEMA_VERSION = 21
V23_WORKFLOW_CONTRACT = (
    "dual-objective-training-v23-standalone-maximum-completion-two-seed-"
    "webots-regression-six-seed-evaluation-nn-fcfs")
V23_WORKFLOW_STATE_SCHEMA_VERSION = 22
V24_WORKFLOW_CONTRACT = (
    "dual-objective-training-v24-webots-timeout-fallback-audit-only-"
    "six-seed-evaluation-nn-fcfs")
V24_WORKFLOW_STATE_SCHEMA_VERSION = 23
WORKFLOW_CONTRACT = (
    "dual-objective-training-v25-unfinetuned-ten-percent-webots-regression-"
    "six-seed-evaluation-nn-fcfs")
V17_WORKFLOW_CONTRACT = (
    "dual-objective-training-v17-five-percent-regression-model-pipeline-"
    "nn-fcfs-evaluation")
V17_WORKFLOW_STATE_SCHEMA_VERSION = 16
V16_WORKFLOW_CONTRACT = (
    "dual-objective-training-v16-parallel-headless-model-pipeline-"
    "nn-fcfs-evaluation")
V16_WORKFLOW_STATE_SCHEMA_VERSION = 15
V15_WORKFLOW_CONTRACT = (
    "dual-objective-training-v15-convergent-webots-model-pipeline-"
    "nn-fcfs-evaluation")
V15_WORKFLOW_STATE_SCHEMA_VERSION = 14
V15_MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS = 21
V14_WORKFLOW_CONTRACT = (
    "dual-objective-training-v14-efficient-tuning-model-pipeline-"
    "nn-fcfs-evaluation")
V14_WORKFLOW_STATE_SCHEMA_VERSION = 13
PREVIOUS_WORKFLOW_CONTRACT = (
    "dual-objective-training-v13-model-pipeline-nn-fcfs-evaluation")
PREVIOUS_WORKFLOW_STATE_SCHEMA_VERSION = 12
WORKFLOW_STATE_SCHEMA_VERSION = 24
TRAINING_SEMANTICS_VERSION = (
    "task-lifecycle-attribution-v6-convergence-bounded-regression")
TRAINING_RECORD_SCHEMA_VERSION = 7
DEFAULT_RUN_DIR = (
    ROOT / "results" / "all_scheduler_model_remediation" /
    "formal_1800_dual_v6")
DEFAULT_REPORT_PATH = (
    ROOT / "docs" / "all_scheduler_model_remediation_training_report_v6.md")
TRAINING_SEEDS = STANDALONE_TRAINING_SEEDS
VALIDATION_SEEDS = FORMAL_VALIDATION_SEEDS
# Standalone checkpoint protection remains at the existing five-percent
# budget. Webots uses a separately registered per-policy budget so changing
# the fine-tune rule cannot silently weaken standalone training protection.
STANDALONE_COUNT_RELATIVE_DEGRADATION_TOLERANCE = 0.05
STANDALONE_UTILITY_RELATIVE_DEGRADATION_TOLERANCE = 0.05
TUNING_TRAINING_SEEDS = tuple(range(11000, 11005))
TUNING_VALIDATION_SEEDS = tuple(range(12000, 12002))
HYPERPARAMETER_TRIALS_MAXIMUM = 12
PARALLEL_HYPERPARAMETER_TRIAL_WORKERS = 2
PARALLEL_FROZEN_VALIDATION_WORKERS = 3
PARALLEL_HEADLESS_MODEL_WORKERS = 2
PARALLEL_HEADLESS_PROCESS_BUDGET = 6
V16_PARALLEL_HEADLESS_EXECUTION_CONTRACT = (
    "isolated-trials-parent-state-writer-v1")
PARALLEL_HEADLESS_EXECUTION_CONTRACT = (
    "two-model-isolated-artifacts-parent-state-writer-serial-physical-v2")
STANDALONE_EXIT_OVERRIDE_VERSION = (
    "operator-requested-webots-transition-v1")
HYPERPARAMETER_GROUP_AUDIT_VERSION = (
    "hyperparameter-group-audit-v2-six-validation-episodes")
AI_NEAREST_NEIGHBOUR_REVIEW_RATIO = 0.70
HYPERPARAMETER_TRIAL_COUNTS = {
    policy_id: min(HYPERPARAMETER_TRIALS_MAXIMUM, space.size)
    for policy_id, space in SEARCH_SPACES.items()
}
CALIBRATION = (
    ROOT / "config" / "formal_1800" / "calibration.json")
MANIFEST_ROOT = (
    ROOT / "config" / "formal_1800" / "manifests")

_SEED_PARTITIONS = {
    "tuning_training": set(TUNING_TRAINING_SEEDS),
    "tuning_validation": set(TUNING_VALIDATION_SEEDS),
    "standalone_training": set(TRAINING_SEEDS),
    "standalone_validation": set(VALIDATION_SEEDS),
    "webots_finetune": set(WEBOTS_FINETUNE_SEEDS),
    "webots_regression": set(WEBOTS_REGRESSION_SEEDS),
    "final_evaluation": set(FINAL_EVALUATION_SEEDS),
}
for _left_name, _left_values in _SEED_PARTITIONS.items():
    for _right_name, _right_values in _SEED_PARTITIONS.items():
        if (_left_name < _right_name and
                _left_values.intersection(_right_values)):
            raise RuntimeError(
                f"seed partitions overlap: {_left_name}/{_right_name}")


def _source(*parts: str) -> Path:
    return ROOT.joinpath("results", "all_algorithms_formal", *parts)


_AI_BASE_SPECS = {
    "LearnedHungarian": {
        "extension": ".npz",
        "source": _source("ml", "candidates", "learned_cost.npz")},
    "GraphImitation": {
        "extension": ".npz",
        "source": _source("ml", "candidates", "graph_imitation.npz")},
    "GraphPPO": {
        "extension": ".npz",
        "sources": {
            "count": _source("dual_auto", "count", "winner", "model.npz"),
            "utility_v2": _source(
                "dual_auto", "utility_v2", "winner", "model.npz")}},
    "RainbowDQN": {
        "extension": ".npz",
        "source": _source("advanced_ai", "candidates", "rainbow_dqn.npz")},
    "QRDQN": {
        "extension": ".npz",
        "source": _source("advanced_ai", "candidates", "qrdqn.npz")},
    "CQL": {
        "extension": ".npz",
        "source": _source("advanced_ai", "candidates", "cql.npz")},
    "LinUCB": {
        "extension": ".npz",
        "source": _source("advanced_ai", "candidates", "linucb.npz")},
    "PPO_RL": {
        "extension": ".npz",
        "source": _source("legacy_rl", "candidates", "ppo_pairwise.npz")},
    "SARSA": {
        "extension": ".json",
        "source": _source("legacy_rl", "candidates", "sarsa.json")},
    "DQN": {
        "extension": ".pkl",
        "source": _source("legacy_rl", "candidates", "dqn.pkl")},
}


def _dual_objective_spec(scheduler: str, objective: str) -> dict:
    base = _AI_BASE_SPECS[scheduler]
    source = base.get("source")
    if source is None:
        source = base["sources"][objective]
    return {
        "policy_id": f"{scheduler}_{objective}",
        "scheduler": scheduler,
        "objective": objective,
        "extension": base["extension"],
        "source": source,
    }


AI_SPECS = tuple(
    _dual_objective_spec(scheduler, objective)
    for scheduler in TRAINABLE_ALGORITHMS
    for objective in TRAINABLE_OBJECTIVES)
if tuple(spec["policy_id"] for spec in AI_SPECS) != tuple(SEARCH_SPACES):
    raise RuntimeError("workflow and dual-objective search registries differ")

SUPERVISED_WEBOTS_ALGORITHMS = frozenset({
    "LearnedHungarian", "GraphImitation",
})

# Finish one objective-specific model before starting the next.  LinUCB has
# the smallest non-supervised search grid, so it runs first to produce the
# earliest useful end-to-end result.  The two supervised families run last:
# their Webots update consumes a validated, training-partition-only corpus
# from every non-supervised model.
PIPELINE_ALGORITHM_ORDER = (
    "LinUCB", "GraphPPO", "RainbowDQN", "QRDQN", "CQL", "PPO_RL",
    "SARSA", "DQN", "LearnedHungarian", "GraphImitation",
)
if (len(PIPELINE_ALGORITHM_ORDER) != len(TRAINABLE_ALGORITHMS) or
        set(PIPELINE_ALGORITHM_ORDER) != set(TRAINABLE_ALGORITHMS) or
        set(PIPELINE_ALGORITHM_ORDER[-2:]) !=
            SUPERVISED_WEBOTS_ALGORITHMS):
    raise RuntimeError("model pipeline algorithm order is incomplete")
PIPELINE_AI_SPECS = tuple(
    next(
        spec for spec in AI_SPECS
        if spec["scheduler"] == scheduler and
        spec["objective"] == objective)
    for scheduler in PIPELINE_ALGORITHM_ORDER
    for objective in TRAINABLE_OBJECTIVES
)
PIPELINE_POLICY_IDS = tuple(
    spec["policy_id"] for spec in PIPELINE_AI_SPECS)
if (len(PIPELINE_POLICY_IDS) != len(AI_SPECS) or
        len(set(PIPELINE_POLICY_IDS)) != len(AI_SPECS) or
        set(PIPELINE_POLICY_IDS) != {
            spec["policy_id"] for spec in AI_SPECS}):
    raise RuntimeError("model pipeline policy order is not a permutation")
DIRECT_FINETUNED_EVALUATION_POLICY_IDS = PIPELINE_POLICY_IDS
DIRECT_FINETUNED_SELECTION_VERSION = (
    "direct-finetuned-final-evaluation-selection-v1")

EVALUATED_NON_AI_SCHEDULERS = ("FCFS", "NearestNeighbour")
if (not set(EVALUATED_NON_AI_SCHEDULERS).issubset(
        run_experiments.SCHEDULER_TYPES) or
        set(EVALUATED_NON_AI_SCHEDULERS).intersection(TRAINABLE_ALGORITHMS)):
    raise RuntimeError("non-AI evaluation roster is invalid")
FINAL_EVALUATION_POLICY_COUNT = (
    len(AI_SPECS) + len(EVALUATED_NON_AI_SCHEDULERS))

ADVANCED_VALUE = {
    "RainbowDQN": RainbowDQNAgent,
    "QRDQN": QRDQNAgent,
    "CQL": CQLAgent,
}


def atomic_json(path: Path, document) -> None:
    webots_workflow.atomic_json(path, document)


def atomic_copy(source: Path, destination: Path) -> None:
    webots_workflow.atomic_copy(source, destination)


def load_json(path: Path):
    return webots_workflow.load_json(path)


def sha256_file(path: Path) -> str:
    return webots_workflow.sha256_file(path)


MODEL_IDENTITY_VERSION = (
    "dual-objective-model-identity-v8-convergence-bounded-regression")
EVALUATION_IDENTITY_VERSION = "dual-metric-evaluation-identity-v2-common-seeds"
STANDALONE_NONREGRESSION_GUARD_VERSION = (
    "all-ai-standalone-bounded-regression-v5-ineligible-candidate-rollback")
# Kept as a compatibility name for audit readers/tests written while the
# defect was first isolated on GraphPPO.  The guard now covers every AI
# scheduler and both objective-specific model identities.
GRAPH_PPO_STANDALONE_GUARD_VERSION = (
    STANDALONE_NONREGRESSION_GUARD_VERSION)


def training_parameters(spec: dict) -> dict:
    """Return the stable formal configuration for one objective instance."""
    space = SEARCH_SPACES[spec["policy_id"]]
    defaults = {
        name: values[len(values)//2]
        for name, values in space.choices
    }
    selected = spec.get("training_parameters")
    if selected is None:
        return defaults
    if not isinstance(selected, dict) or set(selected) != set(defaults):
        raise ValueError("selected training parameters do not match search space")
    for name, values in space.choices:
        if selected[name] not in values:
            raise ValueError("selected training parameter is outside search space")
    return dict(selected)


def training_config_sha256(spec: dict) -> str:
    payload = {
        "training_semantics_version": TRAINING_SEMANTICS_VERSION,
        "formal_protocol_sha256": formal_protocol_sha256(),
        "policy_id": spec["policy_id"],
        "objective": spec["objective"],
        "search_space_sha256": SEARCH_SPACES[spec["policy_id"]].sha256,
        "parameters": training_parameters(spec),
        "trial_config_sha256": spec.get("trial_config_sha256"),
    }
    if spec.get("traffic_version"):
        payload["traffic_version"] = spec["traffic_version"]
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def model_identity_path(model_path: Path) -> Path:
    path = Path(model_path)
    return path.with_name(path.name + ".identity.json")


def _write_model_identity(spec: dict, model_path: Path) -> dict:
    path = Path(model_path).resolve()
    document = {
        "version": MODEL_IDENTITY_VERSION,
        "training_semantics_version": TRAINING_SEMANTICS_VERSION,
        "policy_id": spec["policy_id"],
        "scheduler": spec["scheduler"],
        "objective": spec["objective"],
        "search_space_sha256": SEARCH_SPACES[spec["policy_id"]].sha256,
        "training_parameters": training_parameters(spec),
        "training_config_sha256": training_config_sha256(spec),
        "trial_config_sha256": spec.get("trial_config_sha256"),
        "checkpoint_sha256": sha256_file(path),
    }
    atomic_json(model_identity_path(path), document)
    return document


def validate_model_identity(spec: dict, model_path: Path) -> dict:
    path = Path(model_path).resolve()
    identity = load_json(model_identity_path(path))
    expected = {
        "version": MODEL_IDENTITY_VERSION,
        "training_semantics_version": TRAINING_SEMANTICS_VERSION,
        "policy_id": spec["policy_id"],
        "scheduler": spec["scheduler"],
        "objective": spec["objective"],
        "search_space_sha256": SEARCH_SPACES[spec["policy_id"]].sha256,
        "training_parameters": training_parameters(spec),
        "training_config_sha256": training_config_sha256(spec),
        "trial_config_sha256": spec.get("trial_config_sha256"),
        "checkpoint_sha256": sha256_file(path),
    }
    if identity != expected:
        raise ModelValidationError(
            "checkpoint objective identity is missing or mismatched")
    return identity


def atomic_copy_model(spec: dict, source: Path, destination: Path) -> None:
    validate_model_identity(spec, source)
    atomic_copy(source, destination)
    _write_model_identity(spec, destination)
    validate_model_identity(spec, destination)


def evaluation_identity_path(result_path: Path) -> Path:
    return Path(result_path).with_name("policy_identity.json")


def ensure_evaluation_identity(policy: dict, result_path: Path,
                               scenario: str, seed: int) -> dict:
    result_path = Path(result_path).resolve()
    document = {
        "version": EVALUATION_IDENTITY_VERSION,
        "policy_id": policy["policy_id"],
        "scheduler": policy["scheduler"],
        "training_objective": policy.get("training_objective"),
        "evaluation_metrics": ["count", "utility_v2"],
        "evaluation_seed_contract_sha256":
            final_evaluation_seed_contract_sha256(),
        "model_sha256": policy.get("model_sha256"),
        "model_identity_sha256": policy.get("model_identity_sha256"),
        "scenario": scenario,
        "seed": seed,
        "result_sha256": sha256_file(result_path),
    }
    path = evaluation_identity_path(result_path)
    existing = load_json(path)
    if existing is None and path.exists():
        raise RuntimeError("final evaluation identity is unreadable")
    if existing is None:
        atomic_json(path, document)
    elif existing != document:
        raise RuntimeError(
            "final evaluation policy/objective identity mismatch")
    return document


def manifest_path(partition: str, scenario: str, seed: int) -> Path:
    return MANIFEST_ROOT / partition / scenario / f"{seed}.json"


def _canonical_sha256(document) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_policy_id_subset(values) -> list[str]:
    """Normalize an operator policy list without weakening identity checks."""
    if values is None:
        return []
    if isinstance(values, str):
        raise ValueError("policy-id collection must not be a string")
    requested = list(values)
    if any(not isinstance(value, str) for value in requested):
        raise ValueError("operator transition policy ids must be strings")
    unknown = sorted(set(requested).difference(PIPELINE_POLICY_IDS))
    if unknown:
        raise ValueError(
            "unknown operator transition policy ids: " + ", ".join(unknown))
    selected = set(requested)
    return [policy_id for policy_id in PIPELINE_POLICY_IDS
            if policy_id in selected]


def _standalone_exit_override_path(output_dir: Path, spec: dict) -> Path:
    return (Path(output_dir) / "headless" / spec["policy_id"] /
            "standalone_exit_override.json")


def _validate_standalone_exit_override(spec: dict, request: dict) -> dict:
    if not isinstance(request, dict):
        raise RuntimeError("standalone exit override is unreadable")
    unsigned = dict(request)
    report_sha256 = unsigned.pop("report_sha256", None)
    if (report_sha256 != _canonical_sha256(unsigned) or
            request.get("version") != STANDALONE_EXIT_OVERRIDE_VERSION or
            request.get("status") != "requested" or
            request.get("action") !=
                "enter_webots_without_further_standalone_training" or
            request.get("policy_id") != spec["policy_id"] or
            request.get("scheduler") != spec["scheduler"] or
            request.get("objective") != spec["objective"] or
            request.get("requested_by") != "user" or
            request.get("reason") !=
                "user_requested_direct_webots_transition" or
            not isinstance(request.get("requested_at"), str) or
            not request["requested_at"]):
        raise RuntimeError(
            f"standalone exit override identity is invalid: "
            f"{spec['policy_id']}")
    return request


def _ensure_standalone_exit_override(
        output_dir: Path, spec: dict) -> dict:
    """Persist a restart-safe, identity-bound user transition request."""
    path = _standalone_exit_override_path(output_dir, spec)
    existing = load_json(path)
    if existing is not None:
        return _validate_standalone_exit_override(spec, existing)
    request = {
        "version": STANDALONE_EXIT_OVERRIDE_VERSION,
        "status": "requested",
        "action": "enter_webots_without_further_standalone_training",
        "requested_by": "user",
        "requested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reason": "user_requested_direct_webots_transition",
        "policy_id": spec["policy_id"],
        "scheduler": spec["scheduler"],
        "objective": spec["objective"],
    }
    request["report_sha256"] = _canonical_sha256(request)
    atomic_json(path, request)
    return _validate_standalone_exit_override(spec, load_json(path))


def _load_standalone_exit_override(
        output_dir: Path, spec: dict) -> dict | None:
    path = _standalone_exit_override_path(output_dir, spec)
    request = load_json(path)
    if request is None:
        return None
    request = _validate_standalone_exit_override(spec, request)
    state = load_json(Path(output_dir) / "workflow_state.json")
    if not (
            isinstance(state, dict) and
            state.get("schema_version") == WORKFLOW_STATE_SCHEMA_VERSION and
            state.get("workflow_contract") == WORKFLOW_CONTRACT and
            spec["policy_id"] in state.get(
                "standalone_exit_to_webots_policy_ids", [])):
        raise RuntimeError(
            f"standalone exit override is not registered by workflow state: "
            f"{spec['policy_id']}")
    return request


def _operator_webots_readiness(
        spec: dict, natural_readiness: dict, request: dict) -> dict:
    """Allow a user-requested physical transition without faking convergence."""
    _validate_standalone_exit_override(spec, request)
    if not isinstance(natural_readiness, dict):
        raise RuntimeError("operator transition has no readiness evidence")
    unsigned = dict(natural_readiness)
    natural_report_sha256 = unsigned.pop("report_sha256", None)
    if natural_report_sha256 != _canonical_sha256(unsigned):
        raise RuntimeError("operator transition readiness hash mismatch")
    allowed_statuses = {
        "continue_training", "research_candidate_not_converged",
        "qualified_for_webots_fine_tune",
        "maximum_completed_for_webots_fine_tune",
    }
    completed_rounds = natural_readiness.get("completed_balanced_rounds")
    best_round = natural_readiness.get("best_validation_round")
    if (natural_readiness.get("policy_id") != spec["policy_id"] or
            natural_readiness.get("objective") != spec["objective"] or
            natural_readiness.get("status") not in allowed_statuses or
            isinstance(completed_rounds, bool) or
            not isinstance(completed_rounds, int) or
            completed_rounds < MIN_STANDALONE_BALANCED_ROUNDS or
            isinstance(best_round, bool) or not isinstance(best_round, int) or
            best_round < MIN_STANDALONE_BALANCED_ROUNDS or
            best_round > completed_rounds or
            natural_readiness.get("invalid_training_evidence") or
            natural_readiness.get("invalid_validation_evidence") or
            natural_readiness.get("missing_validation_rounds") or
            natural_readiness.get("training_validation_seed_overlap") is not
                False):
        raise RuntimeError(
            f"{spec['policy_id']} lacks safe standalone evidence for an "
            "operator-requested Webots transition")
    effective = dict(natural_readiness)
    effective.update({
        "status": "operator_requested_webots_transition",
        "natural_status": natural_readiness["status"],
        "natural_webots_fine_tune_eligible": bool(
            natural_readiness.get("webots_fine_tune_eligible")),
        "webots_fine_tune_eligible": True,
        "standalone_exit_mode": "operator_requested_webots_transition",
        "standalone_exit_override_version":
            STANDALONE_EXIT_OVERRIDE_VERSION,
        "standalone_exit_override_sha256": request["report_sha256"],
        "natural_readiness_report_sha256": natural_report_sha256,
    })
    effective.pop("report_sha256", None)
    effective["report_sha256"] = _canonical_sha256(effective)
    return effective


def ensure_formal_manifests() -> None:
    """Create deterministic, purpose-separated manifests when absent."""
    partitions = (
        ("standalone_tuning_training", TUNING_TRAINING_SEEDS),
        ("standalone_tuning_validation", TUNING_VALIDATION_SEEDS),
        ("standalone_training", TRAINING_SEEDS),
        ("standalone_validation", VALIDATION_SEEDS),
        ("webots_finetune", WEBOTS_FINETUNE_SEEDS),
        ("webots_regression", WEBOTS_REGRESSION_SEEDS),
        ("final_evaluation", FINAL_EVALUATION_SEEDS),
    )
    for partition, seeds in partitions:
        for scenario in SCENARIOS:
            config = formal_scenario_config(scenario)
            for seed in seeds:
                path = manifest_path(partition, scenario, seed)
                generated = generate_task_manifest(
                    scenario, config, seed,
                    duration_seconds=DURATION_SECONDS)
                if path.is_file():
                    existing = _load_manifest(path)
                    if (existing.get("manifest_sha256") !=
                            generated["manifest_sha256"]):
                        raise RuntimeError(
                            f"canonical manifest drift detected: {path}")
                    continue
                write_task_manifest(path, generated)


def _load_manifest(path: Path) -> dict:
    document = load_json(path)
    if not isinstance(document, dict):
        raise ValueError(f"invalid task manifest: {path}")
    return document


def _load_calibration() -> dict:
    """Load the immutable formal protocol and verify both embedded hashes."""
    document = _load_manifest(CALIBRATION)
    for field, hash_field in (
            ("utility_reward_profile", "utility_reward_profile_sha256"),
            ("utility_score_config", "utility_score_config_sha256")):
        values = document.get(field)
        expected = document.get(hash_field)
        if (not isinstance(values, dict) or not isinstance(expected, str) or
                len(expected) != 64 or _canonical_sha256(values) != expected):
            raise ValueError(f"formal calibration hash mismatch: {field}")
    return document


def formal_protocol_sha256() -> str:
    return _canonical_sha256(_load_calibration())


def final_evaluation_seed_contract() -> dict:
    """Return the policy-independent common-random-numbers contract."""
    manifests = {}
    for scenario in SCENARIOS:
        for seed in FINAL_EVALUATION_SEEDS:
            path = manifest_path("final_evaluation", scenario, seed)
            if not path.is_file():
                raise FileNotFoundError(
                    f"final evaluation manifest is missing: {path}")
            manifests[f"{scenario}:{seed}"] = sha256_file(path)
    return {
        "version": "final-evaluation-common-random-numbers-v1",
        "policy_independent": True,
        "scenarios": list(SCENARIOS),
        "seeds": list(FINAL_EVALUATION_SEEDS),
        "manifest_file_sha256_by_scenario_seed": manifests,
    }


def final_evaluation_seed_contract_sha256() -> str:
    return _canonical_sha256(final_evaluation_seed_contract())


def utility_profile() -> UtilityV2RewardProfile:
    document = _load_calibration()
    values = document.get("utility_reward_profile")
    if not isinstance(values, dict):
        raise ValueError("calibration has no Utility reward profile")
    allowed = {item.name for item in fields(UtilityV2RewardProfile)}
    profile = UtilityV2RewardProfile(**{
        key: value for key, value in values.items() if key in allowed})
    if profile.horizon_seconds != DURATION_SECONDS:
        raise ValueError("Utility reward profile is not a 1800s profile")
    return profile


def utility_score_config() -> UtilityScoreConfig:
    document = _load_calibration()
    values = document.get("utility_score_config")
    if not isinstance(values, dict):
        raise ValueError("calibration has no Utility score configuration")
    config = UtilityScoreConfig(**values)
    if config.horizon_seconds != DURATION_SECONDS:
        raise ValueError("Utility score configuration is not a 1800s profile")
    return config


def _episode_environment(manifest: dict, scenario: str, seed: int,
                         objective: str):
    robots, tasks, context = factory_scenario(
        seed, scenario_id=scenario, duration_seconds=DURATION_SECONDS,
        manifest=manifest)
    step_limit = max(10000, len(tasks) * 30 + 100)
    profile = (CountRewardProfile(horizon_seconds=DURATION_SECONDS)
               if objective == "count" else utility_profile())
    environment = SchedulingEnvironment(
        RLEnvironmentConfig(
            max_robots=8, max_tasks=20,
            max_steps_per_episode=step_limit),
        reward_profile=profile,
        simulation_mode="headless",
        runtime_config=HeadlessRuntimeConfig(
            episode_end_time=DURATION_SECONDS,
            max_advance_seconds=DURATION_SECONDS,
            fixed_horizon=True))
    return environment, robots, tasks, context, step_limit, profile


def _dimensions():
    environment = SchedulingEnvironment(simulation_mode="webots")
    return (environment.observation_dim, environment.action_dim,
            environment.no_op_action)


def _ppo_hidden_size(path: Path) -> int:
    with np.load(path, allow_pickle=False) as archive:
        return int(archive["hidden_size"][0])


def load_model(spec: dict, path: Path, seed: int, *,
               require_identity: bool | None = None):
    path = Path(path).resolve()
    if path == Path(spec["source"]).resolve():
        raise ModelValidationError(
            "legacy source checkpoints are family markers only and may not "
            "be loaded by the from-scratch workflow")
    if require_identity is None:
        require_identity = True
    if require_identity:
        validate_model_identity(spec, path)
    state_dim, action_dim, no_op = _dimensions()
    name = spec["scheduler"]
    if name == "LearnedHungarian":
        model = RidgeCostModel.load(path)
        if model.uses_traffic != bool(spec.get("traffic_version")):
            raise ModelValidationError("Ridge traffic training contract mismatch")
        if model.uses_traffic and model.target_name != f"{spec['objective']}_expert_cost":
            raise ModelValidationError("Ridge objective target mismatch")
        return model
    if name == "GraphImitation":
        model = GraphEdgeImitationModel.load(path)
        if model.uses_traffic != bool(spec.get("traffic_version")):
            raise ModelValidationError("graph traffic training contract mismatch")
        return model
    if name == "GraphPPO":
        return GraphPPOModel.load(path, seed=seed)
    if name in ADVANCED_VALUE:
        return ADVANCED_VALUE[name].load(
            path, state_dim, action_dim, no_op, seed=seed)
    if name == "LinUCB":
        return LinUCBModel.load(path)
    if name == "PPO_RL":
        model = PPONetwork(state_dim, action_dim, _ppo_hidden_size(path))
        model.load(str(path))
        return model
    if name == "SARSA":
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError("invalid SARSA checkpoint")
        allowed = SarsaConfig.__dataclass_fields__
        config = SarsaConfig(**{
            key: value for key, value in payload.get("config", {}).items()
            if key in allowed})
        model = SarsaAgent(action_dim, no_op, config=config, seed=seed)
        model.load(str(path))
        return model
    if name == "DQN":
        model = DQNAgent(
            state_dim, action_dim, no_op,
            dqn_config_from_checkpoint(str(path)), seed=seed)
        model.load(str(path))
        return model
    raise ValueError(f"unsupported trainable algorithm: {name}")


def _policy_parameter_tensors(spec: dict, checkpoint: Path) -> dict[str, np.ndarray]:
    """Return only parameters that can change the scheduler's decisions.

    Optimizer moments, replay buffers, counters, RNG state, and checkpoint
    metadata are deliberately excluded: changes to training state are not
    evidence that the policy itself is still moving.
    """
    model = load_model(spec, checkpoint, seed=0)
    name = spec["scheduler"]
    if name == "LearnedHungarian":
        tensors = {
            "mean": model.mean, "scale": model.scale,
            "coefficients": model.coefficients,
            "intercept": np.asarray([model.intercept]),
        }
    elif name == "GraphImitation":
        tensors = {
            "mean": model.mean, "scale": model.scale,
            "weights": model.weights,
            "intercept": np.asarray([model.intercept]),
        }
    elif name == "GraphPPO":
        tensors = {
            f"network.{key}": value
            for key, value in model.network.params.items()}
    elif name in ADVANCED_VALUE or name == "DQN":
        tensors = {
            f"online.{key}": value
            for key, value in model.online.params.items()}
    elif name == "LinUCB":
        theta = np.stack([
            np.linalg.solve(matrix, rewards)
            for matrix, rewards in zip(model.covariance, model.reward_sum)
        ])
        precision = np.stack([
            np.linalg.inv(matrix) for matrix in model.covariance])
        tensors = {
            "posterior.theta": theta,
            "posterior.exploration_precision": model.alpha * precision,
        }
    elif name == "PPO_RL":
        tensors = {
            key: getattr(model, key) for key in (
                "W1", "b1", "W2", "b2", "W_policy", "b_policy",
                "W_value", "b_value")}
    elif name == "SARSA":
        tensors = {
            "q." + "|".join(map(str, state)): values
            for state, values in model.q_table.items()}
        # An empty table is a valid bootstrap but cannot be declared converged.
        if not tensors:
            tensors = {"q.__empty__": np.zeros(model.action_dim)}
    else:  # pragma: no cover - AI_SPECS is checked at module import.
        raise ValueError(f"no parameter convergence adapter for {name}")
    normalized = {
        str(key): np.asarray(value, dtype=np.float64).reshape(-1)
        for key, value in tensors.items()}
    if (not normalized or any(
            not value.size or not np.all(np.isfinite(value))
            for value in normalized.values())):
        raise RuntimeError("policy parameter snapshot is empty or non-finite")
    return normalized


def _policy_parameter_change(spec: dict, before: Path, after: Path,
                             *, reference_round: int) -> dict:
    before_tensors = _policy_parameter_tensors(spec, before)
    after_tensors = _policy_parameter_tensors(spec, after)
    delta_squared = before_squared = after_squared = 0.0
    parameter_count = 0
    for key in sorted(set(before_tensors) | set(after_tensors)):
        left = before_tensors.get(key)
        right = after_tensors.get(key)
        if left is None:
            left = np.zeros_like(right)
        if right is None:
            right = np.zeros_like(left)
        if left.shape != right.shape:
            raise RuntimeError(f"policy parameter shape changed: {key}")
        difference = right-left
        delta_squared += float(np.dot(difference, difference))
        before_squared += float(np.dot(left, left))
        after_squared += float(np.dot(right, right))
        parameter_count += int(left.size)
    delta_l2 = math.sqrt(max(0.0, delta_squared))
    scale_l2 = max(
        math.sqrt(max(0.0, before_squared)),
        math.sqrt(max(0.0, after_squared)), 1e-12)
    relative = delta_l2/scale_l2
    if not all(math.isfinite(value) for value in (
            delta_l2, scale_l2, relative)):
        raise RuntimeError("policy parameter change is non-finite")
    return {
        "schema_version": 1,
        "reference_round": int(reference_round),
        "parameter_count": parameter_count,
        "delta_l2": delta_l2,
        "scale_l2": scale_l2,
        "relative_l2_change": relative,
    }


def _fresh_model(spec: dict, seed: int):
    """Create a repaired-contract bootstrap model for a legacy source only."""
    state_dim, action_dim, no_op = _dimensions()
    name = spec["scheduler"]
    parameters = training_parameters(spec)
    if name == "GraphPPO":
        return GraphPPOModel(GraphPPOConfig(**parameters), seed=seed)
    if name in ADVANCED_VALUE:
        config_type = {
            "RainbowDQN": RainbowConfig,
            "QRDQN": QRDQNConfig,
            "CQL": CQLConfig,
        }[name]
        return ADVANCED_VALUE[name](
            state_dim, action_dim, no_op,
            config=config_type(**parameters), seed=seed)
    if name == "LinUCB":
        return LinUCBModel.create(alpha=parameters["alpha"])
    if name == "PPO_RL":
        return PPONetwork(
            state_dim, action_dim, hidden_size=parameters["hidden_size"])
    if name == "SARSA":
        return SarsaAgent(
            action_dim, no_op, config=SarsaConfig(**parameters), seed=seed)
    if name == "DQN":
        return DQNAgent(
            state_dim, action_dim, no_op,
            config=DQNConfig(**parameters), seed=seed)
    raise ValueError(f"no bootstrap constructor for {name}")


def _load_training_model(spec: dict, path: Path, seed: int):
    if Path(path).resolve() == Path(spec["source"]).resolve():
        # Formal v11 runs are intentionally from-scratch. Legacy candidates
        # identify the algorithm family only; none of their learned parameters,
        # replay, optimizer state, or exploration state may enter a new trial.
        return _fresh_model(spec, seed)
    return load_model(spec, path, seed)


def _training_input_provenance(spec: dict, path: Path, seed: int) -> dict:
    """Identify a training input without reading a legacy source checkpoint."""
    if Path(path).resolve() == Path(spec["source"]).resolve():
        kind = ("fresh_expert_bootstrap" if
                spec["scheduler"] in SUPERVISED_WEBOTS_ALGORITHMS else
                "fresh_seeded_model")
        payload = {
            "schema_version": 1,
            "kind": kind,
            "policy_id": spec["policy_id"],
            "training_config_sha256": training_config_sha256(spec),
            "seed": int(seed),
            "legacy_source_bytes_read": False,
        }
        return {**payload, "identity_sha256": _canonical_sha256(payload)}
    checkpoint = Path(path).resolve()
    return {
        "schema_version": 1,
        "kind": "checkpoint_from_current_fresh_run",
        "policy_id": spec["policy_id"],
        "training_config_sha256": training_config_sha256(spec),
        "checkpoint_path": str(checkpoint),
        "identity_sha256": sha256_file(checkpoint),
        "legacy_source_bytes_read": False,
    }


def atomic_save_model(spec: dict, model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        model.save(str(temporary))
        # A successful load is part of the checkpoint transaction.
        load_model(spec, temporary, seed=0, require_identity=False)
        os.replace(temporary, path)
        _write_model_identity(spec, path)
        validate_model_identity(spec, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_transitions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                states=np.asarray([row["state"] for row in rows], np.float32),
                action_masks=np.asarray(
                    [row["mask"] for row in rows], np.uint8),
                actions=np.asarray([row["action"] for row in rows], np.int64),
                rewards=np.asarray([row["reward"] for row in rows], np.float32),
                next_states=np.asarray(
                    [row["next_state"] for row in rows], np.float32),
                next_action_masks=np.asarray(
                    [row["next_mask"] for row in rows], np.uint8),
                dones=np.asarray([row["done"] for row in rows], np.uint8))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_transitions(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as archive:
        count = len(archive["actions"])
        return [{
            "state": archive["states"][index],
            "mask": archive["action_masks"][index].astype(bool),
            "action": int(archive["actions"][index]),
            "reward": float(archive["rewards"][index]),
            "next_state": archive["next_states"][index],
            "next_mask": archive["next_action_masks"][index].astype(bool),
            "done": bool(archive["dones"][index]),
        } for index in range(count)]


def atomic_supervised_samples(path: Path, kind: str, features,
                              targets) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    allowed_kinds = {
        "ridge_count_expert_cost",
        "ridge_utility_v2_expert_cost",
        "ridge_count_expert_cost_traffic_v1",
        "ridge_utility_v2_expert_cost_traffic_v1",
        "graph_count_imitation",
        "graph_utility_v2_imitation",
        "graph_count_imitation_traffic_v1",
        "graph_utility_v2_imitation_traffic_v1",
    }
    if (kind not in allowed_kinds or
            x.ndim != 2 or x.shape[0] != y.shape[0] or not len(y) or
            not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
        raise ValueError("invalid supervised standalone samples")
    ridge_traffic = kind.startswith("ridge_") and kind.endswith("_traffic_v1")
    if ridge_traffic and x.shape[1] != len(TRAFFIC_PAIR_FEATURE_NAMES):
        raise ValueError("invalid traffic Ridge sample contract")
    if kind.startswith("graph_") and kind.endswith("_traffic_v1") and (
            x.shape[1] != len(TRAFFIC_GRAPH_FEATURE_NAMES) or
            not np.all(np.isin(y, (0.0, 1.0)))):
        raise ValueError("invalid traffic supervised sample contract")
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, kind=np.asarray(kind, dtype=np.str_),
                features=x, targets=y,
                **({"feature_version": np.asarray(TRAFFIC_PAIR_FEATURE_VERSION),
                    "traffic_version": np.asarray(TRAFFIC_VERSION)} if ridge_traffic else {}))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_supervised_samples(path: Path, expected_kind: str):
    with np.load(path, allow_pickle=False) as archive:
        kind = str(archive["kind"].item())
        features = np.asarray(archive["features"], dtype=np.float64)
        targets = np.asarray(archive["targets"], dtype=np.float64).reshape(-1)
        if kind.startswith("ridge_") and kind.endswith("_traffic_v1"):
            if ("feature_version" not in archive or "traffic_version" not in archive or
                    str(archive["feature_version"].item()) != TRAFFIC_PAIR_FEATURE_VERSION or
                    str(archive["traffic_version"].item()) != TRAFFIC_VERSION):
                raise ValueError("traffic Ridge sample version mismatch")
    if (kind != expected_kind or features.ndim != 2 or
            features.shape[0] != targets.shape[0] or not len(targets) or
            not np.all(np.isfinite(features)) or
            not np.all(np.isfinite(targets))):
        raise ValueError("invalid supervised sample artifact")
    if kind.startswith("ridge_") and kind.endswith("_traffic_v1") and (
            features.shape[1] != len(TRAFFIC_PAIR_FEATURE_NAMES)):
        raise ValueError("invalid traffic Ridge sample contract")
    if kind.startswith("graph_") and kind.endswith("_traffic_v1") and (
            features.shape[1] != len(TRAFFIC_GRAPH_FEATURE_NAMES) or
            not np.all(np.isin(targets, (0.0, 1.0)))):
        raise ValueError("invalid traffic supervised sample contract")
    return features, targets


def _remember_rows(name: str, model, paths: list[Path]) -> int:
    if name == "CQL":
        raise RuntimeError(
            "CQL cannot restore self-generated online transitions")
    if name == "SARSA":
        return 0
    if name == "DQN":
        # DQN v2 persists its bounded replay and both RNG streams. Re-adding
        # transition artifacts would duplicate samples after every resume.
        return len(model.replay)
    count = 0
    for path in paths:
        for row in _load_transitions(path):
            if name in ADVANCED_VALUE:
                model.remember(
                    row["state"], row["mask"], row["action"], row["reward"],
                    row["next_state"], row["next_mask"], row["done"])
            count += 1
    return count


def _task_lifecycle_transitions(environment, committed_actions: list[dict],
                                episode_return: float, objective: str,
                                profile, algorithm: str):
    """Convert committed assignments into conservative action-level rows.

    The standalone runtime emits useful operational events between scheduling
    decisions.  Those step rewards must not be learned as no-op rewards: the
    training action is the assignment that owns the task lifecycle.  Shared
    horizon reward is distributed by the versioned attribution contract.
    """
    if not committed_actions:
        raise RuntimeError(
            f"{algorithm} produced no committed assignment actions")
    attribution_profile = None
    if objective == "utility_v2":
        attribution_profile = replace(
            profile, safety_observation="webots_physical")
    attribution = physical_episode_reward_attribution(
        environment._tasks, committed_actions, objective=objective,
        utility_profile=attribution_profile, collision_count=0,
        horizon_seconds=DURATION_SECONDS)
    action_rewards = attribution["row_rewards"]
    if not math.isclose(
            sum(action_rewards), episode_return,
            rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"{algorithm} action rewards do not conserve return: "
            f"actions={sum(action_rewards)!r}, episode={episode_return!r}")
    transitions = []
    for index, (row, action_reward) in enumerate(zip(
            committed_actions, action_rewards)):
        last = index == len(committed_actions)-1
        successor = row if last else committed_actions[index+1]
        transitions.append({
            "state": row["state"], "mask": row["mask"],
            "action": row["action"], "reward": float(action_reward),
            "next_state": successor["state"],
            "next_mask": successor["mask"], "done": last,
        })
    return transitions, attribution


def _collect_strict_cql_dataset(manifest: dict, scenario: str, seed: int,
                                objective: str):
    """Freeze an external deterministic behavior data set before CQL fits."""
    environment, robots, tasks, context, step_limit, profile = (
        _episode_environment(manifest, scenario, seed, objective))
    state, metadata = environment.reset(robots, tasks, context, seed=seed)
    if metadata.get("reward_profile_hash") != profile.sha256:
        raise RuntimeError("CQL collection reward profile mismatch")
    committed_actions = []
    episode_return = 0.0
    terminated = False
    for _index in range(step_limit):
        mask = environment.get_action_mask()
        action = _cost_greedy_action(environment, mask)
        assignment = environment.assignment_for_action(action)
        commit_time = float(environment.policy_snapshot()[2].current_time)
        old_state = state.copy()
        next_state, reward, terminated, truncated, info = environment.step(
            action)
        next_mask = environment.get_action_mask()
        done = bool(terminated or truncated)
        if (info.get("invalid_action") or info.get("assignment_rejected") or
                info.get("invalid_truncation") or
                (truncated and not terminated)):
            raise RuntimeError("invalid fixed-behavior CQL transition")
        if assignment is not None:
            committed_actions.append({
                "state": old_state, "mask": mask.copy(),
                "action": int(action),
                "task_id": int(assignment.task.task_id),
                "committed_at": max(
                    commit_time, float(assignment.task.arrival_time)),
            })
        episode_return += float(reward)
        state = next_state
        if done:
            break
    if not terminated or not committed_actions:
        raise RuntimeError("CQL behavior collection missed the 1800s horizon")
    _validate_runtime(environment, manifest)
    rows, attribution = _task_lifecycle_transitions(
        environment, committed_actions, episode_return, objective, profile,
        "CQL")
    manifest_hash = manifest.get("manifest_sha256")
    if not isinstance(manifest_hash, str):
        raise RuntimeError("CQL collection manifest hash is missing")
    dataset = OfflineTransitionDataset(
        np.asarray([row["state"] for row in rows], np.float32),
        np.asarray([row["mask"] for row in rows], np.uint8),
        np.asarray([row["action"] for row in rows], np.int64),
        np.asarray([row["reward"] for row in rows], np.float32),
        np.asarray([row["next_state"] for row in rows], np.float32),
        np.asarray([row["next_mask"] for row in rows], np.uint8),
        np.asarray([row["done"] for row in rows], np.uint8),
        "MaskedCostGreedy/v1", (seed,),
        np.full(len(rows), seed, dtype=np.int64),
        objective, manifest_hashes=(manifest_hash,))
    return dataset, attribution, episode_return


def _merge_frozen_cql_datasets(previous: OfflineTransitionDataset | None,
                               current: OfflineTransitionDataset, *,
                               capacity: int) -> OfflineTransitionDataset:
    """Build one immutable cumulative CQL data set without online samples."""
    if not isinstance(current, OfflineTransitionDataset):
        raise ValueError("current CQL dataset is invalid")
    if (isinstance(capacity, bool) or not isinstance(capacity, int) or
            capacity < 1):
        raise ValueError("CQL dataset capacity is invalid")
    datasets = [current] if previous is None else [previous, current]
    if previous is not None and (
            previous.state_dim != current.state_dim or
            previous.action_dim != current.action_dim or
            previous.behavior_policy != current.behavior_policy or
            previous.objective != current.objective or
            previous.collection_mode != current.collection_mode or
            previous.dataset_version != current.dataset_version):
        raise RuntimeError("CQL frozen datasets use incompatible contracts")

    arrays = {
        name: np.concatenate([getattr(dataset, name) for dataset in datasets])
        for name in (
            "states", "action_masks", "actions", "rewards", "next_states",
            "next_action_masks", "dones", "transition_seeds")
    }
    total_rows = len(arrays["states"])
    if total_rows > capacity:
        # Stable coverage-preserving thinning bounds replay cost.  Every
        # episode contributes many adjacent rows, so evenly spaced indices
        # retain old and new portions instead of silently keeping only a tail.
        indices = np.linspace(0, total_rows-1, capacity, dtype=np.int64)
        arrays = {name: values[indices] for name, values in arrays.items()}
    transition_seeds = arrays["transition_seeds"].astype(np.int64, copy=False)
    seeds = tuple(dict.fromkeys(int(value) for value in transition_seeds))
    manifest_hashes = tuple(dict.fromkeys(
        digest for dataset in datasets for digest in dataset.manifest_hashes))
    return OfflineTransitionDataset(
        arrays["states"], arrays["action_masks"], arrays["actions"],
        arrays["rewards"], arrays["next_states"],
        arrays["next_action_masks"], arrays["dones"],
        current.behavior_policy, seeds, transition_seeds,
        current.objective, current.collection_mode, manifest_hashes,
        current.dataset_version)


def _validate_runtime(environment, manifest: dict) -> dict:
    runtime = environment.runtime_telemetry()
    actual_task_hash = task_manifest_sha256(
        task.generation_parameters() for task in environment._tasks)
    if not (
            runtime.get("runtime_mode") == "headless_webots_logic" and
            runtime.get("dynamics_version") == HEADLESS_DYNAMICS_VERSION and
            runtime.get("physics_fidelity") == "business_logic_only" and
            runtime.get("joint_runtime") is True and
            runtime.get("route_planner") == "rolling_joint_grid" and
            runtime.get("fixed_horizon") is True and
            runtime.get("horizon_finalized") is True and
            runtime.get("termination_reason") == "episode_horizon" and
            math.isclose(float(runtime.get("current_time", -1)),
                         DURATION_SECONDS, rel_tol=0.0, abs_tol=1e-9) and
            actual_task_hash == manifest.get("task_manifest_sha256")):
        raise RuntimeError("headless episode failed the strict 1800s contract")
    return runtime


def _assert_formal_standalone_runtime(metrics: dict) -> dict:
    """Fail closed if any AI training branch bypasses shared robot logic."""
    runtime = metrics.get("runtime") if isinstance(metrics, dict) else None
    if not isinstance(runtime, dict) or not (
            runtime.get("runtime_mode") == "headless_webots_logic" and
            runtime.get("dynamics_version") == HEADLESS_DYNAMICS_VERSION and
            runtime.get("physics_fidelity") == "business_logic_only" and
            runtime.get("joint_runtime") is True and
            runtime.get("route_planner") == "rolling_joint_grid" and
            runtime.get("fixed_horizon") is True and
            runtime.get("horizon_finalized") is True and
            runtime.get("termination_reason") == "episode_horizon" and
            math.isclose(float(runtime.get("current_time", -1.0)),
                         DURATION_SECONDS, rel_tol=0.0, abs_tol=1e-9)):
        raise RuntimeError(
            "AI training did not use the formal standalone robot runtime")
    return runtime


def _assert_training_behavior_contract(spec: dict, metrics: dict) -> None:
    """Bind every reported training metric to the policy that produced it."""
    name = spec["scheduler"]
    rollout_policy = metrics.get("rollout_policy")
    metric_owner = metrics.get("metric_owner")
    if rollout_policy != metric_owner:
        raise RuntimeError("training metric owner differs from rollout policy")
    if name in SUPERVISED_WEBOTS_ALGORITHMS:
        bootstrap = f"Hungarian_{spec['objective']}_bootstrap"
        if rollout_policy not in {name, bootstrap} or not (
                metrics.get("supervision_expert") ==
                f"Hungarian_{spec['objective']}" and
                metrics.get("supervision_scope") == "all_feasible_edges" and
                metrics.get("training_data_policy") ==
                    f"Hungarian_{spec['objective']}_dense_labels"):
            raise RuntimeError("supervised training behavior contract drifted")
    elif rollout_policy != name or metrics.get("training_data_policy") not in {
            name, "MaskedCostGreedy/v1"}:
        raise RuntimeError("self-training behavior contract drifted")
    if name == "CQL" and metrics.get("training_data_policy") != (
            "MaskedCostGreedy/v1"):
        raise RuntimeError("CQL behavior dataset contract drifted")


def _train_pair_policy(spec: dict, input_checkpoint: Path,
                       manifest: dict, scenario: str, seed: int,
                       prior_transitions: list[Path]):
    name = spec["scheduler"]
    model = _load_training_model(spec, input_checkpoint, seed)
    environment, robots, tasks, context, step_limit, profile = (
        _episode_environment(manifest, scenario, seed, spec["objective"]))
    state, metadata = environment.reset(robots, tasks, context, seed=seed)
    if metadata["reward_profile_hash"] != profile.sha256:
        raise RuntimeError("wrong reward profile installed")
    offline_dataset = None
    if name == "CQL":
        (current_offline_dataset, offline_attribution,
         offline_episode_return) = _collect_strict_cql_dataset(
            manifest, scenario, seed, spec["objective"])
        previous_offline_dataset = None
        if prior_transitions:
            previous_offline_dataset = OfflineTransitionDataset.load(
                prior_transitions[-1])
        offline_dataset = _merge_frozen_cql_datasets(
            previous_offline_dataset, current_offline_dataset,
            capacity=model.config.replay_capacity)
        contract = model.ingest_offline_dataset(offline_dataset)
        replay_rows = len(offline_dataset.states)
        current_offline_rows = len(current_offline_dataset.states)
        offline_losses = [
            model.train_step() for _ in range(current_offline_rows)]
        losses = [
            float(loss) for loss in offline_losses if loss is not None]
        warmup_rows = max(model.config.batch_size, model.config.warmup_steps)
        if ((replay_rows >= warmup_rows and not losses) or
                not np.all(np.isfinite(losses))):
            raise RuntimeError("CQL produced no finite offline update")
    else:
        replay_rows = _remember_rows(name, model, prior_transitions)
        contract = None
        losses = []
    transitions = []
    committed_actions = []
    episode_return = 0.0
    terminated = False

    if name == "SARSA":
        discrete = model.discretize(state)
        action = model.select_action(
            discrete, environment.get_action_mask(), training=True)

    for _index in range(step_limit):
        mask = environment.get_action_mask()
        old_state = state.copy()
        if name in ADVANCED_VALUE:
            action = (model.select_action(state, mask, training=True)
                      if name != "CQL" else model.select_action(state, mask))
            assignment = environment.assignment_for_action(action)
            commit_time = float(
                environment.policy_snapshot()[2].current_time)
        elif name == "DQN":
            action = model.select_action(state, mask, training=True)
            assignment = environment.assignment_for_action(action)
            commit_time = float(
                environment.policy_snapshot()[2].current_time)
        elif name == "SARSA":
            # ``action`` was selected on-policy after the previous state.
            assignment = environment.assignment_for_action(action)
            commit_time = float(
                environment.policy_snapshot()[2].current_time)
        else:
            raise ValueError(f"not a pair-value policy: {name}")
        next_state, reward, terminated, truncated, info = environment.step(
            action)
        next_mask = environment.get_action_mask()
        done = bool(terminated or truncated)
        if (info.get("invalid_action") or info.get("assignment_rejected") or
                info.get("invalid_truncation") or
                (truncated and not terminated)):
            raise RuntimeError(f"invalid strict transition for {name}")
        if name in ADVANCED_VALUE or name in {"SARSA", "DQN"}:
            if assignment is not None:
                committed = {
                    "state": old_state,
                    "mask": mask.copy(), "action": int(action),
                    "task_id": int(assignment.task.task_id),
                    "committed_at": max(
                        commit_time, float(assignment.task.arrival_time)),
                }
                if name == "SARSA":
                    committed["discrete_state"] = discrete
                committed_actions.append(committed)
        if name in ADVANCED_VALUE and name != "CQL":
            # Formal learning happens after the 1800s horizon so every
            # committed assignment receives its complete task-lifecycle
            # reward.  Intermediate no-op/event steps are not training rows.
            loss = None
        elif name == "CQL":
            # Evaluation is separated from the already-frozen behavior data.
            # No policy-generated transition enters CQL replay.
            loss = None
        elif name == "DQN":
            # Formal DQN learns after the horizon, when task lifecycle reward
            # can be attributed to the actual committed assignment actions.
            loss = None
        else:
            next_discrete = model.discretize(next_state)
            next_action = model.select_action(
                next_discrete, next_mask, training=True)
            loss = None
            discrete, action = next_discrete, next_action
        if loss is not None:
            losses.append(float(loss))
        episode_return += float(reward)
        state = next_state
        if done:
            break
    if not terminated:
        raise RuntimeError(f"{name} did not reach the 1800s horizon")
    transitions, attribution = _task_lifecycle_transitions(
        environment, committed_actions, episode_return,
        spec["objective"], profile, name)
    action_rewards = attribution["row_rewards"]

    if name in {"SARSA", "DQN"}:
        for index, (row, action_reward) in enumerate(zip(
                committed_actions, action_rewards)):
            last = index == len(committed_actions)-1
            successor = row if last else committed_actions[index+1]
            if name == "SARSA":
                next_discrete = successor["discrete_state"]
                next_action = (
                    model.no_op_action if last else successor["action"])
                losses.append(abs(float(model.update(
                    row["discrete_state"], row["action"], action_reward,
                    next_discrete, next_action, last))))
            else:
                model.remember(
                    row["state"], row["action"], action_reward,
                    successor["state"], last, successor["mask"])
        if name == "SARSA":
            model.end_episode()
        else:
            for _ in transitions:
                loss = model.train_step()
                if loss is not None:
                    losses.append(float(loss))
            model.episode += 1
    elif name in {"RainbowDQN", "QRDQN"}:
        for row in transitions:
            model.remember(
                row["state"], row["mask"], row["action"], row["reward"],
                row["next_state"], row["next_mask"], row["done"])
            loss = model.train_step()
            if loss is not None:
                losses.append(float(loss))
    runtime = _validate_runtime(environment, manifest)
    metrics = {
        "episode_return": episode_return,
        "decisions": len(transitions),
        "optimizer_updates": len(losses),
        "mean_loss": statistics.fmean(losses) if losses else None,
        "replay_rows_restored": replay_rows,
        "rollout_policy": name,
        "metric_owner": name,
        "training_data_policy": (
            "MaskedCostGreedy/v1" if name == "CQL" else name),
        "runtime": runtime,
    }
    if offline_dataset is not None:
        metrics["offline_dataset"] = {
            **contract,
            "strictly_offline_updates": True,
            "evaluation_transitions_added": 0,
            "cumulative_frozen_rows": replay_rows,
            "current_episode_samples": current_offline_rows,
            "warmup_rows": warmup_rows,
            "warmup_reached": replay_rows >= warmup_rows,
            "reward_attribution": "task_lifecycle_action_level",
            "episode_return": offline_episode_return,
            "reward_conservation_error": (
                sum(offline_attribution["row_rewards"])-
                offline_episode_return),
        }
    if name in ADVANCED_VALUE or name in {"SARSA", "DQN"}:
        metrics.update({
            "reward_attribution": "task_lifecycle_action_level",
            "reward_conservation_error": (
                sum(action_rewards) - episode_return),
            "gamma": model.config.gamma,
        })
        if name in {"SARSA", "DQN"}:
            metrics.update({
                "epsilon": model.epsilon,
                "physical_epsilon_cap": model.config.physical_epsilon_cap,
            })
        if name == "SARSA":
            metrics.update({
                "epsilon_decay": model.config.epsilon_decay,
                "state_version": "sarsa-state-v2-priority-robot-task-slots",
            })
        elif name == "DQN":
            metrics.update({
                "epsilon_decay_steps": model.config.epsilon_decay_steps,
                "model_version": DQN_MODEL_VERSION,
                "replay_size": len(model.replay),
            })
        else:
            metrics.update({
                "replay_size": len(model.replay),
                "n_step": getattr(model.config, "n_step", 1),
            })
    return model, transitions, metrics, offline_dataset


def _train_ppo(spec: dict, input_checkpoint: Path, manifest: dict,
               scenario: str, seed: int):
    network = _load_training_model(spec, input_checkpoint, seed)
    environment, robots, tasks, context, step_limit, _profile = (
        _episode_environment(manifest, scenario, seed, spec["objective"]))
    state, _ = environment.reset(robots, tasks, context, seed=seed)
    rng = np.random.default_rng(seed)
    rollout = []
    action_rows = []
    episode_return = 0.0
    terminated = False
    for _index in range(step_limit):
        mask = environment.get_action_mask()
        probabilities, value = _masked_probabilities(network, state, mask)
        action = int(rng.choice(network.action_dim, p=probabilities))
        old_log = math.log(max(float(probabilities[action]), 1e-12))
        assignment = environment.assignment_for_action(action)
        commit_time = float(environment.policy_snapshot()[2].current_time)
        next_state, reward, terminated, truncated, info = environment.step(
            action)
        done = bool(terminated or truncated)
        if (info.get("invalid_action") or info.get("assignment_rejected") or
                info.get("invalid_truncation") or
                (truncated and not terminated)):
            raise RuntimeError("PPO_RL encountered an invalid transition")
        if assignment is not None:
            rollout.append((state.copy(), mask.copy(), action, 0.0,
                            old_log, value, False))
            action_rows.append({
                "task_id": int(assignment.task.task_id),
                "committed_at": max(
                    commit_time, float(assignment.task.arrival_time)),
            })
        episode_return += float(reward)
        state = next_state
        if done:
            break
    if not terminated or not rollout:
        raise RuntimeError("PPO_RL did not reach the 1800s horizon")
    attribution_profile = None
    if spec["objective"] == "utility_v2":
        attribution_profile = replace(
            _profile, safety_observation="webots_physical")
    attribution = physical_episode_reward_attribution(
        environment._tasks, action_rows, objective=spec["objective"],
        utility_profile=attribution_profile, collision_count=0,
        horizon_seconds=DURATION_SECONDS)
    action_rewards = attribution["row_rewards"]
    if not math.isclose(
            sum(action_rewards), episode_return,
            rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            "PPO action rewards do not conserve return: "
            f"actions={sum(action_rewards)!r}, episode={episode_return!r}")
    rollout = [
        row[:3] + (float(action_rewards[index]),) + row[4:6]
        + (index == len(rollout) - 1,)
        for index, row in enumerate(rollout)
    ]
    parameters = training_parameters(spec)
    update = _pairwise_ppo_update(
        network, rollout,
        learning_rate=parameters["learning_rate"],
        clip_ratio=parameters["clip_ratio"],
        value_coefficient=parameters["value_coefficient"],
        entropy_coefficient=parameters["entropy_coefficient"],
        update_epochs=parameters["update_epochs"],
        max_grad_norm=parameters["max_grad_norm"],
        gamma=parameters["gamma"],
        value_huber_delta=parameters["value_huber_delta"])
    return network, [], {
        "episode_return": episode_return, "decisions": len(rollout),
        "optimizer_updates": 1, "mean_loss": update["mean_loss"],
        "replay_rows_restored": 0,
        "reward_attribution": "task_lifecycle_action_level",
        "reward_conservation_error": sum(action_rewards)-episode_return,
        "rollout_policy": "PPO_RL",
        "metric_owner": "PPO_RL",
        "training_data_policy": "PPO_RL",
        "ppo_update": update,
        "runtime": _validate_runtime(environment, manifest),
    }


def _train_linucb(spec: dict, input_checkpoint: Path, manifest: dict,
                  scenario: str, seed: int):
    model = _load_training_model(spec, input_checkpoint, seed)
    environment, robots, tasks, context, step_limit, _profile = (
        _episode_environment(manifest, scenario, seed, spec["objective"]))
    environment.reset(robots, tasks, context, seed=seed)
    arms = {name: _arm_factory(name) for name in model.arms}
    samples = []
    episode_return = 0.0
    terminated = False
    decisions = 0
    for _index in range(step_limit):
        mask = environment.get_action_mask()
        if mask[environment.no_op_action]:
            _, reward, terminated, truncated, info = environment.step(
                environment.no_op_action)
        else:
            live_robots, live_tasks, live_context = environment.policy_snapshot()
            vector = bandit_context_vector(
                live_robots, live_tasks, live_context)
            arm_name, _scores = model.select(vector)
            decision = arms[arm_name].assign(
                list(live_tasks), live_robots, live_context)
            valid, reason = validate_assignments(
                decision.assignments, list(live_tasks), live_robots,
                live_context)
            if not decision.is_feasible or not valid or not decision.assignments:
                raise RuntimeError(f"LinUCB arm rejected: {reason}")
            selected = decision.assignments[0]
            action = environment.action_for_pair(
                selected.robot_id, selected.task.task_id)
            _, reward, terminated, truncated, info = environment.step(action)
            samples.append({
                "context": vector,
                "arm": arm_name,
                "task_id": int(selected.task.task_id),
                # Visibility admits a 1e-9 boundary tolerance.  Persist the
                # canonical event time, never a value microscopically before
                # the task's declared arrival.
                "committed_at": max(
                    float(live_context.current_time),
                    float(selected.task.arrival_time)),
            })
            decisions += 1
        if (info.get("invalid_action") or info.get("assignment_rejected") or
                info.get("invalid_truncation") or
                (truncated and not terminated)):
            raise RuntimeError("LinUCB encountered an invalid transition")
        episode_return += float(reward)
        if terminated or truncated:
            break
    if not terminated or not samples:
        raise RuntimeError("LinUCB did not reach the 1800s horizon")
    attribution_profile = None
    if spec["objective"] == "utility_v2":
        attribution_profile = replace(
            _profile, safety_observation="webots_physical")
    attribution = physical_episode_reward_attribution(
        environment._tasks, samples, objective=spec["objective"],
        utility_profile=attribution_profile, collision_count=0,
        horizon_seconds=DURATION_SECONDS)
    action_rewards = attribution["row_rewards"]
    if not math.isclose(
            sum(action_rewards), episode_return,
            rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            "LinUCB action rewards do not conserve return: "
            f"actions={sum(action_rewards)!r}, episode={episode_return!r}")
    selection_counts = {arm: 0 for arm in model.arms}
    reward_by_arm = {arm: [] for arm in model.arms}
    for sample, action_reward in zip(samples, action_rewards):
        model.update(sample["context"], sample["arm"], action_reward)
        selection_counts[sample["arm"]] += 1
        reward_by_arm[sample["arm"]].append(float(action_reward))
    return model, [], {
        "episode_return": episode_return, "decisions": decisions,
        "optimizer_updates": len(samples), "mean_loss": None,
        "replay_rows_restored": 0,
        "reward_attribution": "task_lifecycle_action_level",
        "reward_conservation_error": sum(action_rewards)-episode_return,
        "rollout_policy": "LinUCB",
        "metric_owner": "LinUCB",
        "training_data_policy": "LinUCB",
        "arm_selection_counts": selection_counts,
        "arm_failure_reasons": {arm: {} for arm in model.arms},
        "mean_action_reward_by_arm": {
            arm: (statistics.fmean(values) if values else None)
            for arm, values in reward_by_arm.items()},
        "model_training_audit": model.training_audit(),
        "runtime": _validate_runtime(environment, manifest),
    }


def _supervised_behavior_pairs(name: str, model, matrix,
                               pair_features: np.ndarray,
                               graph_features: np.ndarray,
                               context) -> tuple[tuple[int, int], ...]:
    """Decode the current learned model exactly as its scheduler does."""
    if name == "LearnedHungarian":
        if model.uses_traffic:
            pair_features = traffic_pair_feature_tensor(matrix, pair_features, context)
        values = learned_cost_values(model, matrix, pair_features, context)
    elif name == "GraphImitation":
        values, _probabilities = graph_imitation_values(
            model, matrix, graph_features)
    else:
        raise ValueError("unsupported supervised behavior policy")
    return tuple(solve_hungarian(values, matrix.feasible))


def _bounded_supervised_batches(feature_batches: list[np.ndarray],
                                target_batches: list[np.ndarray], *,
                                sample_limit: int, feature_dim: int,
                                binary: bool = False
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Materialize a deterministic bounded fit set without a full-size copy.

    Episode artifacts remain complete.  Only the in-memory matrix passed to a
    supervised optimizer is bounded.  Global evenly spaced indices retain old,
    middle, and new cases.  Binary imitation labels are sampled per class so a
    large corpus cannot accidentally lose all positives (or all negatives)
    because labels have a regular within-decision ordering.
    """
    if (isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or
            sample_limit < 2):
        raise ValueError("supervised sample limit must be at least two")
    if (isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or
            feature_dim < 1):
        raise ValueError("supervised feature dimension is invalid")
    if (not feature_batches or
            len(feature_batches) != len(target_batches)):
        raise ValueError("supervised feature/target batches are not aligned")

    normalized_features = []
    normalized_targets = []
    total_rows = 0
    for features, targets in zip(feature_batches, target_batches):
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64).reshape(-1)
        if (x.ndim != 2 or x.shape != (len(y), feature_dim) or
                not len(y) or not np.all(np.isfinite(x)) or
                not np.all(np.isfinite(y))):
            raise ValueError("supervised batch is invalid")
        if binary and not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError("binary supervised targets are invalid")
        normalized_features.append(x)
        normalized_targets.append(y)
        total_rows += len(y)

    retained_rows = min(total_rows, sample_limit)
    if binary:
        class_positions = []
        offset = 0
        for value in (0.0, 1.0):
            positions = []
            offset = 0
            for targets in normalized_targets:
                local = np.flatnonzero(targets == value).astype(
                    np.int64, copy=False)
                if len(local):
                    positions.append(local + offset)
                offset += len(targets)
            class_positions.append(
                np.concatenate(positions) if positions else
                np.empty(0, dtype=np.int64))
        negative_positions, positive_positions = class_positions
        if not len(negative_positions) or not len(positive_positions):
            raise ValueError(
                "binary supervised corpus needs positive and negative rows")
        if total_rows <= sample_limit:
            selected = np.arange(total_rows, dtype=np.int64)
        else:
            # Preserve the observed class distribution while reserving at
            # least one row for either class.  The downstream graph fit also
            # applies positive class weighting.
            positive_budget = int(round(
                retained_rows * len(positive_positions) / total_rows))
            positive_budget = min(
                len(positive_positions), max(1, positive_budget))
            negative_budget = retained_rows - positive_budget
            if negative_budget < 1:
                negative_budget = 1
                positive_budget = retained_rows - 1
            if negative_budget > len(negative_positions):
                negative_budget = len(negative_positions)
                positive_budget = retained_rows - negative_budget
            if positive_budget > len(positive_positions):
                positive_budget = len(positive_positions)
                negative_budget = retained_rows - positive_budget
            positive_choice = positive_positions[np.linspace(
                0, len(positive_positions)-1, positive_budget,
                dtype=np.int64)]
            negative_choice = negative_positions[np.linspace(
                0, len(negative_positions)-1, negative_budget,
                dtype=np.int64)]
            selected = np.sort(np.concatenate(
                (positive_choice, negative_choice)))
    else:
        selected = np.linspace(
            0, total_rows-1, retained_rows, dtype=np.int64)

    selected_feature_batches = []
    selected_target_batches = []
    offset = 0
    for features, targets in zip(normalized_features, normalized_targets):
        left = int(np.searchsorted(selected, offset, side="left"))
        right = int(np.searchsorted(
            selected, offset + len(targets), side="left"))
        if right > left:
            local = selected[left:right] - offset
            selected_feature_batches.append(features[local])
            selected_target_batches.append(targets[local])
        offset += len(targets)
    fit_features = np.concatenate(selected_feature_batches, axis=0)
    fit_targets = np.concatenate(selected_target_batches, axis=0)
    if (fit_features.shape != (retained_rows, feature_dim) or
            fit_targets.shape != (retained_rows,)):
        raise RuntimeError("bounded supervised sampling lost rows")
    return fit_features, fit_targets


def _bounded_supervised_artifacts(prior_sample_paths: list[Path],
                                  current_features: np.ndarray,
                                  current_targets: np.ndarray, *, kind: str,
                                  sample_limit: int, feature_dim: int,
                                  binary: bool = False
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Read episode artifacts one at a time with coverage for every episode."""
    source_count = len(prior_sample_paths) + 1
    if source_count * 2 > sample_limit:
        raise RuntimeError(
            "supervised sample limit cannot cover every episode artifact")
    selected_features = []
    selected_targets = []
    retained = 0
    for index in range(source_count):
        if index < len(prior_sample_paths):
            features, targets = _load_supervised_samples(
                prior_sample_paths[index], kind)
        else:
            features, targets = current_features, current_targets
        sources_left = source_count - index
        quota = max(2, (sample_limit - retained) // sources_left)
        sampled_features, sampled_targets = _bounded_supervised_batches(
            [features], [targets], sample_limit=quota,
            feature_dim=feature_dim, binary=binary)
        selected_features.append(sampled_features)
        selected_targets.append(sampled_targets)
        retained += len(sampled_targets)
    # The per-episode quotas already enforce the bound.  A final pass verifies
    # dimensions/classes and gives a single optimizer matrix.
    return _bounded_supervised_batches(
        selected_features, selected_targets, sample_limit=sample_limit,
        feature_dim=feature_dim, binary=binary)


def _train_supervised_policy(spec: dict, input_checkpoint: Path,
                             manifest: dict, scenario: str, seed: int,
                             prior_sample_paths: list[Path],
                             fit_model: bool):
    """Collect dense expert labels on states visited by the learned policy.

    Only the first legacy-bootstrap episode is driven by the expert.  Once a
    compatible checkpoint exists, the learned model owns every dispatch while
    the objective expert labels the complete feasible edge set off-policy.
    This is deterministic DAgger-style state coverage and keeps the reported
    rollout metric attached to the policy that actually generated it.
    """
    name = spec["scheduler"]
    uses_traffic = bool(spec.get("traffic_version"))
    if Path(input_checkpoint).resolve() == Path(spec["source"]).resolve():
        # Supervised policies bootstrap only from labels collected in this new
        # run; no prior candidate parameters or samples are loaded.
        model = None
    else:
        model = load_model(spec, input_checkpoint, seed)
    requires_refit = bool(fit_model or model is None)
    bootstrap_expert = model is None
    environment, robots, tasks, context, step_limit, profile = (
        _episode_environment(manifest, scenario, seed, spec["objective"]))
    if uses_traffic:
        context = replace(context, configuration={
            **context.configuration, "graph_traffic_enabled": True})
    _state, metadata = environment.reset(robots, tasks, context, seed=seed)
    if metadata["reward_profile_hash"] != profile.sha256:
        raise RuntimeError("supervised training reward profile mismatch")
    feature_batches = []
    target_batches = []
    episode_return = 0.0
    decisions = 0
    terminated = False
    for _index in range(step_limit):
        mask = environment.get_action_mask()
        if mask[environment.no_op_action]:
            _state, reward, terminated, truncated, info = environment.step(
                environment.no_op_action)
        else:
            live_robots, live_tasks, live_context = (
                environment.policy_snapshot())
            matrix = build_cost_matrix(
                list(live_tasks), live_robots, live_context)
            pair_features = pair_feature_tensor(
                matrix, live_robots, list(live_tasks), live_context)
            edge_features = (None if name == "LearnedHungarian" and uses_traffic else
                             traffic_graph_feature_tensor(matrix, pair_features, live_context)
                             if uses_traffic else graph_edge_feature_tensor(matrix, pair_features))
            expert_values = (traffic_objective_expert_values(matrix, live_context, spec["objective"])
                             if uses_traffic else objective_expert_values(matrix, live_context, spec["objective"]))
            expert_pairs = tuple(solve_hungarian(
                expert_values, matrix.feasible))
            if not expert_pairs:
                raise RuntimeError("Hungarian expert produced no assignment")
            behavior_pairs = (expert_pairs if bootstrap_expert else
                              _supervised_behavior_pairs(
                                  name, model, matrix, pair_features,
                                  edge_features, live_context))
            if not behavior_pairs:
                raise RuntimeError(
                    f"{name} learned behavior produced no assignment")
            selected_row, selected_column = behavior_pairs[0]
            if uses_traffic and name == "LearnedHungarian":
                ridge_features = traffic_pair_feature_tensor(matrix, pair_features, live_context)
                supervision = {"ridge_by_objective": {spec["objective"]:
                    traffic_ridge_supervision(matrix, ridge_features, live_context, spec["objective"])}}
            elif uses_traffic:
                supervision = {"graph_by_objective": {spec["objective"]:
                    traffic_graph_supervision(
                        matrix, edge_features, live_context, spec["objective"],
                        (selected_row, selected_column))}}
            else:
                supervision = objective_supervision_batches(
                    matrix, pair_features, edge_features, live_context,
                    behavior_coordinate=(selected_row, selected_column))
            if name == "LearnedHungarian":
                batch = supervision["ridge_by_objective"][spec["objective"]]
                feature_batches.append(np.asarray(
                    batch["features"], dtype=np.float64))
                target_batches.append(np.asarray(
                    batch["targets"], dtype=np.float64))
            elif name == "GraphImitation":
                batch = supervision["graph_by_objective"][spec["objective"]]
                feature_batches.append(np.asarray(
                    batch["features"], dtype=np.float64))
                target_batches.append(np.asarray(
                    batch["labels"], dtype=np.float64))
            else:
                raise ValueError("unsupported supervised policy")
            task = matrix.tasks[selected_column]
            assignment = Assignment(
                matrix.robot_ids[selected_row], task,
                float(matrix.values[selected_row, selected_column]))
            action = environment.action_for_pair(
                assignment.robot_id, task.task_id)
            _state, reward, terminated, truncated, info = environment.step(
                action)
            decisions += 1
        if (info.get("invalid_action") or info.get("assignment_rejected") or
                info.get("invalid_truncation") or
                (truncated and not terminated)):
            raise RuntimeError("invalid supervised standalone transition")
        episode_return += float(reward)
        if terminated or truncated:
            break
    if not terminated:
        raise RuntimeError(
            f"{name} supervised episode missed the 1800s horizon")
    runtime = _validate_runtime(environment, manifest)

    if not feature_batches or not target_batches:
        raise RuntimeError("supervised standalone episode has no samples")
    features = np.concatenate(feature_batches)
    targets = np.concatenate(target_batches)
    if name == "LearnedHungarian":
        target_name = f"{spec['objective']}_expert_cost"
        kind = f"ridge_{target_name}" + ("_traffic_v1" if uses_traffic else "")
    elif name == "GraphImitation":
        target_name = None
        kind = f"graph_{spec['objective']}_imitation" + ("_traffic_v1" if uses_traffic else "")
    else:
        raise ValueError("unsupported supervised policy")
    if features.ndim != 2 or len(features) < 2:
        raise RuntimeError("supervised standalone episode has no samples")

    fit_sample_count = 0
    if requires_refit:
        sample_limit = 200_000 if name == "LearnedHungarian" else 50_000
        feature_dim = (len(TRAFFIC_PAIR_FEATURE_NAMES) if name == "LearnedHungarian" and uses_traffic
                       else len(PAIR_FEATURE_NAMES) if name == "LearnedHungarian"
                       else len(TRAFFIC_GRAPH_FEATURE_NAMES) if uses_traffic
                       else len(GRAPH_FEATURE_NAMES))
        fit_features, fit_targets = _bounded_supervised_artifacts(
            prior_sample_paths, features, targets, kind=kind,
            sample_limit=sample_limit, feature_dim=feature_dim,
            binary=name == "GraphImitation")
        fit_sample_count = len(fit_targets)
        parameters = training_parameters(spec)
        if name == "LearnedHungarian":
            model = RidgeCostModel.fit(
                fit_features, fit_targets, l2=parameters["l2"],
                target_name=target_name)
        else:
            model = GraphEdgeImitationModel.fit(
                fit_features, fit_targets,
                epochs=parameters["epochs"],
                learning_rate=parameters["learning_rate"],
                l2=parameters["l2"])
    return model, {"kind": kind, "features": features,
                   "targets": targets}, {
        "episode_return": episode_return,
        "decisions": decisions,
        "optimizer_updates": int(requires_refit),
        "mean_loss": None,
        "replay_rows_restored": 0,
        "supervised_samples_collected": len(targets),
        "supervised_fit_sample_count": fit_sample_count,
        "supervised_model_refit": requires_refit,
        "supervised_target": target_name,
        "supervision_expert": f"Hungarian_{spec['objective']}",
        "supervision_scope": "all_feasible_edges",
        "training_data_policy":
            f"Hungarian_{spec['objective']}_dense_labels",
        "rollout_policy": (
            f"Hungarian_{spec['objective']}_bootstrap" if bootstrap_expert
            else name),
        "metric_owner": (
            f"Hungarian_{spec['objective']}_bootstrap" if bootstrap_expert
            else name),
        "policy_generated_decisions": 0 if bootstrap_expert else decisions,
        "expert_bootstrap_decisions": decisions if bootstrap_expert else 0,
        "dagger_state_collection": not bootstrap_expert,
        "runtime": runtime,
    }


def run_headless_episode(spec: dict, input_checkpoint: Path,
                         output_checkpoint: Path, transition_path: Path,
                         manifest: dict, scenario: str, seed: int,
                         prior_transitions: list[Path], *,
                         fit_supervised: bool = False,
                         training_seed_partition: str =
                         "standalone_training") -> dict:
    name = spec["scheduler"]
    supervised_samples = None
    if name in {"LearnedHungarian", "GraphImitation"}:
        model, supervised_samples, metrics = _train_supervised_policy(
            spec, input_checkpoint, manifest, scenario, seed,
            prior_transitions, fit_supervised)
        transitions = []
    elif name == "GraphPPO":
        model = _load_training_model(spec, input_checkpoint, seed)
        training = train_objective_graph_ppo(
            model, spec["objective"], [(scenario, seed)],
            duration_seconds=DURATION_SECONDS,
            utility_profile=(utility_profile()
                             if spec["objective"] == "utility_v2" else None),
            manifests={(scenario, seed): manifest},
            seed_partition=training_seed_partition)
        transitions = []
        metrics = {
            "episode_return": training["mean_return"],
            "decisions": training["decisions"],
            "optimizer_updates": training["optimizer_updates"],
            "mean_loss": training["mean_loss"],
            "replay_rows_restored": 0,
            "reward_attribution": training["reward_attribution"],
            "reward_conservation_verified":
                training["reward_conservation_verified"],
            "rollout_policy": "GraphPPO",
            "metric_owner": "GraphPPO",
            "training_data_policy": "GraphPPO",
            "runtime": training["episode_reports"][0]["runtime"],
        }
    elif name in ADVANCED_VALUE or name in {"SARSA", "DQN"}:
        model, transitions, metrics, offline_dataset = _train_pair_policy(
            spec, input_checkpoint, manifest, scenario, seed,
            prior_transitions)
    elif name == "PPO_RL":
        model, transitions, metrics = _train_ppo(
            spec, input_checkpoint, manifest, scenario, seed)
    elif name == "LinUCB":
        model, transitions, metrics = _train_linucb(
            spec, input_checkpoint, manifest, scenario, seed)
    else:
        raise ValueError(f"unsupported algorithm: {name}")

    _assert_formal_standalone_runtime(metrics)
    _assert_training_behavior_contract(spec, metrics)

    if (name not in ADVANCED_VALUE and name not in {"SARSA", "DQN"} and
            name not in {"LearnedHungarian", "GraphImitation"}):
        offline_dataset = None
    elif name in {"LearnedHungarian", "GraphImitation"}:
        offline_dataset = None

    atomic_save_model(spec, model, output_checkpoint)
    if supervised_samples is not None:
        atomic_supervised_samples(
            transition_path, supervised_samples["kind"],
            supervised_samples["features"], supervised_samples["targets"])
        artifact_count = len(supervised_samples["targets"])
        artifact_type = supervised_samples["kind"]
    elif offline_dataset is not None:
        offline_dataset.save(transition_path)
        artifact_count = len(offline_dataset.states)
        artifact_type = "frozen_offline_dataset"
    elif transitions:
        atomic_transitions(transition_path, transitions)
        artifact_count = len(transitions)
        artifact_type = "policy_transitions"
    else:
        transition_path.unlink(missing_ok=True)
        artifact_count = 0
        artifact_type = None
    return {
        **metrics,
        "transition_count": artifact_count,
        "transition_artifact_type": artifact_type,
        "transition_sha256": (
            sha256_file(transition_path) if artifact_count else None),
    }


def _valid_headless_record(record, spec: dict, episode_index: int,
                           scenario: str, seed: int,
                           input_provenance: dict,
                           checkpoint: Path, transitions: Path) -> bool:
    if not isinstance(record, dict) or not checkpoint.is_file():
        return False
    try:
        validate_model_identity(spec, checkpoint)
    except (ModelValidationError, OSError, ValueError):
        return False
    metrics = record.get("metrics", {})
    runtime = metrics.get("runtime", {})
    transition_count = metrics.get("transition_count", 0)
    valid = bool(
        record.get("schema_version") == TRAINING_RECORD_SCHEMA_VERSION and
        record.get("training_semantics_version") ==
            TRAINING_SEMANTICS_VERSION and
        record.get("partition") == "training" and
        record.get("policy_id") == spec["policy_id"] and
        record.get("scheduler") == spec["scheduler"] and
        record.get("objective") == spec["objective"] and
        record.get("training_mode") ==
            SEARCH_SPACES[spec["policy_id"]].training_mode and
        record.get("rollout_policy") == metrics.get("rollout_policy") and
        record.get("metric_owner") == metrics.get("metric_owner") and
        record.get("training_data_policy") ==
            metrics.get("training_data_policy") and
        record.get("rollout_policy") == record.get("metric_owner") and
        record.get("training_config_sha256") == training_config_sha256(spec) and
        record.get("trial_config_sha256") ==
            spec.get("trial_config_sha256") and
        record.get("hyperparameter_selection_sha256") ==
            spec.get("hyperparameter_selection_sha256") and
        record.get("episode_index") == episode_index and
        record.get("round_index") == record.get("round") and
        record.get("scenario") == scenario and record.get("seed") == seed and
        record.get("input_provenance") == input_provenance and
        record.get("input_checkpoint_sha256") ==
            input_provenance.get("identity_sha256") and
        record.get("output_checkpoint_sha256") == sha256_file(checkpoint) and
        record.get("output_model_identity_sha256") ==
            sha256_file(model_identity_path(checkpoint)) and
        record.get("status") == "complete" and
        record.get("model_round_trip_valid") is True and
        record.get("duration_seconds") == DURATION_SECONDS and
        record.get("current_time") == DURATION_SECONDS and
        record.get("fixed_horizon") is True and
        record.get("horizon_finalized") is True and
        record.get("termination_reason") == "episode_horizon" and
        isinstance(record.get("task_manifest_sha256"), str) and
        len(record["task_manifest_sha256"]) == 64 and
        runtime.get("fixed_horizon") is True and
        runtime.get("runtime_mode") == "headless_webots_logic" and
        runtime.get("physics_fidelity") == "business_logic_only" and
        runtime.get("dynamics_version") == HEADLESS_DYNAMICS_VERSION and
        runtime.get("joint_runtime") is True and
        runtime.get("route_planner") == "rolling_joint_grid" and
        runtime.get("horizon_finalized") is True and
        runtime.get("termination_reason") == "episode_horizon" and
        math.isclose(float(runtime.get("current_time", -1)),
                     DURATION_SECONDS, rel_tol=0.0, abs_tol=1e-9) and
        ((transition_count == 0 and
          record.get("transition_sha256") is None) or
         (transition_count > 0 and transitions.is_file() and
          record.get("transition_sha256") == sha256_file(transitions))))
    if not valid:
        return False
    if spec["scheduler"] == "CQL":
        try:
            dataset = OfflineTransitionDataset.load(transitions)
        except Exception:
            return False
        contract = metrics.get("offline_dataset", {})
        return bool(
            metrics.get("transition_artifact_type") ==
            "frozen_offline_dataset" and
            record.get("offline_dataset_frozen") is True and
            record.get("policy_generated_transitions") == 0 and
            contract.get("dataset_sha256") == dataset.sha256 and
            contract.get("strictly_offline_updates") is True and
            contract.get("evaluation_transitions_added") == 0)
    if spec["scheduler"] in {"LearnedHungarian", "GraphImitation"}:
        expected_kind = (
            f"ridge_{spec['objective']}_expert_cost" if
            spec["scheduler"] == "LearnedHungarian" else
            f"graph_{spec['objective']}_imitation")
        if spec.get("traffic_version"):
            expected_kind += "_traffic_v1"
        try:
            features, targets = _load_supervised_samples(
                transitions, expected_kind)
        except Exception:
            return False
        return bool(
            record.get("policy_generated_transitions") == 0 and
            metrics.get("transition_artifact_type") == expected_kind and
            metrics.get("transition_count") == len(targets) and
            len(features) == len(targets))
    return bool(
        record.get("policy_generated_transitions") ==
        metrics.get("decisions") and
        record.get("rollout_policy") == spec["scheduler"])


def headless_final_path(output_dir: Path, spec: dict) -> Path:
    return (output_dir / "headless" / spec["policy_id"] /
            ("strict_1800_model" + spec["extension"]))


def fine_tuned_final_path(output_dir: Path, spec: dict) -> Path:
    return (output_dir / "webots_fine_tune" / spec["policy_id"] /
            ("webots_finetuned_model" + spec["extension"]))


def selected_final_path(output_dir: Path, spec: dict) -> Path:
    return (output_dir / "webots_fine_tune" / spec["policy_id"] /
            ("webots_selected_model" + spec["extension"]))


def fine_tune_round_checkpoint_path(
        policy_dir: Path, spec: dict, round_index: int) -> Path:
    """Keep a JSON model distinct from the JSON round commit marker."""
    suffix = ("_model.json" if spec["extension"] == ".json"
              else spec["extension"])
    return (policy_dir / "checkpoints" /
            (f"after_round_{round_index:02d}" + suffix))


def _run_standalone_reference_episode(reference: str, objective: str,
                                      manifest: dict, scenario: str,
                                      seed: int) -> dict:
    """Run a frozen same-seed 1800s reference through the shared runtime."""
    if reference not in {"NearestNeighbour", "ObjectiveHungarian"}:
        raise ValueError("unknown standalone validation reference")
    environment, robots, tasks, context, step_limit, profile = (
        _episode_environment(manifest, scenario, seed, objective))
    _state, metadata = environment.reset(robots, tasks, context, seed=seed)
    if metadata.get("reward_profile_hash") != profile.sha256:
        raise RuntimeError("standalone reference reward profile mismatch")
    scheduler = (create_scheduler("NearestNeighbour")
                 if reference == "NearestNeighbour" else None)
    terminated = False
    invalid_outputs = 0
    for _index in range(step_limit):
        if scheduler is not None:
            _state, _reward, terminated, truncated, info = (
                environment.step_scheduler(scheduler))
        else:
            mask = environment.get_action_mask()
            if mask[environment.no_op_action]:
                _state, _reward, terminated, truncated, info = (
                    environment.step(environment.no_op_action))
            else:
                live_robots, live_tasks, live_context = (
                    environment.policy_snapshot())
                matrix = build_cost_matrix(
                    list(live_tasks), live_robots, live_context)
                values = objective_expert_values(
                    matrix, live_context, objective)
                pairs = solve_hungarian(values, matrix.feasible)
                if not pairs:
                    raise RuntimeError(
                        "objective reference produced no assignment")
                row, column = pairs[0]
                action = environment.action_for_pair(
                    matrix.robot_ids[row], matrix.tasks[column].task_id)
                _state, _reward, terminated, truncated, info = (
                    environment.step(action))
        invalid_outputs += int(bool(
            info.get("invalid_action") or info.get("assignment_rejected") or
            info.get("invalid_truncation") or
            (truncated and not terminated)))
        if terminated or truncated:
            break
    if not terminated or invalid_outputs:
        raise RuntimeError("standalone reference failed the 1800s contract")
    runtime = _validate_runtime(environment, manifest)
    utility = environment.formal_utility_evaluation(utility_score_config())
    return {
        "schema_version": 1,
        "reference": reference,
        "objective": objective,
        "scenario": scenario,
        "seed": seed,
        "duration_seconds": DURATION_SECONDS,
        "current_time": float(runtime["current_time"]),
        "completed_tasks": int(runtime["tasks_completed"]),
        "utility_score_deadline_v2": float(
            utility["utility_score_deadline_v2"]),
        "invalid_scheduler_outputs": invalid_outputs,
        "runtime_mode": runtime["runtime_mode"],
        "dynamics_version": runtime["dynamics_version"],
        "physics_fidelity": runtime["physics_fidelity"],
        "joint_runtime": runtime["joint_runtime"],
        "route_planner": runtime["route_planner"],
        "fixed_horizon": runtime["fixed_horizon"],
        "horizon_finalized": runtime["horizon_finalized"],
        "termination_reason": runtime["termination_reason"],
        "task_manifest_sha256": manifest["task_manifest_sha256"],
    }


def _valid_standalone_reference(row, reference: str, objective: str,
                                manifest: dict, scenario: str,
                                seed: int) -> bool:
    if not isinstance(row, dict):
        return False
    values = (
        row.get("completed_tasks"),
        row.get("utility_score_deadline_v2"),
    )
    return bool(
        row.get("schema_version") == 1 and
        row.get("reference") == reference and
        row.get("objective") == objective and
        row.get("scenario") == scenario and row.get("seed") == seed and
        row.get("duration_seconds") == DURATION_SECONDS and
        row.get("current_time") == DURATION_SECONDS and
        row.get("invalid_scheduler_outputs") == 0 and
        row.get("runtime_mode") == "headless_webots_logic" and
        row.get("dynamics_version") == HEADLESS_DYNAMICS_VERSION and
        row.get("physics_fidelity") == "business_logic_only" and
        row.get("joint_runtime") is True and
        row.get("route_planner") == "rolling_joint_grid" and
        row.get("fixed_horizon") is True and
        row.get("horizon_finalized") is True and
        row.get("termination_reason") == "episode_horizon" and
        row.get("task_manifest_sha256") ==
            manifest.get("task_manifest_sha256") and
        all(not isinstance(value, bool) and
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values))


def _same_seed_reference_results(output_dir: Path, partition: str,
                                 objective: str, manifest: dict,
                                 scenario: str, seed: int, *,
                                 allow_create: bool = True) -> dict:
    references = {}
    for reference in ("NearestNeighbour", "ObjectiveHungarian"):
        path = (output_dir / "headless" / "reference_baselines" /
                partition / objective / scenario / str(seed) /
                f"{reference}.json")
        row = load_json(path)
        if not _valid_standalone_reference(
                row, reference, objective, manifest, scenario, seed):
            if not allow_create:
                raise RuntimeError(
                    "parallel validation reference cache is incomplete")
            row = _run_standalone_reference_episode(
                reference, objective, manifest, scenario, seed)
            atomic_json(path, row)
        references[reference] = row
    return references


def _run_standalone_validation_episode(spec: dict, checkpoint: Path,
                                       manifest: dict, scenario: str,
                                       seed: int, *,
                                       same_seed_references: dict | None = None
                                       ) -> dict:
    """Evaluate a frozen checkpoint without exposing a training entry point."""
    validate_model_identity(spec, checkpoint)
    before_hash = sha256_file(checkpoint)
    scheduler = create_scheduler(
        spec["scheduler"], model_path=str(checkpoint), seed=seed,
        allow_safe_fallback=False)
    environment, robots, tasks, context, step_limit, profile = (
        _episode_environment(
            manifest, scenario, seed, spec["objective"]))
    if getattr(scheduler, "uses_traffic", False):
        context = replace(context, configuration={
            **context.configuration, "graph_traffic_enabled": True})
    _state, metadata = environment.reset(robots, tasks, context, seed=seed)
    if metadata.get("reward_profile_hash") != profile.sha256:
        raise RuntimeError("standalone validation reward profile mismatch")
    fallback_commits = 0
    invalid_outputs = 0
    timeout_count = 0
    terminated = False
    for _index in range(step_limit):
        _state, _reward, terminated, truncated, info = (
            environment.step_scheduler(scheduler))
        diagnostics = info.get("scheduler_diagnostics", {})
        fallback_commits += int(bool(
            isinstance(diagnostics, dict) and
            diagnostics.get("fallback", False) and
            not info.get("assignment_rejected")))
        invalid_outputs += int(bool(
            info.get("invalid_action") or
            info.get("assignment_rejected") or
            info.get("invalid_truncation")))
        timeout_count += int(bool(
            isinstance(diagnostics, dict) and
            diagnostics.get("timeout", False)))
        if truncated and not terminated:
            invalid_outputs += 1
        if terminated or truncated:
            break
    if not terminated:
        raise RuntimeError(
            "standalone validation did not reach the 1800s horizon")
    runtime = _validate_runtime(environment, manifest)
    after_hash = sha256_file(checkpoint)
    if after_hash != before_hash:
        raise RuntimeError("pure standalone validation mutated checkpoint")
    utility = environment.formal_utility_evaluation(utility_score_config())
    completed = int(runtime.get("tasks_completed", -1))
    utility_score = utility.get("utility_score_deadline_v2")
    if completed < 0 or not isinstance(utility_score, (int, float)) or not (
            math.isfinite(float(utility_score))):
        raise RuntimeError("standalone validation metric is invalid")
    if same_seed_references is not None:
        if not isinstance(same_seed_references, dict) or not all(
                _valid_standalone_reference(
                    same_seed_references.get(reference), reference,
                    spec["objective"], manifest, scenario, seed)
                for reference in (
                    "NearestNeighbour", "ObjectiveHungarian")):
            raise RuntimeError(
                "standalone validation references are invalid")
    elif spec["scheduler"] in SUPERVISED_WEBOTS_ALGORITHMS:
        raise RuntimeError("supervised validation references are missing")
    return {
        "schema_version": 3,
        "partition": "validation",
        "policy_id": spec["policy_id"],
        "scheduler": spec["scheduler"],
        "objective": spec["objective"],
        "scenario": scenario,
        "seed": seed,
        "duration_seconds": DURATION_SECONDS,
        "current_time": float(runtime["current_time"]),
        "fixed_horizon": runtime["fixed_horizon"],
        "horizon_finalized": runtime["horizon_finalized"],
        "termination_reason": runtime["termination_reason"],
        "runtime_mode": runtime["runtime_mode"],
        "dynamics_version": runtime["dynamics_version"],
        "physics_fidelity": runtime["physics_fidelity"],
        "joint_runtime": runtime["joint_runtime"],
        "route_planner": runtime["route_planner"],
        "completed_tasks": completed,
        "utility_score_deadline_v2": float(utility_score),
        "rollout_policy": spec["scheduler"],
        "metric_owner": spec["scheduler"],
        "same_seed_references": same_seed_references,
        "pure_policy_eligible": bool(
            fallback_commits == 0 and invalid_outputs == 0 and
            timeout_count == 0),
        "fallback_scheduler_commits": fallback_commits,
        "invalid_scheduler_outputs": invalid_outputs,
        "scheduler_timeout_count": timeout_count,
        "checkpoint_sha256": before_hash,
        "checkpoint_identity_sha256": sha256_file(
            model_identity_path(checkpoint)),
        "task_manifest_sha256": manifest["task_manifest_sha256"],
    }


def _frozen_validation_worker(
        spec: dict, checkpoint: Path, manifest: dict, scenario: str,
        seed: int, same_seed_references: dict) -> tuple[str, int, dict]:
    """Run one read-only validation unit in an isolated process."""
    row = _run_standalone_validation_episode(
        spec, checkpoint, manifest, scenario, seed,
        same_seed_references=same_seed_references)
    return scenario, seed, row


def _valid_standalone_validation_episode(
        row, spec: dict, checkpoint_hash: str,
        checkpoint_identity_hash: str, manifest: dict,
        scenario: str, seed: int) -> bool:
    if not isinstance(row, dict):
        return False
    metric = ("completed_tasks" if spec["objective"] == "count" else
              "utility_score_deadline_v2")
    value = row.get(metric)
    references = row.get("same_seed_references")
    if spec["scheduler"] in SUPERVISED_WEBOTS_ALGORITHMS:
        references_valid = bool(
            isinstance(references, dict) and all(
                _valid_standalone_reference(
                    references.get(reference), reference, spec["objective"],
                    manifest, scenario, seed)
                for reference in (
                    "NearestNeighbour", "ObjectiveHungarian")))
    else:
        references_valid = bool(
            references is None or (
                isinstance(references, dict) and all(
                    _valid_standalone_reference(
                        references.get(reference), reference,
                        spec["objective"], manifest, scenario, seed)
                    for reference in (
                        "NearestNeighbour", "ObjectiveHungarian"))))
    purity_counters = tuple(
        row.get(field) for field in (
            "fallback_scheduler_commits",
            "invalid_scheduler_outputs",
            "scheduler_timeout_count",
        ))
    purity_counters_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in purity_counters)
    expected_pure_policy_eligible = bool(
        purity_counters_valid and all(value == 0 for value in purity_counters))
    return bool(
        row.get("schema_version") == 3 and
        row.get("partition") == "validation" and
        row.get("policy_id") == spec["policy_id"] and
        row.get("scheduler") == spec["scheduler"] and
        row.get("objective") == spec["objective"] and
        row.get("rollout_policy") == spec["scheduler"] and
        row.get("metric_owner") == spec["scheduler"] and
        references_valid and
        row.get("scenario") == scenario and row.get("seed") == seed and
        row.get("duration_seconds") == DURATION_SECONDS and
        row.get("current_time") == DURATION_SECONDS and
        row.get("fixed_horizon") is True and
        row.get("horizon_finalized") is True and
        row.get("termination_reason") == "episode_horizon" and
        row.get("runtime_mode") == "headless_webots_logic" and
        row.get("dynamics_version") == HEADLESS_DYNAMICS_VERSION and
        row.get("physics_fidelity") == "business_logic_only" and
        row.get("joint_runtime") is True and
        row.get("route_planner") == "rolling_joint_grid" and
        purity_counters_valid and
        row.get("pure_policy_eligible") is expected_pure_policy_eligible and
        row.get("checkpoint_sha256") == checkpoint_hash and
        row.get("checkpoint_identity_sha256") == checkpoint_identity_hash and
        row.get("task_manifest_sha256") ==
            manifest.get("task_manifest_sha256") and
        not isinstance(value, bool) and isinstance(value, (int, float)) and
        math.isfinite(float(value)))


def _pure_policy_gate(rows: list[dict]) -> dict:
    """Describe impure validation episodes without invalidating their audit."""
    impure_episodes = [
        {
            "scenario": row["scenario"],
            "seed": row["seed"],
            "fallback_scheduler_commits":
                row["fallback_scheduler_commits"],
            "invalid_scheduler_outputs": row["invalid_scheduler_outputs"],
            "scheduler_timeout_count": row["scheduler_timeout_count"],
        }
        for row in rows if not row["pure_policy_eligible"]
    ]
    return {
        "required": True,
        "eligible": not impure_episodes,
        "impure_episode_count": len(impure_episodes),
        "impure_episodes": impure_episodes,
    }


def _supervised_baseline_gate(rows: list[dict], spec: dict) -> dict:
    """Require supervised policies to stay near same-seed NN and expert."""
    if spec["scheduler"] not in SUPERVISED_WEBOTS_ALGORITHMS:
        return {"required": False, "eligible": True, "scenarios": {}}
    metric = ("completed_tasks" if spec["objective"] == "count" else
              "utility_score_deadline_v2")

    def floor(reference_value: float, ratio: float) -> float:
        return reference_value - (1.0-ratio)*abs(reference_value)

    comparisons = {}
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        if len(scenario_rows) == 0:
            raise RuntimeError("supervised baseline gate lacks a scenario")
        model_value = statistics.fmean(
            float(row[metric]) for row in scenario_rows)
        nearest = statistics.fmean(float(
            row["same_seed_references"]["NearestNeighbour"][metric])
            for row in scenario_rows)
        expert = statistics.fmean(float(
            row["same_seed_references"]["ObjectiveHungarian"][metric])
            for row in scenario_rows)
        nearest_floor = floor(
            nearest, SUPERVISED_MIN_NEAREST_NEIGHBOUR_RATIO)
        expert_floor = floor(
            expert, SUPERVISED_MIN_OBJECTIVE_EXPERT_RATIO)
        comparisons[scenario] = {
            "model_mean": model_value,
            "nearest_neighbour_mean": nearest,
            "objective_expert_mean": expert,
            "nearest_neighbour_floor": nearest_floor,
            "objective_expert_floor": expert_floor,
            "eligible": bool(
                model_value >= nearest_floor and
                model_value >= expert_floor),
        }
    return {
        "required": True,
        "metric": metric,
        "nearest_neighbour_min_ratio":
            SUPERVISED_MIN_NEAREST_NEIGHBOUR_RATIO,
        "objective_expert_min_ratio":
            SUPERVISED_MIN_OBJECTIVE_EXPERT_RATIO,
        "eligible": all(row["eligible"] for row in comparisons.values()),
        "scenarios": comparisons,
    }


def _nearest_neighbour_diagnostic(rows: list[dict], spec: dict) -> dict:
    """Describe large same-seed gaps for every AI without cross-seed claims."""
    metric = ("completed_tasks" if spec["objective"] == "count" else
              "utility_score_deadline_v2")
    comparisons = {}
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        if not scenario_rows:
            raise RuntimeError("nearest-neighbour diagnostic lacks a scenario")
        try:
            model_value = statistics.fmean(
                float(row[metric]) for row in scenario_rows)
            reference_value = statistics.fmean(float(
                row["same_seed_references"]["NearestNeighbour"][metric])
                for row in scenario_rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "nearest-neighbour diagnostic lacks paired references") from exc
        floor = reference_value - (
            1.0-AI_NEAREST_NEIGHBOUR_REVIEW_RATIO)*abs(reference_value)
        comparisons[scenario] = {
            "model_mean": model_value,
            "nearest_neighbour_mean": reference_value,
            "review_floor": floor,
            "delta": model_value-reference_value,
            "large_gap": model_value < floor,
        }
    return {
        "paired_same_seed": True,
        "review_ratio": AI_NEAREST_NEIGHBOUR_REVIEW_RATIO,
        "requires_review": any(
            row["large_gap"] for row in comparisons.values()),
        "scenarios": comparisons,
    }


def _valid_standalone_validation_report(
        report, output_dir: Path, spec: dict, checkpoint: Path,
        checkpoint_hash: str, checkpoint_identity_hash: str,
        round_index: int, reference_round: int) -> bool:
    """Validate current or legacy-superset evidence without rewriting it."""
    if not isinstance(report, dict):
        return False
    seeds = tuple(report.get("validation_seeds", ()))
    metric_name = ("completed_tasks" if spec["objective"] == "count" else
                   "utility_score_deadline_v2")
    unsigned = dict(report)
    report_hash = unsigned.pop("selector_report_sha256", None)
    parameter_change = report.get("parameter_change")
    if not (
            seeds in ACCEPTED_FORMAL_VALIDATION_SEED_CONTRACTS and
            report.get("policy_id") == spec["policy_id"] and
            report.get("round_index") == round_index and
            report.get("objective") == spec["objective"] and
            report.get("selection_metric") == metric_name and
            report.get("duration_seconds") == DURATION_SECONDS and
            report.get("checkpoint_sha256") == checkpoint_hash and
            report.get("checkpoint_identity_sha256") ==
                checkpoint_identity_hash and
            isinstance(report.get("checkpoint_path"), str) and
            Path(report["checkpoint_path"]).resolve() == checkpoint.resolve() and
            report.get("episode_count") == len(SCENARIOS) * len(seeds) and
            isinstance(parameter_change, dict) and
            parameter_change.get("reference_round") == reference_round and
            report_hash == _canonical_sha256(unsigned)):
        return False
    rows = []
    for scenario in SCENARIOS:
        for seed in seeds:
            manifest = _load_manifest(manifest_path(
                "standalone_validation", scenario, seed))
            result_path = (
                output_dir / "headless" / spec["policy_id"] /
                "validation" / f"round_{round_index:02d}" / scenario /
                str(seed) / "result.json")
            row = load_json(result_path)
            if not _valid_standalone_validation_episode(
                    row, spec, checkpoint_hash, checkpoint_identity_hash,
                    manifest, scenario, seed):
                return False
            rows.append(row)
    try:
        metric = statistics.fmean(float(row[metric_name]) for row in rows)
        stored_metric = float(report["validation_metric"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        math.isclose(metric, stored_metric, rel_tol=0.0, abs_tol=1e-12) and
        report.get("episodes_sha256") == _canonical_sha256(rows))


def _validate_standalone_round(output_dir: Path, spec: dict,
                               checkpoint: Path, round_index: int,
                               update_state, *,
                               reference_checkpoint: Path,
                               reference_round: int) -> dict:
    validate_model_identity(spec, checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    checkpoint_identity_hash = sha256_file(model_identity_path(checkpoint))
    report_path = (
        output_dir / "headless" / spec["policy_id"] / "validation" /
        f"round_{round_index:02d}" / "report.json")
    existing_report = load_json(report_path)
    if _valid_standalone_validation_report(
            existing_report, output_dir, spec, checkpoint, checkpoint_hash,
            checkpoint_identity_hash, round_index, reference_round):
        print(
            f"[RESUME] standalone-validation/{spec['policy_id']}/"
            f"round-{round_index} seeds="
            f"{len(existing_report['validation_seeds'])}", flush=True)
        return existing_report
    rows_by_key = {}
    pending_units = []
    for scenario in SCENARIOS:
        for seed in VALIDATION_SEEDS:
            manifest = _load_manifest(manifest_path(
                "standalone_validation", scenario, seed))
            result_path = (
                output_dir / "headless" / spec["policy_id"] /
                "validation" / f"round_{round_index:02d}" / scenario /
                str(seed) / "result.json")
            row = load_json(result_path)
            if _valid_standalone_validation_episode(
                    row, spec, checkpoint_hash, checkpoint_identity_hash,
                    manifest, scenario, seed):
                rows_by_key[(scenario, seed)] = row
                continue
            update_state(
                stage="standalone_validation_reference_cache",
                current={"policy_id": spec["policy_id"],
                         "round": round_index, "scenario": scenario,
                         "seed": seed})
            references = _same_seed_reference_results(
                output_dir, "standalone_validation",
                spec["objective"], manifest, scenario, seed)
            pending_units.append(
                (scenario, seed, manifest, result_path, references))
    if pending_units:
        worker_count = min(
            PARALLEL_FROZEN_VALIDATION_WORKERS, len(pending_units))
        update_state(
            stage="standalone_validation_parallel",
            current={"policy_id": spec["policy_id"],
                     "round": round_index,
                     "parallel_workers": worker_count,
                     "pending_episodes": len(pending_units),
                     "committed_episodes": len(rows_by_key)})
        future_units = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for scenario, seed, manifest, result_path, references in (
                    pending_units):
                future = executor.submit(
                    _frozen_validation_worker, spec, checkpoint, manifest,
                    scenario, seed, references)
                future_units[future] = (scenario, seed, result_path)
            for future in as_completed(future_units):
                expected_scenario, expected_seed, result_path = (
                    future_units[future])
                scenario, seed, row = future.result()
                if (scenario, seed) != (expected_scenario, expected_seed):
                    raise RuntimeError(
                        "parallel standalone validation identity mismatch")
                atomic_json(result_path, row)
                rows_by_key[(scenario, seed)] = row
                update_state(
                    stage="standalone_validation_parallel",
                    current={"policy_id": spec["policy_id"],
                             "round": round_index,
                             "parallel_workers": worker_count,
                             "pending_episodes": len(pending_units),
                             "committed_episodes": len(rows_by_key)})
    rows = [
        rows_by_key[(scenario, seed)]
        for scenario in SCENARIOS for seed in VALIDATION_SEEDS]
    metric_name = ("completed_tasks" if spec["objective"] == "count" else
                   "utility_score_deadline_v2")
    validation_metric = statistics.fmean(
        float(row[metric_name]) for row in rows)
    baseline_gate = _supervised_baseline_gate(rows, spec)
    nearest_diagnostic = _nearest_neighbour_diagnostic(rows, spec)
    parameter_change = _policy_parameter_change(
        spec, reference_checkpoint, checkpoint,
        reference_round=reference_round)
    evidence = {
        "policy_id": spec["policy_id"],
        "round_index": round_index,
        "objective": spec["objective"],
        "selection_metric": metric_name,
        "eligible": bool(
            all(row["pure_policy_eligible"] for row in rows) and
            baseline_gate["eligible"]),
        "same_seed_baseline_gate": baseline_gate,
        "nearest_neighbour_diagnostic": nearest_diagnostic,
        "duration_seconds": DURATION_SECONDS,
        "validation_seeds": list(VALIDATION_SEEDS),
        "validation_metric": validation_metric,
        "parameter_change": parameter_change,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_identity_sha256": sha256_file(
            model_identity_path(checkpoint)),
        "episode_count": len(rows),
        "episodes_sha256": _canonical_sha256(rows),
    }
    evidence["selector_report_sha256"] = _canonical_sha256(evidence)
    atomic_json(report_path, evidence)
    return evidence


def _standalone_nonregression_decision(
        spec: dict, incumbent: dict, candidate: dict, *,
        training_baseline: dict | None = None) -> dict:
    """Accept normal validation noise but reject material degradation.

    Every candidate is checked against both the latest accepted checkpoint and
    the immutable pre-training baseline. The second comparison prevents a
    sequence of individually small regressions from accumulating into a large
    loss over a long convergence-driven run.
    """
    policy_id = spec.get("policy_id")
    space = SEARCH_SPACES.get(policy_id)
    if (space is None or spec.get("scheduler") != space.scheduler or
            spec.get("objective") != space.objective):
        raise ValueError("standalone guard received an unknown AI identity")
    if training_baseline is None:
        training_baseline = incumbent
    if (not isinstance(incumbent, dict) or
            not isinstance(candidate, dict) or
            not isinstance(training_baseline, dict)):
        raise ValueError("standalone guard requires validation reports")
    initial_baseline_recovery = bool(
        training_baseline.get("eligible") is False and
        incumbent.get("eligible") is False and
        candidate.get("eligible") is True and
        training_baseline.get("round_index") == 0 and
        incumbent.get("round_index") == 0 and
        isinstance(candidate.get("round_index"), int) and
        candidate.get("round_index") > 0 and
        training_baseline.get("checkpoint_sha256") ==
            incumbent.get("checkpoint_sha256"))
    for label, report in (("training baseline", training_baseline),
                          ("incumbent", incumbent),
                          ("candidate", candidate)):
        if (report.get("policy_id") != spec["policy_id"] or
                report.get("objective") != spec["objective"] or
                tuple(report.get("validation_seeds", ())) !=
                tuple(VALIDATION_SEEDS)):
            raise ValueError(
                f"{label} standalone validation is incompatible")
        eligible = report.get("eligible")
        if (not isinstance(eligible, bool) or
                (label != "candidate" and eligible is not True and
                 not initial_baseline_recovery)):
            raise ValueError(
                f"{label} standalone validation is incompatible")
        metric = report.get("validation_metric")
        if (isinstance(metric, bool) or not isinstance(metric, (int, float)) or
                not math.isfinite(float(metric))):
            raise ValueError(
                f"{label} standalone validation metric is invalid")

    def scenario_means(report: dict) -> dict[str, float]:
        try:
            rows = report["nearest_neighbour_diagnostic"]["scenarios"]
            means = {
                scenario: float(rows[scenario]["model_mean"])
                for scenario in SCENARIOS}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "standalone validation lacks scenario means") from exc
        if not all(math.isfinite(value) for value in means.values()):
            raise ValueError("standalone scenario mean is non-finite")
        return means

    incumbent_metric = float(incumbent["validation_metric"])
    candidate_metric = float(candidate["validation_metric"])
    baseline_metric = float(training_baseline["validation_metric"])
    incumbent_scenarios = scenario_means(incumbent)
    candidate_scenarios = scenario_means(candidate)
    baseline_scenarios = scenario_means(training_baseline)
    scenario_differences = {
        scenario: candidate_scenarios[scenario]-incumbent_scenarios[scenario]
        for scenario in SCENARIOS}
    baseline_scenario_differences = {
        scenario: candidate_scenarios[scenario]-baseline_scenarios[scenario]
        for scenario in SCENARIOS}
    mean_difference = candidate_metric-incumbent_metric
    baseline_mean_difference = candidate_metric-baseline_metric

    tolerance = (
        STANDALONE_COUNT_RELATIVE_DEGRADATION_TOLERANCE
        if spec["objective"] == "count" else
        STANDALONE_UTILITY_RELATIVE_DEGRADATION_TOLERANCE)
    tolerance_kind = "relative"

    def lower_bound(reference: float) -> float:
        return reference - tolerance*abs(reference)

    candidate_eligible = candidate["eligible"] is True
    incumbent_gate = bool(
        candidate_metric >= lower_bound(incumbent_metric) and
        all(candidate_scenarios[scenario] >= lower_bound(
            incumbent_scenarios[scenario]) for scenario in SCENARIOS))
    baseline_gate = bool(
        candidate_metric >= lower_bound(baseline_metric) and
        all(candidate_scenarios[scenario] >= lower_bound(
            baseline_scenarios[scenario]) for scenario in SCENARIOS))
    accepted = bool(
        candidate_eligible and (
            initial_baseline_recovery or
            (incumbent_gate and baseline_gate)))
    strict_improvement = bool(
        candidate_eligible and mean_difference > 0.0 and
        all(value >= 0.0 for value in scenario_differences.values()))
    selected = candidate if accepted else incumbent
    if initial_baseline_recovery:
        decision_name = "establish_first_eligible_training_baseline"
    elif strict_improvement:
        decision_name = "promote_candidate"
    elif accepted:
        decision_name = "continue_candidate_within_tolerance"
    else:
        decision_name = "rollback_to_incumbent"
    decision = {
        "version": STANDALONE_NONREGRESSION_GUARD_VERSION,
        "policy_id": spec["policy_id"],
        "scheduler": spec["scheduler"],
        "objective": spec["objective"],
        "duration_seconds": DURATION_SECONDS,
        "validation_seeds": list(VALIDATION_SEEDS),
        "strict_improvement_required": False,
        "candidate_eligible": candidate_eligible,
        "candidate_rejection_reason": (
            None if candidate_eligible else "validation_ineligible"),
        "training_baseline_rebased": initial_baseline_recovery,
        "training_baseline_rebase_reason": (
            "initial_validation_ineligible_first_eligible_candidate"
            if initial_baseline_recovery else None),
        "degradation_tolerance": tolerance,
        "degradation_tolerance_kind": tolerance_kind,
        "training_baseline_round": training_baseline["round_index"],
        "training_baseline_checkpoint": training_baseline["checkpoint_path"],
        "training_baseline_checkpoint_sha256":
            training_baseline["checkpoint_sha256"],
        "training_baseline_metric": baseline_metric,
        "incumbent_round": incumbent["round_index"],
        "incumbent_checkpoint": incumbent["checkpoint_path"],
        "incumbent_checkpoint_sha256": incumbent["checkpoint_sha256"],
        "incumbent_metric": incumbent_metric,
        "candidate_round": candidate["round_index"],
        "candidate_checkpoint": candidate["checkpoint_path"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_metric": candidate_metric,
        "mean_difference": mean_difference,
        "scenario_mean_differences": scenario_differences,
        "baseline_mean_difference": baseline_mean_difference,
        "baseline_scenario_mean_differences":
            baseline_scenario_differences,
        "incumbent_within_tolerance": (
            None if initial_baseline_recovery else incumbent_gate),
        "training_baseline_within_tolerance": (
            None if initial_baseline_recovery else baseline_gate),
        "strict_improvement": strict_improvement,
        "severe_regression": bool(
            not initial_baseline_recovery and
            not (incumbent_gate and baseline_gate)),
        "accepted": accepted,
        "decision": decision_name,
        "selected_checkpoint": selected["checkpoint_path"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_metric": float(selected["validation_metric"]),
    }
    decision["report_sha256"] = _canonical_sha256(decision)
    return decision


def _graph_ppo_nonregression_decision(
        spec: dict, incumbent: dict, candidate: dict) -> dict:
    """Backward-compatible GraphPPO-only entry point."""
    if spec.get("scheduler") != "GraphPPO":
        raise ValueError("standalone GraphPPO guard received another scheduler")
    return _standalone_nonregression_decision(spec, incumbent, candidate)


def _standalone_effective_validation(
        output_dir: Path, spec: dict, incumbent: dict, candidate: dict, *,
        training_baseline: dict,
        previous_selected_checkpoint: Path,
        previous_selected_round: int) -> tuple[dict, dict]:
    """Write the promoted/rolled-back validation used by readiness checks."""
    decision = _standalone_nonregression_decision(
        spec, incumbent, candidate, training_baseline=training_baseline)
    selected = candidate if decision["accepted"] else incumbent
    selected_checkpoint = Path(selected["checkpoint_path"])
    if sha256_file(selected_checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("selected standalone checkpoint hash mismatch")
    validate_model_identity(spec, selected_checkpoint)
    effective = dict(selected)
    effective.update({
        "round_index": candidate["round_index"],
        "parameter_change": _policy_parameter_change(
            spec, previous_selected_checkpoint, selected_checkpoint,
            reference_round=previous_selected_round),
        "standalone_nonregression_guard_version":
            STANDALONE_NONREGRESSION_GUARD_VERSION,
        "promotion_accepted": decision["accepted"],
        "promotion_decision_sha256": decision["report_sha256"],
        "candidate_parameter_change": candidate["parameter_change"],
        "candidate_checkpoint_path": candidate["checkpoint_path"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_validation_metric": candidate["validation_metric"],
        "selected_checkpoint_origin_round": selected["round_index"],
        "validation_evidence_reused": not decision["accepted"],
    })
    effective.pop("selector_report_sha256", None)
    effective["selector_report_sha256"] = _canonical_sha256(effective)
    round_dir = (
        output_dir / "headless" / spec["policy_id"] / "validation" /
        f"round_{candidate['round_index']:02d}")
    atomic_json(round_dir / "promotion_decision.json", decision)
    atomic_json(round_dir / "effective_report.json", effective)
    return effective, decision


def _trial_spec(spec: dict, trial: dict) -> dict:
    validate_trial_config(trial)
    if trial["policy_id"] != spec["policy_id"]:
        raise ValueError("hyperparameter trial belongs to another policy")
    selected = dict(spec)
    selected["training_parameters"] = dict(trial["parameters"])
    selected["trial_config_sha256"] = trial["trial_config_sha256"]
    return selected


def _tuning_selection_path(output_dir: Path, spec: dict) -> Path:
    return (output_dir / "headless" / spec["policy_id"] / "tuning" /
            "selection.json")


def _tuning_group_audit_path(
        output_dir: Path, spec: dict, trial_id: str) -> Path:
    return (output_dir / "headless" / spec["policy_id"] / "tuning" /
            trial_id / "group_audit.json")


def _result_evidence(output_dir: Path, path: Path, *, partition: str,
                     scenario: str, seed: int,
                     episode_index: int | None = None) -> dict:
    resolved_output = output_dir.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_output):
        raise RuntimeError("tuning evidence is outside the output directory")
    evidence = {
        "partition": partition,
        "scenario": scenario,
        "seed": seed,
        "result_path": resolved_path.relative_to(resolved_output).as_posix(),
        "result_file_sha256": sha256_file(resolved_path),
    }
    if episode_index is not None:
        evidence["episode_index"] = episode_index
    return evidence


def _valid_tuning_group_audit(
        output_dir: Path, spec: dict, candidate: dict) -> bool:
    """Fail closed unless one completed tuning group is still immutable."""
    try:
        trial_id = candidate.get("trial_id")
        trial = candidate.get("trial")
        if not isinstance(trial_id, str) or not isinstance(trial, dict):
            return False
        path = _tuning_group_audit_path(output_dir, spec, trial_id)
        document = load_json(path)
        if not isinstance(document, dict):
            return False
        unhashed = {
            key: value for key, value in document.items()
            if key != "report_sha256"
        }
        candidate_payload = {
            key: value for key, value in candidate.items()
            if key != "group_audit_sha256"
        }
        training_evidence = document.get("training_evidence")
        validation_evidence = document.get("validation_evidence")
        expected_training_units = [
            (scenario, seed, episode_index)
            for episode_index, (seed, scenario) in enumerate(
                ((seed, scenario) for seed in TUNING_TRAINING_SEEDS
                 for scenario in SCENARIOS), start=1)
        ]
        expected_validation_units = [
            (scenario, seed)
            for scenario in SCENARIOS for seed in TUNING_VALIDATION_SEEDS
        ]
        valid = bool(
            document.get("schema_version") == 1 and
            document.get("audit_contract") ==
                HYPERPARAMETER_GROUP_AUDIT_VERSION and
            document.get("status") == "complete_and_verified" and
            document.get("policy_id") == spec["policy_id"] and
            document.get("scheduler") == spec["scheduler"] and
            document.get("objective") == spec["objective"] and
            document.get("trial_id") == trial_id and
            document.get("trial_config_sha256") ==
                trial.get("trial_config_sha256") and
            document.get("search_space_sha256") ==
                SEARCH_SPACES[spec["policy_id"]].sha256 and
            document.get("duration_seconds_per_episode") ==
                DURATION_SECONDS and
            document.get("training_episode_count") ==
                len(expected_training_units) and
            document.get("validation_episode_count") ==
                len(expected_validation_units) and
            document.get("training_seed_contract") ==
                list(TUNING_TRAINING_SEEDS) and
            document.get("validation_seed_contract") ==
                list(TUNING_VALIDATION_SEEDS) and
            isinstance(training_evidence, list) and
            isinstance(validation_evidence, list) and
            len(training_evidence) == len(expected_training_units) and
            len(validation_evidence) == len(expected_validation_units) and
            document.get("training_evidence_sha256") ==
                _canonical_sha256(training_evidence) and
            document.get("validation_evidence_sha256") ==
                _canonical_sha256(validation_evidence) and
            document.get("candidate") == candidate_payload and
            document.get("report_sha256") == _canonical_sha256(unhashed) and
            candidate.get("group_audit_sha256") ==
                document.get("report_sha256"))
        if not valid:
            return False
        actual_training_units = [
            (row.get("scenario"), row.get("seed"), row.get("episode_index"))
            for row in training_evidence
        ]
        actual_validation_units = [
            (row.get("scenario"), row.get("seed"))
            for row in validation_evidence
        ]
        if (actual_training_units != expected_training_units or
                actual_validation_units != expected_validation_units):
            return False
        output_root = output_dir.resolve()
        trial_root = path.parent.resolve()
        for row in (*training_evidence, *validation_evidence):
            result_path = (output_root / str(row["result_path"])).resolve()
            if (not result_path.is_relative_to(trial_root) or
                    not result_path.is_file() or
                    sha256_file(result_path) != row["result_file_sha256"]):
                return False
        checkpoint = Path(str(candidate["checkpoint"])).resolve()
        if (not checkpoint.is_relative_to(trial_root) or
                not checkpoint.is_file() or
                sha256_file(checkpoint) != candidate["checkpoint_sha256"] or
                sha256_file(model_identity_path(checkpoint)) !=
                    candidate["checkpoint_identity_sha256"]):
            return False
        validate_model_identity(_trial_spec(spec, trial), checkpoint)
        return True
    except (AttributeError, KeyError, ModelValidationError, OSError,
            TypeError, ValueError):
        return False


def _tuning_candidate_eligible(candidate: dict) -> bool:
    """Apply all selection gates, retaining old pure audit compatibility."""
    baseline_gate = candidate.get("same_seed_baseline_gate")
    if (not isinstance(baseline_gate, dict) or
            baseline_gate.get("eligible") is not True):
        return False
    if "pure_policy_gate" not in candidate:
        return True
    pure_policy_gate = candidate.get("pure_policy_gate")
    return bool(
        isinstance(pure_policy_gate, dict) and
        pure_policy_gate.get("eligible") is True)


def _apply_tuning_selection(output_dir: Path, spec: dict,
                            *, required: bool = False) -> bool:
    """Load one fail-closed winner and bind its parameters to ``spec``."""
    path = _tuning_selection_path(output_dir, spec)
    document = load_json(path)
    if not isinstance(document, dict):
        if required:
            raise RuntimeError(
                f"hyperparameter selection is missing: {spec['policy_id']}")
        return False
    unhashed = {key: value for key, value in document.items()
                if key != "report_sha256"}
    trial = document.get("winner_trial")
    checkpoint_value = document.get("winner_checkpoint")
    candidates = document.get("candidates")
    valid = bool(
        document.get("schema_version") == 3 and
        document.get("optimization_mode") ==
            "automatic_deterministic_grid_sample" and
        document.get("selection_source") ==
            "isolated_tuning_validation_metric_only" and
        document.get("policy_id") == spec["policy_id"] and
        document.get("scheduler") == spec["scheduler"] and
        document.get("objective") == spec["objective"] and
        document.get("duration_seconds_per_episode") == DURATION_SECONDS and
        document.get("trial_count") ==
            HYPERPARAMETER_TRIAL_COUNTS[spec["policy_id"]] and
        document.get("tuning_rounds_per_trial") ==
            len(TUNING_TRAINING_SEEDS) and
        document.get("training_episodes_per_trial") ==
            len(SCENARIOS) * len(TUNING_TRAINING_SEEDS) and
        document.get("validation_episodes_per_trial") ==
            len(SCENARIOS) * len(TUNING_VALIDATION_SEEDS) and
        document.get("search_space_sha256") ==
            SEARCH_SPACES[spec["policy_id"]].sha256 and
        isinstance(candidates, list) and
        len(candidates) == HYPERPARAMETER_TRIAL_COUNTS[spec["policy_id"]] and
        isinstance(trial, dict) and isinstance(checkpoint_value, str) and
        document.get("report_sha256") == _canonical_sha256(unhashed))
    if valid:
        valid = all(
            isinstance(candidate, dict) and
            isinstance(candidate.get("trial_id"), str) and
            isinstance(candidate.get("trial"), dict) and
            isinstance(
                candidate["trial"].get("trial_config_sha256"), str) and
            isinstance(candidate.get("validation_metric"), (int, float)) and
            isinstance(candidate.get("same_seed_baseline_gate"), dict) and
            isinstance(
                candidate["same_seed_baseline_gate"].get("eligible"), bool) and
            ("pure_policy_gate" not in candidate or (
                isinstance(candidate.get("pure_policy_gate"), dict) and
                isinstance(
                    candidate["pure_policy_gate"].get("eligible"), bool))) and
            _valid_tuning_group_audit(output_dir, spec, candidate)
            for candidate in candidates)
    if valid:
        expected_trial_ids = [
            f"trial_{index:02d}"
            for index in range(HYPERPARAMETER_TRIAL_COUNTS[spec["policy_id"]])
        ]
        candidate_by_id = {
            candidate["trial_id"]: candidate for candidate in candidates
        }
        winner_id = document.get("winner_trial_id")
        winner_candidate = candidate_by_id.get(winner_id)
        rejected = [
            candidate["trial_id"] for candidate in candidates
            if not candidate["same_seed_baseline_gate"]["eligible"]
        ]
        rejected_by_pure_policy = [
            candidate["trial_id"] for candidate in candidates
            if isinstance(candidate.get("pure_policy_gate"), dict) and
            candidate["pure_policy_gate"].get("eligible") is False
        ]
        eligible_ranked = sorted(
            (candidate for candidate in candidates
             if _tuning_candidate_eligible(candidate)),
            key=lambda row: (-row["validation_metric"],
                             row["trial"]["trial_config_sha256"]))
        valid = bool(
            [candidate["trial_id"] for candidate in candidates] ==
                expected_trial_ids and
            len(candidate_by_id) == len(candidates) and
            winner_candidate is not None and
            eligible_ranked and winner_candidate == eligible_ranked[0] and
            document.get("ranking") == [
                candidate["trial_id"] for candidate in eligible_ranked] and
            document.get("rejected_by_same_seed_baseline") == rejected and
            document.get("rejected_by_pure_policy", []) ==
                rejected_by_pure_policy and
            document.get("winner_trial") == winner_candidate["trial"] and
            document.get("winner_validation_metric") ==
                winner_candidate["validation_metric"] and
            document.get("winner_checkpoint") ==
                winner_candidate["checkpoint"] and
            document.get("winner_checkpoint_sha256") ==
                winner_candidate["checkpoint_sha256"] and
            document.get("winner_checkpoint_identity_sha256") ==
                winner_candidate["checkpoint_identity_sha256"])
    if valid:
        try:
            selected = _trial_spec(spec, trial)
            checkpoint = Path(checkpoint_value).resolve()
            trial_root = (
                output_dir / "headless" / spec["policy_id"] / "tuning" /
                str(document.get("winner_trial_id", ""))).resolve()
            valid = bool(
                checkpoint.is_relative_to(trial_root) and
                checkpoint.is_file() and
                sha256_file(checkpoint) ==
                    document.get("winner_checkpoint_sha256") and
                model_identity_path(checkpoint).is_file() and
                sha256_file(model_identity_path(checkpoint)) ==
                    document.get("winner_checkpoint_identity_sha256"))
            if valid:
                validate_model_identity(selected, checkpoint)
        except (KeyError, ModelValidationError, OSError, ValueError):
            valid = False
    if not valid:
        if required:
            raise RuntimeError(
                f"hyperparameter selection is invalid: {spec['policy_id']}")
        return False
    spec["training_parameters"] = dict(selected["training_parameters"])
    spec["trial_config_sha256"] = selected["trial_config_sha256"]
    spec["bootstrap_checkpoint"] = checkpoint
    spec["hyperparameter_selection_sha256"] = document["report_sha256"]
    return True


def _run_hyperparameter_tuning_core(
        output_dir: Path, spec: dict, update_state,
        max_new_episodes: int | None = None, *,
        trial_indices: tuple[int, ...] | None = None,
        select_winner: bool = True,
        parallel_validation: bool = False) -> tuple[bool, int]:
    """Train isolated 1800s trials and select only on isolated validation."""
    if select_winner and _apply_tuning_selection(output_dir, spec):
        return True, 0
    policy_dir = output_dir / "headless" / spec["policy_id"] / "tuning"
    sampler_seed = 17000 + tuple(
        item["policy_id"] for item in AI_SPECS).index(spec["policy_id"])
    trial_count = HYPERPARAMETER_TRIAL_COUNTS[spec["policy_id"]]
    trials = deterministic_trial_configs(
        spec["policy_id"], trial_count, sampler_seed)
    executed = 0
    candidates = []
    for trial_index, frozen_trial in enumerate(trials):
        if trial_indices is not None and trial_index not in trial_indices:
            continue
        trial = dict(frozen_trial)
        trial["parameters"] = dict(frozen_trial["parameters"])
        trial_id = f"trial_{trial_index:02d}"
        candidate_spec = _trial_spec(spec, trial)
        trial_dir = policy_dir / trial_id
        current = Path(spec["source"])
        transition_history = []
        training_evidence = []
        episode_index = 0
        for round_index, seed in enumerate(TUNING_TRAINING_SEEDS, start=1):
            for scenario in SCENARIOS:
                episode_index += 1
                unit_dir = (trial_dir / "episodes" /
                            f"{episode_index:02d}_{scenario}_{seed}")
                checkpoint = unit_dir / ("model" + spec["extension"])
                transitions = unit_dir / "transitions.npz"
                record_path = unit_dir / "result.json"
                input_provenance = _training_input_provenance(
                    candidate_spec, current, seed)
                input_hash = input_provenance["identity_sha256"]
                record = load_json(record_path)
                if _valid_headless_record(
                        record, candidate_spec, episode_index, scenario, seed,
                        input_provenance, checkpoint, transitions):
                    current = checkpoint
                    if transitions.is_file():
                        transition_history.append(transitions)
                    training_evidence.append(_result_evidence(
                        output_dir, record_path, partition="training",
                        scenario=scenario, seed=seed,
                        episode_index=episode_index))
                    continue
                if (max_new_episodes is not None and
                        executed >= max_new_episodes):
                    return False, executed
                update_state(
                    stage="standalone_hyperparameter_tuning",
                    current={"policy_id": spec["policy_id"],
                             "trial_id": trial_id,
                             "episode_index": episode_index,
                             "scenario": scenario, "seed": seed})
                print(
                    f"[RUN] tuning policy={spec['policy_id']} "
                    f"trial={trial_id} scenario={scenario} seed={seed} "
                    "duration=1800s", flush=True)
                manifest = _load_manifest(manifest_path(
                    "standalone_tuning_training", scenario, seed))
                metrics = run_headless_episode(
                    candidate_spec, current, checkpoint, transitions,
                    manifest, scenario, seed, transition_history,
                    fit_supervised=(
                        spec["scheduler"] in {
                            "LearnedHungarian", "GraphImitation"} and
                        scenario == SCENARIOS[-1]),
                    training_seed_partition=
                        "standalone_tuning_training")
                runtime = metrics["runtime"]
                record = {
                    "schema_version": TRAINING_RECORD_SCHEMA_VERSION,
                    "training_semantics_version":
                        TRAINING_SEMANTICS_VERSION,
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "status": "complete", "partition": "training",
                    "policy_id": spec["policy_id"],
                    "scheduler": spec["scheduler"],
                    "objective": spec["objective"],
                    "training_mode":
                        SEARCH_SPACES[spec["policy_id"]].training_mode,
                    "training_config_sha256":
                        training_config_sha256(candidate_spec),
                    "hyperparameter_selection_sha256": None,
                    "episode_index": episode_index,
                    "round": round_index, "round_index": round_index,
                    "scenario": scenario, "seed": seed,
                    "duration_seconds": DURATION_SECONDS,
                    "current_time": float(runtime["current_time"]),
                    "fixed_horizon": runtime["fixed_horizon"],
                    "horizon_finalized": runtime["horizon_finalized"],
                    "termination_reason": runtime["termination_reason"],
                    "model_round_trip_valid": True,
                    "input_provenance": input_provenance,
                    "input_checkpoint_sha256": input_hash,
                    "output_checkpoint_sha256": sha256_file(checkpoint),
                    "output_model_identity_sha256": sha256_file(
                        model_identity_path(checkpoint)),
                    "task_manifest_sha256": manifest["task_manifest_sha256"],
                    "manifest_file_sha256": sha256_file(manifest_path(
                        "standalone_tuning_training", scenario, seed)),
                    "offline_dataset_frozen": spec["scheduler"] == "CQL",
                    "rollout_policy": metrics["rollout_policy"],
                    "metric_owner": metrics["metric_owner"],
                    "training_data_policy": metrics["training_data_policy"],
                    "policy_generated_transitions": (
                        0 if spec["scheduler"] in {
                            "CQL", "LearnedHungarian", "GraphImitation"}
                        else metrics["decisions"]),
                    "metrics": metrics,
                    "transition_sha256": metrics["transition_sha256"],
                    "hyperparameter_trial_id": trial_id,
                    "trial_config_sha256": trial["trial_config_sha256"],
                }
                atomic_json(record_path, record)
                training_evidence.append(_result_evidence(
                    output_dir, record_path, partition="training",
                    scenario=scenario, seed=seed,
                    episode_index=episode_index))
                current = checkpoint
                if transitions.is_file():
                    transition_history.append(transitions)
                executed += 1

        validate_model_identity(candidate_spec, current)
        checkpoint_hash = sha256_file(current)
        identity_hash = sha256_file(model_identity_path(current))
        rows_by_key = {}
        validation_paths = {}
        pending_validation = []
        for scenario in SCENARIOS:
            for seed in TUNING_VALIDATION_SEEDS:
                manifest = _load_manifest(manifest_path(
                    "standalone_tuning_validation", scenario, seed))
                result_path = (trial_dir / "validation" / scenario /
                               str(seed) / "result.json")
                row = load_json(result_path)
                validation_paths[(scenario, seed)] = result_path
                if _valid_standalone_validation_episode(
                        row, candidate_spec, checkpoint_hash, identity_hash,
                        manifest, scenario, seed):
                    rows_by_key[(scenario, seed)] = row
                    continue
                references = _same_seed_reference_results(
                    output_dir, "tuning_validation",
                    spec["objective"], manifest, scenario, seed,
                    allow_create=not parallel_validation)
                pending_validation.append(
                    (scenario, seed, manifest, result_path, references))
        if pending_validation and parallel_validation:
            worker_count = min(
                PARALLEL_FROZEN_VALIDATION_WORKERS,
                len(pending_validation))
            future_units = {}
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                for scenario, seed, manifest, result_path, references in (
                        pending_validation):
                    future = executor.submit(
                        _frozen_validation_worker, candidate_spec, current,
                        manifest, scenario, seed, references)
                    future_units[future] = (scenario, seed, result_path)
                for future in as_completed(future_units):
                    expected_scenario, expected_seed, result_path = (
                        future_units[future])
                    scenario, seed, row = future.result()
                    if (scenario, seed) != (expected_scenario, expected_seed):
                        raise RuntimeError(
                            "parallel tuning validation identity mismatch")
                    atomic_json(result_path, row)
                    rows_by_key[(scenario, seed)] = row
        else:
            for scenario, seed, manifest, result_path, references in (
                    pending_validation):
                    update_state(
                        stage="standalone_hyperparameter_validation",
                        current={"policy_id": spec["policy_id"],
                                  "trial_id": trial_id,
                                  "scenario": scenario, "seed": seed})
                    row = _run_standalone_validation_episode(
                        candidate_spec, current, manifest, scenario, seed,
                        same_seed_references=references)
                    atomic_json(result_path, row)
                    rows_by_key[(scenario, seed)] = row
        rows = [
            rows_by_key[(scenario, seed)]
            for scenario in SCENARIOS for seed in TUNING_VALIDATION_SEEDS]
        validation_evidence = [
            _result_evidence(
                output_dir, validation_paths[(scenario, seed)],
                partition="validation", scenario=scenario, seed=seed)
            for scenario in SCENARIOS for seed in TUNING_VALIDATION_SEEDS]
        metric_name = ("completed_tasks" if spec["objective"] == "count"
                       else "utility_score_deadline_v2")
        pure_policy_gate = _pure_policy_gate(rows)
        baseline_gate = _supervised_baseline_gate(rows, candidate_spec)
        nearest_diagnostic = _nearest_neighbour_diagnostic(
            rows, candidate_spec)
        candidate = {
            "trial_id": trial_id,
            "trial": trial,
            "validation_metric": statistics.fmean(
                float(row[metric_name]) for row in rows),
            "selection_metric": metric_name,
            "checkpoint": str(current.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_identity_sha256": identity_hash,
            "training_episode_count": episode_index,
            "validation_episode_count": len(rows),
            "same_seed_baseline_gate": baseline_gate,
            "nearest_neighbour_diagnostic": nearest_diagnostic,
            "validation_rows_sha256": _canonical_sha256(rows),
        }
        # Pure candidates retain their original payload and audit hash. An
        # impure result is valid evidence, but cannot win model selection.
        if not pure_policy_gate["eligible"]:
            candidate["pure_policy_gate"] = pure_policy_gate
        audit_path = _tuning_group_audit_path(
            output_dir, spec, trial_id)
        audit = load_json(audit_path)
        if isinstance(audit, dict) and isinstance(
                audit.get("report_sha256"), str):
            candidate["group_audit_sha256"] = audit["report_sha256"]
        if not _valid_tuning_group_audit(output_dir, spec, candidate):
            candidate.pop("group_audit_sha256", None)
            audit = {
                "schema_version": 1,
                "audit_contract": HYPERPARAMETER_GROUP_AUDIT_VERSION,
                "status": "complete_and_verified",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "policy_id": spec["policy_id"],
                "scheduler": spec["scheduler"],
                "objective": spec["objective"],
                "trial_id": trial_id,
                "trial_config_sha256": trial["trial_config_sha256"],
                "search_space_sha256":
                    SEARCH_SPACES[spec["policy_id"]].sha256,
                "duration_seconds_per_episode": DURATION_SECONDS,
                "training_episode_count": len(training_evidence),
                "validation_episode_count": len(validation_evidence),
                "training_seed_contract": list(TUNING_TRAINING_SEEDS),
                "validation_seed_contract": list(TUNING_VALIDATION_SEEDS),
                "training_evidence": training_evidence,
                "validation_evidence": validation_evidence,
                "training_evidence_sha256":
                    _canonical_sha256(training_evidence),
                "validation_evidence_sha256":
                    _canonical_sha256(validation_evidence),
                # Keep the hashed payload independent from the audit hash
                # subsequently attached to the selection candidate.
                "candidate": dict(candidate),
            }
            audit["report_sha256"] = _canonical_sha256(audit)
            candidate["group_audit_sha256"] = audit["report_sha256"]
            atomic_json(audit_path, audit)
            if not _valid_tuning_group_audit(
                    output_dir, spec, candidate):
                raise RuntimeError(
                    f"hyperparameter group audit failed: "
                    f"{spec['policy_id']} {trial_id}")
        candidates.append(candidate)
        update_state(
            stage="standalone_hyperparameter_group_verified",
            current={"policy_id": spec["policy_id"],
                     "trial_id": trial_id,
                     "group_audit_sha256": audit["report_sha256"]})
        print(
            f"[GROUP CHECK OK] policy={spec['policy_id']} "
            f"trial={trial_id} training={len(training_evidence)} "
            f"validation={len(validation_evidence)} "
            f"audit={audit['report_sha256']}", flush=True)

    if not select_winner:
        return True, executed
    eligible_candidates = [
        row for row in candidates
        if _tuning_candidate_eligible(row)]
    if not eligible_candidates:
        raise RuntimeError(
            f"no hyperparameter trial passed selection gates: "
            f"{spec['policy_id']}")
    ranked = sorted(
        eligible_candidates,
        key=lambda row: (-row["validation_metric"],
                         row["trial"]["trial_config_sha256"]))
    winner = ranked[0]
    selection = {
        "schema_version": 3,
        "optimization_mode": "automatic_deterministic_grid_sample",
        "selection_source": "isolated_tuning_validation_metric_only",
        "policy_id": spec["policy_id"],
        "scheduler": spec["scheduler"],
        "objective": spec["objective"],
        "search_space_sha256": SEARCH_SPACES[spec["policy_id"]].sha256,
        "duration_seconds_per_episode": DURATION_SECONDS,
        "trial_count": trial_count,
        "tuning_rounds_per_trial": len(TUNING_TRAINING_SEEDS),
        "training_episodes_per_trial": (
            len(SCENARIOS) * len(TUNING_TRAINING_SEEDS)),
        "validation_episodes_per_trial": (
            len(SCENARIOS) * len(TUNING_VALIDATION_SEEDS)),
        "selection_metric": winner["selection_metric"],
        "winner_trial_id": winner["trial_id"],
        "winner_trial": winner["trial"],
        "winner_validation_metric": winner["validation_metric"],
        "winner_checkpoint": winner["checkpoint"],
        "winner_checkpoint_sha256": winner["checkpoint_sha256"],
        "winner_checkpoint_identity_sha256":
            winner["checkpoint_identity_sha256"],
        "ranking": [row["trial_id"] for row in ranked],
        "rejected_by_same_seed_baseline": [
            row["trial_id"] for row in candidates
            if not row["same_seed_baseline_gate"]["eligible"]],
        "rejected_by_pure_policy": [
            row["trial_id"] for row in candidates
            if isinstance(row.get("pure_policy_gate"), dict) and
            row["pure_policy_gate"].get("eligible") is False],
        "candidates": candidates,
        "seed_contract": {
            "training": list(TUNING_TRAINING_SEEDS),
            "validation": list(TUNING_VALIDATION_SEEDS),
        },
    }
    selection["report_sha256"] = _canonical_sha256(selection)
    atomic_json(_tuning_selection_path(output_dir, spec), selection)
    if not _apply_tuning_selection(output_dir, spec, required=True):
        raise RuntimeError("failed to reload hyperparameter winner")
    print(
        f"[TUNING WINNER] policy={spec['policy_id']} "
        f"trial={winner['trial_id']} "
        f"metric={winner['validation_metric']:.6f}", flush=True)
    return True, executed


def _ignore_parallel_worker_state(**_values) -> None:
    """Workers never mutate the shared workflow state document."""


def _hyperparameter_trial_worker(
        output_dir: Path, spec: dict, trial_index: int) -> tuple[int, int]:
    complete, executed = _run_hyperparameter_tuning_core(
        output_dir, spec, _ignore_parallel_worker_state,
        trial_indices=(trial_index,), select_winner=False,
        parallel_validation=True)
    if not complete:
        raise RuntimeError("isolated hyperparameter worker stopped early")
    return trial_index, executed


def _prepare_tuning_reference_cache(
        output_dir: Path, spec: dict, update_state) -> None:
    """Materialize shared references before isolated trial workers start."""
    for scenario in SCENARIOS:
        for seed in TUNING_VALIDATION_SEEDS:
            manifest = _load_manifest(manifest_path(
                "standalone_tuning_validation", scenario, seed))
            update_state(
                stage="standalone_hyperparameter_reference_cache",
                current={"policy_id": spec["policy_id"],
                         "scenario": scenario, "seed": seed})
            _same_seed_reference_results(
                output_dir, "tuning_validation", spec["objective"],
                manifest, scenario, seed)


def _parallel_tuning_commit_counts(
        output_dir: Path, spec: dict, trial_count: int) -> dict:
    tuning = output_dir / "headless" / spec["policy_id"] / "tuning"
    trial_dirs = [tuning / f"trial_{index:02d}"
                  for index in range(trial_count)]
    return {
        "training_episodes": sum(
            1 for trial_dir in trial_dirs
            for _path in (trial_dir / "episodes").glob("*/result.json")),
        "validation_episodes": sum(
            1 for trial_dir in trial_dirs
            for _path in (trial_dir / "validation").glob("*/*/result.json")),
        "verified_groups": sum(
            int((trial_dir / "group_audit.json").is_file())
            for trial_dir in trial_dirs),
    }


def _run_hyperparameter_tuning(
        output_dir: Path, spec: dict, update_state,
        max_new_episodes: int | None = None) -> tuple[bool, int]:
    """Run isolated trials concurrently while the parent owns shared state."""
    if _apply_tuning_selection(output_dir, spec):
        return True, 0
    # Explicit run limits are primarily used by interruption/resume tests and
    # operators requesting an exact number of new training episodes. Preserve
    # their strictly sequential accounting contract.
    if max_new_episodes is not None:
        return _run_hyperparameter_tuning_core(
            output_dir, spec, update_state, max_new_episodes)

    _prepare_tuning_reference_cache(output_dir, spec, update_state)
    trial_count = HYPERPARAMETER_TRIAL_COUNTS[spec["policy_id"]]
    worker_count = min(PARALLEL_HYPERPARAMETER_TRIAL_WORKERS, trial_count)
    if worker_count <= 1:
        return _run_hyperparameter_tuning_core(
            output_dir, spec, update_state,
            parallel_validation=True)

    update_state(
        stage="standalone_hyperparameter_parallel",
        current={"policy_id": spec["policy_id"],
                 "parallel_trial_workers": worker_count,
                 "trial_count": trial_count,
                 **_parallel_tuning_commit_counts(
                     output_dir, spec, trial_count)})
    executed = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_indices = {
            executor.submit(
                _hyperparameter_trial_worker, output_dir, spec, trial_index):
                trial_index
            for trial_index in range(trial_count)
        }
        pending = set(future_indices)
        while pending:
            completed, pending = wait(
                pending, timeout=15.0, return_when=FIRST_COMPLETED)
            for future in completed:
                expected_index = future_indices[future]
                try:
                    trial_index, trial_executed = future.result()
                except Exception:
                    for queued in pending:
                        queued.cancel()
                    raise
                if trial_index != expected_index:
                    raise RuntimeError(
                        "parallel hyperparameter trial identity mismatch")
                executed += trial_executed
            update_state(
                stage="standalone_hyperparameter_parallel",
                current={"policy_id": spec["policy_id"],
                         "parallel_trial_workers": worker_count,
                         "trial_count": trial_count,
                         "completed_worker_trials":
                             trial_count-len(pending),
                         **_parallel_tuning_commit_counts(
                             output_dir, spec, trial_count)})

    # Rebuild every candidate in deterministic trial order from its validated
    # atomic artifacts, then use the unchanged selector to commit the winner.
    complete, unexpected_executed = _run_hyperparameter_tuning_core(
        output_dir, spec, update_state, max_new_episodes=0)
    if unexpected_executed:
        raise RuntimeError(
            "parallel tuning finalization unexpectedly trained an episode")
    return complete, executed


def train_headless(output_dir: Path, update_state,
                   max_new_episodes: int | None = None, *, specs=None,
                   progress: dict | None = None) -> bool:
    """Tune and train the requested policies, preserving a shared run limit."""
    requested_specs = tuple(
        PIPELINE_AI_SPECS if specs is None else specs)
    executed = int((progress or {}).get("standalone", 0))
    for spec in requested_specs:
        remaining = (None if max_new_episodes is None else
                     max_new_episodes - executed)
        tuning_complete, tuning_executed = _run_hyperparameter_tuning(
            output_dir, spec, update_state, remaining)
        executed += tuning_executed
        if progress is not None:
            progress["standalone"] = executed
        if not tuning_complete:
            return False
        policy_dir = output_dir / "headless" / spec["policy_id"]
        operator_exit_request = _load_standalone_exit_override(
            output_dir, spec)
        operator_stopped_training = False
        current = Path(spec["bootstrap_checkpoint"])
        transition_history = []
        training_records = []
        validation_history = []
        promotion_history = []
        readiness = None
        episode_index = 0
        previous_validation_checkpoint = Path(spec["bootstrap_checkpoint"])
        previous_validation_round = 0
        standalone_baseline = _validate_standalone_round(
            output_dir, spec, current, 0, update_state,
            reference_checkpoint=current, reference_round=0)
        standalone_incumbent = standalone_baseline
        print(
            f"[BASELINE] standalone/{spec['policy_id']} "
            f"metric={standalone_incumbent['validation_metric']:.6f} "
            "source=tuning_winner", flush=True)
        for round_index, seed in enumerate(TRAINING_SEEDS, start=1):
            for scenario in SCENARIOS:
                episode_index += 1
                unit_dir = (policy_dir / "episodes" /
                            f"{episode_index:02d}_{scenario}_{seed}")
                checkpoint = unit_dir / ("model" + spec["extension"])
                transitions = unit_dir / "transitions.npz"
                record_path = unit_dir / "result.json"
                input_provenance = _training_input_provenance(
                    spec, current, seed)
                input_hash = input_provenance["identity_sha256"]
                record = load_json(record_path)
                if _valid_headless_record(
                        record, spec, episode_index, scenario, seed,
                        input_provenance, checkpoint, transitions):
                    print(
                        f"[RESUME] headless/{spec['policy_id']}/"
                        f"{scenario}/{seed}", flush=True)
                    current = checkpoint
                    if transitions.is_file():
                        transition_history.append(transitions)
                    training_records.append(record)
                    continue
                if operator_exit_request is not None:
                    operator_stopped_training = True
                    print(
                        f"[OPERATOR TRANSITION] standalone/"
                        f"{spec['policy_id']} stops before incomplete "
                        f"episode={episode_index}; no new standalone "
                        "episode will run", flush=True)
                    break
                if (max_new_episodes is not None and
                        executed >= max_new_episodes):
                    return False
                update_state(
                    stage="standalone_sufficient_training",
                    current={"policy_id": spec["policy_id"],
                             "episode_index": episode_index,
                             "scenario": scenario, "seed": seed})
                print(
                    f"[RUN] sufficient-standalone policy={spec['policy_id']} "
                    f"scenario={scenario} seed={seed} duration=1800s",
                    flush=True)
                manifest = _load_manifest(manifest_path(
                    "standalone_training", scenario, seed))
                metrics = run_headless_episode(
                    spec, current, checkpoint, transitions,
                    manifest,
                    scenario, seed, transition_history,
                    fit_supervised=(
                        spec["scheduler"] in {
                            "LearnedHungarian", "GraphImitation"} and
                        scenario == SCENARIOS[-1] and
                        round_index % VALIDATION_INTERVAL_ROUNDS == 0))
                runtime = metrics["runtime"]
                record = {
                    "schema_version": TRAINING_RECORD_SCHEMA_VERSION,
                    "training_semantics_version":
                        TRAINING_SEMANTICS_VERSION,
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "status": "complete",
                    "partition": "training",
                    "policy_id": spec["policy_id"],
                    "scheduler": spec["scheduler"],
                    "objective": spec["objective"],
                    "training_mode":
                        SEARCH_SPACES[spec["policy_id"]].training_mode,
                    "training_config_sha256": training_config_sha256(spec),
                    "trial_config_sha256": spec["trial_config_sha256"],
                    "hyperparameter_selection_sha256":
                        spec["hyperparameter_selection_sha256"],
                    "episode_index": episode_index,
                    "round": round_index,
                    "round_index": round_index,
                    "scenario": scenario,
                    "seed": seed,
                    "duration_seconds": DURATION_SECONDS,
                    "current_time": float(runtime["current_time"]),
                    "fixed_horizon": runtime["fixed_horizon"],
                    "horizon_finalized": runtime["horizon_finalized"],
                    "termination_reason": runtime["termination_reason"],
                    "model_round_trip_valid": True,
                    "input_provenance": input_provenance,
                    "input_checkpoint_sha256": input_hash,
                    "output_checkpoint_sha256": sha256_file(checkpoint),
                    "output_model_identity_sha256": sha256_file(
                        model_identity_path(checkpoint)),
                    "task_manifest_sha256":
                        manifest["task_manifest_sha256"],
                    "manifest_file_sha256": sha256_file(manifest_path(
                        "standalone_training", scenario, seed)),
                    "offline_dataset_frozen": (
                        spec["scheduler"] == "CQL"),
                    "rollout_policy": metrics["rollout_policy"],
                    "metric_owner": metrics["metric_owner"],
                    "training_data_policy": metrics["training_data_policy"],
                    "policy_generated_transitions": (
                        0 if spec["scheduler"] in {
                            "CQL", "LearnedHungarian", "GraphImitation"}
                        else metrics["decisions"]),
                    "metrics": metrics,
                    "transition_sha256": metrics["transition_sha256"],
                }
                atomic_json(record_path, record)
                current = checkpoint
                if transitions.is_file():
                    transition_history.append(transitions)
                training_records.append(record)
                executed += 1
                if progress is not None:
                    progress["standalone"] = executed
                print(
                    f"[CHECKPOINT] headless/{spec['policy_id']} "
                    f"episode={episode_index}/"
                    f"{MAX_STANDALONE_BALANCED_ROUNDS*len(SCENARIOS)} "
                    f"sha256={record['output_checkpoint_sha256'][:12]}",
                    flush=True)

            if operator_stopped_training:
                break
            if round_index % VALIDATION_INTERVAL_ROUNDS == 0:
                candidate_validation = _validate_standalone_round(
                    output_dir, spec, current, round_index, update_state,
                    reference_checkpoint=previous_validation_checkpoint,
                    reference_round=previous_validation_round)
                validation, promotion = _standalone_effective_validation(
                    output_dir, spec, standalone_incumbent,
                    candidate_validation,
                    training_baseline=standalone_baseline,
                    previous_selected_checkpoint=
                        previous_validation_checkpoint,
                    previous_selected_round=previous_validation_round)
                if promotion.get("training_baseline_rebased") is True:
                    standalone_baseline = validation
                current = Path(validation["checkpoint_path"])
                standalone_incumbent = validation
                promotion_history.append(promotion)
                print(
                    f"[PROMOTION] standalone/{spec['policy_id']} "
                    f"round={round_index} "
                    f"candidate={candidate_validation['validation_metric']:.6f} "
                    f"selected={validation['validation_metric']:.6f} "
                    f"decision={promotion['decision']}", flush=True)
                validation_history.append(validation)
                previous_validation_checkpoint = current
                previous_validation_round = round_index
                readiness = standalone_training_readiness(
                    spec["policy_id"], training_records,
                    validation_history)
                atomic_json(policy_dir / "standalone_readiness.json", readiness)
                print(
                    f"[VALIDATION] standalone/{spec['policy_id']} "
                    f"round={round_index} "
                    f"metric={validation['validation_metric']:.6f} "
                    "parameter_delta="
                    f"{validation['parameter_change']['relative_l2_change']:.6g} "
                    f"status={readiness['status']}", flush=True)
                if readiness["webots_fine_tune_eligible"]:
                    break

        natural_readiness = readiness
        if operator_exit_request is not None:
            readiness = _operator_webots_readiness(
                spec, natural_readiness, operator_exit_request)
            atomic_json(policy_dir / "standalone_readiness.json", readiness)
            print(
                f"[OPERATOR TRANSITION] standalone/{spec['policy_id']} "
                f"selects validated round={readiness['best_validation_round']} "
                "for Webots fine-tuning", flush=True)
        elif (readiness is not None and readiness.get("status") ==
              "maximum_completed_for_webots_fine_tune"):
            print(
                f"[MAXIMUM] standalone/{spec['policy_id']} completed "
                f"{readiness['completed_balanced_rounds']} balanced rounds "
                "without full convergence; selecting the best guarded "
                "checkpoint for Webots fine-tuning",
                flush=True)
        if not readiness or not readiness["webots_fine_tune_eligible"]:
            atomic_json(policy_dir / "training_report.json", {
                "status": (readiness or {}).get(
                    "status", "insufficient_training"),
                "policy_id": spec["policy_id"],
                "completed_episode_count": len(training_records),
                "webots_fine_tune_eligible": False,
                "readiness": readiness,
            })
            raise RuntimeError(
                f"{spec['policy_id']} did not qualify after sufficient "
                "standalone training; Webots fine-tuning is blocked")
        best_round = readiness["best_validation_round"]
        best = next(
            row for row in validation_history
            if row["round_index"] == best_round)
        best_checkpoint = Path(best["checkpoint_path"])
        if sha256_file(best_checkpoint) != readiness["best_checkpoint_sha256"]:
            raise RuntimeError("standalone winner checkpoint hash mismatch")
        validate_model_identity(spec, best_checkpoint)
        if (sha256_file(model_identity_path(best_checkpoint)) !=
                readiness["best_checkpoint_identity_sha256"]):
            raise RuntimeError(
                "standalone winner objective identity hash mismatch")
        final = headless_final_path(output_dir, spec)
        atomic_copy_model(spec, best_checkpoint, final)
        load_model(spec, final, seed=0)
        atomic_json(policy_dir / "training_report.json", {
            "status": "complete", "policy_id": spec["policy_id"],
            "scheduler": spec["scheduler"], "objective": spec["objective"],
            "training_contract": "standalone_sufficient_training",
            "standalone_exit_mode": readiness.get(
                "standalone_exit_mode",
                "maximum_balanced_rounds_reached"
                if readiness.get("maximum_rounds_reached") and
                readiness.get("status") ==
                    "maximum_completed_for_webots_fine_tune"
                else "natural_convergence"),
            "standalone_completion_reason": readiness.get(
                "standalone_completion_reason"),
            "maximum_rounds_reached": readiness.get(
                "maximum_rounds_reached", False),
            "natural_readiness_status": (
                natural_readiness or {}).get("status"),
            "natural_webots_fine_tune_eligible": bool(
                (natural_readiness or {}).get(
                    "webots_fine_tune_eligible")),
            "standalone_exit_override_sha256": readiness.get(
                "standalone_exit_override_sha256"),
            "episode_count": len(training_records),
            "balanced_rounds": readiness["completed_balanced_rounds"],
            "minimum_episode_count":
                MIN_STANDALONE_BALANCED_ROUNDS*len(SCENARIOS),
            "duration_seconds_per_episode": 1800.0,
            "fixed_horizon": True,
            "physics_fidelity": "business_logic_only",
            "webots_fine_tune_eligible": True,
            "readiness_report_sha256": readiness["report_sha256"],
            "best_validation_round": best_round,
            "best_validation_metric": readiness["best_validation_metric"],
            "nearest_neighbour_diagnostic":
                best["nearest_neighbour_diagnostic"],
            "validation_plateau": readiness["validation_plateau"],
            "parameter_converged": readiness["parameter_converged"],
            "parameter_relative_change_threshold":
                PARAMETER_RELATIVE_CHANGE_THRESHOLD,
            "parameter_convergence_patience_checks":
                PARAMETER_CONVERGENCE_PATIENCE_CHECKS,
            "training_parameters": training_parameters(spec),
            "trial_config_sha256": spec["trial_config_sha256"],
            "hyperparameter_selection_sha256":
                spec["hyperparameter_selection_sha256"],
            "standalone_nonregression_guard_version":
                STANDALONE_NONREGRESSION_GUARD_VERSION,
            "standalone_baseline_metric":
                standalone_baseline["validation_metric"],
            "standalone_baseline_checkpoint_sha256":
                standalone_baseline["checkpoint_sha256"],
            "promotion_checks": len(promotion_history),
            "promotion_acceptances": sum(
                int(row["accepted"]) for row in promotion_history),
            "promotion_rollbacks": sum(
                int(not row["accepted"]) for row in promotion_history),
            "promotion_decision_sha256": [
                row["report_sha256"] for row in promotion_history],
            "output_checkpoint": str(final.resolve()),
            "output_checkpoint_sha256": sha256_file(final),
            "output_model_identity_sha256": sha256_file(
                model_identity_path(final)),
            "resume_granularity": "each completed 1800s episode",
        })
    return True


HEADLESS_MODEL_LANE_VERSION = "isolated-headless-model-lane-v1"


def _headless_lane_state_path(output_dir: Path, policy_id: str) -> Path:
    return (Path(output_dir) / "model_pipeline" / "headless_lanes" /
            policy_id / "state.json")


@contextmanager
def _exclusive_headless_lane(output_dir: Path, policy_id: str):
    """Serialize writers for one model without deleting a lock file.

    The persistent one-byte file is only an OS-lock anchor.  Locks are
    released automatically when a process exits, including abnormal exits.
    Different policy identities use different files and remain independent.
    """
    lock_path = (_headless_lane_state_path(output_dir, policy_id).parent /
                 "standalone_worker.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        if lock_path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(2.0)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def _headless_model_lane_worker(output_dir: Path, spec: dict) -> dict:
    """Train one identity while excluding another writer for that identity."""
    with _exclusive_headless_lane(
            output_dir, str(spec.get("policy_id", "unknown"))):
        return _headless_model_lane_worker_locked(output_dir, spec)


def _completed_headless_lane_result(
        output_dir: Path, spec: dict) -> dict | None:
    """Rebuild a lane receipt from an already committed training result."""
    policy_id = spec["policy_id"]
    report_path = output_dir / "headless" / policy_id / "training_report.json"
    report = load_json(report_path)
    checkpoint = headless_final_path(output_dir, spec).resolve()
    try:
        if not (
                isinstance(report, dict) and
                report.get("status") == "complete" and
                report.get("policy_id") == policy_id and
                report.get("scheduler") == spec["scheduler"] and
                report.get("objective") == spec["objective"] and
                report.get("webots_fine_tune_eligible") is True and
                Path(report.get("output_checkpoint", "")).resolve() ==
                    checkpoint and
                checkpoint.is_file() and
                report.get("output_checkpoint_sha256") ==
                    sha256_file(checkpoint) and
                report.get("output_model_identity_sha256") ==
                    sha256_file(model_identity_path(checkpoint))):
            return None
        identity = validate_model_identity(spec, checkpoint)
    except (KeyError, ModelValidationError, OSError, TypeError, ValueError):
        return None
    result = {
        "version": HEADLESS_MODEL_LANE_VERSION,
        "status": "complete",
        "policy_id": policy_id,
        "scheduler": spec["scheduler"],
        "objective": spec["objective"],
        "training_report": str(report_path.resolve()),
        "training_report_sha256": sha256_file(report_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_identity_sha256": sha256_file(
            model_identity_path(checkpoint)),
        "training_config_sha256": identity["training_config_sha256"],
        "trial_config_sha256": identity["trial_config_sha256"],
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


def _headless_model_lane_worker_locked(output_dir: Path, spec: dict) -> dict:
    """Train exactly one objective identity in an isolated child process."""
    output_dir = Path(output_dir).resolve()
    lane_spec = dict(spec)
    policy_id = lane_spec["policy_id"]
    space = SEARCH_SPACES.get(policy_id)
    if (space is None or lane_spec.get("scheduler") != space.scheduler or
            lane_spec.get("objective") != space.objective):
        raise ValueError("parallel headless lane received an unknown identity")
    state_path = _headless_lane_state_path(output_dir, policy_id)

    def lane_state(status: str, **values) -> None:
        atomic_json(state_path, {
            "version": HEADLESS_MODEL_LANE_VERSION,
            "status": status,
            "policy_id": policy_id,
            "scheduler": lane_spec["scheduler"],
            "objective": lane_spec["objective"],
            "worker_pid": os.getpid(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **values,
        })

    def update_lane_state(**values) -> None:
        lane_state("running", **values)

    lane_state("running", stage="starting", current=None)
    try:
        selection_bound = _apply_tuning_selection(output_dir, lane_spec)
        if selection_bound:
            existing_result = _completed_headless_lane_result(
                output_dir, lane_spec)
            if existing_result is not None:
                lane_state(
                    "complete", stage="standalone_complete", current=None,
                    result=existing_result,
                    resumed_from_committed_result=True)
                print(
                    f"[RESUME] headless/{policy_id} committed training result",
                    flush=True)
                return existing_result
        else:
            print(
                f"[RESUME] headless/{policy_id} has no committed "
                "hyperparameter winner; starting or resuming automatic "
                "optimization",
                flush=True)
        complete = train_headless(
            output_dir, update_lane_state, specs=(lane_spec,), progress={})
        if not complete:
            raise RuntimeError(
                f"parallel headless lane stopped early: {policy_id}")
        result = _completed_headless_lane_result(output_dir, lane_spec)
        if result is None:
            raise RuntimeError(
                f"parallel headless lane report is invalid: {policy_id}")
        lane_state(
            "complete", stage="standalone_complete", current=None,
            result=result)
        return result
    except Exception as exc:
        lane_state(
            "failed", stage="standalone_failed", current=None,
            error=f"{type(exc).__name__}: {exc}")
        raise


def _validate_headless_lane_result(
        output_dir: Path, spec: dict, result: dict) -> None:
    """Fail closed if a child returns another model/objective's artifacts."""
    if not isinstance(result, dict):
        raise RuntimeError("parallel headless lane returned no result")
    unsigned = dict(result)
    result_sha256 = unsigned.pop("result_sha256", None)
    if (result_sha256 != _canonical_sha256(unsigned) or
            result.get("version") != HEADLESS_MODEL_LANE_VERSION or
            result.get("status") != "complete" or
            result.get("policy_id") != spec["policy_id"] or
            result.get("scheduler") != spec["scheduler"] or
            result.get("objective") != spec["objective"]):
        raise RuntimeError("parallel headless lane identity mismatch")
    policy_root = (
        Path(output_dir).resolve() / "headless" / spec["policy_id"])
    report_path = Path(result["training_report"]).resolve()
    checkpoint = Path(result["checkpoint"]).resolve()
    if (not report_path.is_relative_to(policy_root) or
            not checkpoint.is_relative_to(policy_root) or
            report_path != policy_root / "training_report.json" or
            checkpoint != headless_final_path(output_dir, spec).resolve() or
            sha256_file(report_path) != result["training_report_sha256"] or
            sha256_file(checkpoint) != result["checkpoint_sha256"] or
            sha256_file(model_identity_path(checkpoint)) !=
                result["checkpoint_identity_sha256"]):
        raise RuntimeError("parallel headless lane crossed its artifact scope")
    identity = validate_model_identity(spec, checkpoint)
    if (identity["training_config_sha256"] !=
            result["training_config_sha256"] or
            identity["trial_config_sha256"] !=
            result["trial_config_sha256"]):
        raise RuntimeError("parallel headless lane training identity mismatch")


def _headless_lane_snapshot(output_dir: Path, specs) -> dict:
    snapshots = {}
    for spec in specs:
        row = load_json(_headless_lane_state_path(
            output_dir, spec["policy_id"]))
        if (isinstance(row, dict) and
                row.get("version") == HEADLESS_MODEL_LANE_VERSION and
                row.get("policy_id") == spec["policy_id"] and
                row.get("scheduler") == spec["scheduler"] and
                row.get("objective") == spec["objective"]):
            snapshots[spec["policy_id"]] = row
        else:
            snapshots[spec["policy_id"]] = {
                "version": HEADLESS_MODEL_LANE_VERSION,
                "status": "starting",
                "policy_id": spec["policy_id"],
                "scheduler": spec["scheduler"],
                "objective": spec["objective"],
            }
    return snapshots


def _physical_groups(envelopes: list[dict]) -> list[list[dict]]:
    groups = []
    for envelope in envelopes:
        physical = envelope["result"]["formal_evaluation"][
            "physical_fine_tune"]
        rows = [dict(row) for row in physical["rollout"]]
        if not rows or sum(bool(row.get("done")) for row in rows) != 1 or not (
                rows[-1].get("done") is True):
            raise RuntimeError("physical rollout has an invalid terminal marker")
        groups.append(rows)
    return groups


def _shared_supervision(row: dict) -> dict:
    document = row.get("shared_supervision")
    if not webots_workflow._valid_shared_supervision(document):
        raise RuntimeError("Webots shared supervision telemetry is invalid")
    if (document.get("dataset_version") !=
            WEBOTS_SHARED_SUPERVISION_VERSION or
            document.get("feature_version") != FEATURE_VERSION):
        raise RuntimeError("Webots supervision feature contract drifted")
    return document


def _load_supervision_envelope(value) -> dict:
    """Load one validated-corpus envelope without retaining the whole corpus."""
    document = load_json(Path(value)) if isinstance(value, (str, Path)) else value
    if not isinstance(document, dict):
        raise RuntimeError("Webots supervision envelope is invalid")
    return document


def _learned_hungarian_shared_samples(
        envelopes: list[dict | Path], objective: str, *,
        sample_limit: int | None = None, traffic: bool = False):
    names = TRAFFIC_PAIR_FEATURE_NAMES if traffic else PAIR_FEATURE_NAMES
    feature_batches = []
    target_batches = []
    raw_sample_count = 0
    if (sample_limit is not None and
            len(envelopes) * 2 > sample_limit):
        raise RuntimeError(
            "LearnedHungarian sample limit cannot cover every Webots episode")
    retained = 0
    for envelope_index, envelope_value in enumerate(envelopes):
        envelope = _load_supervision_envelope(envelope_value)
        episode_features = []
        episode_targets = []
        physical = envelope["result"]["formal_evaluation"][
            "physical_fine_tune"]
        for row in physical["rollout"]:
            supervision = _shared_supervision(row)
            field = "traffic_ridge_by_objective" if traffic else "ridge_by_objective"
            batch = supervision.get(field, {}).get(objective)
            if not isinstance(batch, dict):
                raise RuntimeError(
                    "LearnedHungarian objective supervision is missing")
            if traffic and (batch.get("feature_version") != TRAFFIC_PAIR_FEATURE_VERSION or
                            batch.get("traffic_version") != TRAFFIC_VERSION):
                raise RuntimeError("LearnedHungarian traffic supervision version mismatch")
            features = np.asarray(batch.get("features"), dtype=np.float64)
            targets = np.asarray(
                batch.get("targets"), dtype=np.float64).reshape(-1)
            if (features.shape !=
                    (len(targets), len(names)) or
                    not len(targets) or
                    not np.all(np.isfinite(features)) or
                    not np.all(np.isfinite(targets))):
                raise RuntimeError(
                    "LearnedHungarian shared sample is invalid")
            episode_features.append(features)
            episode_targets.append(targets)
            raw_sample_count += len(targets)
        if not episode_features:
            raise RuntimeError(
                "LearnedHungarian Webots episode has no supervision")
        if sample_limit is None:
            feature_batches.extend(episode_features)
            target_batches.extend(episode_targets)
        else:
            envelopes_left = len(envelopes) - envelope_index
            quota = max(2, (sample_limit - retained) // envelopes_left)
            features, targets = _bounded_supervised_batches(
                episode_features, episode_targets, sample_limit=quota,
                feature_dim=len(names))
            feature_batches.append(features)
            target_batches.append(targets)
            retained += len(targets)
    if not feature_batches:
        raise RuntimeError("LearnedHungarian shared corpus is empty")
    return feature_batches, target_batches, raw_sample_count


def _graph_imitation_shared_samples(
        envelopes: list[dict | Path], objective: str, *,
        sample_limit: int | None = None, traffic: bool = False):
    names = TRAFFIC_GRAPH_FEATURE_NAMES if traffic else GRAPH_FEATURE_NAMES
    feature_batches = []
    label_batches = []
    raw_sample_count = 0
    if (sample_limit is not None and
            len(envelopes) * 2 > sample_limit):
        raise RuntimeError(
            "GraphImitation sample limit cannot cover every Webots episode")
    retained = 0
    for envelope_index, envelope_value in enumerate(envelopes):
        envelope = _load_supervision_envelope(envelope_value)
        episode_features = []
        episode_labels = []
        for rows in _physical_groups([envelope]):
            for row in rows:
                supervision = _shared_supervision(row)
                field = "traffic_graph_by_objective" if traffic else "graph_by_objective"
                batch = supervision.get(field, {}).get(objective)
                if not isinstance(batch, dict):
                    raise RuntimeError(
                        "GraphImitation objective supervision is missing")
                if traffic and batch.get("feature_version") != TRAFFIC_GRAPH_FEATURE_VERSION:
                    raise RuntimeError("GraphImitation traffic supervision version mismatch")
                features = np.asarray(batch.get("features"), dtype=np.float64)
                labels = np.asarray(
                    batch.get("labels"), dtype=np.float64).reshape(-1)
                if (features.shape !=
                        (len(labels), len(names)) or
                        not len(labels) or
                        not np.all(np.isfinite(features)) or
                        not np.all(np.isin(labels, (0.0, 1.0)))):
                    raise RuntimeError(
                        "GraphImitation shared samples are invalid")
                episode_features.append(features)
                episode_labels.append(labels)
                raw_sample_count += len(labels)
        if not episode_features:
            raise RuntimeError(
                "GraphImitation Webots episode has no supervision")
        if sample_limit is None:
            feature_batches.extend(episode_features)
            label_batches.extend(episode_labels)
        else:
            envelopes_left = len(envelopes) - envelope_index
            quota = max(2, (sample_limit - retained) // envelopes_left)
            features, labels = _bounded_supervised_batches(
                episode_features, episode_labels, sample_limit=quota,
                feature_dim=len(names), binary=True)
            feature_batches.append(features)
            label_batches.append(labels)
            retained += len(labels)
    if not feature_batches:
        raise RuntimeError("GraphImitation shared corpus is empty")
    return feature_batches, label_batches, raw_sample_count


def _pair_rows(groups: list[list[dict]], expected_kind: str):
    transitions = []
    for rows in groups:
        for index, row in enumerate(rows):
            if row.get("kind") != expected_kind:
                raise RuntimeError("physical rollout kind mismatch")
            state = np.asarray(row["state"], np.float32)
            mask = np.asarray(row["action_mask"], bool)
            successor = rows[index + 1] if index + 1 < len(rows) else row
            next_state = np.asarray(successor["state"], np.float32)
            next_mask = np.asarray(successor["action_mask"], bool)
            transitions.append({
                "state": state, "mask": mask, "action": int(row["action"]),
                "reward": float(row["reward"]), "next_state": next_state,
                "next_mask": next_mask, "done": bool(row["done"]),
            })
    return transitions


def _frozen_webots_cql_dataset(envelopes: list[dict], objective: str):
    transitions = []
    transition_seeds = []
    seeds = []
    manifest_hashes = []
    for envelope in envelopes:
        info = envelope["result"]["experiment_info"]
        seed = info.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RuntimeError("Webots CQL rollout seed is invalid")
        if seed not in seeds:
            seeds.append(seed)
        manifest_hash = envelope.get("manifest_sha256")
        if not isinstance(manifest_hash, str):
            raise RuntimeError("Webots CQL manifest hash is invalid")
        if manifest_hash not in manifest_hashes:
            manifest_hashes.append(manifest_hash)
        group = _physical_groups([envelope])
        rows = _pair_rows(group, "pair_value")
        transitions.extend(rows)
        transition_seeds.extend([seed] * len(rows))
    if not transitions:
        raise RuntimeError("Webots CQL dataset is empty")
    return OfflineTransitionDataset(
        np.asarray([row["state"] for row in transitions], np.float32),
        np.asarray([row["mask"] for row in transitions], np.uint8),
        np.asarray([row["action"] for row in transitions], np.int64),
        np.asarray([row["reward"] for row in transitions], np.float32),
        np.asarray([row["next_state"] for row in transitions], np.float32),
        np.asarray([row["next_mask"] for row in transitions], np.uint8),
        np.asarray([row["done"] for row in transitions], np.uint8),
        "FrozenWebotsCQLPolicy/v1", tuple(seeds),
        np.asarray(transition_seeds, np.int64), objective,
        manifest_hashes=tuple(manifest_hashes))


def _set_fine_learning_rate(spec: dict, model, base_learning_rate):
    name = spec["scheduler"]
    desired = None if base_learning_rate is None else base_learning_rate * 0.1
    if name == "GraphPPO":
        model.config = replace(model.config, learning_rate=desired)
        model.network.config = model.config
        model.optimizer.learning_rate = desired
    elif name in ADVANCED_VALUE:
        model.config = replace(model.config, learning_rate=desired)
        model.optimizer.learning_rate = desired
    elif name in {"SARSA", "DQN"}:
        model.config.learning_rate = desired
    return desired


def _base_learning_rate(spec: dict, path: Path):
    name = spec["scheduler"]
    if name in {"GraphPPO", "SARSA", "DQN"} or name in ADVANCED_VALUE:
        return float(load_model(spec, path, seed=0).config.learning_rate)
    if name == "PPO_RL":
        return float(training_parameters(spec)["learning_rate"])
    return None


def _webots_round_objective_metric(envelopes: list[dict],
                                   spec: dict) -> dict:
    metric_name = ("completed_tasks" if spec["objective"] == "count" else
                   "utility_score_deadline_v2")
    scenario_values = {}
    for envelope in envelopes:
        result = envelope.get("result", {})
        scenario = result.get("experiment_info", {}).get("scenario")
        formal = result.get("formal_evaluation", {})
        if metric_name == "completed_tasks":
            value = formal.get("count_evaluation", {}).get(
                "total_tasks_completed")
            valid = (not isinstance(value, bool) and
                     isinstance(value, int) and value >= 0)
        else:
            value = formal.get("utility_v2_evaluation", {}).get(metric_name)
            valid = (not isinstance(value, bool) and
                     isinstance(value, (int, float)) and
                     math.isfinite(float(value)))
        if scenario not in SCENARIOS or scenario in scenario_values or not valid:
            raise RuntimeError("Webots fine-tune objective metric is invalid")
        scenario_values[scenario] = float(value)
    if set(scenario_values) != set(SCENARIOS):
        raise RuntimeError("Webots fine-tune round is not A/B/C balanced")
    return {
        "name": metric_name,
        "mean": statistics.fmean(scenario_values.values()),
        "by_scenario": {
            scenario: scenario_values[scenario] for scenario in SCENARIOS},
    }


def _webots_parameter_convergence(round_records: list[dict]) -> dict:
    changes = [
        float(row["parameter_change"]["relative_l2_change"])
        for row in round_records]
    recent = changes[-WEBOTS_FINETUNE_CONVERGENCE_PATIENCE:]
    converged = bool(
        len(round_records) >= MIN_WEBOTS_FINETUNE_BALANCED_ROUNDS and
        len(recent) == WEBOTS_FINETUNE_CONVERGENCE_PATIENCE and
        all(value <= WEBOTS_FINETUNE_PARAMETER_CHANGE_THRESHOLD
            for value in recent))
    return {
        "schema_version": 1,
        "converged": converged,
        "completed_balanced_rounds": len(round_records),
        "minimum_balanced_rounds": MIN_WEBOTS_FINETUNE_BALANCED_ROUNDS,
        "maximum_balanced_rounds": MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS,
        "patience_rounds": WEBOTS_FINETUNE_CONVERGENCE_PATIENCE,
        "relative_l2_change_threshold":
            WEBOTS_FINETUNE_PARAMETER_CHANGE_THRESHOLD,
        "recent_relative_l2_changes": recent,
        "all_relative_l2_changes": changes,
    }


def _webots_fine_tune_completion(convergence: dict) -> dict:
    """Classify a physical fine-tune stop without faking convergence.

    Parameter convergence remains the preferred early-stop condition.  The
    operator's ten-round ceiling is also a completed collection stage: its
    candidate proceeds to the bounded Webots regression gate instead of
    blocking the pipeline or claiming parameter convergence.
    """
    if not isinstance(convergence, dict):
        raise TypeError("Webots convergence evidence must be an object")
    rounds = convergence.get("completed_balanced_rounds")
    converged = convergence.get("converged")
    if (not isinstance(rounds, int) or isinstance(rounds, bool) or
            rounds < 0 or not isinstance(converged, bool)):
        raise ValueError("invalid Webots convergence evidence")
    maximum_reached = rounds == MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS
    completed = converged or maximum_reached
    return {
        "completed": completed,
        "completion_reason": (
            "parameter_converged" if converged else
            "maximum_balanced_rounds_reached" if maximum_reached else
            "incomplete"),
        "parameter_converged": converged,
        "maximum_rounds_reached": maximum_reached,
        "direct_final_evaluation_eligible": completed,
        # Retained for backward-compatible reads of existing reports.
        "nonregression_eligible": completed,
        "final_evaluation_seed_count": len(FINAL_EVALUATION_SEEDS),
    }


def update_physical_model(spec: dict, input_checkpoint: Path,
                          output_checkpoint: Path,
                          current_envelopes: list[dict],
                          history_envelopes: list[dict],
                          base_learning_rate,
                          shared_supervision_envelopes:
                              list[dict | Path] | None = None
                          ) -> dict:
    name = spec["scheduler"]
    model = load_model(spec, input_checkpoint, seed=0)
    fine_rate = _set_fine_learning_rate(
        spec, model, base_learning_rate)
    current_groups = _physical_groups(current_envelopes)
    all_groups = _physical_groups(history_envelopes + current_envelopes)
    losses = []
    offline_contract = None
    ppo_update = None
    shared_supervision_envelopes = list(
        shared_supervision_envelopes or [])
    shared_supervision_samples = 0

    if name == "LearnedHungarian":
        parameters = training_parameters(spec)
        traffic = bool(spec.get("traffic_version"))
        names = TRAFFIC_PAIR_FEATURE_NAMES if traffic else PAIR_FEATURE_NAMES
        if not shared_supervision_envelopes and not traffic:
            raise RuntimeError(
                "LearnedHungarian requires the shared Webots corpus")
        fit_features, fit_targets, shared_supervision_samples = (
            _learned_hungarian_shared_samples(
                shared_supervision_envelopes, spec["objective"],
                sample_limit=200_000, traffic=traffic)
            if shared_supervision_envelopes else ([], [], 0))
        own_features, own_targets, _own_raw_count = (
            _learned_hungarian_shared_samples(
                history_envelopes + current_envelopes, spec["objective"],
                sample_limit=200_000, traffic=traffic))
        fit_features.extend(own_features)
        fit_targets.extend(own_targets)
        _current_features, _current_targets, current_count = (
            _learned_hungarian_shared_samples(
                current_envelopes, spec["objective"],
                sample_limit=200_000, traffic=traffic))
        fit_features, fit_targets = _bounded_supervised_batches(
            fit_features, fit_targets, sample_limit=200_000,
            feature_dim=len(names))
        if current_count < 1:
            raise RuntimeError(
                "LearnedHungarian Webots fine tune has no physical samples")
        model = RidgeCostModel.fit(
            fit_features, fit_targets,
            l2=parameters["l2"],
            target_name=f"{spec['objective']}_expert_cost")
        updates = 1
        samples = current_count
    elif name == "GraphImitation":
        parameters = training_parameters(spec)
        traffic = bool(spec.get("traffic_version"))
        names = TRAFFIC_GRAPH_FEATURE_NAMES if traffic else GRAPH_FEATURE_NAMES
        if not shared_supervision_envelopes and not traffic:
            raise RuntimeError(
                "GraphImitation requires the shared Webots corpus")
        fit_features, fit_labels, shared_supervision_samples = (
            _graph_imitation_shared_samples(
                shared_supervision_envelopes, spec["objective"],
                sample_limit=50_000, traffic=traffic)
            if shared_supervision_envelopes else ([], [], 0))
        own_features, own_labels, _own_raw_count = (
            _graph_imitation_shared_samples(
                history_envelopes + current_envelopes, spec["objective"],
                sample_limit=50_000, traffic=traffic))
        fit_features.extend(own_features)
        fit_labels.extend(own_labels)
        _current_features, _current_labels, current_count = (
            _graph_imitation_shared_samples(
                current_envelopes, spec["objective"],
                sample_limit=50_000, traffic=traffic))
        if not fit_labels:
            raise RuntimeError("GraphImitation Webots samples are empty")
        fit_features, fit_labels = _bounded_supervised_batches(
            fit_features, fit_labels, sample_limit=50_000,
            feature_dim=len(names), binary=True)
        model = GraphEdgeImitationModel.fit(
            fit_features, fit_labels,
            epochs=parameters["epochs"],
            learning_rate=parameters["learning_rate"] * 0.1,
            l2=parameters["l2"])
        updates = 1
        samples = current_count
    elif name == "GraphPPO":
        rollout = []
        for rows in current_groups:
            for row in rows:
                if row.get("kind") != "graph_ppo":
                    raise RuntimeError("GraphPPO physical rollout kind mismatch")
                rollout.append(GraphRolloutStep(
                    np.asarray(row["features"], np.float32),
                    int(row["action_index"]),
                    float(row["old_log_probability"]),
                    float(row["value"]), float(row["reward"]),
                    bool(row["done"])))
        update = model.update(rollout)
        # ``update`` processes all terminal markers correctly but owns one
        # aggregate optimizer call.  Preserve the physical episode counter as
        # three A/B/C episodes, matching standalone and the saved evidence.
        model.training_episodes += max(0, len(current_groups) - 1)
        if update["mean_loss"] is not None:
            losses.append(float(update["mean_loss"]))
        updates = int(update["updates"])
        samples = len(rollout)
    elif name == "CQL":
        # All three physical episodes complete before this immutable batch is
        # constructed and fitted; no within-episode online CQL update occurs.
        dataset = _frozen_webots_cql_dataset(
            history_envelopes + current_envelopes, spec["objective"])
        current_count = len(_pair_rows(current_groups, "pair_value"))
        offline_contract = model.ingest_offline_dataset(dataset)
        for _ in range(max(1, current_count)):
            loss = model.train_step()
            if loss is not None:
                losses.append(float(loss))
        if not losses:
            raise RuntimeError("Webots CQL produced no offline update")
        updates = len(losses)
        samples = current_count
        offline_contract = {
            **offline_contract,
            "strictly_offline_updates": True,
            "episodes_frozen_before_update": len(
                history_envelopes + current_envelopes),
            "current_episode_samples": current_count,
        }
    elif name in ADVANCED_VALUE:
        transitions = _pair_rows(all_groups, "pair_value")
        current_count = len(_pair_rows(current_groups, "pair_value"))
        for row in transitions:
            model.remember(
                row["state"], row["mask"], row["action"], row["reward"],
                row["next_state"], row["next_mask"], row["done"])
        for _ in range(max(1, current_count)):
            loss = model.train_step()
            if loss is not None:
                losses.append(float(loss))
        updates = len(losses)
        samples = current_count
    elif name == "DQN":
        # DQN v2 checkpoint already carries the exact bounded replay state;
        # only the newly committed physical actions are appended this round.
        transitions = _pair_rows(current_groups, "dqn")
        current_count = len(transitions)
        for row in transitions:
            model.remember(
                row["state"], row["action"], row["reward"],
                row["next_state"], row["done"], row["next_mask"])
        for _ in range(max(1, current_count)):
            loss = model.train_step()
            if loss is not None:
                losses.append(float(loss))
        model.episode += len(current_groups)
        updates = len(losses)
        samples = current_count
    elif name == "SARSA":
        transitions = []
        for group in current_groups:
            episode_transitions = _pair_rows([group], "sarsa")
            for index, row in enumerate(episode_transitions):
                state = model.discretize(row["state"])
                next_state = model.discretize(row["next_state"])
                next_action = (
                    episode_transitions[index + 1]["action"]
                    if index + 1 < len(episode_transitions) and
                    not row["done"] else model.no_op_action)
                losses.append(abs(float(model.update(
                    state, row["action"], row["reward"], next_state,
                    next_action, row["done"]))))
            transitions.extend(episode_transitions)
            model.end_episode()
        updates = len(losses)
        samples = len(transitions)
    elif name == "PPO_RL":
        parameters = training_parameters(spec)
        rollout = []
        for rows in current_groups:
            for row in rows:
                if row.get("kind") != "pairwise_ppo":
                    raise RuntimeError("PPO physical rollout kind mismatch")
                rollout.append((
                    np.asarray(row["state"], np.float64),
                    np.asarray(row["action_mask"], bool),
                    int(row["action"]), float(row["reward"]),
                    float(row["old_log_probability"]),
                    float(row["value"]), bool(row["done"])))
        ppo_update = _pairwise_ppo_update(
            model, rollout, learning_rate=base_learning_rate * 0.1,
            clip_ratio=parameters["clip_ratio"],
            value_coefficient=parameters["value_coefficient"],
            entropy_coefficient=parameters["entropy_coefficient"],
            update_epochs=parameters["update_epochs"],
            max_grad_norm=parameters["max_grad_norm"],
            gamma=parameters["gamma"],
            value_huber_delta=parameters["value_huber_delta"])
        losses.append(ppo_update["mean_loss"])
        updates = 1
        samples = len(rollout)
    elif name == "LinUCB":
        samples = 0
        selection_counts = {arm: 0 for arm in model.arms}
        failure_reasons = {arm: {} for arm in model.arms}
        reward_by_arm = {arm: [] for arm in model.arms}
        for rows in current_groups:
            for row in rows:
                if row.get("kind") != "linucb":
                    raise RuntimeError("LinUCB physical rollout kind mismatch")
                arm = str(row["arm"])
                action_reward = float(row["reward"])
                model.update(
                    np.asarray(row["context"], np.float64),
                    arm, action_reward)
                selection_counts[arm] += 1
                reward_by_arm[arm].append(action_reward)
                samples += 1
        updates = samples
    else:
        raise ValueError(f"unsupported physical fine tune policy: {name}")

    atomic_save_model(spec, model, output_checkpoint)
    result = {
        "samples": samples,
        "optimizer_updates": updates,
        "mean_loss": statistics.fmean(losses) if losses else None,
        "learning_rate": fine_rate,
        "learning_rate_multiplier": (
            0.1 if base_learning_rate is not None else None),
    }
    if name in {"LearnedHungarian", "GraphImitation"}:
        shared_sources = set()
        shared_identities = []
        for envelope_value in shared_supervision_envelopes:
            envelope = _load_supervision_envelope(envelope_value)
            physical = envelope["result"]["formal_evaluation"][
                "physical_fine_tune"]
            shared_sources.add(
                physical["algorithm"] + "_" + physical["objective"])
            shared_identities.append({
                "manifest_sha256": envelope.get("manifest_sha256"),
                "input_model_sha256": envelope.get("input_model_sha256"),
                "algorithm": physical["algorithm"],
                "objective": physical["objective"],
                "scenario": envelope["result"]["experiment_info"][
                    "scenario"],
                "seed": envelope["result"]["experiment_info"]["seed"],
            })
        shared_sources = sorted(shared_sources)
        shared_corpus_sha256 = _canonical_sha256(shared_identities)
        result.update({
            "shared_supervision_contract":
                WEBOTS_SHARED_SUPERVISION_VERSION,
            "shared_supervision_episode_count":
                len(shared_supervision_envelopes),
            "shared_supervision_sample_count": shared_supervision_samples,
            "shared_supervision_source_policy_ids": shared_sources,
            "shared_supervision_corpus_sha256": shared_corpus_sha256,
            "shared_supervision_objective": spec["objective"],
            "shared_supervision_excludes_evaluation": True,
        })
    if offline_contract is not None:
        result["offline_dataset"] = offline_contract
    if name == "LinUCB":
        result.update({
            "reward_attribution": "physical_task_lifecycle_action_level",
            "arm_selection_counts": selection_counts,
            "arm_failure_reasons": failure_reasons,
            "mean_action_reward_by_arm": {
                arm: (statistics.fmean(values) if values else None)
                for arm, values in reward_by_arm.items()},
            "model_training_audit": model.training_audit(),
        })
    if ppo_update is not None:
        result.update({
            "reward_attribution": "physical_task_lifecycle_action_level",
            "ppo_update": ppo_update,
        })
    if name == "SARSA":
        result.update({
            "reward_attribution": "physical_task_lifecycle_action_level",
            "gamma": model.config.gamma,
            "epsilon": model.epsilon,
            "epsilon_decay": model.config.epsilon_decay,
            "physical_epsilon_cap": model.config.physical_epsilon_cap,
            "state_version": "sarsa-state-v2-priority-robot-task-slots",
        })
    if name == "DQN":
        result.update({
            "reward_attribution": "physical_task_lifecycle_action_level",
            "gamma": model.config.gamma,
            "epsilon": model.epsilon,
            "epsilon_decay_steps": model.config.epsilon_decay_steps,
            "physical_epsilon_cap": model.config.physical_epsilon_cap,
            "model_version": DQN_MODEL_VERSION,
            "replay_size": len(model.replay),
        })
    return result


def _valid_fine_record(record, spec: dict, round_index: int, seed: int,
                       input_hash: str, checkpoint: Path,
                       episode_paths: list[Path]) -> bool:
    if not isinstance(record, dict) or not checkpoint.is_file():
        return False
    try:
        validate_model_identity(spec, checkpoint)
    except (ModelValidationError, OSError, ValueError):
        return False
    valid = bool(
        record.get("schema_version") == TRAINING_RECORD_SCHEMA_VERSION and
        record.get("training_semantics_version") ==
            TRAINING_SEMANTICS_VERSION and
        record.get("policy_id") == spec["policy_id"] and
        record.get("scheduler") == spec["scheduler"] and
        record.get("objective") == spec["objective"] and
        record.get("round") == round_index and record.get("seed") == seed and
        record.get("input_checkpoint_sha256") == input_hash and
        record.get("output_checkpoint_sha256") == sha256_file(checkpoint) and
        record.get("output_model_identity_sha256") ==
            sha256_file(model_identity_path(checkpoint)) and
        record.get("episode_result_sha256") == [
            sha256_file(path) if path.is_file() else None
            for path in episode_paths] and
        all(webots_workflow.valid_episode(
            load_json(path), scenario=scenario, seed=seed,
            manifest=manifest_path("webots_finetune", scenario, seed),
            input_model_sha256=input_hash,
            fine_tune_objective=spec["objective"],
            fine_tune_algorithm=spec["scheduler"],
            expected_scheduler=spec["scheduler"])
            for path, scenario in zip(episode_paths, SCENARIOS)))
    if not valid:
        return False
    try:
        envelopes = [load_json(path) for path in episode_paths]
        metric = _webots_round_objective_metric(envelopes, spec)
        parameter_change = record.get("parameter_change", {})
        if not (
                record.get("objective_metric") == metric and
                isinstance(parameter_change, dict) and
                parameter_change.get("schema_version") == 1 and
                parameter_change.get("reference_round") == round_index-1 and
                isinstance(parameter_change.get("parameter_count"), int) and
                not isinstance(parameter_change.get("parameter_count"), bool) and
                parameter_change["parameter_count"] > 0 and
                isinstance(parameter_change.get("relative_l2_change"),
                           (int, float)) and
                not isinstance(parameter_change.get("relative_l2_change"), bool) and
                math.isfinite(parameter_change["relative_l2_change"]) and
                parameter_change["relative_l2_change"] >= 0):
            return False
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    if spec["scheduler"] == "CQL":
        contract = record.get("model_update", {}).get(
            "offline_dataset", {})
        return bool(
            contract.get("dataset_version") ==
            "cql-offline-dataset-v2-fixed-behavior" and
            contract.get("collection_mode") == "fixed_offline_behavior" and
            contract.get("strictly_offline_updates") is True and
            contract.get("episodes_frozen_before_update", 0) >= 3)
    return True


def _shared_webots_supervision_corpus(
        output_dir: Path, requesting_spec: dict) -> list[Path]:
    """Load only validated training-partition Webots trajectories.

    Regression and final-evaluation directories are never searched.  Model,
    objective, seed, manifest and checkpoint hashes are revalidated through
    the same resume gate used by the source policy before any sample is read.
    """
    source_specs = [
        spec for spec in AI_SPECS
        if spec["scheduler"] not in SUPERVISED_WEBOTS_ALGORITHMS]
    expected_sources = (
        2 * (len(TRAINABLE_ALGORITHMS)-len(SUPERVISED_WEBOTS_ALGORITHMS)))
    if len(source_specs) != expected_sources:
        raise RuntimeError("shared Webots source registry is incomplete")
    corpus = []
    for source in source_specs:
        policy_dir = output_dir / "webots_fine_tune" / source["policy_id"]
        report = load_json(policy_dir / "fine_tune_report.json")
        rounds = report.get("balanced_rounds") if isinstance(report, dict) else None
        if (not isinstance(rounds, int) or isinstance(rounds, bool) or
                report.get("status") != "complete" or
                report.get("policy_id") != source["policy_id"] or
                report.get("scheduler") != source["scheduler"] or
                report.get("objective") != source["objective"] or
                rounds < WEBOTS_FINETUNE_CONVERGENCE_PATIENCE or
                rounds > MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS):
            raise RuntimeError(
                f"shared supervision source is incomplete: "
                f"{source['policy_id']}")
        current = headless_final_path(output_dir, source)
        for round_index in range(1, rounds+1):
            seed = WEBOTS_FINETUNE_SEEDS[round_index-1]
            checkpoint = fine_tune_round_checkpoint_path(
                policy_dir, source, round_index)
            record = load_json(
                policy_dir / "checkpoints" /
                f"after_round_{round_index:02d}.json")
            episode_paths = [
                policy_dir / "episodes" /
                f"round_{round_index:02d}_seed_{seed}" / scenario /
                "result.json" for scenario in SCENARIOS]
            input_hash = sha256_file(current)
            if not _valid_fine_record(
                    record, source, round_index, seed, input_hash,
                    checkpoint, episode_paths):
                raise RuntimeError(
                    f"shared supervision source failed validation: "
                    f"{source['policy_id']}/round-{round_index}")
            # Keep only paths here.  Supervised consumers stream one validated
            # physical episode at a time, avoiding a corpus-sized JSON object
            # graph while still using every source episode.
            corpus.extend(episode_paths)
            current = checkpoint
        final = fine_tuned_final_path(output_dir, source)
        if not (final.is_file() and
                report.get("output_checkpoint_sha256") == sha256_file(final) ==
                sha256_file(current)):
            raise RuntimeError(
                f"shared supervision final checkpoint mismatch: "
                f"{source['policy_id']}")
    if len(corpus) < expected_sources * len(SCENARIOS) * (
            WEBOTS_FINETUNE_CONVERGENCE_PATIENCE):
        raise RuntimeError("shared Webots supervision corpus is insufficient")
    return corpus


def fine_tune_all(output_dir: Path, webots: Path, update_state,
                  max_new_episodes: int | None = None, *, specs=None,
                  progress: dict | None = None) -> bool:
    requested_specs = tuple(
        PIPELINE_AI_SPECS if specs is None else specs)
    executed = int((progress or {}).get("fine_tune", 0))
    fine_tune_specs = tuple(
        spec for spec in requested_specs
        if spec["scheduler"] not in SUPERVISED_WEBOTS_ALGORITHMS) + tuple(
        spec for spec in requested_specs
        if spec["scheduler"] in SUPERVISED_WEBOTS_ALGORITHMS)
    for spec in fine_tune_specs:
        policy_dir = output_dir / "webots_fine_tune" / spec["policy_id"]
        current = headless_final_path(output_dir, spec)
        if not current.is_file():
            raise FileNotFoundError(
                f"sufficient standalone checkpoint is missing: {current}")
        validate_model_identity(spec, current)
        readiness = load_json(
            output_dir / "headless" / spec["policy_id"] /
            "standalone_readiness.json")
        training_report = load_json(
            output_dir / "headless" / spec["policy_id"] /
            "training_report.json")
        standalone_exit_audit_valid = False
        if isinstance(readiness, dict) and isinstance(training_report, dict):
            exit_mode = training_report.get("standalone_exit_mode")
            if exit_mode == "operator_requested_webots_transition":
                request = _load_standalone_exit_override(output_dir, spec)
                standalone_exit_audit_valid = bool(
                    isinstance(request, dict) and
                    readiness.get("standalone_exit_mode") == exit_mode and
                    readiness.get("standalone_exit_override_sha256") ==
                        request["report_sha256"] and
                    training_report.get(
                        "standalone_exit_override_sha256") ==
                        request["report_sha256"] and
                    training_report.get("natural_readiness_status") ==
                        readiness.get("natural_status") and
                    training_report.get("parameter_converged") ==
                        readiness.get("parameter_converged"))
            elif exit_mode in {None, "natural_convergence"}:
                standalone_exit_audit_valid = bool(
                    not readiness.get("standalone_exit_mode") and
                    not training_report.get(
                        "standalone_exit_override_sha256"))
            elif exit_mode == "maximum_balanced_rounds_reached":
                standalone_exit_audit_valid = bool(
                    readiness.get("status") ==
                        "maximum_completed_for_webots_fine_tune" and
                    readiness.get("maximum_rounds_reached") is True and
                    readiness.get("completed_balanced_rounds") ==
                        MAX_STANDALONE_BALANCED_ROUNDS and
                    readiness.get("standalone_completion_reason") ==
                        "maximum_balanced_rounds_reached" and
                    training_report.get("maximum_rounds_reached") is True and
                    training_report.get("standalone_completion_reason") ==
                        "maximum_balanced_rounds_reached" and
                    training_report.get("parameter_converged") ==
                        readiness.get("parameter_converged") and
                    not readiness.get("standalone_exit_mode") and
                    not training_report.get(
                        "standalone_exit_override_sha256"))
        if not (
                isinstance(readiness, dict) and
                standalone_exit_audit_valid and
                readiness.get("webots_fine_tune_eligible") is True and
                readiness.get("policy_id") == spec["policy_id"] and
                readiness.get("objective") == spec["objective"] and
                readiness.get("completed_balanced_rounds", 0) >=
                    MIN_STANDALONE_BALANCED_ROUNDS and
                isinstance(training_report, dict) and
                training_report.get("training_contract") ==
                    "standalone_sufficient_training" and
                training_report.get("policy_id") == spec["policy_id"] and
                training_report.get("scheduler") == spec["scheduler"] and
                training_report.get("objective") == spec["objective"] and
                training_report.get("webots_fine_tune_eligible") is True and
                training_report.get("output_checkpoint_sha256") ==
                    sha256_file(current) and
                training_report.get("output_model_identity_sha256") ==
                    sha256_file(model_identity_path(current))):
            raise RuntimeError(
                f"{spec['policy_id']} has not passed sufficient standalone "
                "training; Webots fine-tuning is forbidden")
        base_lr = _base_learning_rate(spec, current)
        shared_supervision_envelopes = (
            _shared_webots_supervision_corpus(output_dir, spec)
            if spec["scheduler"] in SUPERVISED_WEBOTS_ALGORITHMS else [])
        history_envelopes = []
        round_records = []
        convergence = _webots_parameter_convergence(round_records)
        for round_index, seed in enumerate(
                WEBOTS_FINETUNE_SEEDS, start=1):
            checkpoint_dir = policy_dir / "checkpoints"
            checkpoint = fine_tune_round_checkpoint_path(
                policy_dir, spec, round_index)
            record_path = checkpoint_dir / f"after_round_{round_index:02d}.json"
            input_hash = sha256_file(current)
            episode_paths = [
                policy_dir / "episodes" /
                f"round_{round_index:02d}_seed_{seed}" / scenario /
                "result.json" for scenario in SCENARIOS]
            record = load_json(record_path)
            if _valid_fine_record(
                    record, spec, round_index, seed, input_hash,
                    checkpoint, episode_paths):
                print(
                    f"[RESUME] webots-fine/{spec['policy_id']}/"
                    f"round-{round_index}", flush=True)
                history_envelopes.extend(
                    load_json(path) for path in episode_paths)
                round_records.append(record)
                current = checkpoint
                convergence = _webots_parameter_convergence(round_records)
                if convergence["converged"]:
                    print(
                        f"[CONVERGED] webots-fine/{spec['policy_id']} "
                        f"round={round_index} parameter_delta="
                        f"{record['parameter_change']['relative_l2_change']:.6g}",
                        flush=True)
                    break
                continue
            envelopes = []
            for scenario, canonical in zip(SCENARIOS, episode_paths):
                cached = load_json(canonical)
                if not webots_workflow.valid_episode(
                        cached, scenario=scenario, seed=seed,
                        manifest=manifest_path(
                            "webots_finetune", scenario, seed),
                        input_model_sha256=input_hash,
                        fine_tune_objective=spec["objective"],
                        fine_tune_algorithm=spec["scheduler"],
                        expected_scheduler=spec["scheduler"]):
                    if (max_new_episodes is not None and
                            executed >= max_new_episodes):
                        return False
                    update_state(
                        stage="webots_fine_tune",
                        current={"policy_id": spec["policy_id"],
                                 "round": round_index,
                                 "scenario": scenario, "seed": seed})
                    envelope, _ = webots_workflow.run_or_resume_episode(
                        canonical_path=canonical, webots=webots,
                        scenario=scenario, seed=seed,
                        manifest=manifest_path(
                            "webots_finetune", scenario, seed),
                        scheduler=spec["scheduler"], model_path=current,
                        fine_tune_objective=spec["objective"],
                        fine_tune_algorithm=spec["scheduler"])
                    executed += 1
                    if progress is not None:
                        progress["fine_tune"] = executed
                else:
                    envelope = cached
                    print(
                        f"[RESUME] webots-fine/{spec['policy_id']}/"
                        f"{scenario}/{seed}", flush=True)
                envelopes.append(envelope)
            update = update_physical_model(
                spec, current, checkpoint, envelopes,
                history_envelopes, base_lr,
                shared_supervision_envelopes)
            parameter_change = _policy_parameter_change(
                spec, current, checkpoint,
                reference_round=round_index-1)
            objective_metric = _webots_round_objective_metric(
                envelopes, spec)
            record = {
                "schema_version": TRAINING_RECORD_SCHEMA_VERSION,
                "training_semantics_version": TRAINING_SEMANTICS_VERSION,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "policy_id": spec["policy_id"],
                "scheduler": spec["scheduler"],
                "objective": spec["objective"],
                "round": round_index, "seed": seed,
                "scenarios": list(SCENARIOS),
                "input_checkpoint_sha256": input_hash,
                "output_checkpoint_sha256": sha256_file(checkpoint),
                "output_model_identity_sha256": sha256_file(
                    model_identity_path(checkpoint)),
                "episode_result_sha256": [
                    sha256_file(path) for path in episode_paths],
                "objective_metric": objective_metric,
                "parameter_change": parameter_change,
                "model_update": update,
            }
            atomic_json(record_path, record)
            history_envelopes.extend(envelopes)
            round_records.append(record)
            current = checkpoint
            convergence = _webots_parameter_convergence(round_records)
            print(
                f"[CHECKPOINT] webots-fine/{spec['policy_id']} "
                f"round={round_index}/{MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS} "
                "parameter_delta="
                f"{parameter_change['relative_l2_change']:.6g} "
                f"sha256={record['output_checkpoint_sha256'][:12]}",
                flush=True)
            if convergence["converged"]:
                print(
                    f"[CONVERGED] webots-fine/{spec['policy_id']} "
                    f"round={round_index}", flush=True)
                break
        completion = _webots_fine_tune_completion(convergence)
        if not completion["completed"]:
            raise RuntimeError(
                f"{spec['policy_id']} Webots fine-tuning stopped before "
                "convergence or the configured maximum")
        if not completion["parameter_converged"]:
            print(
                f"[MAXIMUM] webots-fine/{spec['policy_id']} completed "
                f"{len(round_records)} balanced rounds without parameter "
                "convergence; proceeding directly to the "
                f"{len(FINAL_EVALUATION_SEEDS)}-seed final evaluation",
                flush=True)
        final = fine_tuned_final_path(output_dir, spec)
        atomic_copy_model(spec, current, final)
        load_model(spec, final, seed=0)
        atomic_json(policy_dir / "fine_tune_report.json", {
            "status": "complete", "policy_id": spec["policy_id"],
            "scheduler": spec["scheduler"], "objective": spec["objective"],
            "duration_seconds_per_episode": DURATION_SECONDS,
            "balanced_rounds": len(round_records),
            "physical_episode_count": len(SCENARIOS)*len(round_records),
            "fine_tune_seeds": [
                int(row["seed"]) for row in round_records],
            "parameter_convergence": convergence,
            "completion_reason": completion["completion_reason"],
            "parameter_converged": completion["parameter_converged"],
            "maximum_rounds_reached":
                completion["maximum_rounds_reached"],
            "direct_final_evaluation_eligible":
                completion["direct_final_evaluation_eligible"],
            "nonregression_eligible":
                completion["nonregression_eligible"],
            "final_evaluation_seed_count":
                completion["final_evaluation_seed_count"],
            "final_evaluation_seeds": list(FINAL_EVALUATION_SEEDS),
            "output_checkpoint": str(final.resolve()),
            "output_checkpoint_sha256": sha256_file(final),
            "output_model_identity_sha256": sha256_file(
                model_identity_path(final)),
            "resume_granularity": (
                "each completed Webots episode and each A/B/C model update"),
        })
    return True


def _webots_regression_row(envelope: dict, spec: dict,
                           checkpoint_hash: str, scenario: str,
                           seed: int) -> dict:
    result = envelope.get("result", {})
    info = result.get("experiment_info", {})
    summary = result.get("summary_metrics", {})
    formal = result.get("formal_evaluation", {})
    count = formal.get("count_evaluation", {})
    utility = formal.get("utility_v2_evaluation", {})
    return {
        "partition": "webots_regression",
        "policy_id": spec["policy_id"],
        "scheduler": spec["scheduler"],
        "objective": spec["objective"],
        "scenario": scenario,
        "seed": seed,
        "duration_seconds": float(info.get("sim_duration", -1)),
        "current_time": float(info.get("sim_duration", -1)),
        "runtime_mode": info.get("runtime_mode"),
        "physics_fidelity": "webots_physical",
        "fixed_horizon": True,
        "horizon_finalized": True,
        "termination_reason": "episode_horizon",
        "physical_collision_observation_status": summary.get(
            "physical_collision_observation_status"),
        "pure_policy_eligible": bool(
            summary.get("fallback_scheduler_commits") == 0 and
            summary.get("invalid_scheduler_outputs") == 0 and
            summary.get("scheduler_timeout_count") == 0),
        "fallback_scheduler_commits": summary.get(
            "fallback_scheduler_commits"),
        "invalid_scheduler_outputs": summary.get(
            "invalid_scheduler_outputs"),
        "scheduler_timeout_count": summary.get(
            "scheduler_timeout_count"),
        "checkpoint_sha256": checkpoint_hash,
        "task_manifest_sha256": info.get("task_manifest_sha256"),
        "completed_tasks": count.get("total_tasks_completed"),
        "utility_score_deadline_v2": utility.get(
            "utility_score_deadline_v2"),
    }


def _run_webots_nonregression_gate(
        output_dir: Path, webots: Path, update_state,
        max_new_runs: int | None = None, *, specs=None,
        progress: dict | None = None) -> bool:
    """Run paired pre/post Webots checks and atomically accept or roll back."""
    requested_specs = tuple(
        PIPELINE_AI_SPECS if specs is None else specs)
    executed = int((progress or {}).get("regression", 0))
    for spec in requested_specs:
        pre = headless_final_path(output_dir, spec)
        post = fine_tuned_final_path(output_dir, spec)
        if not pre.is_file() or not post.is_file():
            raise FileNotFoundError(
                f"pre/post fine-tune checkpoint missing: {spec['policy_id']}")
        validate_model_identity(spec, pre)
        validate_model_identity(spec, post)
        side_rows = {"pre": [], "post": []}
        for side, checkpoint in (("pre", pre), ("post", post)):
            checkpoint_hash = sha256_file(checkpoint)
            for scenario in SCENARIOS:
                for seed in WEBOTS_REGRESSION_SEEDS:
                    canonical = (
                        output_dir / "webots_regression" / spec["policy_id"] /
                        side / scenario / str(seed) / "result.json")
                    manifest = manifest_path(
                        "webots_regression", scenario, seed)
                    cached = load_json(canonical)
                    if not webots_workflow.valid_episode(
                            cached, scenario=scenario, seed=seed,
                            manifest=manifest,
                            input_model_sha256=checkpoint_hash,
                            fine_tune_objective=None,
                            expected_scheduler=spec["scheduler"]):
                        if (max_new_runs is not None and
                                executed >= max_new_runs):
                            return False
                        update_state(
                            stage="webots_fine_tune_nonregression",
                            current={"policy_id": spec["policy_id"],
                                     "side": side, "scenario": scenario,
                                     "seed": seed})
                        cached, _ = webots_workflow.run_or_resume_episode(
                            canonical_path=canonical, webots=webots,
                            scenario=scenario, seed=seed, manifest=manifest,
                            scheduler=spec["scheduler"],
                            model_path=checkpoint,
                            fine_tune_objective=None)
                        executed += 1
                        if progress is not None:
                            progress["regression"] = executed
                    side_rows[side].append(_webots_regression_row(
                        cached, spec, checkpoint_hash, scenario, seed))
        degradation_tolerance = _registered_webots_regression_tolerance(
            output_dir, spec)
        decision = webots_finetune_regression_decision(
            spec["objective"], pre, post,
            side_rows["pre"], side_rows["post"],
            policy_id=spec["policy_id"], scheduler=spec["scheduler"],
            degradation_tolerance=degradation_tolerance)
        selected = selected_final_path(output_dir, spec)
        source = post if decision["accepted"] else pre
        atomic_copy_model(spec, source, selected)
        if sha256_file(selected) != decision["selected_checkpoint_sha256"]:
            raise RuntimeError("selected Webots checkpoint hash mismatch")
        validate_model_identity(spec, selected)
        decision["selected_checkpoint"] = str(selected.resolve())
        decision["selected_checkpoint_sha256"] = sha256_file(selected)
        decision["selected_model_identity_sha256"] = sha256_file(
            model_identity_path(selected))
        decision["source_checkpoint"] = str(source.resolve())
        decision["source_checkpoint_sha256"] = sha256_file(source)
        decision["source_model_identity_sha256"] = sha256_file(
            model_identity_path(source))
        decision["resume_granularity"] = (
            "each complete pre/post Webots 1800s regression episode")
        decision["report_sha256"] = _canonical_sha256({
            key: value for key, value in decision.items()
            if key != "report_sha256"})
        atomic_json(
            output_dir / "webots_fine_tune" / spec["policy_id"] /
            "nonregression_decision.json", decision)
    return True


def select_fine_tuned_models_for_final_evaluation(
        output_dir: Path, update_state, *, specs=None,
        selection_reason: str =
        "operator_requested_skip_of_post_finetune_nonregression",
        ignored_timeout_gate: bool = False) -> bool:
    """Select fine-tuned checkpoints directly, without extra rollouts.

    Any already committed regression episodes remain untouched, but they are
    deliberately ignored for model selection under this operator request.
    """
    requested_specs = tuple(
        PIPELINE_AI_SPECS if specs is None else specs)
    if (not isinstance(selection_reason, str) or not selection_reason or
            not isinstance(ignored_timeout_gate, bool)):
        raise ValueError("direct fine-tuned selection audit is invalid")
    for spec in requested_specs:
        if spec["policy_id"] not in DIRECT_FINETUNED_EVALUATION_POLICY_IDS:
            raise RuntimeError(
                f"direct fine-tuned selection is not registered for "
                f"{spec['policy_id']}")
        post = fine_tuned_final_path(output_dir, spec)
        if not post.is_file():
            raise FileNotFoundError(
                f"fine-tuned checkpoint missing: {spec['policy_id']}")
        validate_model_identity(spec, post)
        update_state(
            stage="webots_fine_tune_direct_selection",
            current={"policy_id": spec["policy_id"],
                     "scheduler": spec["scheduler"],
                     "objective": spec["objective"],
                     "selection": "fine_tuned_checkpoint"})
        selected = selected_final_path(output_dir, spec)
        atomic_copy_model(spec, post, selected)
        validate_model_identity(spec, selected)
        post_hash = sha256_file(post)
        selected_hash = sha256_file(selected)
        if selected_hash != post_hash:
            raise RuntimeError("direct fine-tuned checkpoint hash mismatch")
        decision = {
            "version": DIRECT_FINETUNED_SELECTION_VERSION,
            "status": "complete",
            "policy_id": spec["policy_id"],
            "scheduler": spec["scheduler"],
            "objective": spec["objective"],
            "decision": "accept_finetuned_without_nonregression",
            "accepted": True,
            "selection_mode": "operator_requested_direct_finetuned_model",
            "nonregression_skipped": True,
            "selection_reason": selection_reason,
            "ignored_timeout_gate": ignored_timeout_gate,
            "paired_run_count": 0,
            "regression_seeds": [],
            "degradation_tolerance": None,
            "degradation_tolerance_kind": None,
            "strict_improvement_required": False,
            "one_sided_bootstrap_95_lower_bound": None,
            "existing_nonregression_artifacts_preserved": True,
            "source_checkpoint": str(post.resolve()),
            "source_checkpoint_sha256": post_hash,
            "source_model_identity_sha256": sha256_file(
                model_identity_path(post)),
            "selected_checkpoint": str(selected.resolve()),
            "selected_checkpoint_sha256": selected_hash,
            "selected_model_identity_sha256": sha256_file(
                model_identity_path(selected)),
            "resume_granularity": "atomic direct checkpoint selection",
        }
        decision["report_sha256"] = _canonical_sha256(decision)
        atomic_json(
            output_dir / "webots_fine_tune" / spec["policy_id"] /
            "nonregression_decision.json", decision)
        print(
            f"[DIRECT EVALUATION] policy={spec['policy_id']} selected "
            "the Webots-fine-tuned checkpoint; non-regression skipped",
            flush=True)
    return True


def _valid_direct_finetuned_selection(
        output_dir: Path, spec: dict) -> bool:
    """Recognize an explicit, identity-bound direct-selection decision."""
    decision_path = (
        output_dir / "webots_fine_tune" / spec["policy_id"] /
        "nonregression_decision.json")
    decision = load_json(decision_path)
    post = fine_tuned_final_path(output_dir, spec)
    selected = selected_final_path(output_dir, spec)
    try:
        unsigned = dict(decision)
        report_hash = unsigned.pop("report_sha256", None)
        if not (
                decision.get("version") == DIRECT_FINETUNED_SELECTION_VERSION and
                decision.get("status") == "complete" and
                decision.get("policy_id") == spec["policy_id"] and
                decision.get("scheduler") == spec["scheduler"] and
                decision.get("objective") == spec["objective"] and
                decision.get("decision") ==
                    "accept_finetuned_without_nonregression" and
                decision.get("accepted") is True and
                decision.get("selection_mode") ==
                    "operator_requested_direct_finetuned_model" and
                decision.get("nonregression_skipped") is True and
                isinstance(decision.get("selection_reason"), str) and
                bool(decision["selection_reason"]) and
                isinstance(decision.get("ignored_timeout_gate", False), bool) and
                decision.get("paired_run_count") == 0 and
                decision.get("regression_seeds") == [] and
                decision.get("existing_nonregression_artifacts_preserved") is
                    True and
                report_hash == _canonical_sha256(unsigned) and
                post.is_file() and selected.is_file() and
                Path(decision["source_checkpoint"]).resolve() == post.resolve() and
                Path(decision["selected_checkpoint"]).resolve() ==
                    selected.resolve() and
                decision.get("source_checkpoint_sha256") ==
                    sha256_file(post) and
                decision.get("selected_checkpoint_sha256") ==
                    sha256_file(selected) and
                decision.get("source_checkpoint_sha256") ==
                    decision.get("selected_checkpoint_sha256") and
                decision.get("source_model_identity_sha256") ==
                    sha256_file(model_identity_path(post)) and
                decision.get("selected_model_identity_sha256") ==
                    sha256_file(model_identity_path(selected))):
            return False
        validate_model_identity(spec, post)
        validate_model_identity(spec, selected)
    except (KeyError, ModelValidationError, OSError, TypeError, ValueError):
        return False
    return True


def guard_all_fine_tuned_models(
        output_dir: Path, webots: Path, update_state,
        max_new_runs: int | None = None, *, specs=None,
        progress: dict | None = None) -> bool:
    """Run the paired gate unless an audited direct selection already exists."""
    requested_specs = tuple(
        PIPELINE_AI_SPECS if specs is None else specs)
    for spec in requested_specs:
        if _valid_direct_finetuned_selection(output_dir, spec):
            print(
                f"[RESUME] direct-finetuned-selection/{spec['policy_id']}",
                flush=True)
            continue
        if not _run_webots_nonregression_gate(
                output_dir, webots, update_state, max_new_runs,
                specs=(spec,), progress=progress):
            return False
    return True


def baseline_evaluation_policies() -> list[dict]:
    return [
        {
            "policy_id": scheduler, "scheduler": scheduler,
            "training_objective": None,
            "model_path": None,
            "model_sha256": None,
            "model_identity_sha256": None,
        }
        for scheduler in EVALUATED_NON_AI_SCHEDULERS
    ]


def trained_evaluation_policy(output_dir: Path, spec: dict) -> dict:
    model = selected_final_path(output_dir, spec)
    validate_model_identity(spec, model)
    return {
        "policy_id": spec["policy_id"],
        "scheduler": spec["scheduler"],
        "training_objective": spec["objective"],
        "model_path": model,
        "model_sha256": sha256_file(model),
        "model_identity_sha256": sha256_file(model_identity_path(model)),
    }


def evaluation_policies(output_dir: Path) -> list[dict]:
    policies = baseline_evaluation_policies()
    for spec in AI_SPECS:
        policies.append(trained_evaluation_policy(output_dir, spec))
    if (len(policies) != FINAL_EVALUATION_POLICY_COUNT or
            len({row["policy_id"] for row in policies}) != len(policies)):
        raise RuntimeError("final dual-evaluation policy roster is invalid")
    return policies


def _write_evaluation_aggregate(
        output_dir: Path, policies: list[dict], destination: Path) -> dict:
    aggregate = webots_workflow.aggregate_evaluation(
        output_dir / "evaluation", policies,
        validation_seeds=FINAL_EVALUATION_SEEDS)
    aggregate.update({
        "evaluation_contract": "count-and-utility-v2-from-every-run",
        "evaluation_metrics": ["count", "utility_v2"],
        "seeds_per_scenario": len(FINAL_EVALUATION_SEEDS),
        "common_random_numbers": True,
        "evaluation_seed_contract": final_evaluation_seed_contract(),
        "evaluation_seed_contract_sha256":
            final_evaluation_seed_contract_sha256(),
        "policy_identity_required": True,
    })
    atomic_json(destination, aggregate)
    return aggregate


def _evaluate_policies(
        output_dir: Path, webots: Path, update_state, policies: list[dict],
        *, stage: str, aggregate_path: Path,
        max_new_runs: int | None = None,
        progress: dict | None = None) -> bool:
    if (not policies or len({row["policy_id"] for row in policies}) !=
            len(policies)):
        raise ValueError("evaluation policies must be nonempty and unique")
    missing = [str(row["model_path"]) for row in policies
               if row["model_path"] is not None and
               not Path(row["model_path"]).is_file()]
    if missing:
        raise FileNotFoundError("evaluation models are missing:\n" +
                                "\n".join(missing))
    executed = int((progress or {}).get("evaluation", 0))
    for policy in policies:
        for scenario in SCENARIOS:
            for seed in FINAL_EVALUATION_SEEDS:
                canonical = (output_dir / "evaluation" / "episodes" /
                             policy["policy_id"] / scenario / str(seed) /
                             "result.json")
                model = policy["model_path"]
                model_hash = sha256_file(model) if model else None
                if model_hash != policy.get("model_sha256"):
                    raise RuntimeError(
                        "final evaluation checkpoint identity changed")
                cached = load_json(canonical)
                if webots_workflow.valid_episode(
                        cached, scenario=scenario, seed=seed,
                        manifest=manifest_path(
                            "final_evaluation", scenario, seed),
                        input_model_sha256=model_hash,
                        fine_tune_objective=None,
                        expected_scheduler=policy["scheduler"]):
                    ensure_evaluation_identity(
                        policy, canonical, scenario, seed)
                    print(
                        f"[RESUME] evaluation/{policy['policy_id']}/"
                        f"{scenario}/{seed}", flush=True)
                    continue
                if max_new_runs is not None and executed >= max_new_runs:
                    return False
                update_state(
                    stage=stage,
                    current={"policy_id": policy["policy_id"],
                             "training_objective":
                                 policy.get("training_objective"),
                             "scenario": scenario, "seed": seed})
                webots_workflow.run_or_resume_episode(
                    canonical_path=canonical, webots=webots,
                    scenario=scenario, seed=seed,
                    manifest=manifest_path(
                        "final_evaluation", scenario, seed),
                    scheduler=policy["scheduler"], model_path=model,
                    fine_tune_objective=None)
                ensure_evaluation_identity(
                    policy, canonical, scenario, seed)
                executed += 1
                if progress is not None:
                    progress["evaluation"] = executed
    _write_evaluation_aggregate(
        output_dir, policies, aggregate_path)
    return True


def evaluate_model(
        output_dir: Path, webots: Path, update_state, spec: dict,
        max_new_runs: int | None = None, *,
        progress: dict | None = None) -> bool:
    """Evaluate one selected model immediately after its physical guard."""
    policy = trained_evaluation_policy(output_dir, spec)
    return _evaluate_policies(
        output_dir, webots, update_state, [policy],
        stage="model_final_evaluation",
        aggregate_path=(output_dir / "evaluation" / "per_model" /
                        f"{spec['policy_id']}.json"),
        max_new_runs=max_new_runs, progress=progress)


def evaluate_baselines(
        output_dir: Path, webots: Path, update_state,
        max_new_runs: int | None = None, *,
        progress: dict | None = None) -> bool:
    return _evaluate_policies(
        output_dir, webots, update_state, baseline_evaluation_policies(),
        stage="baseline_final_evaluation",
        aggregate_path=(output_dir / "evaluation" /
                        "baselines_evaluation.json"),
        max_new_runs=max_new_runs, progress=progress)


def write_full_evaluation_aggregate(output_dir: Path) -> dict:
    return _write_evaluation_aggregate(
        output_dir, evaluation_policies(output_dir),
        output_dir / "evaluation" / "all_schedulers_evaluation.json")


def evaluate_all(output_dir: Path, webots: Path, update_state,
                 max_new_runs: int | None = None, *,
                 progress: dict | None = None) -> bool:
    return _evaluate_policies(
        output_dir, webots, update_state, evaluation_policies(output_dir),
        stage="evaluate_all",
        aggregate_path=(output_dir / "evaluation" /
                        "all_schedulers_evaluation.json"),
        max_new_runs=max_new_runs, progress=progress)


LEGACY_DIRECT_MODEL_PIPELINE_RECEIPT_VERSION = (
    "model-pipeline-receipt-v3-direct-finetuned")
MODEL_PIPELINE_RECEIPT_VERSION = (
    "model-pipeline-receipt-v4-two-seed-webots-regression")


def _model_pipeline_receipt_document(
        output_dir: Path, spec: dict, position: int) -> dict:
    policy_id = spec["policy_id"]
    artifacts = {
        "standalone_training_report": (
            output_dir / "headless" / policy_id / "training_report.json"),
        "webots_fine_tune_report": (
            output_dir / "webots_fine_tune" / policy_id /
            "fine_tune_report.json"),
        "webots_nonregression_decision": (
            output_dir / "webots_fine_tune" / policy_id /
            "nonregression_decision.json"),
        "per_model_final_evaluation": (
            output_dir / "evaluation" / "per_model" / f"{policy_id}.json"),
    }
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "model pipeline artifacts are missing:\n" + "\n".join(missing))
    selected = selected_final_path(output_dir, spec)
    validate_model_identity(spec, selected)
    selected_hash = sha256_file(selected)
    selected_identity_hash = sha256_file(model_identity_path(selected))
    evaluation = load_json(artifacts["per_model_final_evaluation"])
    raw_evaluation_rows = (
        evaluation.get("policies") if isinstance(evaluation, dict) else None)
    evaluation_rows = (
        raw_evaluation_rows if isinstance(raw_evaluation_rows, list) else [])
    if not (
            isinstance(evaluation, dict) and
            evaluation.get("status") == "complete" and
            evaluation.get("policy_count") == 1 and
            evaluation.get("run_count") ==
                len(SCENARIOS) * len(FINAL_EVALUATION_SEEDS) and
            evaluation.get("evaluation_metrics") == ["count", "utility_v2"] and
            evaluation.get("validation_seeds") ==
                list(FINAL_EVALUATION_SEEDS) and
            evaluation.get("common_random_numbers") is True and
            evaluation.get("evaluation_seed_contract_sha256") ==
                final_evaluation_seed_contract_sha256() and
            len(evaluation_rows) == 1 and
            isinstance(evaluation_rows[0], dict) and
            evaluation_rows[0].get("policy_id") == policy_id and
            evaluation_rows[0].get("scheduler") == spec["scheduler"] and
            evaluation_rows[0].get("training_objective") ==
                spec["objective"] and
            evaluation_rows[0].get("model_sha256") == selected_hash and
            evaluation_rows[0].get("model_identity_sha256") ==
                selected_identity_hash):
        raise RuntimeError("per-model final evaluation is invalid")
    document = {
        "version": MODEL_PIPELINE_RECEIPT_VERSION,
        "pipeline_position": position,
        "pipeline_size": len(PIPELINE_AI_SPECS),
        "policy_id": policy_id,
        "scheduler": spec["scheduler"],
        "objective": spec["objective"],
        "stage_order": [
            "automatic_hyperparameter_optimization",
            "standalone_training_until_convergence",
            "webots_fine_tuning_until_convergence_or_maximum",
            "two_seed_webots_bounded_regression_selection",
            "final_dual_metric_evaluation",
        ],
        "selected_checkpoint": str(selected.resolve()),
        "selected_checkpoint_sha256": selected_hash,
        "selected_model_identity_sha256": selected_identity_hash,
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
    }
    document["report_sha256"] = _canonical_sha256(document)
    return document


def _write_model_pipeline_receipt(
        output_dir: Path, spec: dict, position: int) -> dict:
    document = _model_pipeline_receipt_document(output_dir, spec, position)
    atomic_json(
        output_dir / "model_pipeline" / spec["policy_id"] / "complete.json",
        document)
    return document


def _valid_model_pipeline_receipt(
        output_dir: Path, spec: dict, position: int) -> bool:
    path = (output_dir / "model_pipeline" / spec["policy_id"] /
            "complete.json")
    existing = load_json(path)
    if not isinstance(existing, dict):
        return False
    try:
        expected = _model_pipeline_receipt_document(
            output_dir, spec, position)
        if existing == expected:
            return True
        # Pipelines completed before the two-seed gate was requested retain
        # their immutable direct-selection receipt and remain resumable.
        legacy = dict(expected)
        legacy["version"] = LEGACY_DIRECT_MODEL_PIPELINE_RECEIPT_VERSION
        legacy["stage_order"] = list(expected["stage_order"])
        legacy["stage_order"][3] = "direct_finetuned_checkpoint_selection"
        legacy.pop("report_sha256", None)
        legacy["report_sha256"] = _canonical_sha256(legacy)
        return existing == legacy
    except (FileNotFoundError, ModelValidationError, RuntimeError,
            KeyError, OSError, TypeError, ValueError):
        return False


def _validate_parallel_headless_pair(specs) -> tuple[dict, ...]:
    pair = tuple(specs)
    if not 1 <= len(pair) <= PARALLEL_HEADLESS_MODEL_WORKERS:
        raise ValueError("parallel headless pair size is invalid")
    identities = set()
    for spec in pair:
        identity = (
            spec.get("policy_id"), spec.get("scheduler"),
            spec.get("objective"))
        space = SEARCH_SPACES.get(spec.get("policy_id"))
        if (space is None or spec.get("scheduler") != space.scheduler or
                spec.get("objective") != space.objective or
                identity in identities):
            raise ValueError("parallel headless model identity is invalid")
        identities.add(identity)
    if len(pair) == 2 and not (
            pair[0]["scheduler"] == pair[1]["scheduler"] and
            {pair[0]["objective"], pair[1]["objective"]} ==
                {"count", "utility_v2"}):
        raise ValueError(
            "parallel lanes must be the count/utility pair of one scheduler")
    return pair


def _parallel_positioned_groups(positioned_specs):
    """Pair only count/utility siblings, including mid-pair resumes."""
    rows = list(positioned_specs)
    groups = []
    index = 0
    while index < len(rows):
        group = [rows[index]]
        if index + 1 < len(rows):
            left, right = rows[index][1], rows[index+1][1]
            if (left["scheduler"] == right["scheduler"] and
                    {left["objective"], right["objective"]} ==
                    {"count", "utility_v2"}):
                group.append(rows[index+1])
        groups.append(group)
        index += len(group)
    return groups


def _wait_for_parallel_headless_pair(
        futures: dict, output_dir: Path, pair, position: int,
        update_state) -> dict[str, dict]:
    """Audit both lanes to completion, preserving useful sibling progress."""
    pending = set(futures)
    results = {}
    failures = []
    while pending:
        finished, pending = wait(
            pending, timeout=15.0, return_when=FIRST_COMPLETED)
        for future in finished:
            spec = futures[future]
            policy_id = spec["policy_id"]
            try:
                result = future.result()
                # The child binds the tuning winner in its own process.
                # Rebind and independently validate every returned artifact
                # in the sole shared-state writer before accepting the lane.
                _apply_tuning_selection(output_dir, spec, required=True)
                _validate_headless_lane_result(output_dir, spec, result)
                results[policy_id] = result
            except Exception as exc:
                failures.append((policy_id, exc))
        pending_policy_ids = {
            futures[future]["policy_id"] for future in pending}
        pending_ids = [
            spec["policy_id"] for spec in pair
            if spec["policy_id"] in pending_policy_ids]
        update_state(
            stage="parallel_model_headless",
            active_policy_id=(pending_ids[0] if pending_ids else None),
            active_policy_ids=pending_ids,
            pipeline_position=position,
            current={
                "mode": "two_isolated_headless_lanes",
                "lanes": _headless_lane_snapshot(output_dir, pair),
                "completed_headless_policy_ids": [
                    spec["policy_id"] for spec in pair
                    if spec["policy_id"] in results],
                "failed_headless_policy_ids": [
                    spec["policy_id"] for spec in pair
                    if spec["policy_id"] in {
                        policy_id for policy_id, _error in failures}],
                "physical_stages_serial": True,
            })
    if failures:
        policy_id, error = failures[0]
        raise RuntimeError(
            f"parallel headless lane failed: {policy_id}") from error
    if set(results) != {spec["policy_id"] for spec in pair}:
        raise RuntimeError("parallel headless pair completion is incomplete")
    return results


def _run_serial_physical_pipeline(
        output_dir: Path, webots: Path, update_state, spec: dict,
        position: int, pair, progress: dict, *,
        max_fine_tune_episodes: int | None,
        max_regression_runs: int | None,
        max_evaluation_runs: int | None) -> bool:
    policy_id = spec["policy_id"]
    update_state(
        stage="model_pipeline_physical_serial",
        active_policy_id=policy_id,
        active_policy_ids=[row["policy_id"] for row in pair],
        pipeline_position=position,
        current={
            "policy_id": policy_id,
            "scheduler": spec["scheduler"],
            "objective": spec["objective"],
            "step": "webots_fine_tune_then_two_seed_guard_then_evaluate",
            "physical_stages_serial": True,
        })
    if not fine_tune_all(
            output_dir, webots, update_state, max_fine_tune_episodes,
            specs=(spec,), progress=progress):
        return False
    if not guard_all_fine_tuned_models(
            output_dir, webots, update_state, max_regression_runs,
            specs=(spec,), progress=progress):
        return False
    if not evaluate_model(
            output_dir, webots, update_state, spec,
            max_evaluation_runs, progress=progress):
        return False
    _write_model_pipeline_receipt(output_dir, spec, position)
    return True


def _has_completed_fine_tune_for_resume(
        output_dir: Path, spec: dict) -> bool:
    """Recognize a later-stage checkpoint before resubmitting headless work."""
    policy_dir = output_dir / "webots_fine_tune" / spec["policy_id"]
    report = load_json(policy_dir / "fine_tune_report.json")
    checkpoint = fine_tuned_final_path(output_dir, spec)
    try:
        if not (
                isinstance(report, dict) and
                report.get("status") == "complete" and
                report.get("policy_id") == spec["policy_id"] and
                report.get("scheduler") == spec["scheduler"] and
                report.get("objective") == spec["objective"] and
                Path(report.get("output_checkpoint", "")).resolve() ==
                    checkpoint.resolve() and
                checkpoint.is_file() and
                report.get("output_checkpoint_sha256") ==
                    sha256_file(checkpoint) and
                report.get("output_model_identity_sha256") == sha256_file(
                    model_identity_path(checkpoint))):
            return False
        validate_model_identity(spec, checkpoint)
    except (KeyError, ModelValidationError, OSError, TypeError, ValueError):
        return False
    return True


def _has_committed_fine_tune_for_tolerance_migration(
        output_dir: Path, spec: dict) -> bool:
    """Recognize a completed fine-tune without requiring a rebound config.

    At process startup the base policy specs do not yet contain their selected
    hyperparameters.  The normal resume validator intentionally rejects that
    mismatch.  Protocol migration therefore validates the immutable report,
    checkpoint hash, and identity document directly before freezing the old
    five-percent tolerance.
    """
    policy_dir = Path(output_dir) / "webots_fine_tune" / spec["policy_id"]
    report = load_json(policy_dir / "fine_tune_report.json")
    checkpoint = fine_tuned_final_path(Path(output_dir), spec)
    identity_path = model_identity_path(checkpoint)
    identity = load_json(identity_path)
    try:
        checkpoint_hash = sha256_file(checkpoint)
        return bool(
            isinstance(report, dict) and
            report.get("status") == "complete" and
            report.get("policy_id") == spec["policy_id"] and
            report.get("scheduler") == spec["scheduler"] and
            report.get("objective") == spec["objective"] and
            Path(report.get("output_checkpoint", "")).resolve() ==
                checkpoint.resolve() and
            report.get("output_checkpoint_sha256") == checkpoint_hash and
            report.get("output_model_identity_sha256") ==
                sha256_file(identity_path) and
            isinstance(identity, dict) and
            identity.get("policy_id") == spec["policy_id"] and
            identity.get("scheduler") == spec["scheduler"] and
            identity.get("objective") == spec["objective"] and
            identity.get("checkpoint_sha256") == checkpoint_hash)
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _registered_webots_regression_tolerance(
        output_dir: Path, spec: dict) -> float:
    """Load the immutable, identity-bound Webots budget for one policy."""
    state = load_json(Path(output_dir) / "workflow_state.json")
    if not isinstance(state, dict):
        raise RuntimeError("workflow state is unavailable for Webots guard")
    tolerances = state.get(
        "webots_finetune_regression_tolerance_by_policy")
    legacy_policy_ids = state.get(
        "webots_finetune_legacy_tolerance_policy_ids")
    if (not isinstance(legacy_policy_ids, list) or
            legacy_policy_ids != _ordered_policy_id_subset(
                legacy_policy_ids)):
        raise RuntimeError("Webots legacy tolerance registry is invalid")
    expected_tolerances = {
        policy_id: (
            PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE
            if policy_id in legacy_policy_ids else
            COUNT_RELATIVE_DEGRADATION_TOLERANCE)
        for policy_id in PIPELINE_POLICY_IDS}
    if tolerances != expected_tolerances:
        raise RuntimeError("Webots per-policy tolerance registry is invalid")
    tolerance = tolerances.get(spec.get("policy_id"))
    allowed = {
        PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE,
        COUNT_RELATIVE_DEGRADATION_TOLERANCE,
        UTILITY_RELATIVE_DEGRADATION_TOLERANCE,
    }
    if (isinstance(tolerance, bool) or
            not isinstance(tolerance, (int, float)) or
            not math.isfinite(float(tolerance)) or
            float(tolerance) not in allowed):
        raise RuntimeError("registered Webots degradation tolerance is invalid")
    return float(tolerance)


def run_model_by_model_pipeline(
        output_dir: Path, webots: Path, update_state, *,
        completed_policy_ids=(), max_standalone_episodes: int | None = None,
        max_fine_tune_episodes: int | None = None,
        max_regression_runs: int | None = None,
        max_evaluation_runs: int | None = None,
        stop_after_policy_id: str | None = None) -> bool:
    """Train two isolated objective lanes; serialize all physical stages."""
    if (stop_after_policy_id is not None and
            stop_after_policy_id not in PIPELINE_POLICY_IDS):
        raise ValueError("stop-after policy identity is invalid")
    completed = list(completed_policy_ids)
    if completed != list(PIPELINE_POLICY_IDS[:len(completed)]):
        raise RuntimeError("completed model pipelines are not an order prefix")
    progress: dict[str, int] = {}

    for position, spec in enumerate(
            PIPELINE_AI_SPECS[:len(completed)], start=1):
        policy_id = spec["policy_id"]
        # A fresh process starts with an unbound base spec. Reload the winner
        # before validating the completed pipeline receipt.
        _apply_tuning_selection(output_dir, spec, required=True)
        if not _valid_model_pipeline_receipt(output_dir, spec, position):
            raise RuntimeError(
                f"completed model pipeline receipt is invalid: {policy_id}")
        print(f"[RESUME] model-pipeline/{policy_id} complete", flush=True)

    # A restart can occur after Webots fine-tuning but before the per-model
    # receipt is committed. That later-stage artifact is stronger resume
    # evidence than a stale headless lane state, so finish its bounded guard
    # and final evaluation before scheduling any headless workers again.
    while len(completed) < len(PIPELINE_AI_SPECS):
        position = len(completed) + 1
        spec = PIPELINE_AI_SPECS[len(completed)]
        if not (
                (output_dir / "webots_fine_tune" / spec["policy_id"] /
                 "fine_tune_report.json").is_file() and
                fine_tuned_final_path(output_dir, spec).is_file()):
            break
        _apply_tuning_selection(output_dir, spec, required=True)
        if not _has_completed_fine_tune_for_resume(output_dir, spec):
            raise RuntimeError(
                f"committed fine-tune artifact is invalid: "
                f"{spec['policy_id']}")
        print(
            f"[RESUME] webots-fine/{spec['policy_id']} complete; "
            "entering two-seed guard and final evaluation",
            flush=True)
        if not _run_serial_physical_pipeline(
                output_dir, webots, update_state, spec, position, (spec,),
                progress,
                max_fine_tune_episodes=max_fine_tune_episodes,
                max_regression_runs=max_regression_runs,
                max_evaluation_runs=max_evaluation_runs):
            return False
        completed.append(spec["policy_id"])
        update_state(
            stage="model_pipeline_complete", active_policy_id=None,
            active_policy_ids=[], pipeline_position=position,
            completed_model_pipelines=list(completed),
            current={"policy_id": spec["policy_id"],
                     "scheduler": spec["scheduler"],
                     "objective": spec["objective"],
                     "result": "complete"})
        print(
            f"[MODEL COMPLETE] {position}/{len(PIPELINE_AI_SPECS)} "
            f"policy={spec['policy_id']}", flush=True)
        if spec["policy_id"] == stop_after_policy_id:
            return False

    remaining = list(enumerate(
        PIPELINE_AI_SPECS[len(completed):], start=len(completed)+1))
    parallel_enabled = bool(
        PARALLEL_HEADLESS_MODEL_WORKERS >= 2 and
        max_standalone_episodes is None and
        max_fine_tune_episodes is None and
        max_regression_runs is None and
        max_evaluation_runs is None)

    if not parallel_enabled:
        for position, spec in remaining:
            policy_id = spec["policy_id"]
            update_state(
                stage="model_pipeline", active_policy_id=policy_id,
                active_policy_ids=[policy_id], pipeline_position=position,
                current={"policy_id": policy_id,
                         "scheduler": spec["scheduler"],
                         "objective": spec["objective"],
                         "step": "automatic_hyperparameter_optimization"})
            if not train_headless(
                    output_dir, update_state, max_standalone_episodes,
                    specs=(spec,), progress=progress):
                return False
            if not _run_serial_physical_pipeline(
                    output_dir, webots, update_state, spec, position, (spec,),
                    progress,
                    max_fine_tune_episodes=max_fine_tune_episodes,
                    max_regression_runs=max_regression_runs,
                    max_evaluation_runs=max_evaluation_runs):
                return False
            completed.append(policy_id)
            update_state(
                stage="model_pipeline_complete", active_policy_id=None,
                active_policy_ids=[], pipeline_position=position,
                completed_model_pipelines=list(completed),
                current={"policy_id": policy_id,
                         "scheduler": spec["scheduler"],
                         "objective": spec["objective"],
                         "result": "complete"})
            print(
                f"[MODEL COMPLETE] {position}/{len(PIPELINE_AI_SPECS)} "
                f"policy={policy_id}", flush=True)
            if policy_id == stop_after_policy_id:
                return False
    else:
        for positioned_pair in _parallel_positioned_groups(remaining):
            pair = _validate_parallel_headless_pair(
                spec for _position, spec in positioned_pair)
            update_state(
                stage="parallel_model_headless",
                active_policy_id=pair[0]["policy_id"],
                active_policy_ids=[spec["policy_id"] for spec in pair],
                pipeline_position=positioned_pair[0][0],
                current={
                    "mode": "two_isolated_headless_lanes",
                    "lanes": _headless_lane_snapshot(output_dir, pair),
                    "physical_stages_serial": True,
                    "maximum_headless_processes":
                        PARALLEL_HEADLESS_PROCESS_BUDGET,
                })
            with ProcessPoolExecutor(max_workers=len(pair)) as executor:
                futures = {
                    executor.submit(
                        _headless_model_lane_worker, output_dir, dict(spec)):
                        spec
                    for spec in pair
                }
                _wait_for_parallel_headless_pair(
                    futures, output_dir, pair, positioned_pair[0][0],
                    update_state)
            # Both headless workers and their nested validation workers have
            # exited before physical simulation starts. This prevents CPU or
            # memory contention from biasing Webots timing and evaluation.
            for position, spec in positioned_pair:
                policy_id = spec["policy_id"]
                if not _run_serial_physical_pipeline(
                        output_dir, webots, update_state, spec, position,
                        pair, progress,
                        max_fine_tune_episodes=None,
                        max_regression_runs=None,
                        max_evaluation_runs=None):
                    return False
                completed.append(policy_id)
                remaining_lane_ids = [
                    row["policy_id"] for row in pair
                    if row["policy_id"] not in completed]
                update_state(
                    stage="model_pipeline_complete",
                    active_policy_id=(
                        remaining_lane_ids[0]
                        if remaining_lane_ids else None),
                    active_policy_ids=remaining_lane_ids,
                    pipeline_position=position,
                    completed_model_pipelines=list(completed),
                    current={"policy_id": policy_id,
                             "scheduler": spec["scheduler"],
                             "objective": spec["objective"],
                             "result": "complete"})
                print(
                    f"[MODEL COMPLETE] {position}/"
                    f"{len(PIPELINE_AI_SPECS)} policy={policy_id}",
                    flush=True)
                if policy_id == stop_after_policy_id:
                    return False

    if not evaluate_baselines(
            output_dir, webots, update_state, max_evaluation_runs,
            progress=progress):
        return False
    write_full_evaluation_aggregate(output_dir)
    update_state(
        stage="full_evaluation_complete", active_policy_id=None,
        active_policy_ids=[],
        pipeline_position=len(PIPELINE_AI_SPECS),
        completed_model_pipelines=list(completed), current=None)
    return True


def validate_inputs(phase: str, output_dir: Path) -> None:
    ensure_formal_manifests()
    _load_calibration()
    required = [CALIBRATION]
    required.extend(
        manifest_path(partition, scenario, seed)
        for partition, seeds in (
            ("standalone_tuning_training", TUNING_TRAINING_SEEDS),
            ("standalone_tuning_validation", TUNING_VALIDATION_SEEDS),
            ("standalone_training", TRAINING_SEEDS),
            ("standalone_validation", VALIDATION_SEEDS),
            ("webots_finetune", WEBOTS_FINETUNE_SEEDS),
            ("final_evaluation", FINAL_EVALUATION_SEEDS))
        for scenario in SCENARIOS for seed in seeds)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required workflow inputs are missing:\n" +
                                "\n".join(missing))


def parse_args():
    parser = argparse.ArgumentParser(description=(
        "Resume-safe sufficient standalone 1800s training, Webots fine "
        "tuning, and all-scheduler Webots evaluation."))
    parser.add_argument(
        "--output-dir", type=Path,
        default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--report-path", type=Path, default=DEFAULT_REPORT_PATH,
        help="write the detailed final report here after full evaluation")
    parser.add_argument(
        "--webots", type=Path,
        default=Path(r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"))
    parser.add_argument(
        "--phase", choices=(
            "all", "standalone", "headless", "fine-tune", "evaluate",
            "baselines"),
        default="all")
    parser.add_argument(
        "--max-standalone-episodes", "--max-headless-episodes",
        dest="max_standalone_episodes", type=int,
        help=("limit new standalone training episodes; validation episodes "
              "remain resume-safe and are not counted"))
    parser.add_argument("--max-fine-tune-episodes", type=int)
    parser.add_argument("--max-regression-runs", type=int)
    parser.add_argument("--max-evaluation-runs", type=int)
    parser.add_argument(
        "--standalone-exit-to-webots", action="append", default=[],
        choices=PIPELINE_POLICY_IDS,
        help=("persist an identity-bound user request to stop further "
              "standalone training and use the best complete validated "
              "checkpoint for Webots fine-tuning"))
    parser.add_argument(
        "--stop-after-policy", choices=PIPELINE_POLICY_IDS,
        help=("stop cleanly after the selected model completes its full "
              "training, Webots guard, and final-evaluation pipeline"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_standalone_exit_ids = _ordered_policy_id_subset(
        getattr(args, "standalone_exit_to_webots", ()))
    for value in (args.max_standalone_episodes,
                  args.max_fine_tune_episodes,
                  args.max_regression_runs,
                  args.max_evaluation_runs):
        if value is not None and value < 0:
            raise ValueError("run limits must be nonnegative")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_inputs(args.phase, output_dir)
    if args.phase in {"all", "fine-tune", "evaluate", "baselines"}:
        webots_workflow.validate_webots(args.webots)
    state_path = output_dir / "workflow_state.json"
    loaded_state = load_json(state_path)
    if state_path.exists() and not isinstance(loaded_state, dict):
        raise RuntimeError(
            "workflow state exists but is unreadable; refusing unsafe resume")
    if loaded_state is None:
        allowed_without_state = {
            "workflow.stdout.log", "workflow.stderr.log",
            "monitor_history.jsonl", "monitor_state.json", "monitor.lock",
            "monitor.stdout.log", "monitor.stderr.log",
        }
        unexpected = sorted(
            entry.name for entry in output_dir.iterdir()
            if (entry.name not in allowed_without_state and
                not entry.name.startswith("workflow.supervised.")))
        if unexpected:
            raise RuntimeError(
                "output directory contains artifacts but has no workflow "
                "state; refusing unsafe reuse: " + ", ".join(unexpected))
    if (isinstance(loaded_state, dict) and
            loaded_state.get("schema_version") ==
                PREVIOUS_WORKFLOW_STATE_SCHEMA_VERSION and
            loaded_state.get("workflow_contract") ==
                PREVIOUS_WORKFLOW_CONTRACT):
        selections = [
            _tuning_selection_path(output_dir, spec)
            for spec in PIPELINE_AI_SPECS
            if _tuning_selection_path(output_dir, spec).exists()
        ]
        group_audits = list(
            (output_dir / "headless").glob(
                "*/tuning/trial_*/group_audit.json"))
        if (loaded_state.get("completed_model_pipelines") != [] or
                selections or group_audits):
            raise RuntimeError(
                "v13 tuning-validation migration is only safe before the "
                "first hyperparameter selection has been committed")
        migration_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        resume_contract = dict(loaded_state.get("resume_contract", {}))
        resume_contract["standalone_hyperparameter_tuning"] = (
            "automatic selection from up to twelve deterministic isolated "
            "trials per objective instance (exhausting smaller grids); "
            "training uses five seeds per scenario and tuning validation "
            "uses two fixed independent seeds per scenario; every episode "
            "is 1800s; supervised winners must pass same-seed "
            "NearestNeighbour and objective-expert floors")
        migrations = list(loaded_state.get("protocol_migrations", []))
        migrations.append({
            "version": "efficient-tuning-validation-v1",
            "migrated_at": migration_time,
            "from_workflow_contract": PREVIOUS_WORKFLOW_CONTRACT,
            "to_workflow_contract": V14_WORKFLOW_CONTRACT,
            "preserved_committed_training_and_validation_results": True,
            "selected_validation_seeds": list(TUNING_VALIDATION_SEEDS),
            "preexisting_extra_validation_seeds_excluded_from_selection":
                [12002, 12003, 12004],
        })
        loaded_state.update({
            "schema_version": V14_WORKFLOW_STATE_SCHEMA_VERSION,
            "workflow_contract": V14_WORKFLOW_CONTRACT,
            "tuning_validation_seeds_per_scenario":
                len(TUNING_VALIDATION_SEEDS),
            "tuning_validation_seed_contract":
                list(TUNING_VALIDATION_SEEDS),
            "resume_contract": resume_contract,
            "protocol_migrations": migrations,
            "status": "interrupted",
            "stage": "protocol_migration",
            "current": None,
            "interrupted_at": migration_time,
            "updated_at": migration_time,
        })
        atomic_json(state_path, loaded_state)
    if (isinstance(loaded_state, dict) and
            loaded_state.get("schema_version") ==
                V14_WORKFLOW_STATE_SCHEMA_VERSION and
            loaded_state.get("workflow_contract") ==
                V14_WORKFLOW_CONTRACT):
        webots_root = output_dir / "webots_fine_tune"
        webots_artifacts = (
            [path for path in webots_root.glob("**/*") if path.is_file()]
            if webots_root.exists() else [])
        if loaded_state.get("completed_model_pipelines") != [] or webots_artifacts:
            raise RuntimeError(
                "v14 Webots-convergence migration is only safe before the "
                "first physical fine-tune artifact has been committed")
        migration_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        resume_contract = dict(loaded_state.get("resume_contract", {}))
        resume_contract["webots_model"] = (
            "checkpoint after every balanced A/B/C update; evaluate "
            "convergence after every round, stop after three consecutive "
            "stable parameter updates, and enforce a 21-round maximum")
        migrations = list(loaded_state.get("protocol_migrations", []))
        migrations.append({
            "version": "webots-convergence-six-seed-evaluation-v1",
            "migrated_at": migration_time,
            "from_workflow_contract": V14_WORKFLOW_CONTRACT,
            "to_workflow_contract": V15_WORKFLOW_CONTRACT,
            "preserved_committed_headless_results": True,
            "minimum_balanced_rounds":
                MIN_WEBOTS_FINETUNE_BALANCED_ROUNDS,
            "convergence_patience_rounds":
                WEBOTS_FINETUNE_CONVERGENCE_PATIENCE,
            "maximum_balanced_rounds":
                V15_MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS,
            "final_evaluation_seeds": list(FINAL_EVALUATION_SEEDS),
            "final_evaluation_common_random_numbers": True,
        })
        loaded_state.update({
            "schema_version": V15_WORKFLOW_STATE_SCHEMA_VERSION,
            "workflow_contract": V15_WORKFLOW_CONTRACT,
            "webots_finetune_seeds_per_scenario_minimum":
                MIN_WEBOTS_FINETUNE_BALANCED_ROUNDS,
            "webots_finetune_convergence_patience_rounds":
                WEBOTS_FINETUNE_CONVERGENCE_PATIENCE,
            "webots_finetune_seeds_per_scenario_maximum":
                V15_MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS,
            "final_evaluation_seeds_per_scenario":
                len(FINAL_EVALUATION_SEEDS),
            "final_evaluation_seed_contract_sha256":
                final_evaluation_seed_contract_sha256(),
            "resume_contract": resume_contract,
            "protocol_migrations": migrations,
            "status": "interrupted",
            "stage": "protocol_migration",
            "current": None,
            "interrupted_at": migration_time,
            "updated_at": migration_time,
        })
        atomic_json(state_path, loaded_state)
    if (isinstance(loaded_state, dict) and
            loaded_state.get("schema_version") ==
                V15_WORKFLOW_STATE_SCHEMA_VERSION and
            loaded_state.get("workflow_contract") ==
                V15_WORKFLOW_CONTRACT):
        migration_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        resume_contract = dict(loaded_state.get("resume_contract", {}))
        resume_contract["standalone_hyperparameter_tuning"] = (
            "automatic deterministic selection from isolated trials; up to "
            "two trial groups execute concurrently, each model-update chain "
            "remains serial, and only the parent writes workflow state")
        resume_contract["frozen_checkpoint_validation"] = (
            "up to three independent scenario/seed episodes execute in "
            "isolated processes; the checkpoint is read-only and the parent "
            "commits results in deterministic scenario/seed order")
        migrations = list(loaded_state.get("protocol_migrations", []))
        migrations.append({
            "version": "parallel-headless-execution-v1",
            "migrated_at": migration_time,
            "from_workflow_contract": V15_WORKFLOW_CONTRACT,
            "to_workflow_contract": V16_WORKFLOW_CONTRACT,
            "preserved_all_atomic_results": True,
            "hyperparameter_trial_workers":
                PARALLEL_HYPERPARAMETER_TRIAL_WORKERS,
            "frozen_validation_workers":
                PARALLEL_FROZEN_VALIDATION_WORKERS,
            "parent_only_workflow_state_writer": True,
        })
        loaded_state.update({
            "schema_version": V16_WORKFLOW_STATE_SCHEMA_VERSION,
            "workflow_contract": V16_WORKFLOW_CONTRACT,
            "parallel_headless_execution_contract":
                V16_PARALLEL_HEADLESS_EXECUTION_CONTRACT,
            "parallel_hyperparameter_trial_workers":
                PARALLEL_HYPERPARAMETER_TRIAL_WORKERS,
            "parallel_frozen_validation_workers":
                PARALLEL_FROZEN_VALIDATION_WORKERS,
            "resume_contract": resume_contract,
            "protocol_migrations": migrations,
            "status": "interrupted",
            "stage": "protocol_migration",
            "current": None,
            "interrupted_at": migration_time,
            "updated_at": migration_time,
        })
        atomic_json(state_path, loaded_state)
    if (isinstance(loaded_state, dict) and
            loaded_state.get("schema_version") ==
                V16_WORKFLOW_STATE_SCHEMA_VERSION and
            loaded_state.get("workflow_contract") ==
                V16_WORKFLOW_CONTRACT):
        webots_root = output_dir / "webots_fine_tune"
        evaluation_root = output_dir / "evaluation"
        later_stage_artifacts = any(
            path.is_file()
            for root in (webots_root, evaluation_root)
            if root.exists()
            for path in root.glob("**/*"))
        if (loaded_state.get("completed_model_pipelines") != [] or
                later_stage_artifacts):
            raise RuntimeError(
                "v16 five-percent regression migration is only safe before "
                "the first Webots or final-evaluation artifact")
        migration_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        resume_contract = dict(loaded_state.get("resume_contract", {}))
        resume_contract["bounded_regression"] = (
            "accept aggregate and per-scenario metric degradation up to five "
            "percent relative to the immutable pre-training baseline; reject "
            "and roll back any larger degradation")
        migrations = list(loaded_state.get("protocol_migrations", []))
        migrations.append({
            "version": "five-percent-relative-regression-v1",
            "migrated_at": migration_time,
            "from_workflow_contract": V16_WORKFLOW_CONTRACT,
            "to_workflow_contract": V17_WORKFLOW_CONTRACT,
            "previous_count_relative_tolerance": 0.02,
            "previous_utility_absolute_tolerance": 2.0,
            "relative_tolerance":
                STANDALONE_COUNT_RELATIVE_DEGRADATION_TOLERANCE,
            "applies_to_objectives": ["count", "utility_v2"],
            "preserved_standalone_atomic_results": True,
            "recompute_standalone_promotion_decisions": True,
            "revalidate_downstream_episode_provenance": True,
        })
        loaded_state.update({
            "schema_version": V17_WORKFLOW_STATE_SCHEMA_VERSION,
            "workflow_contract": V17_WORKFLOW_CONTRACT,
            "bounded_regression_guard": {
                "count_relative_tolerance":
                    STANDALONE_COUNT_RELATIVE_DEGRADATION_TOLERANCE,
                "utility_relative_tolerance":
                    STANDALONE_UTILITY_RELATIVE_DEGRADATION_TOLERANCE,
                "immutable_training_baseline": True,
            },
            "resume_contract": resume_contract,
            "protocol_migrations": migrations,
            "status": "interrupted",
            "stage": "protocol_migration",
            "current": None,
            "interrupted_at": migration_time,
            "updated_at": migration_time,
        })
        atomic_json(state_path, loaded_state)
    if (isinstance(loaded_state, dict) and
            loaded_state.get("schema_version") ==
                V17_WORKFLOW_STATE_SCHEMA_VERSION and
            loaded_state.get("workflow_contract") ==
                V17_WORKFLOW_CONTRACT):
        migration_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        resume_contract = dict(loaded_state.get("resume_contract", {}))
        resume_contract["model_pipeline"] = (
            "run the count and utility objective models of one scheduler in "
            "two isolated headless processes; each process owns only its "
            "policy directory and lane state; the parent alone writes shared "
            "workflow state and runs Webots fine-tuning, non-regression, and "
            "final evaluation serially in registered policy order")
        migrations = list(loaded_state.get("protocol_migrations", []))
        migrations.append({
            "version": "two-isolated-headless-model-lanes-v1",
            "migrated_at": migration_time,
            "from_workflow_contract": V17_WORKFLOW_CONTRACT,
            "to_workflow_contract": V18_WORKFLOW_CONTRACT,
            "headless_model_workers": PARALLEL_HEADLESS_MODEL_WORKERS,
            "maximum_nested_headless_processes":
                PARALLEL_HEADLESS_PROCESS_BUDGET,
            "artifact_isolation": "headless/<policy_id>",
            "parent_only_shared_state_writer": True,
            "model_identity_revalidated_by_parent": True,
            "webots_and_final_evaluation_serial": True,
            "preserved_all_committed_artifacts": True,
        })
        loaded_state.update({
            "schema_version": V18_WORKFLOW_STATE_SCHEMA_VERSION,
            "workflow_contract": V18_WORKFLOW_CONTRACT,
            "pipeline_execution_contract":
                "two-headless-lanes-serial-physical-v1",
            "parallel_headless_execution_contract":
                PARALLEL_HEADLESS_EXECUTION_CONTRACT,
            "parallel_headless_model_workers":
                PARALLEL_HEADLESS_MODEL_WORKERS,
            "parallel_headless_process_budget":
                PARALLEL_HEADLESS_PROCESS_BUDGET,
            "active_policy_ids": [],
            "resume_contract": resume_contract,
            "protocol_migrations": migrations,
            "status": "interrupted",
            "stage": "protocol_migration",
            "current": None,
            "interrupted_at": migration_time,
            "updated_at": migration_time,
        })
        atomic_json(state_path, loaded_state)
    if (isinstance(loaded_state, dict) and
            loaded_state.get("schema_version") ==
                V18_WORKFLOW_STATE_SCHEMA_VERSION and
            loaded_state.get("workflow_contract") ==
                V18_WORKFLOW_CONTRACT):
        webots_root = output_dir / "webots_fine_tune"
        evaluation_root = output_dir / "evaluation"
        later_stage_artifacts = any(
            path.is_file()
            for root in (webots_root, evaluation_root)
            if root.exists()
            for path in root.glob("**/*"))
        if (loaded_state.get("completed_model_pipelines") != [] or
                later_stage_artifacts):
            raise RuntimeError(
                "v18 ten-round Webots migration is only safe before the "
                "first Webots or final-evaluation artifact")
        if (loaded_state.get(
                "webots_finetune_seeds_per_scenario_maximum") !=
                V15_MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS):
            raise RuntimeError("v18 Webots maximum is not the expected 21")
        migration_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        resume_contract = dict(loaded_state.get("resume_contract", {}))
        resume_contract["standalone_sufficient_training"] = (
            "checkpoint after every complete 1800s episode and normally "
            "continue balanced A/B/C rounds until convergence; an explicit "
            "identity-bound user transition may stop further standalone "
            "training and select only a complete guarded validation "
            "checkpoint for Webots")
        resume_contract["webots_model"] = (
            "checkpoint after every balanced A/B/C update; evaluate "
            "convergence after every round, stop after three consecutive "
            "stable parameter updates, and enforce a 10-round maximum")
        migrations = list(loaded_state.get("protocol_migrations", []))
        migrations.append({
            "version": "operator-webots-transition-ten-round-finetune-v1",
            "migrated_at": migration_time,
            "from_workflow_contract": V18_WORKFLOW_CONTRACT,
            "to_workflow_contract": V19_WORKFLOW_CONTRACT,
            "previous_webots_finetune_maximum_balanced_rounds":
                V15_MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS,
            "webots_finetune_maximum_balanced_rounds":
                MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS,
            "standalone_exit_to_webots_policy_ids":
                list(requested_standalone_exit_ids),
            "preserved_all_committed_standalone_artifacts": True,
            "discard_incomplete_standalone_units": True,
            "webots_and_final_evaluation_serial": True,
        })
        loaded_state.update({
            "schema_version": V19_WORKFLOW_STATE_SCHEMA_VERSION,
            "workflow_contract": V19_WORKFLOW_CONTRACT,
            "webots_finetune_seeds_per_scenario_maximum":
                MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS,
            "standalone_exit_to_webots_policy_ids":
                list(requested_standalone_exit_ids),
            "resume_contract": resume_contract,
            "protocol_migrations": migrations,
            "active_policy_ids": [],
            "status": "interrupted",
            "stage": "protocol_migration",
            "current": None,
            "interrupted_at": migration_time,
            "updated_at": migration_time,
        })
        atomic_json(state_path, loaded_state)
    if (isinstance(loaded_state, dict) and
            loaded_state.get("schema_version") ==
                V19_WORKFLOW_STATE_SCHEMA_VERSION and
            loaded_state.get("workflow_contract") ==
                V19_WORKFLOW_CONTRACT):
        if (loaded_state.get(
                "webots_finetune_seeds_per_scenario_maximum") !=
                MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS or
                loaded_state.get("final_evaluation_seeds_per_scenario") !=
                len(FINAL_EVALUATION_SEEDS) or
                loaded_state.get(
                    "final_evaluation_seed_contract_sha256") !=
                final_evaluation_seed_contract_sha256()):
            raise RuntimeError(
                "v19 maximum-round evaluation migration contract mismatch")
        migration_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        resume_contract = dict(loaded_state.get("resume_contract", {}))
        resume_contract["webots_model"] = (
            "checkpoint after every balanced A/B/C update; stop on three "
            "consecutive stable updates or at the 10-round maximum; a "
            "maximum-completed candidate retains parameter_converged=false "
            "and proceeds to the physical non-regression gate")
        resume_contract["evaluation"] = (
            "after physical non-regression selection, atomically evaluate "
            "each model on the same six final-evaluation seeds per scenario")
        migrations = list(loaded_state.get("protocol_migrations", []))
        migrations.append({
            "version": "maximum-completion-six-seed-evaluation-v1",
            "migrated_at": migration_time,
            "from_workflow_contract": V19_WORKFLOW_CONTRACT,
            "to_workflow_contract": V20_WORKFLOW_CONTRACT,
            "maximum_balanced_rounds":
                MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS,
            "maximum_completion_does_not_fake_convergence": True,
            "maximum_completion_enters_nonregression": True,
            "final_evaluation_seeds": list(FINAL_EVALUATION_SEEDS),
            "final_evaluation_common_random_numbers": True,
            "preserved_all_committed_artifacts": True,
        })
        loaded_state.update({
            "schema_version": V20_WORKFLOW_STATE_SCHEMA_VERSION,
            "workflow_contract": V20_WORKFLOW_CONTRACT,
            "webots_finetune_maximum_completion_action":
                "nonregression_then_six_seed_final_evaluation",
            "resume_contract": resume_contract,
            "protocol_migrations": migrations,
            "status": "interrupted",
            "stage": "protocol_migration",
            "current": None,
            "interrupted_at": migration_time,
            "updated_at": migration_time,
        })
        atomic_json(state_path, loaded_state)
    if (isinstance(loaded_state, dict) and
            loaded_state.get("schema_version") ==
                V20_WORKFLOW_STATE_SCHEMA_VERSION and
            loaded_state.get("workflow_contract") ==
                V20_WORKFLOW_CONTRACT):
        migration_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        resume_contract = dict(loaded_state.get("resume_contract", {}))
        resume_contract["webots_model"] = (
            "checkpoint after every balanced A/B/C update; stop on three "
            "consecutive stable updates or at the 10-round maximum; select "
            "the completed fine-tuned checkpoint directly")
        resume_contract["webots_nonregression"] = (
            "disabled by operator request for every AI model; preserve any "
            "previously committed episodes but do not use them for selection")
        resume_contract["evaluation"] = (
            "after direct fine-tuned-checkpoint selection, atomically "
            "evaluate each model on the same six final-evaluation seeds per "
            "scenario")
        migrations = list(loaded_state.get("protocol_migrations", []))
        migrations.append({
            "version": "direct-finetuned-six-seed-evaluation-v1",
            "migrated_at": migration_time,
            "from_workflow_contract": V20_WORKFLOW_CONTRACT,
            "to_workflow_contract": V21_WORKFLOW_CONTRACT,
            "scope": "all_ai_models",
            "webots_nonregression_enabled": False,
            "selected_checkpoint": "webots_finetuned_model",
            "final_evaluation_seeds": list(FINAL_EVALUATION_SEEDS),
            "final_evaluation_common_random_numbers": True,
            "preserved_all_committed_artifacts": True,
        })
        loaded_state.update({
            "schema_version": V21_WORKFLOW_STATE_SCHEMA_VERSION,
            "workflow_contract": V21_WORKFLOW_CONTRACT,
            "webots_finetune_maximum_completion_action":
                "direct_six_seed_final_evaluation",
            "webots_post_finetune_selection":
                "direct_finetuned_checkpoint",
            "webots_nonregression_enabled": False,
            "resume_contract": resume_contract,
            "protocol_migrations": migrations,
            "status": "interrupted",
            "stage": "protocol_migration",
            "current": None,
            "interrupted_at": migration_time,
            "updated_at": migration_time,
        })
        atomic_json(state_path, loaded_state)
    if (isinstance(loaded_state, dict) and
            loaded_state.get("schema_version") ==
                V21_WORKFLOW_STATE_SCHEMA_VERSION and
            loaded_state.get("workflow_contract") ==
                V21_WORKFLOW_CONTRACT):
        migration_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        resume_contract = dict(loaded_state.get("resume_contract", {}))
        resume_contract["webots_model"] = (
            "checkpoint after every balanced A/B/C update; stop on three "
            "consecutive stable updates or at the 10-round maximum; then "
            "run the two-seed paired Webots regression gate")
        resume_contract["webots_nonregression"] = (
            "compare frozen pre/post checkpoints on the same two isolated "
            "Webots seeds per scenario; accept degradation up to five "
            "percent and otherwise roll back to the pre-tune checkpoint")
        resume_contract["evaluation"] = (
            "after two-seed bounded-regression selection, atomically "
            "evaluate each model on the same six final-evaluation seeds per "
            "scenario")
        migrations = list(loaded_state.get("protocol_migrations", []))
        migrations.append({
            "version": "two-seed-webots-nonregression-v1",
            "migrated_at": migration_time,
            "from_workflow_contract": V21_WORKFLOW_CONTRACT,
            "to_workflow_contract": V22_WORKFLOW_CONTRACT,
            "scope": "all_ai_models_not_already_completed",
            "webots_nonregression_enabled": True,
            "webots_regression_seeds": list(WEBOTS_REGRESSION_SEEDS),
            "webots_regression_common_random_numbers": True,
            "completed_model_artifacts_preserved": True,
            "preserved_all_committed_artifacts": True,
        })
        loaded_state.update({
            "schema_version": V22_WORKFLOW_STATE_SCHEMA_VERSION,
            "workflow_contract": V22_WORKFLOW_CONTRACT,
            "webots_finetune_maximum_completion_action":
                "two_seed_nonregression_then_six_seed_final_evaluation",
            "webots_post_finetune_selection":
                "two_seed_bounded_nonregression",
            "webots_nonregression_enabled": True,
            "webots_regression_seeds_per_scenario":
                len(WEBOTS_REGRESSION_SEEDS),
            "resume_contract": resume_contract,
            "protocol_migrations": migrations,
            "status": "interrupted",
            "stage": "protocol_migration",
            "current": None,
            "interrupted_at": migration_time,
            "updated_at": migration_time,
        })
        atomic_json(state_path, loaded_state)
    if (isinstance(loaded_state, dict) and
            loaded_state.get("schema_version") ==
                V22_WORKFLOW_STATE_SCHEMA_VERSION and
            loaded_state.get("workflow_contract") == V22_WORKFLOW_CONTRACT):
        migration_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        resume_contract = dict(loaded_state.get("resume_contract", {}))
        resume_contract["standalone_sufficient_training"] = (
            "checkpoint after every complete 1800s episode; stop early when "
            "metric and parameters converge, or complete the 200-round "
            "maximum without claiming convergence; select the best guarded "
            "checkpoint and continue to Webots")
        migrations = list(loaded_state.get("protocol_migrations", []))
        migrations.append({
            "version": "standalone-maximum-completion-webots-transition-v1",
            "migrated_at": migration_time,
            "from_workflow_contract": V22_WORKFLOW_CONTRACT,
            "to_workflow_contract": V23_WORKFLOW_CONTRACT,
            "scope": "all_ai_models_not_already_completed",
            "maximum_balanced_rounds": MAX_STANDALONE_BALANCED_ROUNDS,
            "maximum_completion_does_not_fake_parameter_convergence": True,
            "maximum_completion_selects_guarded_checkpoint": True,
            "maximum_completion_enters_webots": True,
            "preserved_all_committed_artifacts": True,
        })
        loaded_state.update({
            "schema_version": V23_WORKFLOW_STATE_SCHEMA_VERSION,
            "workflow_contract": V23_WORKFLOW_CONTRACT,
            "standalone_maximum_completion_action":
                "guarded_checkpoint_then_webots",
            "resume_contract": resume_contract,
            "protocol_migrations": migrations,
            "status": "interrupted",
            "stage": "protocol_migration",
            "current": None,
            "interrupted_at": migration_time,
            "updated_at": migration_time,
        })
        atomic_json(state_path, loaded_state)
    if (isinstance(loaded_state, dict) and
            loaded_state.get("schema_version") ==
                V23_WORKFLOW_STATE_SCHEMA_VERSION and
            loaded_state.get("workflow_contract") == V23_WORKFLOW_CONTRACT):
        migration_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        resume_contract = dict(loaded_state.get("resume_contract", {}))
        resume_contract["webots_nonregression"] = (
            "compare frozen pre/post checkpoints on the same two isolated "
            "Webots seeds per scenario; retain recoverable timeout and "
            "fallback counts as audit evidence without blocking selection, "
            "while invalid scheduler outputs still fail closed; accept metric "
            "degradation up to five percent and otherwise roll back to the "
            "pre-tune checkpoint")
        migrations = list(loaded_state.get("protocol_migrations", []))
        migrations.append({
            "version": "webots-regression-timeout-fallback-audit-only-v1",
            "migrated_at": migration_time,
            "from_workflow_contract": V23_WORKFLOW_CONTRACT,
            "to_workflow_contract": V24_WORKFLOW_CONTRACT,
            "scope": "all_ai_models",
            "runtime_event_policy":
                WEBOTS_REGRESSION_RUNTIME_EVENT_POLICY,
            "timeout_fallback_affect_checkpoint_selection": False,
            "invalid_scheduler_outputs_block_selection": True,
            "bounded_metric_regression_guard_preserved": True,
            "preserved_all_committed_artifacts": True,
        })
        loaded_state.update({
            "schema_version": V24_WORKFLOW_STATE_SCHEMA_VERSION,
            "workflow_contract": V24_WORKFLOW_CONTRACT,
            "webots_regression_runtime_event_policy":
                WEBOTS_REGRESSION_RUNTIME_EVENT_POLICY,
            "resume_contract": resume_contract,
            "protocol_migrations": migrations,
            "status": "interrupted",
            "stage": "protocol_migration",
            "current": None,
            "interrupted_at": migration_time,
            "updated_at": migration_time,
        })
        atomic_json(state_path, loaded_state)
    if (isinstance(loaded_state, dict) and
            loaded_state.get("schema_version") ==
                V24_WORKFLOW_STATE_SCHEMA_VERSION and
            loaded_state.get("workflow_contract") == V24_WORKFLOW_CONTRACT):
        migration_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        already_finetuned_policy_ids = [
            spec["policy_id"] for spec in PIPELINE_AI_SPECS
            if _has_committed_fine_tune_for_tolerance_migration(
                output_dir, spec)]
        already_finetuned = set(already_finetuned_policy_ids)
        webots_tolerances = {
            policy_id: (
                PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE
                if policy_id in already_finetuned else
                COUNT_RELATIVE_DEGRADATION_TOLERANCE)
            for policy_id in PIPELINE_POLICY_IDS}
        resume_contract = dict(loaded_state.get("resume_contract", {}))
        resume_contract["webots_nonregression"] = (
            "compare frozen pre/post checkpoints on the same two isolated "
            "Webots seeds per scenario; retain recoverable timeout and "
            "fallback counts as audit evidence without blocking selection, "
            "while invalid scheduler outputs still fail closed; models with "
            "a completed fine-tune before this migration retain their frozen "
            "five-percent budget, and only models not yet fine-tuned receive "
            "the ten-percent budget")
        migrations = list(loaded_state.get("protocol_migrations", []))
        migrations.append({
            "version": "unfinetuned-ten-percent-webots-regression-v1",
            "migrated_at": migration_time,
            "from_workflow_contract": V24_WORKFLOW_CONTRACT,
            "to_workflow_contract": WORKFLOW_CONTRACT,
            "scope": "ai_models_without_completed_webots_fine_tune",
            "already_finetuned_policy_ids": already_finetuned_policy_ids,
            "already_finetuned_relative_tolerance":
                PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE,
            "not_yet_finetuned_relative_tolerance":
                COUNT_RELATIVE_DEGRADATION_TOLERANCE,
            "standalone_relative_tolerance_preserved":
                STANDALONE_COUNT_RELATIVE_DEGRADATION_TOLERANCE,
            "policy_tolerances_frozen_at_migration": True,
            "preserved_all_committed_artifacts": True,
        })
        loaded_state.update({
            "schema_version": WORKFLOW_STATE_SCHEMA_VERSION,
            "workflow_contract": WORKFLOW_CONTRACT,
            "bounded_regression_guard": {
                "count_relative_tolerance":
                    STANDALONE_COUNT_RELATIVE_DEGRADATION_TOLERANCE,
                "utility_relative_tolerance":
                    STANDALONE_UTILITY_RELATIVE_DEGRADATION_TOLERANCE,
                "immutable_training_baseline": True,
            },
            "webots_finetune_bounded_regression_guard": {
                "already_finetuned_relative_tolerance":
                    PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE,
                "not_yet_finetuned_relative_tolerance":
                    COUNT_RELATIVE_DEGRADATION_TOLERANCE,
                "policy_tolerances_frozen": True,
            },
            "webots_finetune_regression_tolerance_by_policy":
                webots_tolerances,
            "webots_finetune_legacy_tolerance_policy_ids":
                already_finetuned_policy_ids,
            "resume_contract": resume_contract,
            "protocol_migrations": migrations,
            "status": "interrupted",
            "stage": "protocol_migration",
            "current": None,
            "interrupted_at": migration_time,
            "updated_at": migration_time,
        })
        atomic_json(state_path, loaded_state)
    state = loaded_state if loaded_state is not None else {
        "schema_version": WORKFLOW_STATE_SCHEMA_VERSION,
        "workflow_contract": WORKFLOW_CONTRACT,
        "training_semantics_version": TRAINING_SEMANTICS_VERSION,
        "formal_protocol_sha256": formal_protocol_sha256(),
        "ai_objective_instances": len(AI_SPECS),
        "pipeline_execution_contract": (
            "two-headless-lanes-serial-physical-v1"),
        "pipeline_policy_order": list(PIPELINE_POLICY_IDS),
        "completed_model_pipelines": [],
        "active_policy_ids": [],
        "standalone_exit_to_webots_policy_ids":
            list(requested_standalone_exit_ids),
        "final_evaluation_policy_count": FINAL_EVALUATION_POLICY_COUNT,
        "final_evaluation_seeds_per_scenario": len(
            FINAL_EVALUATION_SEEDS),
        "final_evaluation_seed_contract_sha256":
            final_evaluation_seed_contract_sha256(),
        "standalone_training_seeds_per_scenario_minimum": (
            MIN_STANDALONE_BALANCED_ROUNDS),
        "standalone_training_seeds_per_scenario_maximum": (
            MAX_STANDALONE_BALANCED_ROUNDS),
        "standalone_maximum_completion_action":
            "guarded_checkpoint_then_webots",
        "hyperparameter_trials_maximum": HYPERPARAMETER_TRIALS_MAXIMUM,
        "hyperparameter_trials_by_policy": HYPERPARAMETER_TRIAL_COUNTS,
        "parallel_headless_execution_contract":
            PARALLEL_HEADLESS_EXECUTION_CONTRACT,
        "parallel_headless_model_workers":
            PARALLEL_HEADLESS_MODEL_WORKERS,
        "parallel_headless_process_budget":
            PARALLEL_HEADLESS_PROCESS_BUDGET,
        "parallel_hyperparameter_trial_workers":
            PARALLEL_HYPERPARAMETER_TRIAL_WORKERS,
        "parallel_frozen_validation_workers":
            PARALLEL_FROZEN_VALIDATION_WORKERS,
        "tuning_training_seeds_per_scenario": len(TUNING_TRAINING_SEEDS),
        "tuning_validation_seeds_per_scenario": len(TUNING_VALIDATION_SEEDS),
        "tuning_validation_seed_contract": list(TUNING_VALIDATION_SEEDS),
        "webots_finetune_seeds_per_scenario_minimum":
            MIN_WEBOTS_FINETUNE_BALANCED_ROUNDS,
        "webots_finetune_convergence_patience_rounds":
            WEBOTS_FINETUNE_CONVERGENCE_PATIENCE,
        "webots_finetune_seeds_per_scenario_maximum":
            MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS,
        "webots_finetune_maximum_completion_action":
            "two_seed_nonregression_then_six_seed_final_evaluation",
        "webots_post_finetune_selection":
            "two_seed_bounded_nonregression",
        "webots_nonregression_enabled": True,
        "webots_regression_seeds_per_scenario":
            len(WEBOTS_REGRESSION_SEEDS),
        "webots_regression_runtime_event_policy":
            WEBOTS_REGRESSION_RUNTIME_EVENT_POLICY,
        "bounded_regression_guard": {
            "count_relative_tolerance":
                STANDALONE_COUNT_RELATIVE_DEGRADATION_TOLERANCE,
            "utility_relative_tolerance":
                STANDALONE_UTILITY_RELATIVE_DEGRADATION_TOLERANCE,
            "immutable_training_baseline": True,
        },
        "webots_finetune_bounded_regression_guard": {
            "already_finetuned_relative_tolerance":
                PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE,
            "not_yet_finetuned_relative_tolerance":
                COUNT_RELATIVE_DEGRADATION_TOLERANCE,
            "policy_tolerances_frozen": True,
        },
        "webots_finetune_regression_tolerance_by_policy": {
            policy_id: COUNT_RELATIVE_DEGRADATION_TOLERANCE
            for policy_id in PIPELINE_POLICY_IDS
        },
        "webots_finetune_legacy_tolerance_policy_ids": [],
        "evaluation_metrics": ["count", "utility_v2"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_seconds": DURATION_SECONDS,
        "resume_contract": {
            "standalone_hyperparameter_tuning": (
                "automatic deterministic selection from isolated trials; up "
                "to two trial groups execute concurrently, each model-update "
                "chain remains serial, and only the parent writes workflow "
                "state"),
            "frozen_checkpoint_validation": (
                "up to three independent scenario/seed episodes execute in "
                "isolated processes; the checkpoint is read-only and the "
                "parent commits results in deterministic scenario/seed order"),
            "model_pipeline": (
                "run each scheduler's count and utility models in two "
                "isolated headless lanes; the parent revalidates identity and "
                "then serializes Webots fine-tuning, non-regression, and "
                "final evaluation; LearnedHungarian and GraphImitation run "
                "last"),
            "standalone_sufficient_training": (
                "checkpoint after every complete 1800s episode; stop early "
                "when metric and parameters converge, or complete the "
                "200-round maximum without claiming convergence; select the "
                "best guarded checkpoint and continue to Webots"),
            "webots_episode": (
                "atomic result after every complete 1800s physical episode"),
            "webots_model": (
                "checkpoint after every balanced A/B/C update; evaluate "
                "convergence after every round, stop after three consecutive "
                "stable parameter updates or at the 10-round maximum; then "
                "run the two-seed paired Webots regression gate without "
                "claiming parameter convergence"),
            "webots_nonregression": (
                "compare frozen pre/post checkpoints on the same two isolated "
                "Webots seeds per scenario; retain recoverable timeout and "
                "fallback counts as audit evidence without blocking "
                "selection, while invalid scheduler outputs still fail "
                "closed; models not yet fine-tuned use a ten-percent metric "
                "degradation budget, frozen per policy before fine-tuning; "
                "otherwise roll back to the pre-tune checkpoint"),
            "evaluation": (
                "after two-seed bounded-regression selection, atomically "
                "evaluate each model on the same six final-evaluation seeds "
                "per scenario"),
            "incomplete_unit": "restart only the interrupted current unit",
        },
    }
    raw_standalone_exit_ids = state.get(
        "standalone_exit_to_webots_policy_ids", [])
    normalized_standalone_exit_ids = _ordered_policy_id_subset(
        raw_standalone_exit_ids)
    if raw_standalone_exit_ids != normalized_standalone_exit_ids:
        raise RuntimeError(
            "standalone Webots transition policy order is invalid")
    merged_standalone_exit_ids = _ordered_policy_id_subset(
        normalized_standalone_exit_ids + requested_standalone_exit_ids)
    if merged_standalone_exit_ids != normalized_standalone_exit_ids:
        newly_requested = [
            policy_id for policy_id in merged_standalone_exit_ids
            if policy_id not in normalized_standalone_exit_ids]
        if any(
                path.is_file()
                for policy_id in newly_requested
                for root in (
                    output_dir / "webots_fine_tune" / policy_id,
                    output_dir / "evaluation" / policy_id)
                if root.exists()
                for path in root.glob("**/*")):
            raise RuntimeError(
                "cannot add a standalone exit override after physical "
                "artifacts exist")
        transition_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        events = list(state.get("operator_transition_events", []))
        events.append({
            "version": STANDALONE_EXIT_OVERRIDE_VERSION,
            "requested_at": transition_time,
            "requested_by": "user",
            "policy_ids": newly_requested,
            "action": "enter_webots_without_further_standalone_training",
        })
        state.update({
            "standalone_exit_to_webots_policy_ids":
                merged_standalone_exit_ids,
            "operator_transition_events": events,
            "status": "interrupted",
            "stage": "operator_transition_registered",
            "current": None,
            "interrupted_at": transition_time,
            "updated_at": transition_time,
        })
        atomic_json(state_path, state)
        normalized_standalone_exit_ids = merged_standalone_exit_ids

    # Only the explicit, artifact-gated migrations above may reuse a prior
    # state.  Every other contract mismatch fails closed so legacy checkpoints
    # cannot look eligible for the current run.
    completed_model_pipelines = state.get("completed_model_pipelines")
    completed_pipeline_prefix_valid = bool(
        isinstance(completed_model_pipelines, list) and
        completed_model_pipelines == list(
            PIPELINE_POLICY_IDS[:len(completed_model_pipelines)]))
    raw_webots_legacy_tolerance_policy_ids = state.get(
        "webots_finetune_legacy_tolerance_policy_ids")
    webots_legacy_tolerance_policy_ids_valid = bool(
        isinstance(raw_webots_legacy_tolerance_policy_ids, list) and
        raw_webots_legacy_tolerance_policy_ids == _ordered_policy_id_subset(
            raw_webots_legacy_tolerance_policy_ids))
    expected_webots_tolerances = {
        policy_id: (
            PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE
            if policy_id in (raw_webots_legacy_tolerance_policy_ids or [])
            else COUNT_RELATIVE_DEGRADATION_TOLERANCE)
        for policy_id in PIPELINE_POLICY_IDS}
    if not (
            state.get("schema_version") == WORKFLOW_STATE_SCHEMA_VERSION and
            state.get("workflow_contract") == WORKFLOW_CONTRACT and
            state.get("training_semantics_version") ==
                TRAINING_SEMANTICS_VERSION and
            state.get("formal_protocol_sha256") == formal_protocol_sha256() and
            state.get("ai_objective_instances") == 20 and
            state.get("pipeline_execution_contract") ==
                "two-headless-lanes-serial-physical-v1" and
            state.get("pipeline_policy_order") == list(PIPELINE_POLICY_IDS) and
            state.get("standalone_exit_to_webots_policy_ids") ==
                normalized_standalone_exit_ids and
            completed_pipeline_prefix_valid and
            state.get("final_evaluation_policy_count") ==
                FINAL_EVALUATION_POLICY_COUNT and
            state.get("final_evaluation_seeds_per_scenario") ==
                len(FINAL_EVALUATION_SEEDS) and
            state.get("final_evaluation_seed_contract_sha256") ==
                final_evaluation_seed_contract_sha256() and
            state.get("standalone_training_seeds_per_scenario_minimum") ==
                MIN_STANDALONE_BALANCED_ROUNDS and
            state.get("standalone_training_seeds_per_scenario_maximum") ==
                MAX_STANDALONE_BALANCED_ROUNDS and
            state.get("standalone_maximum_completion_action") ==
                "guarded_checkpoint_then_webots" and
            state.get("hyperparameter_trials_maximum") ==
                HYPERPARAMETER_TRIALS_MAXIMUM and
            state.get("hyperparameter_trials_by_policy") ==
                HYPERPARAMETER_TRIAL_COUNTS and
            state.get("parallel_headless_execution_contract") ==
                PARALLEL_HEADLESS_EXECUTION_CONTRACT and
            state.get("parallel_headless_model_workers") ==
                PARALLEL_HEADLESS_MODEL_WORKERS and
            state.get("parallel_headless_process_budget") ==
                PARALLEL_HEADLESS_PROCESS_BUDGET and
            state.get("parallel_hyperparameter_trial_workers") ==
                PARALLEL_HYPERPARAMETER_TRIAL_WORKERS and
            state.get("parallel_frozen_validation_workers") ==
                PARALLEL_FROZEN_VALIDATION_WORKERS and
            state.get("tuning_training_seeds_per_scenario") ==
                len(TUNING_TRAINING_SEEDS) and
            state.get("tuning_validation_seeds_per_scenario") ==
                len(TUNING_VALIDATION_SEEDS) and
            state.get("tuning_validation_seed_contract") ==
                list(TUNING_VALIDATION_SEEDS) and
            state.get("webots_finetune_seeds_per_scenario_minimum") ==
                MIN_WEBOTS_FINETUNE_BALANCED_ROUNDS and
            state.get("webots_finetune_convergence_patience_rounds") ==
                WEBOTS_FINETUNE_CONVERGENCE_PATIENCE and
            state.get("webots_finetune_seeds_per_scenario_maximum") ==
                MAX_WEBOTS_FINETUNE_BALANCED_ROUNDS and
            state.get("webots_finetune_maximum_completion_action") ==
                "two_seed_nonregression_then_six_seed_final_evaluation" and
            state.get("webots_post_finetune_selection") ==
                "two_seed_bounded_nonregression" and
            state.get("webots_nonregression_enabled") is True and
            state.get("webots_regression_seeds_per_scenario") ==
                len(WEBOTS_REGRESSION_SEEDS) and
            state.get("webots_regression_runtime_event_policy") ==
                WEBOTS_REGRESSION_RUNTIME_EVENT_POLICY and
            state.get("bounded_regression_guard") == {
                "count_relative_tolerance":
                    STANDALONE_COUNT_RELATIVE_DEGRADATION_TOLERANCE,
                "utility_relative_tolerance":
                    STANDALONE_UTILITY_RELATIVE_DEGRADATION_TOLERANCE,
                "immutable_training_baseline": True,
            } and
            state.get("webots_finetune_bounded_regression_guard") == {
                "already_finetuned_relative_tolerance":
                    PREVIOUS_WEBOTS_RELATIVE_DEGRADATION_TOLERANCE,
                "not_yet_finetuned_relative_tolerance":
                    COUNT_RELATIVE_DEGRADATION_TOLERANCE,
                "policy_tolerances_frozen": True,
            } and
            webots_legacy_tolerance_policy_ids_valid and
            state.get("webots_finetune_regression_tolerance_by_policy") ==
                expected_webots_tolerances and
            state.get("evaluation_metrics") == ["count", "utility_v2"]):
        raise RuntimeError(
            "output directory belongs to a different workflow contract")

    for policy_id in normalized_standalone_exit_ids:
        spec = next(
            row for row in PIPELINE_AI_SPECS
            if row["policy_id"] == policy_id)
        _ensure_standalone_exit_override(output_dir, spec)

    def update_state(**values):
        for key in ("error", "failed_at", "interrupted_at", "completed_at"):
            state.pop(key, None)
        state.update(values)
        state["status"] = "running"
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_json(state_path, state)

    try:
        update_state(stage="preflight", current=None)
        if args.phase in {"fine-tune", "evaluate"}:
            for spec in PIPELINE_AI_SPECS:
                _apply_tuning_selection(output_dir, spec, required=True)
        progress: dict[str, int] = {}
        if args.phase == "all":
            if not run_model_by_model_pipeline(
                    output_dir, args.webots, update_state,
                    completed_policy_ids=completed_model_pipelines,
                    max_standalone_episodes=args.max_standalone_episodes,
                    max_fine_tune_episodes=args.max_fine_tune_episodes,
                    max_regression_runs=args.max_regression_runs,
                    max_evaluation_runs=args.max_evaluation_runs,
                    stop_after_policy_id=getattr(
                        args, "stop_after_policy", None)):
                state.update({"status": "stopped_at_requested_run_limit"})
                atomic_json(state_path, state)
                return 0
        elif args.phase in {"standalone", "headless"}:
            if not train_headless(
                    output_dir, update_state,
                    max_new_episodes=args.max_standalone_episodes,
                    specs=PIPELINE_AI_SPECS, progress=progress):
                state.update({"status": "stopped_at_requested_run_limit",
                              "current": None})
                atomic_json(state_path, state)
                return 0
        elif args.phase == "fine-tune":
            for spec in PIPELINE_AI_SPECS:
                if not fine_tune_all(
                        output_dir, args.webots, update_state,
                        max_new_episodes=args.max_fine_tune_episodes,
                        specs=(spec,), progress=progress):
                    state.update({
                        "status": "stopped_at_requested_run_limit",
                        "stage": "webots_fine_tune", "current": None})
                    atomic_json(state_path, state)
                    return 0
                if not guard_all_fine_tuned_models(
                        output_dir, args.webots, update_state,
                        max_new_runs=args.max_regression_runs,
                        specs=(spec,), progress=progress):
                    state.update({
                        "status": "stopped_at_requested_run_limit",
                        "stage": "webots_fine_tune_nonregression",
                        "current": None})
                    atomic_json(state_path, state)
                    return 0
        elif args.phase == "evaluate":
            if not evaluate_all(
                    output_dir, args.webots, update_state,
                    max_new_runs=args.max_evaluation_runs,
                    progress=progress):
                state.update({"status": "stopped_at_requested_run_limit",
                              "stage": "evaluate_all", "current": None})
                atomic_json(state_path, state)
                return 0
        elif args.phase == "baselines":
            if not evaluate_baselines(
                    output_dir, args.webots, update_state,
                    max_new_runs=args.max_evaluation_runs,
                    progress=progress):
                state.update({
                    "status": "stopped_at_requested_run_limit",
                    "stage": "baseline_final_evaluation",
                    "current": None,
                })
                atomic_json(state_path, state)
                return 0
            state.update({
                "status": "baseline_evaluation_complete",
                "stage": "baseline_final_evaluation_complete",
                "current": None,
                "baseline_evaluation_completed_at":
                    time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            atomic_json(state_path, state)
            print("[BASELINES COMPLETE] FCFS and NearestNeighbour", flush=True)
            return 0
        state.update({
            "status": "complete", "stage": "complete", "current": None,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        atomic_json(state_path, state)
        if args.phase in {"all", "evaluate"}:
            import generate_strict_1800_report as report_generator
            report_generator.atomic_write_text(
                args.report_path.resolve(),
                report_generator.build_report(output_dir))
            print(
                f"[REPORT] {args.report_path.resolve()}", flush=True)
        print(f"[COMPLETE] report root: {output_dir}", flush=True)
        return 0
    except KeyboardInterrupt:
        state.update({
            "status": "interrupted",
            "interrupted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        atomic_json(state_path, state)
        print(
            "[INTERRUPTED] restart the same command; all committed "
            "episode/model checkpoints will be reused.", flush=True)
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
