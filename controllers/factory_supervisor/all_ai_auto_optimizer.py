"""Deterministic, independent hyperparameter spaces for every trainable AI."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from types import MappingProxyType
from typing import Mapping, Tuple

from advanced_rl_agents import CQLConfig, QRDQNConfig, RainbowConfig
from bandit_scheduler import LinUCBModel
from graph_ppo_scheduler import GraphPPOConfig
from headless_training_runtime import HEADLESS_DYNAMICS_VERSION
from rl_agents import DQNConfig, SarsaConfig


ALL_AI_SEARCH_VERSION = (
    "all-ai-dual-objective-independent-search-v4-stratified-coverage")
VALIDATION_SELECTOR_VERSION = "strict-validation-selector-v3"
FORMAL_VALIDATION_SEEDS = tuple(range(31000, 31005))
FORMAL_SCENARIOS = ("A", "B", "C")
STANDALONE_TRAINING_SEEDS = tuple(range(21000, 21200))
MIN_STANDALONE_BALANCED_ROUNDS = 100
MAX_STANDALONE_BALANCED_ROUNDS = 200
VALIDATION_INTERVAL_ROUNDS = 5
CONVERGENCE_PATIENCE_CHECKS = 3
PARAMETER_CONVERGENCE_PATIENCE_CHECKS = 3
PARAMETER_RELATIVE_CHANGE_THRESHOLD = 1e-2
CONVERGENCE_MIN_IMPROVEMENT = {
    "count": 0.5,
    "utility_v2": 0.5,
}
SUPERVISED_MIN_NEAREST_NEIGHBOUR_RATIO = 0.90
SUPERVISED_MIN_OBJECTIVE_EXPERT_RATIO = 0.80
TRAINABLE_ALGORITHMS = (
    "LearnedHungarian",
    "GraphImitation",
    "GraphPPO",
    "RainbowDQN",
    "QRDQN",
    "CQL",
    "LinUCB",
    "PPO_RL",
    "SARSA",
    "DQN",
)
TRAINABLE_OBJECTIVES = ("count", "utility_v2")
TRAINABLE_POLICY_IDS = tuple(
    f"{algorithm}_{objective}"
    for algorithm in TRAINABLE_ALGORITHMS
    for objective in TRAINABLE_OBJECTIVES)


def policy_algorithm(policy_id: str) -> str:
    """Return the algorithm family of a registered objective instance."""
    if not isinstance(policy_id, str):
        raise ValueError("policy id must be a string")
    for objective in TRAINABLE_OBJECTIVES:
        suffix = f"_{objective}"
        if policy_id.endswith(suffix):
            algorithm = policy_id[:-len(suffix)]
            if algorithm in TRAINABLE_ALGORITHMS:
                return algorithm
    raise ValueError("policy id is not a registered objective instance")


def _canonical_json(document) -> str:
    return json.dumps(
        document, ensure_ascii=True, allow_nan=False,
        sort_keys=True, separators=(",", ":"))


def _sha256(document) -> str:
    return hashlib.sha256(_canonical_json(document).encode("utf-8")).hexdigest()


def _valid_value(value) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


@dataclass(frozen=True)
class AlgorithmSearchSpace:
    policy_id: str
    scheduler: str
    objective: str
    training_mode: str
    choices: Tuple[Tuple[str, Tuple[object, ...]], ...]
    fixed_contract: Tuple[Tuple[str, object], ...] = ()

    def __post_init__(self):
        if (self.policy_id not in TRAINABLE_POLICY_IDS or
                self.scheduler not in TRAINABLE_ALGORITHMS or
                self.objective not in {"count", "utility_v2"} or
                self.policy_id != f"{self.scheduler}_{self.objective}" or
                self.training_mode not in {
                    "supervised", "online", "offline", "bandit"}):
            raise ValueError("invalid algorithm search-space identity")
        names = [name for name, _values in self.choices]
        if not names or len(names) != len(set(names)):
            raise ValueError("search-space parameter names must be unique")
        fixed_names = [name for name, _value in self.fixed_contract]
        if (len(fixed_names) != len(set(fixed_names)) or
                set(names) & set(fixed_names)):
            raise ValueError("fixed/search parameter names overlap or repeat")
        for name, values in self.choices:
            if not isinstance(name, str) or not name or not values:
                raise ValueError("search-space choices must be named/nonempty")
            encoded = [_canonical_json(value) for value in values]
            if (len(encoded) != len(set(encoded)) or
                    any(not _valid_value(value) for value in values)):
                raise ValueError("search-space choices must be finite/unique")
        if any(not isinstance(name, str) or not name or
               not _valid_value(value)
               for name, value in self.fixed_contract):
            raise ValueError("fixed search-space contract is invalid")
        # Every value in every dimension must materialize a valid model or
        # trainer configuration, not merely a syntactically valid grid.
        baseline = {name: values[0] for name, values in self.choices}
        for name, values in self.choices:
            for value in values:
                candidate = dict(baseline)
                candidate[name] = value
                _validate_parameters(self.policy_id, candidate)

    @property
    def size(self) -> int:
        return math.prod(len(values) for _name, values in self.choices)

    def canonical(self) -> dict:
        return {
            "version": ALL_AI_SEARCH_VERSION,
            "policy_id": self.policy_id,
            "scheduler": self.scheduler,
            "objective": self.objective,
            "training_mode": self.training_mode,
            "choices": {name: list(values) for name, values in self.choices},
            "fixed_contract": dict(self.fixed_contract),
            "size": self.size,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical())


def _space(policy_id, scheduler, objective, training_mode, choices,
           fixed_contract=()):
    return AlgorithmSearchSpace(
        policy_id, scheduler, objective, training_mode,
        tuple((name, tuple(values)) for name, values in choices),
        tuple(fixed_contract))


def _validate_parameters(policy_id: str, parameters: Mapping[str, object]
                         ) -> None:
    if not isinstance(parameters, Mapping):
        raise ValueError("trial parameters must be a mapping")
    values = dict(parameters)
    algorithm = policy_algorithm(policy_id)
    if algorithm == "LearnedHungarian":
        l2 = values["l2"]
        if (isinstance(l2, bool) or not isinstance(l2, (int, float)) or
                not math.isfinite(l2) or l2 < 0):
            raise ValueError("invalid ridge regularization")
    elif algorithm == "GraphImitation":
        epochs = values["epochs"]
        rate, l2 = values["learning_rate"], values["l2"]
        if (isinstance(epochs, bool) or not isinstance(epochs, int) or
                epochs < 1 or isinstance(rate, bool) or
                not isinstance(rate, (int, float)) or
                not math.isfinite(rate) or rate <= 0 or
                isinstance(l2, bool) or not isinstance(l2, (int, float)) or
                not math.isfinite(l2) or l2 < 0):
            raise ValueError("invalid graph-imitation parameters")
    elif algorithm == "GraphPPO":
        GraphPPOConfig(**values)
    elif algorithm == "RainbowDQN":
        RainbowConfig(**values)
    elif algorithm == "QRDQN":
        QRDQNConfig(**values)
    elif algorithm == "CQL":
        config = CQLConfig(**values)
        if config.priority_alpha != 0 or config.priority_beta != 0:
            raise ValueError("formal CQL must use an immutable offline dataset")
    elif algorithm == "LinUCB":
        LinUCBModel.create(alpha=values["alpha"])
    elif algorithm == "PPO_RL":
        integers = (values["hidden_size"], values["update_epochs"])
        numeric = (
            values["learning_rate"], values["clip_ratio"],
            values["value_coefficient"], values["entropy_coefficient"],
            values["max_grad_norm"], values["gamma"],
            values["value_huber_delta"],
        )
        if (any(isinstance(value, bool) or not isinstance(value, int) or
                value < 1 for value in integers) or
                any(isinstance(value, bool) or not isinstance(
                    value, (int, float)) or not math.isfinite(value)
                    for value in numeric) or
                values["learning_rate"] <= 0 or
                not 0 < values["clip_ratio"] <= 1 or
                values["value_coefficient"] < 0 or
                values["entropy_coefficient"] < 0 or
                values["max_grad_norm"] <= 0 or values["gamma"] != 1.0 or
                values["value_huber_delta"] <= 0):
            raise ValueError("invalid PPO_RL parameters")
    elif algorithm == "SARSA":
        SarsaConfig(**values)
    elif algorithm == "DQN":
        DQNConfig(**values)
    else:
        raise ValueError("unknown trainable policy")


_UTILITY_SPACES = (
    _space("LearnedHungarian_utility_v2", "LearnedHungarian", "utility_v2",
           "supervised", (("l2", (0.01, 0.1, 1.0, 10.0)),),
           (("target", "utility_v2_objective_expert_cost"),)),
    _space("GraphImitation_utility_v2", "GraphImitation", "utility_v2",
           "supervised", (
               ("epochs", (300, 600, 1000)),
               ("learning_rate", (0.01, 0.03, 0.05)),
               ("l2", (1e-5, 1e-4, 1e-3))),
           (("expert", "UtilityDeadlineHungarian"),)),
    _space("GraphPPO_utility_v2", "GraphPPO", "utility_v2", "online", (
        ("hidden_size", (32, 48, 64)), ("gamma", (1.0,)),
        ("gae_lambda", (0.9, 0.95)), ("clip_epsilon", (0.1, 0.2)),
        ("learning_rate", (0.0002, 0.0005, 0.001)),
        ("update_epochs", (2, 4)), ("value_coefficient", (0.5,)),
        ("entropy_coefficient", (0.005, 0.01)),
        ("max_grad_norm", (5.0,)), ("value_huber_delta", (10.0,)))),
    _space("RainbowDQN_utility_v2", "RainbowDQN", "utility_v2", "online", (
        ("hidden_size", (48, 64, 96)), ("atoms", (51,)),
        ("value_min", (-500.0,)), ("value_max", (100.0,)),
        ("raw_reward_min", (-50000.0,)),
        ("raw_reward_max", (10000.0,)), ("reward_scale", (100.0,)),
        ("gamma", (1.0,)), ("n_step", (1, 3)),
        ("replay_capacity", (20000,)), ("batch_size", (32, 64)),
        ("warmup_steps", (256,)),
        ("learning_rate", (0.0002, 0.0005, 0.001)),
        ("target_update_interval", (128, 250, 500)),
        ("priority_alpha", (0.6,)), ("priority_beta", (0.4,)),
        ("epsilon", (0.02, 0.05)))),
    _space("QRDQN_utility_v2", "QRDQN", "utility_v2", "online", (
        ("hidden_size", (48, 64, 96)), ("quantiles", (32,)),
        ("risk_fraction", (0.25, 0.5, 0.75, 1.0)),
        ("huber_kappa", (1.0, 5.0)), ("gamma", (1.0,)),
        ("n_step", (1, 3)), ("replay_capacity", (20000,)),
        ("batch_size", (32, 64)), ("warmup_steps", (256,)),
        ("learning_rate", (0.0002, 0.0005)),
        ("target_update_interval", (128, 250)),
        ("priority_alpha", (0.6,)), ("priority_beta", (0.4,)),
        ("epsilon", (0.02, 0.05)))),
    _space("CQL_utility_v2", "CQL", "utility_v2", "offline", (
        ("hidden_size", (48, 64, 96)), ("gamma", (1.0,)),
        ("conservative_weight", (0.25, 0.5, 1.0, 2.0)),
        ("replay_capacity", (50000,)), ("batch_size", (32, 64)),
        ("warmup_steps", (64,)),
        ("learning_rate", (0.0002, 0.0005, 0.001)),
        ("target_update_interval", (128, 250)),
        ("priority_alpha", (0.0,)), ("priority_beta", (0.0,))),
           (("dataset", "fixed_external_behavior_v2"),
            ("policy_generated_transitions", False))),
    _space("LinUCB_utility_v2", "LinUCB", "utility_v2", "bandit",
           (("alpha", (0.1, 0.25, 0.5, 1.0)),),
           (("arms", "step4_legal_baselines"),)),
    _space("PPO_RL_utility_v2", "PPO_RL", "utility_v2", "online", (
        ("hidden_size", (64, 128, 256)),
        ("learning_rate", (0.0001, 0.0003, 0.0005)),
        ("clip_ratio", (0.1, 0.2)), ("value_coefficient", (0.25, 0.5)),
        ("entropy_coefficient", (0.005, 0.01)),
        ("update_epochs", (4, 8)), ("max_grad_norm", (1.0,)),
        ("gamma", (1.0,)), ("value_huber_delta", (10.0,)))),
    _space("SARSA_utility_v2", "SARSA", "utility_v2", "online", (
        ("learning_rate", (0.03, 0.05, 0.1)), ("gamma", (1.0,)),
        ("epsilon_start", (1.0,)), ("epsilon_end", (0.02, 0.05)),
        ("epsilon_decay", (0.95, 0.97, 0.98)),
        ("physical_epsilon_cap", (0.02,)))),
    _space("DQN_utility_v2", "DQN", "utility_v2", "online", (
        ("hidden_size", (48, 64, 96)),
        ("learning_rate", (0.0002, 0.0005, 0.001)), ("gamma", (1.0,)),
        ("batch_size", (32, 64)), ("replay_capacity", (20000,)),
        ("warmup_steps", (256,)),
        ("target_update_interval", (128, 250, 500)),
        ("epsilon_start", (1.0,)), ("epsilon_end", (0.05,)),
        ("epsilon_decay_steps", (10000, 20000, 40000)),
        ("physical_epsilon_cap", (0.02,)), ("max_grad_norm", (5.0,)),
        ("double_dqn", (True,)), ("adam_beta1", (0.9,)),
        ("adam_beta2", (0.999,)), ("adam_epsilon", (1e-8,)))),
)


def _count_variant(utility_space: AlgorithmSearchSpace
                   ) -> AlgorithmSearchSpace:
    """Build the Count-specific grid and objective contract.

    Reward-scale-sensitive parameters must not be inherited from Utility V2:
    a 0/1 completion signal on a [-500, 100] distributional support is nearly
    invisible.  Other algorithms also receive Count-appropriate robustness,
    exploration, or regularisation ranges instead of sharing one winner.
    """
    overrides = {
        "LearnedHungarian": {
            "fixed_contract": (("target", "count_objective_expert_cost"),),
        },
        "GraphImitation": {
            "fixed_contract": (("expert", "CountThroughputHungarian"),),
        },
        "GraphPPO": {
            "value_huber_delta": (1.0, 2.0, 5.0),
        },
        "RainbowDQN": {
            "value_min": (0.0,), "value_max": (128.0,),
            "raw_reward_min": (0.0,), "raw_reward_max": (1.0,),
            "reward_scale": (1.0,),
        },
        "QRDQN": {
            "risk_fraction": (0.75, 1.0),
            "huber_kappa": (1.0, 2.0),
        },
        "CQL": {
            "conservative_weight": (0.1, 0.25, 0.5, 1.0),
        },
        "LinUCB": {
            "alpha": (0.05, 0.1, 0.25, 0.5),
        },
        "PPO_RL": {
            "value_huber_delta": (1.0, 2.0, 5.0),
        },
        "SARSA": {
            "epsilon_end": (0.01, 0.02),
            "physical_epsilon_cap": (0.01,),
        },
        "DQN": {
            "epsilon_end": (0.02, 0.05),
            "physical_epsilon_cap": (0.01,),
        },
    }[utility_space.scheduler]
    fixed_contract = overrides.pop(
        "fixed_contract", utility_space.fixed_contract)
    choices = tuple(
        (name, tuple(overrides.get(name, values)))
        for name, values in utility_space.choices)
    return AlgorithmSearchSpace(
        policy_id=f"{utility_space.scheduler}_count",
        scheduler=utility_space.scheduler,
        objective="count",
        training_mode=utility_space.training_mode,
        choices=choices,
        fixed_contract=fixed_contract)


if tuple(space.scheduler for space in _UTILITY_SPACES) != TRAINABLE_ALGORITHMS:
    raise RuntimeError("utility search-space registry is incomplete")
_SPACES = tuple(
    variant
    for utility_space in _UTILITY_SPACES
    for variant in (_count_variant(utility_space), utility_space))


SEARCH_SPACES = MappingProxyType({space.policy_id: space for space in _SPACES})
if tuple(SEARCH_SPACES) != TRAINABLE_POLICY_IDS:
    raise RuntimeError("trainable AI search-space registry is incomplete")
if len({space.sha256 for space in _SPACES}) != len(_SPACES):
    raise RuntimeError("trainable AI search spaces are not independent")


def deterministic_trial_configs(policy_id: str, max_trials: int,
                                sampler_seed: int
                                ) -> Tuple[Mapping[str, object], ...]:
    """Sample a stable no-replacement prefix from one policy's full grid."""
    if policy_id not in SEARCH_SPACES:
        raise ValueError("unknown trainable policy")
    if (isinstance(max_trials, bool) or not isinstance(max_trials, int) or
            max_trials < 1 or isinstance(sampler_seed, bool) or
            not isinstance(sampler_seed, int) or sampler_seed < 0):
        raise ValueError("invalid trial count or sampler seed")
    space = SEARCH_SPACES[policy_id]
    combinations = [
        dict(zip((name for name, _values in space.choices), values))
        for values in itertools.product(
            *(values for _name, values in space.choices))]
    if max_trials > len(combinations):
        raise ValueError("max_trials exceeds the unique search space")
    random.Random(sampler_seed).shuffle(combinations)
    # A short random prefix can accidentally omit an entire boundary value
    # (for example every high learning-rate candidate).  Formal tuning uses a
    # coverage-first deterministic prefix: greedily cover unseen values in all
    # dimensions, then fill the remaining budget from the seeded shuffle.
    uncovered = {
        (name, _canonical_json(value))
        for name, values in space.choices for value in values}
    selected = []
    remaining = list(combinations)
    while remaining and len(selected) < max_trials and uncovered:
        best_index = max(
            range(len(remaining)),
            key=lambda index: sum(
                (name, _canonical_json(remaining[index][name])) in uncovered
                for name, _values in space.choices))
        candidate = remaining.pop(best_index)
        selected.append(candidate)
        uncovered.difference_update(
            (name, _canonical_json(candidate[name]))
            for name, _values in space.choices)
    if len(selected) < max_trials:
        selected.extend(remaining[:max_trials-len(selected)])
    trials = []
    for index, parameters in enumerate(selected):
        _validate_parameters(policy_id, parameters)
        payload = {
            "search_version": ALL_AI_SEARCH_VERSION,
            "policy_id": policy_id,
            "objective": space.objective,
            "training_mode": space.training_mode,
            "space_sha256": space.sha256,
            "trial_index": index,
            "parameters": parameters,
        }
        payload["trial_config_sha256"] = _sha256(payload)
        trials.append(MappingProxyType(payload))
    return tuple(trials)


def validate_trial_config(document: Mapping[str, object]) -> None:
    if not isinstance(document, Mapping):
        raise ValueError("trial configuration must be a mapping")
    required = {
        "search_version", "policy_id", "objective", "training_mode",
        "space_sha256", "trial_index", "parameters",
        "trial_config_sha256",
    }
    if set(document) != required:
        raise ValueError("trial configuration fields do not match contract")
    policy_id = document["policy_id"]
    if policy_id not in SEARCH_SPACES:
        raise ValueError("trial policy is not registered")
    space = SEARCH_SPACES[policy_id]
    if (document["search_version"] != ALL_AI_SEARCH_VERSION or
            document["objective"] != space.objective or
            document["training_mode"] != space.training_mode or
            document["space_sha256"] != space.sha256 or
            isinstance(document["trial_index"], bool) or
            not isinstance(document["trial_index"], int) or
            document["trial_index"] < 0):
        raise ValueError("trial configuration identity mismatch")
    parameters = document["parameters"]
    if (not isinstance(parameters, Mapping) or
            set(parameters) != {name for name, _values in space.choices}):
        raise ValueError("trial parameter fields do not match search space")
    for name, choices in space.choices:
        if _canonical_json(parameters[name]) not in {
                _canonical_json(value) for value in choices}:
            raise ValueError("trial parameter is outside registered choices")
    _validate_parameters(policy_id, parameters)
    unhashed = {key: value for key, value in document.items()
                if key != "trial_config_sha256"}
    if document["trial_config_sha256"] != _sha256(unhashed):
        raise ValueError("trial configuration hash mismatch")


def search_space_manifest() -> dict:
    spaces = {policy_id: space.canonical()
              for policy_id, space in SEARCH_SPACES.items()}
    document = {
        "version": ALL_AI_SEARCH_VERSION,
        "trainable_policy_ids": list(TRAINABLE_POLICY_IDS),
        "spaces": spaces,
        "space_sha256": {
            policy_id: SEARCH_SPACES[policy_id].sha256
            for policy_id in TRAINABLE_POLICY_IDS},
    }
    document["manifest_sha256"] = _sha256(document)
    return document


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def _candidate_validation_record(policy_id: str, candidate,
                                 validation_seeds: Tuple[int, ...]) -> dict:
    result = {
        "trial_id": None,
        "eligible": False,
        "exclusion_reasons": [],
        "validation_metric": None,
        "scenario_metrics": {},
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "same_seed_baseline_gate": None,
    }

    def reject(reason: str) -> None:
        if reason not in result["exclusion_reasons"]:
            result["exclusion_reasons"].append(reason)

    if not isinstance(candidate, Mapping):
        reject("candidate_not_mapping")
        return result
    trial_id = candidate.get("trial_id")
    if not isinstance(trial_id, str) or not trial_id:
        reject("invalid_trial_id")
    else:
        result["trial_id"] = trial_id
    trial = candidate.get("trial")
    try:
        validate_trial_config(trial)
    except (KeyError, TypeError, ValueError):
        reject("invalid_trial_config")
    else:
        if trial["policy_id"] != policy_id:
            reject("trial_policy_mismatch")
        result["trial_config_sha256"] = trial["trial_config_sha256"]
    checkpoint_path = candidate.get("checkpoint_path")
    checkpoint_sha256 = candidate.get("checkpoint_sha256")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        reject("invalid_checkpoint_path")
    else:
        path = Path(checkpoint_path).resolve()
        result["checkpoint_path"] = str(path)
        if not path.is_file():
            reject("checkpoint_missing")
        elif (not _is_sha256(checkpoint_sha256) or
              _file_sha256(path) != checkpoint_sha256):
            reject("checkpoint_hash_mismatch")
        else:
            result["checkpoint_sha256"] = checkpoint_sha256
    if candidate.get("checkpoint_contract_valid") is not True:
        reject("checkpoint_contract_invalid")

    episodes = candidate.get("validation_episodes")
    expected_keys = {
        (scenario, seed) for scenario in FORMAL_SCENARIOS
        for seed in validation_seeds}
    observed = {}
    metric_name = ("completed_tasks" if
                   SEARCH_SPACES[policy_id].objective == "count" else
                   "utility_score_deadline_v2")
    if not isinstance(episodes, list):
        reject("validation_episodes_missing")
        episodes = []
    for row in episodes:
        if not isinstance(row, Mapping):
            reject("validation_episode_not_mapping")
            continue
        scenario, seed = row.get("scenario"), row.get("seed")
        key = (scenario, seed)
        if key in observed:
            reject("duplicate_validation_episode")
            continue
        observed[key] = row
        if key not in expected_keys:
            reject("unexpected_validation_key")
        if (row.get("partition") != "validation" or
                row.get("objective") != SEARCH_SPACES[policy_id].objective):
            reject("validation_identity_mismatch")
        duration = row.get("duration_seconds")
        current_time = row.get("current_time")
        if (isinstance(duration, bool) or
                not isinstance(duration, (int, float)) or
                not math.isfinite(duration) or duration != 1800.0 or
                isinstance(current_time, bool) or
                not isinstance(current_time, (int, float)) or
                not math.isfinite(current_time) or current_time != 1800.0 or
                row.get("fixed_horizon") is not True or
                row.get("horizon_finalized") is not True or
                row.get("termination_reason") != "episode_horizon"):
            reject("incomplete_validation_horizon")
        if (row.get("schema_version") != 3 or
                row.get("runtime_mode") != "headless_webots_logic" or
                row.get("dynamics_version") != HEADLESS_DYNAMICS_VERSION or
                row.get("physics_fidelity") != "business_logic_only" or
                row.get("joint_runtime") is not True or
                row.get("route_planner") != "rolling_joint_grid"):
            reject("validation_runtime_mismatch")
        if (row.get("rollout_policy") !=
                SEARCH_SPACES[policy_id].scheduler or
                row.get("metric_owner") !=
                SEARCH_SPACES[policy_id].scheduler):
            reject("validation_metric_owner_mismatch")
        if (row.get("pure_policy_eligible") is not True or
                row.get("fallback_scheduler_commits") != 0 or
                row.get("invalid_scheduler_outputs") != 0 or
                row.get("scheduler_timeout_count") != 0):
            reject("validation_purity_failure")
        if (row.get("checkpoint_sha256") != checkpoint_sha256 or
                not _is_sha256(row.get("task_manifest_sha256"))):
            reject("validation_provenance_mismatch")
        metric = row.get(metric_name)
        if metric_name == "completed_tasks":
            if (isinstance(metric, bool) or not isinstance(metric, int) or
                    metric < 0):
                reject("invalid_validation_metric")
        elif (isinstance(metric, bool) or
              not isinstance(metric, (int, float)) or
              not math.isfinite(metric)):
            reject("invalid_validation_metric")
    if set(observed) != expected_keys:
        reject("incomplete_balanced_validation_set")
    if not result["exclusion_reasons"]:
        scenario_metrics = {}
        for scenario in FORMAL_SCENARIOS:
            values = [float(observed[(scenario, seed)][metric_name])
                      for seed in validation_seeds]
            scenario_metrics[scenario] = sum(values)/len(values)
        result["scenario_metrics"] = scenario_metrics
        result["validation_metric"] = (
            sum(scenario_metrics.values())/len(FORMAL_SCENARIOS))
        if SEARCH_SPACES[policy_id].scheduler in {
                "LearnedHungarian", "GraphImitation"}:
            comparisons = {}
            for scenario in FORMAL_SCENARIOS:
                scenario_rows = [
                    observed[(scenario, seed)] for seed in validation_seeds]
                try:
                    nearest = sum(float(
                        row["same_seed_references"]["NearestNeighbour"]
                        [metric_name]) for row in scenario_rows) / len(
                            scenario_rows)
                    expert = sum(float(
                        row["same_seed_references"]["ObjectiveHungarian"]
                        [metric_name]) for row in scenario_rows) / len(
                            scenario_rows)
                except (KeyError, TypeError, ValueError):
                    reject("same_seed_baseline_missing")
                    break
                model_value = scenario_metrics[scenario]
                nearest_floor = nearest - (
                    1.0-SUPERVISED_MIN_NEAREST_NEIGHBOUR_RATIO)*abs(nearest)
                expert_floor = expert - (
                    1.0-SUPERVISED_MIN_OBJECTIVE_EXPERT_RATIO)*abs(expert)
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
            if not result["exclusion_reasons"]:
                result["same_seed_baseline_gate"] = {
                    "eligible": all(
                        row["eligible"] for row in comparisons.values()),
                    "scenarios": comparisons,
                }
                if not result["same_seed_baseline_gate"]["eligible"]:
                    reject("same_seed_baseline_regression")
        else:
            result["same_seed_baseline_gate"] = {
                "eligible": True, "required": False}
    if not result["exclusion_reasons"]:
        result["eligible"] = True
    return result


def select_validation_checkpoint(
        policy_id: str, candidates,
        validation_seeds: Tuple[int, ...] = FORMAL_VALIDATION_SEEDS) -> dict:
    """Select only by the registered formal metric on unseen 1800s runs."""
    if policy_id not in SEARCH_SPACES:
        raise ValueError("unknown trainable policy")
    if (not isinstance(validation_seeds, tuple) or not validation_seeds or
            any(isinstance(seed, bool) or not isinstance(seed, int) or
                seed < 0 for seed in validation_seeds) or
            len(validation_seeds) != len(set(validation_seeds))):
        raise ValueError("validation seeds must be unique nonnegative integers")
    if validation_seeds != FORMAL_VALIDATION_SEEDS:
        raise ValueError("strict selection requires pre-registered validation seeds")
    try:
        candidate_rows = list(candidates)
    except TypeError as exc:
        raise ValueError("candidates must be iterable") from exc
    if not candidate_rows:
        raise ValueError("at least one candidate is required")
    records = [
        _candidate_validation_record(policy_id, candidate, validation_seeds)
        for candidate in candidate_rows]
    seen_ids, seen_paths = set(), set()
    for record in records:
        trial_id, checkpoint_path = (
            record.get("trial_id"), record.get("checkpoint_path"))
        if trial_id in seen_ids:
            record["eligible"] = False
            record["exclusion_reasons"].append("duplicate_trial_id")
        elif trial_id is not None:
            seen_ids.add(trial_id)
        if checkpoint_path in seen_paths:
            record["eligible"] = False
            record["exclusion_reasons"].append(
                "shared_checkpoint_path")
        elif checkpoint_path is not None:
            seen_paths.add(checkpoint_path)
    eligible = [record for record in records if record["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -row["validation_metric"], row["trial_config_sha256"],
            row["trial_id"]))
    objective = SEARCH_SPACES[policy_id].objective
    metric_name = ("completed_tasks" if objective == "count" else
                   "utility_score_deadline_v2")
    winner = ranked[0] if ranked else None
    report = {
        "version": VALIDATION_SELECTOR_VERSION,
        "policy_id": policy_id,
        "objective": objective,
        "selection_metric": metric_name,
        "duration_seconds": 1800.0,
        "scenarios": list(FORMAL_SCENARIOS),
        "validation_seeds": list(validation_seeds),
        "candidate_count": len(records),
        "eligible_candidate_count": len(ranked),
        "selection_status": (
            "winner" if winner is not None else
            "research_candidate_no_eligible_checkpoint"),
        "winner_trial_id": winner["trial_id"] if winner else None,
        "winner_checkpoint_path": (
            winner["checkpoint_path"] if winner else None),
        "winner_checkpoint_sha256": (
            winner["checkpoint_sha256"] if winner else None),
        "winner_validation_metric": (
            winner["validation_metric"] if winner else None),
        "ranking": [row["trial_id"] for row in ranked],
        "records": records,
        "ignored_diagnostics": ["training_return", "training_loss"],
    }
    report["report_sha256"] = _sha256(report)
    return report


def build_standalone_resume_plan(policy_id: str, trial_id: str,
                                 balanced_rounds: int):
    """Return the ordered atomic plan through one validation boundary."""
    from atomic_training_resume import ResumeUnit

    if policy_id not in SEARCH_SPACES:
        raise ValueError("unknown trainable policy")
    if (not isinstance(trial_id, str) or not trial_id or
            isinstance(balanced_rounds, bool) or
            not isinstance(balanced_rounds, int) or
            balanced_rounds < VALIDATION_INTERVAL_ROUNDS or
            balanced_rounds > MAX_STANDALONE_BALANCED_ROUNDS or
            balanced_rounds % VALIDATION_INTERVAL_ROUNDS):
        raise ValueError("balanced rounds must end on a validation boundary")
    units = []
    order = 0
    for round_index in range(1, balanced_rounds+1):
        seed = STANDALONE_TRAINING_SEEDS[round_index-1]
        rung_index = (round_index-1)//VALIDATION_INTERVAL_ROUNDS
        for scenario in FORMAL_SCENARIOS:
            units.append(ResumeUnit(
                order, "training_episode", policy_id, trial_id,
                rung_index, round_index, scenario, seed))
            order += 1
        units.append(ResumeUnit(
            order, "model_update", policy_id, trial_id,
            rung_index, round_index))
        order += 1
        if round_index % VALIDATION_INTERVAL_ROUNDS == 0:
            units.append(ResumeUnit(
                order, "validation_batch", policy_id, trial_id,
                rung_index, round_index))
            order += 1
            units.append(ResumeUnit(
                order, "rung", policy_id, trial_id,
                rung_index, round_index))
            order += 1
    units.append(ResumeUnit(
        order, "trial", policy_id, trial_id,
        (balanced_rounds//VALIDATION_INTERVAL_ROUNDS)-1,
        balanced_rounds))
    return tuple(units)


def _valid_standalone_training_record(policy_id: str, row: Mapping,
                                      round_index: int, scenario: str,
                                      seed: int) -> bool:
    if not isinstance(row, Mapping):
        return False
    duration = row.get("duration_seconds")
    current_time = row.get("current_time")
    metrics = row.get("metrics")
    runtime = (metrics.get("runtime")
               if isinstance(metrics, Mapping) else None)
    return bool(
        row.get("policy_id") == policy_id and
        row.get("scheduler") == SEARCH_SPACES[policy_id].scheduler and
        row.get("partition") == "training" and
        row.get("objective") == SEARCH_SPACES[policy_id].objective and
        row.get("training_mode") == SEARCH_SPACES[policy_id].training_mode and
        row.get("round_index") == round_index and
        row.get("scenario") == scenario and row.get("seed") == seed and
        not isinstance(duration, bool) and
        isinstance(duration, (int, float)) and math.isfinite(duration) and
        duration == 1800.0 and not isinstance(current_time, bool) and
        isinstance(current_time, (int, float)) and
        math.isfinite(current_time) and current_time == 1800.0 and
        row.get("fixed_horizon") is True and
        row.get("horizon_finalized") is True and
        row.get("termination_reason") == "episode_horizon" and
        isinstance(runtime, Mapping) and
        runtime.get("runtime_mode") == "headless_webots_logic" and
        runtime.get("dynamics_version") == HEADLESS_DYNAMICS_VERSION and
        runtime.get("physics_fidelity") == "business_logic_only" and
        runtime.get("joint_runtime") is True and
        runtime.get("route_planner") == "rolling_joint_grid" and
        runtime.get("fixed_horizon") is True and
        runtime.get("horizon_finalized") is True and
        runtime.get("termination_reason") == "episode_horizon" and
        runtime.get("current_time") == 1800.0 and
        row.get("status") == "complete" and
        row.get("model_round_trip_valid") is True and
        _is_sha256(row.get("task_manifest_sha256")) and
        _is_sha256(row.get("output_checkpoint_sha256")) and
        _is_sha256(row.get("output_model_identity_sha256")) and
        (SEARCH_SPACES[policy_id].scheduler != "CQL" or (
            row.get("offline_dataset_frozen") is True and
            row.get("policy_generated_transitions") == 0)))


def standalone_training_readiness(policy_id: str, training_records,
                                  validation_history) -> dict:
    """Decide whether a frozen standalone winner may enter Webots tuning."""
    if policy_id not in SEARCH_SPACES:
        raise ValueError("unknown trainable policy")
    try:
        rows = list(training_records)
        validations = list(validation_history)
    except TypeError as exc:
        raise ValueError("training and validation records must be iterable") from exc
    indexed = {}
    invalid_records = []
    for row in rows:
        if not isinstance(row, Mapping):
            invalid_records.append("training_record_not_mapping")
            continue
        key = (row.get("round_index"), row.get("scenario"), row.get("seed"))
        if key in indexed:
            invalid_records.append("duplicate_training_record")
        indexed[key] = row
    completed_rounds = 0
    for round_index, seed in enumerate(
            STANDALONE_TRAINING_SEEDS, start=1):
        expected = []
        missing = False
        for scenario in FORMAL_SCENARIOS:
            key = (round_index, scenario, seed)
            row = indexed.get(key)
            if row is None:
                missing = True
                break
            expected.append((scenario, row))
        if missing:
            break
        if not all(_valid_standalone_training_record(
                policy_id, row, round_index, scenario, seed)
                for scenario, row in expected):
            invalid_records.append("invalid_training_record")
            break
        completed_rounds = round_index
    expected_prefix_keys = {
        (round_index, scenario, STANDALONE_TRAINING_SEEDS[round_index-1])
        for round_index in range(1, completed_rounds+1)
        for scenario in FORMAL_SCENARIOS}
    unexpected = [key for key in indexed
                  if key not in expected_prefix_keys and
                  isinstance(key[0], int) and key[0] <= completed_rounds]
    if unexpected:
        invalid_records.append("unexpected_training_record")

    validation_by_round = {}
    invalid_validations = []
    objective = SEARCH_SPACES[policy_id].objective
    metric_name = ("completed_tasks" if objective == "count" else
                   "utility_score_deadline_v2")
    for row in validations:
        if not isinstance(row, Mapping):
            invalid_validations.append("validation_record_not_mapping")
            continue
        round_index = row.get("round_index")
        if round_index in validation_by_round:
            invalid_validations.append("duplicate_validation_round")
        validation_by_round[round_index] = row
        metric = row.get("validation_metric")
        parameter_change = row.get("parameter_change", {})
        if (isinstance(round_index, bool) or not isinstance(round_index, int) or
                round_index < VALIDATION_INTERVAL_ROUNDS or
                round_index > MAX_STANDALONE_BALANCED_ROUNDS or
                round_index % VALIDATION_INTERVAL_ROUNDS or
                row.get("policy_id") != policy_id or
                row.get("objective") != objective or
                row.get("selection_metric") != metric_name or
                row.get("eligible") is not True or
                row.get("duration_seconds") != 1800.0 or
                tuple(row.get("validation_seeds", ())) !=
                FORMAL_VALIDATION_SEEDS or
                not _is_sha256(row.get("checkpoint_sha256")) or
                not _is_sha256(row.get("checkpoint_identity_sha256")) or
                not _is_sha256(row.get("selector_report_sha256")) or
                isinstance(metric, bool) or
                not isinstance(metric, (int, float)) or
                not math.isfinite(metric) or
                not isinstance(parameter_change, Mapping) or
                parameter_change.get("schema_version") != 1 or
                parameter_change.get("reference_round") !=
                    max(0, round_index-VALIDATION_INTERVAL_ROUNDS) or
                isinstance(parameter_change.get("relative_l2_change"), bool) or
                not isinstance(parameter_change.get("relative_l2_change"),
                               (int, float)) or
                not math.isfinite(parameter_change["relative_l2_change"]) or
                parameter_change["relative_l2_change"] < 0 or
                isinstance(parameter_change.get("parameter_count"), bool) or
                not isinstance(parameter_change.get("parameter_count"), int) or
                parameter_change["parameter_count"] < 1):
            invalid_validations.append("invalid_validation_record")
    required_validation_rounds = tuple(range(
        VALIDATION_INTERVAL_ROUNDS,
        completed_rounds+1,
        VALIDATION_INTERVAL_ROUNDS))
    missing_validation_rounds = [
        round_index for round_index in required_validation_rounds
        if round_index not in validation_by_round]
    extra_validation_rounds = [
        round_index for round_index in validation_by_round
        if not isinstance(round_index, int) or
        round_index not in required_validation_rounds]
    if extra_validation_rounds:
        invalid_validations.append("unexpected_validation_round")

    ordered_validations = [validation_by_round[round_index]
                           for round_index in required_validation_rounds
                           if round_index in validation_by_round]
    # Never promote an early lucky checkpoint after claiming that the model
    # received sufficient training.  Selection is restricted to checkpoints
    # at/after the 100-round floor and to the final parameter-convergence
    # window; older validation results remain diagnostics only.
    sufficiently_trained = [
        row for row in ordered_validations
        if row["round_index"] >= MIN_STANDALONE_BALANCED_ROUNDS]
    selection_window = sufficiently_trained[
        -PARAMETER_CONVERGENCE_PATIENCE_CHECKS:]
    best = (max(selection_window,
                key=lambda row: (row["validation_metric"],
                                 -row["round_index"]))
            if selection_window and not invalid_validations else None)
    stable = False
    improvement = None
    if (not invalid_validations and
            len(ordered_validations) > CONVERGENCE_PATIENCE_CHECKS):
        earlier = ordered_validations[:-CONVERGENCE_PATIENCE_CHECKS]
        recent = ordered_validations[-CONVERGENCE_PATIENCE_CHECKS:]
        reference = max(row["validation_metric"] for row in earlier)
        recent_best = max(row["validation_metric"] for row in recent)
        improvement = recent_best-reference
        stable = improvement < CONVERGENCE_MIN_IMPROVEMENT[objective]

    parameter_converged = False
    recent_parameter_changes = []
    if (not invalid_validations and
            len(ordered_validations) >=
                PARAMETER_CONVERGENCE_PATIENCE_CHECKS):
        recent_parameter_changes = [
            float(row["parameter_change"]["relative_l2_change"])
            for row in ordered_validations[
                -PARAMETER_CONVERGENCE_PATIENCE_CHECKS:]
        ]
        parameter_converged = all(
            value <= PARAMETER_RELATIVE_CHANGE_THRESHOLD
            for value in recent_parameter_changes)

    if invalid_records or invalid_validations:
        status = "invalid_training_evidence"
    elif missing_validation_rounds:
        status = "incomplete_validation"
    elif completed_rounds < MIN_STANDALONE_BALANCED_ROUNDS:
        status = "insufficient_training"
    elif stable and parameter_converged:
        status = "qualified_for_webots_fine_tune"
    elif completed_rounds >= MAX_STANDALONE_BALANCED_ROUNDS:
        status = "research_candidate_not_converged"
    else:
        status = "continue_training"
    report = {
        "policy_id": policy_id,
        "objective": objective,
        "training_mode": SEARCH_SPACES[policy_id].training_mode,
        "status": status,
        "webots_fine_tune_eligible": (
            status == "qualified_for_webots_fine_tune"),
        "completed_balanced_rounds": completed_rounds,
        "completed_episode_count": completed_rounds*len(FORMAL_SCENARIOS),
        "scenario_episode_counts": {
            scenario: completed_rounds for scenario in FORMAL_SCENARIOS},
        "minimum_balanced_rounds": MIN_STANDALONE_BALANCED_ROUNDS,
        "maximum_balanced_rounds": MAX_STANDALONE_BALANCED_ROUNDS,
        "duration_seconds_per_episode": 1800.0,
        "training_seed_count": completed_rounds,
        "training_seeds_unique": True,
        "training_validation_seed_overlap": bool(
            set(STANDALONE_TRAINING_SEEDS[:completed_rounds]) &
            set(FORMAL_VALIDATION_SEEDS)),
        "validation_interval_rounds": VALIDATION_INTERVAL_ROUNDS,
        "validation_checks": len(ordered_validations),
        "missing_validation_rounds": missing_validation_rounds,
        "invalid_training_evidence": sorted(set(invalid_records)),
        "invalid_validation_evidence": sorted(set(invalid_validations)),
        "convergence_patience_checks": CONVERGENCE_PATIENCE_CHECKS,
        "minimum_improvement": CONVERGENCE_MIN_IMPROVEMENT[objective],
        "recent_improvement": improvement,
        "validation_plateau": stable,
        "parameter_convergence_patience_checks":
            PARAMETER_CONVERGENCE_PATIENCE_CHECKS,
        "parameter_relative_change_threshold":
            PARAMETER_RELATIVE_CHANGE_THRESHOLD,
        "recent_parameter_relative_changes": recent_parameter_changes,
        "parameter_converged": parameter_converged,
        "best_validation_round": best["round_index"] if best else None,
        "best_validation_metric": (
            float(best["validation_metric"]) if best else None),
        "best_checkpoint_sha256": (
            best["checkpoint_sha256"] if best else None),
        "best_checkpoint_identity_sha256": (
            best["checkpoint_identity_sha256"] if best else None),
        "checkpoint_selection_window_rounds": [
            row["round_index"] for row in selection_window],
        "early_validation_checkpoints_promotable": False,
    }
    report["report_sha256"] = _sha256(report)
    return report
