"""Deterministic contracts and execution primitives for dual-objective tuning."""

from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from dual_objective import (
    CountRewardProfile, UtilityScoreConfig, UtilityV2RewardProfile,
    count_evaluation, paired_count_delta,
    physical_episode_reward_attribution, utility_v2_evaluation,
)
from graph_ppo_scheduler import (
    GraphPPOConfig, GraphPPOModel, GraphRolloutStep, graph_policy_inputs,
)
from headless_training_runtime import (
    HEADLESS_DYNAMICS_VERSION, HeadlessRuntimeConfig,
    summarize_runtime_telemetry,
)
from rl_environment import RLEnvironmentConfig, SchedulingEnvironment
from schedulers import create_scheduler
from task_generator import (
    generate_task_manifest, task_manifest_sha256, write_task_manifest,
)
from training_scenarios import factory_scenario, formal_scenario_config


AUTO_TUNING_VERSION = "dual-auto-optimization-v1"
SCENARIO_IDS = ("A", "B", "C")


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value) -> bool:
    return bool(
        isinstance(value, str) and len(value) == 64 and
        all(character in "0123456789abcdef" for character in value))


def _strict_integer(value, name: str, *, minimum=None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _strict_float(value, name: str, *, positive=False,
                  nonnegative=False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _strict_unique_seeds(values: Iterable[int], name: str) -> Tuple[int, ...]:
    try:
        seeds = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} seeds must be iterable") from exc
    if not seeds:
        raise ValueError(f"{name} seeds must not be empty")
    canonical = tuple(_strict_integer(seed, f"{name} seed") for seed in seeds)
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"{name} seeds must be unique")
    return canonical


@dataclass(frozen=True)
class SeedPartitions:
    """Pre-registered train/validation split with Webots training subset."""

    training: Tuple[int, ...]
    validation: Tuple[int, ...]
    webots_finetune: Tuple[int, ...]

    def __post_init__(self):
        training = _strict_unique_seeds(self.training, "training")
        validation = _strict_unique_seeds(self.validation, "validation")
        webots = _strict_unique_seeds(
            self.webots_finetune, "Webots fine-tune")
        if set(training) & set(validation):
            raise ValueError("training and validation seeds must be disjoint")
        if not set(webots).issubset(training):
            raise ValueError("Webots fine-tune seeds must be training seeds")
        if any(not 21000 <= seed <= 21999 for seed in training):
            raise ValueError("training seeds must be within 21000-21999")
        if any(not 31000 <= seed <= 31999 for seed in validation):
            raise ValueError("validation seeds must be within 31000-31999")
        object.__setattr__(self, "training", training)
        object.__setattr__(self, "validation", validation)
        object.__setattr__(self, "webots_finetune", webots)

    def canonical(self) -> dict:
        return {
            "training": list(self.training),
            "validation": list(self.validation),
            "webots_finetune": list(self.webots_finetune),
            "final_test_api_exposed": False,
        }


@dataclass(frozen=True)
class GraphPPOSearchSpace:
    hidden_sizes: Tuple[int, ...] = (24, 48, 64)
    learning_rates: Tuple[float, ...] = (0.0002, 0.0005, 0.001)
    clip_epsilons: Tuple[float, ...] = (0.1, 0.2)
    gae_lambdas: Tuple[float, ...] = (0.9, 0.95)
    entropy_coefficients: Tuple[float, ...] = (0.005, 0.01)
    update_epochs: Tuple[int, ...] = (2, 4)

    def __post_init__(self):
        integer_fields = ("hidden_sizes", "update_epochs")
        float_fields = (
            "learning_rates", "clip_epsilons", "gae_lambdas",
            "entropy_coefficients",
        )
        for name in integer_fields:
            raw = tuple(getattr(self, name))
            if not raw:
                raise ValueError(f"{name} must not be empty")
            values = tuple(sorted({_strict_integer(
                value, name, minimum=1) for value in raw}))
            object.__setattr__(self, name, values)
        for name in float_fields:
            raw = tuple(getattr(self, name))
            if not raw:
                raise ValueError(f"{name} must not be empty")
            values = tuple(sorted({_strict_float(
                value, name, positive=(name != "entropy_coefficients"),
                nonnegative=(name == "entropy_coefficients"))
                for value in raw}))
            if name in {"clip_epsilons", "gae_lambdas"} and any(
                    value > 1.0 for value in values):
                raise ValueError(f"{name} values must not exceed 1")
            object.__setattr__(self, name, values)

    def canonical(self) -> dict:
        return {
            "hidden_sizes": list(self.hidden_sizes),
            "learning_rates": list(self.learning_rates),
            "clip_epsilons": list(self.clip_epsilons),
            "gae_lambdas": list(self.gae_lambdas),
            "entropy_coefficients": list(self.entropy_coefficients),
            "update_epochs": list(self.update_epochs),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.canonical())


@dataclass(frozen=True)
class SuccessiveHalvingPlan:
    training_episode_budgets: Tuple[int, ...]
    validation_seed_budgets: Tuple[int, ...]
    reduction_factor: int = 3

    def __post_init__(self):
        training = tuple(_strict_integer(
            value, "training episode budget", minimum=1)
            for value in self.training_episode_budgets)
        validation = tuple(_strict_integer(
            value, "validation seed budget", minimum=1)
            for value in self.validation_seed_budgets)
        reduction = _strict_integer(
            self.reduction_factor, "reduction_factor", minimum=2)
        if not training or len(training) != len(validation):
            raise ValueError("training and validation rung budgets must align")
        if any(right <= left for left, right in zip(training, training[1:])):
            raise ValueError("training episode budgets must strictly increase")
        if any(right < left for left, right in zip(validation, validation[1:])):
            raise ValueError("validation seed budgets must not decrease")
        object.__setattr__(self, "training_episode_budgets", training)
        object.__setattr__(self, "validation_seed_budgets", validation)
        object.__setattr__(self, "reduction_factor", reduction)

    def canonical(self) -> dict:
        return {
            "training_episode_budgets": list(self.training_episode_budgets),
            "validation_seed_budgets": list(self.validation_seed_budgets),
            "reduction_factor": self.reduction_factor,
        }


@dataclass(frozen=True)
class AutoTuningContract:
    mode: str
    duration_seconds: float
    max_trials: int
    sampler_seed: int
    partitions: SeedPartitions
    search_space: GraphPPOSearchSpace
    halving_plan: SuccessiveHalvingPlan

    def __post_init__(self):
        if self.mode not in {"smoke", "formal"}:
            raise ValueError("mode must be smoke or formal")
        duration = _strict_float(
            self.duration_seconds, "duration_seconds", positive=True)
        max_trials = _strict_integer(self.max_trials, "max_trials", minimum=1)
        sampler_seed = _strict_integer(self.sampler_seed, "sampler_seed")
        if not isinstance(self.partitions, SeedPartitions):
            raise ValueError("partitions must be SeedPartitions")
        if not isinstance(self.search_space, GraphPPOSearchSpace):
            raise ValueError("search_space must be GraphPPOSearchSpace")
        if not isinstance(self.halving_plan, SuccessiveHalvingPlan):
            raise ValueError("halving_plan must be SuccessiveHalvingPlan")
        all_training_keys = len(SCENARIO_IDS) * len(
            self.partitions.training)
        if self.halving_plan.training_episode_budgets[-1] != all_training_keys:
            raise ValueError(
                "final training budget must cover every A/B/C training key")
        if self.halving_plan.validation_seed_budgets[-1] != len(
                self.partitions.validation):
            raise ValueError(
                "final validation budget must cover every validation seed")
        if max_trials > _search_space_size(self.search_space):
            raise ValueError("max_trials exceeds the unique search space")
        if self.mode == "formal":
            if duration != 1800.0:
                raise ValueError("formal tuning requires exactly 1800 seconds")
            if len(self.partitions.webots_finetune) != 5:
                raise ValueError(
                    "formal Webots fine-tune requires five seeds per scenario")
        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(self, "max_trials", max_trials)
        object.__setattr__(self, "sampler_seed", sampler_seed)

    def canonical(self) -> dict:
        return {
            "version": AUTO_TUNING_VERSION,
            "mode": self.mode,
            "duration_seconds": self.duration_seconds,
            "max_trials": self.max_trials,
            "sampler_seed": self.sampler_seed,
            "partitions": self.partitions.canonical(),
            "search_space": self.search_space.canonical(),
            "search_space_sha256": self.search_space.sha256,
            "halving_plan": self.halving_plan.canonical(),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.canonical())


def _search_space_size(space: GraphPPOSearchSpace) -> int:
    return math.prod((
        len(space.hidden_sizes), len(space.learning_rates),
        len(space.clip_epsilons), len(space.gae_lambdas),
        len(space.entropy_coefficients), len(space.update_epochs),
    ))


def deterministic_trial_configs(
        space: GraphPPOSearchSpace, max_trials: int,
        sampler_seed: int) -> Tuple[Mapping[str, object], ...]:
    """Return a stable sample without replacement from the full grid."""
    if not isinstance(space, GraphPPOSearchSpace):
        raise ValueError("space must be GraphPPOSearchSpace")
    count = _strict_integer(max_trials, "max_trials", minimum=1)
    seed = _strict_integer(sampler_seed, "sampler_seed")
    combinations = [
        {
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "clip_epsilon": clip_epsilon,
            "gae_lambda": gae_lambda,
            "entropy_coefficient": entropy,
            "update_epochs": epochs,
            "gamma": 1.0,
        }
        for (hidden_size, learning_rate, clip_epsilon, gae_lambda,
             entropy, epochs) in itertools.product(
                space.hidden_sizes, space.learning_rates,
                space.clip_epsilons, space.gae_lambdas,
                space.entropy_coefficients, space.update_epochs)
    ]
    if count > len(combinations):
        raise ValueError("max_trials exceeds the unique search space")
    random.Random(seed).shuffle(combinations)
    result = []
    for configuration in combinations[:count]:
        canonical = dict(configuration)
        canonical["trial_config_sha256"] = _canonical_sha256(configuration)
        result.append(MappingProxyType(canonical))
    return tuple(result)


def scenario_seed_keys(seeds: Sequence[int]) -> Tuple[Tuple[str, int], ...]:
    canonical = _strict_unique_seeds(seeds, "scenario")
    return tuple(
        (scenario_id, seed)
        for seed in canonical for scenario_id in SCENARIO_IDS)


def curriculum_episode_keys(
        training_seeds: Sequence[int]) -> Tuple[Tuple[str, int], ...]:
    """Order all A/B/C keys as 20% A, 30% A/B, then mixed A/B/C."""
    seeds = _strict_unique_seeds(training_seeds, "training curriculum")
    queues = {
        scenario_id: [(scenario_id, seed) for seed in seeds]
        for scenario_id in SCENARIO_IDS
    }
    total = len(SCENARIO_IDS) * len(seeds)
    stage_one_count = min(len(seeds), math.ceil(total * 0.20))
    stage_two_count = min(
        len(queues["A"]) + len(queues["B"]) - stage_one_count,
        math.ceil(total * 0.30))
    result = queues["A"][:stage_one_count]
    offsets = {"A": stage_one_count, "B": 0, "C": 0}
    stage_two = []
    while len(stage_two) < stage_two_count:
        progressed = False
        for scenario_id in ("A", "B"):
            offset = offsets[scenario_id]
            if offset < len(queues[scenario_id]):
                stage_two.append(queues[scenario_id][offset])
                offsets[scenario_id] += 1
                progressed = True
                if len(stage_two) == stage_two_count:
                    break
        if not progressed:
            break
    result.extend(stage_two)
    while len(result) < total:
        progressed = False
        for scenario_id in SCENARIO_IDS:
            offset = offsets[scenario_id]
            if offset < len(queues[scenario_id]):
                result.append(queues[scenario_id][offset])
                offsets[scenario_id] += 1
                progressed = True
        if not progressed:
            raise RuntimeError("curriculum construction did not make progress")
    if len(result) != total or len(set(result)) != total:
        raise RuntimeError("curriculum does not cover every key exactly once")
    return tuple(result)


def webots_finetune_keys(
        partitions: SeedPartitions) -> Tuple[Tuple[str, int], ...]:
    if not isinstance(partitions, SeedPartitions):
        raise ValueError("partitions must be SeedPartitions")
    return scenario_seed_keys(partitions.webots_finetune)


def graph_ppo_config_from_trial(configuration: Mapping[str, object]
                                ) -> GraphPPOConfig:
    """Build the only GraphPPO configuration accepted by this workflow."""
    if not isinstance(configuration, Mapping):
        raise ValueError("trial configuration must be a mapping")
    required = {
        "hidden_size", "learning_rate", "clip_epsilon", "gae_lambda",
        "entropy_coefficient", "update_epochs", "gamma",
        "trial_config_sha256",
    }
    if set(configuration) != required:
        raise ValueError("trial configuration fields do not match contract")
    unhashed = {
        key: configuration[key]
        for key in required if key != "trial_config_sha256"
    }
    if configuration["trial_config_sha256"] != _canonical_sha256(unhashed):
        raise ValueError("trial configuration hash mismatch")
    if configuration["gamma"] != 1.0:
        raise ValueError("dual-objective GraphPPO requires gamma=1")
    return GraphPPOConfig(
        hidden_size=configuration["hidden_size"],
        learning_rate=configuration["learning_rate"],
        clip_epsilon=configuration["clip_epsilon"],
        gae_lambda=configuration["gae_lambda"],
        entropy_coefficient=configuration["entropy_coefficient"],
        update_epochs=configuration["update_epochs"],
        gamma=configuration["gamma"],
    )


def _objective_reward_profile(objective: str, duration_seconds: float,
                              utility_profile=None):
    duration = _strict_float(
        duration_seconds, "duration_seconds", positive=True)
    if objective == "count":
        if utility_profile is not None:
            raise ValueError("Count training does not accept a Utility profile")
        return CountRewardProfile(horizon_seconds=duration)
    if objective != "utility_v2":
        raise ValueError("objective must be count or utility_v2")
    if not isinstance(utility_profile, UtilityV2RewardProfile):
        raise ValueError("Utility training requires UtilityV2RewardProfile")
    if utility_profile.horizon_seconds != duration:
        raise ValueError("Utility reward profile horizon differs from training")
    if utility_profile.gamma != 1.0:
        raise ValueError("Utility reward profile requires gamma=1")
    return utility_profile


def _canonical_episode_manifest(scenario_id: str, seed: int,
                                duration_seconds: float,
                                supplied_manifest=None) -> dict:
    generated = generate_task_manifest(
        scenario_id, formal_scenario_config(scenario_id), seed,
        duration_seconds=duration_seconds)
    if supplied_manifest is None:
        return generated
    # ``factory_scenario`` performs full canonical validation.  Requiring the
    # hash to equal a fresh deterministic generation also prevents a caller
    # from tuning on a modified task stream with otherwise valid metadata.
    supplied_hash = supplied_manifest.get("manifest_sha256") if isinstance(
        supplied_manifest, dict) else None
    if supplied_hash != generated["manifest_sha256"]:
        raise ValueError("supplied training manifest is not canonical")
    return supplied_manifest


def _model_parameter_sha256(model: GraphPPOModel) -> str:
    digest = hashlib.sha256()
    for name in sorted(model.network.params):
        value = np.asarray(model.network.params[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def train_objective_graph_ppo(
        model: GraphPPOModel, objective: str,
        episode_keys: Sequence[Tuple[str, int]], *,
        duration_seconds: float,
        utility_profile: Optional[UtilityV2RewardProfile] = None,
        manifests: Optional[Mapping[Tuple[str, int], dict]] = None,
        max_steps_per_episode: Optional[int] = None) -> dict:
    """Incrementally train GraphPPO on canonical objective-aware episodes.

    The function accepts training keys only; validation and final-test seeds
    have no entry point here.  Callers are expected to pass a curriculum
    prefix from :func:`curriculum_episode_keys`.
    """
    if not isinstance(model, GraphPPOModel):
        raise ValueError("model must be GraphPPOModel")
    if model.config.gamma != 1.0:
        raise ValueError("dual-objective GraphPPO requires gamma=1")
    duration = _strict_float(
        duration_seconds, "duration_seconds", positive=True)
    profile = _objective_reward_profile(
        objective, duration, utility_profile)
    try:
        keys = tuple(episode_keys)
    except TypeError as exc:
        raise ValueError("episode_keys must be iterable") from exc
    if not keys:
        raise ValueError("episode_keys must not be empty")
    canonical_keys = []
    for key in keys:
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise ValueError("episode key must be (scenario_id, seed)")
        scenario_id, seed = key
        if scenario_id not in SCENARIO_IDS:
            raise ValueError("training scenario must be A, B, or C")
        seed = _strict_integer(seed, "training seed")
        if not 21000 <= seed <= 21999:
            raise ValueError("training seed must be within 21000-21999")
        canonical_keys.append((scenario_id, seed))
    if len(canonical_keys) != len(set(canonical_keys)):
        raise ValueError("episode_keys must not contain duplicates")
    if manifests is not None and not isinstance(manifests, Mapping):
        raise ValueError("manifests must be a mapping")
    if max_steps_per_episode is not None:
        step_limit = _strict_integer(
            max_steps_per_episode, "max_steps_per_episode", minimum=1)
    else:
        step_limit = None

    parameter_hash_before = _model_parameter_sha256(model)
    losses = []
    episode_returns = []
    manifest_hashes = []
    runtime_rows = []
    decision_count = 0
    update_count = 0
    episode_reports = []
    for scenario_id, seed in canonical_keys:
        supplied = (None if manifests is None else
                    manifests.get((scenario_id, seed)))
        manifest = _canonical_episode_manifest(
            scenario_id, seed, duration, supplied)
        if manifests is not None and supplied is None:
            raise ValueError("missing canonical training manifest")
        robots, tasks, context = factory_scenario(
            seed, scenario_id=scenario_id, duration_seconds=duration,
            manifest=manifest)
        episode_step_limit = (step_limit if step_limit is not None else
                              max(10000, len(tasks) * 30 + 100))
        environment = SchedulingEnvironment(
            RLEnvironmentConfig(
                max_robots=8, max_tasks=20,
                max_steps_per_episode=episode_step_limit),
            reward_profile=profile,
            runtime_config=HeadlessRuntimeConfig(
                episode_end_time=duration,
                max_advance_seconds=duration,
                fixed_horizon=True),
            simulation_mode="headless")
        _observation, reset_metadata = environment.reset(
            robots, tasks, context, seed=seed)
        del _observation
        if (reset_metadata["reward_profile_hash"] != profile.sha256 or
                reset_metadata["reward_profile_version"] != profile.version):
            raise RuntimeError("training reward profile was not installed")

        rollout = []
        action_rows = []
        episode_return = 0.0
        terminated = False
        for _step_index in range(episode_step_limit):
            live_robots, live_tasks, live_context = (
                environment.policy_snapshot())
            matrix, coordinates, features = graph_policy_inputs(
                live_robots, live_tasks, live_context)
            if not len(coordinates):
                mask = environment.get_action_mask()
                if not mask[environment.no_op_action]:
                    raise RuntimeError(
                        "no graph edge exists while no-op is illegal")
                _state, reward, terminated, truncated, info = (
                    environment.step(environment.no_op_action))
                del _state
            else:
                edge_index, log_probability, value = model.sample_edge(
                    features)
                row, column = coordinates[edge_index]
                task = matrix.tasks[int(column)]
                action = environment.action_for_pair(
                    matrix.robot_ids[int(row)],
                    task.task_id)
                _state, reward, terminated, truncated, info = (
                    environment.step(action))
                del _state
                rollout.append(GraphRolloutStep(
                    features.copy(), edge_index, log_probability, value,
                    0.0, False))
                action_rows.append({
                    "task_id": int(task.task_id),
                    "committed_at": max(
                        float(live_context.current_time),
                        float(task.arrival_time)),
                })
                decision_count += 1
            episode_return += float(reward)
            if (info.get("invalid_action") or
                    info.get("assignment_rejected") or
                    info.get("invalid_truncation")):
                raise RuntimeError(
                    "objective training encountered an invalid transition")
            if truncated:
                raise RuntimeError(
                    "objective training truncated before clean termination")
            if terminated:
                break
        if not terminated:
            raise RuntimeError("objective training did not reach the horizon")
        if not rollout:
            raise RuntimeError("objective training episode had no decisions")
        attribution_profile = None
        if objective == "utility_v2":
            attribution_profile = replace(
                profile, safety_observation="webots_physical")
        attribution = physical_episode_reward_attribution(
            environment._tasks, action_rows, objective=objective,
            utility_profile=attribution_profile, collision_count=0,
            horizon_seconds=duration)
        action_rewards = attribution["row_rewards"]
        if (len(action_rewards) != len(rollout) or not math.isclose(
                sum(action_rewards), episode_return,
                rel_tol=0.0, abs_tol=1e-9)):
            raise RuntimeError(
                "GraphPPO lifecycle rewards do not conserve episode return")
        for index, (step, action_reward) in enumerate(zip(
                rollout, action_rewards)):
            step.reward = float(action_reward)
            step.done = index == len(rollout)-1
        update = model.update(rollout)
        if update["mean_loss"] is None:
            raise RuntimeError("objective training produced no optimizer update")
        if not math.isfinite(float(update["mean_loss"])):
            raise RuntimeError("objective training loss is non-finite")
        losses.append(float(update["mean_loss"]))
        update_count += int(update["updates"])
        attributed_return = float(sum(step.reward for step in rollout))
        if not math.isclose(
                attributed_return, episode_return,
                rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("GraphPPO attributed return drifted")
        if not math.isfinite(episode_return):
            raise RuntimeError("objective training return is non-finite")
        episode_returns.append(episode_return)

        actual_manifest_hash = task_manifest_sha256(
            task.generation_parameters() for task in environment._tasks)
        if actual_manifest_hash != manifest["task_manifest_sha256"]:
            raise RuntimeError("training runtime changed the task manifest")
        runtime = environment.runtime_telemetry()
        if not (
                runtime.get("runtime_mode") == "headless_webots_logic" and
                runtime.get("dynamics_version") ==
                    HEADLESS_DYNAMICS_VERSION and
                runtime.get("physics_fidelity") == "business_logic_only" and
                runtime.get("joint_runtime") is True and
                runtime.get("route_planner") == "rolling_joint_grid" and
                runtime.get("fixed_horizon") is True and
                runtime.get("horizon_finalized") and
                runtime.get("termination_reason") == "episode_horizon" and
                runtime.get("current_time") == duration and
                runtime.get("episode_end_time") == duration and
                runtime.get("reward_profile_hash") == profile.sha256):
            raise RuntimeError("training runtime did not finish exact horizon")
        runtime_rows.append(runtime)
        manifest_hashes.append(manifest["manifest_sha256"])
        episode_reports.append({
            "scenario_id": scenario_id,
            "seed": seed,
            "manifest_sha256": manifest["manifest_sha256"],
            "task_manifest_sha256": actual_manifest_hash,
            "return": episode_return,
            "loss": float(update["mean_loss"]),
            "decisions": len(rollout),
            "runtime": runtime,
        })

    parameter_hash_after = _model_parameter_sha256(model)
    if parameter_hash_before == parameter_hash_after:
        raise RuntimeError("objective training did not update model weights")
    if not (np.all(np.isfinite(episode_returns)) and
            np.all(np.isfinite(losses))):
        raise RuntimeError("objective training produced non-finite metrics")
    return {
        "objective": objective,
        "reward_attribution": "task_lifecycle_action_level",
        "reward_conservation_verified": True,
        "episode_keys": [list(key) for key in canonical_keys],
        "episodes": len(canonical_keys),
        "decisions": decision_count,
        "optimizer_updates": update_count,
        "model_training_step": model.training_step,
        "model_training_episodes": model.training_episodes,
        "mean_return": float(np.mean(episode_returns)),
        "mean_loss": float(np.mean(losses)),
        "parameter_sha256_before": parameter_hash_before,
        "parameter_sha256_after": parameter_hash_after,
        "reward_profile": profile.canonical(),
        "reward_profile_sha256": profile.sha256,
        "manifest_sha256s": manifest_hashes,
        "runtime": summarize_runtime_telemetry(runtime_rows),
        "episode_reports": episode_reports,
    }


def save_and_verify_objective_checkpoint(
        model: GraphPPOModel, path, *, objective: str,
        expected_reward_profile_sha256: str) -> dict:
    """Atomically save via the model API and verify an exact round trip."""
    if objective not in {"count", "utility_v2"}:
        raise ValueError("objective must be count or utility_v2")
    if not _is_sha256(expected_reward_profile_sha256):
        raise ValueError("expected reward profile SHA-256 is invalid")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(target)
    loaded = GraphPPOModel.load(target, seed=model.seed)
    original_hash = _model_parameter_sha256(model)
    loaded_hash = _model_parameter_sha256(loaded)
    if original_hash != loaded_hash:
        raise RuntimeError("checkpoint round-trip changed model weights")
    if loaded.config != model.config:
        raise RuntimeError("checkpoint round-trip changed model config")
    if (loaded.training_step != model.training_step or
            loaded.training_episodes != model.training_episodes):
        raise RuntimeError("checkpoint round-trip changed training counters")
    checkpoint_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    return {
        "objective": objective,
        "checkpoint_path": str(target.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "parameter_sha256": original_hash,
        "reward_profile_sha256": expected_reward_profile_sha256,
        "training_step": model.training_step,
        "training_episodes": model.training_episodes,
        "round_trip_verified": True,
    }


def rank_trial_records(records: Iterable[Mapping[str, object]]) -> tuple:
    """Fail-closed deterministic ranking by validation metric only."""
    try:
        rows = tuple(records)
    except TypeError as exc:
        raise ValueError("trial records must be iterable") from exc
    eligible = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("trial record must be a mapping")
        trial_id = row.get("trial_id")
        trial_hash = row.get("trial_config_sha256")
        if (not isinstance(trial_id, str) or not trial_id or
                trial_id in seen):
            raise ValueError("trial IDs must be unique non-empty strings")
        seen.add(trial_id)
        if not _is_sha256(trial_hash):
            raise ValueError("trial config SHA-256 is invalid")
        if row.get("status") != "ok":
            continue
        metric = row.get("validation_metric")
        if (isinstance(metric, bool) or
                not isinstance(metric, (int, float, np.integer, np.floating))
                or not math.isfinite(float(metric))):
            raise ValueError("eligible trial validation metric is invalid")
        eligible.append(row)
    return tuple(sorted(
        eligible,
        key=lambda row: (
            -float(row["validation_metric"]),
            row["trial_config_sha256"], row["trial_id"])))


def select_successive_halving_survivors(
        records: Iterable[Mapping[str, object]],
        reduction_factor: int) -> Tuple[str, ...]:
    reduction = _strict_integer(
        reduction_factor, "reduction_factor", minimum=2)
    ranked = rank_trial_records(records)
    if not ranked:
        raise RuntimeError("successive halving has no eligible trial")
    survivor_count = max(1, math.ceil(len(ranked) / reduction))
    return tuple(row["trial_id"] for row in ranked[:survivor_count])


@dataclass(frozen=True)
class AutoTuningStudyResult:
    objective: str
    report: Mapping[str, object]
    winner_trial_id: str
    winner_configuration: Mapping[str, object]
    winner_model: GraphPPOModel


def _trial_model_seed(sampler_seed: int, objective: str,
                      trial_hash: str) -> int:
    payload = f"{sampler_seed}:{objective}:{trial_hash}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def run_successive_halving_study(
        contract: AutoTuningContract, objective: str, *,
        validation_callback,
        utility_profile: Optional[UtilityV2RewardProfile] = None,
        manifests: Optional[Mapping[Tuple[str, int], dict]] = None
        ) -> AutoTuningStudyResult:
    """Train, validate and prune deterministic GraphPPO configurations.

    ``validation_callback`` receives only a model plus the registered
    validation-seed prefix.  Training reports are deliberately unavailable to
    it and are never inspected by the ranking implementation.
    """
    if not isinstance(contract, AutoTuningContract):
        raise ValueError("contract must be AutoTuningContract")
    if objective not in {"count", "utility_v2"}:
        raise ValueError("objective must be count or utility_v2")
    _objective_reward_profile(
        objective, contract.duration_seconds, utility_profile)
    if not callable(validation_callback):
        raise ValueError("validation_callback must be callable")
    configurations = deterministic_trial_configs(
        contract.search_space, contract.max_trials, contract.sampler_seed)
    curriculum = curriculum_episode_keys(contract.partitions.training)
    states = {}
    for index, configuration in enumerate(configurations):
        trial_id = f"trial_{index:03d}"
        states[trial_id] = {
            "configuration": configuration,
            "model": GraphPPOModel(
                graph_ppo_config_from_trial(configuration),
                seed=_trial_model_seed(
                    contract.sampler_seed, objective,
                    configuration["trial_config_sha256"])),
            "trained_episodes": 0,
        }
    active = tuple(states)
    rung_reports = []
    budgets = contract.halving_plan
    for rung_index, (training_budget, validation_budget) in enumerate(zip(
            budgets.training_episode_budgets,
            budgets.validation_seed_budgets)):
        records = []
        for trial_id in active:
            state = states[trial_id]
            configuration = state["configuration"]
            record = {
                "trial_id": trial_id,
                "trial_config_sha256": configuration[
                    "trial_config_sha256"],
                "configuration": dict(configuration),
                "rung_index": rung_index,
                "training_episode_budget": training_budget,
                "validation_seed_budget": validation_budget,
            }
            try:
                previous_budget = state["trained_episodes"]
                if previous_budget >= training_budget:
                    raise RuntimeError(
                        "rung training budget did not increase")
                episode_keys = curriculum[previous_budget:training_budget]
                training_report = train_objective_graph_ppo(
                    state["model"], objective, episode_keys,
                    duration_seconds=contract.duration_seconds,
                    utility_profile=utility_profile, manifests=manifests)
                state["trained_episodes"] = training_budget
                validation_seeds = contract.partitions.validation[
                    :validation_budget]
                validation = validation_callback(
                    model=state["model"], objective=objective,
                    validation_seeds=validation_seeds,
                    duration_seconds=contract.duration_seconds,
                    utility_profile=utility_profile,
                    rung_index=rung_index, trial_id=trial_id)
                if not isinstance(validation, Mapping):
                    raise ValueError(
                        "validation callback must return a mapping")
                if validation.get("status") != "ok":
                    raise RuntimeError("validation callback did not pass")
                metric = validation.get("selection_metric")
                if (isinstance(metric, bool) or not isinstance(
                        metric, (int, float, np.integer, np.floating)) or
                        not math.isfinite(float(metric))):
                    raise ValueError("validation metric must be finite")
                record.update({
                    "status": "ok",
                    "validation_metric": float(metric),
                    "training": training_report,
                    "validation": dict(validation),
                })
            except Exception as exc:
                record.update({
                    "status": "failed",
                    "validation_metric": None,
                    "failure_type": type(exc).__name__,
                    "failure_reason": str(exc),
                })
            records.append(record)
        ranked = rank_trial_records(records)
        if not ranked:
            raise RuntimeError(
                f"all trials failed at rung {rung_index}")
        final_rung = rung_index == len(
            budgets.training_episode_budgets) - 1
        promoted = (() if final_rung else
                    select_successive_halving_survivors(
                        records, budgets.reduction_factor))
        rung_reports.append({
            "rung_index": rung_index,
            "active_trial_ids": list(active),
            "training_episode_budget": training_budget,
            "validation_seed_budget": validation_budget,
            "records": records,
            "ranking": [row["trial_id"] for row in ranked],
            "promoted_trial_ids": list(promoted),
        })
        if not final_rung:
            active = promoted

    final_ranked = rank_trial_records(rung_reports[-1]["records"])
    winner_id = final_ranked[0]["trial_id"]
    winner_state = states[winner_id]
    report = MappingProxyType({
        "version": AUTO_TUNING_VERSION,
        "objective": objective,
        "contract_sha256": contract.sha256,
        "selection_source": "validation_metric_only",
        "trial_count": contract.max_trials,
        "rungs": rung_reports,
        "winner_trial_id": winner_id,
        "winner_trial_config_sha256": winner_state[
            "configuration"]["trial_config_sha256"],
        "winner_configuration": dict(winner_state["configuration"]),
    })
    return AutoTuningStudyResult(
        objective=objective, report=report,
        winner_trial_id=winner_id,
        winner_configuration=winner_state["configuration"],
        winner_model=winner_state["model"])


@dataclass(frozen=True)
class OptimizationCalibration:
    utility_reward_profile: UtilityV2RewardProfile
    utility_score_config: UtilityScoreConfig

    def __post_init__(self):
        if not isinstance(
                self.utility_reward_profile, UtilityV2RewardProfile):
            raise ValueError(
                "utility_reward_profile must be UtilityV2RewardProfile")
        if not isinstance(self.utility_score_config, UtilityScoreConfig):
            raise ValueError(
                "utility_score_config must be UtilityScoreConfig")
        if (self.utility_reward_profile.horizon_seconds !=
                self.utility_score_config.horizon_seconds):
            raise ValueError("Utility reward and score horizons differ")

    @classmethod
    def recommended(cls, duration_seconds: float):
        duration = _strict_float(
            duration_seconds, "duration_seconds", positive=True)
        return cls(
            utility_reward_profile=UtilityV2RewardProfile(
                lambda_deadline=1.0, horizon_seconds=duration),
            utility_score_config=UtilityScoreConfig(
                tau_by_priority={1: 100.0, 2: 200.0, 3: 300.0},
                kappa_by_priority={1: 50.0, 2: 100.0, 3: 150.0},
                horizon_seconds=duration),
        )

    def canonical(self) -> dict:
        return {
            "utility_reward_profile":
                self.utility_reward_profile.canonical(),
            "utility_reward_profile_sha256":
                self.utility_reward_profile.sha256,
            "utility_score_config": self.utility_score_config.canonical(),
            "utility_score_config_sha256":
                self.utility_score_config.sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.canonical())


def _run_validation_episode(*, scheduler_name: str, model_path,
                            scenario_id: str, seed: int, manifest: dict,
                            objective: str, duration_seconds: float,
                            calibration: OptimizationCalibration) -> dict:
    robots, tasks, context = factory_scenario(
        seed, scenario_id=scenario_id, duration_seconds=duration_seconds,
        manifest=manifest)
    profile = (CountRewardProfile(horizon_seconds=duration_seconds)
               if objective == "count" else
               calibration.utility_reward_profile)
    step_limit = max(10000, len(tasks) * 30 + 100)
    environment = SchedulingEnvironment(
        RLEnvironmentConfig(
            max_robots=8, max_tasks=20,
            max_steps_per_episode=step_limit),
        reward_profile=profile,
        runtime_config=HeadlessRuntimeConfig(
            episode_end_time=duration_seconds,
            max_advance_seconds=duration_seconds,
            fixed_horizon=True),
        simulation_mode="headless")
    _observation, metadata = environment.reset(
        robots, tasks, context, seed=seed)
    del _observation
    scheduler = create_scheduler(
        scheduler_name, model_path, seed=seed, allow_safe_fallback=False)
    reset = getattr(scheduler, "reset", None)
    if callable(reset):
        reset()
    episode_return = 0.0
    decisions = 0
    invalid_outputs = 0
    fallback_commits = 0
    timeout_count = 0
    latencies_ms = []
    terminated = False
    truncated = False
    while decisions < step_limit:
        _state, reward, terminated, truncated, info = (
            environment.step_scheduler(scheduler))
        del _state
        decisions += 1
        episode_return += float(reward)
        invalid_outputs += int(bool(
            info.get("scheduler_output_rejected") or
            info.get("assignment_rejected") or
            info.get("invalid_action") or
            info.get("invalid_truncation")))
        diagnostics = info.get("scheduler_diagnostics", {})
        fallback_commits += int(bool(diagnostics.get("fallback", False)))
        observed_timeouts = diagnostics.get("timeout_count", 0)
        if (isinstance(observed_timeouts, bool) or
                not isinstance(observed_timeouts, int) or
                observed_timeouts < 0):
            raise RuntimeError("scheduler timeout counter is invalid")
        timeout_count = max(timeout_count, observed_timeouts)
        latency = info.get("scheduler_computation_time")
        if latency is not None:
            milliseconds = float(latency) * 1000.0
            if not math.isfinite(milliseconds) or milliseconds < 0:
                raise RuntimeError("scheduler latency is invalid")
            latencies_ms.append(milliseconds)
        if terminated or truncated:
            break
    final_tasks = environment._tasks
    actual_task_hash = task_manifest_sha256(
        task.generation_parameters() for task in final_tasks)
    runtime = environment.runtime_telemetry()
    valid = bool(
        terminated and not truncated and invalid_outputs == 0 and
        fallback_commits == 0 and timeout_count == 0 and
        actual_task_hash == manifest["task_manifest_sha256"] and
        metadata["reward_profile_hash"] == profile.sha256 and
        runtime.get("reward_profile_hash") == profile.sha256 and
        runtime.get("runtime_mode") == "headless_webots_logic" and
        runtime.get("dynamics_version") == HEADLESS_DYNAMICS_VERSION and
        runtime.get("physics_fidelity") == "business_logic_only" and
        runtime.get("joint_runtime") is True and
        runtime.get("route_planner") == "rolling_joint_grid" and
        runtime.get("fixed_horizon") is True and
        runtime.get("horizon_finalized") and
        runtime.get("termination_reason") == "episode_horizon" and
        runtime.get("current_time") == duration_seconds and
        runtime.get("episode_end_time") == duration_seconds)
    if not valid:
        raise RuntimeError("paired validation episode failed hard gates")
    count_score = count_evaluation(
        final_tasks, horizon_seconds=duration_seconds)
    route_distances = {
        task.task_id: {
            "ideal_distance": task.ideal_distance,
            "actual_distance": task.actual_distance,
        }
        for task in final_tasks if task.assignment_time is not None
    }
    utility_score = utility_v2_evaluation(
        final_tasks, calibration.utility_score_config,
        route_distances=route_distances)
    if not math.isfinite(episode_return):
        raise RuntimeError("validation return is non-finite")
    return {
        "status": "ok",
        "objective": objective,
        "scenario_id": scenario_id,
        "seed": seed,
        "scheduler": scheduler_name,
        "episode_return": episode_return,
        "decisions": decisions,
        "invalid_scheduler_outputs": invalid_outputs,
        "fallback_scheduler_commits": fallback_commits,
        "scheduler_timeout_count": timeout_count,
        "scheduler_latency_mean_ms": (
            float(np.mean(latencies_ms)) if latencies_ms else None),
        "manifest_sha256": manifest["manifest_sha256"],
        "task_manifest_sha256": actual_task_hash,
        "count_evaluation": count_score,
        "utility_v2_evaluation": utility_score,
        "runtime": runtime,
    }


def _paired_validation_summary(objective: str, pairs: Sequence[dict], *,
                               mode: str) -> dict:
    if objective not in {"count", "utility_v2"}:
        raise ValueError("objective must be count or utility_v2")
    if mode not in {"smoke", "formal"}:
        raise ValueError("mode must be smoke or formal")
    by_scenario = {scenario_id: [] for scenario_id in SCENARIO_IDS}
    for pair in pairs:
        by_scenario[pair["scenario_id"]].append(pair)
    if any(not rows for rows in by_scenario.values()):
        raise ValueError("paired validation must cover A, B, and C")
    scenarios = {}
    selection_values = []
    zero_baseline = False
    for scenario_id in SCENARIO_IDS:
        rows = by_scenario[scenario_id]
        if objective == "count":
            candidate = sum(
                row["candidate"]["count_evaluation"][
                    "total_tasks_completed"] for row in rows)
            baseline = sum(
                row["baseline"]["count_evaluation"][
                    "total_tasks_completed"] for row in rows)
            comparison = paired_count_delta(candidate, baseline)
            value = comparison["relative_delta"]
            if value is None:
                zero_baseline = True
                if mode == "smoke":
                    value = float(comparison["absolute_delta"])
            scenarios[scenario_id] = {
                "seed_count": len(rows),
                **comparison,
                "selection_value": value,
            }
        else:
            candidate_values = [
                row["candidate"]["utility_v2_evaluation"][
                    "utility_score_deadline_v2"] for row in rows]
            baseline_values = [
                row["baseline"]["utility_v2_evaluation"][
                    "utility_score_deadline_v2"] for row in rows]
            if any(value is None for value in
                   candidate_values + baseline_values):
                raise RuntimeError(
                    "Utility validation has no evaluable score")
            candidate = float(np.mean(candidate_values))
            baseline = float(np.mean(baseline_values))
            value = candidate - baseline
            scenarios[scenario_id] = {
                "seed_count": len(rows),
                "candidate_score_mean": candidate,
                "baseline_score_mean": baseline,
                "absolute_delta": value,
                "selection_value": value,
                "status": "ok",
            }
        if value is not None:
            selection_values.append(float(value))
    if len(selection_values) != len(SCENARIO_IDS):
        return {
            "status": "failed",
            "failure_reason": "formal_count_baseline_zero",
            "scenarios": scenarios,
            "selection_metric": None,
        }
    metric = float(np.mean(selection_values))
    method = ("a_b_c_macro_relative_delta"
              if objective == "count" and not zero_baseline else
              "a_b_c_macro_smoke_mixed_delta"
              if objective == "count" else
              "a_b_c_macro_absolute_score_delta")
    return {
        "status": "ok",
        "selection_metric": metric,
        "selection_metric_method": method,
        "equal_scenario_weighting": True,
        "scenarios": scenarios,
    }


def promotion_decision(objective: str, validation_summary: Mapping,
                       *, mode: str) -> dict:
    """Apply pre-registered non-regression and material-improvement gates."""
    if objective not in {"count", "utility_v2"}:
        raise ValueError("objective must be count or utility_v2")
    if mode not in {"smoke", "formal"}:
        raise ValueError("mode must be smoke or formal")
    if (not isinstance(validation_summary, Mapping) or
            validation_summary.get("status") != "ok"):
        return {
            "status": "validation_failed",
            "production_ready": False,
            "scenario_non_regression": False,
            "material_improvement": False,
        }
    scenarios = validation_summary["scenarios"]
    metric = float(validation_summary["selection_metric"])
    if objective == "count":
        values = [
            scenarios[scenario_id].get(
                "selection_value",
                scenarios[scenario_id]["relative_delta"])
            for scenario_id in SCENARIO_IDS]
        if (any(value is None for value in values) or
                not math.isclose(
                    metric, float(np.mean(values)),
                    rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError("Count promotion summary is inconsistent")
        scenario_gate = all(
            scenarios[scenario_id]["relative_delta"] is not None and
            scenarios[scenario_id]["relative_delta"] >= -0.02
            for scenario_id in SCENARIO_IDS)
        improvement_gate = metric >= 0.02
        thresholds = {
            "scenario_relative_delta_min": -0.02,
            "macro_relative_delta_min": 0.02,
        }
    else:
        values = [
            scenarios[scenario_id]["absolute_delta"]
            for scenario_id in SCENARIO_IDS]
        if not math.isclose(
                metric, float(np.mean(values)),
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Utility promotion summary is inconsistent")
        scenario_gate = all(
            scenarios[scenario_id]["absolute_delta"] >= -2.0
            for scenario_id in SCENARIO_IDS)
        improvement_gate = metric >= 2.0
        thresholds = {
            "scenario_absolute_score_delta_min": -2.0,
            "macro_absolute_score_delta_min": 2.0,
        }
    production_ready = bool(
        mode == "formal" and scenario_gate and improvement_gate)
    return {
        "status": (
            "promoted" if production_ready else
            "research_candidate_smoke" if mode == "smoke" else
            "gates_failed"),
        "production_ready": production_ready,
        "scenario_non_regression": scenario_gate,
        "material_improvement": improvement_gate,
        "thresholds": thresholds,
    }


class PairedValidationEvaluator:
    """Reusable paired validator with a canonical Hungarian baseline cache."""

    def __init__(self, *, mode: str, calibration: OptimizationCalibration):
        if mode not in {"smoke", "formal"}:
            raise ValueError("mode must be smoke or formal")
        if not isinstance(calibration, OptimizationCalibration):
            raise ValueError("calibration must be OptimizationCalibration")
        self.mode = mode
        self.calibration = calibration
        self._manifest_cache = {}
        self._baseline_cache = {}

    def __call__(self, *, model: GraphPPOModel, objective: str,
                 validation_seeds: Sequence[int], duration_seconds: float,
                 utility_profile, rung_index: int, trial_id: str) -> dict:
        del rung_index, trial_id
        if not isinstance(model, GraphPPOModel):
            raise ValueError("validation model must be GraphPPOModel")
        seeds = _strict_unique_seeds(validation_seeds, "validation")
        if any(not 31000 <= seed <= 31999 for seed in seeds):
            raise ValueError("validation seeds must be within 31000-31999")
        duration = _strict_float(
            duration_seconds, "duration_seconds", positive=True)
        if duration != self.calibration.utility_score_config.horizon_seconds:
            raise ValueError("validation duration differs from calibration")
        expected_utility = (
            self.calibration.utility_reward_profile
            if objective == "utility_v2" else None)
        if utility_profile != expected_utility:
            raise ValueError("study Utility profile differs from calibration")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.npz"
            model.save(checkpoint)
            pairs = []
            for scenario_id in SCENARIO_IDS:
                for seed in seeds:
                    key = (scenario_id, seed, duration)
                    manifest = self._manifest_cache.get(key)
                    if manifest is None:
                        manifest = generate_task_manifest(
                            scenario_id, formal_scenario_config(scenario_id),
                            seed, duration_seconds=duration)
                        self._manifest_cache[key] = manifest
                    baseline_key = (
                        objective, scenario_id, seed, duration,
                        self.calibration.sha256)
                    baseline = self._baseline_cache.get(baseline_key)
                    if baseline is None:
                        baseline = _run_validation_episode(
                            scheduler_name="Hungarian", model_path=None,
                            scenario_id=scenario_id, seed=seed,
                            manifest=manifest, objective=objective,
                            duration_seconds=duration,
                            calibration=self.calibration)
                        self._baseline_cache[baseline_key] = baseline
                    candidate = _run_validation_episode(
                        scheduler_name="GraphPPO", model_path=checkpoint,
                        scenario_id=scenario_id, seed=seed,
                        manifest=manifest, objective=objective,
                        duration_seconds=duration,
                        calibration=self.calibration)
                    if (candidate["manifest_sha256"] !=
                            baseline["manifest_sha256"]):
                        raise RuntimeError(
                            "paired validation manifest mismatch")
                    pairs.append({
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "manifest_sha256": manifest["manifest_sha256"],
                        "candidate": candidate,
                        "baseline": baseline,
                    })
        summary = _paired_validation_summary(
            objective, pairs, mode=self.mode)
        if summary["status"] != "ok":
            return summary
        summary["pairs"] = pairs
        summary["promotion"] = promotion_decision(
            objective, summary, mode=self.mode)
        summary["baseline_cache_entries"] = len(self._baseline_cache)
        return summary


def build_webots_finetune_plan(
        contract: AutoTuningContract,
        checkpoint_descriptors: Mapping[str, Mapping[str, object]], *,
        webots_transition_bridge_available: bool = False,
        physical_safety_observation_available: bool = False) -> dict:
    """Build the frozen 15-combination-per-model on-policy Webots plan."""
    if not isinstance(contract, AutoTuningContract):
        raise ValueError("contract must be AutoTuningContract")
    if (not isinstance(webots_transition_bridge_available, bool) or
            not isinstance(physical_safety_observation_available, bool)):
        raise ValueError("Webots capabilities must be boolean")
    if len(contract.partitions.webots_finetune) != 5:
        raise ValueError(
            "Webots fine-tuning requires exactly five training seeds")
    if not isinstance(checkpoint_descriptors, Mapping):
        raise ValueError("checkpoint_descriptors must be a mapping")
    if set(checkpoint_descriptors) != {"count", "utility_v2"}:
        raise ValueError(
            "Count and Utility checkpoint descriptors are both required")
    descriptors = {}
    for objective in ("count", "utility_v2"):
        descriptor = checkpoint_descriptors[objective]
        if not isinstance(descriptor, Mapping):
            raise ValueError("checkpoint descriptor must be a mapping")
        if descriptor.get("objective") != objective:
            raise ValueError("checkpoint descriptor objective mismatch")
        if not _is_sha256(descriptor.get("checkpoint_sha256")):
            raise ValueError("checkpoint descriptor hash is invalid")
        if descriptor.get("round_trip_verified") is not True:
            raise ValueError("checkpoint descriptor was not round-trip verified")
        descriptors[objective] = dict(descriptor)
    if (descriptors["count"]["checkpoint_sha256"] ==
            descriptors["utility_v2"]["checkpoint_sha256"]):
        raise ValueError(
            "Count and Utility Webots inputs must be distinct checkpoints")

    keys = webots_finetune_keys(contract.partitions)
    if (len(keys) != 15 or len(set(keys)) != 15 or
            {scenario for scenario, _seed in keys} != set(SCENARIO_IDS)):
        raise RuntimeError("Webots fine-tune matrix is not A/B/C x 5")
    manifest_rows = []
    for scenario_id, seed in keys:
        manifest = generate_task_manifest(
            scenario_id, formal_scenario_config(scenario_id), seed,
            duration_seconds=contract.duration_seconds)
        manifest_rows.append({
            "scenario_id": scenario_id,
            "seed": seed,
            "seed_partition": "training",
            "manifest_sha256": manifest["manifest_sha256"],
            "task_manifest_sha256": manifest["task_manifest_sha256"],
        })
    prerequisites = {
        "headless_checkpoint_round_trip_verified": True,
        "webots_transition_bridge_available":
            webots_transition_bridge_available,
        "physical_safety_observation_available":
            physical_safety_observation_available,
    }
    ready = all(prerequisites.values())
    objective_plans = {}
    for objective in ("count", "utility_v2"):
        objective_plans[objective] = {
            "objective": objective,
            "algorithm": "GraphPPO",
            "input_checkpoint_sha256":
                descriptors[objective]["checkpoint_sha256"],
            "input_checkpoint_path":
                descriptors[objective].get("checkpoint_path"),
            "output_checkpoint_name": "webots_finetuned_model.npz",
            "combination_count": len(manifest_rows),
            "combinations": [dict(row) for row in manifest_rows],
            "rounds": [{
                "training_seed": seed,
                "scenario_order": list(SCENARIO_IDS),
                "update_after_balanced_a_b_c_round": True,
            } for seed in contract.partitions.webots_finetune],
        }
    plan = {
        "version": AUTO_TUNING_VERSION,
        "status": "ready_to_execute" if ready else "blocked_prerequisites",
        "execution_claimed": False,
        "contract_sha256": contract.sha256,
        "algorithm": "GraphPPO",
        "fine_tune_semantics": "on_policy_physical_transitions_only",
        "learning_rate_multiplier": 0.1,
        "headless_replay_ratio": 0.0,
        "webots_on_policy_ratio": 1.0,
        "training_seed_only": True,
        "validation_or_final_test_seed_exposed": False,
        "scenario_count": 3,
        "training_seed_count": 5,
        "runs_per_objective": 15,
        "total_physical_runs": 30,
        "prerequisites": prerequisites,
        "objectives": objective_plans,
        "blocked_reasons": [
            name for name, available in prerequisites.items()
            if not available
        ],
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _atomic_write_json(path: Path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(value), indent=2, sort_keys=True,
                       ensure_ascii=False, allow_nan=False),
            encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_fingerprint(paths: Iterable[Path]) -> dict:
    resolved = tuple(sorted({Path(path).resolve() for path in paths},
                            key=lambda path: str(path)))
    common_root = (Path(os.path.commonpath([
        str(path.parent) for path in resolved])).resolve()
        if resolved else None)
    files = {}
    for path in resolved:
        if not path.is_file():
            raise ValueError(f"source fingerprint path is not a file: {path}")
        relative = path.relative_to(common_root).as_posix()
        files[relative] = _file_sha256(path)
    return {
        "sha256": _canonical_sha256(files),
        "files": files,
    }


def _write_study_artifacts(output_dir: Path, objective: str,
                           study: AutoTuningStudyResult) -> None:
    report = _json_safe(study.report)
    _atomic_write_json(output_dir / objective / "study.json", report)
    for rung in report["rungs"]:
        rung_index = rung["rung_index"]
        for record in rung["records"]:
            _atomic_write_json(
                output_dir / objective / "trials" /
                record["trial_id"] / f"rung_{rung_index}.json",
                record)


def run_dual_auto_optimization_workflow(
        contract: AutoTuningContract, output_dir, *,
        calibration: OptimizationCalibration,
        source_paths: Iterable[Path] = ()) -> dict:
    """Execute both studies and write a self-verifying artifact bundle."""
    if not isinstance(contract, AutoTuningContract):
        raise ValueError("contract must be AutoTuningContract")
    if not isinstance(calibration, OptimizationCalibration):
        raise ValueError("calibration must be OptimizationCalibration")
    if (calibration.utility_reward_profile.horizon_seconds !=
            contract.duration_seconds):
        raise ValueError("calibration horizon differs from contract")
    target = Path(output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError("output_dir must be absent or empty")
    target.mkdir(parents=True, exist_ok=True)
    source_paths = tuple(source_paths)
    source_before = _source_fingerprint(source_paths)
    started = time.perf_counter()

    _atomic_write_json(target / "contract.json", contract.canonical())
    _atomic_write_json(target / "calibration.json", calibration.canonical())
    training_manifests = {}
    for partition_name, seeds in (
            ("training", contract.partitions.training),
            ("validation", contract.partitions.validation)):
        for scenario_id in SCENARIO_IDS:
            for seed in seeds:
                manifest = generate_task_manifest(
                    scenario_id, formal_scenario_config(scenario_id), seed,
                    duration_seconds=contract.duration_seconds)
                write_task_manifest(
                    target / "manifests" / partition_name /
                    scenario_id / f"{seed}.json",
                    manifest)
                if partition_name == "training":
                    training_manifests[(scenario_id, seed)] = manifest

    evaluator = PairedValidationEvaluator(
        mode=contract.mode, calibration=calibration)
    studies = {}
    descriptors = {}
    objective_summaries = {}
    for objective in ("count", "utility_v2"):
        utility_profile = (
            calibration.utility_reward_profile
            if objective == "utility_v2" else None)
        study = run_successive_halving_study(
            contract, objective, validation_callback=evaluator,
            utility_profile=utility_profile,
            manifests=training_manifests)
        studies[objective] = study
        _write_study_artifacts(target, objective, study)
        profile = _objective_reward_profile(
            objective, contract.duration_seconds, utility_profile)
        checkpoint_path = target / objective / "winner" / "model.npz"
        checkpoint = save_and_verify_objective_checkpoint(
            study.winner_model, checkpoint_path, objective=objective,
            expected_reward_profile_sha256=profile.sha256)
        final_records = study.report["rungs"][-1]["records"]
        winner_record = next(
            row for row in final_records
            if row["trial_id"] == study.winner_trial_id)
        validation = winner_record["validation"]
        descriptor = {
            **checkpoint,
            "algorithm": "GraphPPO",
            "contract_sha256": contract.sha256,
            "calibration_sha256": calibration.sha256,
            "trial_id": study.winner_trial_id,
            "trial_config_sha256": study.winner_configuration[
                "trial_config_sha256"],
            "trial_configuration": dict(study.winner_configuration),
            "selection_metric": validation["selection_metric"],
            "selection_metric_method": validation[
                "selection_metric_method"],
            "promotion": validation["promotion"],
        }
        descriptors[objective] = descriptor
        _atomic_write_json(
            target / objective / "winner" / "policy_descriptor.json",
            descriptor)
        objective_summaries[objective] = {
            "winner_trial_id": study.winner_trial_id,
            "winner_trial_config_sha256": study.winner_configuration[
                "trial_config_sha256"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "selection_metric": validation["selection_metric"],
            "selection_metric_method": validation[
                "selection_metric_method"],
            "promotion": validation["promotion"],
        }

    webots_plan = build_webots_finetune_plan(contract, descriptors)
    _atomic_write_json(target / "webots_finetune_plan.json", webots_plan)
    source_after = _source_fingerprint(source_paths)
    source_unchanged = source_before == source_after
    if not source_unchanged:
        raise RuntimeError("source files changed during optimization workflow")
    distinct_checkpoints = (
        descriptors["count"]["checkpoint_sha256"] !=
        descriptors["utility_v2"]["checkpoint_sha256"])
    formal_promotions = all(
        summary["promotion"]["production_ready"]
        for summary in objective_summaries.values())
    artifact_hashes = {
        path.relative_to(target).as_posix(): _file_sha256(path)
        for path in sorted(target.rglob("*"))
        if path.is_file() and path.name != "workflow_report.json"
    }
    report = {
        "version": AUTO_TUNING_VERSION,
        "status": "completed",
        "mode": contract.mode,
        "contract_sha256": contract.sha256,
        "calibration_sha256": calibration.sha256,
        "elapsed_seconds": time.perf_counter() - started,
        "source_fingerprint": source_before,
        "source_unchanged_during_run": source_unchanged,
        "objectives": objective_summaries,
        "distinct_objective_checkpoints": distinct_checkpoints,
        "headless_optimization_complete": True,
        "formal_promotion_gates_passed": formal_promotions,
        "final_test": {
            "status": "reserved_not_run",
            "api_exposed_to_tuning": False,
        },
        "webots_finetune": {
            "status": webots_plan["status"],
            "plan_sha256": webots_plan["plan_sha256"],
            "execution_claimed": False,
        },
        "deployment": {
            "status": "not_ready",
            "reason": (
                "Webots physical fine-tuning and safety observation have "
                "not been executed"),
        },
        "artifact_hashes": artifact_hashes,
    }
    _atomic_write_json(target / "workflow_report.json", report)
    return report
