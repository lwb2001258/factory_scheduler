"""Standalone rollout and PPO updates for CostGraphPPO."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

import numpy as np

from cost_graph_ppo_model import CostGraphPPOModel
from cost_guided_graph_observation import (
    CostGuidedGraphObservation, build_cost_guided_graph_observation,
)
from dual_objective import CountRewardProfile, UtilityV2RewardProfile
from headless_training_runtime import (
    HeadlessRuntimeConfig, summarize_runtime_telemetry,
)
from rl_environment import RLEnvironmentConfig, SchedulingEnvironment
from training_scenarios import SCENARIOS, factory_scenario


@dataclass
class CostGraphPPORolloutStep:
    observation: CostGuidedGraphObservation
    action_index: int
    old_log_probability: float
    value: float
    reward: float
    terminated: bool
    truncated: bool
    advantage: float = 0.0
    return_value: float = 0.0

    def __post_init__(self):
        if (not isinstance(self.observation, CostGuidedGraphObservation) or
                isinstance(self.action_index, bool) or
                not isinstance(self.action_index, (int, np.integer)) or
                not 0 <= int(self.action_index) <
                self.observation.action_count or
                not self.observation.action_mask[int(self.action_index)] or
                not all(math.isfinite(float(value)) for value in (
                    self.old_log_probability, self.value, self.reward)) or
                float(self.old_log_probability) > 1e-12 or
                not isinstance(self.terminated, bool) or
                not isinstance(self.truncated, bool)):
            raise ValueError("CostGraphPPO rollout step is invalid")
        self.action_index = int(self.action_index)


def _huber(error: float, delta: float) -> tuple[float, float]:
    error = float(error)
    delta = float(delta)
    if not math.isfinite(error) or not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("CostGraphPPO Huber inputs are invalid")
    magnitude = abs(error)
    if magnitude <= delta:
        return 0.5 * error * error, error
    return delta * (magnitude - 0.5 * delta), math.copysign(delta, error)


def prepare_cost_graph_ppo_advantages(
        model: CostGraphPPOModel, rollout, *,
        bootstrap_value: float) -> None:
    if not rollout:
        return
    bootstrap_value = float(bootstrap_value)
    if not math.isfinite(bootstrap_value):
        raise ValueError("CostGraphPPO bootstrap value is non-finite")
    gae = 0.0
    next_value = bootstrap_value
    for step in reversed(rollout):
        nonterminal = 0.0 if step.terminated else 1.0
        delta = (
            step.reward + model.config.gamma * next_value * nonterminal
            - step.value)
        gae = (
            delta + model.config.gamma * model.config.gae_lambda
            * nonterminal * gae)
        if not math.isfinite(delta) or not math.isfinite(gae):
            raise ValueError("CostGraphPPO advantage is non-finite")
        step.advantage = float(gae)
        step.return_value = float(gae + step.value)
        next_value = step.value
    advantages = np.asarray(
        [step.advantage for step in rollout], dtype=np.float64)
    deviation = float(np.std(advantages))
    if len(advantages) > 1 and deviation > 1e-8:
        normalized = (
            advantages - float(np.mean(advantages))) / deviation
        for step, advantage in zip(rollout, normalized):
            step.advantage = float(advantage)


def update_cost_graph_ppo(
        model: CostGraphPPOModel, rollout, *,
        bootstrap_value: float | None = None) -> dict:
    """Apply clipped PPO to one on-policy, identity-checked rollout."""
    rollout = list(rollout)
    if not rollout:
        return {
            "updates": 0, "mean_loss": None,
            "mean_actor_loss": None, "mean_huber_value_loss": None,
            "mean_entropy": None, "max_abs_value_error": None,
        }
    if any(
            step.observation.prior_identity_sha256 !=
            model.prior_identity.sha256
            for step in rollout):
        raise ValueError("CostGraphPPO rollout identity differs")
    if any(step.terminated or step.truncated for step in rollout[:-1]):
        raise ValueError("CostGraphPPO rollout continues after a boundary")
    if bootstrap_value is None:
        if not rollout[-1].terminated:
            raise ValueError(
                "nonterminal CostGraphPPO rollout requires bootstrap")
        bootstrap_value = 0.0
    prepare_cost_graph_ppo_advantages(
        model, rollout, bootstrap_value=bootstrap_value)
    losses = []
    actor_losses = []
    value_losses = []
    entropies = []
    maximum_value_error = 0.0
    for _ in range(model.config.update_epochs):
        for step in rollout:
            probabilities, value, _, cache = model.network.forward(
                step.observation, cache=True)
            probability = max(
                float(probabilities[step.action_index]), 1e-12)
            log_ratio = math.log(probability) - step.old_log_probability
            ratio = math.exp(min(40.0, max(-40.0, log_ratio)))
            upper = 1.0 + model.config.clip_epsilon
            lower = 1.0 - model.config.clip_epsilon
            active = (
                (step.advantage >= 0.0 and ratio <= upper) or
                (step.advantage < 0.0 and ratio >= lower))
            actor_factor = (
                -step.advantage * ratio if active else 0.0)
            grad_logits = (
                -actor_factor * probabilities).astype(np.float32)
            grad_logits[step.action_index] += actor_factor
            legal = step.observation.action_mask
            log_probabilities = np.zeros_like(probabilities)
            log_probabilities[legal] = np.log(np.maximum(
                probabilities[legal], 1e-12))
            entropy = -float(np.sum(
                probabilities[legal] * log_probabilities[legal]))
            grad_logits += (
                model.config.entropy_coefficient
                * probabilities
                * (log_probabilities + entropy)).astype(np.float32)
            grad_logits[~legal] = 0.0
            value_error = value - step.return_value
            value_loss, value_gradient = _huber(
                value_error, model.config.value_huber_delta)
            grad_value = (
                model.config.value_coefficient * value_gradient)
            grads = model.network.backward(
                cache, grad_logits, grad_value)
            model.optimizer.step(model.network.params, grads)
            model.training_step = model.optimizer.step_count
            clipped_ratio = float(np.clip(ratio, lower, upper))
            actor_loss = -min(
                ratio * step.advantage,
                clipped_ratio * step.advantage)
            total_loss = (
                actor_loss
                + model.config.value_coefficient * value_loss
                - model.config.entropy_coefficient * entropy)
            if not all(math.isfinite(float(value)) for value in (
                    actor_loss, value_loss, entropy, total_loss)):
                raise ValueError("CostGraphPPO loss is non-finite")
            losses.append(float(total_loss))
            actor_losses.append(float(actor_loss))
            value_losses.append(float(value_loss))
            entropies.append(float(entropy))
            maximum_value_error = max(
                maximum_value_error, abs(float(value_error)))
    model.training_episodes += 1
    return {
        "updates": len(losses),
        "mean_loss": float(np.mean(losses)),
        "mean_actor_loss": float(np.mean(actor_losses)),
        "mean_huber_value_loss": float(np.mean(value_losses)),
        "mean_entropy": float(np.mean(entropies)),
        "max_abs_value_error": maximum_value_error,
        "value_huber_delta": model.config.value_huber_delta,
    }


def _strict_training_seeds(model: CostGraphPPOModel,
                           values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(values)
    if (not result or len(set(result)) != len(result) or
            any(isinstance(value, bool) or
                not isinstance(value, (int, np.integer))
                for value in result)):
        raise ValueError("CostGraphPPO training seeds are invalid")
    result = tuple(int(value) for value in result)
    if not set(result).issubset(model.provenance.standalone_seeds):
        raise ValueError("training seed is outside standalone provenance")
    return result


def _strict_scenarios(values) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if (not result or len(set(result)) != len(result) or
            any(value not in SCENARIOS for value in result)):
        raise ValueError("CostGraphPPO training scenarios are invalid")
    return result


def _validate_reward_profile(model, profile):
    if not isinstance(profile, (CountRewardProfile, UtilityV2RewardProfile)):
        raise ValueError("CostGraphPPO requires a formal reward profile")
    canonical = profile.canonical()
    if (canonical.get("objective") != model.prior_identity.objective or
            profile.version != model.provenance.reward_profile_version or
            profile.sha256 != model.provenance.reward_profile_sha256 or
            profile.horizon_seconds <= 0.0):
        raise ValueError("CostGraphPPO reward profile identity differs")


def _environment_action(environment, observation, action_index):
    coordinate = observation.coordinate_for_action(action_index)
    if coordinate is None:
        return environment.no_op_action
    row, column = coordinate
    return environment.action_for_pair(
        observation.matrix.robot_ids[row],
        observation.matrix.tasks[column].task_id)


def _bootstrap_value(model, environment) -> float:
    robots, tasks, context = environment.policy_snapshot()
    observation = build_cost_guided_graph_observation(
        robots, tasks, context, model.prior_identity)
    _, value, _ = model.policy(observation)
    return float(value)


def train_cost_graph_ppo(
        model: CostGraphPPOModel, episode_seeds: Iterable[int], *,
        reward_profile, scenario_ids=("A", "B", "C"),
        env_config: RLEnvironmentConfig | None = None,
        runtime_config: HeadlessRuntimeConfig | None = None,
        scenario_factory=factory_scenario) -> dict:
    """Run formal headless episodes using only the standalone seed partition."""
    if (not isinstance(model, CostGraphPPOModel) or
            model.provenance.training_phase != "standalone"):
        raise ValueError("CostGraphPPO standalone model is required")
    seeds = _strict_training_seeds(model, episode_seeds)
    scenarios = _strict_scenarios(scenario_ids)
    _validate_reward_profile(model, reward_profile)
    horizon = float(reward_profile.horizon_seconds)
    env_config = env_config or RLEnvironmentConfig(
        max_robots=8, max_tasks=20, max_steps_per_episode=10000)
    runtime_config = runtime_config or HeadlessRuntimeConfig(
        episode_end_time=horizon, max_advance_seconds=horizon,
        fixed_horizon=True)
    if (not math.isclose(
            runtime_config.episode_end_time, horizon,
            rel_tol=0.0, abs_tol=1e-9) or
            not runtime_config.fixed_horizon):
        raise ValueError("CostGraphPPO runtime horizon differs from reward")
    environment = SchedulingEnvironment(
        env_config, reward_profile=reward_profile,
        simulation_mode="headless", runtime_config=runtime_config)
    episode_rows = []
    runtime_rows = []
    loss_rows = []
    for seed in seeds:
        for scenario in scenarios:
            robots, tasks, context = scenario_factory(
                seed, scenario_id=scenario, duration_seconds=horizon)
            configuration = dict(context.configuration)
            configuration["graph_traffic_enabled"] = True
            context = replace(context, configuration=configuration)
            environment.reset(robots, tasks, context, seed=seed)
            rollout = []
            episode_return = 0.0
            terminated = truncated = False
            last_info = {}
            for _step in range(env_config.max_steps_per_episode):
                live_robots, live_tasks, live_context = (
                    environment.policy_snapshot())
                observation = build_cost_guided_graph_observation(
                    live_robots, live_tasks, live_context,
                    model.prior_identity)
                action_index, log_probability, value = model.select_action(
                    observation, deterministic=False)
                environment_action = _environment_action(
                    environment, observation, action_index)
                _, reward, terminated, truncated, info = environment.step(
                    environment_action)
                if (info.get("invalid_action") or
                        info.get("assignment_rejected") or
                        info.get("deferred_feasible_assignment") or
                        info.get("invalid_truncation")):
                    raise RuntimeError(
                        "CostGraphPPO standalone episode violated action "
                        f"contract: {info}")
                rollout.append(CostGraphPPORolloutStep(
                    observation, action_index, log_probability, value,
                    float(reward), bool(terminated), bool(truncated)))
                episode_return += float(reward)
                last_info = info
                if terminated or truncated:
                    break
            if not (terminated or truncated):
                raise RuntimeError("CostGraphPPO episode exceeded step budget")
            bootstrap = (
                0.0 if terminated else _bootstrap_value(model, environment))
            telemetry = environment.runtime_telemetry()
            reward_totals = telemetry.get("reward_totals", {})
            accounted = reward_totals.get("reward_total")
            conservation_error = (
                float("inf") if accounted is None else
                abs(episode_return - float(accounted)))
            if (not math.isfinite(episode_return) or
                    conservation_error > 1e-9):
                raise RuntimeError(
                    "CostGraphPPO formal reward conservation failed")
            # Gradient mutation is permitted only after the episode has
            # crossed the action-contract and formal-reward audit boundary.
            update = update_cost_graph_ppo(
                model, rollout, bootstrap_value=bootstrap)
            row = {
                "seed": seed,
                "scenario": scenario,
                "steps": len(rollout),
                "return": float(episode_return),
                "reward_total": float(accounted),
                "reward_conservation_error": conservation_error,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "completed_count": int(
                    last_info.get("completed_count", 0)),
                "update": update,
            }
            episode_rows.append(row)
            runtime_rows.append(telemetry)
            loss_rows.append(update["mean_loss"])
    if not episode_rows:
        raise RuntimeError("CostGraphPPO training produced no episodes")
    return {
        "algorithm": "CostGraphPPO",
        "policy_id": model.prior_identity.policy_id,
        "objective": model.prior_identity.objective,
        "prior_identity_sha256": model.prior_identity.sha256,
        "reward_profile_version": reward_profile.version,
        "reward_profile_sha256": reward_profile.sha256,
        "episode_seeds": list(seeds),
        "scenarios": list(scenarios),
        "episodes": episode_rows,
        "episode_count": len(episode_rows),
        "training_step": model.training_step,
        "training_episodes": model.training_episodes,
        "mean_return": float(np.mean([
            row["return"] for row in episode_rows])),
        "mean_loss": float(np.mean(loss_rows)),
        "max_reward_conservation_error": float(max(
            row["reward_conservation_error"] for row in episode_rows)),
        "runtime": summarize_runtime_telemetry(runtime_rows),
    }
