"""Unified headless integration and baseline training drivers for AI schedulers."""

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterable, Optional

import numpy as np

from headless_training_runtime import summarize_runtime_telemetry
from rl_environment import RLEnvironmentConfig, SchedulingEnvironment
from training_scenarios import factory_scenario


AI_ALGORITHM_NAMES = (
    "LearnedHungarian", "GraphImitation", "PPO_RL", "SARSA", "DQN",
    "GraphPPO", "RainbowDQN", "QRDQN", "CQL", "LinUCB",
)


@dataclass(frozen=True)
class AITrainingCapability:
    policy_interface: str
    learning_method: str
    training_driver: str
    consumes_runtime_reward: bool
    headless_execution: bool = True


AI_TRAINING_CAPABILITIES = MappingProxyType({
    "LearnedHungarian": AITrainingCapability(
        "scheduler_result", "supervised_regression",
        "run_ml_scheduler_workflow", False),
    "GraphImitation": AITrainingCapability(
        "scheduler_result", "expert_imitation",
        "run_ml_scheduler_workflow", False),
    "PPO_RL": AITrainingCapability(
        "pair_action", "online_actor_critic",
        "train_pairwise_ppo", True),
    "SARSA": AITrainingCapability(
        "pair_action", "online_td_control",
        "train_sarsa_agent", True),
    "DQN": AITrainingCapability(
        "pair_action", "online_replay_value_learning",
        "train_dqn_agent", True),
    "GraphPPO": AITrainingCapability(
        "graph_edge_action", "online_actor_critic",
        "train_graph_ppo", True),
    "RainbowDQN": AITrainingCapability(
        "pair_action", "online_distributional_value_learning",
        "train_online_value_agent", True),
    "QRDQN": AITrainingCapability(
        "pair_action", "online_quantile_value_learning",
        "train_online_value_agent", True),
    "CQL": AITrainingCapability(
        "pair_action", "offline_conservative_value_learning",
        "collect_cql_dataset", True),
    "LinUCB": AITrainingCapability(
        "scheduler_result", "contextual_bandit",
        "train_linucb", False),
})

if tuple(AI_TRAINING_CAPABILITIES) != AI_ALGORITHM_NAMES:
    raise RuntimeError("AI scheduler capability registry is incomplete or reordered")


def capability_report() -> dict:
    """Return the stable JSON-safe integration contract for all AI schedulers."""
    return {
        name: {
            "policy_interface": item.policy_interface,
            "learning_method": item.learning_method,
            "training_driver": item.training_driver,
            "consumes_runtime_reward": item.consumes_runtime_reward,
            "headless_execution": item.headless_execution,
        }
        for name, item in AI_TRAINING_CAPABILITIES.items()
    }


def _strict_seeds(values: Iterable[int]) -> tuple:
    seeds = tuple(values)
    if (not seeds or any(isinstance(seed, bool) or
                         not isinstance(seed, (int, np.integer))
                         for seed in seeds)):
        raise ValueError("training seeds must be integers")
    return tuple(int(seed) for seed in seeds)


def _environment(env_config=None) -> SchedulingEnvironment:
    return SchedulingEnvironment(env_config, simulation_mode="headless")


def run_scheduler_episode(scheduler, seed: int, *, max_decisions: int = 64,
                          env_config: Optional[RLEnvironmentConfig] = None
                          ) -> dict:
    """Run any project scheduler against the shared headless runtime."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if isinstance(max_decisions, bool) or not isinstance(max_decisions, int) \
            or max_decisions <= 0:
        raise ValueError("max_decisions must be a positive integer")
    environment = _environment(env_config)
    robots, tasks, context = factory_scenario(int(seed))
    environment.reset(robots, tasks, context, seed=int(seed))
    reset = getattr(scheduler, "reset", None)
    if callable(reset):
        reset()
    episode_return = 0.0
    decisions = 0
    rejected = 0
    fallback_decisions = 0
    terminated = False
    for _ in range(max_decisions):
        _, reward, terminated, truncated, info = (
            environment.step_scheduler(scheduler))
        episode_return += float(reward)
        decisions += 1
        rejected += int(bool(info.get("scheduler_output_rejected") or
                             info.get("assignment_rejected") or
                             info.get("invalid_action")))
        diagnostics = info.get("scheduler_diagnostics", {})
        fallback_decisions += int(bool(diagnostics.get("fallback", False)))
        if terminated or truncated:
            break
    return {
        "scheduler": getattr(scheduler, "name", type(scheduler).__name__),
        "seed": int(seed),
        "decisions": decisions,
        "rejected_outputs": rejected,
        "fallback_decisions": fallback_decisions,
        "terminated": bool(terminated),
        "episode_return": float(episode_return),
        "runtime": environment.runtime_telemetry(),
    }


def train_sarsa_agent(agent, episode_seeds: Iterable[int], *,
                      env_config: Optional[RLEnvironmentConfig] = None,
                      max_steps: int = 64) -> dict:
    """Train the existing tabular SARSA agent on Webots-logic transitions."""
    seeds = _strict_seeds(episode_seeds)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) \
            or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    environment = _environment(env_config)
    if (agent.action_dim != environment.action_dim or
            agent.no_op_action != environment.no_op_action):
        raise ValueError("SARSA dimensions do not match training environment")
    returns, errors, runtime_rows = [], [], []
    for seed in seeds:
        robots, tasks, context = factory_scenario(seed)
        observation, _ = environment.reset(robots, tasks, context, seed=seed)
        state = agent.discretize(observation)
        action = agent.select_action(
            state, environment.get_action_mask(), training=True)
        episode_return = 0.0
        for step_index in range(max_steps):
            next_observation, reward, terminated, truncated, _ = (
                environment.step(action))
            done = bool(terminated or truncated or step_index + 1 >= max_steps)
            next_state = agent.discretize(next_observation)
            next_action = agent.select_action(
                next_state, environment.get_action_mask(), training=True)
            errors.append(float(agent.update(
                state, action, reward, next_state, next_action, done)))
            episode_return += float(reward)
            state, action = next_state, next_action
            if done:
                break
        agent.end_episode()
        returns.append(episode_return)
        runtime_rows.append(environment.runtime_telemetry())
    return {
        "episodes": len(seeds),
        "episode_seeds": list(seeds),
        "updates": len(errors),
        "mean_return": float(np.mean(returns)),
        "mean_abs_td_error": float(np.mean(np.abs(errors))),
        "runtime": summarize_runtime_telemetry(runtime_rows),
    }


def train_dqn_agent(agent, episode_seeds: Iterable[int], *,
                    env_config: Optional[RLEnvironmentConfig] = None,
                    max_steps: int = 64) -> dict:
    """Train the existing DQN agent on Webots-logic transitions."""
    seeds = _strict_seeds(episode_seeds)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) \
            or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    environment = _environment(env_config)
    if (agent.state_dim != environment.observation_dim or
            agent.action_dim != environment.action_dim or
            agent.no_op_action != environment.no_op_action):
        raise ValueError("DQN dimensions do not match training environment")
    returns, losses, runtime_rows = [], [], []
    for seed in seeds:
        robots, tasks, context = factory_scenario(seed)
        state, _ = environment.reset(robots, tasks, context, seed=seed)
        episode_return = 0.0
        for step_index in range(max_steps):
            mask = environment.get_action_mask()
            action = agent.select_action(state, mask, training=True)
            next_state, reward, terminated, truncated, _ = (
                environment.step(action))
            done = bool(terminated or truncated or step_index + 1 >= max_steps)
            next_mask = environment.get_action_mask()
            agent.remember(
                state, action, reward, next_state, done, next_mask)
            loss = agent.train_step()
            if loss is not None:
                losses.append(float(loss))
            episode_return += float(reward)
            state = next_state
            if done:
                break
        agent.episode += 1
        returns.append(episode_return)
        runtime_rows.append(environment.runtime_telemetry())
    return {
        "episodes": len(seeds),
        "episode_seeds": list(seeds),
        "updates": len(losses),
        "mean_return": float(np.mean(returns)),
        "mean_loss": float(np.mean(losses)) if losses else None,
        "runtime": summarize_runtime_telemetry(runtime_rows),
    }


def _masked_probabilities(network, state, mask):
    probabilities, value = network.forward(state)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if (probabilities.shape != (network.action_dim,) or
            not np.isfinite(probabilities).all() or
            not math.isfinite(float(value)) or
            mask.shape != (network.action_dim,) or not mask.any()):
        raise ValueError("PPO action mask is invalid")
    values = np.where(mask, probabilities, 0.0)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 1e-12:
        values = mask.astype(np.float64) / np.count_nonzero(mask)
    else:
        values = values / total
    return values, float(value)


def _ppo_cache(network, state, mask):
    x = np.asarray(state, dtype=np.float64)
    if x.shape != (network.state_dim,) or not np.isfinite(x).all():
        raise ValueError("PPO state is invalid")
    z1 = x @ network.W1 + network.b1
    h1 = np.clip(np.maximum(z1, 0.0), 0.0, 50.0)
    z2 = h1 @ network.W2 + network.b2
    h2 = np.clip(np.maximum(z2, 0.0), 0.0, 50.0)
    logits = np.clip(h2 @ network.W_policy + network.b_policy, -30.0, 30.0)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (network.action_dim,) or not mask.any():
        raise ValueError("PPO action mask is invalid")
    legal_logits = logits[mask]
    legal_logits -= np.max(legal_logits)
    legal_probabilities = np.exp(legal_logits)
    legal_probabilities /= legal_probabilities.sum()
    probabilities = np.zeros(network.action_dim, dtype=np.float64)
    probabilities[mask] = legal_probabilities
    value = float(np.clip((h2 @ network.W_value + network.b_value)[0],
                          -100.0, 100.0))
    return probabilities, value, (x, z1, h1, z2, h2)


def _pairwise_ppo_update(network, rollout, *, learning_rate, clip_ratio,
                         value_coefficient, entropy_coefficient,
                         update_epochs, max_grad_norm) -> float:
    rewards = np.asarray([row[3] for row in rollout], dtype=np.float64)
    dones = np.asarray([row[6] for row in rollout], dtype=bool)
    values = np.asarray([row[5] for row in rollout], dtype=np.float64)
    returns = np.zeros_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + 0.99 * running * (not dones[index])
        returns[index] = running
    advantages = returns - values
    if len(advantages) > 1 and float(np.std(advantages)) > 1e-8:
        advantages = (advantages - advantages.mean()) / advantages.std()
    losses = []
    for _ in range(update_epochs):
        gradients = {
            "W1": np.zeros_like(network.W1), "b1": np.zeros_like(network.b1),
            "W2": np.zeros_like(network.W2), "b2": np.zeros_like(network.b2),
            "W_policy": np.zeros_like(network.W_policy),
            "b_policy": np.zeros_like(network.b_policy),
            "W_value": np.zeros_like(network.W_value),
            "b_value": np.zeros_like(network.b_value),
        }
        epoch_loss = 0.0
        for index, row in enumerate(rollout):
            state, mask, action, _, old_log_probability, _, _ = row
            probabilities, value, cache = _ppo_cache(network, state, mask)
            probability = max(float(probabilities[action]), 1e-12)
            log_ratio = float(np.clip(
                math.log(probability) - old_log_probability, -20.0, 20.0))
            ratio = math.exp(log_ratio)
            advantage = float(advantages[index])
            clipped = float(np.clip(ratio, 1.0 - clip_ratio,
                                    1.0 + clip_ratio))
            objective = min(ratio * advantage, clipped * advantage)
            clipped_region = ((advantage >= 0.0 and ratio > 1.0 + clip_ratio) or
                              (advantage < 0.0 and ratio < 1.0 - clip_ratio))
            log_gradient = 0.0 if clipped_region else -advantage * ratio
            grad_logits = -log_gradient * probabilities
            grad_logits[action] += log_gradient
            legal = probabilities > 0.0
            entropy = -float(np.sum(
                probabilities[legal] * np.log(probabilities[legal])))
            mean_log = float(np.sum(
                probabilities[legal] * np.log(probabilities[legal])))
            grad_logits[legal] += entropy_coefficient * probabilities[legal] * (
                np.log(probabilities[legal]) - mean_log)
            value_error = value - returns[index]
            grad_value = value_coefficient * value_error
            x, z1, h1, z2, h2 = cache
            gradients["W_policy"] += np.outer(h2, grad_logits)
            gradients["b_policy"] += grad_logits
            gradients["W_value"] += np.outer(h2, [grad_value])
            gradients["b_value"] += grad_value
            grad_h2 = (grad_logits @ network.W_policy.T
                       + grad_value * network.W_value[:, 0])
            grad_z2 = grad_h2 * ((z2 > 0.0) & (z2 < 50.0))
            gradients["W2"] += np.outer(h1, grad_z2)
            gradients["b2"] += grad_z2
            grad_h1 = grad_z2 @ network.W2.T
            grad_z1 = grad_h1 * ((z1 > 0.0) & (z1 < 50.0))
            gradients["W1"] += np.outer(x, grad_z1)
            gradients["b1"] += grad_z1
            epoch_loss += (-objective + 0.5 * value_coefficient
                           * value_error * value_error
                           - entropy_coefficient * entropy)
        scale = 1.0 / len(rollout)
        norm = math.sqrt(sum(float(np.sum((value * scale) ** 2))
                             for value in gradients.values()))
        clip = min(1.0, max_grad_norm / max(norm, 1e-12))
        for name, gradient in gradients.items():
            target = getattr(network, name)
            target -= learning_rate * gradient * scale * clip
            if not np.isfinite(target).all():
                raise ValueError("PPO update produced non-finite parameters")
        losses.append(epoch_loss * scale)
    return float(np.mean(losses))


def train_pairwise_ppo(network, episode_seeds: Iterable[int], *,
                       env_config: Optional[RLEnvironmentConfig] = None,
                       max_steps: int = 64, learning_rate: float = 3e-4,
                       clip_ratio: float = 0.2,
                       value_coefficient: float = 0.5,
                       entropy_coefficient: float = 0.01,
                       update_epochs: int = 4,
                       max_grad_norm: float = 1.0) -> dict:
    """Train the pair-action PPO checkpoint used by ``PairwisePPOScheduler``."""
    seeds = _strict_seeds(episode_seeds)
    numeric = (learning_rate, clip_ratio, value_coefficient,
               entropy_coefficient, max_grad_norm)
    if (isinstance(max_steps, bool) or not isinstance(max_steps, int) or
            max_steps <= 0 or isinstance(update_epochs, bool) or
            not isinstance(update_epochs, int) or update_epochs <= 0 or
            any(not math.isfinite(value) or value < 0 for value in numeric) or
            learning_rate <= 0 or not 0 < clip_ratio <= 1.0 or
            max_grad_norm <= 0):
        raise ValueError("invalid pairwise PPO training configuration")
    environment = _environment(env_config)
    if (network.state_dim != environment.observation_dim or
            network.action_dim != environment.action_dim):
        raise ValueError("PPO dimensions do not match training environment")
    losses, returns, runtime_rows = [], [], []
    rng = np.random.default_rng(seeds[0])
    for seed in seeds:
        robots, tasks, context = factory_scenario(seed)
        state, _ = environment.reset(robots, tasks, context, seed=seed)
        rollout = []
        episode_return = 0.0
        for step_index in range(max_steps):
            mask = environment.get_action_mask()
            probabilities, value = _masked_probabilities(network, state, mask)
            action = int(rng.choice(network.action_dim, p=probabilities))
            old_log_probability = math.log(max(probabilities[action], 1e-12))
            next_state, reward, terminated, truncated, _ = (
                environment.step(action))
            done = bool(terminated or truncated or step_index + 1 >= max_steps)
            rollout.append((state.copy(), mask.copy(), action, float(reward),
                            old_log_probability, value, done))
            episode_return += float(reward)
            state = next_state
            if done:
                break
        if rollout:
            losses.append(_pairwise_ppo_update(
                network, rollout, learning_rate=learning_rate,
                clip_ratio=clip_ratio, value_coefficient=value_coefficient,
                entropy_coefficient=entropy_coefficient,
                update_epochs=update_epochs, max_grad_norm=max_grad_norm))
        returns.append(episode_return)
        runtime_rows.append(environment.runtime_telemetry())
    return {
        "episodes": len(seeds),
        "episode_seeds": list(seeds),
        "updates": len(losses),
        "mean_return": float(np.mean(returns)),
        "mean_loss": float(np.mean(losses)) if losses else None,
        "runtime": summarize_runtime_telemetry(runtime_rows),
    }
