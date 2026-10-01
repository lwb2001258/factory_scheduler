"""Deterministic headless Webots-logic training primitives.

These functions use the same versioned scheduling environment as deployment.
They use the unchanged project path coordinator, but never invoke Webots
physics, sensors, radio, or robot-controller collision avoidance.
"""

import math
from typing import Iterable, Optional

import numpy as np

from advanced_rl_agents import OfflineTransitionDataset
from headless_training_runtime import summarize_runtime_telemetry
from rl_environment import RLEnvironmentConfig, SchedulingEnvironment
from training_scenarios import factory_scenario


def _normalise_seeds(values: Iterable[int]) -> tuple:
    seeds = tuple(values)
    if (not seeds or any(
            isinstance(seed, bool) or
            not isinstance(seed, (int, np.integer))
            for seed in seeds)):
        raise ValueError("training seeds must be integers")
    return tuple(int(seed) for seed in seeds)


def _environment_for_agent(agent, env_config=None):
    environment = SchedulingEnvironment(
        env_config, simulation_mode="headless")
    if (agent.state_dim != environment.observation_dim or
            agent.action_dim != environment.action_dim or
            agent.no_op_action != environment.no_op_action):
        raise ValueError("agent dimensions do not match training environment")
    return environment


def train_online_value_agent(agent, episode_seeds: Iterable[int], *,
                             env_config: Optional[RLEnvironmentConfig] = None,
                             max_steps: Optional[int] = None) -> dict:
    """Train Rainbow/QRDQN against deterministic factory snapshots."""
    seeds = _normalise_seeds(episode_seeds)
    environment = _environment_for_agent(agent, env_config)
    step_limit = (environment.config.max_steps_per_episode
                  if max_steps is None else int(max_steps))
    if step_limit <= 0:
        raise ValueError("max_steps must be positive")
    losses = []
    returns = []
    steps = 0
    runtime_rows = []
    for seed in seeds:
        robots, tasks, context = factory_scenario(seed)
        state, _ = environment.reset(robots, tasks, context, seed=seed)
        episode_return = 0.0
        for step in range(step_limit):
            mask = environment.get_action_mask()
            action = agent.select_action(state, mask, training=True)
            next_state, reward, terminated, truncated, _ = (
                environment.step(action))
            done = bool(terminated or truncated or step+1 >= step_limit)
            next_mask = environment.get_action_mask()
            agent.remember(
                state, mask, action, reward, next_state, next_mask, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(float(loss))
            episode_return += float(reward)
            steps += 1
            state = next_state
            if done:
                break
        returns.append(episode_return)
        runtime_rows.append(environment.runtime_telemetry())
    if (not np.all(np.isfinite(returns)) or
            (losses and not np.all(np.isfinite(losses)))):
        raise ValueError("training produced non-finite metrics")
    return {
        "episode_seeds": list(seeds),
        "episodes": len(seeds),
        "steps": steps,
        "updates": len(losses),
        "mean_return": float(np.mean(returns)),
        "mean_loss": float(np.mean(losses)) if losses else None,
        "runtime": summarize_runtime_telemetry(runtime_rows),
    }


def _cost_greedy_action(environment: SchedulingEnvironment,
                        action_mask: np.ndarray) -> int:
    legal = np.flatnonzero(action_mask)
    if legal.size == 1 and legal[0] == environment.no_op_action:
        return int(legal[0])
    candidates = []
    for action in legal:
        assignment = environment.assignment_for_action(int(action))
        if assignment is None:
            continue
        cost = float(assignment.estimated_cost)
        if math.isfinite(cost):
            candidates.append((cost, int(action)))
    if not candidates:
        if action_mask[environment.no_op_action]:
            return environment.no_op_action
        raise ValueError("behaviour policy found no legal assignment")
    return min(candidates)[1]


def collect_cql_dataset(episode_seeds: Iterable[int], *,
                        env_config: Optional[RLEnvironmentConfig] = None,
                        max_steps: Optional[int] = None
                        ) -> OfflineTransitionDataset:
    """Collect a fixed, explicitly labelled cost-greedy behaviour data set."""
    seeds = _normalise_seeds(episode_seeds)
    environment = SchedulingEnvironment(
        env_config, simulation_mode="headless")
    step_limit = (environment.config.max_steps_per_episode
                  if max_steps is None else int(max_steps))
    if step_limit <= 0:
        raise ValueError("max_steps must be positive")
    rows = []
    for seed in seeds:
        robots, tasks, context = factory_scenario(seed)
        state, _ = environment.reset(robots, tasks, context, seed=seed)
        for step in range(step_limit):
            mask = environment.get_action_mask()
            action = _cost_greedy_action(environment, mask)
            next_state, reward, terminated, truncated, _ = (
                environment.step(action))
            done = bool(terminated or truncated or step+1 >= step_limit)
            next_mask = environment.get_action_mask()
            rows.append((
                state.copy(), mask.copy(), action, float(reward),
                next_state.copy(), next_mask.copy(), done))
            state = next_state
            if done:
                break
    if not rows:
        raise ValueError("offline collection produced no transitions")
    return OfflineTransitionDataset(
        np.stack([row[0] for row in rows]),
        np.stack([row[1] for row in rows]),
        np.asarray([row[2] for row in rows], dtype=np.int64),
        np.asarray([row[3] for row in rows], dtype=np.float32),
        np.stack([row[4] for row in rows]),
        np.stack([row[5] for row in rows]),
        np.asarray([row[6] for row in rows], dtype=bool),
        "MaskedCostGreedy", seeds)
