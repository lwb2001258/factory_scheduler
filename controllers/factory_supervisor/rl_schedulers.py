"""Scheduler adapters and fail-closed safety wrapper for DQN/SARSA."""

import os
import time
from typing import List, Optional

import numpy as np

from rl_agents import DQNAgent, SarsaAgent, dqn_config_from_checkpoint
from rl_environment import RLEnvironmentConfig, SchedulingEnvironment
from schedulers import (
    Assignment, BaseScheduler, HungarianScheduler, ModelValidationError,
    PPONetwork, SchedulerResult, SchedulingContext, validate_assignment,
)
from task_generator import TransportTask


class _AgentScheduler(BaseScheduler):
    def __init__(self, name: str, environment: SchedulingEnvironment):
        super().__init__(name)
        self.environment = environment

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        context = SchedulingContext(congestion_map=congestion_map)
        result = self.assign(pending_tasks, robot_states, context)
        if not result.assignments:
            return None
        item = result.assignments[0]
        return item.robot_id, item.task


class SarsaScheduler(_AgentScheduler):
    def __init__(self, model_path: str, seed: int = 42,
                 env_config: Optional[RLEnvironmentConfig] = None):
        env = SchedulingEnvironment(env_config, simulation_mode="webots")
        super().__init__("SARSA", env)
        if not model_path:
            raise ModelValidationError("SARSA checkpoint is required for inference")
        self.agent = SarsaAgent(
            env.action_dim, env.no_op_action, seed=seed)
        self.agent.load(model_path)

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None):
        started = time.perf_counter()
        context = context or SchedulingContext()
        try:
            state = self.environment.set_snapshot(
                robot_states, pending_tasks, context)
            mask = self.environment.get_action_mask()
            action = self.agent.select_action(
                self.agent.discretize(state), mask, training=False)
            assignment = self.environment.assignment_for_action(action)
            valid, reason = validate_assignment(
                assignment, pending_tasks, robot_states, context)
        except Exception as exc:
            assignment, valid = None, False
            reason = f"sarsa_inference_error:{type(exc).__name__}"
        return SchedulerResult(
            [assignment] if valid and assignment else [],
            assignment.estimated_cost if assignment else None,
            time.perf_counter() - started, valid, self.name,
            {"reason": reason, "action": locals().get("action")})


class DQNScheduler(_AgentScheduler):
    def __init__(self, model_path: str, seed: int = 42,
                 env_config: Optional[RLEnvironmentConfig] = None):
        env = SchedulingEnvironment(env_config, simulation_mode="webots")
        super().__init__("DQN", env)
        if not model_path:
            raise ModelValidationError("DQN checkpoint is required for inference")
        self.agent = DQNAgent(
            env.observation_dim, env.action_dim, env.no_op_action,
            dqn_config_from_checkpoint(model_path), seed=seed)
        self.agent.load(model_path)

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None):
        started = time.perf_counter()
        context = context or SchedulingContext()
        try:
            state = self.environment.set_snapshot(
                robot_states, pending_tasks, context)
            mask = self.environment.get_action_mask()
            action = self.agent.select_action(state, mask, training=False)
            assignment = self.environment.assignment_for_action(action)
            valid, reason = validate_assignment(
                assignment, pending_tasks, robot_states, context)
        except Exception as exc:
            assignment, valid = None, False
            reason = f"dqn_inference_error:{type(exc).__name__}"
        return SchedulerResult(
            [assignment] if valid and assignment else [],
            assignment.estimated_cost if assignment else None,
            time.perf_counter() - started, valid, self.name,
            {"reason": reason, "action": locals().get("action")})


class PairwisePPOScheduler(_AgentScheduler):
    """PPO adapter sharing DQN/SARSA observation and pair-action semantics."""

    def __init__(self, model_path: str, seed: int = 42,
                 env_config: Optional[RLEnvironmentConfig] = None):
        env = SchedulingEnvironment(env_config, simulation_mode="webots")
        super().__init__("PPO_RL", env)
        if not model_path:
            raise ModelValidationError("PPO checkpoint is required for inference")
        try:
            with np.load(model_path, allow_pickle=False) as data:
                hidden_size = int(data["hidden_size"][0])
        except Exception as exc:
            raise ModelValidationError(
                f"cannot read PPO checkpoint metadata: {exc}") from exc
        self.network = PPONetwork(
            env.observation_dim, env.action_dim, hidden_size)
        self.network.load(model_path)

    @staticmethod
    def _masked(probabilities, mask):
        mask = np.asarray(mask, dtype=bool)
        values = np.where(mask, probabilities, 0.0)
        total = float(values.sum())
        if not np.isfinite(total) or total <= 1e-12:
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                raise ValueError("PPO snapshot has no legal action")
            values = np.zeros_like(probabilities, dtype=float)
            values[legal] = 1.0 / legal.size
            return values
        return values / total

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None):
        started = time.perf_counter()
        context = context or SchedulingContext()
        try:
            state = self.environment.set_snapshot(
                robot_states, pending_tasks, context)
            mask = self.environment.get_action_mask()
            probabilities, _ = self.network.forward(state)
            masked = self._masked(probabilities, mask)
            action = int(np.argmax(masked))
            assignment = self.environment.assignment_for_action(action)
            valid, reason = validate_assignment(
                assignment, pending_tasks, robot_states, context)
        except Exception as exc:
            assignment, valid = None, False
            reason = f"ppo_inference_error:{type(exc).__name__}"
        return SchedulerResult(
            [assignment] if valid and assignment else [],
            assignment.estimated_cost if assignment else None,
            time.perf_counter() - started, valid, self.name,
            {"reason": reason, "action": locals().get("action"),
             "pairwise_action": True})


class RLSchedulerSafetyWrapper(BaseScheduler):
    """Fail closed on model/policy errors and use deterministic Hungarian."""

    def __init__(self, policy: BaseScheduler, timeout_seconds: float = None,
                 max_consecutive_failures: int = 3):
        super().__init__(policy.name)
        self.policy = policy
        self.fallback = HungarianScheduler()
        if timeout_seconds is None:
            timeout_seconds = float(os.environ.get(
                "RL_SCHEDULER_TIMEOUT_SECONDS", "0.05"))
        self.timeout_seconds = max(0.001, float(timeout_seconds))
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self._last_source = "policy"
        self.policy_decisions = 0
        self.fallback_decisions = 0
        self.timeout_count = 0

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        result = self.assign(
            pending_tasks, robot_states,
            SchedulingContext(congestion_map=congestion_map))
        if not result.assignments:
            return None
        item = result.assignments[0]
        return item.robot_id, item.task

    def assign(self, pending_tasks: List[TransportTask], robot_states: dict,
               context: Optional[SchedulingContext] = None):
        context = context or SchedulingContext()
        if self.consecutive_failures < self.max_consecutive_failures:
            result = self.policy.assign(pending_tasks, robot_states, context)
            timed_out = result.computation_time > self.timeout_seconds
            if result.is_feasible and result.assignments and not timed_out:
                self.consecutive_failures = 0
                self._last_source = "policy"
                self.policy_decisions += 1
                result.diagnostics.update({
                    "fallback": False,
                    "inference_ms": result.computation_time * 1000.0,
                    "policy_decisions": self.policy_decisions,
                    "fallback_decisions": self.fallback_decisions,
                    "timeout_count": self.timeout_count,
                    "timeout_limit_ms": self.timeout_seconds * 1000.0,
                })
                return result
            self.consecutive_failures += 1
            if timed_out:
                self.timeout_count += 1
            failure_reason = (
                "inference_timeout" if timed_out
                else result.diagnostics.get("reason", "invalid_rl_output"))
        else:
            failure_reason = "rl_temporarily_disabled"
        fallback = self.fallback.assign(pending_tasks, robot_states, context)
        self.fallback_decisions += 1
        policy_seconds = (result.computation_time
                          if 'result' in locals() else 0.0)
        fallback.computation_time += policy_seconds
        fallback.algorithm_name = f"{self.policy.name}_FALLBACK_HUNGARIAN"
        fallback.diagnostics.update({
            "fallback": True, "rl_failure": failure_reason,
            "consecutive_failures": self.consecutive_failures})
        fallback.diagnostics.update({
            "inference_ms": policy_seconds * 1000.0,
            "policy_decisions": self.policy_decisions,
            "fallback_decisions": self.fallback_decisions,
            "timeout_count": self.timeout_count,
            "timeout_limit_ms": self.timeout_seconds * 1000.0,
        })
        self._last_source = "fallback"
        return fallback

    def on_assignment_committed(self, assignment: Assignment) -> None:
        target = self.policy if self._last_source == "policy" else self.fallback
        target.on_assignment_committed(assignment)
        super().on_assignment_committed(assignment)

    def on_assignment_rejected(self, assignment, reason):
        target = self.policy if self._last_source == "policy" else self.fallback
        target.on_assignment_rejected(assignment, reason)
        self.consecutive_failures += 1

    def reset(self):
        super().reset()
        self.policy.reset()
        self.fallback.reset()
        self.consecutive_failures = 0
        self.policy_decisions = 0
        self.fallback_decisions = 0
        self.timeout_count = 0
