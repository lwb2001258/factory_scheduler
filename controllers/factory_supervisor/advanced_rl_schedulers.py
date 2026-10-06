"""Safety-preserving scheduler adapters for advanced value policies."""

import os
import time
from typing import Optional

import numpy as np

from advanced_ai_common import masked_argmax
from advanced_rl_agents import CQLAgent, QRDQNAgent, RainbowDQNAgent
from rl_environment import RLEnvironmentConfig, SchedulingEnvironment
from schedulers import (
    BaseScheduler, SchedulerResult, SchedulingContext, validate_assignment,
)


class ValuePolicyScheduler(BaseScheduler):
    """Adapt a finite masked value policy to the shared assignment contract.

    The policy only ranks the fixed robot-task action space.  Feasibility,
    path reachability and the canonical task object remain owned by
    ``SchedulingEnvironment`` and ``validate_assignment``.
    """

    def __init__(self, name: str, policy, *,
                 env_config: Optional[RLEnvironmentConfig] = None):
        super().__init__(name)
        self.environment = SchedulingEnvironment(
            env_config, simulation_mode="webots")
        self.policy = policy

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        result = self.assign(
            pending_tasks, robot_states,
            SchedulingContext(congestion_map=congestion_map))
        if not result.assignments:
            return None
        assignment = result.assignments[0]
        return assignment.robot_id, assignment.task

    def assign(self, pending_tasks, robot_states,
               context: Optional[SchedulingContext] = None):
        started = time.perf_counter()
        context = context or SchedulingContext()
        action = None
        try:
            state = self.environment.set_snapshot(
                robot_states, pending_tasks, context)
            mask = self.environment.get_action_mask()
            values = np.asarray(
                self.policy.action_values(state), dtype=np.float64)
            if values.shape != (self.environment.action_dim,):
                raise ValueError("policy returned an invalid action-value shape")
            physical_fine_tune = os.environ.get(
                "SMART_FACTORY_PHYSICAL_FINE_TUNE", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            if physical_fine_tune and self.name != "CQL":
                action = self.policy.select_action(
                    state, mask, training=True)
            else:
                # CQL deliberately remains greedy: it is an offline policy
                # and has no exploration parameter in its action contract.
                action = masked_argmax(values, mask)
            assignment = self.environment.assignment_for_action(action)
            valid, reason = validate_assignment(
                assignment, pending_tasks, robot_states, context)
        except Exception as exc:
            assignment = None
            valid = False
            reason = f"policy_inference_error:{type(exc).__name__}"
        elapsed = time.perf_counter() - started
        diagnostics = {
            "reason": reason,
            "action": action,
            "pairwise_action": True,
        }
        if (locals().get("physical_fine_tune", False) and valid and
                assignment is not None):
            diagnostics.update({
                "decoder": "physical_policy_action",
                "physical_fine_tune": True,
                "physical_rollout_step": {
                    "kind": "pair_value",
                    "state": state.tolist(),
                    "action_mask": mask.astype(np.uint8).tolist(),
                    "action": int(action),
                    "selected_robot_id": int(assignment.robot_id),
                    "selected_task_id": int(assignment.task.task_id),
                },
            })
        return SchedulerResult(
            assignments=[assignment] if valid and assignment else [],
            objective_value=(assignment.estimated_cost
                             if valid and assignment else None),
            computation_time=elapsed,
            is_feasible=bool(valid and assignment),
            algorithm_name=self.name,
            diagnostics=diagnostics,
        )


class RainbowDQNScheduler(ValuePolicyScheduler):
    def __init__(self, model_path: str, seed: int = 42,
                 env_config: Optional[RLEnvironmentConfig] = None):
        environment = SchedulingEnvironment(
            env_config, simulation_mode="webots")
        policy = RainbowDQNAgent.load(
            model_path, environment.observation_dim, environment.action_dim,
            environment.no_op_action, seed)
        super().__init__("RainbowDQN", policy, env_config=env_config)


class QRDQNScheduler(ValuePolicyScheduler):
    def __init__(self, model_path: str, seed: int = 42,
                 env_config: Optional[RLEnvironmentConfig] = None):
        environment = SchedulingEnvironment(
            env_config, simulation_mode="webots")
        policy = QRDQNAgent.load(
            model_path, environment.observation_dim, environment.action_dim,
            environment.no_op_action, seed)
        super().__init__("QRDQN", policy, env_config=env_config)


class CQLScheduler(ValuePolicyScheduler):
    def __init__(self, model_path: str, seed: int = 42,
                 env_config: Optional[RLEnvironmentConfig] = None):
        environment = SchedulingEnvironment(
            env_config, simulation_mode="webots")
        policy = CQLAgent.load(
            model_path, environment.observation_dim, environment.action_dim,
            environment.no_op_action, seed)
        super().__init__("CQL", policy, env_config=env_config)
