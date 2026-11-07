"""Runtime adapters for CostGraphPPO and variable-edge GraphRainbow."""

from __future__ import annotations

import os
import time

import numpy as np

from advanced_ai_common import masked_argmax
from cost_graph_ppo_model import CostGraphPPOModel
from cost_guided_graph_observation import (
    COST_GUIDED_GRAPH_SCHEMA_SHA256, CostGuidedGraphObservation,
    build_cost_guided_graph_observation,
)
from graph_rainbow_model import GraphRainbowModel
from schedulers import (
    BaseScheduler, SchedulerResult, SchedulingContext,
    _result_from_matching,
)


def _enabled(name) -> bool:
    return os.environ.get(name, "0").strip().lower() in {
        "1", "true", "yes", "on"}


def _fine_tune_enabled() -> bool:
    return any(_enabled(name) for name in (
        "SMART_FACTORY_PHYSICAL_FINE_TUNE",
        "SMART_FACTORY_COST_GRAPH_RL_FINE_TUNE",
    ))


def _exploration_enabled() -> bool:
    """Separate physical reward collection from exploratory inference.

    Final evaluation reuses the supervisor's physical action/reward audit but
    must keep the frozen checkpoint deterministic.  Fine-tuning remains
    exploratory unless the workflow explicitly requests deterministic audit
    collection.
    """
    return (
        _fine_tune_enabled()
        and not _enabled("SMART_FACTORY_COST_GRAPH_RL_DETERMINISTIC")
    )


def _expected_identity():
    policy_id = os.environ.get(
        "SMART_FACTORY_COST_GUIDED_POLICY_ID", "").strip() or None
    reward_hash = os.environ.get(
        "SMART_FACTORY_EXPECTED_REWARD_PROFILE_SHA256", "").strip() or None
    return policy_id, reward_hash


def _observation_payload(observation: CostGuidedGraphObservation) -> dict:
    return {
        "feature_schema_sha256": observation.feature_schema_sha256,
        "prior_identity_sha256": observation.prior_identity_sha256,
        "current_time": float(observation.current_time),
        "matrix_shape": list(observation.matrix.values.shape),
        "coordinates": observation.coordinates.tolist(),
        "action_features": observation.action_features.tolist(),
        "action_mask": observation.action_mask.astype(np.uint8).tolist(),
        "no_op_index": int(observation.no_op_index),
    }


def _wait_result(name, started, identity, model_name):
    return SchedulerResult(
        [], None, time.perf_counter() - started, True, name, {
            "reason": "no_feasible_pair",
            "intentional_wait": True,
            "fallback": False,
            "model": model_name,
            "policy_id": identity.policy_id,
            "objective": identity.objective,
            "prior_identity_sha256": identity.sha256,
        })


class _CostGuidedScheduler(BaseScheduler):
    uses_traffic = True

    def __init__(self, model):
        super().__init__(model.prior_identity.policy_id)
        self.model = model
        self.cost_prior_identity = model.prior_identity
        self.objective = model.prior_identity.objective
        self.expert_coefficient_bundle = (
            model.prior_identity.primary_bundle)
        self.expert_coefficient_sha256 = (
            model.prior_identity.primary_bundle_sha256)
        self.pure_policy_required = _enabled(
            "SMART_FACTORY_PURE_POLICY_REQUIRED")

    def assign_task(self, pending_tasks, robot_states, congestion_map=None):
        result = self.assign(
            pending_tasks, robot_states,
            SchedulingContext(congestion_map=congestion_map))
        if not result.assignments:
            return None
        assignment = result.assignments[0]
        return assignment.robot_id, assignment.task

    def _observation(self, robot_states, pending_tasks, context):
        return build_cost_guided_graph_observation(
            robot_states, pending_tasks, context,
            self.cost_prior_identity)

    def _base_diagnostics(self):
        return {
            "policy_id": self.cost_prior_identity.policy_id,
            "objective": self.cost_prior_identity.objective,
            "prior_identity_sha256": self.cost_prior_identity.sha256,
            "primary_policy": self.cost_prior_identity.primary_policy,
            "primary_bundle_sha256":
                self.cost_prior_identity.primary_bundle_sha256,
            "secondary_policy": self.cost_prior_identity.secondary_policy,
            "secondary_bundle_sha256":
                self.cost_prior_identity.secondary_bundle_sha256,
            "feature_schema_sha256":
                COST_GUIDED_GRAPH_SCHEMA_SHA256,
            "reward_profile_version":
                self.model.provenance.reward_profile_version,
            "reward_profile_sha256":
                self.model.provenance.reward_profile_sha256,
            "training_phase": self.model.provenance.training_phase,
        }


class CostGraphPPOScheduler(_CostGuidedScheduler):
    def __init__(self, model_path, seed=42):
        del seed
        policy_id, reward_hash = _expected_identity()
        super().__init__(CostGraphPPOModel.load(
            model_path, expected_policy_id=policy_id,
            expected_reward_profile_sha256=reward_hash))

    def assign(self, pending_tasks, robot_states,
               context: SchedulingContext | None = None):
        started = time.perf_counter()
        context = context or SchedulingContext()
        try:
            observation = self._observation(
                robot_states, pending_tasks, context)
            if observation.edge_count == 0:
                return _wait_result(
                    self.name, started, self.cost_prior_identity,
                    "CostGraphPPO")
            physical = _fine_tune_enabled()
            explore = _exploration_enabled()
            action, log_probability, value = self.model.select_action(
                observation, deterministic=not explore)
            coordinate = observation.coordinate_for_action(action)
            if coordinate is None:
                raise ValueError("CostGraphPPO selected a masked no-op")
            row, column = coordinate
            diagnostics = {
                **self._base_diagnostics(),
                "model": "CostGraphPPO",
                "decoder": (
                    "on_policy_sample" if explore
                    else "deterministic_policy_argmax"),
                "exploration_enabled": explore,
                "action_semantics":
                    "variable_feasible_edge_plus_masked_noop",
                "selected_edge_index": int(action),
                "selected_probability": float(
                    math_exp(log_probability)),
            }
            if physical:
                diagnostics.update({
                    "physical_fine_tune": True,
                    "physical_rollout_step": {
                        "kind": "cost_graph_ppo",
                        "observation": _observation_payload(observation),
                        "action_index": int(action),
                        "old_log_probability": float(log_probability),
                        "value": float(value),
                        "selected_robot_id": int(
                            observation.matrix.robot_ids[row]),
                        "selected_task_id": int(
                            observation.matrix.tasks[column].task_id),
                    },
                })
            return _result_from_matching(
                self.name, observation.matrix, [(row, column)],
                pending_tasks, robot_states, context, started, diagnostics)
        except Exception as exc:
            return SchedulerResult(
                [], None, time.perf_counter() - started, False, self.name, {
                    **self._base_diagnostics(),
                    "reason":
                        f"cost_graph_ppo_inference_error:{type(exc).__name__}",
                })


class GraphRainbowScheduler(_CostGuidedScheduler):
    def __init__(self, model_path, seed=42):
        del seed
        policy_id, reward_hash = _expected_identity()
        super().__init__(GraphRainbowModel.load(
            model_path, expected_policy_id=policy_id,
            expected_reward_profile_sha256=reward_hash))

    def assign(self, pending_tasks, robot_states,
               context: SchedulingContext | None = None):
        started = time.perf_counter()
        context = context or SchedulingContext()
        try:
            observation = self._observation(
                robot_states, pending_tasks, context)
            physical = _fine_tune_enabled()
            explore = _exploration_enabled()
            if observation.edge_count == 0:
                return _wait_result(
                    self.name, started, self.cost_prior_identity,
                    "GraphRainbow")
            q_values = self.model.q_values(
                observation, noisy=explore)
            action = masked_argmax(
                q_values, observation.action_mask)
            coordinate = observation.coordinate_for_action(action)
            if coordinate is None:
                raise ValueError("GraphRainbow selected a masked no-op")
            row, column = coordinate
            diagnostics = {
                **self._base_diagnostics(),
                "model": "GraphRainbow",
                "decoder": (
                    "noisy_policy_argmax" if explore
                    else "deterministic_policy_argmax"),
                "exploration_enabled": explore,
                "action_semantics":
                    "variable_feasible_edge_plus_masked_noop",
                "selected_edge_index": int(action),
                "selected_q_value": float(q_values[action]),
            }
            if physical:
                diagnostics.update({
                    "physical_fine_tune": True,
                    "physical_rollout_step": {
                        "kind": "graph_rainbow",
                        "observation": _observation_payload(observation),
                        "action_index": int(action),
                        "selected_q_value": float(q_values[action]),
                        "selected_robot_id": int(
                            observation.matrix.robot_ids[row]),
                        "selected_task_id": int(
                            observation.matrix.tasks[column].task_id),
                    },
                })
            return _result_from_matching(
                self.name, observation.matrix, [(row, column)],
                pending_tasks, robot_states, context, started, diagnostics)
        except Exception as exc:
            return SchedulerResult(
                [], None, time.perf_counter() - started, False, self.name, {
                    **self._base_diagnostics(),
                    "reason":
                        f"graph_rainbow_inference_error:{type(exc).__name__}",
                })


def math_exp(log_probability):
    value = float(np.exp(float(log_probability)))
    if not np.isfinite(value):
        raise ValueError("policy probability is non-finite")
    return value
