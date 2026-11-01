"""Execute the current GraphImitation teacher directly, without learning."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from learning_scheduler import (
    TRAFFIC_GRAPH_MODEL_VERSION, _LearnedMatchingScheduler,
    solve_hungarian, traffic_objective_expert_values,
)
from expert_coefficients import (
    bundle_sha256, canonical_bundle,
)
from schedulers import (
    CostMatrix, ModelValidationError, SchedulerResult, SchedulingContext,
    _result_from_matching, build_cost_matrix,
)
from traffic_features import TRAFFIC_VERSION


EXPERT_VERSION = "graph-imitation-direct-expert-v1"
OPTIMIZED_EXPERT_VERSION = "graph-imitation-direct-expert-v2-learned-coefficients"


def expert_descriptor(objective, coefficient_bundle=None):
    """Return a canonical analytical policy identity."""
    if objective not in ("count", "utility_v2"):
        raise ValueError("expert objective must be count or utility_v2")
    descriptor = {
        "version": EXPERT_VERSION,
        "teacher_version": TRAFFIC_GRAPH_MODEL_VERSION,
        "traffic_version": TRAFFIC_VERSION,
        "objective": objective,
        "training_enabled": False,
    }
    if coefficient_bundle is not None:
        bundle = canonical_bundle(coefficient_bundle)
        descriptor.update({
            "version": OPTIMIZED_EXPERT_VERSION,
            "coefficient_bundle": bundle,
            "coefficient_sha256": bundle_sha256(bundle),
            "coefficient_optimization": bundle["optimization"]["method"],
        })
    return descriptor


@dataclass(frozen=True)
class ExpertIdentity:
    objective: str
    descriptor_sha256: str
    coefficient_bundle: dict | None = None


class GraphExpertScheduler(_LearnedMatchingScheduler):
    """Use shared feasible costs, route traffic and exact teacher matching.

    Like GraphImitation, Webots commits the first pair then observes the new
    state before the next matching. Coefficients are learned offline and the
    descriptor is immutable during evaluation; no online update API exists.
    """

    uses_traffic = True
    pure_policy_required = True
    graph_expert = True

    def __init__(self, model_path):
        super().__init__("GraphExpert")
        for key in ("SMART_FACTORY_PHYSICAL_FINE_TUNE",
                    "SMART_FACTORY_GRAPH_PPO_FINE_TUNE"):
            if os.environ.get(key, "0").strip().lower() in {"1", "true", "yes", "on"}:
                raise ModelValidationError("GraphExpert is evaluation-only")
        try:
            raw = Path(model_path).read_bytes()
            descriptor = json.loads(raw)
            if not isinstance(descriptor, dict):
                raise ValueError("descriptor must be a JSON object")
            coefficient_bundle = descriptor.get("coefficient_bundle")
            if descriptor != expert_descriptor(
                    descriptor["objective"], coefficient_bundle):
                raise ValueError("descriptor does not match the current expert")
            # False must be a JSON boolean, not an integer accepted by ==.
            if descriptor["training_enabled"] is not False:
                raise ValueError("training must be disabled")
        except (OSError, TypeError, KeyError, ValueError) as exc:
            raise ModelValidationError(f"invalid GraphExpert descriptor: {exc}") from exc
        self.model_path = str(model_path)
        self.identity = ExpertIdentity(
            descriptor["objective"], hashlib.sha256(raw).hexdigest(),
            (canonical_bundle(coefficient_bundle)
             if coefficient_bundle is not None else None))

    def evaluation_metadata(self):
        return {
            **expert_descriptor(
                self.identity.objective, self.identity.coefficient_bundle),
            "descriptor_sha256": self.identity.descriptor_sha256,
            "parameter_updates": 0,
        }

    def assign(self, pending_tasks, robot_states, context=None):
        context = context or SchedulingContext()
        started = time.perf_counter()
        base = build_cost_matrix(pending_tasks, robot_states, context)
        if not np.any(base.feasible):
            return SchedulerResult(
                computation_time=time.perf_counter() - started,
                is_feasible=True, algorithm_name=self.name,
                diagnostics={"reason": "no_feasible_pair", "intentional_wait": True})
        values = traffic_objective_expert_values(
            base, context, self.identity.objective,
            coefficient_bundle=self.identity.coefficient_bundle)
        expert = CostMatrix(base.robot_ids, base.tasks, values, base.feasible)
        pairs = solve_hungarian(values, base.feasible)
        return _result_from_matching(
            self.name, expert, pairs, pending_tasks, robot_states, context,
            started, {"solver": "scipy.linear_sum_assignment",
                      "expert_version": expert_descriptor(
                          self.identity.objective,
                          self.identity.coefficient_bundle)["version"],
                      "objective": self.identity.objective,
                      "parameter_updates": 0})


class TrafficHungarianScheduler(GraphExpertScheduler):
    """Nonlearning Hungarian baseline sharing the exact route-traffic expert.

    The descriptor binds the objective and optional offline-learned coefficient
    bundle. Reuse the analytical solver so GraphImitation and Ridge have one
    traffic teacher.
    """

    def __init__(self, model_path):
        super().__init__(model_path)
        self.name = "TrafficHungarian"

    def evaluation_metadata(self):
        return {**super().evaluation_metadata(),
                "algorithm": "Hungarian",
                "cost_function": "traffic_objective_expert_values",
                "learning_enabled": False}
