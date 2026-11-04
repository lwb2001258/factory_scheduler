"""Versioned, objective-specific coefficients for the traffic expert.

The coefficients only rank already-feasible robot/task edges.  They never
change path feasibility, matching constraints, or physical safety handling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping


COEFFICIENT_VERSION = "traffic-expert-coefficients-v1"
CONFIGURATION_KEY = "expert_coefficient_bundle"
OBJECTIVES = ("count", "utility_v2")


@dataclass(frozen=True)
class ExpertCostCoefficients:
    objective: str
    travel: float = 1.0
    priority_bonus: float = 0.0
    waiting_bonus: float = 0.0
    congestion_penalty: float = 0.05
    deadline_pressure_bonus: float = 0.0
    traffic_expected_wait_penalty: float = 0.22
    traffic_recent_wait_penalty: float = 0.022
    traffic_stop_event_penalty: float = 0.22

    def __post_init__(self):
        if self.objective not in OBJECTIVES:
            raise ValueError("expert coefficient objective is invalid")
        values = asdict(self)
        values.pop("objective")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(float(value)) or float(value) < 0.0
               for value in values.values()):
            raise ValueError("expert coefficients must be finite and non-negative")
        if not math.isclose(float(self.travel), 1.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("travel coefficient is the fixed scale anchor")
        if self.objective == "count" and any(
                float(value) != 0.0 for value in (
                    self.priority_bonus, self.waiting_bonus,
                    self.deadline_pressure_bonus)):
            raise ValueError("count expert cannot contain utility bonuses")

    def to_dict(self) -> dict:
        return {key: (value if key == "objective" else float(value))
                for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, value: Mapping):
        if not isinstance(value, Mapping):
            raise ValueError("expert coefficient row must be an object")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError("expert coefficient fields are incompatible")
        return cls(**dict(value))


def default_coefficients(objective: str) -> ExpertCostCoefficients:
    if objective == "count":
        return ExpertCostCoefficients(objective="count")
    if objective == "utility_v2":
        return ExpertCostCoefficients(
            objective="utility_v2", priority_bonus=1.0,
            waiting_bonus=0.01, deadline_pressure_bonus=2.0)
    raise ValueError("expert coefficient objective is invalid")


def default_bundle() -> dict:
    return {
        "version": COEFFICIENT_VERSION,
        "objectives": {
            objective: default_coefficients(objective).to_dict()
            for objective in OBJECTIVES},
        "optimization": {
            "method": "default-hand-tuned",
            "training_enabled": False,
        },
    }


def canonical_bundle(value: Mapping | None) -> dict:
    if value is None:
        value = default_bundle()
    if not isinstance(value, Mapping) or value.get("version") != COEFFICIENT_VERSION:
        raise ValueError("expert coefficient bundle version is incompatible")
    objectives = value.get("objectives")
    if not isinstance(objectives, Mapping) or set(objectives) != set(OBJECTIVES):
        raise ValueError("expert coefficient bundle objectives are incomplete")
    normalized = {
        objective: ExpertCostCoefficients.from_dict(
            objectives[objective]).to_dict()
        for objective in OBJECTIVES}
    optimization = value.get("optimization")
    if not isinstance(optimization, Mapping):
        raise ValueError("expert coefficient optimization identity is missing")
    method = optimization.get("method")
    training_enabled = optimization.get("training_enabled")
    if not isinstance(method, str) or not method or not isinstance(training_enabled, bool):
        raise ValueError("expert coefficient optimization identity is invalid")
    return {
        "version": COEFFICIENT_VERSION,
        "objectives": normalized,
        "optimization": {
            "method": method,
            "training_enabled": training_enabled,
        },
    }


def bundle_sha256(value: Mapping | None) -> str:
    normalized = canonical_bundle(value)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def coefficients_for(value: Mapping | None,
                     objective: str) -> ExpertCostCoefficients:
    normalized = canonical_bundle(value)
    return ExpertCostCoefficients.from_dict(
        normalized["objectives"][objective])


def configured_bundle(configuration: Mapping | None) -> dict | None:
    if not isinstance(configuration, Mapping) or CONFIGURATION_KEY not in configuration:
        return None
    return canonical_bundle(configuration[CONFIGURATION_KEY])


def with_optimized_rows(rows: Mapping[str, Mapping], *, method: str) -> dict:
    if set(rows) != set(OBJECTIVES):
        raise ValueError("optimized coefficient rows are incomplete")
    value = {
        "version": COEFFICIENT_VERSION,
        "objectives": dict(rows),
        "optimization": {"method": method, "training_enabled": False},
    }
    return canonical_bundle(value)


def search_fields(objective: str) -> tuple[str, ...]:
    shared = (
        "congestion_penalty",
        "traffic_expected_wait_penalty",
        "traffic_recent_wait_penalty",
        "traffic_stop_event_penalty",
    )
    if objective == "count":
        return shared
    if objective == "utility_v2":
        return (
            "priority_bonus", "waiting_bonus", "deadline_pressure_bonus",
            *shared)
    raise ValueError("expert coefficient objective is invalid")


def row_from_log_multipliers(objective: str, multipliers) -> dict:
    return row_from_baseline_log_multipliers(
        objective, default_coefficients(objective).to_dict(), multipliers)


def row_from_baseline_log_multipliers(
        objective: str, baseline: Mapping, multipliers, *,
        minimum_multiplier: float = 0.1,
        maximum_multiplier: float = 10.0) -> dict:
    """Apply bounded log multipliers to a validated warm-start row.

    The fixed travel anchor and objective-isolation rules are revalidated after
    the update.  Zero warm-start fields stay zero, so callers cannot enable a
    coefficient that the objective deliberately excludes.
    """
    fields = search_fields(objective)
    values = tuple(float(value) for value in multipliers)
    if len(values) != len(fields) or not all(math.isfinite(value) for value in values):
        raise ValueError("coefficient search vector is invalid")
    if (isinstance(minimum_multiplier, bool) or
            isinstance(maximum_multiplier, bool) or
            not isinstance(minimum_multiplier, (int, float)) or
            not isinstance(maximum_multiplier, (int, float)) or
            not math.isfinite(float(minimum_multiplier)) or
            not math.isfinite(float(maximum_multiplier)) or
            float(minimum_multiplier) <= 0.0 or
            float(minimum_multiplier) >= float(maximum_multiplier)):
        raise ValueError("coefficient multiplier bounds are invalid")
    baseline = ExpertCostCoefficients.from_dict(baseline).to_dict()
    if baseline["objective"] != objective:
        raise ValueError("coefficient baseline objective differs")
    lower = math.log(float(minimum_multiplier))
    upper = math.log(float(maximum_multiplier))
    for name, log_multiplier in zip(fields, values):
        clipped = min(upper, max(lower, log_multiplier))
        baseline[name] = float(baseline[name]) * math.exp(clipped)
    return ExpertCostCoefficients.from_dict(baseline).to_dict()


def traffic_penalty(traffic_features: Mapping[str, float],
                    coefficients: ExpertCostCoefficients) -> float:
    required = (
        "traffic_peer_fleet_fraction",
        "traffic_expected_wait_seconds",
        "traffic_peer_recent_wait_seconds",
        "traffic_peer_recent_stop_events",
    )
    if any(name not in traffic_features for name in required):
        raise ValueError("traffic features are incomplete")
    exposure = float(traffic_features["traffic_peer_fleet_fraction"])
    result = (
        coefficients.traffic_expected_wait_penalty *
        float(traffic_features["traffic_expected_wait_seconds"]) +
        exposure * (
            coefficients.traffic_recent_wait_penalty *
            float(traffic_features["traffic_peer_recent_wait_seconds"]) +
            coefficients.traffic_stop_event_penalty *
            float(traffic_features["traffic_peer_recent_stop_events"])))
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("learned traffic penalty is invalid")
    return result
