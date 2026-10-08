"""Fail-closed eligibility rules for reporting a scheduler as a pure policy."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Mapping


PURE_POLICY_COUNTERS = (
    "fallback_scheduler_commits",
    "invalid_scheduler_outputs",
    "scheduler_timeout_count",
)


def _nonnegative_counter(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a non-negative integer")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(f"{field} must be a non-negative integer")
    if isinstance(value, Integral):
        return int(value)
    return int(numeric)


def pure_policy_eligibility(counters: Mapping[str, object]) -> dict:
    """Return a stable, auditable pure-policy decision.

    Missing counters fail closed.  Present counters must be finite,
    non-negative integers so corrupt metrics cannot silently enter rankings.
    """
    if not isinstance(counters, Mapping):
        raise ValueError("pure-policy counters must be a mapping")
    reasons = []
    normalized = {}
    for field in PURE_POLICY_COUNTERS:
        if field not in counters:
            reasons.append({
                "code": "missing_counter",
                "counter": field,
                "value": None,
            })
            continue
        value = _nonnegative_counter(counters[field], field)
        normalized[field] = value
        if value:
            reasons.append({
                "code": "nonzero_counter",
                "counter": field,
                "value": value,
            })
    return {
        "pure_policy_eligible": not reasons,
        "pure_policy_exclusion_reasons": reasons,
        "pure_policy_counters": normalized,
    }
