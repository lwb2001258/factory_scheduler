"""Frozen coefficient priors shared by CostGraphPPO and GraphRainbow.

This module reuses the canonical traffic expert implementation. It does not
own feasibility, rewards, dispatch, or physical safety. Returned matrices are
read-only model inputs derived from one scheduling snapshot and two versioned
coefficient bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

from expert_coefficients import bundle_sha256, canonical_bundle
from learning_scheduler import traffic_objective_expert_values
from schedulers import CostMatrix, SchedulingContext


COST_PRIOR_VERSION = "cost-guided-expert-prior-v1"
COST_PRIOR_NORMALIZATION = "median-mad-iqr-v1-clip8"
COST_GUIDED_POLICY_SPECS = {
    "CostGraphPPO_count": {
        "algorithm": "CostGraphPPO",
        "objective": "count",
        "primary": "GraphImitation_count",
        "secondary": "LearnedHungarian_count",
    },
    "CostGraphPPO_utility_v2": {
        "algorithm": "CostGraphPPO",
        "objective": "utility_v2",
        "primary": "GraphImitation_utility_v2",
        "secondary": "LearnedHungarian_utility_v2",
    },
    "GraphRainbow_count": {
        "algorithm": "GraphRainbow",
        "objective": "count",
        "primary": "LearnedHungarian_count",
        "secondary": "GraphImitation_count",
    },
    "GraphRainbow_utility_v2": {
        "algorithm": "GraphRainbow",
        "objective": "utility_v2",
        "primary": "LearnedHungarian_utility_v2",
        "secondary": "GraphImitation_utility_v2",
    },
}
EXPECTED_BUNDLE_SHA256 = {
    "GraphImitation_count":
        "04676a293d1710686569aac34d9acf88591db3e538a42bfb9f4d11be534f171e",
    "GraphImitation_utility_v2":
        "803c1eec98d5a0c4141922bb6955e5683f79acf00e9f4e7ab3a32f1de85e609e",
    "LearnedHungarian_count":
        "6a3c36f4c8b8cfe6ad4916ca4e4e18fcfa3f81db988268389575804633b4213d",
    "LearnedHungarian_utility_v2":
        "d1881055a0a5f26aaaf34821ca5503a0b36ccd415ce811550c93cfd5f3c54f36",
}


def _canonical_json(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False)


def _bundle_json(bundle: Mapping) -> tuple[str, str]:
    normalized = canonical_bundle(bundle)
    return _canonical_json(normalized), bundle_sha256(normalized)


@dataclass(frozen=True)
class CostPriorIdentity:
    """Immutable policy/objective/bundle identity embedded in checkpoints."""

    version: str
    policy_id: str
    algorithm: str
    objective: str
    primary_policy: str
    secondary_policy: str
    primary_bundle_json: str
    secondary_bundle_json: str
    primary_bundle_sha256: str
    secondary_bundle_sha256: str

    def __post_init__(self):
        spec = COST_GUIDED_POLICY_SPECS.get(self.policy_id)
        if (self.version != COST_PRIOR_VERSION or spec is None or
                self.algorithm != spec["algorithm"] or
                self.objective != spec["objective"] or
                self.primary_policy != spec["primary"] or
                self.secondary_policy != spec["secondary"]):
            raise ValueError("cost-prior policy identity is incompatible")
        try:
            primary_json, primary_hash = _bundle_json(
                json.loads(self.primary_bundle_json))
            secondary_json, secondary_hash = _bundle_json(
                json.loads(self.secondary_bundle_json))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("cost-prior bundle identity is invalid") from exc
        if (primary_json != self.primary_bundle_json or
                secondary_json != self.secondary_bundle_json or
                primary_hash != self.primary_bundle_sha256 or
                secondary_hash != self.secondary_bundle_sha256 or
                primary_hash != EXPECTED_BUNDLE_SHA256[self.primary_policy] or
                secondary_hash != EXPECTED_BUNDLE_SHA256[
                    self.secondary_policy]):
            raise ValueError("cost-prior bundle hash is incompatible")

    @classmethod
    def from_selected_bundles(cls, policy_id: str,
                              selected_bundles: Mapping):
        spec = COST_GUIDED_POLICY_SPECS.get(policy_id)
        if spec is None or not isinstance(selected_bundles, Mapping):
            raise ValueError("unknown cost-guided policy identity")
        rows = {}
        for role in ("primary", "secondary"):
            policy = spec[role]
            row = selected_bundles.get(policy)
            if (not isinstance(row, Mapping) or
                    not isinstance(row.get("bundle"), Mapping) or
                    not isinstance(row.get("bundle_sha256"), str)):
                raise ValueError(f"selected bundle is missing: {policy}")
            bundle_json, actual_hash = _bundle_json(row["bundle"])
            if (row["bundle_sha256"] != actual_hash or
                    EXPECTED_BUNDLE_SHA256[policy] != actual_hash):
                raise ValueError(f"selected bundle hash differs: {policy}")
            rows[role] = (bundle_json, actual_hash)
        return cls(
            COST_PRIOR_VERSION, policy_id, spec["algorithm"],
            spec["objective"], spec["primary"], spec["secondary"],
            rows["primary"][0], rows["secondary"][0],
            rows["primary"][1], rows["secondary"][1])

    @classmethod
    def from_metadata(cls, value: Mapping):
        if not isinstance(value, Mapping):
            raise ValueError("cost-prior metadata must be an object")
        expected = {
            "version", "normalization", "policy_id", "algorithm", "objective",
            "primary_policy", "secondary_policy", "primary_bundle",
            "secondary_bundle", "primary_bundle_sha256",
            "secondary_bundle_sha256", "identity_sha256",
        }
        if set(value) != expected:
            raise ValueError("cost-prior metadata fields are incompatible")
        if value["normalization"] != COST_PRIOR_NORMALIZATION:
            raise ValueError("cost-prior normalization identity differs")
        primary_json, _ = _bundle_json(value["primary_bundle"])
        secondary_json, _ = _bundle_json(value["secondary_bundle"])
        identity = cls(
            value["version"], value["policy_id"], value["algorithm"],
            value["objective"], value["primary_policy"],
            value["secondary_policy"], primary_json, secondary_json,
            value["primary_bundle_sha256"],
            value["secondary_bundle_sha256"])
        if value["identity_sha256"] != identity.sha256:
            raise ValueError("cost-prior metadata identity hash differs")
        return identity

    @property
    def primary_bundle(self) -> dict:
        return json.loads(self.primary_bundle_json)

    @property
    def secondary_bundle(self) -> dict:
        return json.loads(self.secondary_bundle_json)

    def _metadata_without_hash(self) -> dict:
        return {
            "version": self.version,
            "normalization": COST_PRIOR_NORMALIZATION,
            "policy_id": self.policy_id,
            "algorithm": self.algorithm,
            "objective": self.objective,
            "primary_policy": self.primary_policy,
            "secondary_policy": self.secondary_policy,
            "primary_bundle": self.primary_bundle,
            "secondary_bundle": self.secondary_bundle,
            "primary_bundle_sha256": self.primary_bundle_sha256,
            "secondary_bundle_sha256": self.secondary_bundle_sha256,
        }

    @property
    def sha256(self) -> str:
        payload = _canonical_json(self._metadata_without_hash()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_metadata(self) -> dict:
        result = self._metadata_without_hash()
        result["identity_sha256"] = self.sha256
        return result


def robust_cost_normalize(values, feasible) -> np.ndarray:
    """Robustly center legal costs without changing their ordering.

    Infeasible locations are zero and must still be masked by every consumer.
    The separate raw cost matrices remain available for identity checks.
    """
    values = np.asarray(values, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    if values.ndim != 2 or values.shape != feasible.shape:
        raise ValueError("cost-prior value and feasibility shapes differ")
    legal = values[feasible]
    result = np.zeros(values.shape, dtype=np.float32)
    if legal.size == 0:
        return result
    if not np.all(np.isfinite(legal)):
        raise ValueError("legal cost-prior values must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        median = float(np.median(legal))
        deviations = np.abs(legal - median)
        mad_scale = 1.4826 * float(np.median(deviations))
        q25, q75 = np.percentile(legal, (25.0, 75.0))
        iqr_scale = float(q75 - q25) / 1.349
        # Preserve contrast if a mostly tied set has one distinct edge.
        std_fallback = 0.1 * float(np.std(legal))
    if not all(np.isfinite(value) for value in (
            median, mad_scale, iqr_scale, std_fallback)):
        raise ValueError("cost-prior normalization statistics are non-finite")
    scale = max(mad_scale, iqr_scale, std_fallback, 1e-6)
    normalized = np.clip((legal - median) / scale, -8.0, 8.0)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalized cost-prior values must be finite")
    result[feasible] = normalized.astype(np.float32)
    return result


@dataclass(frozen=True)
class CostPriorMatrices:
    """One-snapshot primary/secondary costs and model transforms."""

    primary_cost: np.ndarray
    secondary_cost: np.ndarray
    primary_normalized: np.ndarray
    secondary_normalized: np.ndarray
    feasible: np.ndarray
    identity_sha256: str

    def __post_init__(self):
        mask = np.asarray(self.feasible, dtype=bool)
        arrays = tuple(np.asarray(value).copy() for value in (
            self.primary_cost, self.secondary_cost,
            self.primary_normalized, self.secondary_normalized))
        shape = mask.shape
        if (len(shape) != 2 or any(value.shape != shape for value in arrays)):
            raise ValueError("cost-prior matrices have incompatible shapes")
        if (any(not np.all(np.isfinite(value[mask])) for value in arrays) or
                any(np.any(value[~mask] != 0.0) for value in arrays)):
            raise ValueError("legal cost-prior matrices must be finite")
        try:
            valid_hash = (isinstance(self.identity_sha256, str) and
                          len(self.identity_sha256) == 64 and
                          int(self.identity_sha256, 16) >= 0)
        except ValueError:
            valid_hash = False
        if not valid_hash:
            raise ValueError("cost-prior matrix identity is invalid")
        mask = mask.copy()
        mask.setflags(write=False)
        for value in arrays:
            value.setflags(write=False)
        object.__setattr__(self, "primary_cost", arrays[0])
        object.__setattr__(self, "secondary_cost", arrays[1])
        object.__setattr__(self, "primary_normalized", arrays[2])
        object.__setattr__(self, "secondary_normalized", arrays[3])
        object.__setattr__(self, "feasible", mask)


def cost_prior_matrices(matrix: CostMatrix, context: SchedulingContext,
                        identity: CostPriorIdentity) -> CostPriorMatrices:
    """Evaluate both frozen teachers exactly once on one traffic snapshot."""
    if not isinstance(matrix, CostMatrix):
        raise ValueError("cost priors require a shared CostMatrix")
    if not isinstance(context, SchedulingContext):
        raise ValueError("cost priors require a SchedulingContext")
    if not isinstance(identity, CostPriorIdentity):
        raise ValueError("cost priors require a frozen identity")
    mask = np.asarray(matrix.feasible, dtype=bool)
    if mask.shape != matrix.values.shape:
        raise ValueError("cost-prior feasibility shape differs")
    primary_full = traffic_objective_expert_values(
        matrix, context, identity.objective,
        coefficient_bundle=identity.primary_bundle)
    secondary_full = traffic_objective_expert_values(
        matrix, context, identity.objective,
        coefficient_bundle=identity.secondary_bundle)
    primary_full = np.asarray(primary_full, dtype=np.float64)
    secondary_full = np.asarray(secondary_full, dtype=np.float64)
    if (primary_full.shape != mask.shape or
            secondary_full.shape != mask.shape or
            not np.all(np.isfinite(primary_full[mask])) or
            not np.all(np.isfinite(secondary_full[mask]))):
        raise ValueError("cost-prior expert matrices are invalid")
    primary = np.where(mask, primary_full, 0.0)
    secondary = np.where(mask, secondary_full, 0.0)
    return CostPriorMatrices(
        primary, secondary,
        robust_cost_normalize(primary, mask),
        robust_cost_normalize(secondary, mask),
        mask.copy(), identity.sha256)


def validate_policy_specs() -> None:
    required = {
        spec[role]
        for spec in COST_GUIDED_POLICY_SPECS.values()
        for role in ("primary", "secondary")}
    if required != set(EXPECTED_BUNDLE_SHA256):
        raise RuntimeError("cost-guided expected bundle hashes are incomplete")
    for policy_id, spec in COST_GUIDED_POLICY_SPECS.items():
        if (spec["objective"] not in {"count", "utility_v2"} or
                spec["primary"] == spec["secondary"] or
                not policy_id.startswith(spec["algorithm"])):
            raise RuntimeError(f"invalid cost-guided policy spec: {policy_id}")


validate_policy_specs()
