"""Shared contracts for advanced AI schedulers.

This module deliberately contains no navigation or motion logic.  It owns the
strict checkpoint envelope and numerical action-mask helpers used by the
assignment policies in :mod:`advanced_rl_agents` and
:mod:`advanced_rl_schedulers`.
"""

import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from config import RL_ENVIRONMENT_VERSION
from schedulers import ModelValidationError


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_METADATA_KEY = "metadata"


def _normalise_metadata(metadata: Mapping) -> dict:
    result = dict(metadata)
    required = {
        "algorithm", "environment_version", "state_dim", "action_dim",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"checkpoint metadata missing: {missing}")
    result.setdefault("schema_version", CHECKPOINT_SCHEMA_VERSION)
    if result["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if not isinstance(result["algorithm"], str) or not result["algorithm"]:
        raise ValueError("checkpoint algorithm must be a non-empty string")
    if result["environment_version"] != RL_ENVIRONMENT_VERSION:
        raise ValueError("checkpoint environment version mismatch")
    for name in ("state_dim", "action_dim"):
        value = result[name]
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"checkpoint {name} must be a positive integer") from exc
        if isinstance(value, bool) or integer != value or integer <= 0:
            raise ValueError(f"checkpoint {name} must be a positive integer")
        result[name] = integer
    return result


def _validated_array(name: str, value) -> np.ndarray:
    array = np.asarray(value)
    if name == CHECKPOINT_METADATA_KEY:
        raise ValueError(f"{CHECKPOINT_METADATA_KEY!r} is a reserved array name")
    if array.dtype.kind not in "biuf":
        raise ValueError(f"checkpoint array {name} must be real numeric")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"checkpoint array {name} contains non-finite values")
    return np.array(array, copy=True)


def save_checkpoint(path, metadata: Mapping,
                    arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically save a finite, non-pickle NPZ checkpoint."""
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("checkpoint path must end with .npz")
    checked_metadata = _normalise_metadata(metadata)
    checked_arrays = {
        str(name): _validated_array(str(name), value)
        for name, value in arrays.items()
    }
    if not checked_arrays:
        raise ValueError("checkpoint must contain at least one parameter array")
    checked_arrays[CHECKPOINT_METADATA_KEY] = np.asarray(
        json.dumps(checked_metadata, sort_keys=True, allow_nan=False),
        dtype=np.str_)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **checked_arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, *, expected_algorithm: str,
                    expected_state_dim: Optional[int] = None,
                    expected_action_dim: Optional[int] = None
                    ) -> Tuple[dict, Dict[str, np.ndarray]]:
    """Load and validate the common NPZ envelope without enabling pickle."""
    if not path:
        raise ModelValidationError("model checkpoint path is required")
    source = Path(path)
    if source.suffix.lower() != ".npz" or not source.is_file():
        raise ModelValidationError(f"model checkpoint not found: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if CHECKPOINT_METADATA_KEY not in archive.files:
                raise ModelValidationError("checkpoint metadata is missing")
            metadata = json.loads(str(
                archive[CHECKPOINT_METADATA_KEY].item()))
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
                if name != CHECKPOINT_METADATA_KEY
            }
    except ModelValidationError:
        raise
    except Exception as exc:
        raise ModelValidationError(
            f"cannot read model checkpoint: {type(exc).__name__}") from exc
    try:
        metadata = _normalise_metadata(metadata)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(str(exc)) from exc
    if metadata["algorithm"] != expected_algorithm:
        raise ModelValidationError(
            f"checkpoint algorithm {metadata['algorithm']!r} is not "
            f"{expected_algorithm!r}")
    expected = (
        ("state_dim", expected_state_dim),
        ("action_dim", expected_action_dim),
    )
    for name, value in expected:
        if value is not None and metadata[name] != int(value):
            raise ModelValidationError(
                f"checkpoint {name} {metadata[name]} != expected {int(value)}")
    if not arrays:
        raise ModelValidationError("checkpoint parameter arrays are missing")
    try:
        arrays = {
            name: _validated_array(name, value)
            for name, value in arrays.items()
        }
    except ValueError as exc:
        raise ModelValidationError(str(exc)) from exc
    return metadata, arrays


def validate_action_mask(mask, action_dim: Optional[int] = None) -> np.ndarray:
    """Return a one-dimensional boolean mask containing a legal action."""
    result = np.asarray(mask)
    if result.ndim != 1:
        raise ValueError("action mask must be one-dimensional")
    if action_dim is not None and result.shape != (int(action_dim),):
        raise ValueError(
            f"action mask shape {result.shape} != ({int(action_dim)},)")
    if result.dtype.kind not in "biuf":
        raise ValueError("action mask must be boolean or numeric")
    if result.dtype.kind == "f" and not np.all(np.isfinite(result)):
        raise ValueError("action mask contains non-finite values")
    result = result.astype(bool, copy=False)
    if not np.any(result):
        raise ValueError("action mask has no legal action")
    return result


def masked_argmax(values, mask) -> int:
    values = np.asarray(values, dtype=np.float64)
    legal = validate_action_mask(mask, values.size)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("action values must be a finite one-dimensional vector")
    indices = np.flatnonzero(legal)
    return int(indices[np.argmax(values[indices])])


def masked_softmax(logits, mask) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    legal = validate_action_mask(mask, logits.size)
    if logits.ndim != 1 or not np.all(np.isfinite(logits)):
        raise ValueError("action logits must be a finite one-dimensional vector")
    indices = np.flatnonzero(legal)
    shifted = logits[indices] - float(np.max(logits[indices]))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("masked softmax normalisation failed")
    result = np.zeros_like(logits, dtype=np.float64)
    result[indices] = weights / total
    return result
