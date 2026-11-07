"""Cost-guided variable-edge GraphPPO model and strict checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Mapping

import numpy as np

from advanced_ai_common import (
    load_checkpoint, masked_argmax, masked_softmax, save_checkpoint,
)
from advanced_rl_agents import Adam
from config import RL_ENVIRONMENT_VERSION
from cost_guided_graph import CostPriorIdentity
from cost_guided_graph_observation import (
    COST_GUIDED_GRAPH_FEATURE_DIM,
    COST_GUIDED_GRAPH_FEATURE_NAMES,
    COST_GUIDED_GRAPH_FEATURE_VERSION,
    COST_GUIDED_GRAPH_SCHEMA_SHA256,
    CostGuidedGraphObservation,
)
from schedulers import ModelValidationError


COST_GRAPH_PPO_ALGORITHM = "CostGraphPPO"
COST_GRAPH_PPO_MODEL_VERSION = "cost-graph-ppo-v1-bipartite-message"
COST_GRAPH_PPO_ACTION_SEMANTICS = "variable_feasible_edge_plus_masked_noop"
TRAINING_PHASES = ("standalone", "webots_fine_tuned")
PRIMARY_NORMALIZED_INDEX = COST_GUIDED_GRAPH_FEATURE_NAMES.index(
    "primary_cost_normalized")


def _positive_integer(name, value, minimum=1) -> int:
    if (isinstance(value, bool) or
            not isinstance(value, (int, np.integer)) or
            int(value) < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_number(name, value, *, positive=False,
                   nonnegative=False) -> float:
    if (isinstance(value, bool) or
            not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if (not math.isfinite(result) or
            (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        raise ValueError(f"{name} has an invalid value")
    return result


def _probability(name, value, *, positive=False) -> float:
    result = _finite_number(name, value)
    if result > 1.0 or result < (1e-15 if positive else 0.0):
        raise ValueError(f"{name} must be within the unit interval")
    return result


def _valid_sha256(value) -> bool:
    return bool(
        isinstance(value, str) and len(value) == 64 and
        all(character in "0123456789abcdef" for character in value))


def _strict_seeds(name, values, *, allow_empty=False) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a seed sequence")
    result = tuple(
        _positive_integer(f"{name} seed", value, minimum=0)
        for value in values)
    if (not allow_empty and not result) or len(set(result)) != len(result):
        raise ValueError(f"{name} seeds are empty or duplicated")
    return result


@dataclass(frozen=True)
class CostGraphPPOConfig:
    hidden_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 0.0003
    update_epochs: int = 4
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 5.0
    value_huber_delta: float = 10.0
    prior_logit_scale: float = 1.0

    def __post_init__(self):
        _positive_integer("hidden_size", self.hidden_size)
        _positive_integer("update_epochs", self.update_epochs)
        _probability("gamma", self.gamma)
        _probability("gae_lambda", self.gae_lambda)
        _probability("clip_epsilon", self.clip_epsilon, positive=True)
        _finite_number("learning_rate", self.learning_rate, positive=True)
        _finite_number(
            "value_coefficient", self.value_coefficient, nonnegative=True)
        _finite_number(
            "entropy_coefficient", self.entropy_coefficient,
            nonnegative=True)
        _finite_number("max_grad_norm", self.max_grad_norm, positive=True)
        _finite_number(
            "value_huber_delta", self.value_huber_delta, positive=True)
        _finite_number(
            "prior_logit_scale", self.prior_logit_scale,
            positive=True)


@dataclass(frozen=True)
class CostGraphPPOProvenance:
    objective: str
    reward_profile_version: str
    reward_profile_sha256: str
    training_phase: str
    standalone_seeds: tuple[int, ...]
    webots_fine_tune_seeds: tuple[int, ...] = ()

    def __post_init__(self):
        if (self.objective not in {"count", "utility_v2"} or
                not isinstance(self.reward_profile_version, str) or
                not self.reward_profile_version or
                not _valid_sha256(self.reward_profile_sha256) or
                self.training_phase not in TRAINING_PHASES):
            raise ValueError("CostGraphPPO provenance identity is invalid")
        standalone = _strict_seeds(
            "standalone", self.standalone_seeds)
        fine_tune = _strict_seeds(
            "webots fine-tune", self.webots_fine_tune_seeds,
            allow_empty=True)
        if set(standalone).intersection(fine_tune):
            raise ValueError("CostGraphPPO seed partitions overlap")
        if self.training_phase == "standalone" and fine_tune:
            raise ValueError("standalone checkpoint cannot contain fine-tune seeds")
        if self.training_phase == "webots_fine_tuned" and not fine_tune:
            raise ValueError("fine-tuned checkpoint requires Webots seeds")
        object.__setattr__(self, "standalone_seeds", standalone)
        object.__setattr__(self, "webots_fine_tune_seeds", fine_tune)

    def to_metadata(self) -> dict:
        return {
            "objective": self.objective,
            "reward_profile_version": self.reward_profile_version,
            "reward_profile_sha256": self.reward_profile_sha256,
            "training_phase": self.training_phase,
            "standalone_seeds": list(self.standalone_seeds),
            "webots_fine_tune_seeds": list(self.webots_fine_tune_seeds),
        }

    @classmethod
    def from_metadata(cls, value: Mapping):
        expected = {
            "objective", "reward_profile_version", "reward_profile_sha256",
            "training_phase", "standalone_seeds",
            "webots_fine_tune_seeds",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("CostGraphPPO provenance fields differ")
        return cls(**dict(value))


def _transform_features(features) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if (values.ndim != 2 or
            values.shape[1] != COST_GUIDED_GRAPH_FEATURE_DIM or
            not np.all(np.isfinite(values))):
        raise ValueError("CostGraphPPO action features are invalid")
    return np.sign(values) * np.log1p(np.abs(values))


def _message_inputs(observation: CostGuidedGraphObservation) -> np.ndarray:
    """Concatenate edge, robot-neighbour, task-neighbour and global messages."""
    x = _transform_features(observation.action_features)
    edge_count = observation.edge_count
    global_message = np.mean(
        x[:edge_count] if edge_count else x, axis=0)
    output = np.zeros(
        (observation.action_count, 4 * COST_GUIDED_GRAPH_FEATURE_DIM),
        dtype=np.float32)
    if edge_count:
        coordinates = observation.coordinates
        for index, (row, column) in enumerate(coordinates):
            row_values = x[:edge_count][coordinates[:, 0] == row]
            column_values = x[:edge_count][coordinates[:, 1] == column]
            output[index] = np.concatenate((
                x[index],
                np.mean(row_values, axis=0),
                np.mean(column_values, axis=0),
                global_message))
        output[observation.no_op_index] = np.concatenate((
            x[observation.no_op_index],
            np.zeros_like(global_message),
            np.zeros_like(global_message),
            global_message))
    else:
        output[0] = np.concatenate((
            x[0], np.zeros_like(global_message),
            np.zeros_like(global_message), global_message))
    if not np.all(np.isfinite(output)):
        raise ValueError("CostGraphPPO graph messages are non-finite")
    return output


class CostGraphPPONetwork:
    def __init__(self, config: CostGraphPPOConfig,
                 rng: np.random.Generator):
        self.config = config
        input_dim = 4 * COST_GUIDED_GRAPH_FEATURE_DIM
        scale = math.sqrt(2.0 / input_dim)
        self.params = {
            "w1": rng.normal(
                0.0, scale, (input_dim, config.hidden_size)
            ).astype(np.float32),
            "b1": np.zeros(config.hidden_size, dtype=np.float32),
            # Zero heads make the initial actor exactly the frozen cost prior
            # and the initial critic neutral.
            "wa": np.zeros(config.hidden_size, dtype=np.float32),
            "ba": np.zeros(1, dtype=np.float32),
            "wv": np.zeros(config.hidden_size, dtype=np.float32),
            "bv": np.zeros(1, dtype=np.float32),
        }

    def forward(self, observation: CostGuidedGraphObservation, *,
                cache=False):
        if not isinstance(observation, CostGuidedGraphObservation):
            raise ValueError("CostGraphPPO requires its graph observation")
        augmented = _message_inputs(observation)
        hidden = np.tanh(
            augmented @ self.params["w1"] + self.params["b1"])
        residual_logits = hidden @ self.params["wa"] + self.params["ba"][0]
        primary = observation.action_features[:, PRIMARY_NORMALIZED_INDEX]
        logits = (
            residual_logits -
            self.config.prior_logit_scale * primary)
        probabilities = masked_softmax(logits, observation.action_mask)
        pool_indices = np.flatnonzero(observation.action_mask)
        pooled = np.mean(hidden[pool_indices], axis=0)
        value = float(pooled @ self.params["wv"] + self.params["bv"][0])
        if (not np.all(np.isfinite(logits)) or
                not np.all(np.isfinite(probabilities)) or
                not math.isfinite(value)):
            raise ValueError("CostGraphPPO forward pass is non-finite")
        if not cache:
            return probabilities, value, logits
        return probabilities, value, logits, (
            augmented, hidden, pooled, pool_indices)

    def backward(self, cache, grad_logits, grad_value) -> dict:
        augmented, hidden, pooled, pool_indices = cache
        grad_logits = np.asarray(grad_logits, dtype=np.float32)
        if (grad_logits.shape != (len(hidden),) or
                not np.all(np.isfinite(grad_logits)) or
                not math.isfinite(float(grad_value)) or
                len(pool_indices) == 0):
            raise ValueError("CostGraphPPO gradients are invalid")
        grads = {
            "wa": hidden.T @ grad_logits,
            "ba": np.asarray([np.sum(grad_logits)], dtype=np.float32),
            "wv": pooled * float(grad_value),
            "bv": np.asarray([grad_value], dtype=np.float32),
        }
        hidden_gradient = grad_logits[:, None] * self.params["wa"][None, :]
        hidden_gradient[pool_indices] += (
            float(grad_value) / len(pool_indices)
            * self.params["wv"][None, :])
        z_gradient = hidden_gradient * (1.0 - hidden ** 2)
        grads["w1"] = augmented.T @ z_gradient
        grads["b1"] = np.sum(z_gradient, axis=0)
        if any(value.shape != self.params[name].shape or
               not np.all(np.isfinite(value))
               for name, value in grads.items()):
            raise ValueError("CostGraphPPO parameter gradients are invalid")
        return grads


class CostGraphPPOModel:
    def __init__(
            self, prior_identity: CostPriorIdentity,
            provenance: CostGraphPPOProvenance,
            config: CostGraphPPOConfig | None = None, seed: int = 42):
        if (not isinstance(prior_identity, CostPriorIdentity) or
                prior_identity.algorithm != COST_GRAPH_PPO_ALGORITHM):
            raise ValueError("CostGraphPPO prior identity is incompatible")
        if (not isinstance(provenance, CostGraphPPOProvenance) or
                provenance.objective != prior_identity.objective):
            raise ValueError("CostGraphPPO provenance is required")
        self.config = config or CostGraphPPOConfig()
        self.prior_identity = prior_identity
        self.provenance = provenance
        self.seed = _positive_integer("seed", seed, minimum=0)
        self.rng = np.random.default_rng(self.seed)
        self.network = CostGraphPPONetwork(self.config, self.rng)
        self.optimizer = Adam(
            self.network.params, self.config.learning_rate,
            self.config.max_grad_norm)
        self.training_step = 0
        self.training_episodes = 0

    def validate_observation(
            self, observation: CostGuidedGraphObservation) -> None:
        if (not isinstance(observation, CostGuidedGraphObservation) or
                observation.feature_schema_sha256 !=
                COST_GUIDED_GRAPH_SCHEMA_SHA256 or
                observation.prior_identity_sha256 !=
                self.prior_identity.sha256):
            raise ValueError("CostGraphPPO observation identity differs")

    def policy(self, observation: CostGuidedGraphObservation):
        self.validate_observation(observation)
        return self.network.forward(observation)

    def select_action(self, observation: CostGuidedGraphObservation, *,
                      deterministic: bool):
        self.validate_observation(observation)
        probabilities, value, logits = self.network.forward(observation)
        if deterministic:
            index = masked_argmax(logits, observation.action_mask)
        else:
            index = int(self.rng.choice(
                observation.action_count, p=probabilities))
        log_probability = math.log(max(float(probabilities[index]), 1e-12))
        return index, log_probability, value

    def mark_webots_fine_tuned(self, seeds) -> None:
        if self.provenance.training_phase != "standalone":
            raise ValueError("CostGraphPPO checkpoint is already fine-tuned")
        self.provenance = replace(
            self.provenance, training_phase="webots_fine_tuned",
            webots_fine_tune_seeds=tuple(seeds))

    def save(self, path) -> None:
        metadata = {
            "algorithm": COST_GRAPH_PPO_ALGORITHM,
            "environment_version": RL_ENVIRONMENT_VERSION,
            "state_dim": COST_GUIDED_GRAPH_FEATURE_DIM,
            "action_dim": 1,
            "model_version": COST_GRAPH_PPO_MODEL_VERSION,
            "action_semantics": COST_GRAPH_PPO_ACTION_SEMANTICS,
            "feature_version": COST_GUIDED_GRAPH_FEATURE_VERSION,
            "feature_schema_sha256": COST_GUIDED_GRAPH_SCHEMA_SHA256,
            "config": asdict(self.config),
            "prior_identity": self.prior_identity.to_metadata(),
            "provenance": self.provenance.to_metadata(),
            "seed": self.seed,
            "rng_state": self.rng.bit_generator.state,
            "training_step": self.training_step,
            "training_episodes": self.training_episodes,
        }
        arrays = {
            f"network_{name}": value
            for name, value in self.network.params.items()}
        arrays.update(self.optimizer.arrays("adam"))
        save_checkpoint(path, metadata, arrays)

    @classmethod
    def load(cls, path, *, expected_policy_id=None,
             expected_reward_profile_sha256=None):
        metadata, arrays = load_checkpoint(
            path, expected_algorithm=COST_GRAPH_PPO_ALGORITHM,
            expected_state_dim=COST_GUIDED_GRAPH_FEATURE_DIM,
            expected_action_dim=1)
        try:
            required = {
                "model_version", "action_semantics", "feature_version",
                "feature_schema_sha256", "config", "prior_identity",
                "provenance", "seed", "rng_state", "training_step",
                "training_episodes",
            }
            if not required.issubset(metadata):
                raise ValueError("checkpoint metadata is incomplete")
            if (metadata.get("model_version") !=
                    COST_GRAPH_PPO_MODEL_VERSION or
                    metadata.get("action_semantics") !=
                    COST_GRAPH_PPO_ACTION_SEMANTICS or
                    metadata.get("feature_version") !=
                    COST_GUIDED_GRAPH_FEATURE_VERSION or
                    metadata.get("feature_schema_sha256") !=
                    COST_GUIDED_GRAPH_SCHEMA_SHA256):
                raise ValueError("model or feature identity differs")
            config = CostGraphPPOConfig(**metadata["config"])
            prior = CostPriorIdentity.from_metadata(
                metadata["prior_identity"])
            provenance = CostGraphPPOProvenance.from_metadata(
                metadata["provenance"])
            if (expected_policy_id is not None and
                    prior.policy_id != expected_policy_id):
                raise ValueError("checkpoint policy identity differs")
            if (expected_reward_profile_sha256 is not None and
                    provenance.reward_profile_sha256 !=
                    expected_reward_profile_sha256):
                raise ValueError("checkpoint reward profile differs")
            seed = _positive_integer("seed", metadata["seed"], minimum=0)
            model = cls(prior, provenance, config, seed)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelValidationError(
                f"invalid CostGraphPPO metadata: {exc}") from exc
        for name, target in model.network.params.items():
            value = np.asarray(
                arrays.get(f"network_{name}"), dtype=np.float32)
            if value.shape != target.shape or not np.all(np.isfinite(value)):
                raise ModelValidationError(
                    f"invalid CostGraphPPO parameter {name}")
            target[...] = value
        try:
            model.optimizer.restore(arrays, "adam")
            training_step = _positive_integer(
                "training_step", metadata["training_step"], minimum=0)
            training_episodes = _positive_integer(
                "training_episodes",
                metadata["training_episodes"], minimum=0)
            if model.optimizer.step_count != 0:
                raise ValueError("optimizer restore changed step unexpectedly")
            model.training_step = training_step
            model.training_episodes = training_episodes
            model.optimizer.step_count = training_step
            model.rng.bit_generator.state = metadata["rng_state"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelValidationError(
                f"invalid CostGraphPPO training state: {exc}") from exc
        return model
