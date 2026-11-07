"""Variable-edge dueling C51 GraphRainbow model with NoisyNet heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Mapping

import numpy as np

from advanced_ai_common import (
    load_checkpoint, masked_argmax, save_checkpoint,
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


GRAPH_RAINBOW_ALGORITHM = "GraphRainbow"
GRAPH_RAINBOW_MODEL_VERSION = "graph-rainbow-v1-variable-dueling-c51-noisy"
GRAPH_RAINBOW_ACTION_SEMANTICS = "variable_feasible_edge_plus_masked_noop"
PRIMARY_NORMALIZED_INDEX = COST_GUIDED_GRAPH_FEATURE_NAMES.index(
    "primary_cost_normalized")
TRAINING_PHASES = ("standalone", "webots_fine_tuned")


def _integer(name, value, minimum=1) -> int:
    if (isinstance(value, bool) or
            not isinstance(value, (int, np.integer)) or
            int(value) < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(name, value, *, positive=False, nonnegative=False) -> float:
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
    result = _number(name, value)
    if result > 1.0 or result < (1e-15 if positive else 0.0):
        raise ValueError(f"{name} must be within the unit interval")
    return result


def _sha256(value) -> bool:
    return bool(
        isinstance(value, str) and len(value) == 64 and
        all(character in "0123456789abcdef" for character in value))


def _seeds(name, values, *, allow_empty=False) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} seeds must be a sequence")
    result = tuple(_integer(f"{name} seed", value, 0) for value in values)
    if (not allow_empty and not result) or len(result) != len(set(result)):
        raise ValueError(f"{name} seeds are empty or duplicated")
    return result


@dataclass(frozen=True)
class GraphRainbowConfig:
    hidden_size: int = 64
    atom_count: int = 51
    v_min: float = -200.0
    v_max: float = 200.0
    gamma: float = 0.99
    learning_rate: float = 0.0003
    max_grad_norm: float = 10.0
    noisy_sigma: float = 0.5
    prior_q_scale: float = 1.0
    n_step: int = 3
    replay_capacity: int = 20000
    batch_size: int = 32
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 100000
    priority_epsilon: float = 1e-5
    target_update_interval: int = 250

    def __post_init__(self):
        _integer("hidden_size", self.hidden_size)
        atoms = _integer("atom_count", self.atom_count, 3)
        if atoms % 2 == 0:
            raise ValueError("GraphRainbow atom_count must be odd")
        lower = _number("v_min", self.v_min)
        upper = _number("v_max", self.v_max)
        if lower >= upper:
            raise ValueError("GraphRainbow support bounds are invalid")
        _probability("gamma", self.gamma)
        _number("learning_rate", self.learning_rate, positive=True)
        _number("max_grad_norm", self.max_grad_norm, positive=True)
        _number("noisy_sigma", self.noisy_sigma, positive=True)
        _number("prior_q_scale", self.prior_q_scale, positive=True)
        _integer("n_step", self.n_step)
        _integer("replay_capacity", self.replay_capacity)
        batch = _integer("batch_size", self.batch_size)
        if batch > self.replay_capacity:
            raise ValueError("GraphRainbow batch exceeds replay capacity")
        _probability("per_alpha", self.per_alpha)
        _probability("per_beta_start", self.per_beta_start, positive=True)
        _integer("per_beta_steps", self.per_beta_steps)
        _number(
            "priority_epsilon", self.priority_epsilon, positive=True)
        _integer("target_update_interval", self.target_update_interval)

    @property
    def support(self) -> np.ndarray:
        return np.linspace(
            self.v_min, self.v_max, self.atom_count, dtype=np.float32)


@dataclass(frozen=True)
class GraphRainbowProvenance:
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
                not _sha256(self.reward_profile_sha256) or
                self.training_phase not in TRAINING_PHASES):
            raise ValueError("GraphRainbow provenance identity is invalid")
        standalone = _seeds("standalone", self.standalone_seeds)
        fine_tune = _seeds(
            "webots fine-tune", self.webots_fine_tune_seeds,
            allow_empty=True)
        if set(standalone).intersection(fine_tune):
            raise ValueError("GraphRainbow seed partitions overlap")
        if self.training_phase == "standalone" and fine_tune:
            raise ValueError("standalone checkpoint contains fine-tune seeds")
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
            raise ValueError("GraphRainbow provenance fields differ")
        return cls(**dict(value))


def _transform(features) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if (values.ndim != 2 or
            values.shape[1] != COST_GUIDED_GRAPH_FEATURE_DIM or
            not np.all(np.isfinite(values))):
        raise ValueError("GraphRainbow action features are invalid")
    return np.sign(values) * np.log1p(np.abs(values))


def _messages(observation: CostGuidedGraphObservation) -> np.ndarray:
    x = _transform(observation.action_features)
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
                x[index], np.mean(row_values, axis=0),
                np.mean(column_values, axis=0), global_message))
        output[observation.no_op_index] = np.concatenate((
            x[observation.no_op_index],
            np.zeros_like(global_message),
            np.zeros_like(global_message), global_message))
    else:
        output[0] = np.concatenate((
            x[0], np.zeros_like(global_message),
            np.zeros_like(global_message), global_message))
    if not np.all(np.isfinite(output)):
        raise ValueError("GraphRainbow messages are non-finite")
    return output


def _factorized_noise(rng, input_size, output_size):
    def transformed(size):
        values = rng.normal(size=size).astype(np.float32)
        return np.sign(values) * np.sqrt(np.abs(values))
    left = transformed(input_size)
    right = transformed(output_size)
    return np.outer(left, right), right


def _softmax_atoms(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(shifted)
    result = weights / np.sum(weights, axis=1, keepdims=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("GraphRainbow atom softmax is non-finite")
    return result


class GraphRainbowNetwork:
    def __init__(self, config: GraphRainbowConfig,
                 rng: np.random.Generator):
        self.config = config
        input_dim = 4 * COST_GUIDED_GRAPH_FEATURE_DIM
        hidden = config.hidden_size
        atoms = config.atom_count
        encoder_scale = math.sqrt(2.0 / input_dim)
        bound = 1.0 / math.sqrt(hidden)
        sigma = config.noisy_sigma / math.sqrt(hidden)
        self.params = {
            "w1": rng.normal(
                0.0, encoder_scale, (input_dim, hidden)).astype(np.float32),
            "b1": np.zeros(hidden, dtype=np.float32),
            "wa_mu": np.zeros((hidden, atoms), dtype=np.float32),
            "wa_sigma": np.full((hidden, atoms), sigma, dtype=np.float32),
            "ba_mu": np.zeros(atoms, dtype=np.float32),
            "ba_sigma": np.full(atoms, sigma, dtype=np.float32),
            "wv_mu": np.zeros((hidden, atoms), dtype=np.float32),
            "wv_sigma": np.full((hidden, atoms), sigma, dtype=np.float32),
            "bv_mu": np.zeros(atoms, dtype=np.float32),
            "bv_sigma": np.full(atoms, sigma, dtype=np.float32),
        }
        # Keep deterministic initial distributions uniform; the cost prior
        # therefore owns initial greedy behavior. The bound is retained as an
        # explicit finite initialization invariant for future versions.
        if not math.isfinite(bound):
            raise ValueError("GraphRainbow NoisyNet bound is invalid")

    def copy_from(self, other) -> None:
        if not isinstance(other, GraphRainbowNetwork):
            raise ValueError("GraphRainbow target source is invalid")
        for name, target in self.params.items():
            source = other.params.get(name)
            if source is None or source.shape != target.shape:
                raise ValueError("GraphRainbow network shapes differ")
            target[...] = source

    def forward(self, observation: CostGuidedGraphObservation, *,
                rng: np.random.Generator | None = None,
                noisy=False, cache=False):
        if not isinstance(observation, CostGuidedGraphObservation):
            raise ValueError("GraphRainbow requires its graph observation")
        if noisy and not isinstance(rng, np.random.Generator):
            raise ValueError("GraphRainbow noisy forward requires an RNG")
        augmented = _messages(observation)
        hidden = np.tanh(
            augmented @ self.params["w1"] + self.params["b1"])
        legal_indices = np.flatnonzero(observation.action_mask)
        pooled = np.mean(hidden[legal_indices], axis=0)
        if noisy:
            a_weight_noise, a_bias_noise = _factorized_noise(
                rng, self.config.hidden_size, self.config.atom_count)
            v_weight_noise, v_bias_noise = _factorized_noise(
                rng, self.config.hidden_size, self.config.atom_count)
        else:
            a_weight_noise = np.zeros_like(self.params["wa_mu"])
            a_bias_noise = np.zeros_like(self.params["ba_mu"])
            v_weight_noise = np.zeros_like(self.params["wv_mu"])
            v_bias_noise = np.zeros_like(self.params["bv_mu"])
        wa = (
            self.params["wa_mu"]
            + self.params["wa_sigma"] * a_weight_noise)
        ba = (
            self.params["ba_mu"]
            + self.params["ba_sigma"] * a_bias_noise)
        wv = (
            self.params["wv_mu"]
            + self.params["wv_sigma"] * v_weight_noise)
        bv = (
            self.params["bv_mu"]
            + self.params["bv_sigma"] * v_bias_noise)
        advantage = hidden @ wa + ba
        value = pooled @ wv + bv
        legal_mean = np.mean(advantage[legal_indices], axis=0)
        logits = value[None, :] + advantage - legal_mean[None, :]
        distributions = _softmax_atoms(logits)
        if not cache:
            return distributions, logits
        return distributions, logits, (
            augmented, hidden, pooled, legal_indices,
            a_weight_noise, a_bias_noise,
            v_weight_noise, v_bias_noise, wa, wv)

    def backward(self, cache, grad_logits) -> dict:
        (augmented, hidden, pooled, legal_indices,
         a_weight_noise, a_bias_noise,
         v_weight_noise, v_bias_noise, wa, wv) = cache
        grad_logits = np.asarray(grad_logits, dtype=np.float32)
        expected = (len(hidden), self.config.atom_count)
        if (grad_logits.shape != expected or
                not np.all(np.isfinite(grad_logits)) or
                len(legal_indices) == 0):
            raise ValueError("GraphRainbow distribution gradients are invalid")
        grad_value = np.sum(grad_logits, axis=0)
        grad_advantage = grad_logits.copy()
        grad_advantage[legal_indices] -= (
            np.sum(grad_logits, axis=0)[None, :] / len(legal_indices))
        grad_wa = hidden.T @ grad_advantage
        grad_ba = np.sum(grad_advantage, axis=0)
        grad_wv = np.outer(pooled, grad_value)
        grad_bv = grad_value
        hidden_gradient = grad_advantage @ wa.T
        hidden_gradient[legal_indices] += (
            (grad_value @ wv.T)[None, :] / len(legal_indices))
        z_gradient = hidden_gradient * (1.0 - hidden ** 2)
        grads = {
            "w1": augmented.T @ z_gradient,
            "b1": np.sum(z_gradient, axis=0),
            "wa_mu": grad_wa,
            "wa_sigma": grad_wa * a_weight_noise,
            "ba_mu": grad_ba,
            "ba_sigma": grad_ba * a_bias_noise,
            "wv_mu": grad_wv,
            "wv_sigma": grad_wv * v_weight_noise,
            "bv_mu": grad_bv,
            "bv_sigma": grad_bv * v_bias_noise,
        }
        if any(value.shape != self.params[name].shape or
               not np.all(np.isfinite(value))
               for name, value in grads.items()):
            raise ValueError("GraphRainbow parameter gradients are invalid")
        return grads


class GraphRainbowModel:
    def __init__(
            self, prior_identity: CostPriorIdentity,
            provenance: GraphRainbowProvenance,
            config: GraphRainbowConfig | None = None, seed: int = 42):
        if (not isinstance(prior_identity, CostPriorIdentity) or
                prior_identity.algorithm != GRAPH_RAINBOW_ALGORITHM):
            raise ValueError("GraphRainbow prior identity is incompatible")
        if (not isinstance(provenance, GraphRainbowProvenance) or
                provenance.objective != prior_identity.objective):
            raise ValueError("GraphRainbow provenance is incompatible")
        self.config = config or GraphRainbowConfig()
        self.prior_identity = prior_identity
        self.provenance = provenance
        self.seed = _integer("seed", seed, 0)
        self.rng = np.random.default_rng(self.seed)
        self.online = GraphRainbowNetwork(self.config, self.rng)
        self.target = GraphRainbowNetwork(self.config, self.rng)
        self.target.copy_from(self.online)
        self.optimizer = Adam(
            self.online.params, self.config.learning_rate,
            self.config.max_grad_norm)
        self.training_step = 0
        self.training_episodes = 0

    def validate_observation(self, observation) -> None:
        if (not isinstance(observation, CostGuidedGraphObservation) or
                observation.feature_schema_sha256 !=
                COST_GUIDED_GRAPH_SCHEMA_SHA256 or
                observation.prior_identity_sha256 !=
                self.prior_identity.sha256):
            raise ValueError("GraphRainbow observation identity differs")

    def distributions(self, observation, *, target=False, noisy=False,
                      cache=False):
        self.validate_observation(observation)
        network = self.target if target else self.online
        return network.forward(
            observation, rng=self.rng if noisy else None,
            noisy=noisy, cache=cache)

    def q_values(self, observation, *, target=False, noisy=False):
        distributions, _ = self.distributions(
            observation, target=target, noisy=noisy)
        expected = distributions @ self.config.support
        prior = observation.action_features[:, PRIMARY_NORMALIZED_INDEX]
        result = expected - self.config.prior_q_scale * prior
        if not np.all(np.isfinite(result)):
            raise ValueError("GraphRainbow Q values are non-finite")
        return result

    def select_action(self, observation, *, deterministic: bool) -> int:
        values = self.q_values(
            observation, target=False, noisy=not deterministic)
        return masked_argmax(values, observation.action_mask)

    def sync_target(self) -> None:
        self.target.copy_from(self.online)

    def mark_webots_fine_tuned(self, seeds) -> None:
        if self.provenance.training_phase != "standalone":
            raise ValueError("GraphRainbow checkpoint is already fine-tuned")
        self.provenance = replace(
            self.provenance, training_phase="webots_fine_tuned",
            webots_fine_tune_seeds=tuple(seeds))

    def save(self, path) -> None:
        metadata = {
            "algorithm": GRAPH_RAINBOW_ALGORITHM,
            "environment_version": RL_ENVIRONMENT_VERSION,
            "state_dim": COST_GUIDED_GRAPH_FEATURE_DIM,
            "action_dim": 1,
            "model_version": GRAPH_RAINBOW_MODEL_VERSION,
            "action_semantics": GRAPH_RAINBOW_ACTION_SEMANTICS,
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
            f"online_{name}": value
            for name, value in self.online.params.items()}
        arrays.update({
            f"target_{name}": value
            for name, value in self.target.params.items()})
        arrays.update(self.optimizer.arrays("adam"))
        save_checkpoint(path, metadata, arrays)

    @classmethod
    def load(cls, path, *, expected_policy_id=None,
             expected_reward_profile_sha256=None):
        metadata, arrays = load_checkpoint(
            path, expected_algorithm=GRAPH_RAINBOW_ALGORITHM,
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
            if (metadata["model_version"] != GRAPH_RAINBOW_MODEL_VERSION or
                    metadata["action_semantics"] !=
                    GRAPH_RAINBOW_ACTION_SEMANTICS or
                    metadata["feature_version"] !=
                    COST_GUIDED_GRAPH_FEATURE_VERSION or
                    metadata["feature_schema_sha256"] !=
                    COST_GUIDED_GRAPH_SCHEMA_SHA256):
                raise ValueError("model or feature identity differs")
            config = GraphRainbowConfig(**metadata["config"])
            prior = CostPriorIdentity.from_metadata(
                metadata["prior_identity"])
            provenance = GraphRainbowProvenance.from_metadata(
                metadata["provenance"])
            if (expected_policy_id is not None and
                    prior.policy_id != expected_policy_id):
                raise ValueError("checkpoint policy identity differs")
            if (expected_reward_profile_sha256 is not None and
                    provenance.reward_profile_sha256 !=
                    expected_reward_profile_sha256):
                raise ValueError("checkpoint reward profile differs")
            model = cls(
                prior, provenance, config,
                _integer("seed", metadata["seed"], 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelValidationError(
                f"invalid GraphRainbow metadata: {exc}") from exc
        for prefix, network in (
                ("online", model.online), ("target", model.target)):
            for name, target in network.params.items():
                value = np.asarray(
                    arrays.get(f"{prefix}_{name}"), dtype=np.float32)
                if (value.shape != target.shape or
                        not np.all(np.isfinite(value))):
                    raise ModelValidationError(
                        f"invalid GraphRainbow parameter {prefix}_{name}")
                target[...] = value
        try:
            model.optimizer.restore(arrays, "adam")
            training_step = _integer(
                "training_step", metadata["training_step"], 0)
            training_episodes = _integer(
                "training_episodes", metadata["training_episodes"], 0)
            model.training_step = training_step
            model.training_episodes = training_episodes
            model.optimizer.step_count = training_step
            model.rng.bit_generator.state = metadata["rng_state"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelValidationError(
                f"invalid GraphRainbow training state: {exc}") from exc
        return model
