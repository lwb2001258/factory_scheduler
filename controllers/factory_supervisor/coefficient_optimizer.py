"""Deterministic, serializable separable CMA-ES for expert coefficients."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


OPTIMIZER_VERSION = "separable-cma-es-v1"


def default_population_size(dimensions: int) -> int:
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("optimizer dimensions must be a positive integer")
    return 4 + int(math.floor(3.0 * math.log(dimensions)))


@dataclass(frozen=True)
class SeparableCMAES:
    """Diagonal CMA-ES state with generation-derived random streams.

    A generation can be regenerated after interruption from only this state.
    Fitness is maximized. Candidate vectors are bounded log-multipliers, so
    the caller can exponentiate them into positive expert coefficients.
    """

    seed: int
    generation: int
    mean: np.ndarray
    sigma: float
    covariance: np.ndarray
    evolution_path_sigma: np.ndarray
    evolution_path_covariance: np.ndarray
    lower_bound: float = -math.log(10.0)
    upper_bound: float = math.log(10.0)

    def __post_init__(self):
        dimensions = np.asarray(self.mean).reshape(-1).size
        arrays = (
            np.asarray(self.mean, dtype=np.float64).reshape(-1),
            np.asarray(self.covariance, dtype=np.float64).reshape(-1),
            np.asarray(self.evolution_path_sigma, dtype=np.float64).reshape(-1),
            np.asarray(self.evolution_path_covariance, dtype=np.float64).reshape(-1),
        )
        if (isinstance(self.seed, bool) or not isinstance(self.seed, int) or
                self.seed < 0 or
                isinstance(self.generation, bool) or
                not isinstance(self.generation, int) or self.generation < 0):
            raise ValueError("optimizer seed/generation is invalid")
        if dimensions < 1 or any(value.shape != (dimensions,) for value in arrays):
            raise ValueError("optimizer state vectors are misaligned")
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("optimizer state must be finite")
        if np.any(arrays[1] <= 0.0):
            raise ValueError("optimizer covariance must be positive")
        if (not math.isfinite(float(self.sigma)) or float(self.sigma) <= 0.0 or
                not math.isfinite(float(self.lower_bound)) or
                not math.isfinite(float(self.upper_bound)) or
                float(self.lower_bound) >= float(self.upper_bound)):
            raise ValueError("optimizer scale or bounds are invalid")
        if np.any(arrays[0] < self.lower_bound) or np.any(arrays[0] > self.upper_bound):
            raise ValueError("optimizer mean is outside its bounds")
        immutable = []
        for value in arrays:
            copied = value.copy()
            copied.setflags(write=False)
            immutable.append(copied)
        object.__setattr__(self, "mean", immutable[0])
        object.__setattr__(self, "covariance", immutable[1])
        object.__setattr__(self, "evolution_path_sigma", immutable[2])
        object.__setattr__(self, "evolution_path_covariance", immutable[3])
        object.__setattr__(self, "sigma", float(self.sigma))
        object.__setattr__(self, "lower_bound", float(self.lower_bound))
        object.__setattr__(self, "upper_bound", float(self.upper_bound))

    @property
    def dimensions(self) -> int:
        return int(self.mean.size)

    @classmethod
    def create(cls, dimensions: int, *, seed: int,
               initial_mean=None, initial_sigma: float = 0.55):
        default_population_size(dimensions)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("optimizer seed must be a non-negative integer")
        mean = (np.zeros(dimensions, dtype=np.float64)
                if initial_mean is None else
                np.asarray(initial_mean, dtype=np.float64).reshape(-1))
        if mean.shape != (dimensions,):
            raise ValueError("initial optimizer mean has an invalid shape")
        return cls(
            seed=seed, generation=0, mean=mean,
            sigma=float(initial_sigma), covariance=np.ones(dimensions),
            evolution_path_sigma=np.zeros(dimensions),
            evolution_path_covariance=np.zeros(dimensions))

    def ask(self, population_size: int | None = None) -> np.ndarray:
        size = (default_population_size(self.dimensions)
                if population_size is None else population_size)
        if isinstance(size, bool) or not isinstance(size, int) or size < 2:
            raise ValueError("optimizer population must contain at least two candidates")
        # A generation-specific stream makes repeated ask() and resume exact.
        random = np.random.default_rng(np.random.SeedSequence(
            [self.seed, self.generation, self.dimensions, size]))
        standard = random.standard_normal((size, self.dimensions))
        candidates = (
            self.mean + self.sigma * np.sqrt(self.covariance) * standard)
        return np.clip(candidates, self.lower_bound, self.upper_bound)

    def tell(self, candidates, fitness) -> "SeparableCMAES":
        population = np.asarray(candidates, dtype=np.float64)
        scores = np.asarray(fitness, dtype=np.float64).reshape(-1)
        if (population.ndim != 2 or population.shape[1] != self.dimensions or
                population.shape[0] < 2 or scores.shape != (population.shape[0],)):
            raise ValueError("optimizer candidates and fitness are misaligned")
        if (not np.all(np.isfinite(population)) or
                not np.all(np.isfinite(scores)) or
                np.any(population < self.lower_bound) or
                np.any(population > self.upper_bound)):
            raise ValueError("optimizer observations must be finite and bounded")

        count = population.shape[0]
        parents = max(1, count // 2)
        raw_weights = np.log(parents + 0.5) - np.log(np.arange(1, parents + 1))
        weights = raw_weights / np.sum(raw_weights)
        effective = 1.0 / float(np.sum(weights ** 2))
        dimensions = self.dimensions
        c_sigma = (effective + 2.0) / (dimensions + effective + 5.0)
        damping = (
            1.0 + 2.0 * max(
                0.0, math.sqrt((effective - 1.0) / (dimensions + 1.0)) - 1.0
            ) + c_sigma)
        c_covariance = (
            (4.0 + effective / dimensions) /
            (dimensions + 4.0 + 2.0 * effective / dimensions))
        c_one = 2.0 / ((dimensions + 1.3) ** 2 + effective)
        c_mu = min(
            1.0 - c_one,
            2.0 * (effective - 2.0 + 1.0 / effective) /
            ((dimensions + 2.0) ** 2 + effective))

        order = np.argsort(-scores, kind="stable")
        selected = population[order[:parents]]
        previous_mean = self.mean
        mean = np.sum(weights[:, None] * selected, axis=0)
        mean = np.clip(mean, self.lower_bound, self.upper_bound)
        normalized = (selected - previous_mean) / self.sigma
        weighted_step = (mean - previous_mean) / self.sigma
        inverse_root_covariance = 1.0 / np.sqrt(self.covariance)
        path_sigma = (
            (1.0 - c_sigma) * self.evolution_path_sigma +
            math.sqrt(c_sigma * (2.0 - c_sigma) * effective) *
            inverse_root_covariance * weighted_step)
        expected_norm = (
            math.sqrt(dimensions) *
            (1.0 - 1.0 / (4.0 * dimensions) +
             1.0 / (21.0 * dimensions ** 2)))
        path_normalizer = math.sqrt(
            max(1e-15, 1.0 - (1.0 - c_sigma) **
                (2.0 * (self.generation + 1))))
        path_test = (
            np.linalg.norm(path_sigma) / path_normalizer /
            expected_norm < 1.4 + 2.0 / (dimensions + 1.0))
        path_covariance = (
            (1.0 - c_covariance) * self.evolution_path_covariance +
            (math.sqrt(c_covariance * (2.0 - c_covariance) * effective) *
             weighted_step if path_test else 0.0))
        rank_mu = np.sum(weights[:, None] * normalized ** 2, axis=0)
        covariance = (
            (1.0 - c_one - c_mu) * self.covariance +
            c_one * (
                path_covariance ** 2 +
                (0.0 if path_test else
                 c_covariance * (2.0 - c_covariance) * self.covariance)) +
            c_mu * rank_mu)
        covariance = np.maximum(covariance, 1e-12)
        sigma = self.sigma * math.exp(
            (c_sigma / damping) *
            (float(np.linalg.norm(path_sigma)) / expected_norm - 1.0))
        sigma = min(2.0, max(1e-3, sigma))
        return SeparableCMAES(
            seed=self.seed, generation=self.generation + 1,
            mean=mean, sigma=sigma, covariance=covariance,
            evolution_path_sigma=path_sigma,
            evolution_path_covariance=path_covariance,
            lower_bound=self.lower_bound, upper_bound=self.upper_bound)

    def to_dict(self) -> dict:
        return {
            "version": OPTIMIZER_VERSION,
            "seed": self.seed,
            "generation": self.generation,
            "mean": self.mean.tolist(),
            "sigma": self.sigma,
            "covariance": self.covariance.tolist(),
            "evolution_path_sigma": self.evolution_path_sigma.tolist(),
            "evolution_path_covariance":
                self.evolution_path_covariance.tolist(),
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

    @classmethod
    def from_dict(cls, value: Mapping):
        expected = {
            "version", "seed", "generation", "mean", "sigma",
            "covariance", "evolution_path_sigma",
            "evolution_path_covariance", "lower_bound", "upper_bound",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("optimizer checkpoint schema is incompatible")
        if value.get("version") != OPTIMIZER_VERSION:
            raise ValueError("optimizer checkpoint version is incompatible")
        return cls(
            seed=value["seed"], generation=value["generation"],
            mean=value["mean"], sigma=value["sigma"],
            covariance=value["covariance"],
            evolution_path_sigma=value["evolution_path_sigma"],
            evolution_path_covariance=value["evolution_path_covariance"],
            lower_bound=value["lower_bound"],
            upper_bound=value["upper_bound"])
