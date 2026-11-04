"""Tests for deterministic, resumable separable CMA-ES."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/factory_supervisor"))

from coefficient_optimizer import SeparableCMAES, default_population_size


class SeparableCMAESTests(unittest.TestCase):
    def test_repeated_ask_and_json_resume_are_bit_exact(self):
        state = SeparableCMAES.create(4, seed=23117)
        first = state.ask(8)
        np.testing.assert_array_equal(first, state.ask(8))
        resumed = SeparableCMAES.from_dict(json.loads(json.dumps(
            state.to_dict(), sort_keys=True)))
        np.testing.assert_array_equal(first, resumed.ask(8))
        scores = -np.sum(first ** 2, axis=1)
        advanced = state.tell(first, scores)
        advanced_resumed = resumed.tell(resumed.ask(8), scores)
        self.assertEqual(advanced.to_dict(), advanced_resumed.to_dict())
        np.testing.assert_array_equal(advanced.ask(8), advanced_resumed.ask(8))

    def test_maximization_converges_on_bounded_sphere(self):
        state = SeparableCMAES.create(
            4, seed=81, initial_mean=np.full(4, 1.8), initial_sigma=0.7)
        initial_distance = float(np.linalg.norm(state.mean))
        best = float("-inf")
        for _ in range(50):
            candidates = state.ask(10)
            scores = -np.sum(candidates ** 2, axis=1)
            best = max(best, float(np.max(scores)))
            state = state.tell(candidates, scores)
        self.assertLess(np.linalg.norm(state.mean), initial_distance * 0.1)
        self.assertGreater(best, -0.02)

    def test_invalid_state_observations_and_schema_fail_closed(self):
        self.assertEqual(8, default_population_size(4))
        with self.assertRaises(ValueError):
            SeparableCMAES.create(0, seed=1)
        state = SeparableCMAES.create(2, seed=1)
        with self.assertRaises(ValueError):
            state.mean[0] = 1.0
        for invalid_seed in (True, -1, 1.5):
            with self.subTest(seed=invalid_seed), self.assertRaises(ValueError):
                SeparableCMAES.create(2, seed=invalid_seed)
        candidates = state.ask(4)
        with self.assertRaises(ValueError):
            state.tell(candidates, [1.0])
        with self.assertRaises(ValueError):
            state.tell(candidates, [1.0, 2.0, float("nan"), 4.0])
        bad = state.to_dict()
        bad["extra"] = True
        with self.assertRaises(ValueError):
            SeparableCMAES.from_dict(bad)

    def test_custom_bounds_are_enforced_and_survive_resume(self):
        bound = float(np.log(4.0))
        state = SeparableCMAES.create(
            3, seed=42, initial_sigma=10.0,
            lower_bound=-bound, upper_bound=bound)
        candidates = state.ask(20)
        self.assertTrue(np.all(candidates >= -bound))
        self.assertTrue(np.all(candidates <= bound))
        resumed = SeparableCMAES.from_dict(state.to_dict())
        self.assertEqual(-bound, resumed.lower_bound)
        self.assertEqual(bound, resumed.upper_bound)
        np.testing.assert_array_equal(candidates, resumed.ask(20))


if __name__ == "__main__":
    unittest.main()
