from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from auction_moment_research.research_world_model import acceptance, count_law, sample

ROOT = Path(__file__).resolve().parents[1]


class ResearchWorldModelTests(unittest.TestCase):
    def test_recipe_reserved_slot_matches_small_multinomial(self):
        # At N=10, reserving one colored leaves exactly one random extra.
        component = {
            "quality_probabilities": [0.1, 0.2, 0.3, 0.15, 0.25],
            "base_n_probabilities": [1 / 51] * 51,
            "reserved_colored": 1,
        }
        law = count_law(component, "R2")
        self.assertAlmostEqual(law[0, 1, 0], 0.6 / 51)
        self.assertAlmostEqual(law[0, 1, 1], 0.15 / 51)
        self.assertAlmostEqual(law[0, 2, 0], 0.25 / 51)
        self.assertEqual(law[:, 0, :].sum(), 0)
        self.assertAlmostEqual(law.sum(), 1)

    def test_composition_stop_matches_two_draw_enumeration(self):
        component = {
            "quality_probabilities": [0.1, 0.2, 0.3, 0.15, 0.25],
            "base_n_probabilities": [1 / 51] * 51,
            "reserved_colored": 0,
            "stop_rates": [0.2, 0.3, 0.1],
        }
        law = count_law(component, "KG_mix2")
        expected = np.zeros((53, 53))
        # Enumerate the first two independent draws into other/gold/colored.
        draws = [(0, 0, 0.6), (0, 1, 0.15), (1, 0, 0.25)]
        for k1, g1, p1 in draws:
            for k2, g2, p2 in draws:
                k, g = k1 + k2, g1 + g2
                stop = 1 - (1 - 1 / 51) * math.exp(-0.2 * k - 0.3 * g - 0.1 * k * g)
                expected[k, g] += p1 * p2 * stop
        np.testing.assert_allclose(law[0], expected, rtol=1e-13, atol=1e-15)
        self.assertAlmostEqual(law.sum(), 1)
        n, k, g = np.indices(law.shape)
        self.assertEqual(law[k + g > n + 2].sum(), 0)

    def test_portable_candidates_generate_valid_nontruncated_worlds(self):
        for family in ["R2", "KG_mix2"]:
            with self.subTest(family=family):
                model = json.loads(
                    (
                        ROOT
                        / f"models/research-world-model-stage-20261027/{family}.json"
                    ).read_text(encoding="utf-8")
                )
                worlds = sample(model, 2048, 20261027)
                self.assertTrue(np.all((worlds["n"] >= 10) & (worlds["n"] <= 60)))
                np.testing.assert_array_equal(worlds["counts"].sum(1), worlds["n"])
                self.assertTrue(np.all(worlds["counts"] >= np.array([1, 1, 3, 3, 0])))
                self.assertTrue(np.all(worlds["total"] > 0))
                self.assertTrue(np.all(worlds["area"] >= worlds["n"]))
                self.assertGreater(int((worlds["total"] > 7000000).sum()), 0)
                weights = acceptance(
                    model, np.full(4, 35), np.array([0, 6000000, 7000000, 12000000])
                )
                self.assertTrue(np.all(np.diff(weights) <= 0))
                self.assertTrue(np.all((weights > 0) & (weights <= 1)))


if __name__ == "__main__":
    unittest.main()
