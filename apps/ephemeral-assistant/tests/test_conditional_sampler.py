from __future__ import annotations

import math
import random
import time
import unittest
from collections import Counter, OrderedDict
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from auction_moment_assistant.conditional_constraints import compile_hard_constraints
from auction_moment_assistant.conditional_sampler import (
    ConditionalWorldSampler,
    ConditionalSamples,
    chain_diagnostics,
    enumerate_block,
    ordered_log_probability,
)
from auction_moment_assistant.models import CatalogItem
from auction_moment_assistant.observations import ObservationSnapshot
from auction_moment_assistant.predictor import EmpiricalWorldPredictor
from auction_moment_assistant.world_layout import layout_compatible


CATALOG = (
    CatalogItem("w", "white", "白", 1, 1, 10),
    CatalogItem("b", "blue", "蓝", 1, 1, 100),
)


class TinyPrior:
    """N=2 or 3 uniformly; iid identities with P(blue)=1/2."""

    catalog = CATALOG
    min_count = 2
    max_count = 3
    _count_probabilities = {2: 0.5, 3: 0.5}
    minimum_quotas = {}

    def sample_prior_counts(self, rng):
        n = rng.choice((2, 3))
        blue = sum(rng.random() < 0.5 for _ in range(n))
        return n - blue, blue

    def log_probability(self, counts, *, rows, rows_exact):
        if any(c < 0 for c in counts) or sum(counts) not in (2, 3):
            return SimpleNamespace(total=-math.inf)
        n = sum(counts)
        return SimpleNamespace(total=math.log(0.5 * math.comb(n, counts[1]) / 2**n))


def snapshot(**changes):
    return replace(
        ObservationSnapshot(
            revision=1,
            round_number=1,
            bankroll=None,
            events=(),
            map_items=(),
            completeness_confirmed=True,
            completeness_source="controlled",
            map_rows=None,
            map_height_exact=False,
            map_rows_source="missing",
            pre_bid_confirmed=True,
            pre_bid_source="controlled",
        ),
        **changes,
    )


def no_filter(snapshot, counts):
    return np.ones(len(counts), dtype=bool)


class ConditionalSamplerTests(unittest.TestCase):
    def test_order_target_removes_multiset_multiplicity(self):
        prior = TinyPrior()
        self.assertAlmostEqual(ordered_log_probability(prior, (1, 1)), math.log(0.125))
        self.assertAlmostEqual(ordered_log_probability(prior, (2, 0)), math.log(0.125))

    def test_iid_small_world_matches_exact_position_selection_posterior(self):
        # Enumerating all ordered worlds and uniformly selected single items
        # gives P(N=2 | selected anchor 0 is blue)=3/5, E(value)=177.
        events = (
            {
                "round_number": 1,
                "kind": "public",
                "confidence": 1,
                "semantics": {
                    "effect": "reveal_position_quality_random",
                    "count": 1,
                    "parsed": True,
                },
            },
            {
                "round_number": 1,
                "kind": "personal",
                "confidence": 1,
                "semantics": {
                    "effect": "highest_quality",
                    "observed_quality": "蓝",
                    "parsed": True,
                },
            },
        )
        clue = {
            "row": 0,
            "column": 0,
            "quality": "蓝",
            "spatial": "top_left",
            "first_seen_round": 1,
            "confidence": 1,
        }
        observed = snapshot(events=events, map_items=(clue,))
        prior = TinyPrior()
        pool_rng = random.Random(91)
        pool = np.asarray([prior.sample_prior_counts(pool_rng) for _ in range(1024)])
        result = ConditionalWorldSampler(prior, CATALOG, budget_seconds=4).sample(
            observed,
            rng=random.Random(42),
            filter_population=no_filter,
            prior_population=pool,
        )
        self.assertTrue(result.converged, result.diagnostics)
        self.assertEqual(result.diagnostics["method"], "independent_prior_rejection")
        self.assertEqual(result.diagnostics["iid_draws"], 1024)
        self.assertGreater(len(result.counts), 100)
        self.assertAlmostEqual(
            float(np.mean(result.counts.sum(axis=1) == 2)), 0.6, delta=0.09
        )
        # N=2: 100+E(other)=155; N=3:210; mixture .6/.4 =>177.
        self.assertAlmostEqual(float(np.mean(result.counts @ [10, 100])), 177, delta=18)

    def test_exact_height_changes_compatible_distribution(self):
        catalog = (
            CatalogItem("a", "a", "白", 3, 1, 10),
            CatalogItem("b", "b", "白", 2, 1, 20),
            CatalogItem("c", "c", "白", 3, 2, 30),
        )
        low = snapshot(map_rows=2, map_height_exact=True)
        high = snapshot(map_rows=3, map_height_exact=True)
        self.assertTrue(layout_compatible((0, 0, 2, 1), catalog, low))
        self.assertFalse(layout_compatible((0, 0, 2, 1), catalog, high))
        self.assertNotEqual(
            EmpiricalWorldPredictor._conditioning_signature(low),
            EmpiricalWorldPredictor._conditioning_signature(high),
        )

    def test_cache_key_includes_reveal_history_and_trust(self):
        first = snapshot(map_items=({"row": 0, "column": 0, "first_seen_round": 1},))
        second = replace(
            first, map_items=({"row": 0, "column": 0, "first_seen_round": 2},)
        )
        self.assertNotEqual(
            EmpiricalWorldPredictor._conditioning_signature(first),
            EmpiricalWorldPredictor._conditioning_signature(second),
        )

    def test_block_enumeration_proves_complete_at_cap_and_never_uses_partial(self):
        base, variables, notes = compile_hard_constraints(
            snapshot(), CATALOG, TinyPrior()
        )
        self.assertFalse(notes)
        # N2 supports 3 compositions, N3 supports4 =7 exactly.
        full = enumerate_block(
            base, variables, (1, 1), (0, 1), cap=7, deadline=time.monotonic() + 2
        )
        self.assertEqual(len(full), 7)
        self.assertEqual(Counter(sum(c) for c in full), {2: 3, 3: 4})
        partial = enumerate_block(
            base, variables, (1, 1), (0, 1), cap=6, deadline=time.monotonic() + 2
        )
        self.assertIsNone(partial)
        self.assertIsNone(
            enumerate_block(
                base, variables, (1, 1), (0, 1), deadline=time.monotonic() - 1
            )
        )

    def test_chain_diagnostics_reject_stuck_different_modes(self):
        rhat, ess = chain_diagnostics(np.asarray([[1] * 64, [2] * 64]))
        self.assertTrue(math.isinf(rhat))
        self.assertEqual(ess, 0)
        independent = np.random.default_rng(3).normal(size=(4, 256))
        rhat, ess = chain_diagnostics(independent)
        self.assertLess(rhat, 1.05)
        self.assertGreater(ess, 600)
        slow = np.cumsum(independent, axis=1)
        self.assertLess(chain_diagnostics(slow)[1], ess / 5)

    def test_gibbs_mh_matches_exact_small_world_prior(self):
        result = ConditionalWorldSampler(
            TinyPrior(), CATALOG, budget_seconds=0.7, independent_draws=0
        ).sample(snapshot(), rng=random.Random(12), filter_population=no_filter)
        self.assertEqual(result.diagnostics["method"], "ordered_block_gibbs_mh")
        self.assertGreater(len(result.counts), 100)
        self.assertAlmostEqual(
            float(np.mean(result.counts.sum(axis=1) == 2)), 0.5, delta=0.12
        )
        self.assertAlmostEqual(
            float(np.mean(result.counts @ [10, 100])), 137.5, delta=15
        )

    def test_unsupported_event_cannot_receive_iid_ready_status(self):
        obs = snapshot(
            events=(
                {
                    "round_number": 1,
                    "kind": "public",
                    "semantics": {"parsed": True, "effect": "unknown"},
                },
            )
        )
        result = ConditionalWorldSampler(TinyPrior(), CATALOG).sample(
            obs, rng=random.Random(1), filter_population=no_filter
        )
        self.assertFalse(result.converged)
        self.assertEqual(len(result.counts), 0)
        self.assertEqual(result.diagnostics["method"], "unsupported_evidence")

    def test_timeout_is_not_numeric_conflict_or_convergence(self):
        obs = snapshot(
            events=(
                {
                    "round_number": 1,
                    "kind": "public",
                    "semantics": {
                        "parsed": True,
                        "effect": "total_item_count",
                        "observed_count": 99,
                    },
                },
            )
        )
        result = ConditionalWorldSampler(
            TinyPrior(), CATALOG, budget_seconds=0.02
        ).sample(
            obs,
            rng=random.Random(1),
            filter_population=lambda s, c: np.zeros(len(c), bool),
        )
        self.assertFalse(result.converged)
        self.assertLess(result.effective_sample_size, 32)

    def test_nonconverged_evidence_can_retry_then_cache_success(self):
        predictor = object.__new__(EmpiricalWorldPredictor)
        predictor.catalog = CATALOG
        predictor.values = np.asarray([10, 100])
        predictor._world_model = TinyPrior()
        predictor._prior_counts = np.empty((0, 2))
        predictor.conditional_budget_seconds = 0.1
        predictor._posterior_cache = OrderedDict()
        predictor.empirical_prior_diagnostics = {}
        predictor._filter_population = lambda *args, **kwargs: None
        count_array = np.asarray([[1, 1]] * 64)
        diagnostic = {"notes": [], "method": "test"}
        with patch.object(
            ConditionalWorldSampler,
            "sample",
            side_effect=[
                ConditionalSamples(count_array, 2, False, diagnostic),
                ConditionalSamples(count_array, 64, True, diagnostic),
            ],
        ) as sampler:
            first = predictor._predict_conditional(snapshot())
            second = predictor._predict_conditional(snapshot())
            third = predictor._predict_conditional(snapshot(revision=99))
        self.assertFalse(first.posterior_converged)
        self.assertTrue(second.posterior_converged)
        self.assertEqual(third.revision, 99)
        self.assertEqual(sampler.call_count, 2)

    def test_unbound_largest_outline_is_explicitly_unsupported(self):
        obs = snapshot(
            events=(
                {
                    "round_number": 1,
                    "kind": "public",
                    "semantics": {
                        "parsed": True,
                        "effect": "reveal_largest_area_item_outline",
                    },
                },
            )
        )
        _, _, notes = compile_hard_constraints(obs, CATALOG, TinyPrior())
        self.assertTrue(
            any(
                note.startswith("unsupported:largest_outline_binding") for note in notes
            )
        )
        missing = replace(
            obs,
            events=(
                {
                    **obs.events[0],
                    "semantics": {
                        **obs.events[0]["semantics"],
                        "reveal_outcome_available": False,
                    },
                },
            ),
        )
        self.assertFalse(compile_hard_constraints(missing, CATALOG, TinyPrior())[2])

    def test_nonfinite_budget_is_rejected(self):
        for budget in (float("inf"), float("nan"), 0, -1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                ConditionalWorldSampler(TinyPrior(), CATALOG, budget_seconds=budget)


if __name__ == "__main__":
    unittest.main()
