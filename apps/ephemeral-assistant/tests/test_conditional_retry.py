from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import numpy as np

from auction_moment_assistant.conditional_sampler import ConditionalSamples
from auction_moment_assistant.models import CatalogItem
from auction_moment_assistant.observations import ObservationStore
from auction_moment_assistant.predictor import EmpiricalWorldPredictor


class ConditionalRetryTests(unittest.TestCase):
    def predictor(self):
        predictor = object.__new__(EmpiricalWorldPredictor)
        predictor.catalog = (
            CatalogItem("C001", "one", "白", 1, 1, 10),
            CatalogItem("C002", "two", "金", 1, 2, 100),
        )
        predictor.values = np.asarray([10, 100])
        predictor._world_model = object()
        predictor._prior_counts = np.asarray([[1, 1], [2, 1]])
        predictor.conditional_budget_seconds = 2.0
        predictor._posterior_cache = OrderedDict()
        predictor._conditional_retry_indices = OrderedDict()
        predictor.empirical_prior_diagnostics = {}
        predictor._filter_population = lambda *args, **kwargs: None
        return predictor

    def samples(self, *, converged=False):
        return ConditionalSamples(
            np.asarray([[1, 1], [2, 1]], dtype=np.int16),
            64.0 if converged else 0.0,
            converged,
            {"notes": [], "method": "mock_conditional"},
        )

    def test_first_call_keeps_frozen_seed_and_sampler_inputs(self):
        predictor = self.predictor()
        snapshot = ObservationStore().snapshot()
        seed = int(predictor._conditioning_signature(snapshot)[:16], 16)
        with patch(
            "auction_moment_assistant.conditional_sampler.ConditionalWorldSampler.sample",
            return_value=self.samples(),
        ) as sample:
            result = predictor._predict_conditional(snapshot)
        args, kwargs = sample.call_args
        self.assertEqual(args, (snapshot,))
        self.assertEqual(kwargs["rng"].getstate(), random.Random(seed).getstate())
        self.assertIs(kwargs["prior_population"], predictor._prior_counts)
        self.assertEqual(kwargs["filter_population"], predictor._filter_numeric_population)
        self.assertEqual(result.p50, 110)
        self.assertEqual(result.conditional_diagnostics["sampling_attempt"], 1)
        self.assertEqual(result.conditional_diagnostics["sampling_retry_index"], 0)
        self.assertEqual(result.conditional_diagnostics["sampling_rng_seed"], seed)

    def test_nonconverged_retry_changes_seed_without_changing_evidence(self):
        predictor = self.predictor()
        snapshot = ObservationStore().snapshot()
        with patch(
            "auction_moment_assistant.conditional_sampler.ConditionalWorldSampler.sample",
            side_effect=lambda *args, **kwargs: self.samples(),
        ) as sample:
            first = predictor._predict_conditional(snapshot)
            second = predictor._predict_conditional(snapshot)
        first_rng = sample.call_args_list[0].kwargs["rng"]
        second_rng = sample.call_args_list[1].kwargs["rng"]
        self.assertNotEqual(first_rng.getstate(), second_rng.getstate())
        self.assertEqual(second.conditional_diagnostics["sampling_attempt"], 2)
        self.assertEqual(second.conditional_diagnostics["sampling_retry_index"], 1)
        self.assertNotEqual(
            first.conditional_diagnostics["sampling_rng_seed"],
            second.conditional_diagnostics["sampling_rng_seed"],
        )
        self.assertEqual(sample.call_args_list[0].args, sample.call_args_list[1].args)

    def test_sampler_exception_also_advances_retry_seed(self):
        predictor = self.predictor()
        snapshot = ObservationStore().snapshot()
        with patch(
            "auction_moment_assistant.conditional_sampler.ConditionalWorldSampler.sample",
            side_effect=[RuntimeError("budget interrupted"), self.samples()],
        ) as sample:
            with self.assertRaisesRegex(RuntimeError, "budget interrupted"):
                predictor._predict_conditional(snapshot)
            retried = predictor._predict_conditional(snapshot)
        self.assertEqual(retried.conditional_diagnostics["sampling_attempt"], 2)
        self.assertNotEqual(
            sample.call_args_list[0].kwargs["rng"].getstate(),
            sample.call_args_list[1].kwargs["rng"].getstate(),
        )

    def test_converged_cache_does_not_resample_and_updates_revision(self):
        predictor = self.predictor()
        snapshot = ObservationStore().snapshot()
        with patch(
            "auction_moment_assistant.conditional_sampler.ConditionalWorldSampler.sample",
            return_value=self.samples(converged=True),
        ) as sample:
            first = predictor._predict_conditional(snapshot)
            cached = predictor._predict_conditional(replace(snapshot, revision=99))
        self.assertEqual(sample.call_count, 1)
        self.assertEqual(cached.revision, 99)
        self.assertEqual(cached.p50, first.p50)
        self.assertEqual(cached.conditional_diagnostics, first.conditional_diagnostics)
        self.assertEqual(len(predictor._conditional_retry_indices), 0)

    def test_retry_signature_ledger_is_bounded(self):
        predictor = self.predictor()
        snapshot = ObservationStore().snapshot()
        first_signature = predictor._conditioning_signature(snapshot)
        with patch(
            "auction_moment_assistant.conditional_sampler.ConditionalWorldSampler.sample",
            side_effect=lambda *args, **kwargs: self.samples(),
        ):
            predictor._predict_conditional(snapshot)
            for rows in range(1, 9):
                predictor._predict_conditional(replace(snapshot, map_rows=rows))
        self.assertEqual(len(predictor._conditional_retry_indices), 8)
        self.assertNotIn(first_signature, predictor._conditional_retry_indices)

    def test_counter_wrap_remains_bounded_and_skips_first_seed(self):
        predictor = self.predictor()
        snapshot = ObservationStore().snapshot()
        signature = predictor._conditioning_signature(snapshot)
        predictor._conditional_retry_indices[signature] = (1 << 64) - 1
        with patch(
            "auction_moment_assistant.conditional_sampler.ConditionalWorldSampler.sample",
            return_value=self.samples(),
        ):
            result = predictor._predict_conditional(snapshot)
        self.assertEqual(predictor._conditional_retry_indices[signature], 1)
        self.assertNotEqual(
            result.conditional_diagnostics["sampling_rng_seed"],
            int(signature[:16], 16),
        )


if __name__ == "__main__":
    unittest.main()
