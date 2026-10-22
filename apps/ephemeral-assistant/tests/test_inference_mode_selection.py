from __future__ import annotations

import inspect
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from auction_moment_assistant.cli import build_parser
from auction_moment_assistant.latest_model import (
    LatestV6V2Predictor,
    AdaptedDecision,
    V6_CANDIDATE_ID,
    _generation_weight,
    _released_generation_weight,
)
from auction_moment_assistant.predictor import PredictionResult
from auction_moment_assistant.models import load_catalog
from auction_moment_assistant.observations import ObservationSnapshot


PUBLIC_ROOT = Path(__file__).resolve().parents[3]
FROZEN_V6 = PUBLIC_ROOT / "models/latest-model-v6-v2.0.0-beta.1/latest-value-model-v6.joblib"


class InferenceModeSelectionTests(unittest.TestCase):
    def world(self, **updates):
        return replace(
            PredictionResult(
                revision=1,
                status="provisional_low_support",
                compatible_worlds=3,
                p10=80,
                p50=100,
                p90=140,
                minimum=80,
                maximum=140,
                actionable=False,
                issues=(),
            ),
            **updates,
        )

    def test_unqualified_conditional_sampler_is_explicit_opt_in(self):
        parameter = inspect.signature(LatestV6V2Predictor).parameters[
            "conditional_sampling"
        ]
        self.assertIs(parameter.default, False)
        self.assertFalse(
            build_parser().parse_args([]).experimental_conditional_sampling
        )
        self.assertTrue(
            build_parser()
            .parse_args(["--experimental-conditional-sampling"])
            .experimental_conditional_sampling
        )

    def test_v2_fusion_is_a_separate_explicit_opt_in(self):
        parameter = inspect.signature(LatestV6V2Predictor).parameters[
            "experimental_v2_fusion"
        ]
        self.assertIs(parameter.default, False)
        defaults = build_parser().parse_args([])
        self.assertFalse(defaults.experimental_v2_fusion)
        self.assertFalse(defaults.experimental_conditional_sampling)
        released = build_parser().parse_args(["--experimental-v2-fusion"])
        self.assertTrue(released.experimental_v2_fusion)
        self.assertFalse(released.experimental_conditional_sampling)
        conditional = build_parser().parse_args(["--experimental-conditional-sampling"])
        self.assertTrue(conditional.experimental_conditional_sampling)
        self.assertFalse(conditional.experimental_v2_fusion)

    def predictor(self, world=None, **mode):
        v6 = SimpleNamespace(
            candidate_id=V6_CANDIDATE_ID,
            predict=lambda _: {"prediction": 1000, "p10": 800, "p90": 1400},
        )
        result = world or self.world(p10=1600, p50=2000, p90=2800)
        world_predictor = SimpleNamespace(
            world_model_version="world_model_v2", predict=lambda _: result,
        )
        with patch.dict("sys.modules", {"joblib": SimpleNamespace(load=lambda _: v6)}):
            with patch(
                "auction_moment_assistant.latest_model.EmpiricalWorldPredictor",
                return_value=world_predictor,
            ):
                predictor = LatestV6V2Predictor(
                    v6_model_path=Path("unused-v6.joblib"),
                    world_model_v2_path=Path("unused-v2.json"),
                    treasures_csv=Path("unused-treasures.csv"),
                    catalog=(), **mode,
                )
        predictor.adapter = SimpleNamespace(
            adapt=lambda _: AdaptedDecision({}, (), (), 5),
        )
        return predictor

    def test_default_preserves_v6_quantiles_and_manual_advice_with_v2_diagnostics(self):
        result = self.predictor().predict(SimpleNamespace(revision=1, round_number=1))
        self.assertEqual((result.p10, result.p50, result.p90), (800, 1000, 1400))
        self.assertEqual(result.generation_weight, 0)
        self.assertEqual(result.estimate_source, "v6_default")
        self.assertEqual(result.status, "ocr_adapted_v6_default_estimate")
        self.assertEqual(result.recommended_bid, 720)
        self.assertEqual(result.v2_prediction, 2000)
        self.assertEqual(
            result.conditional_diagnostics["fusion_policy"],
            "v6_only_pending_fusion_qualification",
        )
        self.assertTrue(any("默认使用 v6" in issue for issue in result.issues))
        self.assertFalse(any("fallback" in issue or "failed" in issue for issue in result.issues))

    def test_default_v2_failure_does_not_mislabel_successful_v6_as_fallback(self):
        predictor = self.predictor()

        def fail(_):
            raise RuntimeError("diagnostic world unavailable")

        predictor.world_predictor.predict = fail
        result = predictor.predict(SimpleNamespace(revision=1, round_number=1))
        self.assertEqual(result.p50, 1000)
        self.assertEqual(result.estimate_source, "v6_default")
        self.assertEqual(result.status, "ocr_adapted_v6_default_estimate")
        self.assertEqual(
            result.conditional_diagnostics["v2_diagnostic_result"]["status"],
            "sampling_failed",
        )
        self.assertFalse(any("failed" in issue for issue in result.issues))

    def test_missing_restored_mode_attributes_do_not_activate_fusion(self):
        predictor = self.predictor()
        del predictor.conditional_sampling
        del predictor.experimental_v2_fusion
        result = predictor.predict(SimpleNamespace(revision=1, round_number=1))
        self.assertEqual(result.p50, 1000)
        self.assertEqual(result.generation_weight, 0)
        self.assertEqual(result.estimate_source, "v6_default")
        self.assertFalse(result.conditional_diagnostics["experimental_conditional_sampling"])
        self.assertFalse(result.conditional_diagnostics["experimental_v2_fusion"])

    def test_opt_in_released_fusion_preserves_original_heuristic(self):
        result = self.predictor(experimental_v2_fusion=True).predict(
            SimpleNamespace(revision=1, round_number=1)
        )
        self.assertEqual(result.generation_weight, .8)
        self.assertEqual(result.p50, 1741)
        self.assertEqual(result.estimate_source, "v6_v2_blend")
        self.assertEqual(
            result.conditional_diagnostics["fusion_policy"],
            "released_event_count_heuristic_uncalibrated",
        )

    def test_conditional_opt_in_automatically_allows_only_strict_gate(self):
        snapshot = SimpleNamespace(revision=1, round_number=1)
        rejected = self.predictor(conditional_sampling=True).predict(snapshot)
        self.assertEqual(rejected.generation_weight, 0)
        self.assertEqual(rejected.estimate_source, "v6_fallback")
        supported = self.world(
            p10=1600, p50=2000, p90=2800, posterior_converged=True,
            effective_sample_size=512.0, compatible_worlds=512,
        )
        for released_flag in (False, True):
            result = self.predictor(
                world=supported, conditional_sampling=True,
                experimental_v2_fusion=released_flag,
            ).predict(snapshot)
            self.assertEqual(result.generation_weight, .4)
            self.assertEqual(result.estimate_source, "v6_v2_blend")
            self.assertEqual(
                result.conditional_diagnostics["fusion_policy"],
                "converged_effective_support_cap_v1",
            )

    @unittest.skipUnless(FROZEN_V6.is_file(), "optional released v6 artifact unavailable")
    def test_default_matches_real_frozen_v6_after_reveal_adapter_fix(self):
        catalog_path = PUBLIC_ROOT / "data/v1/core/treasures.csv"
        catalog = load_catalog(catalog_path)
        rainbow = next(item for item in catalog if item.quality == "彩" and item.size == "1x1")
        snapshot = ObservationSnapshot(
            revision=1, round_number=1, bankroll=None,
            events=(
                {"round_number": 1, "kind": "public", "confidence": .95,
                 "semantics": {"parsed": True, "effect": "reveal_identity_random", "count": 1}},
                {"round_number": 1, "kind": "personal", "confidence": .95,
                 "semantics": {"parsed": True, "effect": "total_item_count", "observed_count": 30}},
            ),
            map_items=({"row": 0, "column": 0, "width": 1, "height": 1,
                        "quality": "彩", "catalog_id": rainbow.catalog_id, "confidence": .95,
                        "spatial": "complete", "first_seen_round": 1},),
            completeness_confirmed=True, completeness_source="controlled_test",
            map_rows=12, map_height_exact=True, map_rows_source="controlled_test",
            pre_bid_confirmed=True, pre_bid_source="controlled_test",
        )
        diagnostic_world = SimpleNamespace(
            world_model_version="world_model_v2", predict=lambda _: self.world(),
        )
        with patch(
            "auction_moment_assistant.latest_model.EmpiricalWorldPredictor",
            return_value=diagnostic_world,
        ):
            predictor = LatestV6V2Predictor(
                v6_model_path=FROZEN_V6,
                world_model_v2_path=Path("unused-v2.json"),
                treasures_csv=catalog_path, catalog=catalog,
            )
        adapted = predictor.adapter.adapt(snapshot)
        self.assertTrue(adapted.accepted, adapted.issues)
        expected = predictor.v6_model.predict(adapted.decision)
        self.assertEqual(expected["random_identity_sample"]["sample_count"], 1)
        result = predictor.predict(snapshot)
        self.assertEqual((result.p10, result.p50, result.p90),
                         (expected["p10"], expected["prediction"], expected["p90"]))
        self.assertEqual(result.generation_weight, 0)
        self.assertEqual(result.estimate_source, "v6_default")

    def test_released_policy_is_preserved_without_fabricating_convergence(self):
        world = self.world()
        weight, reasons = _released_generation_weight(world, 5)
        self.assertEqual((weight, reasons), (0.8, ()))
        self.assertFalse(world.posterior_converged)
        self.assertIsNone(world.effective_sample_size)
        self.assertEqual(_generation_weight(world)[0], 0)

    def test_released_policy_keeps_empty_or_invalid_estimates_out(self):
        for changes in (
            {"p50": None},
            {"p90": float("nan")},
            {"p10": 110},
            {"compatible_worlds": 0},
        ):
            self.assertEqual(
                _released_generation_weight(self.world(**changes), 8)[0], 0
            )


if __name__ == "__main__":
    unittest.main()
