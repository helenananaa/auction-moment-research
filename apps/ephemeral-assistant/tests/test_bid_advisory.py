from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from auction_moment_assistant.latest_model import (
    AdaptedDecision,
    LatestV6V2Predictor,
    _geometric_blend,
    manual_bid_advice,
)
from auction_moment_assistant.observations import ObservationStore
from auction_moment_assistant.predictor import PredictionResult
from auction_moment_assistant.presentation import (
    describe_diagnostic,
    format_prediction,
)


class BidAdvisoryTests(unittest.TestCase):
    def predictor_with_world(self, world: PredictionResult) -> LatestV6V2Predictor:
        predictor = object.__new__(LatestV6V2Predictor)
        predictor.conditional_sampling = True
        predictor.experimental_v2_fusion = False
        predictor.adapter = SimpleNamespace(
            adapt=lambda _snapshot: AdaptedDecision({}, (), (), 0)
        )
        predictor.v6_model = SimpleNamespace(
            predict=lambda _decision: {
                "prediction": 5_600_000,
                "p10": 3_500_000,
                "p90": 7_000_000,
            }
        )
        predictor.world_predictor = SimpleNamespace(
            predict=lambda _snapshot: world
        )
        predictor.bid_safety_factor = 0.90
        return predictor

    def test_manual_bid_uses_p10_safety_and_round_multiplier(self) -> None:
        advice = manual_bid_advice(3_500_000, 2, safety_factor=0.90)
        self.assertEqual(advice["recommended_bid"], 3_150_000)
        self.assertEqual(advice["required_winning_multiplier"], 1.6)
        self.assertEqual(advice["maximum_opponent_bid"], 1_968_750)

    def test_v2_conflict_keeps_v6_estimate_and_manual_bid(self) -> None:
        world = PredictionResult(
            revision=1,
            status="constraint_conflict",
            compatible_worlds=0,
            p10=None,
            p50=None,
            p90=None,
            minimum=None,
            maximum=None,
            actionable=False,
            issues=("no_compatible_public_world",),
            diagnostics=("event_conflict:R2:public:total_item_count:999",),
            conditional_diagnostics={"numeric_infeasible": True},
        )
        store = ObservationStore()
        store.set_round(2)
        result = self.predictor_with_world(world).predict(store.snapshot())

        self.assertEqual(result.status, "v6_fallback_v2_conditioning_conflict")
        self.assertEqual(
            (result.p10, result.p50, result.p90),
            (3_500_000, 5_600_000, 7_000_000),
        )
        self.assertEqual(result.estimate_source, "v6_fallback")
        self.assertEqual(result.recommended_bid, 3_150_000)
        self.assertEqual(result.maximum_opponent_bid, 1_968_750)
        self.assertFalse(result.actionable)
        rendered = format_prediction(result)
        self.assertIn("建议最高出价（仅人工参考）：3,150,000", rendered)
        self.assertIn("v2 条件冲突，显示 v6 保守回退", rendered)
        self.assertIn("R2 公共事件（本局藏品总数=999）", rendered)

    def test_normal_blend_also_emits_manual_bid(self) -> None:
        world = PredictionResult(
            revision=1,
            status="ready_research_estimate",
            compatible_worlds=200,
            p10=4_100_000,
            p50=5_300_000,
            p90=7_800_000,
            minimum=2_000_000,
            maximum=10_000_000,
            actionable=False,
            issues=(),
            effective_sample_size=256.0,
            posterior_converged=True,
            posterior_method="conditional_mcmc",
        )
        store = ObservationStore()
        store.set_round(1)
        result = self.predictor_with_world(world).predict(store.snapshot())

        expected_p10 = _geometric_blend(3_500_000, 4_100_000, 0.30)
        self.assertEqual(result.p10, expected_p10)
        self.assertEqual(result.recommended_bid, math.floor(expected_p10 * 0.90))
        self.assertEqual(result.estimate_source, "v6_v2_blend")
        self.assertEqual(result.required_winning_multiplier, 2.0)
        self.assertIn("手动出价建议", format_prediction(result))

    def test_conflict_diagnostic_is_chinese(self) -> None:
        self.assertEqual(
            describe_diagnostic("map_conflict:quality_size:金:2x2:3"),
            "要求金品质且 2x2 尺寸至少 3 件后兼容世界归零",
        )


if __name__ == "__main__":
    unittest.main()
