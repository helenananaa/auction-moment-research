from __future__ import annotations

import unittest

from auction_moment_assistant.predictor import PredictionResult
from auction_moment_assistant.presentation import format_prediction


class DefaultPredictionPresentationTests(unittest.TestCase):
    def test_default_v6_is_not_presented_as_fusion_or_failure(self):
        result = PredictionResult(
            revision=1,
            status="ocr_adapted_v6_default_estimate",
            compatible_worlds=40,
            p10=100,
            p50=200,
            p90=400,
            minimum=None,
            maximum=None,
            actionable=False,
            issues=(),
            estimate_source="v6_default",
            v6_prediction=200,
            v2_prediction=300,
            generation_weight=0.0,
            diagnostics=("conditioned_resample:100:40",),
        )
        rendered = format_prediction(result)
        self.assertIn("v6 P10 / P50 / P90：100 / 200 / 400", rendered)
        self.assertIn("v2 诊断估值（未参与融合）：300", rendered)
        self.assertIn("v2 诊断：", rendered)
        self.assertNotIn("融合 P10", rendered)
        self.assertNotIn("冲突", rendered)
        self.assertNotIn("回退", rendered)


if __name__ == "__main__":
    unittest.main()
