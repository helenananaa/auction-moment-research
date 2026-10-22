from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from auction_moment_assistant.latest_model import (
    AdaptedDecision,
    LatestV6V2Predictor,
    OcrDecisionAdapter,
    V6_CANDIDATE_ID,
    _generation_weight,
)
from auction_moment_assistant.models import CatalogItem
from auction_moment_assistant.observations import (
    EventObservation,
    MapObservation,
    ObservationStore,
)
from auction_moment_assistant.predictor import PredictionResult


CATALOG = (
    CatalogItem("C001", "catalog_C001", "金", 2, 2, 123456),
)


def event(round_number: int, kind: str, confidence: float = 0.95):
    return EventObservation(
        round_number,
        kind,
        "显示本局藏品总数量24",
        confidence,
        {
            "parsed": True,
            "effect": "total_item_count",
            "observed_count": 24,
        },
    )


class OcrDecisionAdapterTests(unittest.TestCase):
    def reviewed_store(self) -> ObservationStore:
        store = ObservationStore()
        store.apply_capture(
            round_number=1,
            events=[event(1, "public"), event(1, "personal")],
            bankroll=999,
            map_items=[
                MapObservation(
                    row=0,
                    column=0,
                    width=2,
                    height=2,
                    quality="金",
                    catalog_id="C001",
                    confidence=0.96,
                )
            ],
        )
        store.confirm_complete(True)
        store.confirm_pre_bid(True)
        store.correct_map_extent(10, True)
        return store

    def test_translates_reviewed_ocr_without_inventing_packet_fields(self) -> None:
        adapted = OcrDecisionAdapter(CATALOG).adapt(
            self.reviewed_store().snapshot()
        )
        self.assertTrue(adapted.accepted, adapted.issues)
        decision = adapted.decision
        self.assertEqual(
            decision["native_evidence"]["mode"], "ocr_visible_pre_bid_v1"
        )
        self.assertNotIn("packet", decision)
        self.assertNotIn("settlement", decision)
        self.assertNotIn("item_index", decision["visible_items"][0])
        self.assertEqual(
            decision["events"][0]["semantics"]["protocol_source"],
            "ocr_thresholded",
        )
        self.assertFalse(decision["visible_items"][0]["identity_known"])
        self.assertEqual(decision["visible_items"][0]["identity"], {})

    def test_manual_identity_becomes_exact_human_evidence(self) -> None:
        store = self.reviewed_store()
        store.upsert_map_item(
            MapObservation(
                row=0,
                column=0,
                width=2,
                height=2,
                quality="金",
                catalog_id="C001",
                spatial="complete",
            )
        )
        adapted = OcrDecisionAdapter(CATALOG).adapt(store.snapshot())
        item = adapted.decision["visible_items"][0]
        self.assertTrue(item["identity_known"])
        self.assertEqual(item["identity"]["known_value"], 123456)
        self.assertEqual(
            item["identity"]["match_confidence_status"], "human_confirmed"
        )

    def test_preserves_all_five_evidence_states_without_inventing_size(self) -> None:
        store = ObservationStore()
        store.apply_capture(
            round_number=1,
            events=[event(1, "public"), event(1, "personal")],
            bankroll=999,
            map_items=[
                MapObservation(
                    row=0,
                    column=0,
                    marker_width=3,
                    marker_height=2,
                    confidence=0.96,
                    spatial="top_left",
                ),
                MapObservation(
                    row=0,
                    column=1,
                    marker_width=2,
                    marker_height=2,
                    quality="金",
                    confidence=0.96,
                    spatial="top_left",
                ),
                MapObservation(
                    row=0,
                    column=2,
                    width=1,
                    height=1,
                    confidence=0.96,
                    spatial="outline",
                ),
                MapObservation(
                    row=0,
                    column=3,
                    width=1,
                    height=1,
                    quality="金",
                    confidence=0.96,
                    spatial="outline",
                ),
                MapObservation(
                    row=0,
                    column=4,
                    width=2,
                    height=2,
                    quality="金",
                    catalog_id="C001",
                    confidence=0.96,
                    spatial="complete",
                ),
            ],
        )
        store.confirm_complete(True)
        store.confirm_pre_bid(True)
        store.correct_map_extent(10, True)

        adapted = OcrDecisionAdapter(CATALOG).adapt(store.snapshot())
        self.assertTrue(adapted.accepted, adapted.issues)
        items = {
            value["position"]["column"]: value
            for value in adapted.decision["visible_items"]
        }
        self.assertIsNone(items[0]["known_size"])
        self.assertEqual(items[0]["spatial_knowledge"], "top_left")
        self.assertIsNone(items[0]["quality"])
        self.assertIsNone(items[1]["known_size"])
        self.assertEqual(items[1]["quality"], "金")
        self.assertEqual(items[2]["known_size"], {"width": 1, "height": 1})
        self.assertEqual(items[2]["spatial_knowledge"], "outline")
        self.assertIsNone(items[2]["quality"])
        self.assertEqual(items[3]["quality"], "金")
        self.assertTrue(items[4]["identity_known"])
        self.assertEqual(
            items[4]["identity"]["match_confidence_status"],
            "ocr_thresholded",
        )

    def test_blocks_missing_proofs_and_low_confidence_event(self) -> None:
        store = ObservationStore()
        store.apply_capture(
            round_number=1,
            events=[event(1, "public", 0.5), event(1, "personal")],
            bankroll=None,
            map_items=[],
        )
        adapted = OcrDecisionAdapter(CATALOG).adapt(store.snapshot())
        self.assertFalse(adapted.accepted)
        self.assertIn("pre_bid_not_confirmed", adapted.issues)
        self.assertIn("exact_map_rows_not_confirmed", adapted.issues)
        self.assertIn("visible_map_not_reviewed", adapted.issues)
        self.assertIn("low_confidence_event:R1:public", adapted.issues)

    def test_auto_scroll_estimate_runs_with_explicit_warnings(self) -> None:
        store = ObservationStore()
        store.apply_capture(
            round_number=1,
            events=[event(1, "public"), event(1, "personal")],
            bankroll=999,
            map_items=[],
        )
        store.apply_automatic_scan_proof(
            round_number=1, map_rows=12, complete=True, pre_bid=True
        )
        adapted = OcrDecisionAdapter(CATALOG).adapt(store.snapshot())
        self.assertTrue(adapted.accepted, adapted.issues)
        self.assertIn("map_rows_auto_estimated", adapted.warnings)
        self.assertEqual(
            adapted.decision["observed_map"]["source"],
            "ocr_visible_map_estimated",
        )
        self.assertEqual(
            adapted.decision["observed_map"]["height_proof"]["status"],
            "estimated",
        )


class LatestFusionGateTests(unittest.TestCase):
    def test_constructor_forwards_explicit_sampling_budget_and_mode(self) -> None:
        model = SimpleNamespace(candidate_id=V6_CANDIDATE_ID)
        world = SimpleNamespace(world_model_version="world_model_v2")
        with patch.dict("sys.modules", {"joblib": SimpleNamespace(load=lambda _: model)}):
            with patch(
                "auction_moment_assistant.latest_model.EmpiricalWorldPredictor",
                return_value=world,
            ) as factory:
                predictor = LatestV6V2Predictor(
                    v6_model_path=Path("unused-v6.joblib"),
                    world_model_v2_path=Path("unused-v2.json"),
                    treasures_csv=Path("unused-public.csv"),
                    catalog=CATALOG,
                    conditional_sampling=False,
                    conditional_budget_seconds=8.0,
                )
        self.assertIs(predictor.world_predictor, world)
        self.assertFalse(factory.call_args.kwargs["conditional_sampling"])
        self.assertEqual(factory.call_args.kwargs["conditional_budget_seconds"], 8.0)

    def world(self, **changes) -> PredictionResult:
        base = PredictionResult(
            revision=1,
            status="ready_research_estimate",
            compatible_worlds=512,
            p10=200,
            p50=300,
            p90=400,
            minimum=150,
            maximum=450,
            actionable=False,
            issues=(),
            effective_sample_size=256.0,
            posterior_converged=True,
            posterior_method="conditional_mcmc",
            conditional_diagnostics={"chains": 2, "rhat": 1.01},
        )
        return replace(base, **changes)

    def predict(self, world=None, *, strong_count=0, error=None):
        predictor = object.__new__(LatestV6V2Predictor)
        predictor.conditional_sampling = True
        predictor.experimental_v2_fusion = False
        predictor.adapter = SimpleNamespace(
            adapt=lambda _snapshot: AdaptedDecision({}, (), (), strong_count)
        )
        predictor.v6_model = SimpleNamespace(
            predict=lambda _decision: {"prediction": 100, "p10": 80, "p90": 120}
        )

        def predict_world(_snapshot):
            if error is not None:
                raise error
            return world

        predictor.world_predictor = SimpleNamespace(predict=predict_world)
        predictor.bid_safety_factor = 0.9
        return predictor.predict(ObservationStore().snapshot())

    def test_convergence_and_true_effective_support_required(self) -> None:
        for changes in (
            {"posterior_converged": False},
            {"effective_sample_size": None},
            {"effective_sample_size": 31.99},
            {"effective_sample_size": float("nan")},
            {"effective_sample_size": float("inf")},
            {"compatible_worlds": 31},
        ):
            with self.subTest(**changes):
                result = self.predict(self.world(**changes), strong_count=5)
                self.assertEqual(result.generation_weight, 0.0)
                self.assertEqual((result.p10, result.p50, result.p90), (80, 100, 120))
                self.assertEqual(result.estimate_source, "v6_fallback")
                self.assertFalse(result.actionable)

    def test_weight_is_capped_by_effective_and_retained_support(self) -> None:
        for ess, count, expected in (
            (32, 512, 0.1),
            (63, 512, 0.1),
            (64, 512, 0.2),
            (127, 512, 0.2),
            (128, 512, 0.3),
            (255, 512, 0.3),
            (256, 512, 0.4),
            (4096, 512, 0.4),
            (4096, 32, 0.1),
        ):
            with self.subTest(ess=ess, count=count):
                world = self.world(effective_sample_size=ess, compatible_worlds=count)
                self.assertEqual(_generation_weight(world), (expected, ()))

    def test_more_strong_events_do_not_raise_fusion_weight(self) -> None:
        world = self.world(effective_sample_size=32.0)
        first = self.predict(world, strong_count=0)
        second = self.predict(world, strong_count=5)
        self.assertEqual(first.generation_weight, 0.1)
        self.assertEqual(first.p50, second.p50)
        self.assertEqual(second.generation_weight, first.generation_weight)

    def test_invalid_quantiles_and_evidence_failure_block_fusion(self) -> None:
        for changes in (
            {"p10": None},
            {"p50": float("nan")},
            {"p90": 299},
            {"p10": 0},
            {"status": "constraint_conflict"},
            {"issues": ("unsupported_event_effect:example",)},
            {"conditional_diagnostics": {"sampling_failed": True}},
            {"conditional_diagnostics": {"evidence_conflict": True}},
            {"conditional_diagnostics": {"truncated": True}},
        ):
            with self.subTest(**changes):
                self.assertEqual(_generation_weight(self.world(**changes))[0], 0.0)

    def test_sampling_exception_falls_back_and_retains_reason(self) -> None:
        result = self.predict(error=RuntimeError("solver budget exhausted"))
        self.assertEqual(result.estimate_source, "v6_fallback")
        self.assertEqual(result.status, "v6_fallback_v2_insufficient_posterior")
        self.assertEqual(result.p50, 100)
        self.assertEqual(result.generation_weight, 0.0)
        self.assertIn("v2_sampling_failed:RuntimeError:solver budget exhausted", result.issues)
        self.assertNotIn("v2_conflict_using_v6_fallback", result.issues)
        self.assertFalse(result.actionable)

    def test_blend_preserves_posterior_diagnostics_and_uncalibrated_contract(self) -> None:
        result = self.predict(self.world())
        self.assertEqual(result.estimate_source, "v6_v2_blend")
        self.assertEqual(result.generation_weight, 0.4)
        self.assertEqual(result.effective_sample_size, 256.0)
        self.assertTrue(result.posterior_converged)
        self.assertEqual(result.posterior_method, "conditional_mcmc")
        self.assertEqual(result.conditional_diagnostics["chains"], 2)
        self.assertEqual(result.conditional_diagnostics["fusion_weight"], 0.4)
        self.assertEqual(result.conditional_diagnostics["fusion_gate_reasons"], [])
        self.assertIn("interval_uncalibrated", result.issues)
        self.assertFalse(result.actionable)


if __name__ == "__main__":
    unittest.main()
