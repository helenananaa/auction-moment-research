from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from pathlib import Path

from auction_moment_assistant.latest_model import OcrDecisionAdapter
from auction_moment_assistant.models import CatalogItem, load_catalog
from auction_moment_assistant.observations import ObservationSnapshot
from protocol_analog_value_model import _random_identity_quality_sample
from protocol_repeated_quality_sampling import extract_event_quality_samples


CATALOG = (CatalogItem("rainbow", "rainbow", "彩", 1, 1, 500_000),)
PUBLIC_ROOT = Path(__file__).resolve().parents[3]
FROZEN_MODEL = PUBLIC_ROOT / "models/latest-model-v6-v2.0.0-beta.1/latest-value-model-v6.joblib"


def event(effect="reveal_identity_random", *, kind="public", round_number=1, **semantics):
    return {
        "round_number": round_number, "kind": kind,
        "confidence": 0.95, "human_locked": False,
        "semantics": {"parsed": True, "effect": effect, **semantics},
    }


def snapshot(*, effect="reveal_identity_random", human_locked=False):
    return ObservationSnapshot(
        revision=1, round_number=1, bankroll=None,
        events=(
            event(effect, count=1),
            event("total_item_count", kind="personal", observed_count=30),
        ),
        map_items=({
            "row": 0, "column": 0, "width": 1, "height": 1,
            "quality": "彩", "catalog_id": "rainbow", "confidence": 0.95,
            "human_locked": human_locked, "spatial": "complete", "first_seen_round": 1,
        },),
        completeness_confirmed=True, completeness_source="human_confirmed",
        map_rows=12, map_height_exact=True, map_rows_source="human_confirmed",
        pre_bid_confirmed=True, pre_bid_source="human_confirmed",
    )


class AdapterRevealFeaturesTests(unittest.TestCase):
    def adapt(self, evidence=None):
        return OcrDecisionAdapter(CATALOG).adapt(evidence or snapshot())

    def test_ocr_identity_sample_preserves_source_and_activates_feature(self):
        adapted = self.adapt()
        self.assertTrue(adapted.accepted, adapted.issues)
        decision = adapted.decision
        item = decision["visible_items"][0]
        self.assertEqual(item["quality_validation"], "ocr_thresholded")
        self.assertEqual(item["identity"]["match_confidence_status"], "ocr_thresholded")
        self.assertNotIn("item_index", item)
        self.assertEqual(item["history"][0]["event_kind"], "public")
        self.assertEqual(_random_identity_quality_sample(decision)["quality_counts"], {"彩": 1})
        self.assertEqual(len(extract_event_quality_samples(decision)["accepted"]), 1)

    def test_personal_position_quality_source_is_preserved(self):
        evidence = snapshot(effect="reveal_position_quality_random")
        evidence = replace(evidence, events=(
            event("total_item_count", observed_count=30),
            event("reveal_position_quality_random", kind="personal", count=1),
        ))
        decision = self.adapt(evidence).decision
        self.assertEqual(decision["visible_items"][0]["history"][0]["event_kind"], "personal")
        samples = extract_event_quality_samples(decision)
        self.assertEqual(samples["accepted"][0].counts(), {"彩": 1})

    def test_unattributed_or_incomplete_evidence_does_not_gain_history(self):
        base = snapshot()
        cases = [
            replace(base, completeness_confirmed=False),
            replace(base, map_items=()),
            replace(base, map_items=({**base.map_items[0], "first_seen_round": None},)),
            replace(base, map_items=({**base.map_items[0], "confidence": 0.5},)),
            replace(base, events=(
                event(count=1),
                event("reveal_position_quality_random", kind="personal", count=1),
            )),
            snapshot(effect="reveal_position_random"),
        ]
        for evidence in cases:
            with self.subTest(evidence=evidence):
                decision = self.adapt(evidence).decision
                self.assertFalse(decision["reveal_attribution"]["certified_events"])
                self.assertTrue(all(not item["history"] for item in decision["visible_items"]))
                self.assertFalse(extract_event_quality_samples(decision)["accepted"])

    def test_old_position_later_attribute_is_not_random_sample_history(self):
        base = snapshot()
        evidence = replace(base, round_number=2, events=(
            event("reveal_position_random", count=1),
            event("total_item_count", kind="personal", observed_count=30),
            event(round_number=2, count=1),
            event("total_item_count", round_number=2, kind="personal", observed_count=30),
        ))
        decision = self.adapt(evidence).decision
        self.assertFalse(decision["reveal_attribution"]["certified_events"])
        self.assertFalse(decision["visible_items"][0]["history"])

    def test_later_random_identity_does_not_invalidate_first_certified_sample(self):
        base = snapshot()
        evidence = replace(base, round_number=2, events=(
            *base.events,
            event(round_number=2, count=1),
            event("total_item_count", round_number=2, kind="personal", observed_count=30),
        ), map_items=(
            *base.map_items,
            {**base.map_items[0], "column": 1, "first_seen_round": 2},
        ))
        decision = self.adapt(evidence).decision
        sample = _random_identity_quality_sample(decision)
        self.assertEqual(sample["sample_count"], 1)
        self.assertEqual(sample["expected_count"], 1)
        self.assertEqual(sample["event_rounds"], [1])
        self.assertFalse(decision["visible_items"][1]["history"])

    def test_thresholded_ocr_requires_complete_adapter_contract(self):
        base = self.adapt().decision
        for mutation in (
            "certificate", "adapter", "native", "confidence", "history",
            "nan_quality", "nan_evidence", "nan_event",
        ):
            decision = copy.deepcopy(base)
            if mutation == "certificate":
                decision.pop("reveal_attribution")
            elif mutation == "adapter":
                decision.pop("adapter_schema_version")
            elif mutation == "native":
                decision["native_evidence"]["ocr_used_for_live_features"] = False
            elif mutation == "confidence":
                decision["visible_items"][0]["quality_confidence"] = 0.5
            elif mutation == "nan_quality":
                decision["visible_items"][0]["quality_confidence"] = float("nan")
            elif mutation == "nan_evidence":
                decision["visible_items"][0]["evidence_confidence"] = float("nan")
            elif mutation == "nan_event":
                decision["events"][0]["ocr_confidence"] = float("nan")
            else:
                decision["visible_items"][0]["history"][0].pop("attribution_schema")
            with self.subTest(mutation=mutation):
                self.assertEqual(_random_identity_quality_sample(decision)["sample_count"], 0)
                self.assertFalse(extract_event_quality_samples(decision)["accepted"])

    def test_adapter_rejects_nonfinite_unlocked_confidence(self):
        base = snapshot()
        for value in (float("nan"), float("inf"), float("-inf")):
            for field in ("events", "map_items"):
                records = list(getattr(base, field))
                records[0] = {**records[0], "confidence": value}
                adapted = self.adapt(replace(base, **{field: tuple(records)}))
                with self.subTest(value=value, field=field):
                    self.assertFalse(adapted.accepted)
                    self.assertFalse(adapted.decision["reveal_attribution"]["certified_events"])

    def test_legacy_exact_packet_sample_still_works_without_ocr_contract(self):
        decision = self.adapt(snapshot(human_locked=True)).decision
        decision.pop("adapter_schema_version")
        decision.pop("reveal_attribution")
        item = decision["visible_items"][0]
        item["quality_validation"] = "protocol_exact"
        item["identity"]["match_confidence_status"] = "protocol_exact"
        item["history"] = [{"round": 1, "source_effect": "reveal_identity_random"}]
        self.assertEqual(_random_identity_quality_sample(decision)["quality_counts"], {"彩": 1})
        self.assertEqual(len(extract_event_quality_samples(decision)["accepted"]), 1)

    @unittest.skipUnless(FROZEN_MODEL.is_file(), "optional released v6 artifact unavailable")
    def test_frozen_v6_responds_to_attributed_rainbow_sample(self):
        import joblib

        catalog = load_catalog(PUBLIC_ROOT / "data/v1/core/treasures.csv")
        rainbow = next(item for item in catalog if item.quality == "彩" and item.size == "1x1")
        model = joblib.load(FROZEN_MODEL)
        for effect in ("reveal_identity_random", "reveal_position_quality_random"):
            for human_locked in (False, True):
                evidence = snapshot(effect=effect, human_locked=human_locked)
                evidence = replace(evidence, map_items=({
                    **evidence.map_items[0], "catalog_id": rainbow.catalog_id,
                },))
                decision = OcrDecisionAdapter(catalog).adapt(evidence).decision
                previous = copy.deepcopy(decision)
                previous["visible_items"][0]["history"] = []
                previous.pop("reveal_attribution")
                before, after = model.predict(previous), model.predict(decision)
                with self.subTest(effect=effect, human_locked=human_locked):
                    self.assertGreater(after["prediction"], before["prediction"])
                    self.assertEqual(model.candidate_id, after["model_version"])
                    if effect == "reveal_identity_random":
                        self.assertEqual(after["random_identity_sample"]["sample_count"], 1)
                    else:
                        self.assertEqual(after["position_quality_sample"]["used_event_count"], 1)


if __name__ == "__main__":
    unittest.main()
