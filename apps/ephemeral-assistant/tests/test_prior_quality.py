from __future__ import annotations

from collections import Counter, defaultdict
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

from auction_moment_assistant.models import CatalogItem, load_catalog
from auction_moment_assistant.prior_quality import load_empirical_counts


ROOT = Path(__file__).resolve().parents[3]
TREASURES = ROOT / "data/v1/core/treasures.csv"
RECEIPT = ROOT / "results/assistant-conditioning-development-20261019-stable-2s-sanitized.json"


class PublicPriorQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "public_world_quality_audit", ROOT / "scripts/audit_public_world_quality.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.audit = module.audit(ROOT / "data/v1", RECEIPT)
        cls.catalog = load_catalog(TREASURES)

    def test_runtime_matches_independent_audit_eligible_world_population(self):
        counts, quality = load_empirical_counts(TREASURES, self.catalog)
        self.assertEqual(quality["input_worlds"], 1794)
        self.assertEqual(quality["accepted_worlds"], 1284)
        self.assertEqual(quality["excluded_worlds"], 510)
        self.assertEqual(quality["verified_worlds"], 1284)
        self.assertEqual(
            quality["accepted_worlds"],
            self.audit["denominators"]["prior_eligible_worlds"],
        )
        eligible_ids = {
            row["session_id"] for row in self.audit["world_records"]
            if row["prior_eligible"]
        }
        index = {item.catalog_id: i for i, item in enumerate(self.catalog)}
        expected = defaultdict(lambda: [0] * len(self.catalog))
        with TREASURES.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["session_id"] in eligible_ids:
                    expected[row["session_id"]][index[row["catalog_id"]]] += 1
        self.assertEqual(
            Counter(tuple(map(int, row)) for row in counts),
            Counter(tuple(row) for row in expected.values()),
        )

    def test_conflicting_identities_do_not_remove_frozen_protocol_labels(self):
        rows = self.audit["evaluation_decisions"]
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(row["event_only_label_kept"] for row in rows))
        self.assertTrue(all(row["protocol_label_eligible"] for row in rows))
        self.assertEqual(sum(not row["prior_eligible"] for row in rows), 4)
        self.assertEqual(
            sum(row["exclusion_from_prior_does_not_exclude_label"] for row in rows), 4
        )


class PriorQualityBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.catalog = (
            CatalogItem("C001", "one", "白", 1, 1, 100),
            CatalogItem("C002", "two", "金", 1, 2, 200),
        )
        self.items = [
            {
                "session_id": "world", "treasure_id": f"T{index + 1}",
                "auction_tier": "S", "catalog_id": item.catalog_id,
                "quality": item.quality, "width": item.width,
                "height": item.height, "area": item.width * item.height,
                "value": item.value, "settlement_total": 300,
            }
            for index, item in enumerate(self.catalog)
        ]
        self.session = {
            "session_id": "world", "auction_tier": "S",
            "protocol_settlement_valid": "true", "settlement_total": 300,
            "treasure_count": 2, "detected_value_sum": 300,
            "value_sum_difference": 0, "value_sum_matches": "true",
            "postgame_value_gate_matches": "true",
            "postgame_value_gate_status": "matched",
            "final_treasure_review_status": "auto_confirmed",
            "catalog_independent_of_settlement_total": "true",
            "geometry_settlement_independent": "true",
        }

    def write_csv(self, name, rows, *, fields=None):
        path = self.directory / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields or tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def load(self, *, session_rows=True, quality_filter=True):
        treasures = self.write_csv("treasures.csv", self.items)
        if session_rows is True:
            self.write_csv("sessions.csv", [self.session])
        elif session_rows is not False:
            self.write_csv("sessions.csv", session_rows, fields=tuple(self.session))
        return load_empirical_counts(
            treasures, self.catalog, quality_filter=quality_filter
        )

    def assert_rejected(self, *, reason=None, session_rows=True):
        counts, report = self.load(session_rows=session_rows)
        self.assertEqual(counts.shape, (0, len(self.catalog)))
        self.assertEqual(report["excluded_worlds"], 1)
        if reason:
            self.assertIn(reason, report["exclusion_reason_counts_nonexclusive"])

    def test_consistent_reviewed_world_is_verified(self):
        counts, report = self.load()
        self.assertEqual(counts.tolist(), [[1, 1]])
        self.assertEqual(report["verified_worlds"], 1)

    def test_duplicate_item_key_rejected(self):
        self.items[1]["treasure_id"] = self.items[0]["treasure_id"]
        self.assert_rejected(reason="duplicate_item_key")

    def test_duplicate_session_metadata_rejected(self):
        self.assert_rejected(
            session_rows=[self.session, self.session],
            reason="missing_or_duplicate_session",
        )

    def test_conflicting_embedded_label_rejected_with_metadata(self):
        self.items[0]["settlement_total"] = 301
        self.assert_rejected(reason="catalog_sum_not_settlement")

    def test_conflicting_embedded_label_rejected_without_metadata(self):
        self.items[0]["settlement_total"] = 301
        self.assert_rejected(
            session_rows=False, reason="catalog_sum_not_settlement"
        )

    def test_false_review_flag_rejected(self):
        self.session["postgame_value_gate_matches"] = "false"
        self.assert_rejected(reason="review_or_independence_flag_not_confirmed")

    def test_catalog_metadata_conflict_rejected(self):
        self.items[0]["value"] = 101
        self.assert_rejected(reason="catalog_metadata_conflict")

    def test_present_empty_metadata_is_strict_and_rejects_orphan_world(self):
        self.assert_rejected(
            session_rows=[], reason="missing_or_duplicate_session"
        )

    def test_missing_metadata_is_excluded_even_with_matching_embedded_label(self):
        counts, report = self.load(session_rows=False)
        self.assertEqual(counts.shape, (0, len(self.catalog)))
        self.assertFalse(report["metadata_available"])
        self.assertEqual(report["verified_worlds"], 0)
        self.assertEqual(report["excluded_worlds"], 1)
        self.assertEqual(
            report["exclusion_reason_counts_nonexclusive"],
            {"review_metadata_unavailable": 1},
        )

    def test_missing_metadata_and_missing_label_are_excluded(self):
        for item in self.items:
            item.pop("settlement_total")
        self.assert_rejected(
            session_rows=False, reason="review_metadata_unavailable"
        )

    def test_missing_metadata_legacy_reproduction_requires_explicit_opt_out(self):
        counts, report = self.load(session_rows=False, quality_filter=False)
        self.assertEqual(counts.tolist(), [[1, 1]])
        self.assertEqual(report["rule"], "legacy_unfiltered_for_reproduction")
        self.assertFalse(report["metadata_available"])
        self.assertEqual(report["verified_worlds"], 0)

    def test_legacy_unsafe_is_explicit_and_reproduces_rejected_world(self):
        self.session["postgame_value_gate_matches"] = "false"
        counts, report = self.load(quality_filter=False)
        self.assertEqual(counts.tolist(), [[1, 1]])
        self.assertEqual(report["rule"], "legacy_unfiltered_for_reproduction")
        self.assertEqual(report["verified_worlds"], 0)


if __name__ == "__main__":
    unittest.main()
