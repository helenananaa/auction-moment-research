from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "evaluate_assistant_conditioning.py"
SPEC = importlib.util.spec_from_file_location("conditioning_development_replay", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class DevelopmentReplayTests(unittest.TestCase):
    def test_receipt_path_sanitization_preserves_predictions_and_hashes(self):
        source = str(SCRIPT.resolve())
        report = {
            "provenance_sha256": {source: "input_hash"},
            "runtime_source_sha256": {source: "source_hash"},
            "runtime_source_sha256_after": {source: "source_hash"},
            "summary": {"mape": 0.42},
            "decisions": [{"p50": 123}],
        }
        sanitized = replay.sanitize_receipt_paths(report, original_sha256="receipt_hash")
        self.assertEqual(sanitized["provenance_sha256"], {"scripts/evaluate_assistant_conditioning.py": "input_hash"})
        self.assertEqual(sanitized["summary"], report["summary"])
        self.assertEqual(sanitized["decisions"], report["decisions"])
        self.assertFalse(sanitized["sanitization"]["resampled"])
        self.assertNotIn(str(SCRIPT.resolve()), str(sanitized))

    def test_nonfinite_failed_chain_diagnostics_still_form_valid_json(self):
        self.assertEqual(replay.finite_json({"rhat": float("inf")}), {"rhat": "nonfinite:inf"})

    def event_row(self, **changes):
        return {
            "session_id": "session_public",
            "round": "1",
            "scope": "public",
            "effect": "quality_average_area",
            "group_dimension": "quality",
            "group_value": "金",
            "observed_value": "233",
            "value_scale": "100",
            "validation_expected": "999999999",
            "reported_item_count": "777",
            "quality_codes": "[5]",
            **changes,
        }

    def test_semantics_use_event_time_measurements_without_validation_fields(self):
        self.assertEqual(replay.event_semantics(self.event_row()), {
            "parsed": True,
            "effect": "quality_average_area",
            "quality": "金",
            "observed_average_cells": 2.33,
        })
        result = replay.event_semantics(self.event_row(
            effect="reveal_position_quality_random",
            group_dimension="random",
            group_value="3",
        ))
        self.assertEqual(result, {
            "parsed": True,
            "effect": "reveal_position_quality_random",
            "reveal_outcome_available": False,
            "count": 3,
        })

    def write_rows(self, path, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def test_snapshot_excludes_label_future_events_and_final_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            core = Path(directory)
            self.write_rows(core / "sessions.csv", [{
                "session_id": "session_public", "protocol_settlement_valid": "true",
                "settlement_total": "987654321", "date": "2026-08-02",
            }])
            self.write_rows(core / "rounds.csv", [{
                "session_id": "session_public", "round": "1", "map_rows": "12",
                "own_bid": "444444", "map_height_exact": "false",
            }])
            self.write_rows(core / "event_instances.csv", [
                self.event_row(scope="public"),
                self.event_row(scope="personal", effect="quality_item_count", observed_value="7"),
                self.event_row(round="2", observed_value="555555"),
            ])
            selected = replay.select_inputs(core, limit=10, rounds=(1,))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["label"], 987654321)
        snapshot = selected[0]["snapshot"]
        self.assertEqual(snapshot.map_items, ())
        self.assertEqual(snapshot.map_rows, 12)
        self.assertFalse(snapshot.map_height_exact)
        self.assertFalse(snapshot.completeness_confirmed)
        self.assertEqual(len(snapshot.events), 2)
        feature_text = str(asdict(snapshot))
        for canary in ("987654321", "555555", "999999999", "444444", "777"):
            self.assertNotIn(canary, feature_text)


if __name__ == "__main__":
    unittest.main()
