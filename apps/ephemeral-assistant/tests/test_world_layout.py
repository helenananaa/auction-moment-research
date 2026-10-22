from __future__ import annotations

import itertools
import random
import time
import unittest
from collections import Counter
from dataclasses import replace

from auction_moment_assistant.models import CatalogItem
from auction_moment_assistant.observations import ObservationSnapshot
from auction_moment_assistant.world_layout import (
    find_compatible_sequence,
    layout_compatible,
    layout_diagnostics,
    layout_matches,
    pack_sequence,
)


def snapshot(**changes) -> ObservationSnapshot:
    return replace(
        ObservationSnapshot(
            revision=1,
            round_number=1,
            bankroll=None,
            events=(),
            map_items=(),
            completeness_confirmed=True,
            completeness_source="human",
            map_rows=None,
            map_height_exact=False,
            map_rows_source="unknown",
            pre_bid_confirmed=True,
            pre_bid_source="human",
        ),
        **changes,
    )


def clue(row=0, column=0, **changes) -> dict:
    return {
        "row": row,
        "column": column,
        "spatial": "top_left",
        "confidence": 0.95,
        **changes,
    }


def grid_oracle(sequence, catalog):
    """Independent cell-set first-fit reference for tiny exhaustive worlds."""
    occupied = set()
    result = []
    for index in sequence:
        item = catalog[index]
        for row in itertools.count():
            found = False
            for column in range(11 - item.width):
                cells = {
                    (r, c)
                    for r in range(row, row + item.height)
                    for c in range(column, column + item.width)
                }
                if occupied.isdisjoint(cells):
                    occupied.update(cells)
                    result.append((index, row, column, item.width, item.height))
                    found = True
                    break
            if found:
                break
    return result, max((r + 1 for r, _c in occupied), default=0)


class WorldLayoutTests(unittest.TestCase):
    def setUp(self):
        self.catalog = (
            CatalogItem("wide", "Wide", "蓝", 3, 1, 10),
            CatalogItem("short", "Short", "白", 2, 1, 20),
            CatalogItem("tall", "Tall", "紫", 3, 2, 30),
        )
        self.low_sequence = (0, 0, 2, 1)
        self.high_sequence = (0, 1, 0, 2)

    def test_bit_rows_match_independent_cell_oracle_exhaustively(self):
        catalog = tuple(
            CatalogItem(str(index), "x", "白", width, height, 1)
            for index, (width, height) in enumerate(
                itertools.product(range(1, 4), repeat=2)
            )
        )
        for length in range(4):
            for sequence in itertools.product(range(len(catalog)), repeat=length):
                expected, expected_height = grid_oracle(sequence, catalog)
                packed = pack_sequence(sequence, catalog)
                self.assertEqual(packed.used_rows, expected_height, sequence)
                actual = [
                    (p.catalog_index, p.row, p.column, p.width, p.height)
                    for p in packed.placements
                ]
                self.assertEqual(actual, expected, sequence)

    def test_same_counts_different_entry_order_changes_height(self):
        self.assertEqual(Counter(self.low_sequence), Counter(self.high_sequence))
        self.assertEqual(pack_sequence(self.low_sequence, self.catalog).used_rows, 2)
        self.assertEqual(pack_sequence(self.high_sequence, self.catalog).used_rows, 3)

    def test_area_and_arbitrary_feasible_packing_do_not_replace_first_fit(self):
        # These same rectangles fit in two rows in low_sequence's arrangement.
        # high_sequence must still realize three rows under the generation rule.
        evidence = snapshot(map_rows=2, map_height_exact=True)
        self.assertTrue(layout_compatible(self.low_sequence, self.catalog, evidence))
        self.assertFalse(layout_compatible(self.high_sequence, self.catalog, evidence))
        all_orders = set(itertools.permutations(self.low_sequence))
        compatible = [
            order
            for order in all_orders
            if layout_compatible(order, self.catalog, evidence)
        ]
        self.assertGreater(len(compatible), 0)
        self.assertLess(len(compatible), len(all_orders))

    def test_exact_height_is_equality_not_capacity(self):
        packed = pack_sequence(self.low_sequence, self.catalog)
        for rows in (1, 3, 20):
            evidence = snapshot(map_rows=rows, map_height_exact=True)
            self.assertFalse(layout_matches(packed, self.catalog, evidence))
        self.assertTrue(
            layout_matches(
                packed, self.catalog, snapshot(map_rows=2, map_height_exact=True)
            )
        )

    def test_estimated_scroll_rows_are_not_capacity(self):
        evidence = snapshot(map_rows=1, map_rows_source="automatic_scroll_estimate")
        self.assertTrue(layout_compatible(self.high_sequence, self.catalog, evidence))
        evidence = replace(evidence, map_rows=50)
        self.assertTrue(layout_compatible(self.high_sequence, self.catalog, evidence))

    def test_anchor_only_geometry_does_not_become_known_size(self):
        evidence = snapshot(
            map_items=(clue(width=1, height=3, marker_width=1, marker_height=1),)
        )
        self.assertTrue(layout_compatible(self.low_sequence, self.catalog, evidence))
        outline = replace(
            evidence, map_items=(clue(spatial="outline", width=1, height=3),)
        )
        self.assertFalse(layout_compatible(self.low_sequence, self.catalog, outline))

    def test_outline_shape_quality_and_complete_identity_are_hard_evidence(self):
        good = clue(0, 6, spatial="outline", width=3, height=2, quality="紫")
        self.assertTrue(
            layout_compatible(
                self.low_sequence, self.catalog, snapshot(map_items=(good,))
            )
        )
        for update in (
            {"width": 2},
            {"quality": "金"},
            {"spatial": "complete", "catalog_id": "short"},
        ):
            self.assertFalse(
                layout_compatible(
                    self.low_sequence,
                    self.catalog,
                    snapshot(map_items=({**good, **update},)),
                )
            )
        complete = clue(0, 6, spatial="complete", catalog_id="tall")
        self.assertTrue(
            layout_compatible(
                self.low_sequence, self.catalog, snapshot(map_items=(complete,))
            )
        )

    def test_anchor_must_be_top_left_not_any_occupied_cell(self):
        evidence = snapshot(map_items=(clue(0, 0), clue(0, 1)))
        self.assertFalse(layout_compatible((0,), self.catalog, evidence))

    def test_same_anchor_evidence_cannot_rebind_to_different_items(self):
        evidence = snapshot(
            map_items=(
                clue(spatial="complete", catalog_id="wide"),
                clue(spatial="complete", catalog_id="short"),
            )
        )
        # Both identities exist but only one can own (0, 0).
        self.assertFalse(layout_compatible((0, 1), self.catalog, evidence))
        duplicate = replace(
            evidence,
            map_items=(clue(quality="蓝"), clue(spatial="complete", catalog_id="wide")),
        )
        self.assertTrue(layout_compatible((0, 1), self.catalog, duplicate))

    def test_threshold_and_human_lock_apply_before_geometry(self):
        bad = clue(8, 9, confidence=0.849)
        self.assertTrue(
            layout_compatible(
                self.low_sequence, self.catalog, snapshot(map_items=(bad,))
            )
        )
        for updates in (
            {"confidence": 0.85},
            {"confidence": 0.0, "human_locked": True},
        ):
            self.assertFalse(
                layout_compatible(
                    self.low_sequence,
                    self.catalog,
                    snapshot(map_items=({**bad, **updates},)),
                )
            )
        self.assertTrue(
            layout_compatible(
                self.low_sequence,
                self.catalog,
                snapshot(map_items=({**bad, "confidence": float("nan")},)),
            )
        )

    def test_uncertain_identity_candidates_are_not_promoted(self):
        uncertain = clue(identity_candidates=(("short", 0.99),), catalog_id="short")
        self.assertTrue(
            layout_compatible((0,), self.catalog, snapshot(map_items=(uncertain,)))
        )

    def test_malformed_trusted_evidence_fails_closed(self):
        for bad in (
            clue(row=0.5),
            clue(column=10),
            clue(spatial="made_up"),
            clue(spatial="outline", width=1),
            clue(spatial="outline", width=0, height=1),
        ):
            self.assertFalse(
                layout_compatible((0,), self.catalog, snapshot(map_items=(bad,)))
            )
        packed = pack_sequence((0,), self.catalog)
        self.assertIn(
            "invalid_exact_map_rows",
            layout_diagnostics(packed, self.catalog, snapshot(map_height_exact=True)),
        )

    def test_empty_sequence_and_invalid_catalog_inputs(self):
        packed = pack_sequence((), self.catalog)
        self.assertEqual(packed.used_rows, 0)
        self.assertEqual(packed.placements, ())
        for invalid in ((-1,), (3,), (0.5,), (True,)):
            with self.assertRaises(ValueError):
                pack_sequence(invalid, self.catalog)
        with self.assertRaises(ValueError):
            pack_sequence((0,), (CatalogItem("bad", "Bad", "白", 11, 1, 1),))

    def test_seed_search_finds_full_anchor_order_with_count_conservation(self):
        catalog = tuple(
            CatalogItem(str(index), "x", "蓝", 1, 1, 1) for index in range(12)
        )
        expected = (7, 0, 9, 3, 10, 4, 1, 11, 5, 8, 2, 6)
        packed = pack_sequence(expected, catalog)
        evidence = snapshot(
            map_rows=2,
            map_height_exact=True,
            map_items=tuple(
                clue(
                    p.row,
                    p.column,
                    spatial="complete",
                    catalog_id=catalog[p.catalog_index].catalog_id,
                )
                for p in packed.placements
            ),
        )
        sequence = find_compatible_sequence(
            (1,) * 12, catalog, evidence, random.Random(10)
        )
        self.assertEqual(sequence, expected)
        self.assertEqual(Counter(sequence), Counter(expected))
        self.assertTrue(layout_compatible(sequence, catalog, evidence))

    def test_seed_search_preserves_rectangles_and_exact_realized_height(self):
        evidence = snapshot(
            map_rows=2,
            map_height_exact=True,
            map_items=(
                clue(0, 6, spatial="outline", width=3, height=2, quality="紫"),
                clue(1, 0, spatial="complete", catalog_id="short"),
            ),
        )
        sequence = find_compatible_sequence(
            (2, 1, 1), self.catalog, evidence, random.Random(8)
        )
        self.assertIsNotNone(sequence)
        self.assertEqual(Counter(sequence), Counter(self.low_sequence))
        self.assertTrue(layout_compatible(sequence, self.catalog, evidence))

    def test_seed_search_does_not_relax_impossible_evidence_or_budget(self):
        evidence = snapshot(map_items=(clue(0, 1),))
        self.assertIsNone(
            find_compatible_sequence(
                (1, 0, 0), self.catalog, evidence, random.Random(1)
            )
        )
        self.assertIsNone(
            find_compatible_sequence(
                (1, 0, 0), self.catalog, snapshot(), random.Random(1), maximum_nodes=0
            )
        )
        self.assertIsNone(
            find_compatible_sequence(
                (1, 0, 0),
                self.catalog,
                snapshot(),
                random.Random(1),
                deadline=time.monotonic() - 1,
            )
        )
        with self.assertRaises(ValueError):
            find_compatible_sequence(
                (1, -1, 0), self.catalog, snapshot(), random.Random(1)
            )

    def test_complete_all_quality_reveal_excludes_unobserved_group_anchors(self):
        for effect in ("reveal_position_by_quality", "reveal_outline_by_quality"):
            event = {
                "round_number": 1,
                "confidence": 1.0,
                "semantics": {"parsed": True, "effect": effect, "quality": "蓝"},
            }
            evidence = snapshot(events=(event,), map_items=(clue(quality="蓝"),))
            self.assertTrue(layout_compatible((0,), self.catalog, evidence))
            self.assertFalse(layout_compatible((0, 0), self.catalog, evidence))
            diagnostics = layout_diagnostics(
                pack_sequence((0, 0), self.catalog), self.catalog, evidence
            )
            self.assertIn(f"unobserved_revealed_anchor:{effect}:0:3", diagnostics)

    def test_complete_all_size_reveal_only_requires_visible_anchor(self):
        event = {
            "round_number": 1,
            "confidence": 1.0,
            "semantics": {
                "parsed": True,
                "effect": "reveal_outline_by_size",
                "width": 3,
                "height": 1,
            },
        }
        missing = snapshot(events=(event,), map_items=(clue(),))
        self.assertFalse(layout_compatible((0, 0), self.catalog, missing))
        # The same marker may have been merged with evidence from another
        # round/source; its current classification need not repeat known size.
        complete = replace(missing, map_items=(clue(), clue(0, 3)))
        self.assertTrue(layout_compatible((0, 0), self.catalog, complete))

    def test_all_group_reveal_allows_cumulative_anchors_from_other_sources(self):
        event = {
            "round_number": 1,
            "confidence": 1.0,
            "semantics": {
                "parsed": True,
                "effect": "reveal_position_by_quality",
                "quality": "蓝",
            },
        }
        other_event = {
            "round_number": 2,
            "confidence": 1.0,
            "semantics": {
                "parsed": True,
                "effect": "reveal_position_random",
                "count": 1,
            },
        }
        evidence = snapshot(
            round_number=2,
            events=(event, other_event),
            map_items=(clue(), clue(0, 3), clue(0, 6, quality="白")),
        )
        self.assertTrue(layout_compatible((0, 0, 1), self.catalog, evidence))
        missing_blue = replace(evidence, map_items=(clue(), clue(0, 6, quality="白")))
        self.assertFalse(layout_compatible((0, 0, 1), self.catalog, missing_blue))

    def test_incomplete_or_unreliable_map_does_not_assert_marker_absence(self):
        event = {
            "round_number": 1,
            "confidence": 1.0,
            "semantics": {
                "parsed": True,
                "effect": "reveal_position_by_quality",
                "quality": "蓝",
            },
        }
        evidence = snapshot(events=(event,), map_items=(clue(),))
        self.assertTrue(
            layout_compatible(
                (0, 0), self.catalog, replace(evidence, completeness_confirmed=False)
            )
        )
        uncertain = replace(evidence, map_items=(clue(), clue(9, 9, confidence=0.8)))
        self.assertTrue(layout_compatible((0, 0), self.catalog, uncertain))
        locked = replace(
            evidence,
            map_items=(clue(), clue(0, 3, confidence=0.0, human_locked=True)),
        )
        self.assertTrue(layout_compatible((0, 0), self.catalog, locked))

    def test_future_untrusted_or_unavailable_reveal_cannot_assert_absence(self):
        event = {
            "round_number": 1,
            "confidence": 1.0,
            "semantics": {
                "parsed": True,
                "effect": "reveal_position_by_quality",
                "quality": "蓝",
            },
        }
        for change in (
            {"round_number": 2},
            {"confidence": 0.8},
            {"semantics": {**event["semantics"], "parsed": False}},
            {"semantics": {**event["semantics"], "reveal_outcome_available": False}},
        ):
            evidence = snapshot(events=({**event, **change},), map_items=(clue(),))
            self.assertTrue(layout_compatible((0, 0), self.catalog, evidence))


if __name__ == "__main__":
    unittest.main()
