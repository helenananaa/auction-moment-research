from __future__ import annotations

import itertools
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from ortools.sat.python import cp_model

from auction_moment_assistant.conditional_constraints import compile_hard_constraints
from auction_moment_assistant.models import CatalogItem
from auction_moment_assistant.observations import ObservationSnapshot


QUALITY_ORDER = ("白", "蓝", "紫", "金", "彩")


def snapshot(*semantics, map_items=(), **changes):
    return replace(
        ObservationSnapshot(
            revision=1,
            round_number=1,
            bankroll=None,
            events=tuple(
                {
                    "round_number": 1,
                    "kind": "public",
                    "semantics": {"parsed": True, **event},
                }
                for event in semantics
            ),
            map_items=map_items,
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


def clue(row=0, column=0, **changes):
    return {
        "row": row,
        "column": column,
        "confidence": 0.95,
        "spatial": "top_left",
        **changes,
    }


class Solutions(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        super().__init__()
        self.variables = variables
        self.counts = set()

    def on_solution_callback(self):
        self.counts.add(tuple(self.value(variable) for variable in self.variables))


def enumerate_counts(model, variables):
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    callback = Solutions(variables)
    solver.solve(model, callback)
    return callback.counts


def numeric_oracle(counts, catalog, semantics):
    """Direct integer evaluation, independent of the compiled inequalities."""
    effect = semantics["effect"]
    indices = range(len(catalog))
    if effect.startswith("quality_"):
        indices = [
            index for index in indices if catalog[index].quality == semantics["quality"]
        ]
    elif effect.startswith("size_"):
        indices = [
            index
            for index in indices
            if (catalog[index].width, catalog[index].height)
            == (semantics["width"], semantics["height"])
        ]
    count = sum(counts[index] for index in indices)
    value = sum(counts[index] * catalog[index].value for index in indices)
    area = sum(
        counts[index] * catalog[index].width * catalog[index].height
        for index in indices
    )
    if effect.endswith("item_count"):
        return count == semantics["observed_count"]
    if effect.endswith("total_value"):
        return value == semantics["observed_value"]
    if effect.endswith("average_value"):
        return count > 0 and value // count == semantics["observed_value"]
    if effect.endswith("total_area") or effect == "total_occupied_cells":
        return area == semantics["observed_cells"]
    if effect.endswith("average_area") or effect == "average_item_area":
        observed = math.floor(semantics["observed_average_cells"] * 100 + 1e-9)
        return count > 0 and 100 * area // count == observed
    if effect in {"max_item_value", "max_value_per_cell"}:
        present = [
            item.value
            if effect == "max_item_value"
            else item.value // (item.width * item.height)
            for index, item in enumerate(catalog)
            if counts[index]
        ]
        return max(present, default=-1) == semantics["observed_value"]
    if effect == "highest_quality":
        present = [
            QUALITY_ORDER.index(item.quality)
            for index, item in enumerate(catalog)
            if counts[index]
        ]
        return max(present, default=-1) == QUALITY_ORDER.index(
            semantics["observed_quality"]
        )
    raise AssertionError(effect)


class ConditionalConstraintsTests(unittest.TestCase):
    def setUp(self):
        self.catalog = (
            CatalogItem("white", "White", "白", 1, 1, 10),
            CatalogItem("blue_wide", "Blue wide", "蓝", 2, 1, 23),
            CatalogItem("blue_tall", "Blue tall", "蓝", 1, 2, 31),
            CatalogItem("purple", "Purple", "紫", 2, 2, 77),
        )
        self.world_model = SimpleNamespace(
            _count_probabilities={count: 0.2 for count in range(5)}, minimum_quotas={}
        )
        self.worlds = {
            counts
            for counts in itertools.product(range(5), repeat=len(self.catalog))
            if sum(counts) <= 4
        }

    def compile(self, evidence, world_model=None):
        return compile_hard_constraints(
            evidence, self.catalog, world_model or self.world_model
        )

    def solutions(self, evidence, world_model=None):
        model, variables, diagnostics = self.compile(evidence, world_model)
        return enumerate_counts(model, variables), diagnostics

    def test_exact_numeric_semantics_match_exhaustive_integer_oracle(self):
        cases = (
            {"effect": "total_item_count", "observed_count": 3},
            {"effect": "quality_item_count", "quality": "蓝", "observed_count": 1},
            {"effect": "size_item_count", "width": 2, "height": 1, "observed_count": 1},
            {"effect": "quality_item_count", "quality": "金", "observed_count": 0},
            {"effect": "quality_total_value", "quality": "蓝", "observed_value": 54},
            {
                "effect": "size_total_value",
                "width": 1,
                "height": 1,
                "observed_value": 20,
            },
            {"effect": "quality_total_value", "quality": "紫", "observed_value": 0},
            {"effect": "quality_average_value", "quality": "蓝", "observed_value": 27},
            {
                "effect": "size_average_value",
                "width": 1,
                "height": 1,
                "observed_value": 10,
            },
            {"effect": "quality_average_value", "quality": "蓝", "observed_value": -1},
            {
                "effect": "size_average_value",
                "width": 3,
                "height": 3,
                "observed_value": 0,
            },
            {"effect": "quality_total_area", "quality": "蓝", "observed_cells": 4},
            {"effect": "size_total_area", "width": 1, "height": 1, "observed_cells": 2},
            {"effect": "total_occupied_cells", "observed_cells": 5},
            {
                "effect": "quality_average_area",
                "quality": "蓝",
                "observed_average_cells": 2.0,
            },
            {
                "effect": "size_average_area",
                "width": 1,
                "height": 1,
                "observed_average_cells": 1.0,
            },
            {"effect": "average_item_area", "observed_average_cells": 1.66},
            {"effect": "average_item_area", "observed_average_cells": 1.67},
            {"effect": "max_item_value", "observed_value": 31},
            {"effect": "max_item_value", "observed_value": -1},
            {"effect": "max_item_value", "observed_value": 0},
            {"effect": "max_value_per_cell", "observed_value": 15},
            {"effect": "max_value_per_cell", "observed_value": -1},
            {"effect": "highest_quality", "observed_quality": "蓝"},
            {"effect": "highest_quality", "observed_quality": "紫"},
        )
        for semantics in cases:
            with self.subTest(semantics=semantics):
                actual, diagnostics = self.solutions(snapshot(semantics))
                expected = {
                    counts
                    for counts in self.worlds
                    if numeric_oracle(counts, self.catalog, semantics)
                }
                self.assertEqual(diagnostics, ())
                self.assertEqual(actual, expected)

    def test_multiple_events_intersect_instead_of_replacing_constraints(self):
        events = (
            {"effect": "total_item_count", "observed_count": 3},
            {"effect": "quality_average_value", "quality": "蓝", "observed_value": 27},
            {"effect": "highest_quality", "observed_quality": "蓝"},
        )
        actual, diagnostics = self.solutions(snapshot(*events))
        expected = {
            counts
            for counts in self.worlds
            if all(numeric_oracle(counts, self.catalog, event) for event in events)
        }
        self.assertEqual(actual, expected)
        self.assertEqual(diagnostics, ())

    def test_prior_support_preserves_holes_and_minimum_quality_quotas(self):
        prior = SimpleNamespace(
            _count_probabilities={0: 0.0, 1: 0.3, 2: 0.0, 3: 0.7},
            minimum_quotas={"白": 1, "蓝": 1},
        )
        actual, diagnostics = self.solutions(snapshot(), prior)
        self.assertEqual(
            actual,
            {
                counts
                for counts in self.worlds
                if sum(counts) == 3 and counts[0] >= 1 and counts[1] + counts[2] >= 1
            },
        )
        self.assertEqual(diagnostics, ())

    def test_min_max_fallback_requires_a_declared_support(self):
        actual, diagnostics = self.solutions(
            snapshot(), SimpleNamespace(min_count=1, max_count=2, minimum_quotas={})
        )
        self.assertEqual(
            actual, {counts for counts in self.worlds if 1 <= sum(counts) <= 2}
        )
        self.assertEqual(diagnostics, ())
        actual, diagnostics = self.solutions(snapshot(), SimpleNamespace())
        self.assertEqual(actual, set())
        self.assertIn("unsupported:world_model_count_support", diagnostics)

    def test_unknown_unparsed_and_malformed_evidence_fail_with_diagnostics(self):
        cases = (
            {"effect": "invented_effect"},
            {"effect": "total_item_count", "observed_count": 2, "parsed": False},
            {"effect": "quality_item_count", "quality": "未知", "observed_count": 2},
            {"effect": "size_item_count", "width": 2, "observed_count": 2},
            {"effect": "average_item_area", "observed_average_cells": float("nan")},
        )
        for semantics in cases:
            actual, diagnostics = self.solutions(snapshot(semantics))
            self.assertEqual(actual, set())
            self.assertTrue(
                any(issue.startswith("unsupported:") for issue in diagnostics)
            )

    def test_future_event_is_not_consumed(self):
        future = {
            "round_number": 2,
            "kind": "public",
            "semantics": {"parsed": False, "effect": "unparsed"},
        }
        actual, diagnostics = self.solutions(snapshot(events=(future,)))
        self.assertEqual(actual, self.worlds)
        self.assertEqual(diagnostics, ())

    def test_known_informational_events_are_left_to_reveal_likelihood(self):
        evidence = snapshot({"effect": "reveal_identity_random", "count": 2})
        actual, diagnostics = self.solutions(evidence)
        self.assertEqual(actual, self.worlds)
        self.assertEqual(diagnostics, ())

    def test_same_anchor_clues_intersect_without_double_counting(self):
        evidence = snapshot(
            map_items=(
                clue(quality="蓝"),
                clue(spatial="outline", width=2, height=1),
            )
        )
        actual, diagnostics = self.solutions(evidence)
        self.assertEqual(actual, {counts for counts in self.worlds if counts[1] >= 1})
        self.assertEqual(diagnostics, ())

    def test_different_anchors_with_same_candidate_group_need_distinct_items(self):
        evidence = snapshot(
            map_items=(
                clue(spatial="complete", catalog_id="white"),
                clue(0, 1, spatial="complete", catalog_id="white"),
            )
        )
        actual, diagnostics = self.solutions(evidence)
        self.assertEqual(actual, {counts for counts in self.worlds if counts[0] >= 2})
        self.assertEqual(diagnostics, ())

    def test_anchor_geometry_is_not_item_size_and_low_confidence_is_ignored(self):
        evidence = snapshot(
            map_items=(
                clue(width=3, height=3),
                clue(8, 8, confidence=0.1, spatial="complete", catalog_id="unknown"),
            )
        )
        actual, diagnostics = self.solutions(evidence)
        self.assertEqual(actual, {counts for counts in self.worlds if sum(counts) >= 1})
        self.assertEqual(diagnostics, ())

    def test_human_lock_overrides_confidence_and_conflicting_identity_fails(self):
        evidence = snapshot(
            map_items=(
                clue(
                    confidence=0.0,
                    human_locked=True,
                    spatial="complete",
                    catalog_id="white",
                    quality="蓝",
                ),
            )
        )
        actual, diagnostics = self.solutions(evidence)
        self.assertEqual(actual, set())
        self.assertIn("infeasible:map_identity_shape_quality_conflict", diagnostics)


if __name__ == "__main__":
    unittest.main()
