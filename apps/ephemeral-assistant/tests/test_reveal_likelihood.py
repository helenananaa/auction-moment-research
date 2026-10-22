from __future__ import annotations

import itertools
import math
import unittest
from dataclasses import replace

from auction_moment_assistant.models import CatalogItem
from auction_moment_assistant.observations import ObservationSnapshot
from auction_moment_assistant.reveal_likelihood import (
    RevealObservation,
    build_reveal_observations,
    reveal_log_likelihood,
)


CATALOG = (
    CatalogItem("b1", "blue 1", "蓝", 1, 1, 10),
    CatalogItem("b2", "blue 2", "蓝", 1, 1, 20),
    CatalogItem("w1", "white", "白", 1, 1, 1),
)


def event(
    round_number=1, kind="public", effect="reveal_position_quality_random", count=1
):
    return {
        "round_number": round_number,
        "kind": kind,
        "confidence": 1.0,
        "semantics": {"parsed": True, "effect": effect, "count": count},
    }


def item(row=0, first_seen_round=1, **kwargs):
    return {
        "row": row,
        "column": 0,
        "quality": "蓝",
        "confidence": 1.0,
        "first_seen_round": first_seen_round,
        **kwargs,
    }


def snapshot(*, events=None, items=None, round_number=1):
    return ObservationSnapshot(
        revision=1,
        round_number=round_number,
        bankroll=None,
        events=tuple(
            events
            if events is not None
            else [event(), event(kind="personal", effect="total_item_count", count=20)]
        ),
        map_items=tuple(items if items is not None else [item()]),
        completeness_confirmed=True,
        completeness_source="human_confirmed",
        map_rows=None,
        map_height_exact=False,
        map_rows_source="missing",
        pre_bid_confirmed=True,
        pre_bid_source="human_confirmed",
    )


class RevealLikelihoodTests(unittest.TestCase):
    def test_single_blue_gives_tenfold_likelihood_ratio(self):
        observations, diagnostics = build_reveal_observations(snapshot(), CATALOG)
        self.assertEqual(len(observations), 1)
        self.assertIn(
            "reveal_likelihood:uniform_without_replacement_assumption", diagnostics
        )
        high = reveal_log_likelihood([10, 0, 10], observations, CATALOG)
        low = reveal_log_likelihood([1, 0, 19], observations, CATALOG)
        self.assertAlmostEqual(math.exp(high - low), 10.0)

    def test_quality_likelihood_matches_exact_small_world_enumeration(self):
        physical_items = ["蓝", "蓝", "蓝", "白", "白"]
        samples = list(itertools.combinations(range(5), 3))
        total_probability = 0.0
        for blue_count in range(4):
            white_count = 3 - blue_count
            expected = sum(
                sum(physical_items[index] == "蓝" for index in sample) == blue_count
                for sample in samples
            ) / len(samples)
            categories = tuple(
                (name, count)
                for name, count in (("蓝", blue_count), ("白", white_count))
                if count
            )
            observation = RevealObservation(
                1, "public", "reveal_position_quality_random", categories, 3
            )
            actual = math.exp(reveal_log_likelihood([2, 1, 2], [observation], CATALOG))
            self.assertAlmostEqual(actual, expected)
            total_probability += actual
        self.assertAlmostEqual(total_probability, 1.0)

    def test_duplicate_identity_uses_unordered_combinations(self):
        observation = RevealObservation(
            1, "public", "reveal_identity_random", (("b1", 2), ("w1", 1)), 3
        )
        # C(3,2) C(2,1) / C(6,3), not independent draws or ordered paths.
        actual = math.exp(reveal_log_likelihood([3, 1, 2], [observation], CATALOG))
        self.assertAlmostEqual(actual, 6 / 20)

    def test_identity_likelihood_matches_exact_small_world_enumeration(self):
        physical_items = ["b1", "b1", "b2", "w1", "w1"]
        samples = list(itertools.combinations(range(5), 3))
        probability = 0.0
        for b1 in range(4):
            for b2 in range(4 - b1):
                w1 = 3 - b1 - b2
                selected = {"b1": b1, "b2": b2, "w1": w1}
                expected = sum(
                    all(
                        sum(physical_items[index] == key for index in sample) == count
                        for key, count in selected.items()
                    )
                    for sample in samples
                ) / len(samples)
                observation = RevealObservation(
                    1,
                    "public",
                    "reveal_identity_random",
                    tuple((key, value) for key, value in selected.items() if value),
                    3,
                )
                actual = math.exp(
                    reveal_log_likelihood([2, 1, 2], [observation], CATALOG)
                )
                self.assertAlmostEqual(actual, expected)
                probability += actual
        self.assertAlmostEqual(probability, 1.0)

    def test_zero_or_insufficient_population_has_zero_likelihood(self):
        observation = RevealObservation(
            1, "public", "reveal_identity_random", (("b1", 2),), 2
        )
        for population in ([0, 0, 0], [1, 0, 10], [0, 3, 10]):
            self.assertEqual(
                reveal_log_likelihood(population, [observation], CATALOG), -math.inf
            )

    def test_position_likelihood_sums_to_bag_likelihood_over_matching_subsets(self):
        physical_items = ["蓝", "蓝", "蓝", "白", "白"]
        observation = RevealObservation(
            1, "public", "reveal_position_quality_random", (("蓝", 2), ("白", 1)), 3
        )
        position_probability = math.exp(
            reveal_log_likelihood(
                [2, 1, 2], [observation], CATALOG, conditioned_on_positions=True
            )
        )
        self.assertAlmostEqual(position_probability, 1 / math.comb(5, 3))
        matching_positions = sum(
            sum(physical_items[index] == "蓝" for index in positions) == 2
            for positions in itertools.combinations(range(5), 3)
        )
        bag_probability = math.exp(
            reveal_log_likelihood([2, 1, 2], [observation], CATALOG)
        )
        self.assertAlmostEqual(
            matching_positions * position_probability, bag_probability
        )

    def test_exact_position_selection_depends_on_population_not_category_count(self):
        observation = RevealObservation(
            1, "public", "reveal_position_quality_random", (("蓝", 1),), 1
        )
        large = reveal_log_likelihood(
            [10, 0, 10], [observation], CATALOG, conditioned_on_positions=True
        )
        small = reveal_log_likelihood(
            [1, 0, 9], [observation], CATALOG, conditioned_on_positions=True
        )
        self.assertAlmostEqual(math.exp(large), 1 / 20)
        self.assertAlmostEqual(math.exp(small), 1 / 10)
        self.assertAlmostEqual(math.exp(large - small), 0.5)
        same_size = reveal_log_likelihood(
            [1, 0, 19], [observation], CATALOG, conditioned_on_positions=True
        )
        self.assertEqual(large, same_size)
        self.assertEqual(
            reveal_log_likelihood(
                [0, 0, 20], [observation], CATALOG, conditioned_on_positions=True
            ),
            -math.inf,
        )

    def test_exact_repeated_event_is_not_counted_twice(self):
        observation = RevealObservation(
            1, "public", "reveal_identity_random", (("b1", 1),), 1
        )
        once = reveal_log_likelihood([3, 1, 2], [observation], CATALOG)
        self.assertEqual(
            once, reveal_log_likelihood([3, 1, 2], [observation, observation], CATALOG)
        )
        with self.assertRaisesRegex(ValueError, "independence"):
            reveal_log_likelihood(
                [3, 1, 2], [observation, replace(observation, round_number=2)], CATALOG
            )

    def test_pure_position_event_does_not_use_visible_quality(self):
        value = snapshot(
            events=[
                event(effect="reveal_position_random"),
                event(kind="personal", effect="total_item_count"),
            ]
        )
        observations, diagnostics = build_reveal_observations(value, CATALOG)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].sample_counts, ())
        self.assertEqual(reveal_log_likelihood([10, 0, 10], observations, CATALOG), 0.0)
        self.assertAlmostEqual(
            math.exp(
                reveal_log_likelihood(
                    [10, 0, 10], observations, CATALOG, conditioned_on_positions=True
                )
            ),
            1 / 20,
        )

    def test_position_only_sample_needs_no_quality_or_identity(self):
        value = snapshot(
            events=[
                event(effect="reveal_position_random", count=2),
                event(kind="personal", effect="total_item_count"),
            ],
            items=[item(quality=None), item(row=1, quality=None)],
        )
        observations, _ = build_reveal_observations(value, CATALOG)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].sample_size, 2)
        self.assertEqual(observations[0].sample_counts, ())
        self.assertAlmostEqual(
            math.exp(
                reveal_log_likelihood(
                    [0, 0, 5], observations, CATALOG, conditioned_on_positions=True
                )
            ),
            1 / math.comb(5, 2),
        )
        self.assertEqual(
            reveal_log_likelihood([0, 0, 1], observations, CATALOG), -math.inf
        )

    def test_position_only_rejects_category_counts_and_invalid_sample_size(self):
        for observation in (
            RevealObservation(1, "public", "reveal_position_random", (("蓝", 1),), 1),
            RevealObservation(1, "public", "reveal_position_random", (), -1),
            RevealObservation(1, "public", "reveal_position_random", (), 0),
        ):
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                reveal_log_likelihood([10, 0, 10], [observation], CATALOG)

    def test_position_only_requires_trusted_valid_anchors(self):
        for observed, reason in (
            (item(quality=None, confidence=0.2), "sample_attributes_untrusted"),
            (item(row=-1, quality=None), "sample_position_invalid"),
            (item(column=10, quality=None), "sample_position_invalid"),
            (item(row=0.5, quality=None), "sample_position_invalid"),
            (item(row=True, quality=None), "sample_position_invalid"),
        ):
            value = snapshot(
                events=[
                    event(effect="reveal_position_random"),
                    event(kind="personal", effect="total_item_count"),
                ],
                items=[observed],
            )
            with self.subTest(observed=observed):
                observations, diagnostics = build_reveal_observations(value, CATALOG)
                self.assertFalse(observations)
                self.assertIn(f"reveal_likelihood:{reason}", diagnostics)

    def test_negative_or_fractional_random_count_is_not_evidence(self):
        for count in (-1, 0, 1.5, True):
            value = snapshot(
                events=[
                    event(effect="reveal_position_random", count=count),
                    event(kind="personal", effect="total_item_count"),
                ]
            )
            with self.subTest(count=count):
                observations, diagnostics = build_reveal_observations(value, CATALOG)
                self.assertFalse(observations)
                self.assertIn(
                    "reveal_likelihood:random_sample_count_missing", diagnostics
                )

    def test_no_random_reveal_event_does_not_construct_a_sample(self):
        value = snapshot(events=[event(effect="reveal_outline_by_size")])
        observations, diagnostics = build_reveal_observations(value, CATALOG)
        self.assertFalse(observations)
        self.assertIn("reveal_likelihood:no_random_reveal_event", diagnostics)

    def test_later_visible_items_are_not_part_of_first_random_sample(self):
        value = snapshot(
            round_number=2,
            items=[item(), item(row=1, first_seen_round=2, quality="白")],
        )
        observations, _ = build_reveal_observations(value, CATALOG)
        self.assertEqual(observations[0].sample_counts, (("蓝", 1),))

    def test_earlier_visible_pool_makes_later_reveal_unattributable(self):
        value = snapshot(
            round_number=2,
            events=[event(2), event(2, kind="personal", effect="total_item_count")],
            items=[item(), item(row=1, first_seen_round=2)],
        )
        observations, diagnostics = build_reveal_observations(value, CATALOG)
        self.assertFalse(observations)
        self.assertIn(
            "reveal_likelihood:earlier_visible_items_unknown_reveal_pool", diagnostics
        )

    def test_cross_round_events_are_not_assumed_independent(self):
        value = snapshot(
            round_number=2,
            events=[
                event(),
                event(kind="personal", effect="total_item_count"),
                event(2),
            ],
            items=[item(), item(row=1, first_seen_round=2)],
        )
        observations, diagnostics = build_reveal_observations(value, CATALOG)
        self.assertEqual(len(observations), 1)
        self.assertIn(
            "reveal_likelihood:later_rounds_skipped_unknown_reveal_pool", diagnostics
        )

    def test_delayed_first_seen_does_not_hide_earlier_reveal(self):
        value = snapshot(
            round_number=2,
            events=[
                event(effect="reveal_position_random"),
                event(kind="personal", effect="total_item_count"),
                event(2),
                event(2, kind="personal", effect="total_item_count"),
            ],
            items=[item(first_seen_round=2)],
        )
        observations, diagnostics = build_reveal_observations(value, CATALOG)
        self.assertFalse(observations)
        self.assertIn("reveal_likelihood:first_seen_sample_count_mismatch", diagnostics)

    def test_first_reveal_after_complete_statistical_history_is_usable(self):
        value = snapshot(
            round_number=2,
            events=[
                event(effect="total_item_count"),
                event(kind="personal", effect="highest_quality"),
                event(2),
                event(2, kind="personal", effect="total_item_count"),
            ],
            items=[item(first_seen_round=2)],
        )
        observations, _ = build_reveal_observations(value, CATALOG)
        self.assertEqual(len(observations), 1)

    def test_unique_complete_identity_sample_is_accepted(self):
        value = snapshot(
            events=[
                event(effect="reveal_identity_random", count=2),
                event(kind="personal", effect="total_item_count"),
            ],
            items=[
                item(catalog_id="b1", spatial="complete"),
                item(row=1, catalog_id="b1", spatial="complete"),
            ],
        )
        observations, _ = build_reveal_observations(value, CATALOG)
        self.assertEqual(observations[0].sample_counts, (("b1", 2),))

    def test_missing_or_conflicted_attribution_fails_closed(self):
        cases = [
            (
                replace(snapshot(), completeness_confirmed=False),
                "map_completeness_not_confirmed",
            ),
            (snapshot(events=[event()]), "event_ledger_incomplete"),
            (
                snapshot(
                    events=[
                        event(),
                        event(kind="personal", effect="reveal_position_random"),
                    ]
                ),
                "multiple_reveal_sources",
            ),
            (
                snapshot(items=[item(first_seen_round=None)]),
                "first_seen_history_missing",
            ),
            (snapshot(items=[]), "first_seen_sample_count_mismatch"),
            (snapshot(items=[item(confidence=0.2)]), "sample_attributes_untrusted"),
            (snapshot(items=[item(quality=None)]), "sample_quality_incomplete"),
            (snapshot(items=[item(catalog_id="w1")]), "identity_quality_conflict"),
            (
                snapshot(
                    events=[
                        event(),
                        {**event(kind="personal"), "semantics": {"parsed": False}},
                    ]
                ),
                "event_attributes_untrusted",
            ),
        ]
        for value, reason in cases:
            with self.subTest(reason=reason):
                observations, diagnostics = build_reveal_observations(value, CATALOG)
                self.assertFalse(observations)
                self.assertIn(f"reveal_likelihood:{reason}", diagnostics)

    def test_summary_observed_count_does_not_replace_random_count(self):
        random_event = event()
        random_event["semantics"].pop("count")
        random_event["semantics"]["observed_count"] = 1
        value = snapshot(
            events=[random_event, event(kind="personal", effect="total_item_count")]
        )
        observations, diagnostics = build_reveal_observations(value, CATALOG)
        self.assertFalse(observations)
        self.assertIn("reveal_likelihood:random_sample_count_missing", diagnostics)

    def test_explicitly_unavailable_reveal_outcome_is_not_used(self):
        random_event = event()
        random_event["semantics"]["reveal_outcome_available"] = False
        value = snapshot(
            events=[random_event, event(kind="personal", effect="total_item_count")]
        )
        observations, diagnostics = build_reveal_observations(value, CATALOG)
        self.assertFalse(observations)
        self.assertIn("reveal_likelihood:reveal_outcome_unavailable", diagnostics)

    def test_duplicates_do_not_create_extra_sample_items(self):
        value = snapshot(items=[item(), item()])
        observations, _ = build_reveal_observations(value, CATALOG)
        self.assertEqual(observations[0].sample_size, 1)

    def test_invalid_population_input_is_rejected(self):
        for counts in ([1, 2], [1, -1, 2], [1.2, 1, 1], [math.nan, 1, 1]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                reveal_log_likelihood(counts, [], CATALOG)


if __name__ == "__main__":
    unittest.main()
