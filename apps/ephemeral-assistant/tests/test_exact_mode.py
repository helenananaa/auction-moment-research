"""Experimental exact-constraint mode: opt-in, evidence, toy exactness, early start."""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import itertools
import math
import multiprocessing
import threading
import time
import types
import unittest
from collections import Counter

import numpy as np

from auction_moment_assistant.cli import build_parser
from auction_moment_assistant.exact_evidence import RowsTable
from auction_moment_assistant.latest_model import LatestV6V2Predictor
from auction_moment_assistant.models import CatalogItem
from auction_moment_assistant.observations import (
    EventObservation,
    MapObservation,
    ObservationSnapshot,
    ObservationStore,
)
from auction_moment_assistant.ocr import OcrCapture
from auction_moment_assistant.predictor import PredictionResult
from auction_moment_assistant.runtime import CapturePipeline, InferenceCoordinator
from auction_moment_assistant.vision import ViewportRecognition

HAS_SCIPY = importlib.util.find_spec("scipy") is not None


def semantics(effect, **fields):
    return {"parsed": True, "effect": effect, **fields}


def event(round_number, kind, effect, **fields):
    return {
        "round_number": round_number,
        "kind": kind,
        "confidence": 0.99,
        "human_locked": False,
        "semantics": semantics(effect, **fields),
    }


def item(row, column, *, quality=None, catalog_id=None, width=None, height=None,
         spatial="top_left", first_seen_round=1, quality_round=None):
    # Like the store: a quality seen at the first sighting is dated to it.
    if quality_round is None and quality is not None:
        quality_round = first_seen_round
    return {
        "row": row, "column": column, "quality": quality, "catalog_id": catalog_id,
        "width": width, "height": height, "spatial": spatial, "confidence": 0.99,
        "human_locked": False, "first_seen_round": first_seen_round,
        "quality_first_round": quality_round,
    }


def snapshot(events, items=(), *, round_number=1, complete=True, rows=8, exact=True):
    return ObservationSnapshot(
        revision=1, round_number=round_number, bankroll=None, events=tuple(events),
        map_items=tuple(items), completeness_confirmed=complete,
        completeness_source="automatic_scroll_scan" if complete else "missing",
        map_rows=rows, map_height_exact=exact,
        map_rows_source="human_confirmed" if exact else "automatic_scroll_estimate",
        pre_bid_confirmed=True, pre_bid_source="human_confirmed",
    )


CATALOG = (
    CatalogItem("w1", "白1", "白", 1, 1, 10),
    CatalogItem("b1", "蓝1", "蓝", 2, 1, 45),
    CatalogItem("p1", "紫1", "紫", 2, 2, 151),
    CatalogItem("g1", "金1", "金", 1, 2, 409),
    CatalogItem("g2", "金2", "金", 2, 2, 517),
    CatalogItem("c1", "彩1", "彩", 1, 1, 1500),
)


class ExactModeOptInTests(unittest.TestCase):
    def test_cli_flags_default_to_the_released_path(self):
        args = build_parser().parse_args([])
        self.assertFalse(args.experimental_exact_sampling)
        self.assertEqual(args.exact_budget_seconds, 8.0)
        self.assertEqual(args.exact_lanes, 4)
        self.assertEqual(args.exact_refine_seconds, 45.0)
        args = build_parser().parse_args(
            ["--experimental-exact-sampling", "--exact-budget-seconds", "5", "--exact-lanes", "2",
             "--exact-refine-seconds", "0"]
        )
        self.assertTrue(args.experimental_exact_sampling)
        self.assertEqual((args.exact_budget_seconds, args.exact_lanes, args.exact_refine_seconds), (5.0, 2, 0.0))

    def test_predictor_keeps_exact_mode_off_by_default(self):
        parameters = inspect.signature(LatestV6V2Predictor.__init__).parameters
        self.assertIs(parameters["exact_sampling"].default, False)

    def test_packaged_rows_table_scores_map_heights(self):
        table = RowsTable.packaged()
        scores = table.log_likelihood(11, np.array([100, 100, 100]))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertEqual(table.log_likelihood(30, np.array([100]))[0], -np.inf)


class PreviewSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.store = ObservationStore()
        self.store.apply_capture(
            round_number=1,
            events=(
                EventObservation(1, "public", "a", 0.99, semantics("highest_quality", observed_quality="金")),
                EventObservation(1, "personal", "b", 0.99, semantics("max_item_value", observed_value=517)),
            ),
            bankroll=None,
            map_items=(MapObservation(row=0, column=0, quality="金", confidence=0.95),),
        )
        self.store.apply_automatic_scan_proof(round_number=1, map_rows=8, complete=True, pre_bid=True)

    def test_preview_adds_the_new_round_without_changing_the_store(self):
        new = (EventObservation(2, "public", "c", 0.99, semantics("total_occupied_cells", observed_cells=20)),)
        preview = self.store.preview_snapshot(2, new)
        self.assertEqual(preview.round_number, 2)
        self.assertIn(2, {e["round_number"] for e in preview.events})
        self.assertTrue(preview.completeness_confirmed)
        self.assertFalse(preview.pre_bid_confirmed)
        self.assertEqual(len(preview.map_items), 1)
        current = self.store.snapshot()
        self.assertEqual(current.round_number, 1)
        self.assertNotIn(2, {e["round_number"] for e in current.events})

    def test_human_corrections_win_in_the_preview(self):
        self.store.correct_event(2, "public", "人工", semantics("total_occupied_cells", observed_cells=30))
        new = (EventObservation(2, "public", "c", 0.99, semantics("total_occupied_cells", observed_cells=20)),)
        preview = self.store.preview_snapshot(2, new)
        public = [e for e in preview.events if e["round_number"] == 2 and e["kind"] == "public"]
        self.assertEqual(public[0]["semantics"]["observed_cells"], 30)


class QualityDatingTests(unittest.TestCase):
    def capture(self, store, round_number, **fields):
        store.apply_capture(round_number=round_number, events=(), bankroll=None,
                            map_items=(MapObservation(row=2, column=3, **fields),))
        return store.snapshot().map_items

    def test_store_keeps_the_round_a_quality_first_appeared(self):
        store = ObservationStore()
        self.assertIsNone(self.capture(store, 1, confidence=0.9)[0]["quality_first_round"])
        self.assertEqual(self.capture(store, 2, quality="金", confidence=0.9)[0]["quality_first_round"], 2)
        # a later read that misses the quality keeps the date
        missed = self.capture(store, 3, confidence=0.95)[0]
        self.assertEqual((missed["quality"], missed["quality_first_round"]), (None, 2))
        again = self.capture(store, 4, quality="金", confidence=0.97)[0]
        self.assertEqual((again["first_seen_round"], again["quality_first_round"]), (1, 2))

    def test_human_entries_are_dated_to_the_current_round(self):
        store = ObservationStore()
        store.set_round(3)
        store.upsert_map_item(MapObservation(row=0, column=0, quality="紫"))
        self.assertEqual(store.snapshot().map_items[0]["quality_first_round"], 3)


class CountingOCR:
    def __init__(self):
        self.calls = 0

    def recognize(self, image, *, expected_round):
        self.calls += 1
        return OcrCapture(
            round_number=expected_round, round_verified=True,
            events=(EventObservation(expected_round, "public", "x", 0.99,
                                     semantics("total_occupied_cells", observed_cells=20)),),
            bankroll=None, timing_ms=1.0,
        )


class ScrollSource:
    def __init__(self, frames):
        self.frames = list(frames)

    def capture(self):
        return self.frames.pop(0).copy()

    def swipe_map(self, start, end, duration_ms):
        pass


class Vision:
    def recognize_viewports(self, frames):
        return ViewportRecognition(items=(), offsets_pixels=tuple(0 for _ in frames),
                                   estimated_rows=8, alignment_stable=True)


class TestablePipeline(CapturePipeline):
    def _round_frame(self):
        return self.source.capture()


class EarlyStartTests(unittest.TestCase):
    def frames(self):
        first = np.full((720, 1280, 3), 80, dtype=np.uint8)
        second = first.copy()
        second[110:705, 49:647] = 180
        return [first, first, second, second, first, first]

    def test_round_events_reach_the_callback_before_scrolling_and_ocr_runs_once(self):
        calls = []
        ocr = CountingOCR()
        store = ObservationStore()
        source = ScrollSource(self.frames())
        pipeline = TestablePipeline(
            source=source, store=store, ocr=ocr, vision=Vision(), scroll_wait=0.05,
            on_round_events=lambda round_number, capture: calls.append((round_number, len(source.frames))),
        )
        outcome = pipeline.scan_round(expected_round=1)
        self.assertTrue(outcome.success, outcome.errors)
        self.assertEqual(ocr.calls, 1)
        # called after the top frame, while the downward frames were still unread
        self.assertEqual(calls, [(1, 4)])
        self.assertEqual(len(store.snapshot().events), 1)

    def test_without_a_callback_the_scan_is_unchanged(self):
        ocr = CountingOCR()
        store = ObservationStore()
        pipeline = TestablePipeline(source=ScrollSource(self.frames()), store=store, ocr=ocr,
                                    vision=Vision(), scroll_wait=0.05)
        self.assertTrue(pipeline.scan_round(expected_round=1).success)
        self.assertEqual(ocr.calls, 1)
        self.assertEqual(len(store.snapshot().events), 1)

    def test_coordinator_prewarms_only_predictors_that_support_it(self):
        store = ObservationStore()
        capture = CountingOCR().recognize(None, expected_round=1)

        class Plain:
            def predict(self, snapshot):
                raise AssertionError("not called")

        class Warm(Plain):
            def __init__(self):
                self.seen, self.closed = [], False

            def prewarm(self, snapshot):
                self.seen.append(snapshot)
                return True

            def close(self):
                self.closed = True

        self.assertFalse(InferenceCoordinator(Plain()).prewarm(store, 1, capture))
        warm = Warm()
        coordinator = InferenceCoordinator(warm)
        self.assertTrue(coordinator.prewarm(store, 1, capture))
        self.assertEqual(warm.seen[0].round_number, 1)
        coordinator.close()
        self.assertTrue(warm.closed)


def estimate(revision, status):
    return PredictionResult(revision=revision, status=status, compatible_worlds=1, p10=1, p50=2, p90=3,
                            minimum=1, maximum=3, actionable=False, issues=())


class RefinementCoordinatorTests(unittest.TestCase):
    class Predictor:
        def __init__(self):
            self.refined, self.poll = [], ("running", None)

        def predict(self, snapshot):
            raise AssertionError("not called")

        def refine(self, revision):
            self.refined.append(revision)
            return True

        def poll_refined(self):
            return self.poll

    def test_only_a_current_estimate_cut_by_the_budget_is_refined(self):
        store = ObservationStore()
        predictor = self.Predictor()
        coordinator = InferenceCoordinator(predictor)
        self.assertFalse(coordinator.refine(store, estimate(store.revision, "exact_v2_posterior_estimate")))
        self.assertFalse(coordinator.refine(store, estimate(store.revision + 1, "exact_v2_posterior_low_ess")))
        self.assertTrue(coordinator.refine(store, estimate(store.revision, "exact_v2_posterior_low_ess")))
        self.assertEqual(predictor.refined, [store.revision])
        self.assertEqual(coordinator.refined(store), ("running", None))
        predictor.poll = ("done", estimate(store.revision, "exact_v2_posterior_estimate"))
        self.assertEqual(coordinator.refined(store)[1].status, "exact_v2_posterior_estimate")
        store.set_round(2)  # the evidence moved on: the refined estimate is stale
        self.assertEqual(coordinator.refined(store), ("gone", None))
        coordinator.close()

    def test_window_shows_the_refined_estimate_when_it_arrives(self):
        from concurrent.futures import Future

        from auction_moment_assistant.ui import AssistantWindow

        class Root:
            def __init__(self):
                self.callbacks = []

            def after(self, delay, callback):
                self.callbacks.append(callback)

        store = ObservationStore()
        predictor = self.Predictor()
        window = object.__new__(AssistantWindow)
        window.root, window.store, window.inference = Root(), store, InferenceCoordinator(predictor)
        shown, logged = [], []
        window.prediction_var = types.SimpleNamespace(set=shown.append)
        window._append_log = lambda category, message, details=(): logged.append(category)
        cut = dataclasses.replace(estimate(store.revision, "exact_v2_posterior_low_ess"), point_estimate=2)
        window._prediction_future = Future()
        window._prediction_future.set_result(cut)
        window._poll_prediction()
        self.assertEqual((len(shown), predictor.refined), (1, [store.revision]))
        window.root.callbacks.pop()()  # still running: polls again
        self.assertEqual(len(window.root.callbacks), 1)
        predictor.poll = ("done", dataclasses.replace(cut, status="exact_v2_posterior_estimate"))
        window.root.callbacks.pop()()
        self.assertEqual(len(shown), 2)
        self.assertIn("prediction-refined", logged)
        self.assertEqual(window.root.callbacks, [])
        window.inference.close()


@unittest.skipUnless(HAS_SCIPY, "exact sampler needs scipy")
class ExactEvidenceTests(unittest.TestCase):
    def convert(self, snap, **kwargs):
        from auction_moment_assistant.exact_evidence import evidence_from_snapshot

        return evidence_from_snapshot(snap, CATALOG, **kwargs)

    def test_numeric_clues(self):
        result = self.convert(snapshot([
            event(1, "public", "quality_total_value", quality="金", observed_value=926),
            event(1, "personal", "size_average_value", width=2, height=2, observed_value=334),
            event(2, "public", "quality_average_area", quality="紫", observed_average_cells=2.35),
            event(2, "personal", "highest_quality", observed_quality="彩"),
            event(3, "public", "max_item_value", observed_value=1500),
            event(3, "personal", "total_occupied_cells", observed_cells=14),
        ], round_number=3))
        self.assertTrue(result.usable, result.unsupported)
        ev = result.evidence
        self.assertEqual(ev.total_value, {3: 926})
        self.assertEqual(ev.size_avg_value, {4: 334})
        self.assertEqual(ev.avg_area, {2: (235, 100)})
        self.assertEqual((ev.highest, ev.max_value, ev.total_cells, ev.map_rows), (4, 1500, 14, 8))

    def test_count_reveals_need_a_complete_trusted_scan(self):
        events = [event(1, "public", "reveal_position_by_quality", quality="金"),
                  event(1, "personal", "reveal_outline_by_size", width=2, height=2)]
        items = [item(0, 0, quality="金"), item(0, 3, quality="金", width=2, height=2, spatial="outline"),
                 item(3, 0, quality="紫", width=2, height=2, spatial="outline")]
        complete = self.convert(snapshot(events, items))
        self.assertEqual(complete.evidence.count, {3: 2})
        self.assertEqual(complete.evidence.size_count, {4: 2})
        partial = self.convert(snapshot(events, items, complete=False))
        self.assertEqual((partial.evidence.count, partial.evidence.size_count), ({}, {}))
        self.assertTrue(any(r.startswith("count_reveal_without_trusted_scan") for r in partial.ignored))

    def test_prewarm_leaves_out_the_unscanned_round(self):
        events = [event(1, "public", "reveal_position_by_quality", quality="金"),
                  event(1, "personal", "max_item_value", observed_value=1500),
                  event(2, "public", "reveal_position_by_quality", quality="紫"),
                  event(2, "personal", "highest_quality", observed_quality="彩")]
        items = [item(0, 0, quality="金")]
        result = self.convert(snapshot(events, items, round_number=2), prewarm=True)
        self.assertTrue(result.usable)
        self.assertEqual(result.evidence.count, {3: 1})
        self.assertEqual(result.evidence.highest, 4)

    def test_unsupported_or_conflicting_clues_block_the_mode(self):
        unsupported = self.convert(snapshot([event(1, "public", "total_item_count", observed_count=9),
                                             event(1, "personal", "highest_quality", observed_quality="金")]))
        self.assertFalse(unsupported.usable)
        self.assertTrue(any("total_item_count" in r for r in unsupported.unsupported))
        conflict = self.convert(snapshot(
            [event(1, "public", "quality_item_count", quality="金", observed_count=3),
             event(1, "personal", "reveal_position_by_quality", quality="金")],
            [item(0, 0, quality="金"), item(0, 2, quality="金")]))
        self.assertFalse(conflict.usable)
        self.assertTrue(any(r.startswith("conflicting_clues") for r in conflict.unsupported))

    def test_random_quality_reveals_are_dated_by_when_the_quality_appeared(self):
        events = [event(1, "public", "reveal_identity_random", count=1),
                  event(1, "personal", "reveal_position_quality_random", count=1),
                  event(2, "public", "reveal_position_quality_random", count=2),
                  event(2, "personal", "reveal_position_by_quality", quality="紫"),
                  event(3, "public", "reveal_position_by_quality", quality="白"),
                  event(3, "personal", "max_item_value", observed_value=1500)]
        items = [item(0, 0, catalog_id="g1", spatial="complete", first_seen_round=1),
                 item(0, 5, quality="蓝", first_seen_round=1),
                 # seen bare in round 1, its quality drawn in round 2
                 item(2, 0, quality="金", first_seen_round=1, quality_round=2),
                 item(4, 0, quality="紫", first_seen_round=2),
                 # seen bare in a round with a random quality reveal, its
                 # quality shown later by another event: not a draw
                 item(6, 0, quality="白", first_seen_round=1, quality_round=3)]
        result = self.convert(snapshot(events, items, round_number=3))
        self.assertEqual(result.evidence.revealed_identities, Counter({3: 1}))
        self.assertEqual(result.evidence.revealed_qualities, Counter({1: 1, 3: 1}))
        self.assertIn("quality_reveal_unattributed:R2", result.ignored)

    def test_more_new_qualities_than_draws_are_left_out(self):
        events = [event(1, "public", "reveal_position_quality_random", count=1),
                  event(1, "personal", "max_item_value", observed_value=1500)]
        items = [item(0, 0, quality="蓝"), item(0, 5, quality="金")]
        result = self.convert(snapshot(events, items))
        self.assertEqual(result.evidence.revealed_qualities, Counter())
        self.assertIn("quality_reveal_overflow:R1", result.ignored)
        undated = dict(item(0, 0, quality="蓝"), quality_first_round=None)
        result = self.convert(snapshot(events, [undated]))
        self.assertEqual(result.evidence.revealed_qualities, Counter())
        self.assertIn("quality_reveal_undated", result.ignored)

    def test_largest_outline_is_the_largest_known_size_on_a_complete_scan(self):
        events = [event(1, "public", "reveal_largest_area_item_outline"),
                  event(1, "personal", "reveal_identity_random", count=1)]
        items = [item(0, 0, width=2, height=2, spatial="outline"),
                 item(0, 3, catalog_id="g1", spatial="complete"),
                 item(3, 0, quality="蓝", width=2, height=1, spatial="outline", first_seen_round=2)]
        result = self.convert(snapshot(events, items, round_number=2))
        self.assertTrue(result.usable)
        self.assertEqual(result.evidence.largest_area, 4)
        partial = self.convert(snapshot(events, items, round_number=2, complete=False))
        self.assertIsNone(partial.evidence.largest_area)
        self.assertTrue(any(r.startswith("largest_outline_without_trusted_scan") for r in partial.ignored))
        # the outline is not read until the round's scan
        pending = self.convert(snapshot(events, items[1:2], round_number=1), prewarm=True)
        self.assertIsNone(pending.evidence.largest_area)
        self.assertFalse(any("largest" in r for r in pending.ignored))

    def test_largest_outline_needs_every_new_bare_anchor_explained(self):
        events = [event(1, "public", "reveal_largest_area_item_outline"),
                  event(1, "personal", "max_item_value", observed_value=1500)]
        # the largest item read as a bare anchor: its outline was missed
        items = [item(0, 0), item(0, 3, width=1, height=2, spatial="outline")]
        missed = self.convert(snapshot(events, items))
        self.assertIsNone(missed.evidence.largest_area)
        self.assertTrue(any(r.startswith("largest_outline_not_read") for r in missed.ignored))
        # a position-only reveal in the same round explains one bare anchor
        explained = [events[0], event(1, "personal", "reveal_position_random", count=1)]
        self.assertEqual(self.convert(snapshot(explained, items)).evidence.largest_area, 2)

    def test_position_only_and_unreadable_clues_are_ignored_not_relaxed(self):
        result = self.convert(snapshot([event(1, "public", "reveal_position_random", count=2),
                                        event(1, "personal", "reveal_largest_area_item_outline")],
                                       exact=False))
        self.assertTrue(result.usable)
        self.assertIsNone(result.evidence.map_rows)
        kinds = {r.split(":")[0] for r in result.ignored}
        self.assertEqual(kinds, {"map_rows_not_exact", "position_only_reveal", "largest_outline_not_read"})


# A toy world-model-v2-shaped prior small enough to enumerate every world.
TOY = [(0, 1, 1, 10), (0, 1, 2, 17), (1, 1, 1, 31), (1, 2, 1, 45), (2, 1, 1, 103), (2, 2, 2, 151),
       (2, 1, 2, 122), (3, 1, 2, 409), (3, 1, 1, 333), (3, 2, 2, 517), (4, 1, 1, 1500), (4, 2, 1, 2203)]
TOY_QUOTAS = (1, 0, 1, 1, 0)
TOY_SIZES = [(w, h) for w in range(1, 4) for h in range(1, 4)]


def toy_prior():
    from auction_moment_assistant.exact_sampler import Prior

    quality = np.array([c[0] for c in TOY])
    values = np.array([c[3] for c in TOY])

    def colored_p(n):
        rest = n - sum(TOY_QUOTAS)
        return [math.comb(rest, c) * 0.3 ** c * 0.7 ** (rest - c) for c in range(rest + 1)]

    def noncolored_p(n):
        base = np.array([0.3, 0.25, 0.25, 0.2]) * np.array([1, 1 + 0.1 * (n - 4), 1, 1 - 0.05 * (n - 4)])
        return base / base.sum()

    def identity_p(q, n):
        index = np.flatnonzero(quality == q)
        if q != 4:
            return np.full(len(index), 1 / len(index))
        weights = values[index].astype(float) ** (-0.3 - 0.2 * (n - 5))
        return weights / weights.sum()

    return Prior(values=values, areas=np.array([c[1] * c[2] for c in TOY]), quality=quality,
                 size=np.array([TOY_SIZES.index((c[1], c[2])) for c in TOY]),
                 count_p={4: 0.2, 5: 0.45, 6: 0.35}, quotas=TOY_QUOTAS, colored_p=colored_p,
                 noncolored_p=noncolored_p, identity_p=identity_p)


def compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def multinomial(counts, probs):
    return math.exp(math.lgamma(sum(counts) + 1) - sum(math.lgamma(c + 1) for c in counts)
                    + sum(c * math.log(p) for c, p in zip(counts, probs) if c))


def toy_worlds(prior):
    worlds = []
    for n_total, p_n in prior.count_p.items():
        rest = n_total - sum(TOY_QUOTAS)
        p_colored, p_other = prior.colored_p(n_total), prior.noncolored_p(n_total)
        for colored in range(rest + 1):
            for extras in compositions(rest - colored, 4):
                n = [TOY_QUOTAS[q] + (extras[q] if q < 4 else 0) for q in range(5)]
                n[4] = colored
                base = p_n * p_colored[colored] * multinomial(extras, p_other)
                per_quality = []
                for q in range(5):
                    index = prior.by_quality[q]
                    pi = prior.identity_p(q, n_total)
                    per_quality.append([(c, multinomial(c, pi)) for c in compositions(n[q], len(index))])
                for combo in itertools.product(*per_quality):
                    counts = np.zeros(len(TOY), np.int64)
                    p = base
                    for q, (c, pq) in enumerate(combo):
                        counts[prior.by_quality[q]] = c
                        p *= pq
                    worlds.append((counts, p))
    return worlds


@unittest.skipUnless(HAS_SCIPY, "exact sampler needs scipy")
class ExactSamplerToyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = toy_prior()
        cls.worlds = toy_worlds(cls.prior)
        cls.totals = np.array([float(m @ cls.prior.values) for m, _ in cls.worlds])

    def check(self, ev, accept):
        from auction_moment_assistant.exact_sampler import sample_posterior

        post = np.array([p if accept(m) else 0.0 for m, p in self.worlds])
        self.assertGreater(post.sum(), 0)
        exact = float(post @ self.totals / post.sum())
        result = sample_posterior(self.prior, ev, np.random.default_rng(7), pool_size=1500, min_ess=500)
        weights = result.weights / result.weights.sum()
        self.assertLess(abs(float(weights @ result.totals) / exact - 1), 0.025)

    def test_size_average_value(self):
        from auction_moment_assistant.exact_sampler import Evidence

        size = self.prior.size
        values = self.prior.values

        def accept(m):
            count = int(m[size == 0].sum())
            return count > 0 and int(m[size == 0] @ values[size == 0]) // count == 333

        self.check(Evidence(size_avg_value={0: 333}), accept)

    def test_total_value_with_count(self):
        from auction_moment_assistant.exact_sampler import Evidence

        quality, values = self.prior.quality, self.prior.values

        def accept(m):
            return int(m[quality == 3] @ values[quality == 3]) == 926 and int(m[quality == 0].sum()) == 1

        self.check(Evidence(total_value={3: 926}, count={0: 1}), accept)

    def test_cancel_stops_a_run(self):
        from auction_moment_assistant.exact_sampler import Cancelled, Evidence, sample_posterior

        with self.assertRaises(Cancelled):
            sample_posterior(self.prior, Evidence(), np.random.default_rng(0), cancel=lambda: True)


@unittest.skipUnless(HAS_SCIPY, "exact sampler needs scipy")
class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from auction_moment_assistant.exact_sampler import Evidence, Result, sample_posterior

        cls.lane_result = sample_posterior(toy_prior(), Evidence(), np.random.default_rng(3), pool_size=500)
        totals = np.array([100.0, 120.0, 150.0])
        cls.converged = Result("ok", totals, np.ones(3), {})
        cls.short = Result("low_ess", totals, np.ones(3), {})

    def lanes(self, count=2):
        """ExactLanes over in-process fake workers that hold a refinement until released."""
        from auction_moment_assistant.exact_lanes import ExactLanes

        lanes = ExactLanes.__new__(ExactLanes)
        lanes._conns, lanes._procs, lanes._cancel = [], [], []
        lanes._lock = threading.Lock()
        lanes._background, lanes._replies, lanes._refine_target = None, [], 0.0
        lanes.last_prewarm_wait = 0.0
        self.release = threading.Event()
        for _ in range(count):
            parent, child = multiprocessing.Pipe()
            cancel = threading.Event()
            threading.Thread(target=self.fake_worker, args=(child, cancel), daemon=True).start()
            lanes._conns.append(parent)
            lanes._cancel.append(cancel)
        self.addCleanup(lanes.close)
        return lanes

    def fake_worker(self, conn, cancel):
        while (message := conn.recv()) is not None:
            if message[0] == "refine":
                while not (cancel.is_set() or self.release.is_set()):
                    time.sleep(0.002)
                conn.send(("cancelled", None) if cancel.is_set() else ("ok", self.lane_result))
            else:
                conn.send(("ok", self.lane_result))

    def test_refinement_runs_in_the_background_until_collected(self):
        from auction_moment_assistant.exact_sampler import Evidence

        lanes = self.lanes()
        self.assertTrue(lanes.refine(Evidence(), 1, time_budget=5))
        self.assertEqual(lanes.poll_refine(), ("running", None))
        self.release.set()
        deadline = time.monotonic() + 5
        state, merged = "running", None
        while state == "running" and time.monotonic() < deadline:
            state, merged = lanes.poll_refine()
        self.assertEqual(state, "done")
        self.assertEqual(merged.diagnostics["lanes"], 2)
        self.assertEqual(lanes.poll_refine(), ("gone", "not_refining"))

    def test_a_new_estimate_cancels_the_refinement(self):
        from auction_moment_assistant.exact_sampler import Evidence

        lanes = self.lanes()
        self.assertTrue(lanes.refine(Evidence(), 1))
        merged = lanes.sample(Evidence(), 2, time_budget=None)
        self.assertEqual(merged.diagnostics["lanes"], 2)
        self.assertEqual(lanes.poll_refine(), ("gone", "not_refining"))

    def predictor(self, poll):
        from auction_moment_assistant.exact_evidence import ExactEvidence
        from auction_moment_assistant.exact_sampler import Evidence

        engine = types.SimpleNamespace(started=[], poll=poll)
        engine.refine = lambda ev, seed, *, time_budget: engine.started.append(time_budget) or True
        engine.poll_refine = lambda: engine.poll
        predictor = LatestV6V2Predictor.__new__(LatestV6V2Predictor)
        predictor.exact_engine, predictor.exact_refine_seconds, predictor.bid_safety_factor = engine, 30.0, 0.9
        predictor._exact_cut = {
            "snapshot": snapshot([], round_number=2), "adapted": types.SimpleNamespace(warnings=()),
            "v6": (90, 100, 120), "converted": ExactEvidence(Evidence(), ("map_rows_not_exact",), ()),
            "diagnostics": {"exact_seconds": 8.0}, "point": 95,
        }
        return predictor, engine

    def test_predictor_replaces_a_cut_estimate_only_once_the_rerun_converges(self):
        predictor, engine = self.predictor(("running", None))
        self.assertFalse(predictor.refine(2))  # not the cut decision's revision
        self.assertTrue(predictor.refine(1))
        self.assertEqual(engine.started, [30.0])
        self.assertEqual(predictor.poll_refined(), ("running", None))
        engine.poll = ("done", self.converged)
        state, refined = predictor.poll_refined()
        self.assertEqual((state, refined.status), ("done", "exact_v2_posterior_estimate"))
        self.assertTrue(refined.conditional_diagnostics["exact_refined"])
        self.assertEqual(refined.conditional_diagnostics["exact_cut_point"], 95)
        self.assertEqual(predictor.poll_refined(), ("gone", None))
        predictor, engine = self.predictor(("done", self.short))
        self.assertTrue(predictor.refine(1))
        self.assertEqual(predictor.poll_refined(), ("done", None))
        predictor, _ = self.predictor(("running", None))
        predictor.exact_refine_seconds = 0.0
        self.assertFalse(predictor.refine(1))


if __name__ == "__main__":
    unittest.main()
