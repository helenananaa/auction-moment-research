"""Likelihoods for random reveal events with attributable, complete samples.

``first_seen_round`` tracks the first *position* observation. It is not an
attribute-reveal history. Consequently, current visible items must never all be
treated as independent random draws. This module only extracts a first reveal
whose full sample can be uniquely attributed, before any observed map items.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from .models import CatalogItem
from .observations import ObservationSnapshot


RANDOM_ATTRIBUTE_EFFECTS = frozenset(
    {"reveal_position_quality_random", "reveal_identity_random"}
)
RANDOM_REVEAL_EFFECTS = RANDOM_ATTRIBUTE_EFFECTS | {"reveal_position_random"}
MAP_REVEAL_EFFECTS = RANDOM_REVEAL_EFFECTS | {
    "reveal_outline_by_size",
    "reveal_largest_area_item_outline",
    "reveal_outline_by_quality",
    "reveal_position_by_quality",
}
QUALITY_NAMES = frozenset({"白", "蓝", "紫", "金", "彩"})


@dataclass(frozen=True)
class RevealObservation:
    round_number: int
    event_kind: str
    effect: str
    sample_counts: tuple[tuple[str, int], ...]
    sample_size: int


def _positive_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
        if parsed <= 0 or parsed != float(value):
            return None
        return parsed
    except (TypeError, ValueError, OverflowError):
        return None


def _trusted(record: dict) -> bool:
    if record.get("human_locked"):
        return True
    try:
        confidence = float(record.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(confidence) and confidence >= 0.85


def build_reveal_observations(
    snapshot: ObservationSnapshot,
    catalog: Sequence[CatalogItem],
) -> tuple[tuple[RevealObservation, ...], tuple[str, ...]]:
    """Extract at most one sample; return explicit reasons for skipped evidence.

    The likelihood assumes the named random event samples items uniformly
    without replacement. To avoid guessing whether later events sample the full
    map or only unrevealed items, only the earliest random reveal event can be
    used, and it must precede every other observed map item. Both event slots
    must be parsed, reliable, and contain exactly one map-reveal source.
    """
    diagnostics: list[str] = []
    events_by_round: dict[int, list[dict]] = defaultdict(list)
    for event in snapshot.events:
        round_number = _positive_integer(event.get("round_number"))
        if round_number is not None and round_number <= snapshot.round_number:
            events_by_round[round_number].append(event)
    candidate_rounds = sorted(
        round_number
        for round_number, events in events_by_round.items()
        if any(
            (event.get("semantics") or {}).get("effect") in RANDOM_REVEAL_EFFECTS
            for event in events
        )
    )
    if not candidate_rounds:
        return (), ("reveal_likelihood:no_random_reveal_event",)
    round_number = candidate_rounds[0]
    if len(candidate_rounds) > 1:
        diagnostics.append("reveal_likelihood:later_rounds_skipped_unknown_reveal_pool")

    def unavailable(reason: str):
        return (), tuple([*diagnostics, f"reveal_likelihood:{reason}"])

    if not snapshot.completeness_confirmed:
        return unavailable("map_completeness_not_confirmed")
    events = events_by_round[round_number]
    # ObservationStore has one event per (round, kind). Exact duplicate records
    # from external adapters are harmless; conflicting records fail closed.
    unique_events: dict[str, dict] = {}
    for event in events:
        kind = str(event.get("kind") or "")
        previous = unique_events.get(kind)
        if previous is not None:
            if previous != event:
                return unavailable("conflicting_event_records")
            continue
        unique_events[kind] = event
    if set(unique_events) != {"public", "personal"}:
        return unavailable("event_ledger_incomplete")
    if any(
        not _trusted(event) or not (event.get("semantics") or {}).get("parsed")
        for event in unique_events.values()
    ):
        return unavailable("event_attributes_untrusted")
    reveal_events = [
        event
        for event in unique_events.values()
        if (event.get("semantics") or {}).get("effect") in MAP_REVEAL_EFFECTS
    ]
    if len(reveal_events) != 1:
        return unavailable("multiple_reveal_sources")
    event = reveal_events[0]
    semantics = event["semantics"]
    if semantics.get("reveal_outcome_available") is False:
        return unavailable("reveal_outcome_unavailable")
    effect = str(semantics["effect"])
    # count is the actual random-event schema. observed_count belongs to
    # population summary events and must not silently substitute for it.
    sample_size = _positive_integer(semantics.get("count"))
    if sample_size is None:
        return unavailable("random_sample_count_missing")
    if "observed_count" in semantics and semantics["observed_count"] != sample_size:
        return unavailable("random_sample_count_conflict")

    items_by_position: dict[tuple[object, object], dict] = {}
    for item in snapshot.map_items:
        first_seen = _positive_integer(item.get("first_seen_round"))
        if first_seen is None or first_seen > snapshot.round_number:
            return unavailable("first_seen_history_missing")
        if first_seen < round_number:
            return unavailable("earlier_visible_items_unknown_reveal_pool")
        if first_seen != round_number:
            continue
        if item.get("row") is None or item.get("column") is None:
            return unavailable("sample_position_missing")
        try:
            row, column = int(item["row"]), int(item["column"])
            if (
                isinstance(item["row"], bool)
                or isinstance(item["column"], bool)
                or row != float(item["row"])
                or column != float(item["column"])
                or row < 0
                or not 0 <= column < 10
            ):
                return unavailable("sample_position_invalid")
        except (TypeError, ValueError, OverflowError):
            return unavailable("sample_position_invalid")
        key = (row, column)
        previous = items_by_position.get(key)
        if previous is not None and previous != item:
            return unavailable("conflicting_map_records")
        items_by_position[key] = item
    if len(items_by_position) != sample_size:
        return unavailable("first_seen_sample_count_mismatch")

    # A scan may first encounter an old revealed object in a later round. The
    # absence of earlier first_seen items alone therefore does not establish
    # that the current event sampled a completely hidden map.
    for earlier_round in range(1, round_number):
        earlier_events = events_by_round.get(earlier_round, [])
        if {str(value.get("kind")) for value in earlier_events} != {
            "public",
            "personal",
        }:
            return unavailable("earlier_event_history_incomplete")
        for earlier_event in earlier_events:
            earlier_semantics = earlier_event.get("semantics") or {}
            if not _trusted(earlier_event) or not earlier_semantics.get("parsed"):
                return unavailable("earlier_event_history_untrusted")
            if earlier_semantics.get("effect") in MAP_REVEAL_EFFECTS:
                return unavailable("earlier_reveal_unknown_sampling_pool")

    catalog_by_id = {item.catalog_id: item for item in catalog}
    sample_counts: Counter[str] = Counter()
    for item in items_by_position.values():
        if not _trusted(item):
            return unavailable("sample_attributes_untrusted")
        if effect == "reveal_position_random":
            # Later attribute upgrades must not turn a position-only draw
            # into a random quality or identity sample.
            continue
        catalog_id = str(item.get("catalog_id") or "")
        quality = str(item.get("quality") or "")
        identity = catalog_by_id.get(catalog_id)
        if identity is not None and quality and quality != identity.quality:
            return unavailable("identity_quality_conflict")
        if effect == "reveal_identity_random":
            if identity is None or item.get("spatial") != "complete":
                return unavailable("sample_identity_incomplete")
            category = identity.catalog_id
        else:
            if quality not in QUALITY_NAMES:
                return unavailable("sample_quality_incomplete")
            category = quality
        sample_counts[category] += 1

    diagnostics.extend(
        (
            "reveal_likelihood:uniform_without_replacement_assumption",
            "reveal_likelihood:first_seen_attribution_single_reveal_source",
            f"reveal_likelihood:accepted:{round_number}:{event['kind']}:{sample_size}",
        )
    )
    observation = RevealObservation(
        round_number=round_number,
        event_kind=str(event["kind"]),
        effect=effect,
        sample_counts=tuple(sorted(sample_counts.items())),
        sample_size=sample_size,
    )
    return (observation,), tuple(diagnostics)


def _log_combination(population: int, selected: int) -> float:
    if selected < 0 or selected > population:
        return -math.inf
    # Integer comb avoids cancellation in lgamma(n+1)-lgamma(n-k+1).
    return math.log(math.comb(population, selected))


def reveal_log_likelihood(
    counts: Sequence[int],
    observations: Sequence[RevealObservation],
    catalog: Sequence[CatalogItem],
    *,
    conditioned_on_positions: bool = False,
) -> float:
    """Return the random reveal sample log likelihood.

    ``counts`` and ``catalog`` must have identical indexing. Multiple rounds are
    rejected because the current schema does not establish independent pools.
    Exact repeated copies of a single event are deduplicated.

    By default only the unordered category counts are observed: the likelihood
    is multivariate hypergeometric. If ``conditioned_on_positions`` is true,
    the caller must already have checked the sample's concrete anchors and
    attributes against a particular generated layout. Selecting that exact set
    of items then has probability 1/C(N,n). Multiplying the hypergeometric
    numerator again would count the same category evidence a second time.
    A position-only event has no category information: its bag likelihood is
    one; its exact-position selection likelihood remains 1/C(N,n).
    """
    if len(counts) != len(catalog):
        raise ValueError("counts and catalog must have the same length")
    population: list[int] = []
    for value in counts:
        if isinstance(value, bool):
            raise ValueError("population counts must be nonnegative integers")
        try:
            parsed = int(value)
            valid = math.isfinite(float(value)) and parsed == value and parsed >= 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("population counts must be nonnegative integers")
        population.append(parsed)
    unique = tuple(dict.fromkeys(observations))
    if not unique:
        return 0.0
    if len(unique) != 1:
        raise ValueError("independence of multiple reveal events is not established")
    observation = unique[0]
    if observation.effect not in RANDOM_REVEAL_EFFECTS:
        raise ValueError("only random reveal events have a likelihood")
    sample_counts = {
        category: _positive_integer(selected)
        for category, selected in observation.sample_counts
    }
    sample_size = _positive_integer(observation.sample_size)
    if (
        len(sample_counts) != len(observation.sample_counts)
        or any(value is None for value in sample_counts.values())
        or sample_size is None
    ):
        raise ValueError(
            "sample counts must be positive integers summing to sample size"
        )
    total = sum(population)
    if observation.effect == "reveal_position_random":
        if sample_counts:
            raise ValueError("position-only events cannot carry category counts")
        if total < sample_size:
            return -math.inf
        return (
            -_log_combination(total, sample_size) if conditioned_on_positions else 0.0
        )
    if sum(sample_counts.values()) != sample_size:
        raise ValueError(
            "sample counts must be positive integers summing to sample size"
        )
    if total < sample_size:
        return -math.inf
    category_counts: Counter[str] = Counter()
    for value, item in zip(population, catalog, strict=True):
        category = (
            item.catalog_id
            if observation.effect == "reveal_identity_random"
            else item.quality
        )
        category_counts[category] += value
    terms = [
        _log_combination(category_counts[category], selected)
        for category, selected in sample_counts.items()
    ]
    if any(not math.isfinite(term) for term in terms):
        return -math.inf
    if conditioned_on_positions:
        return -_log_combination(total, sample_size)
    return sum(terms) - _log_combination(total, sample_size)
