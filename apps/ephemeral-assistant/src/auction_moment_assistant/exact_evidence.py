"""Decision-time evidence and prior for the experimental exact-constraint sampler.

``evidence_from_snapshot`` translates an ObservationSnapshot into
``exact_sampler.Evidence`` with the semantics the sampler was validated on
(structured transcriptions of 277 live decisions). Clues that only narrow
positions, or whose outcome cannot be attributed, are listed as ignored and
leave the posterior valid but less informed. Clues the sampler cannot
represent, and conflicting clues, are listed as unsupported; callers must
then not use the sampler, so no condition is ever silently relaxed.

The sampler itself needs scipy; this module imports it only when called.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from .models import CatalogItem
from .observations import ObservationSnapshot


QUALITY_ORDER = ("白", "蓝", "紫", "金", "彩")
SIZES = tuple((width, height) for width in range(1, 4) for height in range(1, 4))
RANDOM_IDENTITY = "reveal_identity_random"
RANDOM_QUALITY = "reveal_position_quality_random"
RANDOM_POSITION = "reveal_position_random"
COUNT_BY_QUALITY = frozenset({"reveal_position_by_quality", "reveal_outline_by_quality"})
COUNT_BY_SIZE = "reveal_outline_by_size"
LARGEST_OUTLINE = "reveal_largest_area_item_outline"
# exact-rows-table-v1: counts of map rows minus ceil(occupied cells / 10), by
# occupied-cell band (cells // 40, capped at 7), from clean public final maps
# (freeze_rows_table.py in the 2026-10-20 experiments). Kept in code so the
# archived package manifest stays unchanged.
ROWS_TABLE_V1 = {
    "band_cells": 40,
    "max_band": 7,
    "bands": {
        "0": {"0": 33, "1": 89, "2": 16},
        "1": {"0": 98, "1": 208, "2": 44},
        "2": {"0": 101, "1": 233, "2": 46, "3": 2},
        "3": {"0": 62, "1": 191, "2": 35},
        "4": {"0": 29, "1": 66, "2": 13},
        "5": {"0": 4, "1": 12, "2": 2},
    },
}


@dataclass(frozen=True)
class ExactEvidence:
    evidence: object | None
    ignored: tuple[str, ...]
    unsupported: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.evidence is not None and not self.unsupported


def _trusted(record: dict, minimum_confidence: float) -> bool:
    if record.get("human_locked"):
        return True
    try:
        confidence = float(record.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(confidence) and confidence >= minimum_confidence


def evidence_from_snapshot(
    snapshot: ObservationSnapshot,
    catalog: Sequence[CatalogItem],
    *,
    minimum_confidence: float = 0.85,
    prewarm: bool = False,
) -> ExactEvidence:
    """Exact sampler evidence for a snapshot.

    With ``prewarm`` the current round's map has not been scanned yet: its
    map-reveal outcomes are left out, so the result is the same evidence
    minus what the scan will add (used only to fill caches early).
    """
    from .exact_sampler import Evidence

    ignored: list[str] = []
    unsupported: list[str] = []
    quality_index = {quality: index for index, quality in enumerate(QUALITY_ORDER)}
    size_index = {size: index for index, size in enumerate(SIZES)}
    catalog_index = {item.catalog_id: index for index, item in enumerate(catalog)}
    current = int(snapshot.round_number)
    ev = Evidence()
    if snapshot.map_rows is not None and snapshot.map_height_exact:
        ev.map_rows = int(snapshot.map_rows)
    else:
        ignored.append("map_rows_not_exact")
    items = [dict(raw) for raw in snapshot.map_items]
    map_trusted = bool(snapshot.completeness_confirmed) and all(
        _trusted(item, minimum_confidence) for item in items
    )
    by_id = {item.catalog_id: item for item in catalog}

    def identity(item: dict):
        if str(item.get("spatial") or "") != "complete":
            return None
        return by_id.get(str(item.get("catalog_id") or ""))

    def item_quality(item: dict):
        known = identity(item)
        return known.quality if known is not None else item.get("quality")

    def item_size(item: dict):
        known = identity(item)
        if known is not None:
            return known.width, known.height
        if str(item.get("spatial") or "") == "outline" and item.get("width") and item.get("height"):
            return int(item["width"]), int(item["height"])
        return None

    def put(target: dict, key, value, label: str) -> None:
        if key in target and target[key] != value:
            unsupported.append(f"conflicting_clues:{label}")
        target[key] = value

    def put_scalar(name: str, value, label: str) -> None:
        previous = getattr(ev, name)
        if previous is not None and previous != value:
            unsupported.append(f"conflicting_clues:{label}")
        setattr(ev, name, value)

    def size_of(semantics: dict) -> tuple[int, int]:
        size = (int(semantics["width"]), int(semantics["height"]))
        if size not in size_index:
            raise ValueError("unknown size")
        return size

    def quality_of(name: object) -> int:
        if name not in quality_index:
            raise ValueError("unknown quality")
        return quality_index[str(name)]

    by_round: dict[int, list[dict]] = defaultdict(list)
    largest_rounds: list[tuple[int, str]] = []
    for position, event in enumerate(snapshot.events):
        try:
            round_number = int(event["round_number"])
        except (KeyError, TypeError, ValueError):
            unsupported.append(f"event_round:{position}")
            continue
        if round_number > current:
            continue
        semantics = event.get("semantics") or {}
        effect = str(semantics.get("effect") or "unparsed")
        label = f"R{round_number}:{event.get('kind')}:{effect}"
        if not semantics.get("parsed") or not _trusted(event, minimum_confidence):
            unsupported.append(f"untrusted_event:{label}")
            continue
        by_round[round_number].append(semantics)
        # The scan that shows this round's map reveals has not happened yet.
        pending = prewarm and round_number == current
        try:
            if effect == "quality_item_count":
                put(ev.count, quality_of(semantics["quality"]), int(semantics["observed_count"]), label)
            elif effect in COUNT_BY_QUALITY:
                if pending:
                    continue
                if semantics.get("reveal_outcome_available") is False or not map_trusted:
                    ignored.append(f"count_reveal_without_trusted_scan:{label}")
                    continue
                quality = quality_of(semantics["quality"])
                visible = sum(1 for item in items if item_quality(item) == QUALITY_ORDER[quality])
                put(ev.count, quality, visible, label)
            elif effect == COUNT_BY_SIZE:
                if pending:
                    continue
                if semantics.get("reveal_outcome_available") is False or not map_trusted:
                    ignored.append(f"count_reveal_without_trusted_scan:{label}")
                    continue
                size = size_of(semantics)
                visible = sum(1 for item in items if item_size(item) == size)
                put(ev.size_count, size_index[size], visible, label)
            elif effect == "size_item_count":
                put(ev.size_count, size_index[size_of(semantics)], int(semantics["observed_count"]), label)
            elif effect == "highest_quality":
                put_scalar("highest", quality_of(semantics["observed_quality"]), label)
            elif effect in {"max_item_value", "max_value_per_cell"}:
                value = int(semantics["observed_value"])
                if value < 0:
                    unsupported.append(f"empty_world_sentinel:{label}")
                    continue
                put_scalar("max_value" if effect == "max_item_value" else "max_per_cell", value, label)
            elif effect == "total_occupied_cells":
                put_scalar("total_cells", int(semantics["observed_cells"]), label)
            elif effect == "quality_average_area":
                shown = math.floor(float(semantics["observed_average_cells"]) * 100 + 1e-9)
                put(ev.avg_area, quality_of(semantics["quality"]), (shown, 100), label)
            elif effect == "quality_total_value":
                put(ev.total_value, quality_of(semantics["quality"]), int(semantics["observed_value"]), label)
            elif effect == "size_average_value":
                put(ev.size_avg_value, size_index[size_of(semantics)], int(semantics["observed_value"]), label)
            elif effect in {RANDOM_IDENTITY, RANDOM_QUALITY}:
                pass  # attributed from the map below
            elif effect == RANDOM_POSITION:
                ignored.append(f"position_only_reveal:{label}")
            elif effect == LARGEST_OUTLINE:
                if not pending:
                    largest_rounds.append((round_number, label))
            else:
                unsupported.append(f"unsupported_effect:{label}")
        except (KeyError, TypeError, ValueError, OverflowError):
            unsupported.append(f"invalid_event_fields:{label}")
    if len(ev.size_avg_value) > 1:
        unsupported.append("multiple_size_average_clues")

    # The largest item's outline stays on the map, so on a complete scan no
    # item of known size is larger and it is one of them. An anchor that
    # first appeared bare (no size, no quality) in that round and that no
    # position-only reveal explains means an outline was not read, and then
    # the maximum may be too small.
    def bare_from(item: dict, round_number: int) -> bool:
        if item.get("first_seen_round") != round_number or item_size(item) is not None:
            return False
        first_quality = item.get("quality_first_round")
        return item.get("quality") is None or first_quality is None or int(first_quality) > round_number

    for round_number, label in largest_rounds:
        if not map_trusted:
            ignored.append(f"largest_outline_without_trusted_scan:{label}")
            continue
        areas = [size[0] * size[1] for size in map(item_size, items) if size is not None]
        if not areas:
            ignored.append(f"largest_outline_not_read:{label}")
            continue
        seen = by_round[round_number]
        bare = sum(1 for item in items if bare_from(item, round_number))
        try:
            explained = sum(int(s["count"]) for s in seen if s.get("effect") == RANDOM_POSITION)
        except (KeyError, TypeError, ValueError):
            explained = 0
        # A 1x1 outline looks the same as a bare anchor.
        tiny_outlines = any(
            s.get("effect") == COUNT_BY_SIZE and (s.get("width"), s.get("height")) == (1, 1) for s in seen
        )
        if bare > explained and not tiny_outlines:
            ignored.append(f"largest_outline_not_read:{label}")
            continue
        put_scalar("largest_area", max(areas), label)

    # Random reveals: a complete identity can only come from an identity
    # reveal; a bare quality counts as a random draw when it first appeared
    # in a round with a random quality reveal and no event that shows every
    # item of that quality. Draws on items whose quality was already visible
    # cannot be seen and are left out.
    identities: Counter = Counter()
    qualities: Counter = Counter()
    if items and map_trusted:
        drawn: dict[int, list[int]] = defaultdict(list)
        for item in items:
            known = identity(item)
            if known is not None:
                identities[catalog_index[known.catalog_id]] += 1
                continue
            quality = item.get("quality")
            if quality not in quality_index:
                continue
            first = item.get("quality_first_round")
            if first is None:
                if any(s.get("effect") == RANDOM_QUALITY for sems in by_round.values() for s in sems):
                    ignored.append("quality_reveal_undated")
                continue
            seen = by_round.get(int(first), [])
            if not any(s.get("effect") == RANDOM_QUALITY for s in seen):
                continue
            if any(s.get("effect") in COUNT_BY_QUALITY and s.get("quality") == quality for s in seen):
                ignored.append(f"quality_reveal_unattributed:R{int(first)}")
                continue
            drawn[int(first)].append(quality_index[quality])
        for round_number, drawn_qualities in drawn.items():
            try:
                limit = sum(
                    int(s["count"]) for s in by_round[round_number] if s.get("effect") == RANDOM_QUALITY
                )
            except (KeyError, TypeError, ValueError):
                limit = None
            # More new qualities than draws means another source gave some.
            if limit is None or len(drawn_qualities) > limit:
                ignored.append(f"quality_reveal_overflow:R{round_number}")
                continue
            qualities.update(drawn_qualities)
    elif any(s.get("effect") in {RANDOM_IDENTITY, RANDOM_QUALITY} for sems in by_round.values() for s in sems):
        ignored.append("random_reveals_without_trusted_scan")
    ev.revealed_identities = identities
    ev.revealed_qualities = qualities
    return ExactEvidence(
        evidence=None if unsupported else ev,
        ignored=tuple(dict.fromkeys(ignored)),
        unsupported=tuple(dict.fromkeys(unsupported)),
    )


def evidence_from_decision(decision: dict, catalog: Sequence[CatalogItem]) -> ExactEvidence:
    """Exact sampler evidence from a packet-capture decision record.

    Capture shows which items every random reveal drew and the map height
    from the match opening, so nothing has to be attributed from the map.
    The translation is the one the sampler was benchmarked with.
    """
    from .exact_sampler import Evidence

    ignored: list[str] = []
    unsupported: list[str] = []
    quality_index = {quality: index for index, quality in enumerate(QUALITY_ORDER)}
    size_index = {f"{width}x{height}": index for index, (width, height) in enumerate(SIZES)}
    ev = Evidence(map_rows=decision.get("map_rows"))
    if ev.map_rows is None:
        ignored.append("map_rows_missing")

    def put(target: dict, key, value, label: str) -> None:
        if key in target and target[key] != value:
            unsupported.append(f"conflicting_clues:{label}")
        target[key] = value

    def put_scalar(name: str, value, label: str) -> None:
        previous = getattr(ev, name)
        if previous is not None and previous != value:
            unsupported.append(f"conflicting_clues:{label}")
        setattr(ev, name, value)

    def identity_index(value: int, quality) -> int:
        matching = [index for index, item in enumerate(catalog)
                    if item.value == value and (quality is None or item.quality == quality)]
        if len(matching) != 1:
            raise ValueError("revealed value does not match one catalog item")
        return matching[0]

    identity_items: dict[int, int] = {}
    quality_items: dict[int, int] = {}
    for event in decision.get("events") or []:
        effect = str(event.get("effect") or "")
        label = f"R{event.get('round')}:{event.get('scope')}:{effect}"
        try:
            group, value = event["group_value"], int(event["observed_value"])
            if effect == "quality_item_count":
                put(ev.count, quality_index[group], value, label)
            elif effect in COUNT_BY_QUALITY:
                put(ev.count, quality_index[group], int(event["reported_item_count"]), label)
            elif effect == COUNT_BY_SIZE:
                put(ev.size_count, size_index[group], int(event["reported_item_count"]), label)
            elif effect == "highest_quality":
                put_scalar("highest", value - 1, label)
            elif effect == LARGEST_OUTLINE:
                put_scalar("largest_area", int(event["reported_cell_count"]), label)
            elif effect in {"max_item_value", "max_value_per_cell"}:
                if value < 0:
                    unsupported.append(f"empty_world_sentinel:{label}")
                    continue
                put_scalar("max_value" if effect == "max_item_value" else "max_per_cell", value, label)
            elif effect == "total_occupied_cells":
                put_scalar("total_cells", value, label)
            elif effect == "quality_average_area":
                put(ev.avg_area, quality_index[group], (value, int(event["value_scale"])), label)
            elif effect == "quality_total_value":
                put(ev.total_value, quality_index[group], value, label)
            elif effect == "size_average_value":
                put(ev.size_avg_value, size_index[group], value, label)
            elif effect == RANDOM_POSITION:
                ignored.append(f"position_only_reveal:{label}")
            elif effect not in {RANDOM_IDENTITY, RANDOM_QUALITY}:
                unsupported.append(f"unsupported_effect:{label}")
            if effect in {RANDOM_IDENTITY, RANDOM_QUALITY}:
                for item in event.get("items") or []:
                    if item.get("value"):
                        identity_items[int(item["index"])] = identity_index(int(item["value"]), item.get("quality"))
                    elif item.get("quality"):
                        quality_items[int(item["index"])] = quality_index[item["quality"]]
        except (KeyError, TypeError, ValueError, OverflowError):
            unsupported.append(f"invalid_event_fields:{label}")
    if len(ev.size_avg_value) > 1:
        unsupported.append("multiple_size_average_clues")
    ev.revealed_identities = Counter(identity_items.values())
    ev.revealed_qualities = Counter(q for index, q in quality_items.items() if index not in identity_items)
    return ExactEvidence(
        evidence=None if unsupported else ev,
        ignored=tuple(dict.fromkeys(ignored)),
        unsupported=tuple(dict.fromkeys(unsupported)),
    )


def evidence_seed(snapshot: ObservationSnapshot) -> int:
    """A seed fixed by the evidence, so the same evidence gives the same estimate."""
    payload = json.dumps(
        {
            "round": snapshot.round_number,
            "events": [
                [event.get("round_number"), event.get("kind"), event.get("semantics")]
                for event in snapshot.events
            ],
            "items": [
                [item.get("row"), item.get("column"), item.get("quality"), item.get("catalog_id"),
                 item.get("width"), item.get("height"), item.get("spatial"), item.get("quality_first_round")]
                for item in snapshot.map_items
            ],
            "rows": [snapshot.map_rows, snapshot.map_height_exact],
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12], 16)


def build_prior(world_model, catalog: Sequence[CatalogItem]):
    """The exact sampler's view of a loaded GenerativeWorldModelV2."""
    import numpy as np

    from .exact_sampler import Prior

    cache: dict = {}

    def memo(name, function):
        def wrapped(*args):
            key = (name, *args)
            if key not in cache:
                cache[key] = function(*args)
            return cache[key]

        return wrapped

    return Prior(
        values=[item.value for item in catalog],
        areas=[item.width * item.height for item in catalog],
        quality=[QUALITY_ORDER.index(item.quality) for item in catalog],
        size=[SIZES.index((item.width, item.height)) for item in catalog],
        count_p=dict(world_model._count_probabilities),
        quotas=tuple(world_model.minimum_quotas[quality] for quality in QUALITY_ORDER),
        colored_p=memo("c", lambda n: np.asarray(world_model._colored_count_probabilities(int(n)), float)),
        noncolored_p=memo("nc", lambda n: np.asarray(world_model._non_colored_probabilities(int(n)), float)),
        identity_p=memo("id", lambda q, n: np.asarray(
            world_model._identity_probabilities(QUALITY_ORDER[q], total_count=int(n)), float)),
    )


class RowsTable:
    """Map-height likelihood: P(rows - ceil(cells / 10) | occupied-cell band).

    Counted on clean public final maps and frozen as ``ROWS_TABLE_V1``;
    half-count smoothing over each band's support widened by one row.
    """

    def __init__(self, payload: dict) -> None:
        self.band_cells = int(payload.get("band_cells", 40))
        self.max_band = int(payload.get("max_band", 7))
        self.bands = {
            int(band): {int(slack): int(count) for slack, count in table.items()}
            for band, table in payload["bands"].items()
        }
        if not self.bands:
            raise ValueError("rows table has no bands")

    @classmethod
    def packaged(cls) -> "RowsTable":
        return cls(ROWS_TABLE_V1)

    def log_likelihood(self, rows, cells):
        import numpy as np

        cells = np.asarray(cells)
        out = np.full(len(cells), -np.inf)
        slack = int(rows) - np.ceil(cells / 10).astype(np.int64)
        bands = np.minimum(cells // self.band_cells, self.max_band)
        for band in np.unique(bands):
            table = self.bands.get(int(band)) or self.bands[max(self.bands)]
            support = range(min(table) - 1, max(table) + 2)
            total = sum(table.values()) + 0.5 * len(support)
            for k in support:
                out[(bands == band) & (slack == k)] = math.log((table.get(k, 0) + 0.5) / total)
        return out
