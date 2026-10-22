"""Deterministic generation-rule layout and decision-time spatial evidence.

An identity-count vector alone does not determine a map.  These helpers keep
the concrete entry sequence explicit and use the declared ten-column,
non-rotating, row-major first-fit rule.  They deliberately do not replace that
rule with arbitrary rectangle packing or an area/height approximation.
"""

from __future__ import annotations

import math
import operator
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import random

    from .models import CatalogItem
    from .observations import ObservationSnapshot


COLUMNS = 10
MINIMUM_CONFIDENCE = 0.85
_FULL_ROW = (1 << COLUMNS) - 1


@dataclass(frozen=True, slots=True)
class Placement:
    catalog_index: int
    row: int
    column: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class PackedLayout:
    placements: tuple[Placement, ...]
    used_rows: int


def _first_free(width: int, height: int, occupied: Sequence[int]) -> tuple[int, int]:
    for row in range(len(occupied) + 1):
        union = 0
        for mask in occupied[row : row + height]:
            union |= mask
        free = _FULL_ROW ^ union
        starts = free
        for shift in range(1, width):
            starts &= free >> shift
        starts &= (1 << (COLUMNS - width + 1)) - 1
        if starts:
            return row, (starts & -starts).bit_length() - 1
    raise AssertionError("the first empty row must fit")


def _occupy(placement: Placement, occupied: Sequence[int]) -> list[int]:
    result = list(occupied)
    bottom = placement.row + placement.height
    result.extend([0] * max(0, bottom - len(result)))
    bits = ((1 << placement.width) - 1) << placement.column
    for item_row in range(placement.row, bottom):
        result[item_row] |= bits
    return result


def pack_sequence(
    sequence: Sequence[int], catalog: Sequence[CatalogItem]
) -> PackedLayout:
    """Place identities in entry order, at the first free row/column each time.

    Coordinates are zero based.  ``used_rows`` is the exclusive bottom edge,
    and is zero for an empty sequence.  Invalid identities or dimensions raise
    ``ValueError`` instead of silently rotating, skipping, or clipping them.
    """
    occupied: list[int] = []
    placements: list[Placement] = []
    for raw_index in sequence:
        try:
            index = operator.index(raw_index)
        except TypeError as exc:
            raise ValueError("catalog indices must be integers") from exc
        if isinstance(raw_index, bool) or not 0 <= index < len(catalog):
            raise ValueError(f"invalid catalog index: {raw_index!r}")
        item = catalog[index]
        try:
            width = operator.index(item.width)
            height = operator.index(item.height)
        except TypeError as exc:
            raise ValueError("catalog dimensions must be integers") from exc
        if not 1 <= width <= COLUMNS or height < 1:
            raise ValueError(f"invalid catalog dimensions: {width}x{height}")
        # The first entirely new row always fits, so this scan is bounded by
        # the current occupied height, even for shapes larger than one row.
        row, column = _first_free(width, height, occupied)
        placement = Placement(index, row, column, width, height)
        occupied = _occupy(placement, occupied)
        placements.append(placement)
    return PackedLayout(tuple(placements), len(occupied))


def _integer(value: object) -> int:
    """Read an exact integer without truncating malformed spatial evidence."""
    if isinstance(value, bool):
        raise ValueError("boolean coordinate")
    result = int(value)  # type: ignore[arg-type]
    if not isinstance(value, str) and value != result:
        raise ValueError("non-integer coordinate")
    return result


def _trusted(raw: dict) -> bool:
    if raw.get("human_locked"):
        return True
    try:
        confidence = float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(confidence) and confidence >= MINIMUM_CONFIDENCE


def layout_diagnostics(
    packed: PackedLayout,
    catalog: Sequence[CatalogItem],
    snapshot: ObservationSnapshot,
) -> tuple[str, ...]:
    """Return hard-constraint mismatches for an already packed candidate.

    Estimated scroll height is not a capacity or a lower/upper bound.  Exact
    height, however, is the *realized* occupied height, not merely capacity.
    Every trusted clue binds to the item at its own anchor.  Repeated clues at
    one anchor all constrain that same item, and a placement cannot satisfy a
    different anchor.  Detector marker dimensions and uncertain identity
    candidates are not known item dimensions or identities.
    """
    issues: list[str] = []
    if snapshot.map_height_exact:
        try:
            expected_rows = _integer(snapshot.map_rows)
            if expected_rows < 1:
                raise ValueError("nonpositive map height")
        except (TypeError, ValueError, OverflowError):
            issues.append("invalid_exact_map_rows")
        else:
            if packed.used_rows != expected_rows:
                issues.append(f"map_height_mismatch:{expected_rows}:{packed.used_rows}")
    anchors = {
        (placement.row, placement.column): placement for placement in packed.placements
    }
    for observation_index, raw in enumerate(snapshot.map_items):
        if not _trusted(raw):
            continue
        try:
            row = _integer(raw["row"])
            column = _integer(raw["column"])
            if row < 0 or not 0 <= column < COLUMNS:
                raise ValueError("out of bounds coordinate")
        except (KeyError, TypeError, ValueError, OverflowError):
            issues.append(f"invalid_map_position:{observation_index}")
            continue
        anchor = f"{row}:{column}"
        placement = anchors.get((row, column))
        if placement is None:
            issues.append(f"missing_map_anchor:{anchor}")
            continue
        spatial = str(raw.get("spatial") or "top_left")
        if spatial not in {"top_left", "outline", "complete"}:
            issues.append(f"invalid_map_spatial:{anchor}")
            continue
        item = catalog[placement.catalog_index]
        quality = str(raw.get("quality") or "")
        if quality and quality != item.quality:
            issues.append(f"map_quality_mismatch:{anchor}")
        if spatial in {"outline", "complete"}:
            raw_width, raw_height = raw.get("width"), raw.get("height")
            if raw_width is not None or raw_height is not None:
                try:
                    width, height = _integer(raw_width), _integer(raw_height)
                    if width < 1 or height < 1:
                        raise ValueError("nonpositive item size")
                except (TypeError, ValueError, OverflowError):
                    issues.append(f"invalid_map_size:{anchor}")
                else:
                    if (width, height) != (placement.width, placement.height):
                        issues.append(f"map_size_mismatch:{anchor}")
        catalog_id = str(raw.get("catalog_id") or "")
        if spatial == "complete" and catalog_id != "":
            if catalog_id != item.catalog_id:
                issues.append(f"map_identity_mismatch:{anchor}")
    # A complete, reliable scan also observes the absence of markers.  An
    # "all members of this group" event guarantees every target member has an
    # anchor in the cumulative visible set, even if another event first exposed
    # that anchor or its current attributes remain partially classified.
    if snapshot.completeness_confirmed and all(
        _trusted(raw) for raw in snapshot.map_items
    ):
        visible_anchors: set[tuple[int, int]] = set()
        for raw in snapshot.map_items:
            try:
                visible_anchors.add((_integer(raw["row"]), _integer(raw["column"])))
            except (KeyError, TypeError, ValueError, OverflowError):
                # The positive-observation pass already rejects malformed
                # anchors; they cannot establish a visible marker here.
                continue
        for event_index, event in enumerate(snapshot.events):
            semantics = event.get("semantics") or {}
            effect = str(semantics.get("effect") or "")
            if effect not in {
                "reveal_position_by_quality",
                "reveal_outline_by_quality",
                "reveal_outline_by_size",
            }:
                continue
            if (
                not semantics.get("parsed")
                or not _trusted(event)
                or semantics.get("reveal_outcome_available") is False
            ):
                continue
            try:
                if _integer(event["round_number"]) > snapshot.round_number:
                    continue
                if effect == "reveal_outline_by_size":
                    width = _integer(semantics["width"])
                    height = _integer(semantics["height"])
                    if width < 1 or height < 1:
                        raise ValueError("nonpositive reveal size")
                    target_indices = {
                        index
                        for index, item in enumerate(catalog)
                        if (item.width, item.height) == (width, height)
                    }
                else:
                    quality = str(semantics["quality"])
                    if quality not in {"白", "蓝", "紫", "金", "彩"}:
                        raise ValueError("unknown reveal quality")
                    target_indices = {
                        index
                        for index, item in enumerate(catalog)
                        if item.quality == quality
                    }
            except (KeyError, TypeError, ValueError, OverflowError):
                issues.append(f"invalid_deterministic_reveal:{event_index}")
                continue
            for placement in packed.placements:
                anchor_position = (placement.row, placement.column)
                if (
                    placement.catalog_index in target_indices
                    and anchor_position not in visible_anchors
                ):
                    issues.append(
                        f"unobserved_revealed_anchor:{effect}:"
                        f"{placement.row}:{placement.column}"
                    )
    return tuple(issues)


def layout_matches(
    packed: PackedLayout,
    catalog: Sequence[CatalogItem],
    snapshot: ObservationSnapshot,
) -> bool:
    """Check exact observations against a reusable concrete packed layout."""
    return not layout_diagnostics(packed, catalog, snapshot)


def layout_compatible(
    sequence: Sequence[int],
    catalog: Sequence[CatalogItem],
    snapshot: ObservationSnapshot,
) -> bool:
    """Pack once and test hard observations; never smooth away a mismatch."""
    return layout_matches(pack_sequence(sequence, catalog), catalog, snapshot)


def find_compatible_sequence(
    counts: Sequence[int],
    catalog: Sequence[CatalogItem],
    snapshot: ObservationSnapshot,
    rng: random.Random,
    maximum_nodes: int = 2000,
    *,
    deadline: float | None = None,
) -> tuple[int, ...] | None:
    """Find a valid initialization order, without assigning it probability.

    Try uniform shuffles first, then a bounded randomized depth-first search.
    The latter is deliberately a *seed search*, not a posterior draw or an
    estimator of the fraction of compatible permutations.  ``deadline`` is an
    absolute ``time.monotonic()`` deadline.  Exhaustion returns ``None`` and
    never relaxes a clue or the generation rule.
    """
    if len(counts) != len(catalog):
        raise ValueError("counts and catalog lengths differ")
    remaining = [_integer(count) for count in counts]
    if any(count < 0 for count in remaining):
        raise ValueError("negative identity count")
    sequence = [index for index, count in enumerate(remaining) for _ in range(count)]
    # Validate the supplied identities/dimensions even if no evidence exists.
    pack_sequence(sequence, catalog)
    budget = max(0, int(maximum_nodes))
    nodes = 0

    def exhausted() -> bool:
        return nodes >= budget or (
            deadline is not None and time.monotonic() >= deadline
        )

    for _ in range(min(24, budget)):
        if exhausted():
            return None
        rng.shuffle(sequence)
        nodes += 1
        if layout_compatible(sequence, catalog, snapshot):
            return tuple(sequence)
    if exhausted():
        return None

    exact_rows: int | None = None
    if snapshot.map_height_exact:
        try:
            exact_rows = _integer(snapshot.map_rows)
            if exact_rows < 1:
                return None
        except (TypeError, ValueError, OverflowError):
            return None
    observations: dict[tuple[int, int], list[dict]] = {}
    reserved: list[tuple[int, int, int, int]] = []
    index_by_id = {item.catalog_id: index for index, item in enumerate(catalog)}
    for raw in snapshot.map_items:
        if not _trusted(raw):
            continue
        try:
            row, column = _integer(raw["row"]), _integer(raw["column"])
            if row < 0 or not 0 <= column < COLUMNS:
                return None
            spatial = str(raw.get("spatial") or "top_left")
            if spatial not in {"top_left", "outline", "complete"}:
                return None
            observations.setdefault((row, column), []).append(raw)
            width, height = 1, 1
            if spatial in {"outline", "complete"}:
                if raw.get("width") is not None or raw.get("height") is not None:
                    width, height = (
                        _integer(raw.get("width")),
                        _integer(raw.get("height")),
                    )
                if spatial == "complete" and raw.get("catalog_id"):
                    item_index = index_by_id.get(str(raw["catalog_id"]))
                    if item_index is None:
                        return None
                    item = catalog[item_index]
                    width, height = item.width, item.height
            if width < 1 or height < 1 or column + width > COLUMNS:
                return None
            if exact_rows is not None and row + height > exact_rows:
                return None
            reserved.append((row, column, width, height))
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
    anchor_snapshots = {
        anchor: replace(snapshot, map_items=tuple(raw_items), map_height_exact=False)
        for anchor, raw_items in observations.items()
    }
    dead_states: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    prefix: list[int] = []
    placements: list[Placement] = []

    def search(occupied: list[int]) -> tuple[int, ...] | None:
        nonlocal nodes
        if exhausted():
            return None
        if len(prefix) == len(sequence):
            packed = PackedLayout(tuple(placements), len(occupied))
            return tuple(prefix) if layout_matches(packed, catalog, snapshot) else None
        state = (tuple(occupied), tuple(remaining))
        if state in dead_states:
            return None
        candidates: list[tuple[int, Placement]] = []
        indices = [index for index, count in enumerate(remaining) if count]
        rng.shuffle(indices)
        for index in indices:
            if exhausted():
                return None
            nodes += 1
            item = catalog[index]
            row, column = _first_free(item.width, item.height, occupied)
            placement = Placement(index, row, column, item.width, item.height)
            if exact_rows is not None and row + item.height > exact_rows:
                continue
            anchor = (row, column)
            evidence = anchor_snapshots.get(anchor)
            if evidence is not None and not layout_matches(
                PackedLayout((placement,), row + item.height), catalog, evidence
            ):
                continue
            # No previously placed item can cover a future clue's anchor or
            # its known rectangle.  All clues at this anchor bind this item.
            if any(
                (known_row, known_column) != anchor
                and row < known_row + known_height
                and known_row < row + item.height
                and column < known_column + known_width
                and known_column < column + item.width
                for known_row, known_column, known_width, known_height in reserved
            ):
                continue
            candidates.append((0 if evidence is not None else 1, placement))
        # Prefer filling a known anchor; randomized ties avoid a fixed identity
        # bias in the search heuristic.  This ordering has no sampling meaning.
        candidates.sort(key=lambda candidate: candidate[0])
        for _, placement in candidates:
            if exhausted():
                return None
            index = placement.catalog_index
            remaining[index] -= 1
            prefix.append(index)
            placements.append(placement)
            result = search(_occupy(placement, occupied))
            placements.pop()
            prefix.pop()
            remaining[index] += 1
            if result is not None:
                return result
        dead_states.add(state)
        return None

    return search([])
