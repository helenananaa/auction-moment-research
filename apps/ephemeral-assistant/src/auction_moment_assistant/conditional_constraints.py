"""Compile decision-time numeric evidence into exact identity-count constraints.

The model is a feasible-count generator, not a probability model.  Entry order,
the deterministic packing rule, and random-reveal likelihood are handled by
the caller.  Unsupported evidence is diagnosed and makes this model infeasible
so a caller cannot accidentally sample a silently relaxed condition.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence

from ortools.sat.python import cp_model

from .models import CatalogItem
from .observations import ObservationSnapshot


QUALITY_ORDER = ("白", "蓝", "紫", "金", "彩")
INFORMATIONAL_EFFECTS = {
    "reveal_identity_random",
    "reveal_position_quality_random",
    "reveal_outline_by_size",
    "reveal_position_random",
    "reveal_largest_area_item_outline",
    "reveal_outline_by_quality",
    "reveal_position_by_quality",
}


def _integer(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer observation")
    result = int(value)  # type: ignore[arg-type]
    if not isinstance(value, str) and value != result:
        raise ValueError("non-integer observation")
    return result


def compile_hard_constraints(
    snapshot: ObservationSnapshot,
    catalog: Sequence[CatalogItem],
    world_model: object,
) -> tuple[cp_model.CpModel, tuple[cp_model.IntVar, ...], tuple[str, ...]]:
    """Return an exact numeric count model and explicit unsupported diagnostics.

    Each returned variable is one catalog identity's multiplicity, in catalog
    order.  Total-count support preserves any zero-probability holes.  Visible
    anchor groups contribute necessary count bounds; the caller must still
    check the actual sequence/layout, which enforces their spatial matching.
    """
    model = cp_model.CpModel()
    diagnostics: list[str] = []

    def unsupported(label: str) -> None:
        diagnostics.append(f"unsupported:{label}")
        model.add(False)

    try:
        probabilities = getattr(world_model, "_count_probabilities", None)
        if isinstance(probabilities, Mapping):
            support = sorted(
                {
                    _integer(count)
                    for count, probability in probabilities.items()
                    if math.isfinite(float(probability)) and float(probability) > 0
                }
            )
        else:
            minimum = _integer(getattr(world_model, "min_count"))
            maximum = _integer(getattr(world_model, "max_count"))
            support = list(range(minimum, maximum + 1))
        if not support or support[0] < 0:
            raise ValueError("empty/negative count support")
    except (AttributeError, TypeError, ValueError, OverflowError):
        unsupported("world_model_count_support")
        support = [0]
    maximum = max(support)
    counts = tuple(
        model.new_int_var(0, maximum, f"count_{index}") for index in range(len(catalog))
    )
    total_count = model.new_int_var(min(support), maximum, "total_count")
    model.add(total_count == sum(counts))
    model.add_allowed_assignments([total_count], [(count,) for count in support])

    values: list[int] = []
    areas: list[int] = []
    for index, item in enumerate(catalog):
        try:
            value, width, height = (
                _integer(item.value),
                _integer(item.width),
                _integer(item.height),
            )
            if (
                value < 0
                or width < 1
                or height < 1
                or item.quality not in QUALITY_ORDER
            ):
                raise ValueError("invalid catalog semantics")
        except (TypeError, ValueError, OverflowError):
            unsupported(f"catalog:{index}")
            value, width, height = 0, 1, 1
        values.append(value)
        areas.append(width * height)

    def linear(indices: Sequence[int], coefficients: Sequence[int] | None = None):
        if coefficients is None:
            return sum(counts[index] for index in indices)
        return sum(counts[index] * coefficients[index] for index in indices)

    all_indices = tuple(range(len(catalog)))
    qualities = {
        quality: tuple(
            index for index, item in enumerate(catalog) if item.quality == quality
        )
        for quality in QUALITY_ORDER
    }
    for quality, raw_minimum in (
        getattr(world_model, "minimum_quotas", {}) or {}
    ).items():
        try:
            minimum = _integer(raw_minimum)
            if quality not in qualities or minimum < 0:
                raise ValueError("invalid minimum quota")
        except (TypeError, ValueError, OverflowError):
            unsupported(f"minimum_quota:{quality}")
            continue
        model.add(linear(qualities[quality]) >= minimum)

    for event_index, event in enumerate(snapshot.events):
        try:
            if _integer(event["round_number"]) > snapshot.round_number:
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            unsupported(f"event_round:{event_index}")
            continue
        semantics = event.get("semantics") or {}
        if not isinstance(semantics, Mapping):
            unsupported(f"event_semantics:{event_index}")
            continue
        effect = str(semantics.get("effect") or "unparsed")
        if (
            effect == "reveal_largest_area_item_outline"
            and semantics.get("reveal_outcome_available") is not False
        ):
            unsupported(f"largest_outline_binding:{event_index}")
            continue
        if not semantics.get("parsed"):
            unsupported(f"unparsed_event:{event_index}:{effect}")
            continue
        if effect in INFORMATIONAL_EFFECTS:
            continue
        try:
            indices = all_indices
            if effect.startswith("quality_"):
                quality = str(semantics["quality"])
                if quality not in qualities:
                    raise ValueError("unknown quality")
                indices = qualities[quality]
            elif effect.startswith("size_"):
                width, height = (
                    _integer(semantics["width"]),
                    _integer(semantics["height"]),
                )
                if width < 1 or height < 1:
                    raise ValueError("invalid size")
                indices = tuple(
                    index
                    for index, item in enumerate(catalog)
                    if (item.width, item.height) == (width, height)
                )
            count = linear(indices)
            value = linear(indices, values)
            area = linear(indices, areas)
            if effect in {"quality_item_count", "size_item_count", "total_item_count"}:
                model.add(count == _integer(semantics["observed_count"]))
            elif effect in {"quality_total_value", "size_total_value"}:
                model.add(value == _integer(semantics["observed_value"]))
            elif effect in {"quality_average_value", "size_average_value"}:
                observed = _integer(semantics["observed_value"])
                model.add(count >= 1)
                model.add(value >= observed * count)
                model.add(value <= (observed + 1) * count - 1)
            elif effect in {
                "quality_total_area",
                "size_total_area",
                "total_occupied_cells",
            }:
                model.add(area == _integer(semantics["observed_cells"]))
            elif effect in {
                "quality_average_area",
                "size_average_area",
                "average_item_area",
            }:
                observed = math.floor(
                    float(semantics["observed_average_cells"]) * 100 + 1e-9
                )
                model.add(count >= 1)
                model.add(100 * area >= observed * count)
                model.add(100 * area <= (observed + 1) * count - 1)
            elif effect in {"max_item_value", "max_value_per_cell"}:
                observed = _integer(semantics["observed_value"])
                if observed == -1:
                    # Match the explicit empty-world sentinel, rather than
                    # treating an empty maximum as zero-valued evidence.
                    model.add(total_count == 0)
                else:
                    scores = (
                        values
                        if effect == "max_item_value"
                        else [value // area for value, area in zip(values, areas)]
                    )
                    model.add(
                        linear(
                            tuple(
                                index
                                for index, score in enumerate(scores)
                                if score > observed
                            )
                        )
                        == 0
                    )
                    model.add(
                        linear(
                            tuple(
                                index
                                for index, score in enumerate(scores)
                                if score == observed
                            )
                        )
                        >= 1
                    )
            elif effect == "highest_quality":
                observed_quality = str(semantics["observed_quality"])
                if observed_quality not in qualities:
                    raise ValueError("unknown highest quality")
                observed_index = QUALITY_ORDER.index(observed_quality)
                model.add(linear(qualities[observed_quality]) >= 1)
                model.add(
                    linear(
                        tuple(
                            index
                            for index, item in enumerate(catalog)
                            if QUALITY_ORDER.index(item.quality) > observed_index
                        )
                    )
                    == 0
                )
            else:
                unsupported(f"event_effect:{event_index}:{effect}")
        except (KeyError, TypeError, ValueError, OverflowError):
            unsupported(f"event_fields:{event_index}:{effect}")

    candidates_by_anchor: dict[tuple[int, int], set[int]] = {}
    for observation_index, raw in enumerate(snapshot.map_items):
        try:
            confidence = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if not raw.get("human_locked") and not (
            math.isfinite(confidence) and confidence >= 0.85
        ):
            continue
        try:
            row, column = _integer(raw["row"]), _integer(raw["column"])
            if row < 0 or not 0 <= column < 10:
                raise ValueError("invalid anchor")
            spatial = str(raw.get("spatial") or "top_left")
            if spatial not in {"top_left", "outline", "complete"}:
                raise ValueError("invalid spatial state")
            candidates = set(all_indices)
            quality = str(raw.get("quality") or "")
            if quality:
                if quality not in qualities:
                    raise ValueError("invalid quality")
                candidates.intersection_update(qualities[quality])
            if spatial in {"outline", "complete"} and (
                raw.get("width") is not None or raw.get("height") is not None
            ):
                width, height = _integer(raw.get("width")), _integer(raw.get("height"))
                if width < 1 or height < 1:
                    raise ValueError("invalid known size")
                candidates.intersection_update(
                    index
                    for index, item in enumerate(catalog)
                    if (item.width, item.height) == (width, height)
                )
            catalog_id = str(raw.get("catalog_id") or "")
            if spatial == "complete" and catalog_id:
                candidates.intersection_update(
                    index
                    for index, item in enumerate(catalog)
                    if item.catalog_id == catalog_id
                )
            anchor = (row, column)
            candidates_by_anchor.setdefault(
                anchor, set(all_indices)
            ).intersection_update(candidates)
        except (KeyError, TypeError, ValueError, OverflowError):
            unsupported(f"map_item:{observation_index}")
    model.add(total_count >= len(candidates_by_anchor))
    group_minimum = Counter(
        tuple(sorted(indices)) for indices in candidates_by_anchor.values()
    )
    for indices, minimum in group_minimum.items():
        model.add(linear(indices) >= minimum)
        if not indices:
            diagnostics.append("infeasible:map_identity_shape_quality_conflict")
    return model, counts, tuple(diagnostics)
