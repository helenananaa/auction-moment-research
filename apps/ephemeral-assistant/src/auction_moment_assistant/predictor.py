from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

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
_CONDITIONAL_RETRY_SIGNATURE_LIMIT = 8
_CONDITIONAL_RETRY_INDEX_MASK = (1 << 64) - 1
# An odd step gives distinct 64-bit seeds across the bounded retry cycle.
_CONDITIONAL_RETRY_SEED_STEP = 0x9E3779B97F4A7C15


class PredictionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PredictionResult:
    revision: int
    status: str
    compatible_worlds: int
    p10: int | None
    p50: int | None
    p90: int | None
    minimum: int | None
    maximum: int | None
    actionable: bool
    issues: tuple[str, ...]
    contract: str = "empirical_compatible_worlds_uncalibrated"
    v6_prediction: int | None = None
    v2_prediction: int | None = None
    generation_weight: float | None = None
    model_version: str | None = None
    estimate_source: str | None = None
    recommended_bid: int | None = None
    bid_safety_factor: float | None = None
    required_winning_multiplier: float | None = None
    maximum_opponent_bid: int | None = None
    bid_basis: str | None = None
    diagnostics: tuple[str, ...] = ()
    effective_sample_size: float | None = None
    posterior_converged: bool = False
    posterior_method: str = "empirical_filter"
    conditional_diagnostics: Mapping | None = None


class EmpiricalWorldPredictor:
    """Filter frozen/public-prior worlds by decision-time evidence.

    Historical settlements only audit eligibility of independent prior worlds.
    Every world value is recomputed from public catalog counts, and no current
    settlement enters inference. Returned quantiles describe compatible
    frozen worlds and are intentionally never marked actionable/calibrated.
    """

    def __init__(
        self,
        treasures_csv: Path,
        catalog: Sequence[CatalogItem],
        *,
        world_model_path: Path | None = None,
        prior_samples: int = 2048,
        conditional_resample_max_draws: int = 32768,
        conditional_resample_target: int = 32,
        conditional_resample_batch_size: int = 2048,
        conditional_sampling: bool = True,
        conditional_budget_seconds: float = 2.0,
        sessions_csv: Path | None = None,
        filter_empirical_quality: bool = True,
    ) -> None:
        self.catalog = tuple(catalog)
        self.index_by_id = {
            item.catalog_id: index for index, item in enumerate(self.catalog)
        }
        from .prior_quality import load_empirical_counts

        empirical_counts, self.empirical_prior_diagnostics = load_empirical_counts(
            treasures_csv,
            self.catalog,
            sessions_csv=sessions_csv,
            quality_filter=filter_empirical_quality,
        )
        if not len(empirical_counts) and world_model_path is None:
            raise PredictionError("没有通过复核的公开世界样本")
        self.empirical_world_count = len(empirical_counts)
        prior_counts = np.empty((0, len(self.catalog)), dtype=np.int16)
        self.world_model_version = "none"
        self._world_model = None
        if world_model_path is not None:
            try:
                payload = json.loads(Path(world_model_path).read_text(encoding="utf-8"))
                schema = str(payload.get("schema_version") or "")
                if schema == "world-model-v2":
                    from generative_world_model_v2 import GenerativeWorldModelV2

                    model = GenerativeWorldModelV2.load(
                        Path(world_model_path), self.catalog
                    )
                    self.world_model_version = "world_model_v2"
                else:
                    from auction_moment_research.world_model import (
                        ProbabilisticWorldModel,
                    )

                    model = ProbabilisticWorldModel.load(
                        Path(world_model_path), self.catalog
                    )
                    self.world_model_version = (
                        schema.replace("-", "_") if schema else "world_model_v1"
                    )
            except Exception as exc:
                raise PredictionError(f"世界模型加载失败：{exc}") from exc
            rng = random.Random(20260804)
            self._world_model = model
            prior_counts = np.asarray(
                [
                    model.sample_prior_counts(rng)
                    for _ in range(max(0, int(prior_samples)))
                ],
                dtype=np.int16,
            )
        self.prior_world_count = len(prior_counts)
        self._prior_counts = prior_counts
        self.conditional_sampling = bool(conditional_sampling)
        self.conditional_budget_seconds = float(conditional_budget_seconds)
        if (
            not math.isfinite(self.conditional_budget_seconds)
            or self.conditional_budget_seconds <= 0
        ):
            raise ValueError("conditional_budget_seconds must be finite and positive")
        self._posterior_cache: OrderedDict[str, PredictionResult] = OrderedDict()
        self._conditional_retry_indices: OrderedDict[str, int] = OrderedDict()
        self.conditional_resample_max_draws = max(
            0, int(conditional_resample_max_draws)
        )
        self.conditional_resample_target = max(1, int(conditional_resample_target))
        self.conditional_resample_batch_size = max(
            1, int(conditional_resample_batch_size)
        )
        self._conditional_cache: OrderedDict[str, tuple[np.ndarray, int]] = (
            OrderedDict()
        )
        self.world_counts = (
            np.concatenate([empirical_counts, prior_counts], axis=0)
            if len(prior_counts)
            else empirical_counts
        )
        self.values = np.asarray([item.value for item in self.catalog], dtype=np.int64)
        self.areas = np.asarray(
            [item.width * item.height for item in self.catalog], dtype=np.int16
        )
        self.total_counts = self.world_counts.sum(axis=1, dtype=np.int64)
        self.total_values = self.world_counts @ self.values
        self.total_areas = self.world_counts @ self.areas.astype(np.int64)
        self.quality_masks = {
            quality: np.asarray(
                [item.quality == quality for item in self.catalog], dtype=bool
            )
            for quality in QUALITY_ORDER
        }
        self.size_masks = {
            (width, height): np.asarray(
                [
                    item.width == width and item.height == height
                    for item in self.catalog
                ],
                dtype=bool,
            )
            for width in range(1, 4)
            for height in range(1, 4)
        }

    def _aggregate(
        self,
        mask: np.ndarray,
        *,
        world_counts: np.ndarray | None = None,
    ):
        population = self.world_counts if world_counts is None else world_counts
        selected = population[:, mask].astype(np.int64)
        return (
            selected.sum(axis=1),
            selected @ self.values[mask],
            selected @ self.areas[mask].astype(np.int64),
        )

    @staticmethod
    def _floor_average(total: np.ndarray, count: np.ndarray) -> np.ndarray:
        return np.divide(
            total,
            count,
            out=np.full(total.shape, -1, dtype=np.float64),
            where=count > 0,
        ).astype(np.int64)

    @staticmethod
    def _floor_area_average_hundredths(
        total: np.ndarray, count: np.ndarray
    ) -> np.ndarray:
        numerator = total * 100
        return np.divide(
            numerator,
            count,
            out=np.full(total.shape, -1, dtype=np.float64),
            where=count > 0,
        ).astype(np.int64)

    def _apply_event(
        self,
        mask: np.ndarray,
        semantics: Mapping,
        *,
        world_counts: np.ndarray | None = None,
        total_counts: np.ndarray | None = None,
        total_values: np.ndarray | None = None,
        total_areas: np.ndarray | None = None,
    ) -> np.ndarray:
        population = self.world_counts if world_counts is None else world_counts
        population_counts = self.total_counts if total_counts is None else total_counts
        population_values = self.total_values if total_values is None else total_values
        population_areas = self.total_areas if total_areas is None else total_areas
        effect = str(semantics.get("effect") or "unparsed")
        if effect in INFORMATIONAL_EFFECTS:
            return mask
        quality = str(semantics.get("quality") or "")
        width = int(semantics.get("width") or 0)
        height = int(semantics.get("height") or 0)
        if quality:
            group_mask = self.quality_masks.get(quality)
        elif width and height:
            group_mask = self.size_masks.get((width, height))
        else:
            group_mask = None
        if group_mask is not None:
            count, value, area = self._aggregate(group_mask, world_counts=population)
        else:
            count, value, area = (
                population_counts,
                population_values,
                population_areas,
            )
        if effect in {"quality_item_count", "size_item_count", "total_item_count"}:
            return mask & (count == int(semantics["observed_count"]))
        if effect in {"quality_total_value", "size_total_value"}:
            return mask & (value == int(semantics["observed_value"]))
        if effect in {"quality_average_value", "size_average_value"}:
            observed = int(semantics["observed_value"])
            return mask & (self._floor_average(value, count) == observed)
        if effect in {
            "quality_total_area",
            "size_total_area",
            "total_occupied_cells",
        }:
            return mask & (area == int(semantics["observed_cells"]))
        if effect in {
            "quality_average_area",
            "size_average_area",
            "average_item_area",
        }:
            observed = int(
                math.floor(float(semantics["observed_average_cells"]) * 100 + 1e-9)
            )
            return mask & (self._floor_area_average_hundredths(area, count) == observed)
        if effect == "max_item_value":
            present = population > 0
            maximum = np.where(present, self.values, -1).max(axis=1)
            return mask & (maximum == int(semantics["observed_value"]))
        if effect == "max_value_per_cell":
            per_cell = self.values // self.areas
            present = population > 0
            maximum = np.where(present, per_cell, -1).max(axis=1)
            return mask & (maximum == int(semantics["observed_value"]))
        if effect == "highest_quality":
            observed_quality = str(semantics["observed_quality"])
            observed_index = QUALITY_ORDER.index(observed_quality)
            quality_present = np.column_stack(
                [
                    population[:, quality_mask].sum(axis=1) > 0
                    for quality_mask in self.quality_masks.values()
                ]
            )
            highest = np.where(
                quality_present,
                np.arange(len(QUALITY_ORDER)),
                -1,
            ).max(axis=1)
            return mask & (highest == observed_index)
        return mask

    def _apply_visible_items(
        self,
        mask: np.ndarray,
        items: Iterable[Mapping],
        diagnostics: list[str] | None = None,
        *,
        world_counts: np.ndarray | None = None,
        total_counts: np.ndarray | None = None,
    ) -> np.ndarray:
        population = self.world_counts if world_counts is None else world_counts
        population_counts = self.total_counts if total_counts is None else total_counts
        identity_minimum: Counter[str] = Counter()
        quality_minimum: Counter[str] = Counter()
        size_minimum: Counter[tuple[int, int]] = Counter()
        joint_minimum: Counter[tuple[str, int, int]] = Counter()
        minimum_visible_item_count = 0
        for item in items:
            human_locked = bool(item.get("human_locked"))
            confidence = float(item.get("confidence") or 0.0)
            if not human_locked and confidence < 0.85:
                continue
            minimum_visible_item_count += 1
            spatial = str(item.get("spatial") or "top_left")
            catalog_id = (
                str(item.get("catalog_id") or "") if spatial == "complete" else ""
            )
            quality = str(item.get("quality") or "")
            width = int(item.get("width") or 0)
            height = int(item.get("height") or 0)
            if catalog_id:
                identity_minimum[catalog_id] += 1
            if quality:
                quality_minimum[quality] += 1
            if spatial in {"outline", "complete"} and width and height:
                size_minimum[(width, height)] += 1
            if spatial in {"outline", "complete"} and quality and width and height:
                joint_minimum[(quality, width, height)] += 1

        def constrain(condition: np.ndarray, label: str) -> None:
            nonlocal mask
            had_candidates = bool(np.any(mask))
            mask = mask & condition
            if diagnostics is not None and had_candidates and not np.any(mask):
                diagnostics.append(label)

        constrain(
            population_counts >= minimum_visible_item_count,
            f"map_conflict:visible_count:{minimum_visible_item_count}",
        )
        for catalog_id, minimum in identity_minimum.items():
            index = self.index_by_id.get(catalog_id)
            if index is None:
                constrain(
                    np.zeros_like(mask),
                    f"map_conflict:unknown_identity:{catalog_id}",
                )
                continue
            constrain(
                population[:, index] >= minimum,
                f"map_conflict:identity:{catalog_id}:{minimum}",
            )
        for quality, minimum in quality_minimum.items():
            group = self.quality_masks.get(quality)
            if group is None:
                constrain(
                    np.zeros_like(mask),
                    f"map_conflict:unknown_quality:{quality}",
                )
                continue
            constrain(
                population[:, group].sum(axis=1) >= minimum,
                f"map_conflict:quality:{quality}:{minimum}",
            )
        for size, minimum in size_minimum.items():
            group = self.size_masks.get(size)
            if group is None:
                constrain(
                    np.zeros_like(mask),
                    f"map_conflict:unknown_size:{size[0]}x{size[1]}",
                )
                continue
            constrain(
                population[:, group].sum(axis=1) >= minimum,
                f"map_conflict:size:{size[0]}x{size[1]}:{minimum}",
            )
        for (quality, width, height), minimum in joint_minimum.items():
            group = self.quality_masks[quality] & self.size_masks[(width, height)]
            constrain(
                population[:, group].sum(axis=1) >= minimum,
                (f"map_conflict:quality_size:{quality}:{width}x{height}:{minimum}"),
            )
        return mask

    def _filter_population(
        self,
        snapshot: ObservationSnapshot,
        world_counts: np.ndarray,
        *,
        issues: list[str] | None = None,
        diagnostics: list[str] | None = None,
    ) -> np.ndarray:
        population = np.asarray(world_counts, dtype=np.int16)
        total_counts = population.sum(axis=1, dtype=np.int64)
        total_values = population @ self.values
        total_areas = population @ self.areas.astype(np.int64)
        mask = np.ones(len(population), dtype=bool)
        current_events = [
            event
            for event in snapshot.events
            if int(event["round_number"]) <= snapshot.round_number
        ]
        if issues is not None and len(current_events) < snapshot.round_number * 2:
            issues.append("missing_event")
        for event in current_events:
            semantics = event.get("semantics") or {}
            if not semantics.get("parsed"):
                if issues is not None:
                    issues.append(
                        f"unparsed_event:R{event['round_number']}:{event['kind']}"
                    )
                continue
            updated = self._apply_event(
                mask,
                semantics,
                world_counts=population,
                total_counts=total_counts,
                total_values=total_values,
                total_areas=total_areas,
            )
            if diagnostics is not None and np.any(mask) and not np.any(updated):
                diagnostic = (
                    "event_conflict:"
                    f"R{event['round_number']}:"
                    f"{event['kind']}:"
                    f"{semantics.get('effect') or 'unparsed'}"
                )
                observed = next(
                    (
                        semantics.get(key)
                        for key in (
                            "observed_count",
                            "observed_value",
                            "observed_cells",
                            "observed_average_cells",
                            "observed_quality",
                        )
                        if semantics.get(key) is not None
                    ),
                    None,
                )
                if observed is not None:
                    diagnostic += f":{observed}"
                diagnostics.append(diagnostic)
            mask = updated
        return self._apply_visible_items(
            mask,
            snapshot.map_items,
            diagnostics,
            world_counts=population,
            total_counts=total_counts,
        )

    @staticmethod
    def _conditioning_signature(snapshot: ObservationSnapshot) -> str:
        payload = {
            "round_number": snapshot.round_number,
            "map_rows": snapshot.map_rows,
            "map_height_exact": snapshot.map_height_exact,
            "completeness_confirmed": snapshot.completeness_confirmed,
            "events": [
                {
                    "round_number": event.get("round_number"),
                    "kind": event.get("kind"),
                    "semantics": event.get("semantics") or {},
                    "confidence": event.get("confidence"),
                    "human_locked": event.get("human_locked"),
                }
                for event in snapshot.events
                if int(event["round_number"]) <= snapshot.round_number
            ],
            "map_items": [
                {
                    key: item.get(key)
                    for key in (
                        "row",
                        "column",
                        "width",
                        "height",
                        "quality",
                        "catalog_id",
                        "confidence",
                        "spatial",
                        "human_locked",
                        "first_seen_round",
                    )
                }
                for item in snapshot.map_items
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _conditioned_resample(
        self,
        snapshot: ObservationSnapshot,
    ) -> tuple[np.ndarray, int]:
        if self._world_model is None or self.conditional_resample_max_draws <= 0:
            return np.empty((0, len(self.catalog)), dtype=np.int16), 0
        signature = self._legacy_conditioning_signature(snapshot)
        cached = self._conditional_cache.get(signature)
        if cached is not None:
            self._conditional_cache.move_to_end(signature)
            return cached

        rng = random.Random(int(signature[:16], 16) ^ 20260804)
        accepted: list[np.ndarray] = []
        accepted_count = 0
        draws = 0
        while (
            draws < self.conditional_resample_max_draws
            and accepted_count < self.conditional_resample_target
        ):
            batch_size = min(
                self.conditional_resample_batch_size,
                self.conditional_resample_max_draws - draws,
            )
            batch = np.asarray(
                [self._world_model.sample_prior_counts(rng) for _ in range(batch_size)],
                dtype=np.int16,
            )
            draws += batch_size
            mask = self._filter_population(snapshot, batch)
            if np.any(mask):
                selected = batch[mask]
                accepted.append(selected)
                accepted_count += len(selected)
        compatible = (
            np.concatenate(accepted, axis=0)
            if accepted
            else np.empty((0, len(self.catalog)), dtype=np.int16)
        )
        result = (compatible, draws)
        self._conditional_cache[signature] = result
        self._conditional_cache.move_to_end(signature)
        while len(self._conditional_cache) > 8:
            self._conditional_cache.popitem(last=False)
        return result

    @staticmethod
    def _legacy_conditioning_signature(snapshot: ObservationSnapshot) -> str:
        """Keep released rejection draws reproducible on the same evidence.

        The legacy filter does not consume layout/history. The experimental
        sampler uses the full conditioning signature, including those fields.
        """
        payload = {
            "round_number": snapshot.round_number,
            "events": [
                {
                    "round_number": event.get("round_number"),
                    "kind": event.get("kind"),
                    "semantics": event.get("semantics") or {},
                }
                for event in snapshot.events
                if int(event["round_number"]) <= snapshot.round_number
            ],
            "map_items": [
                {
                    key: item.get(key)
                    for key in (
                        "row",
                        "column",
                        "width",
                        "height",
                        "quality",
                        "catalog_id",
                        "confidence",
                        "spatial",
                        "human_locked",
                    )
                }
                for item in snapshot.map_items
            ],
        }
        return hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    def predict(self, snapshot: ObservationSnapshot) -> PredictionResult:
        if (
            self.conditional_sampling
            and self.world_model_version == "world_model_v2"
            and self._world_model is not None
            and hasattr(self._world_model, "log_probability")
        ):
            return self._predict_conditional(snapshot)
        issues = []
        if (
            self.empirical_prior_diagnostics["verified_worlds"]
            < self.empirical_world_count
        ):
            issues.append("empirical_prior_review_unverified")
        diagnostics: list[str] = []
        mask = self._filter_population(
            snapshot,
            self.world_counts,
            issues=issues,
            diagnostics=diagnostics,
        )
        compatible_counts = self.world_counts[mask]
        resampled = False
        resample_draws = 0
        if compatible_counts.size == 0:
            compatible_counts, resample_draws = self._conditioned_resample(snapshot)
            if compatible_counts.size:
                resampled = True
                diagnostics = [
                    *(f"particle_collapse:{value}" for value in diagnostics),
                    f"conditioned_resample:{resample_draws}:{len(compatible_counts)}",
                ]
                issues.append("conditioned_resample_provisional")
            elif resample_draws:
                diagnostics.append(f"conditioned_resample_exhausted:{resample_draws}")
        compatible = compatible_counts.astype(np.int64) @ self.values
        if compatible.size == 0:
            status = "constraint_conflict"
            issues.append("no_compatible_public_world")
            values = (None, None, None, None, None)
        else:
            quantiles = np.quantile(compatible, [0.10, 0.50, 0.90], method="nearest")
            values = (
                int(quantiles[0]),
                int(quantiles[1]),
                int(quantiles[2]),
                int(compatible.min()),
                int(compatible.max()),
            )
            if resampled:
                status = "provisional_conditioned_resample"
            elif issues:
                status = "provisional_event_ocr"
            elif not snapshot.completeness_confirmed:
                status = "provisional_map_review"
                issues.append("map_completeness_not_confirmed")
            elif compatible.size < 10:
                status = "provisional_low_support"
                issues.append("compatible_worlds_below_10")
            else:
                status = "ready_research_estimate"
        return PredictionResult(
            revision=snapshot.revision,
            status=status,
            compatible_worlds=int(compatible.size),
            p10=values[0],
            p50=values[1],
            p90=values[2],
            minimum=values[3],
            maximum=values[4],
            actionable=False,
            issues=tuple(dict.fromkeys(issues)),
            diagnostics=tuple(dict.fromkeys(diagnostics)),
            contract=(
                "world_model_v2_conditioned_rejection_resample_uncalibrated"
                if resampled
                else (
                    f"public_empirical_plus_world_model_prior_{self.world_model_version}_uncalibrated"
                    if self.prior_world_count
                    else "empirical_compatible_worlds_uncalibrated"
                )
            ),
        )

    def _predict_conditional(self, snapshot: ObservationSnapshot) -> PredictionResult:
        from dataclasses import replace
        from .conditional_sampler import ConditionalWorldSampler

        signature = self._conditioning_signature(snapshot)
        if signature in self._posterior_cache:
            self._posterior_cache.move_to_end(signature)
            return replace(self._posterior_cache[signature], revision=snapshot.revision)
        issues: list[str] = []
        # Gather observation diagnostics independently from finite-bank support.
        self._filter_population(
            snapshot, np.empty((0, len(self.catalog)), dtype=np.int16), issues=issues
        )
        sampler = ConditionalWorldSampler(
            self._world_model,
            self.catalog,
            budget_seconds=self.conditional_budget_seconds,
        )
        retry_indices = getattr(self, "_conditional_retry_indices", None)
        if retry_indices is None:
            self._conditional_retry_indices = retry_indices = OrderedDict()
        retry_index = retry_indices.get(signature, 0)
        # Retry zero preserves the exact first-run seed used by frozen replays.
        seed = (
            int(signature[:16], 16) + retry_index * _CONDITIONAL_RETRY_SEED_STEP
        ) & _CONDITIONAL_RETRY_INDEX_MASK
        # Advance before sampling so exceptions are retryable with a fresh seed.
        # Keep both the signature ledger and each counter bounded. A wrap skips
        # zero, which is reserved for a signature's first retained invocation.
        retry_indices[signature] = 1 + retry_index % _CONDITIONAL_RETRY_INDEX_MASK
        retry_indices.move_to_end(signature)
        while len(retry_indices) > _CONDITIONAL_RETRY_SIGNATURE_LIMIT:
            retry_indices.popitem(last=False)
        samples = sampler.sample(
            snapshot,
            rng=random.Random(seed),
            filter_population=self._filter_numeric_population,
            prior_population=self._prior_counts,
        )
        samples.diagnostics.update(
            sampling_attempt=retry_index + 1,
            sampling_retry_index=retry_index,
            sampling_rng_seed=seed,
        )
        samples.diagnostics["empirical_prior"] = self.empirical_prior_diagnostics
        counts = samples.counts
        totals = (
            counts.astype(np.int64) @ self.values if counts.size else np.asarray([])
        )
        values = [None] * 5
        if len(totals):
            values = [
                *map(int, np.quantile(totals, [0.1, 0.5, 0.9], method="nearest")),
                int(totals.min()),
                int(totals.max()),
            ]
        if samples.diagnostics.get("numeric_infeasible"):
            status = "constraint_conflict"
            issues.append("numeric_constraints_infeasible")
        elif any(
            note.startswith("unsupported:") for note in samples.diagnostics["notes"]
        ):
            status = "unsupported_evidence"
            issues.append("unsupported_evidence")
        elif not samples.converged:
            status = "provisional_conditional_sampling"
            issues.append("posterior_not_converged")
        elif issues:
            status = "provisional_event_ocr"
        elif not snapshot.completeness_confirmed:
            status = "provisional_map_review"
            issues.append("map_completeness_not_confirmed")
        else:
            status = "ready_research_estimate"
        result = PredictionResult(
            revision=snapshot.revision,
            status=status,
            compatible_worlds=len(totals),
            p10=values[0],
            p50=values[1],
            p90=values[2],
            minimum=values[3],
            maximum=values[4],
            actionable=False,
            issues=tuple(dict.fromkeys(issues)),
            contract="conditional_ordered_world_posterior_uncalibrated",
            diagnostics=tuple(samples.diagnostics["notes"]),
            effective_sample_size=samples.effective_sample_size,
            posterior_converged=samples.converged,
            posterior_method=str(samples.diagnostics.get("method")),
            conditional_diagnostics=samples.diagnostics,
        )
        # A finite-budget failure is retryable; it must not permanently freeze
        # the same evidence at the first unlucky initialization/timeout.
        if samples.converged or samples.diagnostics.get("numeric_infeasible"):
            self._posterior_cache[signature] = result
            retry_indices.pop(signature, None)
        while len(self._posterior_cache) > 8:
            self._posterior_cache.popitem(last=False)
        return result

    def _filter_numeric_population(self, snapshot, population):
        """Exact vectorized event filter; spatial evidence uses world_layout.

        The legacy visible-count filter is intentionally absent: duplicate
        anchors and partially overlapping candidate groups need concrete item
        matching, not independent lower-bound counts.
        """
        mask = np.ones(len(population), dtype=bool)
        total_counts = population.sum(axis=1, dtype=np.int64)
        total_values = population.astype(np.int64) @ self.values
        total_areas = population.astype(np.int64) @ self.areas.astype(np.int64)
        for event in snapshot.events:
            if int(event["round_number"]) <= snapshot.round_number:
                mask = self._apply_event(
                    mask,
                    event.get("semantics") or {},
                    world_counts=population,
                    total_counts=total_counts,
                    total_values=total_values,
                    total_areas=total_areas,
                )
        return mask
