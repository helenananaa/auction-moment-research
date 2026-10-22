from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .models import CatalogItem
from .observations import ObservationSnapshot
from .predictor import EmpiricalWorldPredictor, PredictionResult
from .reveal_likelihood import build_reveal_observations


V6_CANDIDATE_ID = "protocol-round-residual-v6-72a08074c3d8af81"
ADAPTER_SCHEMA = "ocr-visible-pre-bid-adapter-v1"
REVEAL_ATTRIBUTION_SCHEMA = "ocr-first-reveal-single-source-v1"
MINIMUM_FUSION_EFFECTIVE_SAMPLES = 32
# These are conservative engineering gates, not fitted or calibrated weights.
# More conditioning events can reduce support; they are not evidence of trust.
FUSION_WEIGHT_BY_EFFECTIVE_SAMPLES = (
    (256, 0.40),
    (128, 0.30),
    (64, 0.20),
    (MINIMUM_FUSION_EFFECTIVE_SAMPLES, 0.10),
)
FUSION_POLICY = "converged_effective_support_cap_v1"
RELEASED_FUSION_POLICY = "released_event_count_heuristic_uncalibrated"
DEFAULT_POINT_POLICY = "v6_only_pending_fusion_qualification"
EXACT_POSTERIOR_METHOD = "exact_constraint_importance_sampling"
# The 1/value-weighted posterior median minimises expected absolute % error.
EXACT_POINT_POLICY = "mape_optimal_weighted_median"
DEFAULT_EXACT_BUDGET_SECONDS = 8.0
DEFAULT_EXACT_LANES = 4
# A decision cut by the budget keeps running in the background this long.
DEFAULT_EXACT_REFINE_SECONDS = 45.0
DEFAULT_BID_SAFETY_FACTOR = 0.90
ROUND_WINNING_MULTIPLIERS = {1: 2.0, 2: 1.6, 3: 1.3, 4: 1.1, 5: 1.0}
STRONG_CONDITIONING_EFFECTS = frozenset(
    {
        "quality_total_value",
        "quality_item_count",
        "total_occupied_cells",
        "max_item_value",
        "max_value_per_cell",
    }
)


class LatestModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdaptedDecision:
    decision: dict
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    strong_condition_count: int

    @property
    def accepted(self) -> bool:
        return not self.issues


class OcrDecisionAdapter:
    """Translate reviewed screen evidence into the frozen v6 input plane.

    This adapter never invents packet indexes, event codes, exact identities,
    settlement fields, or post-bid outcomes. OCR observations remain explicitly
    marked as thresholded OCR; only user-locked values are human-confirmed.
    """

    def __init__(
        self,
        catalog: Sequence[CatalogItem],
        *,
        minimum_ocr_confidence: float = 0.85,
    ) -> None:
        threshold = float(minimum_ocr_confidence)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_ocr_confidence must be in [0, 1]")
        self.minimum_ocr_confidence = threshold
        self.catalog = {item.catalog_id: item for item in catalog}

    def adapt(self, snapshot: ObservationSnapshot) -> AdaptedDecision:
        issues: list[str] = []
        warnings: list[str] = []
        if not snapshot.pre_bid_confirmed:
            issues.append("pre_bid_not_confirmed")
        elif snapshot.pre_bid_source == "visual_state_machine":
            warnings.append("pre_bid_visual_state_machine")
        if snapshot.map_rows is None:
            issues.append("exact_map_rows_not_confirmed")
        elif not snapshot.map_height_exact:
            if snapshot.map_rows_source == "automatic_scroll_estimate":
                warnings.append("map_rows_auto_estimated")
            else:
                issues.append("exact_map_rows_not_confirmed")
        if not snapshot.completeness_confirmed:
            issues.append("visible_map_not_reviewed")
        elif snapshot.completeness_source == "automatic_scroll_scan":
            warnings.append("visible_map_auto_scanned_not_human_reviewed")

        events: list[dict] = []
        seen_event_keys: set[tuple[int, str]] = set()
        for raw in snapshot.events:
            try:
                round_number = int(raw["round_number"])
            except (KeyError, TypeError, ValueError):
                issues.append("invalid_event_round")
                continue
            if round_number > snapshot.round_number:
                continue
            kind = str(raw.get("kind") or "")
            key = (round_number, kind)
            if kind not in {"public", "personal"} or key in seen_event_keys:
                issues.append(f"invalid_event_key:R{round_number}:{kind or 'missing'}")
                continue
            seen_event_keys.add(key)
            semantics = copy.deepcopy(raw.get("semantics") or {})
            if semantics.get("parsed") is not True:
                issues.append(f"unparsed_event:R{round_number}:{kind}")
                continue
            human_locked = bool(raw.get("human_locked"))
            confidence = float(raw.get("confidence") or 0.0)
            if not human_locked and (
                not math.isfinite(confidence)
                or confidence < self.minimum_ocr_confidence
            ):
                issues.append(f"low_confidence_event:R{round_number}:{kind}")
                continue
            semantics["protocol_source"] = (
                "human_confirmed" if human_locked else "ocr_thresholded"
            )
            events.append(
                {
                    "round": round_number,
                    "kind": kind,
                    "ocr_confidence": 1.0 if human_locked else confidence,
                    "semantics": semantics,
                }
            )

        expected_keys = {
            (round_number, kind)
            for round_number in range(1, snapshot.round_number + 1)
            for kind in ("public", "personal")
        }
        missing = sorted(expected_keys - seen_event_keys)
        issues.extend(
            f"missing_event:R{round_number}:{kind}" for round_number, kind in missing
        )

        visible_items: list[dict] = []
        occupied_cells: set[tuple[int, int]] = set()
        for index, raw in enumerate(snapshot.map_items, start=1):
            confidence = float(raw.get("confidence") or 0.0)
            human_locked = bool(raw.get("human_locked"))
            if not human_locked and (
                not math.isfinite(confidence)
                or confidence < self.minimum_ocr_confidence
            ):
                issues.append(
                    f"low_confidence_map_item:{raw.get('row')}:{raw.get('column')}"
                )
                continue
            try:
                row = int(raw["row"])
                column = int(raw["column"])
            except (KeyError, TypeError, ValueError):
                issues.append(f"invalid_map_position:{index}")
                continue
            if row < 0 or column not in range(10):
                issues.append(f"invalid_map_position:{row}:{column}")
                continue
            width = int(raw.get("width") or 0)
            height = int(raw.get("height") or 0)
            spatial = str(raw.get("spatial") or "top_left")
            if spatial not in {"top_left", "outline", "complete"}:
                issues.append(f"invalid_map_spatial:{row}:{column}")
                continue
            if (width > 0) != (height > 0):
                issues.append(f"invalid_map_size:{row}:{column}")
                continue
            known_size = None
            if spatial in {"outline", "complete"} and width > 0 and height > 0:
                if width > 3 or height > 3:
                    issues.append(f"invalid_map_size:{row}:{column}")
                    continue
                if column + width > 10 or (
                    snapshot.map_rows is not None and row + height > snapshot.map_rows
                ):
                    issues.append(f"map_item_out_of_bounds:{row}:{column}")
                    continue
                cells = {
                    (item_row, item_column)
                    for item_row in range(row, row + height)
                    for item_column in range(column, column + width)
                }
                if occupied_cells & cells:
                    issues.append(f"overlapping_map_item:{row}:{column}")
                    continue
                occupied_cells.update(cells)
                known_size = {"width": width, "height": height}

            catalog_id = str(raw.get("catalog_id") or "")
            catalog_item = self.catalog.get(catalog_id)
            # A detector "complete" class includes identity classification.
            # Thresholded OCR is accepted by default; manual promotion still
            # requires the UI to attach an explicit catalog id.
            identity_known = bool(spatial == "complete" and catalog_item is not None)
            quality = str(raw.get("quality") or "") or None
            if quality not in {None, "白", "蓝", "紫", "金", "彩"}:
                issues.append(f"invalid_map_quality:{row}:{column}")
                continue
            if identity_known:
                if quality is not None and quality != catalog_item.quality:
                    issues.append(f"identity_quality_conflict:{row}:{column}")
                    continue
                if known_size is not None and (
                    width != catalog_item.width or height != catalog_item.height
                ):
                    issues.append(f"identity_size_conflict:{row}:{column}")
                    continue
                quality = catalog_item.quality
                if known_size is None:
                    width = catalog_item.width
                    height = catalog_item.height
                    if column + width > 10 or (
                        snapshot.map_rows is not None
                        and row + height > snapshot.map_rows
                    ):
                        issues.append(f"map_item_out_of_bounds:{row}:{column}")
                        continue
                    cells = {
                        (item_row, item_column)
                        for item_row in range(row, row + height)
                        for item_column in range(column, column + width)
                    }
                    if occupied_cells & cells:
                        issues.append(f"overlapping_map_item:{row}:{column}")
                        continue
                    occupied_cells.update(cells)
                    known_size = {"width": width, "height": height}
            item = {
                "track_id": f"screen-r{row}-c{column}",
                "position": {"row": row, "column": column},
                "known_size": known_size,
                "quality": quality,
                "quality_validation": (
                    "human_confirmed" if human_locked and quality else "ocr_thresholded"
                ),
                "spatial_knowledge": (
                    "complete"
                    if identity_known
                    else "outline"
                    if spatial in {"outline", "complete"}
                    else "top_left"
                ),
                "evidence_confidence": 1.0 if human_locked else confidence,
                "quality_confidence": 1.0 if human_locked else confidence,
                "identity_known": identity_known,
                "identity": (
                    {
                        "catalog_id": catalog_id,
                        "known_value": int(catalog_item.value),
                        "match_confidence_status": (
                            "human_confirmed" if human_locked else "ocr_thresholded"
                        ),
                    }
                    if identity_known
                    else {}
                ),
                "first_seen_round": int(
                    raw.get("first_seen_round") or snapshot.round_number
                ),
                "history": [],
            }
            visible_items.append(item)

        # A position's first-seen round alone is not an attribute history.
        # Reuse the conservative first-reveal attribution contract: a complete
        # sample, no earlier revealed pool, and exactly one reveal source.
        # The frozen v6 feature helpers consume this history separately from
        # the visible-item lower bound and retain the original OCR provenance.
        reveals, reveal_diagnostics = build_reveal_observations(
            snapshot, tuple(self.catalog.values())
        )
        certified_reveals: list[dict] = []
        for reveal in reveals:
            if reveal.effect not in {
                "reveal_identity_random",
                "reveal_position_quality_random",
            }:
                continue
            certified = {
                "round": reveal.round_number,
                "source_effect": reveal.effect,
                "event_kind": reveal.event_kind,
                "sample_size": reveal.sample_size,
                "attribution_schema": REVEAL_ATTRIBUTION_SCHEMA,
            }
            matched = [
                item
                for item in visible_items
                if item["first_seen_round"] == reveal.round_number
            ]
            if len(matched) != reveal.sample_size:
                continue
            for item in matched:
                item["history"] = [dict(certified)]
            certified_reveals.append(certified)

        map_rows = int(snapshot.map_rows or 0)
        automatic_rows = bool(
            not snapshot.map_height_exact
            and snapshot.map_rows_source == "automatic_scroll_estimate"
        )
        decision = {
            "adapter_schema_version": ADAPTER_SCHEMA,
            "round": int(snapshot.round_number),
            "events": events,
            "visible_items": visible_items,
            "reveal_attribution": {
                "schema_version": REVEAL_ATTRIBUTION_SCHEMA,
                "certified_events": certified_reveals,
                "diagnostics": list(reveal_diagnostics),
            },
            "observed_map": {
                "rows": map_rows,
                "columns": 10,
                "height_exact": bool(snapshot.map_height_exact),
                "bottom_visible": bool(snapshot.map_height_exact),
                "canvas_bottom_visible": bool(snapshot.map_height_exact),
                "source": (
                    "ocr_visible_map_estimated"
                    if automatic_rows
                    else "ocr_visible_map_confirmed"
                ),
                "height_proof": {
                    "schema_version": "hidden-map-bottom-proof-v1",
                    "status": (
                        "estimated"
                        if automatic_rows
                        else "proven"
                        if snapshot.map_height_exact
                        else "missing"
                    ),
                    "rows": map_rows,
                    "proof_source": (
                        "automatic_scroll_alignment"
                        if automatic_rows
                        else "human_screen_confirmation"
                    ),
                },
            },
            "map_recognition": {
                "completeness": {
                    "status": (
                        "automatic_scroll_scan"
                        if snapshot.completeness_source == "automatic_scroll_scan"
                        else "human_reviewed_visible_screen"
                    ),
                    "exact_counts_safe": False,
                    "checks": {
                        "visible_map_reviewed": bool(snapshot.completeness_confirmed),
                        "map_height_human_confirmed": bool(snapshot.map_height_exact),
                        "pre_bid_human_confirmed": bool(
                            snapshot.pre_bid_source == "human_confirmed"
                        ),
                        "pre_bid_visual_state_machine": bool(
                            snapshot.pre_bid_source == "visual_state_machine"
                        ),
                    },
                    "exact_quality_counts": {},
                    "exact_size_counts": {},
                }
            },
            "native_evidence": {
                "mode": "ocr_visible_pre_bid_v1",
                "pre_bid_proven": bool(snapshot.pre_bid_confirmed),
                "ocr_used_for_live_features": True,
                "settlement_or_final_truth_in_features": False,
            },
        }
        strong_count = len(
            {
                str((event.get("semantics") or {}).get("effect") or "")
                for event in events
            }
            & STRONG_CONDITIONING_EFFECTS
        )
        return AdaptedDecision(
            decision=decision,
            issues=tuple(dict.fromkeys(issues)),
            warnings=tuple(dict.fromkeys(warnings)),
            strong_condition_count=strong_count,
        )


def _geometric_blend(first: float, second: float, weight: float) -> int:
    if first <= 0 or second <= 0:
        raise LatestModelError("model predictions must be positive")
    return round(
        math.exp(
            (1.0 - float(weight)) * math.log(first) + float(weight) * math.log(second)
        )
    )


def _generation_weight(world: PredictionResult) -> tuple[float, tuple[str, ...]]:
    """Gate fusion on sampling evidence, never on event count or solver rows.

    An ESS must already account for importance weights and chain dependence.
    Neither duplicated particles nor CP-SAT solution counts qualify as ESS.
    """
    reasons: list[str] = []
    quantiles = (world.p10, world.p50, world.p90)
    try:
        valid_quantiles = all(
            value is not None and math.isfinite(float(value)) and value > 0
            for value in quantiles
        ) and float(quantiles[0]) <= float(quantiles[1]) <= float(quantiles[2])
    except (TypeError, ValueError, OverflowError):
        valid_quantiles = False
    if not valid_quantiles:
        reasons.append("v2_invalid_or_missing_quantiles")
    evidence_flags = (world.status, *world.issues)
    if any(
        marker in str(flag).lower()
        for flag in evidence_flags
        for marker in ("conflict", "failed", "unsupported", "unparsed", "invalid")
    ):
        reasons.append("v2_evidence_or_sampling_failure")
    diagnostics = world.conditional_diagnostics or {}
    if any(
        diagnostics.get(flag) is True
        for flag in ("evidence_conflict", "sampling_failed", "truncated")
    ):
        reasons.append("v2_evidence_or_sampling_failure")
    if world.posterior_converged is not True:
        reasons.append("v2_posterior_not_converged")
    try:
        effective_samples = float(world.effective_sample_size)
    except (TypeError, ValueError, OverflowError):
        effective_samples = float("nan")
    if (
        not math.isfinite(effective_samples)
        or effective_samples < MINIMUM_FUSION_EFFECTIVE_SAMPLES
        or world.compatible_worlds < MINIMUM_FUSION_EFFECTIVE_SAMPLES
    ):
        reasons.append("v2_effective_support_below_32")
    if reasons:
        return 0.0, tuple(dict.fromkeys(reasons))
    # A reported ESS above the retained sample count cannot raise the gate.
    support = min(effective_samples, float(world.compatible_worlds))
    return next(
        (weight, ())
        for threshold, weight in FUSION_WEIGHT_BY_EFFECTIVE_SAMPLES
        if support >= threshold
    )


def _released_generation_weight(world: PredictionResult, strong_count: int):
    """Preserve the released heuristic while the new sampler is experimental.

    This policy has no posterior-convergence claim. Its weak finite-bank
    support and uncalibrated weights remain explicit in the result contract.
    """
    try:
        quantiles = (world.p10, world.p50, world.p90)
        valid = all(
            v is not None and math.isfinite(float(v)) and v > 0 for v in quantiles
        )
        valid = valid and quantiles[0] <= quantiles[1] <= quantiles[2]
    except (ValueError, TypeError, OverflowError):
        valid = False
    if not valid or world.compatible_worlds < 1:
        return 0.0, ("v2_no_supported_finite_world",)
    if world.status in {"sampling_failed", "unsupported_evidence"}:
        return 0.0, ("v2_evidence_or_sampling_failure",)
    return min(0.8, 0.4 + 0.1 * int(strong_count)), ()


def manual_bid_advice(
    p10: int,
    round_number: int,
    *,
    safety_factor: float = DEFAULT_BID_SAFETY_FACTOR,
) -> dict:
    """Build a display-only value-side acquisition ceiling."""

    factor = float(safety_factor)
    if not 0.0 < factor <= 1.0:
        raise ValueError("bid safety factor must be in (0, 1]")
    multiplier = ROUND_WINNING_MULTIPLIERS.get(int(round_number))
    if multiplier is None:
        raise ValueError("round number must be in [1, 5]")
    recommended = max(1, math.floor(int(p10) * factor))
    return {
        "recommended_bid": recommended,
        "bid_safety_factor": factor,
        "required_winning_multiplier": multiplier,
        "maximum_opponent_bid": math.floor(recommended / multiplier),
        "bid_basis": "p10_times_safety_value_side_ceiling",
    }


class LatestV6V2Predictor:
    """Use repaired v6 by default; retain v2 diagnostics and opt-in fusion."""

    def __init__(
        self,
        *,
        v6_model_path: Path,
        world_model_v2_path: Path,
        treasures_csv: Path,
        catalog: Sequence[CatalogItem],
        minimum_ocr_confidence: float = 0.85,
        prior_samples: int = 4096,
        conditional_sampling: bool = False,
        experimental_v2_fusion: bool = False,
        conditional_budget_seconds: float = 2.0,
        bid_safety_factor: float = DEFAULT_BID_SAFETY_FACTOR,
        exact_sampling: bool = False,
        exact_budget_seconds: float = DEFAULT_EXACT_BUDGET_SECONDS,
        exact_lanes: int = DEFAULT_EXACT_LANES,
        exact_refine_seconds: float = DEFAULT_EXACT_REFINE_SECONDS,
    ) -> None:
        try:
            import joblib

            self.v6_model = joblib.load(Path(v6_model_path))
        except Exception as exc:
            raise LatestModelError(f"v6 model load failed: {exc}") from exc
        if str(getattr(self.v6_model, "candidate_id", "")) != V6_CANDIDATE_ID:
            raise LatestModelError("v6 candidate id mismatch")
        if not 0.0 < float(bid_safety_factor) <= 1.0:
            raise ValueError("bid_safety_factor must be in (0, 1]")
        if (
            not math.isfinite(float(conditional_budget_seconds))
            or float(conditional_budget_seconds) <= 0.0
        ):
            raise ValueError("conditional_budget_seconds must be finite and positive")
        if exact_sampling and (conditional_sampling or experimental_v2_fusion):
            raise ValueError("exact sampling cannot be combined with another v2 mode")
        if not math.isfinite(float(exact_budget_seconds)) or float(exact_budget_seconds) <= 0.0:
            raise ValueError("exact_budget_seconds must be finite and positive")
        if int(exact_lanes) < 1:
            raise ValueError("exact_lanes must be at least 1")
        if not math.isfinite(float(exact_refine_seconds)) or float(exact_refine_seconds) < 0.0:
            raise ValueError("exact_refine_seconds must be finite and non-negative")
        self.bid_safety_factor = float(bid_safety_factor)
        self.conditional_sampling = bool(conditional_sampling)
        self.experimental_v2_fusion = bool(experimental_v2_fusion)
        self.exact_sampling = bool(exact_sampling)
        self.exact_budget_seconds = float(exact_budget_seconds)
        self.exact_refine_seconds = float(exact_refine_seconds)
        self.exact_engine = None
        # The last decision cut by the budget, kept so it can be refined.
        self._exact_cut = None
        self.adapter = OcrDecisionAdapter(
            catalog,
            minimum_ocr_confidence=minimum_ocr_confidence,
        )
        self.world_predictor = EmpiricalWorldPredictor(
            treasures_csv,
            catalog,
            world_model_path=world_model_v2_path,
            prior_samples=prior_samples,
            conditional_sampling=conditional_sampling,
            conditional_budget_seconds=conditional_budget_seconds,
        )
        if self.world_predictor.world_model_version != "world_model_v2":
            raise LatestModelError("world-model-v2 artifact required")
        if self.exact_sampling:
            # Worker processes load the same frozen v2 artifact and catalog.
            from .exact_lanes import ExactLanes

            self.exact_catalog = tuple(catalog)
            self.exact_engine = ExactLanes(
                Path(world_model_v2_path), Path(treasures_csv), lanes=int(exact_lanes),
            )

    def predict(self, snapshot: ObservationSnapshot) -> PredictionResult:
        adapted = self.adapter.adapt(snapshot)
        if not adapted.accepted:
            return PredictionResult(
                revision=snapshot.revision,
                status="ocr_adapter_blocked",
                compatible_worlds=0,
                p10=None,
                p50=None,
                p90=None,
                minimum=None,
                maximum=None,
                actionable=False,
                issues=adapted.issues,
                contract="ocr_adapted_v6_v2_uncalibrated",
                model_version=V6_CANDIDATE_ID,
            )
        try:
            v6 = self.v6_model.predict(adapted.decision)
        except Exception as exc:
            return PredictionResult(
                revision=snapshot.revision,
                status="v6_inference_failed",
                compatible_worlds=0,
                p10=None,
                p50=None,
                p90=None,
                minimum=None,
                maximum=None,
                actionable=False,
                issues=(
                    *adapted.warnings,
                    f"v6_error:{type(exc).__name__}:{exc}",
                ),
                contract="ocr_adapted_v6_v2_uncalibrated",
                model_version=V6_CANDIDATE_ID,
            )
        v6_p50 = int(v6["prediction"])
        v6_p10 = int(v6.get("p10") or v6_p50)
        v6_p90 = max(v6_p10, v6_p50, int(v6.get("p90") or v6_p50))
        # Exact sampling is its own explicit opt-in; a restored predictor
        # without the attribute keeps the default path.
        if getattr(self, "exact_engine", None) is not None:
            return self._exact_estimate(snapshot, adapted, v6_p10, v6_p50, v6_p90)
        try:
            world = self.world_predictor.predict(snapshot)
        except Exception as exc:
            world = PredictionResult(
                revision=snapshot.revision,
                status="sampling_failed",
                compatible_worlds=0,
                p10=None,
                p50=None,
                p90=None,
                minimum=None,
                maximum=None,
                actionable=False,
                issues=(f"v2_sampling_failed:{type(exc).__name__}:{exc}",),
            )
        # Missing attributes on a restored predictor do not authorize an
        # experimental mode. Both fusion paths require explicit opt-in.
        experimental = getattr(self, "conditional_sampling", False)
        released_fusion_opt_in = getattr(self, "experimental_v2_fusion", False)
        fusion_enabled = experimental or released_fusion_opt_in
        if not fusion_enabled:
            weight, gate_reasons = 0.0, ("v2_fusion_disabled_pending_validation",)
        elif experimental:
            weight, gate_reasons = _generation_weight(world)
        else:
            weight, gate_reasons = _released_generation_weight(
                world, adapted.strong_condition_count
            )
        posterior_metadata = {
            "effective_sample_size": world.effective_sample_size,
            "posterior_converged": world.posterior_converged,
            "posterior_method": world.posterior_method,
            "conditional_diagnostics": {
                **(world.conditional_diagnostics or {}),
                "fusion_policy": (
                    DEFAULT_POINT_POLICY if not fusion_enabled
                    else FUSION_POLICY if experimental else RELEASED_FUSION_POLICY
                ),
                "experimental_conditional_sampling": experimental,
                "experimental_v2_fusion": released_fusion_opt_in,
                "fusion_weight": weight,
                "fusion_gate_reasons": list(gate_reasons),
                "v2_diagnostic_result": {
                    "status": world.status,
                    "issues": list(world.issues),
                    "p10": world.p10,
                    "p50": world.p50,
                    "p90": world.p90,
                    "compatible_worlds": world.compatible_worlds,
                },
            },
        }
        if not fusion_enabled:
            return PredictionResult(
                revision=snapshot.revision,
                status="ocr_adapted_v6_default_estimate",
                compatible_worlds=world.compatible_worlds,
                p10=v6_p10,
                p50=v6_p50,
                p90=v6_p90,
                minimum=None,
                maximum=None,
                actionable=False,
                issues=tuple(dict.fromkeys((
                    *adapted.warnings,
                    "默认使用 v6；v2 仅供诊断，融合尚待独立验证",
                    "ocr_adapter_not_packet_equivalent",
                    "interval_uncalibrated",
                ))),
                contract="ocr_adapted_v6_default_with_v2_diagnostics_uncalibrated",
                v6_prediction=v6_p50,
                v2_prediction=world.p50,
                generation_weight=0.0,
                model_version=V6_CANDIDATE_ID,
                estimate_source="v6_default",
                diagnostics=(*world.diagnostics, "v2_diagnostics_only"),
                **posterior_metadata,
                **manual_bid_advice(
                    v6_p10, snapshot.round_number,
                    safety_factor=self.bid_safety_factor,
                ),
            )
        if weight == 0.0:
            proven_conflict = (world.conditional_diagnostics or {}).get(
                "numeric_infeasible"
            ) is True
            return PredictionResult(
                revision=snapshot.revision,
                status=(
                    "v6_fallback_v2_conditioning_conflict"
                    if proven_conflict
                    else "v6_fallback_v2_insufficient_posterior"
                ),
                compatible_worlds=world.compatible_worlds,
                p10=v6_p10,
                p50=v6_p50,
                p90=v6_p90,
                minimum=None,
                maximum=None,
                actionable=False,
                issues=tuple(
                    dict.fromkeys(
                        (
                            *adapted.warnings,
                            *world.issues,
                            *gate_reasons,
                            (
                                "v2_conflict_using_v6_fallback"
                                if proven_conflict
                                else "v2_unreliable_posterior_using_v6_fallback"
                            ),
                            "ocr_adapter_not_packet_equivalent",
                            "interval_uncalibrated",
                        )
                    )
                ),
                contract="ocr_adapted_v6_fallback_v2_gated_uncalibrated",
                v6_prediction=v6_p50,
                v2_prediction=world.p50,
                generation_weight=0.0,
                model_version=V6_CANDIDATE_ID,
                estimate_source="v6_fallback",
                diagnostics=(
                    *world.diagnostics,
                    *(f"fusion_gate:{reason}" for reason in gate_reasons),
                ),
                **posterior_metadata,
                **manual_bid_advice(
                    v6_p10,
                    snapshot.round_number,
                    safety_factor=self.bid_safety_factor,
                ),
            )
        p10 = _geometric_blend(v6_p10, int(world.p10), weight)
        p50 = _geometric_blend(v6_p50, int(world.p50), weight)
        p90 = max(
            p10,
            p50,
            _geometric_blend(v6_p90, int(world.p90), weight),
        )
        issues = tuple(
            dict.fromkeys(
                (
                    *adapted.warnings,
                    *world.issues,
                    "ocr_adapter_not_packet_equivalent",
                    "interval_uncalibrated",
                )
            )
        )
        return PredictionResult(
            revision=snapshot.revision,
            status="ocr_adapted_latest_model_estimate",
            compatible_worlds=world.compatible_worlds,
            p10=p10,
            p50=p50,
            p90=p90,
            minimum=world.minimum,
            maximum=world.maximum,
            actionable=False,
            issues=issues,
            contract="ocr_adapted_v6_plus_world_model_v2_uncalibrated",
            v6_prediction=v6_p50,
            v2_prediction=int(world.p50),
            generation_weight=weight,
            model_version=V6_CANDIDATE_ID,
            estimate_source="v6_v2_blend",
            diagnostics=world.diagnostics,
            **posterior_metadata,
            **manual_bid_advice(
                p10,
                snapshot.round_number,
                safety_factor=self.bid_safety_factor,
            ),
        )

    def prewarm(self, snapshot: ObservationSnapshot) -> bool:
        """Start the exact sampler on evidence known before the map scan ends.

        Only fills caches in the worker processes (pools, clue tables) and
        returns at once; the next prediction cancels it. False if not in
        exact mode, the evidence cannot be used, or the workers are busy.
        """
        engine = getattr(self, "exact_engine", None)
        if engine is None:
            return False
        from .exact_evidence import evidence_seed

        converted = self._exact_evidence(snapshot, prewarm=True)
        if not converted.usable:
            return False
        return engine.prewarm(converted.evidence, evidence_seed(snapshot))

    def _exact_evidence(self, snapshot: ObservationSnapshot, *, prewarm: bool = False):
        """Packet-capture decision records when the snapshot carries one, else the screen rules."""
        from .exact_evidence import evidence_from_decision, evidence_from_snapshot

        decision = getattr(snapshot, "packet_decision", None)
        if decision is not None:
            return evidence_from_decision(decision, self.exact_catalog)
        return evidence_from_snapshot(
            snapshot,
            self.exact_catalog,
            minimum_confidence=self.adapter.minimum_ocr_confidence,
            prewarm=prewarm,
        )

    def refine(self, revision: int) -> bool:
        """Rerun the last decision cut by the time budget, without waiting.

        Uses ``exact_refine_seconds`` and the caches the cut run filled; the
        next prewarm or prediction cancels it. False if that decision is not
        ``revision``, refinement is off, or the workers are busy.
        """
        import time

        from .exact_evidence import evidence_seed

        engine = getattr(self, "exact_engine", None)
        cut = getattr(self, "_exact_cut", None)
        if engine is None or cut is None or int(cut["snapshot"].revision) != int(revision):
            return False
        if getattr(self, "exact_refine_seconds", 0.0) <= 0.0:
            return False
        # A seed of its own: an independent run, not a repeat of the cut one.
        seed = evidence_seed(cut["snapshot"]) + 1
        if not engine.refine(cut["converted"].evidence, seed, time_budget=self.exact_refine_seconds):
            return False
        cut["refine_started"] = time.monotonic()
        return True

    def poll_refined(self) -> tuple[str, PredictionResult | None]:
        """("running", None) while refining; ("done", result) once it converged.

        ("done", None) when the longer run still fell short, ("gone", None)
        when nothing is being refined (cancelled by newer work, or failed).
        """
        import time

        engine = getattr(self, "exact_engine", None)
        cut = getattr(self, "_exact_cut", None)
        if engine is None or cut is None or "refine_started" not in cut:
            return "gone", None
        state, outcome = engine.poll_refine()
        if state == "running":
            return state, None
        self._exact_cut = None
        if state != "done" or outcome.status != "ok":
            return ("done" if state == "done" else "gone"), None
        diagnostics = dict(cut["diagnostics"])
        diagnostics.update(
            exact_refined=True,
            exact_refine_seconds=round(time.monotonic() - cut["refine_started"], 3),
            exact_cut_point=cut["point"],
        )
        return "done", self._exact_result(
            cut["snapshot"], cut["adapted"], *cut["v6"], cut["converted"], outcome, diagnostics,
        )

    def close(self) -> None:
        engine = getattr(self, "exact_engine", None)
        if engine is not None:
            engine.close()
            self.exact_engine = None

    def _exact_estimate(
        self,
        snapshot: ObservationSnapshot,
        adapted: AdaptedDecision,
        v6_p10: int,
        v6_p50: int,
        v6_p90: int,
    ) -> PredictionResult:
        """Experimental exact-constraint posterior; v6 when it cannot run."""
        import time

        from .exact_evidence import evidence_seed

        converted = self._exact_evidence(snapshot)
        packet = getattr(snapshot, "packet_decision", None) is not None
        diagnostics = {
            "experimental_exact_sampling": True,
            "exact_input": "packet_capture" if packet else "ocr_screen",
            "exact_lanes": self.exact_engine.lanes,
            "exact_budget_seconds": self.exact_budget_seconds,
            "exact_ignored_clues": list(converted.ignored),
            "exact_unsupported": list(converted.unsupported),
        }
        failure = None
        result = None
        self._exact_cut = None
        started = time.monotonic()
        if not converted.usable:
            failure = "exact_unsupported_evidence"
        else:
            try:
                result = self.exact_engine.sample(
                    converted.evidence,
                    evidence_seed(snapshot),
                    time_budget=self.exact_budget_seconds,
                )
            except Exception as exc:
                failure = "exact_sampling_failed"
                diagnostics["exact_error"] = f"{type(exc).__name__}: {exc}"
            else:
                if result.status == "no_support":
                    failure = "exact_no_posterior_support"
        diagnostics["exact_seconds"] = round(time.monotonic() - started, 3)
        diagnostics["exact_prewarm_wait_seconds"] = round(getattr(self.exact_engine, "last_prewarm_wait", 0.0), 3)
        if failure is not None:
            diagnostics["exact_status"] = failure
            return PredictionResult(
                revision=snapshot.revision,
                status="v6_fallback_exact_unavailable",
                compatible_worlds=0,
                p10=v6_p10,
                p50=v6_p50,
                p90=v6_p90,
                minimum=None,
                maximum=None,
                actionable=False,
                issues=tuple(dict.fromkeys((
                    *adapted.warnings,
                    failure,
                    "exact_unavailable_using_v6_fallback",
                    "packet_capture_inputs" if packet else "ocr_adapter_not_packet_equivalent",
                    "interval_uncalibrated",
                ))),
                contract=(
                    "packet_capture_v6_fallback_exact_unavailable_uncalibrated"
                    if packet
                    else "ocr_adapted_v6_fallback_exact_unavailable_uncalibrated"
                ),
                v6_prediction=v6_p50,
                generation_weight=0.0,
                model_version=V6_CANDIDATE_ID,
                estimate_source="v6_fallback",
                diagnostics=tuple(f"exact_unsupported:{reason}" for reason in converted.unsupported),
                posterior_method=EXACT_POSTERIOR_METHOD,
                conditional_diagnostics=diagnostics,
                **manual_bid_advice(
                    v6_p10, snapshot.round_number, safety_factor=self.bid_safety_factor,
                ),
            )
        estimate = self._exact_result(snapshot, adapted, v6_p10, v6_p50, v6_p90, converted, result, diagnostics)
        if result.status != "ok":
            self._exact_cut = {
                "snapshot": snapshot, "adapted": adapted, "v6": (v6_p10, v6_p50, v6_p90),
                "converted": converted, "diagnostics": diagnostics, "point": estimate.point_estimate,
            }
        return estimate

    def _exact_result(
        self,
        snapshot: ObservationSnapshot,
        adapted: AdaptedDecision,
        v6_p10: int,
        v6_p50: int,
        v6_p90: int,
        converted,
        result,
        diagnostics: dict,
    ) -> PredictionResult:
        """The exact-mode estimate for a sampler result that has support."""
        diagnostics = dict(diagnostics)
        packet = getattr(snapshot, "packet_decision", None) is not None
        summary = result.summary()
        p10 = max(1, round(summary["p10"]))
        p50 = max(p10, round(summary["p50"]))
        p90 = max(p50, round(summary["p90"]))
        point = round(summary["mape_opt"])
        reliable = result.status == "ok"
        diagnostics.update(
            exact_status=result.status,
            point_policy=EXACT_POINT_POLICY,
            lane_rel_se=summary.get("lane_rel_se"),
            sa_skip_bound=summary.get("sa_skip_bound"),
            first_round_cut=bool(summary.get("first_round_cut")),
            expected_ape_point=summary.get("expected_ape_mape_opt"),
        )
        return PredictionResult(
            revision=snapshot.revision,
            status="exact_v2_posterior_estimate" if reliable else "exact_v2_posterior_low_ess",
            compatible_worlds=int(result.totals.size),
            p10=p10,
            p50=p50,
            p90=p90,
            minimum=round(float(result.totals.min())),
            maximum=round(float(result.totals.max())),
            actionable=False,
            issues=tuple(dict.fromkeys((
                *adapted.warnings,
                *(() if reliable else ("exact_posterior_low_ess",)),
                "packet_capture_inputs" if packet else "ocr_adapter_not_packet_equivalent",
                "interval_uncalibrated",
            ))),
            contract=(
                "packet_capture_exact_v2_posterior_uncalibrated"
                if packet
                else "ocr_adapted_exact_v2_posterior_uncalibrated"
            ),
            v6_prediction=v6_p50,
            v2_prediction=point,
            generation_weight=1.0,
            model_version=V6_CANDIDATE_ID,
            estimate_source="exact_v2_posterior",
            diagnostics=tuple(f"exact_ignored:{reason}" for reason in converted.ignored),
            effective_sample_size=float(result.ess),
            posterior_converged=reliable,
            posterior_method=EXACT_POSTERIOR_METHOD,
            conditional_diagnostics=diagnostics,
            point_estimate=point,
            **manual_bid_advice(
                p10, snapshot.round_number, safety_factor=self.bid_safety_factor,
            ),
        )
