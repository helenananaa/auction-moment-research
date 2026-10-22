"""Mechanism-grounded analogue update for exact packet evidence.

The model deliberately keeps the frozen additive v1 point estimator as its
base.  It uses reviewed historical games only to estimate the *change* implied
by two packet facts whose semantics are independently auditable:

* an exact count of coloured (``彩``) treasures;
* the quality composition of exact ``reveal_identity_random`` outcomes.

Historical games near the packet-proven map height receive a Gaussian kernel
weight.  Exact coloured counts add a narrow count likelihood and random
identity qualities add a multivariate-hypergeometric likelihood.  The update
is the difference between the conditioned and unconditioned weighted medians,
so a decision with no supported evidence is bit-for-bit the v1 point estimate.

Identity *values* are intentionally excluded.  Earlier exchangeability audits
found the quality composition materially more defensible than a pooled
sample-value estimator, especially for one-item samples containing a coloured
treasure.
"""

from __future__ import annotations

import bisect
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from protocol_additive_value_model import (
    ProtocolAdditiveExactEventModel,
    ProtocolAdditiveValueModelError,
    canonical_sha256,
    sha256_file,
)


MODEL_SCHEMA_VERSION = "protocol-analog-colour-sample-v2"
QUALITY_ORDER = ("白", "蓝", "紫", "金", "彩")
MODEL_CONFIG = {
    "row_bandwidth": 1.0,
    "colour_count_sigma": 0.5,
    "random_identity_quality_temperature": 1.0,
    "point_quantile": 0.5,
    "interval_lower_quantile": 0.1,
    "interval_upper_quantile": 0.9,
    "interval_scale": 1.0,
    "minimum_weight_sum": 1e-12,
    "selective_risk_posterior_quantile": 0.95,
    "selective_risk_interval_weight": 0.5,
    "selective_risk_posterior_weight": 0.5,
    "selective_risk_minimum_reference_sessions": 20,
}


class ProtocolAnalogValueModelError(RuntimeError):
    """Raised when analogue evidence or historical truth is invalid."""


@dataclass(frozen=True)
class AnalogWorld:
    session_id: str
    map_rows: int
    total_value: int
    item_count: int
    quality_counts: tuple[int, ...]

    def quality_count(self, quality: str) -> int:
        return int(self.quality_counts[QUALITY_ORDER.index(quality)])


def _reviewed_world(record: Mapping) -> AnalogWorld:
    annotations = [
        item
        for item in record.get("annotations") or []
        if isinstance(item, Mapping)
    ]
    if not annotations:
        raise ProtocolAnalogValueModelError(
            f"历史对局缺少 annotations：{record.get('session_id')}"
        )
    try:
        map_rows = max(
            int(item["row"]) + int(item["height"])
            for item in annotations
        )
        values = [int(item["value"]) for item in annotations]
        qualities = [str(item["quality"]) for item in annotations]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolAnalogValueModelError(
            f"历史对局字段无效：{record.get('session_id')}"
        ) from exc
    if (
        map_rows <= 0
        or any(value <= 0 for value in values)
        or any(quality not in QUALITY_ORDER for quality in qualities)
    ):
        raise ProtocolAnalogValueModelError(
            f"历史对局含非法行数、价值或品质："
            f"{record.get('session_id')}"
        )
    counts = Counter(qualities)
    return AnalogWorld(
        session_id=str(record.get("session_id") or ""),
        map_rows=map_rows,
        total_value=sum(values),
        item_count=len(values),
        quality_counts=tuple(
            int(counts[quality]) for quality in QUALITY_ORDER
        ),
    )


def load_analog_worlds(
    final_annotations_path: Path,
) -> tuple[list[AnalogWorld], dict]:
    try:
        payload = json.loads(
            Path(final_annotations_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolAnalogValueModelError(
            f"无法读取历史金标：{final_annotations_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("images"), list
    ):
        raise ProtocolAnalogValueModelError(
            "历史金标缺少 images 数组"
        )
    records = [
        record
        for record in payload["images"]
        if isinstance(record, Mapping)
        and record.get("status") == "reviewed"
        and record.get("session_id")
    ]
    worlds = [_reviewed_world(record) for record in records]
    session_ids = [world.session_id for world in worlds]
    if (
        not worlds
        or len(session_ids) != len(set(session_ids))
        or session_ids != sorted(session_ids)
    ):
        raise ProtocolAnalogValueModelError(
            "历史会话为空、重复或未按时间排序"
        )
    provenance = {
        "path": str(Path(final_annotations_path).resolve()),
        "sha256": sha256_file(final_annotations_path),
        "sessions": len(worlds),
        "first_session_id": session_ids[0],
        "last_session_id": session_ids[-1],
        "session_fingerprint": canonical_sha256(session_ids),
        "world_fingerprint": canonical_sha256(
            [
                {
                    "session_id": world.session_id,
                    "map_rows": world.map_rows,
                    "total_value": world.total_value,
                    "item_count": world.item_count,
                    "quality_counts": world.quality_counts,
                }
                for world in worlds
            ]
        ),
    }
    return worlds, provenance


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    if (
        values.ndim != 1
        or weights.ndim != 1
        or len(values) == 0
        or len(values) != len(weights)
        or not 0.0 <= probability <= 1.0
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
        or float(weights.sum()) <= 0.0
    ):
        raise ProtocolAnalogValueModelError(
            "加权分位数输入无效"
        )
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    threshold = float(probability) * float(cumulative[-1])
    index = min(
        int(np.searchsorted(cumulative, threshold, side="left")),
        len(ordered_values) - 1,
    )
    return float(ordered_values[index])


def _empirical_cdf(
    sorted_reference: Sequence[float],
    value: float,
) -> float:
    if (
        not sorted_reference
        or not math.isfinite(float(value))
        or any(
            not math.isfinite(float(reference))
            for reference in sorted_reference
        )
        or any(
            float(left) > float(right)
            for left, right in zip(
                sorted_reference,
                sorted_reference[1:],
            )
        )
    ):
        raise ProtocolAnalogValueModelError(
            "经验风险分位数参考无效"
        )
    return bisect.bisect_right(
        sorted_reference,
        float(value),
    ) / len(sorted_reference)


def _contains_forbidden_label_key(value: object) -> bool:
    forbidden = {
        "actual",
        "decision_outcome",
        "label",
        "settlement",
        "settlement_total",
    }
    if isinstance(value, Mapping):
        if forbidden & {str(key) for key in value}:
            return True
        return any(
            _contains_forbidden_label_key(item)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_label_key(item) for item in value)
    return False


def _multivariate_hypergeometric_log_likelihood(
    world: AnalogWorld,
    sample_quality_counts: Mapping[str, int],
) -> float:
    trials = sum(
        max(0, int(count))
        for count in sample_quality_counts.values()
    )
    if trials <= 0:
        return 0.0
    if trials > world.item_count:
        return -math.inf
    result = (
        math.lgamma(trials + 1)
        + math.lgamma(world.item_count - trials + 1)
        - math.lgamma(world.item_count + 1)
    )
    for quality, raw_count in sample_quality_counts.items():
        if quality not in QUALITY_ORDER:
            return -math.inf
        count = max(0, int(raw_count))
        available = world.quality_count(quality)
        if count > available:
            return -math.inf
        result += (
            math.lgamma(available + 1)
            - math.lgamma(count + 1)
            - math.lgamma(available - count + 1)
        )
    return result


def _exact_colour_count(decision: Mapping) -> int | None:
    completeness = (
        (decision.get("map_recognition") or {}).get("completeness")
        or {}
    )
    counts = completeness.get("exact_quality_counts") or {}
    if "彩" not in counts:
        return None
    matching_event = any(
        isinstance(event, Mapping)
        and isinstance(event.get("semantics"), Mapping)
        and event["semantics"].get("protocol_source")
        in {
            "packet_capture",
            "ocr_exact",
            "ocr_thresholded",
            "human_confirmed",
        }
        and event["semantics"].get("effect")
        == "reveal_position_by_quality"
        and event["semantics"].get("quality") == "彩"
        for event in decision.get("events") or []
    )
    if not matching_event:
        raise ProtocolAnalogValueModelError(
            "彩色精确计数没有对应抓包事件证明"
        )
    try:
        value = int(counts["彩"])
    except (TypeError, ValueError) as exc:
        raise ProtocolAnalogValueModelError(
            "彩色精确计数字段无效"
        ) from exc
    if value < 0:
        raise ProtocolAnalogValueModelError(
            "彩色精确计数不能为负"
        )
    return value


OCR_REVEAL_ATTRIBUTION_SCHEMA = "ocr-first-reveal-single-source-v1"


def _certified_ocr_reveal_events(decision: Mapping) -> frozenset[tuple[int, str, str]]:
    """Recognize the reviewed OCR adapter's explicit sample attribution.

    Thresholded OCR is never promoted to packet-exact or human-confirmed.
    Other callers retain the original exact-item contract.
    """
    native = decision.get("native_evidence") or {}
    attribution = decision.get("reveal_attribution") or {}
    if (
        decision.get("adapter_schema_version") != "ocr-visible-pre-bid-adapter-v1"
        or native.get("mode") != "ocr_visible_pre_bid_v1"
        or native.get("pre_bid_proven") is not True
        or native.get("ocr_used_for_live_features") is not True
        or native.get("settlement_or_final_truth_in_features") is not False
        or attribution.get("schema_version") != OCR_REVEAL_ATTRIBUTION_SCHEMA
    ):
        return frozenset()
    result = set()
    for certificate in attribution.get("certified_events") or []:
        try:
            event_round = int(certificate["round"])
            sample_size = int(certificate["sample_size"])
            effect = str(certificate["source_effect"])
            kind = str(certificate["event_kind"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            certificate.get("attribution_schema") != OCR_REVEAL_ATTRIBUTION_SCHEMA
            or not 0 < event_round <= int(decision.get("round") or 0)
            or sample_size <= 0
            or kind not in {"public", "personal"}
            or effect not in {"reveal_identity_random", "reveal_position_quality_random"}
        ):
            continue
        matching = [
            event for event in decision.get("events") or []
            if int(event.get("round") or 0) == event_round
            and event.get("kind") == kind
            and (event.get("semantics") or {}).get("effect") == effect
            and (event.get("semantics") or {}).get("count") == sample_size
            and (event.get("semantics") or {}).get("parsed") is True
            and (event.get("semantics") or {}).get("protocol_source")
            in {"ocr_thresholded", "human_confirmed"}
        ]
        try:
            confidence = float(matching[0].get("ocr_confidence") or 0.0) if matching else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        if len(matching) == 1 and math.isfinite(confidence) and confidence >= 0.85:
            result.add((event_round, effect, kind))
    return frozenset(result)


def _supported_reveal_attributes(
    item: Mapping, decision: Mapping, history: Mapping
) -> bool:
    """Accept exact legacy evidence or attributed thresholded OCR evidence."""
    effect = str(history.get("source_effect") or "")
    identity = item.get("identity") or {}
    exact = {"protocol_exact", "human_confirmed"}
    quality_status = item.get("quality_validation")
    identity_status = identity.get("match_confidence_status")
    if quality_status in exact and (
        effect != "reveal_identity_random" or identity_status in exact
    ):
        return True
    if quality_status not in exact | {"ocr_thresholded"} or (
        effect == "reveal_identity_random"
        and identity_status not in exact | {"ocr_thresholded"}
    ):
        return False
    try:
        confidence = float(item.get("quality_confidence") or 0.0)
        evidence_confidence = float(item.get("evidence_confidence") or 0.0)
        event_key = (
            int(history.get("round") or 0), effect,
            str(history.get("event_kind") or ""),
        )
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(confidence) and confidence >= 0.85
        and math.isfinite(evidence_confidence) and evidence_confidence >= 0.85
        and history.get("attribution_schema") == OCR_REVEAL_ATTRIBUTION_SCHEMA
        and event_key in _certified_ocr_reveal_events(decision)
    )


def _random_identity_quality_sample(decision: Mapping) -> dict:
    expected = 0
    event_rounds = []
    certified = _certified_ocr_reveal_events(decision)
    for event in decision.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        semantics = event.get("semantics") or {}
        if (
            not isinstance(semantics, Mapping)
            or semantics.get("effect")
            != "reveal_identity_random"
        ):
            continue
        if semantics.get("protocol_source") not in {
            "packet_capture",
            "ocr_exact",
            "ocr_thresholded",
            "human_confirmed",
        }:
            raise ProtocolAnalogValueModelError(
                "随机身份揭示不是已确认的决策时来源"
            )
        try:
            count = int(semantics["count"])
            event_round = int(event["round"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolAnalogValueModelError(
                "随机身份揭示缺少 count 或 round"
            ) from exc
        if count <= 0 or event_round <= 0:
            raise ProtocolAnalogValueModelError(
                "随机身份揭示 count/round 无效"
            )
        if certified and (
            event_round, "reveal_identity_random", str(event.get("kind") or "")
        ) not in certified:
            # The OCR store does not retain later attribute histories. Only
            # the first safely attributed sample belongs to this likelihood.
            continue
        expected += count
        event_rounds.append(event_round)

    samples: dict[str, str] = {}
    for item in decision.get("visible_items") or []:
        if not isinstance(item, Mapping):
            continue
        history = [
            row
            for row in item.get("history") or []
            if isinstance(row, Mapping)
            and row.get("source_effect")
            == "reveal_identity_random"
            and int(row.get("round") or 0)
            <= int(decision.get("round") or 0)
        ]
        if not history:
            continue
        quality = str(item.get("quality") or "")
        track_id = str(item.get("track_id") or "")
        if (
            not track_id
            or quality not in QUALITY_ORDER
            or not all(
                _supported_reveal_attributes(item, decision, row) for row in history
            )
        ):
            return {
                "status": "feature_rejected_invalid_exact_item",
                "expected_count": expected,
                "sample_count": 0,
                "quality_counts": {},
                "event_rounds": sorted(set(event_rounds)),
            }
        prior = samples.get(track_id)
        if prior is not None and prior != quality:
            return {
                "status": "feature_rejected_conflicting_track",
                "expected_count": expected,
                "sample_count": 0,
                "quality_counts": {},
                "event_rounds": sorted(set(event_rounds)),
            }
        samples[track_id] = quality

    observed = len(samples)
    if expected == 0 and observed == 0:
        status = "no_random_identity_event"
    elif expected != observed:
        status = "feature_rejected_count_mismatch"
        samples = {}
        observed = 0
    else:
        status = "exact_random_identity_quality_sample"
    return {
        "status": status,
        "expected_count": expected,
        "sample_count": observed,
        "quality_counts": dict(
            sorted(Counter(samples.values()).items())
        ),
        "event_rounds": sorted(set(event_rounds)),
    }


@dataclass
class ProtocolAnalogEvidenceModel:
    base_model: ProtocolAdditiveExactEventModel
    worlds: tuple[AnalogWorld, ...]
    training_provenance: dict
    config: dict
    selective_risk_reference: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        *,
        base_model: ProtocolAdditiveExactEventModel,
        worlds: Sequence[AnalogWorld],
        training_provenance: Mapping,
        selective_risk_decisions: Sequence[Mapping] | None = None,
        selective_risk_provenance: Mapping | None = None,
    ) -> "ProtocolAnalogEvidenceModel":
        normalized = tuple(worlds)
        if len(normalized) < 30:
            raise ProtocolAnalogValueModelError(
                "经验后验至少需要 30 局历史世界"
            )
        if not isinstance(
            base_model, ProtocolAdditiveExactEventModel
        ):
            raise ProtocolAnalogValueModelError(
                "base_model 类型错误"
            )
        model = cls(
            base_model=base_model,
            worlds=normalized,
            training_provenance=dict(training_provenance),
            config=json.loads(
                json.dumps(MODEL_CONFIG, ensure_ascii=False)
            ),
        )
        if (
            selective_risk_decisions is None
            and selective_risk_provenance is not None
        ) or (
            selective_risk_decisions is not None
            and selective_risk_provenance is None
        ):
            raise ProtocolAnalogValueModelError(
                "风险参考 decisions/provenance 必须同时提供"
            )
        if selective_risk_decisions is not None:
            model.calibrate_selective_risk(
                selective_risk_decisions,
                provenance=selective_risk_provenance or {},
            )
        return model

    def calibrate_selective_risk(
        self,
        decisions: Sequence[Mapping],
        *,
        provenance: Mapping,
    ) -> None:
        """Freeze a feature-only risk scale from historical shadow rows."""

        if self.selective_risk_reference:
            raise ProtocolAnalogValueModelError(
                "选择性风险参考已经冻结，禁止重复校准"
            )
        normalized = [dict(decision) for decision in decisions]
        decision_ids = [
            str(decision.get("decision_id") or "")
            for decision in normalized
        ]
        session_ids = [
            str(decision.get("session_id") or "")
            for decision in normalized
        ]
        if (
            not normalized
            or not all(decision_ids)
            or not all(session_ids)
            or len(decision_ids) != len(set(decision_ids))
            or any(
                str(decision.get("cohort") or "")
                != "historical_shadow_train_only"
                for decision in normalized
            )
            or any(
                _contains_forbidden_label_key(decision)
                for decision in normalized
            )
        ):
            raise ProtocolAnalogValueModelError(
                "风险参考必须是无标签且主键唯一的 shadow 决策"
            )
        unique_sessions = sorted(set(session_ids))
        if len(unique_sessions) < int(
            self.config[
                "selective_risk_minimum_reference_sessions"
            ]
        ):
            raise ProtocolAnalogValueModelError(
                "选择性风险参考的 shadow 会话不足"
            )
        components = [self.predict(decision) for decision in normalized]
        widths = sorted(
            float(row["relative_interval_width"])
            for row in components
        )
        posterior_tails = sorted(
            float(
                row[
                    "posterior_p95_absolute_percentage_error"
                ]
            )
            for row in components
        )
        reference_identity = {
            "schema_version": "protocol-selective-risk-reference-v1",
            "calibration_source": "feature_only_historical_shadow",
            "labels_used": False,
            "decisions": len(normalized),
            "sessions": len(unique_sessions),
            "decision_fingerprint": canonical_sha256(
                sorted(decision_ids)
            ),
            "session_fingerprint": canonical_sha256(
                unique_sessions
            ),
            "relative_interval_widths": widths,
            "posterior_p95_absolute_percentage_errors": (
                posterior_tails
            ),
            "provenance": dict(provenance),
        }
        reference_identity["reference_fingerprint"] = (
            canonical_sha256(reference_identity)
        )
        self.selective_risk_reference = reference_identity

    def _weights(
        self,
        *,
        map_rows: int,
        colour_count: int | None,
        sample_quality_counts: Mapping[str, int],
        conditioned: bool,
    ) -> np.ndarray:
        bandwidth = float(self.config["row_bandwidth"])
        colour_sigma = float(
            self.config["colour_count_sigma"]
        )
        temperature = float(
            self.config[
                "random_identity_quality_temperature"
            ]
        )
        log_weights = []
        for world in self.worlds:
            value = -0.5 * (
                (world.map_rows - map_rows) / bandwidth
            ) ** 2
            if conditioned and colour_count is not None:
                value += -0.5 * (
                    (
                        world.quality_count("彩") - colour_count
                    )
                    / colour_sigma
                ) ** 2
            if conditioned and sample_quality_counts:
                likelihood = (
                    _multivariate_hypergeometric_log_likelihood(
                        world, sample_quality_counts
                    )
                )
                value += temperature * likelihood
            log_weights.append(value)
        finite = [
            value for value in log_weights if math.isfinite(value)
        ]
        if not finite:
            raise ProtocolAnalogValueModelError(
                "经验后验没有有限权重"
            )
        maximum = max(finite)
        weights = np.asarray(
            [
                math.exp(value - maximum)
                if math.isfinite(value)
                else 0.0
                for value in log_weights
            ],
            dtype=float,
        )
        total = float(weights.sum())
        if total < float(self.config["minimum_weight_sum"]):
            raise ProtocolAnalogValueModelError(
                "经验后验权重质量不足"
            )
        return weights / total

    def predict(self, decision: Mapping) -> dict:
        try:
            base = self.base_model.predict(decision)
        except ProtocolAdditiveValueModelError as exc:
            raise ProtocolAnalogValueModelError(str(exc)) from exc
        map_rows = int(base["map_rows"])
        colour_count = _exact_colour_count(decision)
        sample = _random_identity_quality_sample(decision)
        sample_counts = (
            sample["quality_counts"]
            if sample["status"]
            == "exact_random_identity_quality_sample"
            else {}
        )
        evidence_available = bool(
            colour_count is not None or sample_counts
        )
        unconditioned_weights = self._weights(
            map_rows=map_rows,
            colour_count=None,
            sample_quality_counts={},
            conditioned=False,
        )
        conditioned_weights = (
            self._weights(
                map_rows=map_rows,
                colour_count=colour_count,
                sample_quality_counts=sample_counts,
                conditioned=True,
            )
            if evidence_available
            else unconditioned_weights.copy()
        )
        totals = np.asarray(
            [world.total_value for world in self.worlds],
            dtype=float,
        )
        point_quantile = float(self.config["point_quantile"])
        prior_median = _weighted_quantile(
            totals, unconditioned_weights, point_quantile
        )
        conditioned_median = _weighted_quantile(
            totals, conditioned_weights, point_quantile
        )
        evidence_delta = conditioned_median - prior_median
        prediction = max(
            1.0,
            float(base["prediction"]) + evidence_delta,
        )
        interval_scale = float(self.config["interval_scale"])
        support = prediction + interval_scale * (
            totals - conditioned_median
        )
        support = np.maximum(
            support,
            float(base["known_value_lower_bound"]),
        )
        p10 = _weighted_quantile(
            support,
            conditioned_weights,
            float(self.config["interval_lower_quantile"]),
        )
        p90 = _weighted_quantile(
            support,
            conditioned_weights,
            float(self.config["interval_upper_quantile"]),
        )
        effective_sample_size = 1.0 / float(
            np.sum(conditioned_weights**2)
        )
        positive = conditioned_weights[conditioned_weights > 0]
        entropy = -float(
            np.sum(positive * np.log(positive))
        )
        normalized_entropy = (
            entropy / math.log(len(self.worlds))
            if len(self.worlds) > 1
            else 0.0
        )
        posterior_absolute_percentage_errors = (
            np.abs(support - prediction) / np.maximum(support, 1.0)
        )
        posterior_p95_ape = _weighted_quantile(
            posterior_absolute_percentage_errors,
            conditioned_weights,
            float(
                self.config[
                    "selective_risk_posterior_quantile"
                ]
            ),
        )
        relative_interval_width = round(
            (p90 - p10) / max(prediction, 1.0),
            8,
        )
        rounded_posterior_p95_ape = round(
            posterior_p95_ape,
            8,
        )
        selective_risk = {
            "schema_version": "protocol-selective-risk-v1",
            "status": "uncalibrated_feature_components_only",
            "score": None,
            "relative_interval_width_percentile": None,
            "posterior_p95_ape_percentile": None,
            "reference_fingerprint": None,
        }
        if self.selective_risk_reference:
            width_percentile = _empirical_cdf(
                self.selective_risk_reference[
                    "relative_interval_widths"
                ],
                relative_interval_width,
            )
            posterior_percentile = _empirical_cdf(
                self.selective_risk_reference[
                    "posterior_p95_absolute_percentage_errors"
                ],
                rounded_posterior_p95_ape,
            )
            interval_weight = float(
                self.config["selective_risk_interval_weight"]
            )
            posterior_weight = float(
                self.config["selective_risk_posterior_weight"]
            )
            total_weight = interval_weight + posterior_weight
            if total_weight <= 0.0:
                raise ProtocolAnalogValueModelError(
                    "选择性风险权重无效"
                )
            selective_risk = {
                "schema_version": "protocol-selective-risk-v1",
                "status": "calibrated_feature_only",
                "score": round(
                    (
                        interval_weight * width_percentile
                        + posterior_weight * posterior_percentile
                    )
                    / total_weight,
                    8,
                ),
                "relative_interval_width_percentile": round(
                    width_percentile,
                    8,
                ),
                "posterior_p95_ape_percentile": round(
                    posterior_percentile,
                    8,
                ),
                "reference_fingerprint": (
                    self.selective_risk_reference[
                        "reference_fingerprint"
                    ]
                ),
            }
        support_rows = [
            {
                "historical_session_id": world.session_id,
                "total_value": round(float(value)),
                "weight": round(float(weight), 12),
            }
            for world, value, weight in zip(
                self.worlds, support, conditioned_weights
            )
            if float(weight) > 0.0
        ]
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "decision_id": str(decision.get("decision_id") or ""),
            "session_id": str(decision.get("session_id") or ""),
            "round": int(decision.get("round") or 0),
            "map_rows": map_rows,
            "prediction": round(prediction),
            "p10": round(max(0.0, p10)),
            "p90": round(max(p10, p90)),
            "relative_interval_width": relative_interval_width,
            "posterior_p95_absolute_percentage_error": (
                rounded_posterior_p95_ape
            ),
            "selective_risk": selective_risk,
            "base_prediction": int(base["prediction"]),
            "base_row_prediction": int(
                base["base_row_prediction"]
            ),
            "base_active_adjustments": list(base["adjustments"]),
            "known_value_lower_bound": int(
                base["known_value_lower_bound"]
            ),
            "prior_analog_median": round(prior_median),
            "conditioned_analog_median": round(
                conditioned_median
            ),
            "evidence_delta": round(evidence_delta),
            "evidence_available": evidence_available,
            "exact_colour_count": colour_count,
            "random_identity_sample": sample,
            "effective_sample_size": round(
                effective_sample_size, 8
            ),
            "normalized_weight_entropy": round(
                normalized_entropy, 8
            ),
            "maximum_world_weight": round(
                float(conditioned_weights.max()), 8
            ),
            "distribution_support": support_rows,
        }

    def identity(self) -> dict:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "config": self.config,
            "training_provenance": self.training_provenance,
            "base_model_identity": self.base_model.identity(),
            "world_count": len(self.worlds),
            "world_fingerprint": canonical_sha256(
                [
                    {
                        "session_id": world.session_id,
                        "map_rows": world.map_rows,
                        "total_value": world.total_value,
                        "item_count": world.item_count,
                        "quality_counts": world.quality_counts,
                    }
                    for world in self.worlds
                ]
            ),
            "selective_risk_reference": (
                {
                    key: value
                    for key, value in self.selective_risk_reference.items()
                    if key
                    not in {
                        "relative_interval_widths",
                        "posterior_p95_absolute_percentage_errors",
                    }
                }
                if self.selective_risk_reference
                else None
            ),
        }
