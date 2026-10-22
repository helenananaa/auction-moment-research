"""Experimental v3 posterior with independent random-quality events.

Each random reveal event is modeled as a sample without replacement from one
hidden map.  Different events are conditionally independent draws and may
therefore reveal the same treasure again.  This avoids the frozen v2
approximation that pools all random-identity outcomes into one unique sample.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from protocol_analog_value_model import (
    QUALITY_ORDER,
    ProtocolAnalogEvidenceModel,
    ProtocolAnalogValueModelError,
    _exact_colour_count,
    _multivariate_hypergeometric_log_likelihood,
    _supported_reveal_attributes,
    _weighted_quantile,
)


MODEL_SCHEMA_VERSION = "protocol-repeated-random-quality-v3-experimental"
RANDOM_QUALITY_EFFECTS = frozenset(
    {
        "reveal_identity_random",
        "reveal_position_quality_random",
    }
)


class ProtocolRepeatedQualityError(RuntimeError):
    """Raised when event-level random-quality evidence is inconsistent."""


@dataclass(frozen=True)
class EventQualitySample:
    round: int
    effect: str
    expected_count: int
    quality_counts: tuple[int, ...]
    track_ids: tuple[str, ...]

    def counts(self) -> dict[str, int]:
        return {
            quality: int(count)
            for quality, count in zip(
                QUALITY_ORDER,
                self.quality_counts,
            )
            if int(count) > 0
        }


def extract_event_quality_samples(decision: Mapping) -> dict:
    event_keys: Counter = Counter()
    expected_by_key: defaultdict[tuple[int, str], int] = defaultdict(
        int
    )
    for event in decision.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        semantics = event.get("semantics") or {}
        effect = str(
            semantics.get("effect")
            if isinstance(semantics, Mapping)
            else ""
        )
        if effect not in RANDOM_QUALITY_EFFECTS:
            continue
        if semantics.get("protocol_source") not in {
            "packet_capture",
            "ocr_exact",
            "ocr_thresholded",
            "human_confirmed",
        }:
            raise ProtocolRepeatedQualityError(
                "随机品质事件不是已确认的决策时来源"
            )
        try:
            event_round = int(event["round"])
            count = int(semantics["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolRepeatedQualityError(
                "随机品质事件缺少 round/count"
            ) from exc
        if event_round <= 0 or count <= 0:
            raise ProtocolRepeatedQualityError(
                "随机品质事件 round/count 无效"
            )
        key = (event_round, effect)
        event_keys[key] += 1
        expected_by_key[key] += count

    tracks_by_key: defaultdict[
        tuple[int, str], dict[str, str]
    ] = defaultdict(dict)
    invalid_keys = set()
    decision_round = int(decision.get("round") or 0)
    for item in decision.get("visible_items") or []:
        if not isinstance(item, Mapping):
            continue
        track_id = str(item.get("track_id") or "")
        quality = str(item.get("quality") or "")
        history = [
            row
            for row in item.get("history") or []
            if isinstance(row, Mapping)
            and str(row.get("source_effect") or "")
            in RANDOM_QUALITY_EFFECTS
            and int(row.get("round") or 0) <= decision_round
        ]
        for row in history:
            key = (
                int(row["round"]),
                str(row["source_effect"]),
            )
            valid = bool(
                track_id
                and quality in QUALITY_ORDER
                and _supported_reveal_attributes(item, decision, row)
            )
            if not valid:
                invalid_keys.add(key)
                continue
            previous = tracks_by_key[key].get(track_id)
            if previous is not None and previous != quality:
                invalid_keys.add(key)
                continue
            tracks_by_key[key][track_id] = quality

    accepted = []
    rejected = []
    for key in sorted(expected_by_key):
        event_round, effect = key
        expected = int(expected_by_key[key])
        observed = tracks_by_key.get(key, {})
        reason = None
        if event_keys[key] != 1:
            reason = "duplicate_same_round_effect_not_separable"
        elif key in invalid_keys:
            reason = "invalid_exact_quality_track"
        elif len(observed) != expected:
            reason = "event_sample_count_mismatch"
        if reason is not None:
            rejected.append(
                {
                    "round": event_round,
                    "effect": effect,
                    "expected_count": expected,
                    "observed_count": len(observed),
                    "reason": reason,
                }
            )
            continue
        counts = Counter(observed.values())
        accepted.append(
            EventQualitySample(
                round=event_round,
                effect=effect,
                expected_count=expected,
                quality_counts=tuple(
                    int(counts[quality])
                    for quality in QUALITY_ORDER
                ),
                track_ids=tuple(sorted(observed)),
            )
        )
    return {
        "status": (
            "accepted_with_fail_closed_event_rejections"
            if rejected
            else "all_event_samples_exact"
        ),
        "accepted": accepted,
        "rejected": rejected,
    }


@dataclass
class ProtocolRepeatedQualityEvidenceModel:
    frozen_v2: ProtocolAnalogEvidenceModel

    def __post_init__(self) -> None:
        if not isinstance(
            self.frozen_v2,
            ProtocolAnalogEvidenceModel,
        ):
            raise ProtocolRepeatedQualityError(
                "frozen_v2 类型错误"
            )

    def _posterior_weights(
        self,
        *,
        map_rows: int,
        colour_count: int | None,
        samples: Sequence[EventQualitySample],
    ) -> np.ndarray:
        weights = self.frozen_v2._weights(
            map_rows=map_rows,
            colour_count=colour_count,
            sample_quality_counts={},
            conditioned=colour_count is not None,
        )
        if not samples:
            return weights
        log_weights = []
        for world, weight in zip(
            self.frozen_v2.worlds,
            weights,
        ):
            value = math.log(max(float(weight), 1e-300))
            for sample in samples:
                value += _multivariate_hypergeometric_log_likelihood(
                    world,
                    sample.counts(),
                )
            log_weights.append(value)
        finite = [
            value
            for value in log_weights
            if math.isfinite(value)
        ]
        if not finite:
            raise ProtocolRepeatedQualityError(
                "事件级随机品质后验没有有限权重"
            )
        maximum = max(finite)
        result = np.asarray(
            [
                math.exp(value - maximum)
                if math.isfinite(value)
                else 0.0
                for value in log_weights
            ],
            dtype=float,
        )
        if float(result.sum()) <= 0.0:
            raise ProtocolRepeatedQualityError(
                "事件级随机品质后验权重和为零"
            )
        return result / float(result.sum())

    def predict(self, decision: Mapping) -> dict:
        try:
            base = self.frozen_v2.base_model.predict(decision)
            colour_count = _exact_colour_count(decision)
        except ProtocolAnalogValueModelError as exc:
            raise ProtocolRepeatedQualityError(str(exc)) from exc
        samples = extract_event_quality_samples(decision)
        map_rows = int(base["map_rows"])
        prior_weights = self.frozen_v2._weights(
            map_rows=map_rows,
            colour_count=None,
            sample_quality_counts={},
            conditioned=False,
        )
        posterior_weights = self._posterior_weights(
            map_rows=map_rows,
            colour_count=colour_count,
            samples=samples["accepted"],
        )
        totals = np.asarray(
            [
                world.total_value
                for world in self.frozen_v2.worlds
            ],
            dtype=float,
        )
        prior_median = _weighted_quantile(
            totals,
            prior_weights,
            0.5,
        )
        posterior_median = _weighted_quantile(
            totals,
            posterior_weights,
            0.5,
        )
        evidence_delta = posterior_median - prior_median
        prediction = round(
            max(
                1.0,
                float(base["prediction"]) + evidence_delta,
            )
        )
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "decision_id": str(decision.get("decision_id") or ""),
            "session_id": str(decision.get("session_id") or ""),
            "round": int(decision.get("round") or 0),
            "map_rows": map_rows,
            "prediction": prediction,
            "frozen_additive_prediction": int(base["prediction"]),
            "exact_colour_count": colour_count,
            "accepted_event_sample_count": len(
                samples["accepted"]
            ),
            "rejected_event_samples": samples["rejected"],
            "event_samples": [
                {
                    "round": sample.round,
                    "effect": sample.effect,
                    "expected_count": sample.expected_count,
                    "quality_counts": sample.counts(),
                    "track_ids": list(sample.track_ids),
                }
                for sample in samples["accepted"]
            ],
            "prior_analog_median": round(prior_median),
            "posterior_analog_median": round(
                posterior_median
            ),
            "evidence_delta": round(evidence_delta),
            "experimental_only": True,
            "production_enabled": False,
        }
