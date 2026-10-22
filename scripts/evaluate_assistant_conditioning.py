"""Frozen public event-only development comparison; never train or publish.

The public export has no usable per-round object observations or exact map
height. This controlled input comparison cannot measure reveal/layout gains,
screen recognition, unseen accuracy, or interval calibration. Frozen model
training overlaps this time window; HMAC IDs prevent reliable membership joins.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from auction_moment_assistant.latest_model import (
    AdaptedDecision,
    LatestV6V2Predictor,
    OcrDecisionAdapter,
    _geometric_blend,
)
from auction_moment_assistant.models import load_catalog, sha256_file
from auction_moment_assistant.observations import ObservationSnapshot
from auction_moment_assistant.predictor import (
    INFORMATIONAL_EFFECTS,
    EmpiricalWorldPredictor,
    PredictionResult,
)


SCHEMA = "assistant-conditioning-event-only-development-v1"
QUALITY_CODES = {1: "白", 2: "蓝", 3: "紫", 4: "金", 5: "彩"}


def public_path(path: Path | str) -> str:
    """Do not disclose workstation paths in public development receipts."""
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return f"external/{path.name}"


def sanitize_receipt_paths(report: dict, *, original_sha256: str) -> dict:
    """Rewrite only provenance path keys in an existing receipt; no inference."""
    result = dict(report)
    for field in (
        "provenance_sha256",
        "runtime_source_sha256",
        "runtime_source_sha256_after",
    ):
        result[field] = {
            public_path(path): digest for path, digest in result[field].items()
        }
    result["sanitization"] = {
        "schema_version": "public-provenance-paths-v1",
        "original_receipt_sha256": original_sha256,
        "changed_fields": [
            "provenance_sha256",
            "runtime_source_sha256",
            "runtime_source_sha256_after",
        ],
        "operation": "path_keys_only_repo_relative_posix_or_external_basename",
        "resampled": False,
        "prediction_statistics_and_hash_values_unchanged": True,
    }
    return result


def finite_json(value):
    """Keep failed chain diagnostics explicit without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return f"nonfinite:{value}"
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    return value


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def event_semantics(row: dict) -> dict:
    """Translate only event-time values; validation/settlement fields excluded."""
    effect = row["effect"]
    result = {"parsed": True, "effect": effect}
    dimension, group = row["group_dimension"], row["group_value"]
    if dimension == "quality":
        result["quality"] = group
    elif dimension == "size":
        width, height = str(group).lower().split("x")
        result.update(width=int(width), height=int(height))
    if effect in INFORMATIONAL_EFFECTS:
        # The export retains selection metadata, not which objects were revealed.
        # In particular quality_codes/catalog_item_count/reported_cell_count
        # must not be used to invent observed objects or zero-object outcomes.
        result["reveal_outcome_available"] = False
        if dimension == "random":
            result["count"] = int(group)
        return result
    value = int(float(row["observed_value"]))
    scale = int(float(row["value_scale"] or 1))
    if effect in {"quality_item_count", "size_item_count", "total_item_count"}:
        result["observed_count"] = value
    elif effect in {"quality_total_area", "size_total_area", "total_occupied_cells"}:
        result["observed_cells"] = value
    elif effect in {"quality_average_area", "size_average_area", "average_item_area"}:
        result["observed_average_cells"] = value / scale
    elif effect == "highest_quality":
        result["observed_quality"] = QUALITY_CODES[value]
    else:
        result["observed_value"] = value
    return result


def select_inputs(core: Path, *, limit: int, rounds: tuple[int, ...]) -> list[dict]:
    """Freeze selection by event availability and hashed keys, never errors."""
    labels = {}
    for row in read_rows(core / "sessions.csv"):
        if row.get("protocol_settlement_valid", "").lower() != "true":
            continue
        try:
            label = int(float(row["settlement_total"]))
        except (KeyError, ValueError):
            continue
        if label > 0:
            labels[row["session_id"]] = (label, row.get("date"))
    maps = {
        (row["session_id"], int(row["round"])): row
        for row in read_rows(core / "rounds.csv")
    }
    events = defaultdict(dict)
    conflicts = set()
    for row in read_rows(core / "event_instances.csv"):
        session = row["session_id"]
        key = (int(row["round"]), row["scope"])
        if key[1] not in {"public", "personal"}:
            continue
        semantics = event_semantics(row)
        if key in events[session] and events[session][key] != semantics:
            conflicts.add(session)
        events[session][key] = semantics
    candidates = []
    for session, by_key in events.items():
        if session not in labels or session in conflicts:
            continue
        for round_number in rounds:
            keys = [
                (number, scope)
                for number in range(1, round_number + 1)
                for scope in ("public", "personal")
            ]
            if not all(key in by_key for key in keys):
                continue
            map_row = maps.get((session, round_number), {})
            rows_value = map_row.get("map_rows")
            map_rows = int(rows_value) if rows_value else None
            snapshot = ObservationSnapshot(
                revision=round_number,
                round_number=round_number,
                bankroll=None,
                events=tuple(
                    {
                        "round_number": key[0],
                        "kind": key[1],
                        "semantics": by_key[key],
                        "confidence": 1.0,
                        "human_locked": True,
                        "source": "public_event_instances_development_replay",
                    }
                    for key in keys
                ),
                map_items=(),
                completeness_confirmed=False,
                completeness_source="unavailable_public_event_only_replay",
                map_rows=map_rows,
                map_height_exact=False,
                map_rows_source="automatic_scroll_estimate",
                pre_bid_confirmed=True,
                pre_bid_source="public_event_instances_development_replay",
            )
            decision_key = f"{session}:R{round_number}"
            candidates.append(
                {
                    "decision_key": decision_key,
                    "session_id": session,
                    "date": labels[session][1],
                    "label": labels[session][0],
                    "snapshot": snapshot,
                    "selection_key": hashlib.sha256(decision_key.encode()).hexdigest(),
                }
            )
    return sorted(candidates, key=lambda row: row["selection_key"])[:limit]


def score(prediction: dict, label: int) -> dict:
    p10, p50, p90 = (prediction.get(name) for name in ("p10", "p50", "p90"))
    if p50 is None or not math.isfinite(float(p50)):
        return {"absolute_percentage_error": None, "interval_covers_label": None}
    return {
        "absolute_percentage_error": abs(float(p50) - label) / label,
        "interval_covers_label": (
            p10 <= label <= p90 if p10 is not None and p90 is not None else None
        ),
    }


def summary(rows: list[dict]) -> dict:
    methods = sorted({name for row in rows for name in row["predictions"]})
    result = {}
    for name in methods:
        values = [
            row["predictions"][name] for row in rows if name in row["predictions"]
        ]
        errors = [
            v["absolute_percentage_error"]
            for v in values
            if v["absolute_percentage_error"] is not None
        ]
        coverage = [
            v["interval_covers_label"]
            for v in values
            if v["interval_covers_label"] is not None
        ]
        times = [v["elapsed_seconds"] for v in values]
        paired_v6_errors = [
            row["predictions"]["v6"]["absolute_percentage_error"]
            for row in rows
            if name in row["predictions"]
            and row["predictions"][name]["absolute_percentage_error"] is not None
        ]
        result[name] = {
            "attempted": len(values),
            "predicted": len(errors),
            "mape": statistics.fmean(errors) if errors else None,
            "v6_mape_on_same_predicted_rows": statistics.fmean(paired_v6_errors)
            if paired_v6_errors
            else None,
            "missing_prediction_rate": 1.0 - len(errors) / len(values)
            if values
            else None,
            "empirical_interval_coverage": statistics.fmean(coverage)
            if coverage
            else None,
            "fallback_rate": statistics.fmean(
                v.get("estimate_source") == "v6_fallback" for v in values
            ),
            "convergence_rate": statistics.fmean(
                v.get("posterior_converged") is True for v in values
            ),
            "mean_elapsed_seconds": statistics.fmean(times) if times else None,
            "max_elapsed_seconds": max(times, default=None),
        }
    return result


def controlled_blend(adapted, raw_v6, world, snapshot) -> PredictionResult:
    """Explicit harness override of missing live map proofs, reported as such."""
    predictor = object.__new__(LatestV6V2Predictor)
    predictor.adapter = SimpleNamespace(
        adapt=lambda _: AdaptedDecision(
            adapted.decision,
            (),
            (*adapted.warnings, "event_only_development_replay"),
            adapted.strong_condition_count,
        )
    )
    predictor.v6_model = SimpleNamespace(predict=lambda _: raw_v6)
    predictor.world_predictor = SimpleNamespace(predict=lambda _: world)
    predictor.bid_safety_factor = 0.9
    return predictor.predict(snapshot)


def evaluate(args) -> dict:
    import joblib

    source_root = Path(__file__).resolve().parents[1] / "apps/ephemeral-assistant/src"
    source_hashes_before = {
        public_path(path): sha256_file(path)
        for path in sorted(source_root.rglob("*.py"))
    }
    core = args.public_data / "core"
    selected = select_inputs(core, limit=args.limit, rounds=tuple(args.rounds))
    if not selected:
        raise ValueError("no complete public event-only decisions match selection")
    catalog = load_catalog(core / "treasures.csv")
    v6 = joblib.load(args.models / "latest-value-model-v6.joblib")
    adapter = OcrDecisionAdapter(catalog)
    excluded = {row["session_id"] for row in selected}
    rows = []
    with tempfile.TemporaryDirectory(prefix="auction-condition-dev-") as directory:
        prior_csv = Path(directory) / "treasures-without-evaluation-targets.csv"
        with (core / "treasures.csv").open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            with prior_csv.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(
                    row for row in reader if row["session_id"] not in excluded
                )
        initialized = time.perf_counter()
        if args.mode != "legacy":
            # Measure one-time module loading as initialization, not a failure
            # to find posterior seeds within the first decision's CPU budget.
            for module in (
                "ortools.sat.python.cp_model",
                "auction_moment_assistant.conditional_sampler",
                "auction_moment_assistant.conditional_constraints",
            ):
                importlib.import_module(module)
        predictors = {
            mode: EmpiricalWorldPredictor(
                prior_csv,
                catalog,
                world_model_path=args.models / "world-model-v2-S.json",
                prior_samples=args.prior_samples,
                conditional_sampling=mode == "conditional",
                conditional_budget_seconds=args.budget_seconds,
                sessions_csv=core / "sessions.csv",
                filter_empirical_quality=mode != "legacy",
            )
            for mode in (
                ("legacy", "conditional") if args.mode == "both" else (args.mode,)
            )
        }
        initialization_elapsed = time.perf_counter() - initialized
        for index, item in enumerate(selected, 1):
            snapshot, label = item["snapshot"], item["label"]
            adapted = adapter.adapt(snapshot)
            started = time.perf_counter()
            raw_v6 = v6.predict(adapted.decision)
            elapsed_v6 = time.perf_counter() - started
            v6_result = {
                "p10": int(raw_v6.get("p10") or raw_v6["prediction"]),
                "p50": int(raw_v6["prediction"]),
                "p90": int(raw_v6.get("p90") or raw_v6["prediction"]),
                "elapsed_seconds": elapsed_v6,
                "status": "controlled_event_only_v6",
            }
            predictions = {"v6": {**v6_result, **score(v6_result, label)}}
            for mode, predictor in predictors.items():
                started = time.perf_counter()
                world = predictor.predict(snapshot)
                elapsed = time.perf_counter() - started
                result = {**asdict(world), "elapsed_seconds": elapsed}
                predictions[f"{mode}_v2"] = {**result, **score(result, label)}
                if mode == "legacy" and world.p50 is not None:
                    weight = min(0.8, 0.4 + 0.1 * adapted.strong_condition_count)
                    blend = {
                        key: _geometric_blend(
                            v6_result[key], getattr(world, key), weight
                        )
                        for key in ("p10", "p50", "p90")
                    }
                    blend.update(
                        generation_weight=weight, elapsed_seconds=elapsed + elapsed_v6
                    )
                    predictions["legacy_40_80_blend"] = {**blend, **score(blend, label)}
                elif mode == "legacy":
                    result = {
                        **v6_result,
                        "estimate_source": "v6_fallback",
                        "elapsed_seconds": elapsed + elapsed_v6,
                    }
                    predictions["legacy_40_80_blend"] = {
                        **result,
                        **score(result, label),
                    }
                else:
                    blend = controlled_blend(adapted, raw_v6, world, snapshot)
                    result = {**asdict(blend), "elapsed_seconds": elapsed + elapsed_v6}
                    predictions["gated_blend"] = {**result, **score(result, label)}
            rows.append(
                {
                    **{key: value for key, value in item.items() if key != "snapshot"},
                    "round": snapshot.round_number,
                    "snapshot": asdict(snapshot),
                    "live_adapter_issues": list(adapted.issues),
                    "missing_reveal_outcomes": sum(
                        event["semantics"].get("reveal_outcome_available") is False
                        for event in snapshot.events
                    ),
                    "predictions": predictions,
                }
            )
            print(f"{index}/{len(selected)} {item['decision_key']}", flush=True)
    provenance_paths = [
        core / name
        for name in (
            "sessions.csv",
            "rounds.csv",
            "event_instances.csv",
            "treasures.csv",
        )
    ] + [
        args.models / name
        for name in ("latest-value-model-v6.joblib", "world-model-v2-S.json")
    ]
    source_hashes_after = {
        public_path(path): sha256_file(path)
        for path in sorted(source_root.rglob("*.py"))
    }
    common = [
        row
        for row in rows
        if all(
            name in row["predictions"] and row["predictions"][name]["p50"] is not None
            for name in ("legacy_v2", "conditional_v2")
        )
    ]
    return {
        "schema_version": SCHEMA,
        "development_only": True,
        "controlled_input_not_live_eligible": True,
        "production_enabled": False,
        "limitations": [
            "Frozen model training overlaps this public time window; no unseen accuracy claim.",
            "Public HMAC session IDs cannot reliably join model training IDs; membership unknown.",
            "All evaluation target sessions excluded from both empirical prior tables.",
            "No object observations, exact map height, reveal outcomes, or OCR screenshots available.",
            "Missing live map proofs explicitly bypassed only by this controlled development harness.",
            "Cannot measure reveal/layout gains, screen path quality, or forward calibration.",
            "Legacy is the current legacy-filter option, not a byte-for-byte beta.6 binary replay.",
            "Fusion thresholds are engineering gates and are not tuned on these outcomes.",
            "Module cold loading is recorded in initialization; per-decision inference uses warm imports.",
        ],
        "configuration": {
            "mode": args.mode,
            "limit": args.limit,
            "rounds": args.rounds,
            "prior_samples": args.prior_samples,
            "conditional_budget_seconds": args.budget_seconds,
        },
        "initialization_elapsed_seconds": initialization_elapsed,
        "provenance_sha256": {
            public_path(path): sha256_file(path) for path in provenance_paths
        },
        "runtime_source_sha256": source_hashes_before,
        "runtime_source_sha256_after": source_hashes_after,
        "runtime_source_changed_during_evaluation": source_hashes_before
        != source_hashes_after,
        "evaluation_script_sha256": sha256_file(Path(__file__)),
        "evaluation_target_count": len(excluded),
        "excluded_target_ids_sha256": hashlib.sha256(
            json.dumps(sorted(excluded)).encode()
        ).hexdigest(),
        "v6_training_session_count": len(v6.training_session_ids),
        "v6_training_decision_count": v6.training_decision_count,
        "summary": summary(rows),
        "common_predicted_subset": {
            "decisions": len(common),
            "summary": summary(common),
        },
        "decisions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-data", type=Path, default=Path("data/v1"))
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("both", "legacy", "conditional"), default="both"
    )
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--prior-samples", type=int, default=4096)
    parser.add_argument("--budget-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.limit <= 0 or args.prior_samples <= 0 or args.budget_seconds <= 0:
        parser.error("limit, prior-samples, and budget-seconds must be positive")
    if not set(args.rounds) <= set(range(1, 6)):
        parser.error("rounds must be in [1, 5]")
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(finite_json(report), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
