"""Fixed synthetic check of the OCR reveal adapter, without fitting anything.

The generated worlds and ideal observations share the frozen generator's
assumptions. This is a controlled feature-delivery benchmark, not evidence of
accuracy on new real games, OCR robustness, or a validated server generator.
The sample plan is fixed in this source before execution: all seeds 4000..4199,
alternating identity/quality reveals, and sample sizes 1..4 in a fixed cycle.
"""

from __future__ import annotations

# The standalone script bootstraps repository-local modules before imports.
# ruff: noqa: E402

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = ROOT / "apps/ephemeral-assistant/src"
sys.path[:0] = [str(APP_SOURCE), str(ROOT / "src")]

from auction_moment_assistant.latest_model import OcrDecisionAdapter, V6_CANDIDATE_ID
from auction_moment_assistant.models import load_catalog
from auction_moment_assistant.observations import ObservationSnapshot
from auction_moment_assistant.world_layout import pack_sequence
from generative_world_model_v2 import GenerativeWorldModelV2


SCHEMA = "ocr-reveal-adapter-fixed-synthetic-v1"
SEEDS = tuple(range(4000, 4200))
EFFECTS = ("reveal_identity_random", "reveal_position_quality_random")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def public_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"external/{path.name}"


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__).resolve(), *APP_SOURCE.rglob("*.py"), *(ROOT / "src").rglob("*.py")]
    return {public_path(path): sha256_file(path) for path in sorted(paths)}


def synthetic_snapshot(generator, catalog, *, seed: int, case_index: int):
    """Generate a world, then expose only the prescribed decision-time facts."""
    rng = random.Random(seed)
    counts = generator.sample_prior_counts(rng)
    sequence = [index for index, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(sequence)
    layout = pack_sequence(sequence, catalog)
    effect = EFFECTS[case_index % len(EFFECTS)]
    sample_size = case_index % 4 + 1
    selected = tuple(sorted(rng.sample(range(len(sequence)), sample_size)))
    observations = []
    for index in selected:
        placement = layout.placements[index]
        item = catalog[placement.catalog_index]
        identity_reveal = effect == "reveal_identity_random"
        observations.append({
            "row": placement.row,
            "column": placement.column,
            "width": item.width if identity_reveal else None,
            "height": item.height if identity_reveal else None,
            "quality": item.quality,
            "catalog_id": item.catalog_id if identity_reveal else None,
            "spatial": "complete" if identity_reveal else "top_left",
            "confidence": 0.95,
            "human_locked": False,
            "first_seen_round": 1,
            "source": "synthetic_ideal_reveal_outcome",
        })
    snapshot = ObservationSnapshot(
        revision=1,
        round_number=1,
        bankroll=None,
        events=(
            {
                "round_number": 1, "kind": "public", "confidence": 0.95,
                "human_locked": False, "source": "synthetic_prescribed_event",
                "semantics": {"parsed": True, "effect": effect, "count": sample_size},
            },
            {
                "round_number": 1, "kind": "personal", "confidence": 0.95,
                "human_locked": False, "source": "synthetic_prescribed_event",
                "semantics": {
                    "parsed": True, "effect": "total_item_count",
                    "observed_count": len(sequence),
                },
            },
        ),
        map_items=tuple(observations),
        completeness_confirmed=True,
        completeness_source="synthetic_complete_reveal_scan",
        map_rows=layout.used_rows,
        map_height_exact=True,
        map_rows_source="synthetic_generated_exact_height",
        pre_bid_confirmed=True,
        pre_bid_source="synthetic_controlled_pre_bid",
    )
    return snapshot, counts, selected, effect, sample_size


def metrics(cases: list[dict], prediction_field: str) -> dict:
    paired = [case for case in cases if case["status"] == "ok"]
    if not paired:
        return {"case_count": 0, "excluded_failed_count": len(cases)}
    actual = [float(case["actual_value"]) for case in paired]
    errors = [float(case[prediction_field]) - value for case, value in zip(paired, actual)]
    absolute = [abs(error) for error in errors]
    return {
        "case_count": len(paired),
        "excluded_failed_count": len(cases) - len(paired),
        "mape_percent": 100 * statistics.fmean(error / value for error, value in zip(absolute, actual)),
        "wape_percent": 100 * sum(absolute) / sum(actual),
        "mae_currency_units": statistics.fmean(absolute),
        "bias_currency_units": statistics.fmean(errors),
        "bias_percent_of_total_actual": 100 * sum(errors) / sum(actual),
    }


def comparison(cases: list[dict]) -> dict:
    paired = [case for case in cases if case["status"] == "ok"]
    return {
        "baseline": metrics(cases, "baseline_prediction"),
        "attributed": metrics(cases, "attributed_prediction"),
        "response_count": sum(case["prediction_responded"] for case in paired),
        "response_rate": statistics.fmean(case["prediction_responded"] for case in paired) if paired else None,
        "improved_absolute_error_count": sum(case["absolute_error_delta"] < 0 for case in paired),
        "worsened_absolute_error_count": sum(case["absolute_error_delta"] > 0 for case in paired),
        "unchanged_absolute_error_count": sum(case["absolute_error_delta"] == 0 for case in paired),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models/latest-model-v6-v2.0.0-beta.1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/ocr-reveal-adapter-synthetic-20261019.json")
    args = parser.parse_args()
    v6_path = args.models / "latest-value-model-v6.joblib"
    generator_path = args.models / "world-model-v2-S.json"
    catalog_path = ROOT / "data/v1/core/treasures.csv"
    sources_before = source_hashes()
    artifacts_before = {
        public_path(path): sha256_file(path)
        for path in (v6_path, generator_path, catalog_path)
    }
    print(json.dumps({
        "schema_version": SCHEMA,
        "frozen_script_sha256": sha256_file(Path(__file__)),
        "planned_seed_count": len(SEEDS),
        "planned_seeds_sha256": canonical_sha256(SEEDS),
    }), flush=True)

    import joblib

    catalog = load_catalog(catalog_path)
    generator = GenerativeWorldModelV2.load(generator_path, catalog)
    v6 = joblib.load(v6_path)
    if v6.candidate_id != V6_CANDIDATE_ID:
        raise RuntimeError("frozen v6 candidate id mismatch")
    adapter = OcrDecisionAdapter(catalog)
    cases = []
    started = time.monotonic()
    for case_index, seed in enumerate(SEEDS):
        case = {"seed": seed, "case_index": case_index, "status": "failed"}
        try:
            snapshot, counts, selected, effect, sample_size = synthetic_snapshot(
                generator, catalog, seed=seed, case_index=case_index
            )
            case.update(effect=effect, sample_size=sample_size,
                        total_item_count=sum(counts), map_rows=snapshot.map_rows,
                        sampled_entry_indices=list(selected))
            adapted = adapter.adapt(snapshot)
            if not adapted.accepted:
                raise RuntimeError(f"synthetic snapshot unexpectedly blocked: {adapted.issues}")
            current = adapted.decision
            previous = copy.deepcopy(current)
            for item in previous["visible_items"]:
                item["history"] = []
            previous.pop("reveal_attribution", None)
            baseline_prediction = v6.predict(previous)
            attributed_prediction = v6.predict(current)
            # Outcome scoring occurs only after both feature planes and their
            # predictions are materialized. Neither receives the true total.
            actual = sum(count * item.value for count, item in zip(counts, catalog))
            baseline = int(baseline_prediction["prediction"])
            attributed = int(attributed_prediction["prediction"])
            if not all(math.isfinite(value) and value > 0 for value in (actual, baseline, attributed)):
                raise RuntimeError("nonpositive or nonfinite synthetic outcome")
            case.update(
                status="ok", actual_value=actual, baseline_prediction=baseline,
                attributed_prediction=attributed, prediction_responded=baseline != attributed,
                absolute_error_delta=abs(attributed - actual) - abs(baseline - actual),
                baseline_decision_sha256=canonical_sha256(previous),
                attributed_decision_sha256=canonical_sha256(current),
                baseline_identity_sample=baseline_prediction.get("random_identity_sample"),
                attributed_identity_sample=attributed_prediction.get("random_identity_sample"),
                baseline_position_quality_sample=baseline_prediction.get("position_quality_sample"),
                attributed_position_quality_sample=attributed_prediction.get("position_quality_sample"),
            )
        except Exception as exc:
            case["error"] = f"{type(exc).__name__}:{exc}"
        cases.append(case)

    sources_after = source_hashes()
    artifacts_after = {
        public_path(path): sha256_file(path)
        for path in (v6_path, generator_path, catalog_path)
    }
    stable = sources_before == sources_after and artifacts_before == artifacts_after
    report = {
        "schema_version": SCHEMA,
        "experiment_kind": "controlled_synthetic_feature_delivery_benchmark",
        "limitations": [
            "Generated truth and ideal reveal observations use the same frozen generator assumptions.",
            "This is not new real-game evidence, an OCR accuracy test, or server-rule validation.",
            "No model was trained; no threshold, weight, seed, or case was selected from outcomes.",
            "The sampler/fusion is not exercised; both arms use the same frozen v6 model.",
            "Effects alternate and sample sizes cycle as prescribed; their strata are not balanced independently.",
        ],
        "plan": {
            "first_seed": SEEDS[0], "last_seed_inclusive": SEEDS[-1],
            "case_count": len(SEEDS), "seed_list_sha256": canonical_sha256(SEEDS),
            "world_draw": "frozen_generator.sample_prior_counts(random.Random(seed))",
            "entry_order": "uniform_shuffle_of_generated_multiset",
            "layout": "ten_column_row_major_first_fit_without_rotation",
            "reveal_selection": "uniform_without_replacement_from_entry_indices",
            "effect_cycle": list(EFFECTS), "sample_size_cycle": [1, 2, 3, 4],
            "other_event": "personal_total_item_count_exact_generated_N",
            "observation_confidence": 0.95, "human_locked": False,
            "baseline_transform": "clear_visible_item_history_and_remove_reveal_attribution",
            "case_exclusion_policy": "none_all_planned_cases_retained_failures_explicit",
        },
        "candidate_id": v6.candidate_id,
        "runtime_source_sha256": sources_before,
        "runtime_source_sha256_after": sources_after,
        "artifact_sha256": artifacts_before,
        "artifact_sha256_after": artifacts_after,
        "sources_and_artifacts_unchanged_during_run": stable,
        "runtime": {"python": platform.python_version(),
                    **{name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "joblib")}},
        "duration_seconds": time.monotonic() - started,
        "overall": comparison(cases),
        "by_effect": {effect: comparison([case for case in cases if case.get("effect") == effect]) for effect in EFFECTS},
        "failure_count": sum(case["status"] != "ok" for case in cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": public_path(args.output), "overall": report["overall"],
                      "failure_count": report["failure_count"], "source_stable": stable}, ensure_ascii=False))
    return 0 if stable and report["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
