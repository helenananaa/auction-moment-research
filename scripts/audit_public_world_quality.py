"""Audit public settlement-world reconciliation without training or inference.

The empirical-prior gate is stricter than the event-only label evaluation gate:
an independently valid protocol settlement may remain a usable label when its
postgame catalog assignment is wrong. Never filter benchmark rows by errors.
All provenance paths in the receipt are public repository-relative paths.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RULE_VERSION = "public-s-empirical-prior-reconciliation-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"external/{path.name}"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def integer(value) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def is_true(value) -> bool:
    return str(value or "").strip().lower() == "true"


def summary_group(records: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[str(record.get(key) or "missing")].append(record)
    return [
        {
            key: value,
            "worlds": len(group),
            "sum_mismatch_worlds": sum(not r["sum_matches_label"] for r in group),
            "prior_eligible_worlds": sum(r["prior_eligible"] for r in group),
            "protocol_label_eligible_worlds": sum(r["protocol_label_eligible"] for r in group),
        }
        for value, group in sorted(groups.items())
    ]


def audit(public_data: Path, evaluation_receipt: Path | None = None) -> dict:
    core = public_data / "core"
    sessions_path, treasures_path = core / "sessions.csv", core / "treasures.csv"
    sessions, treasures = read_rows(sessions_path), read_rows(treasures_path)
    session_keys = Counter(row["session_id"] for row in sessions)
    sessions_by_id = {row["session_id"]: row for row in sessions}
    treasure_keys = Counter((row["session_id"], row["treasure_id"]) for row in treasures)
    duplicate_keys_by_world = Counter()
    for (session_id, _), count in treasure_keys.items():
        if count > 1:
            duplicate_keys_by_world[session_id] += count - 1
    worlds = defaultdict(list)
    catalog = defaultdict(set)
    for row in treasures:
        worlds[row["session_id"]].append(row)
        catalog[row["catalog_id"]].add(
            (row["quality"], row["width"], row["height"], row["value"])
        )
    conflicting_catalog_ids = {key for key, values in catalog.items() if len(values) != 1}
    records = []
    for session_id, items in sorted(worlds.items()):
        session = sessions_by_id.get(session_id, {})
        label = integer(session.get("settlement_total"))
        values = [integer(row["value"]) for row in items]
        value_sum = sum(value for value in values if value is not None)
        duplicate_item_keys = duplicate_keys_by_world[session_id]
        numeric_valid = all(
            integer(row["width"]) in (1, 2, 3)
            and integer(row["height"]) in (1, 2, 3)
            and integer(row["area"]) == integer(row["width"]) * integer(row["height"])
            and integer(row["value"]) is not None
            and integer(row["value"]) > 0
            and row["quality"] in ("白", "蓝", "紫", "金", "彩")
            for row in items
        )
        sum_matches = label is not None and value_sum == label
        flag_contradictions = []
        for field in ("value_sum_matches", "postgame_value_gate_matches"):
            if session.get(field) and is_true(session[field]) != sum_matches:
                flag_contradictions.append(field)
        if session.get("postgame_value_gate_status") in ("matched", "mismatched"):
            if (session["postgame_value_gate_status"] == "matched") != sum_matches:
                flag_contradictions.append("postgame_value_gate_status")
        for field, expected in (
            ("detected_value_sum", value_sum),
            ("value_sum_difference", value_sum - label if label is not None else None),
            ("treasure_count", len(items)),
        ):
            if session.get(field) and integer(session[field]) != expected:
                flag_contradictions.append(field)
        checks = {
            "session_missing_or_duplicate": session_keys[session_id] != 1,
            "treasure_key_duplicate": duplicate_item_keys > 0,
            "treasure_label_conflict": any(
                row.get("settlement_total")
                and integer(row["settlement_total"]) != label
                for row in items
            ),
            "tier_not_s_or_conflicting": session.get("auction_tier") != "S"
            or any(row["auction_tier"] != "S" for row in items),
            "invalid_catalog_values_or_geometry": not numeric_valid,
            "catalog_id_conflicting": any(row["catalog_id"] in conflicting_catalog_ids for row in items),
            "settlement_missing_or_nonpositive": label is None or label <= 0,
            "catalog_sum_not_settlement": not sum_matches,
            "item_count_not_session_count": integer(session.get("treasure_count")) != len(items),
            "review_not_auto_confirmed": session.get("final_treasure_review_status") != "auto_confirmed",
            "value_match_flags_not_confirmed": not is_true(session.get("value_sum_matches"))
            or not is_true(session.get("postgame_value_gate_matches"))
            or session.get("postgame_value_gate_status") != "matched",
            "identity_not_settlement_independent": not is_true(session.get("catalog_independent_of_settlement_total")),
            "geometry_not_settlement_independent": not is_true(session.get("geometry_settlement_independent")),
            "protocol_settlement_not_confirmed": not is_true(session.get("protocol_settlement_valid")),
            "reconciliation_metadata_contradiction": bool(flag_contradictions),
        }
        failed = [key for key, failed in checks.items() if failed]
        records.append(
            {
                "session_id": session_id,
                "date": session.get("date"),
                "auction_tier": session.get("auction_tier"),
                "item_count": len(items),
                "catalog_value_sum": value_sum,
                "settlement_label": label,
                "value_difference": value_sum - label if label is not None else None,
                "sum_matches_label": sum_matches,
                "protocol_label_eligible": session_keys[session_id] == 1
                and is_true(session.get("protocol_settlement_valid"))
                and label is not None and label > 0,
                "prior_eligible": not failed,
                "prior_eligible_without_protocol_requirement": not [
                    reason for reason in failed if reason != "protocol_settlement_not_confirmed"
                ],
                "prior_exclusion_reasons": failed,
                "metadata_contradictions": flag_contradictions,
                "parse_quality_status": session.get("parse_quality_status"),
                "postgame_value_gate_status": session.get("postgame_value_gate_status"),
                "final_treasure_review_status": session.get("final_treasure_review_status"),
            }
        )
    by_id = {record["session_id"]: record for record in records}
    provenance = {public_path(path): sha256(path) for path in (sessions_path, treasures_path, Path(__file__))}
    runtime_prior = ROOT / "apps/ephemeral-assistant/src/auction_moment_assistant/prior_quality.py"
    frozen_generator = ROOT / "models/latest-model-v6-v2.0.0-beta.1/world-model-v2-S.json"
    for path in (runtime_prior, frozen_generator):
        if path.is_file():
            provenance[public_path(path)] = sha256(path)
    evaluation_rows = []
    if evaluation_receipt is not None:
        provenance[public_path(evaluation_receipt)] = sha256(evaluation_receipt)
        receipt = json.loads(evaluation_receipt.read_text(encoding="utf-8"))
        for decision in receipt["decisions"]:
            record = by_id.get(decision["session_id"], {})
            evaluation_rows.append(
                {
                    "decision_key": decision["decision_key"],
                    "session_id": decision["session_id"],
                    "event_only_label_kept": True,
                    "protocol_label_eligible": record.get("protocol_label_eligible", False),
                    "catalog_sum_matches_label": record.get("sum_matches_label", False),
                    "prior_eligible": record.get("prior_eligible", False),
                    "catalog_value_sum": record.get("catalog_value_sum"),
                    "settlement_label": record.get("settlement_label"),
                    "value_difference": record.get("value_difference"),
                    "exclusion_from_prior_does_not_exclude_label": record.get("protocol_label_eligible", False)
                    and not record.get("prior_eligible", False),
                }
            )
    prior = [record for record in records if record["prior_eligible"]]
    stages = [
        ("unique_join_and_grain", ("session_missing_or_duplicate", "treasure_key_duplicate")),
        ("s_tier", ("tier_not_s_or_conflicting",)),
        ("catalog_valid_and_consistent", ("invalid_catalog_values_or_geometry", "catalog_id_conflicting")),
        ("positive_protocol_settlement", ("settlement_missing_or_nonpositive", "protocol_settlement_not_confirmed")),
        ("catalog_and_count_reconciled", ("catalog_sum_not_settlement", "item_count_not_session_count", "treasure_label_conflict")),
        ("review_and_value_flags_confirmed", ("review_not_auto_confirmed", "value_match_flags_not_confirmed")),
        ("identity_and_geometry_independent", ("identity_not_settlement_independent", "geometry_not_settlement_independent")),
        ("reconciliation_metadata_consistent", ("reconciliation_metadata_contradiction",)),
    ]
    remaining, nested = records, []
    for stage, failures in stages:
        before = len(remaining)
        remaining = [r for r in remaining if not set(r["prior_exclusion_reasons"]).intersection(failures)]
        nested.append({"stage": stage, "input_worlds": before, "excluded_at_stage": before - len(remaining), "remaining_worlds": len(remaining)})
    return {
        "schema_version": "public-world-quality-audit-v1",
        "rule_version": RULE_VERSION,
        "scope": "Public CSV only; no raw/private/future source, training, or prediction rerun.",
        "grain": {"sessions": "session_id", "treasures": ["session_id", "treasure_id"], "prior_world": "session_id"},
        "provenance_sha256": provenance,
        "denominators": {
            "session_rows": len(sessions),
            "unique_session_ids": len(session_keys),
            "treasure_rows": len(treasures),
            "treasure_worlds": len(worlds),
            "sessions_without_treasure_world": len(set(session_keys) - set(worlds)),
            "orphan_treasure_worlds": len(set(worlds) - set(session_keys)),
            "duplicate_session_ids": sum(n > 1 for n in session_keys.values()),
            "duplicate_treasure_keys": sum(n > 1 for n in treasure_keys.values()),
            "conflicting_catalog_ids": len(conflicting_catalog_ids),
            "sum_mismatch_worlds": sum(not r["sum_matches_label"] for r in records),
            "protocol_label_eligible_worlds": sum(r["protocol_label_eligible"] for r in records),
            "protocol_label_eligible_but_sum_mismatch_worlds": sum(
                r["protocol_label_eligible"] and not r["sum_matches_label"] for r in records
            ),
            "prior_eligible_worlds": len(prior),
            "prior_excluded_worlds": len(worlds) - len(prior),
            "prior_eligible_without_protocol_requirement": sum(r["prior_eligible_without_protocol_requirement"] for r in records),
            "metadata_contradiction_worlds": sum(bool(r["metadata_contradictions"]) for r in records),
        },
        "prior_rule": {
            "required": ["unique session and item keys", "S tier in both tables", "valid consistent catalog identities and geometry", "positive independent protocol settlement", "exact item count and catalog sum reconciliation", "auto_confirmed review and all value-match flags", "independent identity and geometry", "no reconciliation metadata contradiction"],
            "failed_reason_counts_nonexclusive": dict(Counter(reason for record in records for reason in record["prior_exclusion_reasons"])),
            "eligible_world_ids_sha256": hashlib.sha256("\n".join(r["session_id"] for r in prior).encode()).hexdigest(),
            "nested_counts": nested,
        },
        "flag_counts": {
            field: dict(Counter(row.get(field) or "missing" for row in sessions if row["session_id"] in worlds))
            for field in ("protocol_settlement_valid", "value_sum_matches", "postgame_value_gate_status", "postgame_value_gate_matches", "final_treasure_review_status", "parse_quality_status")
        },
        "by_tier": summary_group(records, "auction_tier"),
        "by_date": summary_group(records, "date"),
        "by_parse_quality": summary_group(records, "parse_quality_status"),
        "metadata_contradictions": [r for r in records if r["metadata_contradictions"]],
        "evaluation_decisions": evaluation_rows,
        "evaluation_rule": "Keep all frozen decisions and independent positive protocol labels. Identity reconciliation is a prior eligibility rule, not a post-hoc benchmark exclusion.",
        "frozen_generator_limit": "The frozen v2 artifact states trusted_protocol_settlement/1556 training sessions but omits joinable IDs. Public CSV conflicts do not establish which fitted parameters were contaminated. No refit is performed and no clean-training claim is made.",
        "world_records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-data", type=Path, default=ROOT / "data/v1")
    parser.add_argument("--evaluation-receipt", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "build/public-world-quality-audit.json")
    args = parser.parse_args()
    report = audit(args.public_data, args.evaluation_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": public_path(args.output), **report["denominators"]}, indent=2))


if __name__ == "__main__":
    main()
