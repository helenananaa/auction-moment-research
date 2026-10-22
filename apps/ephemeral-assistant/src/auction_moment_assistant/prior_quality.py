"""Historical worlds are eligible priors only after independent reconciliation.

Settlement labels audit historical identities here; they never enter current
decision features or world values. A label can remain valid for scoring even
when its postgame identity assignment is excluded from an empirical prior.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RULE_VERSION = "public-s-empirical-prior-reconciliation-v1"


def _true(value):
    return str(value or "").strip().lower() == "true"


def _integer(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def load_empirical_counts(
    treasures_csv, catalog, *, sessions_csv=None, quality_filter=True
):
    with Path(treasures_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        if row.get("session_id"):
            grouped[row["session_id"]].append(row)
    index = {item.catalog_id: i for i, item in enumerate(catalog)}
    metadata_path = (
        Path(sessions_csv)
        if sessions_csv is not None
        else Path(treasures_csv).with_name("sessions.csv")
    )
    metadata = {}
    session_keys = Counter()
    metadata_available = metadata_path.is_file()
    if metadata_available:
        with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = str(row.get("session_id") or "")
                session_keys[key] += 1
                metadata[key] = row
    rejected, accepted, verified = Counter(), [], 0
    for session_id, items in sorted(grouped.items()):
        failures = set()
        counts = np.zeros(len(catalog), dtype=np.int16)
        total = 0
        known_keys = [row.get("treasure_id") for row in items if row.get("treasure_id")]
        if len(known_keys) != len(set(known_keys)):
            failures.add("duplicate_item_key")
        for row in items:
            item_index = index.get(row.get("catalog_id"))
            if item_index is None:
                failures.add("unknown_identity")
                continue
            item = catalog[item_index]
            if (
                item.value <= 0
                or item.width not in (1, 2, 3)
                or item.height not in (1, 2, 3)
            ):
                failures.add("catalog_metadata_conflict")
            counts[item_index] += 1
            total += int(item.value)
            for field, expected in (
                ("value", item.value),
                ("width", item.width),
                ("height", item.height),
                ("area", item.width * item.height),
            ):
                if row.get(field) and _integer(row[field]) != expected:
                    failures.add("catalog_metadata_conflict")
            if row.get("quality") and row["quality"] != item.quality:
                failures.add("catalog_metadata_conflict")
            if row.get("auction_tier") and row["auction_tier"] != "S":
                failures.add("tier_not_s")
        session = metadata.get(session_id)
        labels = {
            _integer(row.get("settlement_total"))
            for row in items
            if row.get("settlement_total")
        }
        if labels and (len(labels) != 1 or next(iter(labels)) != total or total <= 0):
            failures.add("catalog_sum_not_settlement")
        if metadata_available:
            if session is None or session_keys[session_id] != 1:
                failures.add("missing_or_duplicate_session")
            else:
                if session.get("auction_tier") != "S":
                    failures.add("tier_not_s")
                if not _true(session.get("protocol_settlement_valid")):
                    failures.add("protocol_settlement_not_confirmed")
                if session.get("final_treasure_review_status") != "auto_confirmed":
                    failures.add("review_not_confirmed")
                for field in (
                    "value_sum_matches",
                    "postgame_value_gate_matches",
                    "catalog_independent_of_settlement_total",
                    "geometry_settlement_independent",
                ):
                    if not _true(session.get(field)):
                        failures.add("review_or_independence_flag_not_confirmed")
                if session.get("postgame_value_gate_status") != "matched":
                    failures.add("value_gate_not_matched")
                label = _integer(session.get("settlement_total"))
                if label is None or label <= 0 or total != label:
                    failures.add("catalog_sum_not_settlement")
                if _integer(session.get("treasure_count")) != len(items):
                    failures.add("count_not_reconciled")
                for field, expected in (
                    ("detected_value_sum", total),
                    (
                        "value_sum_difference",
                        total - label if label is not None else None,
                    ),
                ):
                    if session.get(field) and _integer(session[field]) != expected:
                        failures.add("reconciliation_metadata_contradiction")
        else:
            # A matching embedded label cannot establish independent identity
            # review. Legacy reproduction must explicitly disable filtering.
            failures.add("review_metadata_unavailable")
        if quality_filter and failures:
            rejected.update(failures)
            continue
        if counts.sum():
            accepted.append(counts)
            verified += int(bool(session) and not failures)
    result = np.asarray(accepted, dtype=np.int16).reshape(-1, len(catalog))
    return result, {
        "rule": RULE_VERSION
        if quality_filter
        else "legacy_unfiltered_for_reproduction",
        "input_worlds": len(grouped),
        "accepted_worlds": len(accepted),
        "excluded_worlds": len(grouped) - len(accepted),
        "verified_worlds": verified,
        "metadata_available": metadata_available,
        "exclusion_reason_counts_nonexclusive": dict(rejected),
        "used_in_conditional_target": False,
    }
