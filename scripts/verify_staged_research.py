"""Verify the public stage bundle, not the unpublished full historical replay."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from auction_moment_research.research_world_model import validate  # noqa: E402

BUNDLE = ROOT / "results" / "world-model-stage-20261027"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    manifest = read(BUNDLE / "manifest.json")
    for record in manifest["outputs"]:
        path = (BUNDLE / record["path"]).resolve()
        require(path.is_relative_to(ROOT.resolve()), "Manifest path outside repository")
        content = path.read_bytes()
        require(len(content) == record["bytes"], f"Size mismatch: {record['path']}")
        require(
            hashlib.sha256(content).hexdigest() == record["sha256"],
            f"Hash mismatch: {record['path']}",
        )

    replay = read(BUNDLE / "historical-replay.json")
    scores = replay["scores"]
    require(
        scores["truth_games"] == 74 and scores["truth_decisions"] == 277,
        "Wrong replay population",
    )
    for name, arm in scores["arms"].items():
        row = arm["all"]
        require(
            row["valid_decisions"] == 277 and row["failed_decisions"] == 0,
            f"Incomplete arm: {name}",
        )
        require(sum(row["status_counts"].values()) == 277, "Status counts do not sum")
        metrics = row["game_weighted"]
        require(all(math.isfinite(v) for v in metrics.values()), "Non-finite score")
        require(
            metrics["interval_score80"] >= metrics["interval_width"],
            "Interval penalty must be nonnegative",
        )
        require(0 <= metrics["covered80"] <= 1, "Invalid coverage")
    unstable = sum(277 - v["stable"] for v in replay["numerical"].values())
    require(unstable == 119, "Numerical failures were omitted or changed")
    for label, comparison in scores["comparisons"].items():
        left, right = label.split(" minus ")
        require(comparison["paired_games"] == 74, "Incomplete paired comparison")
        for metric, result in comparison["metrics"].items():
            delta = (
                scores["arms"][left]["all"]["game_weighted"][metric]
                - scores["arms"][right]["all"]["game_weighted"][metric]
            )
            require(
                math.isclose(
                    delta, result["left_minus_right"], rel_tol=1e-9, abs_tol=1e-8
                ),
                "Paired difference mismatch",
            )
    base = scores["arms"]["v2"]["all"]["game_weighted"]["interval_score80"]
    gains = {
        a: 100
        * (1 - scores["arms"][a]["all"]["game_weighted"]["interval_score80"] / base)
        for a in ("R2", "KG_mix2")
    }

    recovery = read(BUNDLE / "mechanism-recovery.json")
    require(len(recovery["trials"]) == 40, "Wrong recovery population")
    require(len({r["seed"] for r in recovery["trials"]}) == 40, "Duplicate trial seed")
    for truth, group in recovery["groups"].items():
        rows = [r for r in recovery["trials"] if r["truth"] == truth]
        require(
            len(rows) == 20 and {r["replicate"] for r in rows} == set(range(20)),
            "Missing recovery trial",
        )
        for row in rows:
            require(
                row["qualified"] and row["train"] == 655 and row["test"] == 987,
                "Unqualified recovery trial",
            )
            require(
                row["selected"] == max(row["scores"], key=row["scores"].get),
                "Selection disagrees with score",
            )
            require(
                row["correct"] == (row["selected"] == truth), "Incorrect recovery label"
            )
        require(
            sum(r["correct"] for r in rows) == group["correct"],
            "Recovery count mismatch",
        )
        require(
            sum(r["numerically_ambiguous"] for r in rows)
            == group["numerically_ambiguous"],
            "Ambiguity count mismatch",
        )
        p = group["correct"] / 20
        z = 1.959963984540054
        center = (p + z * z / 40) / (1 + z * z / 20)
        half = z * math.sqrt(p * (1 - p) / 20 + z * z / 1600) / (1 + z * z / 20)
        require(
            all(
                math.isclose(a, b, abs_tol=1e-12)
                for a, b in zip(
                    [center - half, center + half], group["wilson95_among_qualified"]
                )
            ),
            "Wilson interval mismatch",
        )

    comparison = read(BUNDLE / "mechanism-comparison.json")
    observed = comparison["observed"]
    require(
        sum(r["maps"] for r in observed["later987"].values()) == 987,
        "Wrong terminal comparison population",
    )
    require(
        sum(r["over7m"] for r in observed["later987"].values())
        == observed["tail"][0]
        == 13,
        "Tail observation mismatch",
    )
    for row in comparison["screening"]:
        models = comparison["models"][row["family"]]
        candidate = models[f"{row['correction']}/{row['variant']}"]["score"]
        require(
            math.isclose(
                candidate - models["BASE/G"]["score"],
                row["gain_vs_BASE_G"],
                abs_tol=1e-8,
            ),
            "Ablation gain mismatch",
        )

    with (ROOT / "data/v1/core/treasures.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        catalog = {
            r["catalog_id"]: (int(r["value"]), int(r["width"]), int(r["height"]))
            for r in csv.DictReader(handle)
        }
    for family in ("R2", "KG_mix2"):
        model = read(
            ROOT / "models/research-world-model-stage-20261027" / f"{family}.json"
        )
        require(
            model["family"] == family and model["training_games"] == 1556,
            "Wrong model provenance",
        )
        validate(model)
        for group in model["catalog"]:
            for item in group:
                require(
                    catalog[item["catalog_id"]]
                    == (item["value"], item["width"], item["height"]),
                    "Model/catalog mismatch",
                )
    return {
        "success": True,
        "verified_files": len(manifest["outputs"]),
        "predictions": 831,
        "numerically_unresolved_predictions": unstable,
        "recovery_trials": 40,
        "interval_score_improvement_percent": gains,
        "scope": "Frozen aggregates, model probabilities and file integrity; no new historical refit or live validation.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
