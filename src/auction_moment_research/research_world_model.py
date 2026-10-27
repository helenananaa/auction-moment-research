"""Offline terminal-world sampler for the two retained research candidates.

The JSON files contain normalized identity probabilities, exported from the
frozen fits. This module needs no private data, native library, or assistant.
Rejection here implements the fitted terminal density; it is not evidence
that the game itself retries. There is deliberately no runtime integration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

QUOTA = np.array([1, 1, 3, 3, 0], dtype=int)


def count_law(component: dict, family: str) -> np.ndarray:
    """Exact base P(N, extra colored, extra gold), before the value weight."""
    q = np.asarray(component["quality_probabilities"], float)
    pn = np.asarray(component["base_n_probabilities"], float)
    law = np.zeros((51, 53, 53))

    def append(a):
        out = a * q[:3].sum()
        out[1:, :] += a[:-1, :] * q[4]
        out[:, 1:] += a[:, :-1] * q[3]
        return out

    mass = np.zeros((53, 53))
    reserved = component["reserved_colored"]
    mass[reserved, 0] = 1.0
    for _ in range(2 - reserved):
        mass = append(mass)
    if family == "R2":
        for ni in range(51):
            law[ni] = pn[ni] * mass
            if ni < 50:
                mass = append(mass)
    elif family == "KG_mix2":
        k, g = np.indices((53, 53))
        lam = np.asarray(component["stop_rates"], float)
        survive = np.exp(-lam[0] * k - lam[1] * g - lam[2] * k * g)
        hazard = pn / np.cumsum(pn[::-1])[::-1]
        for ni in range(50):
            continuing = mass * (1 - hazard[ni]) * survive
            law[ni] = mass - continuing
            mass = append(continuing)
        law[50] = mass
    else:
        raise ValueError(f"Unsupported family: {family}")
    return law


def acceptance(model: dict, n, total) -> np.ndarray:
    gate = model["value_weight"]
    eta = (
        gate["tail_logit_intercept"]
        + gate["tail_logit_slope"] * (np.asarray(n) - 35) / 25
    )
    eps = np.exp(-np.logaddexp(0, -eta))
    x = (gate["threshold"] - np.asarray(total)) / gate["width"]
    return eps + (1 - eps) * np.exp(-np.logaddexp(0, -x))


def validate(model: dict) -> None:
    if model["schema_version"] != 1 or model["family"] not in ("R2", "KG_mix2"):
        raise ValueError("Unsupported research model")
    components = model["components"]
    if len(components) != 2:
        raise ValueError("Two latent components required")
    mix = np.array([c["weight"] for c in components])
    if not np.all(np.isfinite(mix)) or np.any(mix <= 0) or not np.isclose(mix.sum(), 1):
        raise ValueError("Invalid mixture")
    if len(model["catalog"]) != 5:
        raise ValueError("Five quality catalogs required")
    for z, component in enumerate(components):
        reserved = z if model["family"] == "R2" else 0
        if component["reserved_colored"] != reserved:
            raise ValueError("Invalid reserved slot")
        for name, size in (("quality_probabilities", 5), ("base_n_probabilities", 51)):
            values = np.asarray(component[name], float)
            if (
                values.shape != (size,)
                or not np.all(np.isfinite(values))
                or np.any(values <= 0)
                or not np.isclose(values.sum(), 1)
            ):
                raise ValueError(f"Invalid {name}")
        if model["family"] == "KG_mix2":
            rates = np.asarray(component["stop_rates"], float)
            if (
                rates.shape != (3,)
                or not np.all(np.isfinite(rates))
                or np.any(rates < 0)
            ):
                raise ValueError("Invalid stopping rates")
        ids = component["identity_probabilities"]
        if len(ids) != 5:
            raise ValueError("Five identity distributions required")
        for quality, catalog in enumerate(model["catalog"]):
            p = np.asarray(ids[quality], float)
            if (
                p.shape != (51, len(catalog))
                or not np.all(np.isfinite(p))
                or np.any(p < 0)
                or not np.allclose(p.sum(axis=1), 1)
            ):
                raise ValueError("Invalid identity distribution")
        law = count_law(component, model["family"])
        if (
            not np.all(np.isfinite(law))
            or law.min() < -1e-14
            or not np.isclose(law.sum(), 1, atol=1e-12)
        ):
            raise ValueError("Invalid count distribution")
    gate = model["value_weight"]
    if not all(np.isfinite(v) for v in gate.values()) or gate["width"] <= 0:
        raise ValueError("Invalid value weight")


def sample(model: dict, size: int, seed: int) -> dict[str, np.ndarray]:
    """Generate IID terminal N/counts/value draws, without claiming map layout."""
    validate(model)
    if size < 1:
        raise ValueError("size must be positive")
    rng = np.random.default_rng(seed)
    laws = [count_law(c, model["family"]).ravel() for c in model["components"]]
    weights = [c["weight"] for c in model["components"]]
    chunks = []
    kept = 0
    for _ in range(10000):
        batch = max(1024, min(65536, 2 * (size - kept)))
        z = rng.choice(2, batch, p=weights)
        n = np.empty(batch, int)
        counts = np.tile(QUOTA, (batch, 1))
        total = np.zeros(batch, np.int64)
        area = np.zeros(batch, np.int64)
        for component_index, component in enumerate(model["components"]):
            rows = np.flatnonzero(z == component_index)
            law = laws[component_index]
            picks = rng.choice(len(law), len(rows), p=law / law.sum())
            ni, k, g = np.unravel_index(picks, (51, 53, 53))
            n[rows] = ni + 10
            counts[rows, 4] += k
            counts[rows, 3] += g
            other = np.asarray(component["quality_probabilities"][:3])
            other /= other.sum()
            for remaining in np.unique(ni + 2 - k - g):
                selected = rows[ni + 2 - k - g == remaining]
                counts[selected, :3] += rng.multinomial(
                    int(remaining), other, len(selected)
                )
            for value_n in np.unique(n[rows]):
                group = rows[n[rows] == value_n]
                for quality, catalog in enumerate(model["catalog"]):
                    slots = np.repeat(group, counts[group, quality])
                    p = np.asarray(
                        component["identity_probabilities"][quality][value_n - 10]
                    )
                    item = rng.choice(len(p), len(slots), p=p / p.sum())
                    values = np.array([x["value"] for x in catalog], np.int64)
                    areas = np.array(
                        [x["width"] * x["height"] for x in catalog], np.int64
                    )
                    np.add.at(total, slots, values[item])
                    np.add.at(area, slots, areas[item])
        take = rng.random(batch) < acceptance(model, n, total)
        chunks.append(
            {
                "n": n[take],
                "counts": counts[take],
                "total": total[take],
                "area": area[take],
            }
        )
        kept += int(take.sum())
        if kept >= size:
            return {
                key: np.concatenate([c[key] for c in chunks])[:size]
                for key in chunks[0]
            }
    raise RuntimeError("Acceptance too low; no partial result returned")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    worlds = sample(model, args.worlds, args.seed)
    print(
        json.dumps(
            {
                "scope": "synthetic terminal worlds, not a historical replay or live validation",
                "family": model["family"],
                "seed": args.seed,
                "worlds": args.worlds,
                "mean_n": float(worlds["n"].mean()),
                "mean_total": float(worlds["total"].mean()),
                "total_p10_p50_p90": np.quantile(
                    worlds["total"], [0.1, 0.5, 0.9]
                ).tolist(),
                "fraction_above_seven_million": float(
                    (worlds["total"] > 7000000).mean()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
