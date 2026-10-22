"""Exact-constraint posterior sampler for a world-model-v2-shaped prior.

Every hard clue holds in every returned world; nothing is ever relaxed.

Prior (world-model-v2): N -> rainbow count c | N -> non-rainbow extras
~ Multinomial | N, c -> identities within each quality ~ Multinomial(n_q, pi_q),
pi_q uniform for non-rainbow and N-dependent for rainbow.

1. Quality counts: sampled exactly from the prior given exact quality
   counts and the highest quality (closed-form multinomial marginals), with
   minimums implied by reveals.
2. Within a quality, given n_q:
   * caps from max value / max value per cell / largest item area / highest
     quality shrink the allowed identity set (weight pi(A)^n_q);
   * revealed identities, and a single item named by "max value" or
     "max value per cell", are placed directly; the multinomial-to-forced
     ratio is applied exactly (revealed: 1/r!; at-least-one: 1/m_i);
   * an exact quality total value is sampled exactly by backward sampling
     a dynamic program over value sums; an exact average area likewise over
     area sums (by rejection when both apply).
3. Clues spanning qualities (total cells, size counts, "at least one item
   of area a") are met by exact matching: four qualities are drawn from
   their candidate pools and the fifth is looked up among candidates whose
   statistics equal the remaining deficit.
4. A size-average clue is met clue-first: each quality's size-X items are
   enumerated exactly and only combinations whose average floors to the
   shown value are drawn (k-level tables cached per game, with provably
   negligible cells skipped under a recorded bound).
5. Reveals are scored as one uniform draw of all distinct revealed items;
   map height uses an external log-likelihood on total cells.

With a time budget, work that is cut short is either an unbiased priority
sample flagged low_ess, or a dropped extra round; ``cancel`` (a callable)
aborts a run with Cancelled, leaving every cache consistent.

Remaining approximations: reveal events are pooled into one draw, positions
and layout are not used, and everything is Monte Carlo.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammaln

NQ = 5
COLORED = 4
NSIZES = 9


class Cancelled(Exception):
    """Raised inside sample_posterior when its ``cancel`` callable returns True."""


def _check_cancel(ctx):
    cancel = getattr(ctx, "cancel", None)
    if cancel is not None and cancel():
        raise Cancelled()


@dataclass
class Prior:
    values: np.ndarray
    areas: np.ndarray
    quality: np.ndarray
    size: np.ndarray
    count_p: dict
    quotas: tuple
    colored_p: Callable
    noncolored_p: Callable
    identity_p: Callable
    # Optional world-level log weight (e.g. a maximum-entropy correction);
    # called as world_log_tilt(N, rainbow_count, rainbow_value, total_value).
    world_log_tilt: Callable | None = None

    def __post_init__(self):
        self.values = np.asarray(self.values, np.int64)
        self.areas = np.asarray(self.areas, np.int64)
        self.quality = np.asarray(self.quality, np.int64)
        self.size = np.asarray(self.size, np.int64)
        self.by_quality = [np.flatnonzero(self.quality == q) for q in range(NQ)]
        self.vpc = self.values // self.areas
        if self.quotas[COLORED] != 0:
            raise ValueError("rainbow quota must be zero")


@dataclass
class Evidence:
    count: dict = field(default_factory=dict)
    size_count: dict = field(default_factory=dict)
    highest: int | None = None
    largest_area: int | None = None
    max_value: int | None = None
    max_per_cell: int | None = None
    total_cells: int | None = None
    avg_area: dict = field(default_factory=dict)        # q -> (shown, scale)
    total_value: dict = field(default_factory=dict)     # q -> V
    size_avg_value: dict = field(default_factory=dict)  # s -> shown floor average
    revealed_identities: Counter = field(default_factory=Counter)
    revealed_qualities: Counter = field(default_factory=Counter)
    map_rows: int | None = None


@dataclass
class Result:
    status: str
    totals: np.ndarray
    weights: np.ndarray
    diagnostics: dict

    @property
    def ess(self):
        w = self.weights
        return float(w.sum() ** 2 / (w ** 2).sum()) if w.size and w.sum() > 0 else 0.0

    def quantile(self, p, reweight=None):
        w = self.weights if reweight is None else self.weights * reweight
        order = np.argsort(self.totals)
        cdf = np.cumsum(w[order]) / w.sum()
        return float(self.totals[order][min(np.searchsorted(cdf, p), len(order) - 1)])

    def summary(self):
        if self.status == "no_support":
            return {"status": self.status, "ess": 0.0, **self.diagnostics}
        t, w = self.totals, self.weights / self.weights.sum()
        out = {"status": self.status, "ess": self.ess, "mean": float((t * w).sum()),
               "p10": self.quantile(0.1), "p50": self.quantile(0.5), "p90": self.quantile(0.9),
               "mape_opt": self.quantile(0.5, reweight=1.0 / t)}
        for name in ("p50", "mape_opt"):
            out[f"expected_ape_{name}"] = float((w * np.abs(out[name] - t) / t).sum())
        out.update(self.diagnostics)
        return out


# --------------------------------------------------------------- sum DP
class SumDP:
    """P(sum of k iid draws == s) for s <= target, one rescaled row per k.

    Row k is stored only over its feasible range [k*min, min(target, k*max)]
    (values are positive), which is where all of the work used to go to waste.
    """

    def __init__(self, item_sums, probs, target):
        keep = np.asarray(probs, float) > 0
        self.items = np.asarray(item_sums, np.int64)
        self.probs = np.asarray(probs, float)
        self.target = int(target)
        live = self.items[keep]
        self.vmin = int(live.min()) if len(live) else 0
        self.vmax = int(live.max()) if len(live) else 0
        self.rows = [(0, np.ones(1))]
        self.log_scale = [0.0]

    def bounds(self, k):
        return k * self.vmin, min(self.target, k * self.vmax)

    def ensure(self, kmax):
        while len(self.rows) <= kmax:
            k = len(self.rows)
            lo, hi = self.bounds(k)
            plo, prev = self.rows[-1]
            if hi < lo or not len(prev):
                self.rows.append((lo, np.zeros(0)))
                self.log_scale.append(-math.inf)
                continue
            cur = _dp_row(prev, plo, self.items, self.probs, lo, hi)
            top = cur.max() if len(cur) else 0.0
            if top > 0:
                cur /= top
            self.rows.append((lo, cur))
            self.log_scale.append(self.log_scale[-1] + (math.log(top) if top > 0 else -math.inf))

    def _get(self, k, s):
        lo, row = self.rows[k]
        idx = np.asarray(s) - lo
        ok = (idx >= 0) & (idx < len(row))
        return np.where(ok, row[np.clip(idx, 0, max(len(row) - 1, 0))] if len(row) else 0.0, 0.0)

    def log_prob(self, k, s):
        lo, hi = self.bounds(k)
        if not lo <= s <= hi:
            return -math.inf
        self.ensure(k)
        value = float(self._get(k, s))
        return math.log(value) + self.log_scale[k] if value > 0 else -math.inf

    def range_mass(self, k, lo, hi):
        """Rescaled P(sum == s) for s in [lo, hi] and the row's log scale."""
        self.ensure(k)
        return self._get(k, np.arange(lo, hi + 1)), self.log_scale[k]

    def sample(self, k, finals, rng):
        """Backward-sample k items whose sum is finals[b] (one row per b)."""
        self.ensure(k)
        b = len(finals)
        counts = np.zeros((b, len(self.items)), np.int64)
        s = np.asarray(finals, np.int64).copy()
        for step in range(k, 0, -1):
            rest = s[:, None] - self.items[None, :]
            w = self._get(step - 1, rest) * self.probs[None, :]
            total = w.sum(1)
            if np.any(total <= 0):
                raise RuntimeError("backward sampling reached an unreachable state")
            u = rng.random(b) * total
            choice = np.minimum((np.cumsum(w, 1) < u[:, None]).sum(1), len(self.items) - 1)
            counts[np.arange(b), choice] += 1
            s -= self.items[choice]
        return counts


_VALUE_DP: dict = {}
DP_CHUNK = 1 << 15


def _dp_row(prev, plo, items, probs, lo, hi):
    """One DP row: cur[s] = sum_i p_i prev[s - v_i] on [lo, hi].

    Worked in cache-sized output chunks without temporaries: about 7x faster
    than whole-row slices on multi-million-value totals.
    """
    cur = np.zeros(hi - lo + 1)
    phi = plo + len(prev) - 1
    live = [(int(v), float(p)) for v, p in zip(items, probs) if p > 0]
    buf = np.empty(min(DP_CHUNK, hi - lo + 1))
    for c0 in range(lo, hi + 1, DP_CHUNK):
        c1 = min(hi, c0 + DP_CHUNK - 1)
        for v, p in live:
            a, b = max(c0, plo + v), min(c1, phi + v)
            if a <= b:
                tmp = buf[:b - a + 1]
                np.multiply(prev[a - v - plo:b - v - plo + 1], p, out=tmp)
                seg = cur[a - lo:b - lo + 1]
                np.add(seg, tmp, out=seg)
    return cur


# --------------------------------------------------------------- context
class Context:
    def __init__(self, prior: Prior, ev: Evidence):
        self.prior, self.ev = prior, ev
        self.conflict = None
        allowed = np.ones(len(prior.values), bool)
        if ev.max_value is not None:
            allowed &= prior.values <= ev.max_value
        if ev.max_per_cell is not None:
            allowed &= prior.vpc <= ev.max_per_cell
        if ev.largest_area is not None:
            allowed &= prior.areas <= ev.largest_area
        if ev.highest is not None:
            allowed &= prior.quality <= ev.highest
        self.allowed = allowed
        self.revealed = Counter({int(i): int(r) for i, r in ev.revealed_identities.items() if r})
        forced = Counter(self.revealed)
        self.atleast = set()
        self.flags = []           # identity masks needing >= 1 item somewhere
        self.flag_by_identity = False
        for i in self.revealed:
            if not allowed[i]:
                self.conflict = "revealed identity violates a cap"
        sets = []
        if ev.max_value is not None:
            sets.append(("max_value", allowed & (prior.values == ev.max_value)))
        if ev.max_per_cell is not None:
            sets.append(("max_per_cell", allowed & (prior.vpc == ev.max_per_cell)))
        if ev.largest_area is not None:
            sets.append(("largest_area", allowed & (prior.areas == ev.largest_area)))
        for name, mask in sets:
            members = np.flatnonzero(mask)
            if not len(members):
                self.conflict = f"no identity can show {name}"
                continue
            if any(self.revealed.get(int(i), 0) for i in members):
                continue
            if len(members) == 1:
                i = int(members[0])
                if i not in self.atleast:
                    self.atleast.add(i)
                    forced[i] += 1
            else:
                self.flags.append(mask)
                if name != "largest_area":
                    self.flag_by_identity = True
        self.forced = forced
        self.rho = [sum(r for i, r in self.revealed.items() if prior.quality[i] == q) for q in range(NQ)]
        self.forced_q = [sum(r for i, r in forced.items() if prior.quality[i] == q) for q in range(NQ)]
        self.minimum = [max(self.forced_q[q], self.rho[q] + ev.revealed_qualities.get(q, 0)) for q in range(NQ)]
        if ev.highest is not None:
            self.minimum[ev.highest] = max(1, self.minimum[ev.highest])
        self.reveal_total = sum(self.revealed.values()) + sum(ev.revealed_qualities.values())
        self.exact_sizes = sorted(ev.size_count)
        self.value_dp = {}
        self.class_dist = {}
        # Size-average Rao-Blackwellisation: which (quality, size) classes can
        # be integrated out without touching another clue.
        self.size_avg = None
        if ev.size_avg_value:
            (s, shown), = ev.size_avg_value.items()
            res = []
            for q in range(NQ):
                ok = q != COLORED and q not in ev.total_value
                ok &= not any(prior.quality[i] == q and prior.size[i] == s for i in self.atleast)
                if self.flag_by_identity:
                    ok &= not any(m[(prior.quality == q) & (prior.size == s)].any() for m in self.flags)
                res.append(ok)
            self.size_avg = (s, shown, res)


# --------------------------------------------------------------- pools
@dataclass
class Pool:
    log_f: float
    counts: np.ndarray
    cw: np.ndarray
    cells: np.ndarray
    size_counts: np.ndarray
    flag_bits: np.ndarray
    value: np.ndarray
    s_count: np.ndarray
    s_res: np.ndarray
    s_fixed: np.ndarray
    s_value: np.ndarray
    lookup: dict = field(default_factory=dict)


def _multinomial_rows(rng, n, p, size):
    """``size`` Multinomial(n, p) rows as counted categorical draws (faster for small n)."""
    m = len(p)
    if n == 0:
        return np.zeros((size, m), np.int64)
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    cat = np.minimum(np.searchsorted(cdf, rng.random((size, n)), "right"), m - 1)
    flat = (np.arange(size)[:, None] * m + cat).ravel()
    return np.bincount(flat, minlength=size * m).reshape(size, m)


def build_pool(ctx: Context, q, n_q, N, K, rng, reject_cap=60):
    prior, ev = ctx.prior, ctx.ev
    idx = prior.by_quality[q]
    pi = np.asarray(prior.identity_p(q, N), float)
    amask = ctx.allowed[idx]
    pi_a = float(pi[amask].sum())
    forced = np.asarray([ctx.forced.get(int(i), 0) for i in idx], np.int64)
    nf = int(forced.sum())
    if n_q < nf:
        return None
    if n_q == 0:
        return _stats(ctx, q, np.zeros((1, len(idx)), np.int64), np.ones(1), 0.0)
    if pi_a <= 0:
        return None
    pip = np.where(amask, pi / pi_a, 0.0)
    if np.any((forced > 0) & (pip <= 0)):
        return None
    rest_n = n_q - nf
    log_f = n_q * math.log(pi_a) + gammaln(n_q + 1) - gammaln(rest_n + 1)
    log_f += float(np.sum(forced[forced > 0] * np.log(pip[forced > 0])))
    log_f -= sum(gammaln(r + 1) for i, r in ctx.revealed.items() if prior.quality[i] == q)
    values, areas = prior.values[idx], prior.areas[idx]
    area_range = None
    if q in ev.avg_area:
        shown, scale = ev.avg_area[q]
        lo = -((-shown * n_q) // scale)
        hi = (shown * n_q + n_q - 1) // scale
        fa = int((forced * areas).sum())
        area_range = (lo - fa, hi - fa)
        if area_range[1] < max(area_range[0], 0):
            return None
    sel = np.flatnonzero(amask)
    if q in ev.total_value:
        target = int(ev.total_value[q]) - int((forced * values).sum())
        if target < 0:
            return None
        key = (q, target, tuple(sel), tuple(np.round(pip[sel], 12)))
        if key not in _VALUE_DP:
            while len(_VALUE_DP) >= 2:
                _VALUE_DP.pop(next(iter(_VALUE_DP)))
            _VALUE_DP[key] = SumDP(values[sel], pip[sel], target)
        dp = _VALUE_DP[key]
        logz = dp.log_prob(rest_n, target)
        if not math.isfinite(logz):
            return None
        draws, kept, tries = [], 0, 0
        while kept < K and tries < reject_cap:
            batch = dp.sample(rest_n, np.full(K, target), rng)
            tries += 1
            if area_range is not None:
                a = batch @ areas[sel]
                batch = batch[(a >= area_range[0]) & (a <= area_range[1])]
            draws.append(batch)
            kept += len(batch)
            if area_range is None:
                break
        rest_sel = np.concatenate(draws)[:K] if kept else np.zeros((0, len(sel)), np.int64)
        if area_range is not None:
            if kept == 0:
                return None
            logz += math.log(kept / (tries * K))
    elif area_range is not None:
        lo, hi = max(area_range[0], 0), area_range[1]
        dp = SumDP(areas[sel], pip[sel], hi)
        mass, scale = dp.range_mass(rest_n, lo, hi)
        if mass.sum() <= 0 or not math.isfinite(scale):
            return None
        logz = math.log(mass.sum()) + scale
        finals = lo + rng.choice(len(mass), size=K, p=mass / mass.sum())
        rest_sel = dp.sample(rest_n, finals, rng)
    else:
        logz = 0.0
        rest_sel = _multinomial_rows(rng, rest_n, pip[sel], K)
    counts = np.zeros((len(rest_sel), len(idx)), np.int64)
    counts[:, sel] = rest_sel
    counts += forced[None, :]
    cw = np.ones(len(counts))
    for i in ctx.atleast:
        if prior.quality[i] == q:
            cw /= counts[:, int(np.flatnonzero(idx == i)[0])]
    return _stats(ctx, q, counts, cw, log_f + logz)


def _stats(ctx, q, counts, cw, log_f):
    prior = ctx.prior
    idx = prior.by_quality[q]
    values, areas, sizes = prior.values[idx], prior.areas[idx], prior.size[idx]
    # one float matmul per statistic (exact: counts and sums stay far below 2**53)
    cf = counts.astype(np.float64)
    onehot = (sizes[:, None] == np.arange(NSIZES)[None, :]).astype(np.float64)
    size_counts = np.rint(cf @ onehot).astype(np.int64)
    bits = np.zeros(len(counts), np.int64)
    if ctx.flags:
        hits = cf @ np.stack([m[idx] for m in ctx.flags], 1).astype(np.float64) > 0
        for f in range(len(ctx.flags)):
            bits |= hits[:, f].astype(np.int64) << f
    s_count = s_res = s_fixed = s_value = np.zeros(len(counts), np.int64)
    if ctx.size_avg is not None:
        s, _, res = ctx.size_avg
        in_s = sizes == s
        s_count = counts[:, in_s].sum(1)
        s_value = counts[:, in_s] @ values[in_s]
        forced = np.asarray([ctx.forced.get(int(i), 0) for i in idx], np.int64)
        if res[q]:
            s_res = s_count - forced[in_s].sum()
            s_fixed = np.full(len(counts), int((forced[in_s] * values[in_s]).sum()))
        else:
            s_res = np.zeros(len(counts), np.int64)
            s_fixed = s_value
    return Pool(log_f, counts, cw, counts @ areas, size_counts, bits, counts @ values,
                s_count, s_res, s_fixed, s_value)


# --------------------------------------------------------------- stage 1
_BINOM_LOGC: dict = {}


def _binom_ppf(u, n, p, nmax):
    """Vectorised binomial inverse CDF: smallest x with P(X <= x) >= u, X ~ Bin(n, p)."""
    if p <= 0:
        return np.zeros(len(n), np.int64)
    if p >= 1:
        return np.asarray(n, np.int64).copy()
    if nmax not in _BINOM_LOGC:
        r = np.arange(nmax + 1)
        x = np.arange(nmax + 1)
        with np.errstate(invalid="ignore"):
            logc = gammaln(r[:, None] + 1) - gammaln(x[None, :] + 1) - gammaln(r[:, None] - x[None, :] + 1)
        _BINOM_LOGC[nmax] = np.where(x[None, :] <= r[:, None], logc, -np.inf)
    x = np.arange(nmax + 1)
    rows = np.arange(nmax + 1)[:, None]
    logpmf = _BINOM_LOGC[nmax] + x[None, :] * math.log(p) + np.maximum(rows - x[None, :], 0) * math.log1p(-p)
    cdf = np.cumsum(np.exp(logpmf), 1)
    out = (cdf[n] < u[:, None]).sum(1)
    return np.minimum(out, n).astype(np.int64)


def _stratified_multinomial(total, probs, size, rng, engine=None):
    """Multinomial(total, probs) draws from scrambled Sobol points (total: scalar or per row).

    Sequential conditional binomials with inverse-CDF sampling: each row is
    an exact multinomial draw, while rows cover the space evenly. ``engine``
    (a scrambled Sobol generator shared by the decision's cells) avoids
    re-initialising one per cell.
    """
    from scipy.stats import qmc

    k = len(probs)
    out = np.zeros((size, k), np.int64)
    remaining = np.broadcast_to(np.asarray(total, np.int64), (size,)).copy()
    if k == 1:
        out[:, 0] = remaining
        return out
    if engine is None or engine.d < k - 1:
        engine = qmc.Sobol(k - 1, scramble=True, seed=rng)
    u = np.clip(engine.random(size)[:, :k - 1], 1e-12, 1 - 1e-12)
    nmax = int(remaining.max()) if size else 0
    left = 1.0
    for j in range(k - 1):
        p = min(1.0, probs[j] / left) if left > 0 else 0.0
        draw = _binom_ppf(u[:, j], remaining, p, nmax)
        out[:, j] = np.minimum(draw, remaining)
        remaining -= out[:, j]
        left -= probs[j]
    out[:, k - 1] = remaining
    return out


def sample_counts(ctx: Context, M, rng):
    prior, ev = ctx.prior, ctx.ev
    quotas = np.asarray(prior.quotas)
    if ev.highest is not None and ev.highest < max(q for q in range(NQ) if quotas[q] > 0):
        return []
    table, keys = [], []
    for N, pN in sorted(prior.count_p.items()):
        rem = N - int(quotas.sum())
        if pN <= 0 or rem < 0:
            continue
        pc = np.asarray(prior.colored_p(N), float)
        q4 = np.asarray(prior.noncolored_p(N), float)
        for c in range(rem + 1):
            if c >= len(pc) or pc[c] <= 0:
                continue
            if COLORED in ev.count and ev.count[COLORED] != c:
                continue
            if ev.highest is not None and (ev.highest < COLORED) != (c == 0):
                continue
            t = rem - c
            known = {j: ev.count[j] - int(quotas[j]) for j in range(4) if j in ev.count}
            e = sum(known.values())
            if any(v < 0 for v in known.values()) or e > t:
                continue
            unknown = [j for j in range(4) if j not in known]
            if not unknown and e != t:
                continue
            logp = math.log(pN) + math.log(pc[c]) + gammaln(t + 1) - gammaln(t - e + 1)
            logp += sum(v * math.log(q4[j]) - gammaln(v + 1) for j, v in known.items() if v or q4[j] > 0)
            rest_p = 1.0 - sum(q4[j] for j in known)
            if t - e > 0:
                if rest_p <= 0:
                    continue
                logp += (t - e) * math.log(rest_p)
            table.append(logp)
            keys.append((N, c, t - e, tuple(unknown), tuple(sorted(known.items()))))
    if not table:
        return []
    lp = np.asarray(table)
    p = np.exp(lp - lp.max())
    p /= p.sum()
    from scipy.stats import qmc
    engine = qmc.Sobol(3, scramble=True, seed=rng)
    minimum = np.asarray(ctx.minimum)
    unknown = list(keys[0][3])
    key_n = np.asarray([k[0] for k in keys], np.int64)
    key_free = np.asarray([k[2] for k in keys], np.int64)
    key_base = np.tile(np.asarray(prior.quotas, np.int64), (len(keys), 1))
    for r, (N, c, free, _, known) in enumerate(keys):
        key_base[r, COLORED] = c
        for j, v in known:
            key_base[r, j] += v
    cdf = np.cumsum(p)
    assert key_n.max() < 128, "count codes assume fewer than 128 items"
    radix = np.int64(128) ** np.arange(NQ, -1, -1)
    codes, drawn, attempts = [], 0, 0
    while drawn < M and attempts < 20:
        attempts += 1
        # Systematic allocation over (N, rainbow count): the exact cell
        # probabilities, not a random draw, decide how many vectors each gets.
        picks = np.minimum(np.searchsorted(cdf, (rng.random() + np.arange(M)) / M), len(keys) - 1)
        n = key_base[picks]
        Ns = key_n[picks]
        if unknown:
            # cells of one N share the non-rainbow probabilities: one call each
            for N, rows in _groups(Ns):
                q4 = np.asarray(prior.noncolored_p(N), float)[unknown]
                part = _stratified_multinomial(key_free[picks[rows]], q4 / q4.sum(), len(rows), rng, engine)
                n[np.ix_(rows, unknown)] += part
        ok = (n >= minimum).all(1)
        codes.append(np.concatenate([Ns[ok, None], n[ok]], 1) @ radix)
        drawn += int(ok.sum())
    uniq, mult = np.unique(np.concatenate(codes), return_counts=True)
    digits = (uniq[:, None] // radix[None, :]) % 128
    return [(int(d[0]), tuple(int(x) for x in d[1:]), m / drawn) for d, m in zip(digits, mult.tolist())]


# --------------------------------------------------------------- stage 3
def _class_distribution(ctx, q, s, N):
    key = (q, s, N if q == COLORED else None)
    if key not in ctx.class_dist:
        prior = ctx.prior
        idx = prior.by_quality[q]
        pi = np.asarray(prior.identity_p(q, N), float)
        mask = ctx.allowed[idx] & (prior.size[idx] == s)
        p = np.where(mask, pi, 0.0)
        ctx.class_dist[key] = (prior.values[idx][mask], p[mask] / p[mask].sum()) if p.sum() > 0 else None
    return ctx.class_dist[key]


_SA_STORE: dict = {}


def _size_avg_probabilities(ctx, ks, lo, hi, c_s):
    """P(integrated-out size-class value sum in [lo, hi]) for every world.

    Each per-quality k-item sum distribution is stored only over its feasible
    range [k*min, k*max] (offset array, built by exact shift-adds). Distinct
    item mixes are processed in sorted order so a shared prefix of qualities
    is convolved once; a mix whose feasible range misses every member's
    window is skipped.
    """
    out = np.zeros(len(hi))
    has = ks.sum(1) > 0
    none = ~has & (c_s > 0)
    out[none] = ((lo <= 0) & (hi >= 0))[none]
    live = np.flatnonzero(has & (c_s > 0) & (hi >= 0) & (hi >= lo))
    if not len(live):
        return out
    size = ctx.size_avg[0]
    order_q = [3, 2, 1, 0]  # widest classes first, so prefixes are shared most
    # Process-level store keyed by the integrated classes' exact item
    # distributions: escalation rounds *and later rounds of the same game*
    # (the size-average clue does not change after it appears) reuse every
    # distribution. Only the windows differ per world.
    sig = [size]
    for q in order_q:
        dist = _class_distribution(ctx, q, size, 35)
        sig.append((q, None) if dist is None else (q, tuple(dist[0].tolist()), tuple(np.round(dist[1], 12).tolist())))
    sig = tuple(sig)
    store = _SA_STORE.get(sig)
    if store is None:
        while len(_SA_STORE) >= 2:
            _SA_STORE.pop(next(iter(_SA_STORE)))
        store = _SA_STORE[sig] = {"single": {}, "pmf": {}, "mix": {}, "floats": 0}
    single, cache = store["single"], store["pmf"]

    def one(q):
        if q not in single:
            dv, dp = _class_distribution(ctx, q, size, 35)
            base = int(dv.min())
            single[q] = (base, dv - base, dp)
        return single[q]

    def pmf(q, k):
        if (q, k) not in cache:
            base, shifts, probs = one(q)
            if k == 1:
                arr = np.zeros(int(shifts.max()) + 1)
                np.add.at(arr, shifts, probs)
                cache[(q, k)] = (base, arr)
            else:
                off, prev = pmf(q, k - 1)
                arr = np.zeros(len(prev) + int(shifts.max()))
                for sh, p in zip(shifts, probs):
                    arr[sh:sh + len(prev)] += p * prev
                cache[(q, k)] = (off + base, arr)
        return cache[(q, k)]

    mixes, inverse = np.unique(ks[live][:, order_q], axis=0, return_inverse=True)
    inverse = inverse.ravel()
    chain = []  # [(prefix tuple, offset, array)] along the current sorted prefix
    for u, mix in enumerate(mixes):
        members = live[inverse == u]
        mix = tuple(int(k) for k in mix)
        low = sum(k * one(q)[0] for q, k in zip(order_q, mix) if k)
        high = sum(k * (one(q)[0] + int(one(q)[1].max())) for q, k in zip(order_q, mix) if k)
        lo_m, hi_m = lo[members], hi[members]
        if not np.any((hi_m >= low) & (lo_m <= high)):
            continue
        if mix in store["mix"]:
            off, cdf = store["mix"][mix]
            a = np.clip(lo_m - off, 0, len(cdf) - 1)
            b = np.clip(hi_m - off + 1, 0, len(cdf) - 1)
            out[members] = np.clip(np.where(b > a, cdf[b] - cdf[a], 0.0), 0.0, 1.0)
            continue
        while chain and chain[-1][0] != mix[:len(chain[-1][0])]:
            chain.pop()
        start_j = len(chain[-1][0]) if chain else 0
        off, dist = (chain[-1][1], chain[-1][2]) if chain else (0, None)
        for j in range(start_j, len(order_q)):
            k = mix[j]
            if k:
                o2, part = pmf(order_q[j], k)
                if dist is None:
                    off, dist = o2, part
                else:
                    off, dist = off + o2, np.clip(fftconvolve(dist, part), 0.0, None)
            chain.append((mix[:j + 1], off, dist))
        cdf = np.concatenate([[0.0], np.cumsum(dist)])
        if store["floats"] + len(cdf) <= 150_000_000:
            store["mix"][mix] = (off, cdf)
            store["floats"] += len(cdf)
        a = np.clip(lo_m - off, 0, len(dist))
        b = np.clip(hi_m - off + 1, 0, len(dist))
        out[members] = np.clip(np.where(b > a, cdf[b] - cdf[a], 0.0), 0.0, 1.0)
    return out


def _pool_lookup(ctx, pool, need_mask, radix):
    """Sorted match keys + weight prefix sums for one pool, built once.

    Keyed by the decision's matching components too, because a pool can be
    reused by a later round whose clues match on different statistics.
    """
    lookup_key = (need_mask, ctx.ev.total_cells is not None, tuple(ctx.exact_sizes))
    if lookup_key not in pool.lookup:
        key = pool.cells.copy() if ctx.ev.total_cells is not None else np.zeros(len(pool.cw), np.int64)
        for s in ctx.exact_sizes:
            key = key * radix + pool.size_counts[:, s]
        eligible = np.flatnonzero((pool.flag_bits & need_mask) == need_mask)
        order = eligible[np.argsort(key[eligible], kind="stable")]
        pool.lookup[lookup_key] = (key[order], order, np.concatenate([[0.0], np.cumsum(pool.cw[order])]))
    return pool.lookup[lookup_key]


# --------------------------------------------------------------- size-average: clue first
# A size-average clue ("these c items of size X average V", floor) is a
# near-exact subset-sum constraint: only a few item combinations hit the
# window. Instead of drawing each quality's size-X items blindly and weighting
# afterwards, enumerate every quality's size-X states exactly, pair the
# qualities (meet in the middle) to find the combinations that satisfy the
# clue, and only then pick the rest of each quality from its candidates.
SA_TUPLE = True
SA_TABLE_LIMIT = 100_000
SA_SORTED_LIMIT = 1_500_000
SA_CACHE_ROWS = 200_000
SA_GROUP_SHIFT = 1 << 34
SA_MAX_VECTORS = 600
SA_MIN_VECTOR_ESS = 100.0
SA_MAX_PICKS = 4800


def _sa_enumerable(ctx, q):
    """Can quality q's non-forced size-X identities be enumerated from the prior?"""
    prior, ev = ctx.prior, ctx.ev
    s = ctx.size_avg[0]
    if q in ev.total_value:
        return False
    if any(prior.quality[i] == q and prior.size[i] == s for i in ctx.atleast):
        return False
    if ctx.flag_by_identity and any(m[(prior.quality == q) & (prior.size == s)].any() for m in ctx.flags):
        return False
    return True


def _sa_forced(ctx, q):
    cache = ctx.__dict__.setdefault("sa_forced", {})
    if q not in cache:
        prior, s = ctx.prior, ctx.size_avg[0]
        items = [(i, r) for i, r in ctx.forced.items() if prior.quality[i] == q and prior.size[i] == s]
        cache[q] = (sum(r for _, r in items), sum(r * int(prior.values[i]) for i, r in items))
    return cache[q]


def _class_sum(ctx, q, N, k):
    """Exact distribution of the value sum of k iid size-X identities of quality q."""
    cache = ctx.__dict__.setdefault("sa_class_sum", {})
    key = (q, N if q == COLORED else None, k)
    if key not in cache:
        if k == 0:
            cache[key] = (np.zeros(1, np.int64), np.ones(1))
        else:
            dist = _class_distribution(ctx, q, ctx.size_avg[0], N)
            prev = _class_sum(ctx, q, N, k - 1)
            if dist is None or prev is None:
                cache[key] = None
            else:
                dv, dp = dist
                vals = (prev[0][:, None] + dv[None, :]).ravel()
                probs = (prev[1][:, None] * dp[None, :]).ravel()
                uniq, inv = np.unique(vals, return_inverse=True)
                cache[key] = (uniq, np.bincount(inv.ravel(), weights=probs))
    return cache[key]


def _sa_states(ctx, q, pool, N):
    """One quality's size-X states: count, value (forced items included), mass, group.

    Enumerable quality: group = non-forced size-X count k, mass = P_pool(k) x
    exact P(k-item sum = value). Otherwise: group = a distinct (count, value)
    pair among the pool candidates, mass = its share of the pool weight.
    """
    key = ("sa_states", N if q == COLORED else None)
    if key in pool.lookup:
        return pool.lookup[key]
    K = len(pool.cw)
    fc, fv = _sa_forced(ctx, q)
    if _sa_enumerable(ctx, q):
        group_of = pool.s_count - fc
        parts = []
        for k in np.unique(group_of):
            pk = pool.cw[group_of == k].sum() / K
            dist = _class_sum(ctx, q, N, int(k))
            if dist is None or pk <= 0:
                continue
            parts.append((np.full(len(dist[0]), int(k) + fc), dist[0] + fv, pk * dist[1],
                          np.full(len(dist[0]), int(k))))
        if not parts:
            parts = [(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64))]
        count, value, mass, group = (np.concatenate(x) for x in zip(*parts))
    else:
        pairs = pool.s_count.astype(np.int64) * (1 << 40) + pool.s_value.astype(np.int64)
        uniq, inv = np.unique(pairs, return_inverse=True)
        group_of = inv.ravel()
        count, value = uniq >> 40, uniq & ((1 << 40) - 1)
        mass = np.bincount(group_of, weights=pool.cw, minlength=len(uniq)) / K
        group = np.arange(len(uniq))
    keep = np.asarray(mass) > 0
    out = {"count": np.asarray(count, np.int64)[keep], "value": np.asarray(value, np.int64)[keep],
           "mass": np.asarray(mass, float)[keep], "group": np.asarray(group, np.int64)[keep],
           "group_of": np.asarray(group_of, np.int64)}
    pool.lookup[key] = out
    return out


def _sa_product(parts):
    """Cartesian product of per-quality state tables (None if too large)."""
    count, value, mass, idx = np.zeros(1, np.int64), np.zeros(1, np.int64), np.ones(1), []
    for st in parts:
        prev, new = len(mass), len(st["mass"])
        if prev * new > max(SA_TABLE_LIMIT, SA_SORTED_LIMIT):
            return None
        count = (count[:, None] + st["count"][None, :]).ravel()
        value = (value[:, None] + st["value"][None, :]).ravel()
        mass = (mass[:, None] * st["mass"][None, :]).ravel()
        idx = [np.repeat(i, new) for i in idx] + [np.tile(np.arange(new), prev)]
    return {"count": count, "value": value, "mass": mass, "idx": idx}


def _sa_sorted(table):
    """Second half sorted by (count, value), with mass prefix sums and combined keys."""
    order = np.lexsort((table["value"], table["count"]))
    count, value = table["count"][order], table["value"][order]
    prefix = np.concatenate([[0.0], np.cumsum(table["mass"][order])])
    return {"order": order, "value": value, "prefix": prefix, "counts": np.unique(count),
            "ckey": count * (1 << 40) + value}


def _sa_pair(count1, value1, g2s, shown, exact_count):
    """For first-half entries (count1, value1): compatible second-half mass per count block.

    One vectorised binary search over combined (count, value) keys; returns
    masses and index ranges [a, b) into the sorted second half.
    """
    c2 = g2s["counts"][None, :]
    total = count1[:, None] + c2
    lo = total * shown - value1[:, None]
    hi = total * shown + total - 1 - value1[:, None]
    ok = (total >= 1) & (hi >= 0)
    if exact_count is not None:
        ok &= total == exact_count
    shift = 1 << 40
    a = np.searchsorted(g2s["ckey"], (c2 * shift + np.maximum(lo, 0)).ravel(), "left").reshape(lo.shape)
    b = np.searchsorted(g2s["ckey"], (c2 * shift + np.maximum(hi, -1)).ravel(), "right").reshape(lo.shape)
    mm = np.where(ok & (b > a), g2s["prefix"][b] - g2s["prefix"][a], 0.0)
    return mm, a, b


def _sa_group_index(pool, group_of):
    if "sa_groups" not in pool.lookup:
        order = np.argsort(group_of, kind="stable")
        pool.lookup["sa_groups"] = (order, group_of[order], np.concatenate([[0.0], np.cumsum(pool.cw[order])]))
    return pool.lookup["sa_groups"]


def _sa_pick_in_group(pool, group_of, groups, rng):
    order, g_sorted, prefix = _sa_group_index(pool, group_of)
    a = np.searchsorted(g_sorted, groups, "left")
    b = np.searchsorted(g_sorted, groups, "right")
    u = prefix[a] + rng.random(len(groups)) * (prefix[b] - prefix[a])
    pos = np.clip(np.searchsorted(prefix, u, "right") - 1, a, np.maximum(b - 1, a))
    return order[pos], prefix[b] - prefix[a]


def _sa_last_lookup(ctx, pool, group_of, need_mask, radix):
    key = ("sa_last", need_mask, ctx.ev.total_cells is not None, tuple(ctx.exact_sizes))
    if key not in pool.lookup:
        mk = pool.cells.copy() if ctx.ev.total_cells is not None else np.zeros(len(pool.cw), np.int64)
        for s in ctx.exact_sizes:
            mk = mk * radix + pool.size_counts[:, s]
        combined = group_of * SA_GROUP_SHIFT + mk
        eligible = np.flatnonzero((pool.flag_bits & need_mask) == need_mask)
        order = eligible[np.argsort(combined[eligible], kind="stable")]
        pool.lookup[key] = (combined[order], order, np.concatenate([[0.0], np.cumsum(pool.cw[order])]))
    return pool.lookup[key]


def _sa_split(states):
    """Cheapest bipartition (iterated side, sorted side) of the five state tables.

    An iterated row costs one binary search per count block of the sorted
    side; a sorted row costs its share of the sort. Iterated side capped at
    SA_TABLE_LIMIT rows, sorted side at SA_SORTED_LIMIT. None if nothing fits.
    """
    sizes = [len(st["mass"]) for st in states]
    counts = [np.unique(st["count"]) for st in states]
    best, best_cost = None, None
    for mask in range(1, (1 << NQ) - 1):
        left = tuple(q for q in range(NQ) if mask >> q & 1)
        right = tuple(q for q in range(NQ) if not mask >> q & 1)
        nl = int(np.prod([sizes[q] for q in left]))
        nr = int(np.prod([sizes[q] for q in right]))
        if nl > SA_TABLE_LIMIT or nr > SA_SORTED_LIMIT:
            continue
        span = sum(int(counts[q].max() - counts[q].min()) if len(counts[q]) else 0 for q in right) + 1
        log_nr = math.log2(nr + 2)
        cost = nl * span * 2 * log_nr + nr * log_nr * 3
        if best_cost is None or cost < best_cost:
            best, best_cost = (left, right), cost
    return best


def _sa_combine(ctx, vectors, pools, worlds, rng, rows_loglik, trace=None):
    """Clue-first combination for size-average decisions.

    Count vectors whose state tables are too large for any split go to the
    window-weighted path (_rb_combine); both estimators are on the same scale.
    """
    prior, ev = ctx.prior, ctx.ev
    s, shown, _ = ctx.size_avg
    exact_count = ev.size_count.get(s)
    nflags = len(ctx.flags)
    full_mask = (1 << nflags) - 1
    radix = 101
    stats = Counter(vectors=len(vectors))
    tables = {}

    def table(qs, pq, N, sort):
        key = (qs, sort) + tuple(id(pq[q]) for q in qs) + ((N,) if COLORED in qs else ())
        if key in tables:
            return tables[key]
        states = [_sa_states(ctx, q, pq[q], N) for q in qs]
        t = _sa_product(states)
        out = (t, states, _sa_sorted(t) if sort else None)
        if t is not None and len(t["mass"]) <= SA_CACHE_ROWS:
            tables[key] = out
        return out

    # Pass A: every count vector's prior-and-evidence weight (cheap).
    candidates = _vector_table(ctx, vectors, pools, stats)
    if not candidates:
        return [[], [], dict(stats)]
    # Two-stage sampling: the exact clue pairing is costly per vector. With
    # few vectors, pair them all. Otherwise pair a systematic subsample drawn
    # in proportion to the weight above (a vector picked m times out of M
    # stands for total weight * m / M), doubling M until at least
    # SA_MIN_VECTOR_ESS picked vectors effectively carry the clue mass.
    b = np.asarray([c[3] for c in candidates])
    top = float(b.max())
    w = np.exp(b - top)
    total = float(w.sum())
    paired = {}

    def pair(i):
        if i not in paired:
            N, n, pq, _ = candidates[i]
            split = _sa_split([_sa_states(ctx, q, pq[q], N) for q in range(NQ)])
            if split is None:
                paired[i] = (None, None, None)
            else:
                left, right = split
                t1, _, _ = table(left, pq, N, False)
                _, _, g2s = table(right, pq, N, True)
                mm, _, _ = _sa_pair(t1["count"], t1["value"], g2s, shown, exact_count)
                row_mass = t1["mass"] * mm.sum(1)
                paired[i] = (split, row_mass, float(row_mass.sum()))
        return paired[i]

    import time as _time
    deadline = getattr(ctx, "deadline", None)

    def process(entries):
        """Pair entries in priority order until the deadline (an unbiased priority sample)."""
        proxy = np.asarray([lb for _, lb in entries])
        order, logprio = _priority_order(rng, proxy)
        kept = []
        for j in order:
            _check_cancel(ctx)
            if deadline is not None and len(kept) >= 20 and _time.monotonic() > deadline:
                break
            pair(entries[j][0])
            kept.append(int(j))
        if len(kept) < len(entries):
            stats["vectors_cut_by_deadline"] = len(entries) - len(kept)
        adj = _priority_adjust(proxy, logprio, order, len(kept))
        return [(entries[j][0], entries[j][1] + a) for j, a in zip(kept, adj)]

    if len(candidates) <= SA_MAX_VECTORS:
        chosen = process([(i, c[3]) for i, c in enumerate(candidates)])
    else:
        m_sel = SA_MAX_VECTORS
        while True:
            picks = np.searchsorted(np.cumsum(w) / total, (rng.random() + np.arange(m_sel)) / m_sel)
            picks, mult = np.unique(np.minimum(picks, len(candidates) - 1), return_counts=True)
            chosen = process([(i, top + math.log(total) + math.log(m / m_sel))
                              for i, m in zip(picks.tolist(), mult.tolist())])
            u = np.asarray([math.exp(lb - top) * (pair(i)[2] or 0.0) for i, lb in chosen])
            vector_ess = float(u.sum() ** 2 / (u ** 2).sum()) if u.sum() > 0 else 0.0
            out_of_time = deadline is not None and _time.monotonic() > deadline
            if (vector_ess >= SA_MIN_VECTOR_ESS or out_of_time
                    or m_sel >= len(candidates) * 4 or m_sel >= SA_MAX_PICKS):
                break
            m_sel *= 2
        stats["vector_ess"] = round(vector_ess, 1)
    stats["vectors_paired"] = len(chosen)
    weighted, fallback, fallback_base = [], [], []
    for i, base in chosen:
        N, n, pq, _ = candidates[i]
        split, row_mass, z = pair(i)
        if split is None:
            fallback.append((N, n, 1.0))
            fallback_base.append(base)
            continue
        if z <= 0:
            stats["vectors_clue_infeasible"] += 1
            continue
        weighted.append((N, n, pq, base + math.log(z), split, row_mass, base))
    totals, logw = [], []
    stats["vectors_fallback"] = len(fallback)
    worlds_a = worlds
    if fallback:
        # Fallback vectors get the budget share of their prior mass, clamped
        # to [1,000 worlds, half the budget]; any split keeps both unbiased.
        prior_a = np.asarray([w[6] for w in weighted]) if weighted else np.zeros(0)
        prior_fb = np.asarray(fallback_base)
        top = max(prior_fb.max(), prior_a.max() if len(prior_a) else -np.inf)
        mass_fb = float(np.exp(prior_fb - top).sum())
        mass_a = float(np.exp(prior_a - top).sum()) if len(prior_a) else 0.0
        worlds_fb = int(min(worlds / 2, max(1000, worlds * mass_fb / (mass_fb + mass_a))))
        worlds_a = max(worlds - worlds_fb, 1000)
        t_fb, lw_fb, st_fb = _rb_combine(ctx, fallback, pools, worlds_fb, rng, rows_loglik, bases=fallback_base)
        totals += list(t_fb)
        logw += list(lw_fb)
        stats["fallback_draws"] = st_fb.get("draws", 0)
        stats["_fallback_parts"] = len(lw_fb)
    if not weighted:
        return [totals, logw, dict(stats)]
    alloc, counts = _allocate(rng, np.asarray([w[3] for w in weighted]), worlds_a)
    violations = 0
    recs, vids = [], []
    grp, svs = {q: [] for q in range(NQ)}, {q: [] for q in range(NQ)}
    for (N, n, pq, base, (left, right), row_mass, _), x, draws in zip(weighted, alloc, counts):
        if draws <= 0:
            continue
        draws = int(draws)
        t1, st1, _ = table(left, pq, N, False)
        t2, st2, g2s = table(right, pq, N, True)
        e1 = rng.choice(len(row_mass), size=draws, p=row_mass / row_mass.sum())
        mm, lo_idx, hi_idx = _sa_pair(t1["count"][e1], t1["value"][e1], g2s, shown, exact_count)
        cum = np.cumsum(mm, 1)
        block = np.minimum((cum < rng.random(draws)[:, None] * cum[:, -1:]).sum(1), mm.shape[1] - 1)
        rows_ = np.arange(draws)
        a, b = lo_idx[rows_, block], hi_idx[rows_, block]
        u = g2s["prefix"][a] + rng.random(draws) * (g2s["prefix"][b] - g2s["prefix"][a])
        pos = np.clip(np.searchsorted(g2s["prefix"], u, "right") - 1, a, np.maximum(b - 1, a))
        e2 = g2s["order"][pos]
        state = {q: t1["idx"][col][e1] for col, q in enumerate(left)}
        state.update({q: t2["idx"][col][e2] for col, q in enumerate(right)})
        states = {q: st1[left.index(q)] for q in left}
        states.update({q: st2[right.index(q)] for q in right})
        size_count = sum(states[q]["count"][state[q]] for q in range(NQ))
        size_value = sum(states[q]["value"][state[q]] for q in range(NQ))
        violations += int(np.sum((size_count < 1) | (size_value // np.maximum(size_count, 1) != shown)))
        vids.append(np.full(draws, len(recs)))
        recs.append((N, n, pq, base - math.log(x)))
        for q in range(NQ):
            grp[q].append(states[q]["group"][state[q]])
            svs[q].append(states[q]["value"][state[q]])
    if recs:
        group = {q: np.concatenate(grp[q]) for q in range(NQ)}
        value_add = {q: np.concatenate(svs[q]) if _sa_enumerable(ctx, q) else None for q in range(NQ)}
        t, lw = _draw_worlds(ctx, recs, np.concatenate(vids), rng, rows_loglik, stats, group=group,
                             group_of=lambda q, pool, N: _sa_states(ctx, q, pool, N)["group_of"],
                             value_add=value_add)
        totals.append(t)
        logw.append(lw)
    stats["clue_violations"] = violations
    k = stats.pop("_fallback_parts", 0)
    parts = {"fallback": logw[:k], "clue_first": logw[k:]}
    allw = np.concatenate(logw) if logw else np.zeros(0)
    if len(allw):
        top = allw.max()
        for name, lws in parts.items():
            x = np.concatenate(lws) if lws else np.zeros(0)
            wts = np.exp(x - top)
            stats[f"{name}_weight_share"] = round(float(wts.sum() / np.exp(allw - top).sum()), 4) if len(x) else 0.0
            stats[f"{name}_ess"] = round(float(wts.sum() ** 2 / (wts ** 2).sum()), 1) if len(x) and wts.sum() > 0 else 0.0
    return [totals, logw, dict(stats)]


# --------------------------------------------------------------- size-average: k-level, cached
# When all four non-rainbow qualities are enumerable, the probability that
# their size-X items satisfy the clue depends only on how many size-X items
# each contributes (k) and on the rainbow offset (count, value) -- not on the
# count vector. It is computed once per (k tuple, rainbow offset) from small
# pairwise tables (white+blue vs purple+gold) and cached for the decision and
# for later rounds of the same game; each count vector then only sums cached
# values weighted by its own k probabilities.
SA_K_LEVEL = True
_SA_K_STORE: dict = {}
_SA_K_SERIAL = __import__("itertools").count()
_LEFT_Q, _RIGHT_Q = (0, 1), (2, 3)


def _sa_k_store(ctx):
    """Per-clue cache shared by rounds of a game (same size, shown value, caps)."""
    s, shown, _ = ctx.size_avg
    sig = [s, shown, ctx.ev.size_count.get(s)]
    for q in range(4):
        dist = _class_distribution(ctx, q, s, 35)
        fc, fv = _sa_forced(ctx, q)
        sig.append((q, fc, fv) + (() if dist is None else (tuple(dist[0].tolist()), tuple(np.round(dist[1], 12).tolist()))))
    sig = tuple(sig)
    if sig not in _SA_K_STORE:
        while len(_SA_K_STORE) >= 2:
            _SA_K_STORE.pop(next(iter(_SA_K_STORE)))
        _SA_K_STORE[sig] = {"left": {}, "right": {}, "atoms": {}, "serial": next(_SA_K_SERIAL),
                            "arr": np.full((1, 1, 1, 1, 0), np.nan)}
    return _SA_K_STORE[sig]


def _sa_k_half(ctx, store, side, ks):
    """Exact product table of non-forced class sums for one half given its k's."""
    cache = store[side]
    if ks not in cache:
        qs = _LEFT_Q if side == "left" else _RIGHT_Q
        value, prob, idx = np.zeros(1, np.int64), np.ones(1), []
        for q, k in zip(qs, ks):
            d = _class_sum(ctx, q, 35, int(k))
            if d is None:
                cache[ks] = None
                return None
            prev, new = len(prob), len(d[0])
            value = (value[:, None] + d[0][None, :]).ravel()
            prob = (prob[:, None] * d[1][None, :]).ravel()
            idx = [np.repeat(i, new) for i in idx] + [np.tile(np.arange(new), prev)]
        if side == "left":
            order = np.argsort(value, kind="stable")
            value, prob, idx = value[order], prob[order], [i[order] for i in idx]
        entry = {"value": value, "prob": prob, "idx": idx}
        if side == "right":
            order = np.argsort(value, kind="stable")
            entry.update(order=order, sorted=value[order], prefix=np.concatenate([[0.0], np.cumsum(prob[order])]))
        cache[ks] = entry
    return cache[ks]


def _sa_k_psi(ctx, store, ke, atoms_c, atoms_f):
    """P(clue holds | k tuple ke, rainbow offsets) for each offset (count, value)."""
    s, shown, _ = ctx.size_avg
    exact = ctx.ev.size_count.get(s)
    left = _sa_k_half(ctx, store, "left", ke[:2])
    right = _sa_k_half(ctx, store, "right", ke[2:])
    fc, fv = _sa_forced_total(ctx)
    total = int(sum(ke)) + fc + atoms_c
    lo = total * shown - atoms_f - fv
    hi = total * shown + total - 1 - atoms_f - fv
    ok = total >= 1
    if exact is not None:
        ok &= total == exact
    psi = np.zeros(len(atoms_c))
    if left is None or right is None:
        return psi
    lv = left["value"]
    ok &= (hi >= lv[0] + right["sorted"][0]) & (lo <= lv[-1] + right["sorted"][-1])
    live = np.flatnonzero(ok)
    # all offsets at once, in chunks of about two million pairs
    step = max(1, 2_000_000 // max(len(lv), 1))
    for c0 in range(0, len(live), step):
        j = live[c0:c0 + step]
        a = np.searchsorted(right["sorted"], lo[j, None] - lv[None, :], "left")
        b = np.searchsorted(right["sorted"], hi[j, None] - lv[None, :], "right")
        psi[j] = (right["prefix"][b] - right["prefix"][a]) @ left["prob"]
    return psi


def _sa_forced_total(ctx):
    cache = ctx.__dict__.setdefault("sa_forced", {})
    if "total" not in cache:
        parts = [_sa_forced(ctx, q) for q in range(4)]
        cache["total"] = (sum(p[0] for p in parts), sum(p[1] for p in parts))
    return cache["total"]


def _pool_pairs(pool):
    """Distinct (size-X count, size-X value) pairs of a pool with weight shares."""
    if "sa_pairs" not in pool.lookup:
        pairs = pool.s_count.astype(np.int64) * (1 << 40) + pool.s_value.astype(np.int64)
        uniq, inv = np.unique(pairs, return_inverse=True)
        inv = inv.ravel()
        pool.lookup["sa_pairs"] = {"count": uniq >> 40, "value": uniq & ((1 << 40) - 1),
                                   "mass": np.bincount(inv, weights=pool.cw, minlength=len(uniq)) / len(pool.cw),
                                   "group": np.arange(len(uniq)), "group_of": inv}
    return pool.lookup["sa_pairs"]


def _pool_k(ctx, q, pool):
    """Non-forced size-X count distribution of an enumerable quality's pool."""
    key = ("sa_k", q)
    if key not in pool.lookup:
        fc, _ = _sa_forced(ctx, q)
        group_of = pool.s_count - fc
        ks, inv = np.unique(group_of, return_inverse=True)
        mass = np.bincount(inv.ravel(), weights=pool.cw, minlength=len(ks)) / len(pool.cw)
        pool.lookup[key] = {"k": ks.astype(np.int64), "mass": mass, "group_of": group_of.astype(np.int64)}
    return pool.lookup[key]


def _sa_k_applicable(ctx):
    return SA_K_LEVEL and all(_sa_enumerable(ctx, q) for q in range(4))


# --------------------------------------------------------------- batched world drawing
# Every combination path ends the same way: each quality except a count
# vector's largest ("last") gets a candidate from its pool, and the last
# quality is matched exactly on the remaining deficits (total cells, exact
# size counts, flags). Doing this per count vector cost thousands of small
# numpy calls; here all draws of a round are processed together, looping only
# over distinct pools.
def _groups(keys):
    """(key, rows) for each distinct value of an integer array."""
    if not len(keys):
        return []
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    cuts = np.flatnonzero(np.diff(k)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(k)]])
    return [(int(k[a]), order[a:b]) for a, b in zip(starts, ends)]


def _draw_worlds(ctx, recs, vid, rng, rows_loglik, stats, group=None, group_of=None, value_add=None):
    """Finish the draws of many count vectors at once.

    ``recs[v] = (N, n, pq, lw0)`` with lw0 the per-draw log weight so far;
    ``vid`` gives each draw's vector. Without ``group``, non-last candidates
    are uniform picks whose weight cw enters the log weight; with ``group``
    (per quality, per draw), they are picked in proportion to cw within the
    draw's group (``group_of(q, pool, N)`` maps the pool's candidates to
    groups), and the last quality is matched within its group too.
    ``value_add[q]`` replaces quality q's size-X value: value - s_value + add.
    """
    prior, ev = ctx.prior, ctx.ev
    D = len(vid)
    full_mask = (1 << len(ctx.flags)) - 1
    radix = 101
    last = np.asarray([int(np.argmax(r[1])) for r in recs], np.int64)[vid]
    lw = np.asarray([r[3] for r in recs], float)[vid]
    cells = np.zeros(D, np.int64)
    last_cells = np.zeros(D, np.int64)
    sizes = np.zeros((D, NSIZES), np.int64)
    bits = np.zeros(D, np.int64)
    value = np.zeros(D)
    value_c = np.zeros(D)
    pools_q = []
    for q in range(NQ):
        index, plist, owner = {}, [], []
        vp = np.empty(len(recs), np.int64)
        for v, r in enumerate(recs):
            p = r[2][q]
            if id(p) not in index:
                index[id(p)] = len(plist)
                plist.append(p)
                owner.append(r[0])
            vp[v] = index[id(p)]
        pools_q.append((plist, owner, vp[vid]))

    def add_value(q, pool, rows, c):
        v = pool.value[c].astype(float)
        if value_add is not None and value_add.get(q) is not None:
            v = v - pool.s_value[c] + value_add[q][rows]
        value[rows] += v
        if q == COLORED:
            value_c[rows] = v

    for q in range(NQ):
        plist, owner, pid = pools_q[q]
        sel = np.flatnonzero(last != q)
        for pi, rows in _groups(pid[sel]):
            rows = sel[rows]
            pool = plist[pi]
            if group is None:
                c = rng.integers(len(pool.cw), size=len(rows))
                lw[rows] += np.log(pool.cw[c])
            else:
                c, _ = _sa_pick_in_group(pool, group_of(q, pool, owner[pi]), group[q][rows], rng)
            cells[rows] += pool.cells[c]
            sizes[rows] += pool.size_counts[c]
            bits[rows] |= pool.flag_bits[c]
            add_value(q, pool, rows, c)
    need = full_mask & ~bits
    deficit = np.zeros(D, np.int64)
    bad = np.zeros(D, bool)
    if ev.total_cells is not None:
        deficit = ev.total_cells - cells
        bad |= deficit < 0
    for s in ctx.exact_sizes:
        deficit = deficit * radix + (ev.size_count[s] - sizes[:, s])
        bad |= (ev.size_count[s] - sizes[:, s]) < 0
    mass = np.zeros(D)
    span = full_mask + 1
    for q in range(NQ):
        plist, owner, pid = pools_q[q]
        sel = np.flatnonzero((last == q) & ~bad)
        for gk, rows in _groups(pid[sel] * span + need[sel]):
            rows = sel[rows]
            pi, m = divmod(gk, span)
            pool = plist[pi]
            if group is None:
                keys, order, prefix = _pool_lookup(ctx, pool, m, radix)
                qk = deficit[rows]
                denom = np.full(len(rows), float(len(pool.cw)))
            else:
                gof = group_of(q, pool, owner[pi])
                keys, order, prefix = _sa_last_lookup(ctx, pool, gof, m, radix)
                g = group[q][rows]
                qk = g * SA_GROUP_SHIFT + deficit[rows]
                _, g_sorted, g_prefix = _sa_group_index(pool, gof)
                denom = (g_prefix[np.searchsorted(g_sorted, g, "right")]
                         - g_prefix[np.searchsorted(g_sorted, g, "left")])
            a = np.searchsorted(keys, qk, "left")
            b = np.searchsorted(keys, qk, "right")
            tot = prefix[b] - prefix[a]
            hit = tot > 0
            if not hit.any():
                continue
            u = prefix[a] + rng.random(len(rows)) * tot
            pos = np.clip(np.searchsorted(prefix, u, "right") - 1, a, np.maximum(b - 1, a))
            rows, c = rows[hit], order[pos[hit]]
            mass[rows] = tot[hit] / denom[hit]
            last_cells[rows] = pool.cells[c]
            add_value(q, pool, rows, c)
    ok = mass > 0
    stats["draws"] += D
    stats["matched_draws"] += int(ok.sum())
    lw = lw[ok] + np.log(mass[ok])
    total_value = value[ok]
    if rows_loglik is not None and ev.map_rows and len(lw):
        lw = lw + rows_loglik(int(ev.map_rows), (cells + last_cells)[ok])
    if prior.world_log_tilt is not None and len(lw):
        n_all = np.asarray([r[0] for r in recs])[vid][ok]
        c_all = np.asarray([r[1][COLORED] for r in recs])[vid][ok]
        lw = lw + prior.world_log_tilt(n_all, c_all, value_c[ok], total_value)
    keep = np.isfinite(lw)
    return total_value[keep], lw[keep]


def _vector_table(ctx, vectors, pools, stats):
    """(N, n, pools, base log weight) of every supported count vector, computed at once."""
    ev = ctx.ev
    if not vectors:
        return []
    V = len(vectors)
    Ns = np.fromiter((v[0] for v in vectors), np.int64, V)
    ns = np.asarray([v[1] for v in vectors], np.int64)
    base = np.log(np.fromiter((v[2] for v in vectors), float, V))
    pidx = np.empty((V, NQ), np.int64)
    plist, missing = [], np.zeros(V, bool)
    for q in range(NQ):
        key = ns[:, q] * 1000 + Ns if q == COLORED else ns[:, q]
        uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
        logf = np.empty(len(uniq))
        for j, f in enumerate(first):
            p = pools(q, int(ns[f, q]), int(Ns[f]))
            plist.append(p)
            logf[j] = np.nan if p is None else p.log_f
        pidx[:, q] = len(plist) - len(uniq) + inv.ravel()
        lf = logf[inv.ravel()]
        missing |= np.isnan(lf)
        base += np.where(np.isnan(lf), 0.0, lf)
    if ctx.reveal_total:
        with np.errstate(invalid="ignore", divide="ignore"):
            for q in range(NQ):
                r = ev.revealed_qualities.get(q, 0)
                if r:
                    a = ns[:, q] - ctx.rho[q]
                    base += gammaln(a + 1) - gammaln(r + 1) - gammaln(a - r + 1)
            base -= gammaln(Ns + 1) - gammaln(ctx.reveal_total + 1) - gammaln(Ns - ctx.reveal_total + 1)
    stats["vectors_without_support"] += int(missing.sum())
    keep = np.flatnonzero(~missing & np.isfinite(base))
    return [(vectors[v][0], vectors[v][1], [plist[i] for i in pidx[v]], float(base[v])) for v in keep]


def _allocate(rng, bases, worlds):
    """Worlds per vector in proportion to exp(bases), stochastically rounded (unbiased)."""
    rel = np.exp(bases - bases.max())
    alloc = worlds * rel / rel.sum()
    counts = np.floor(alloc).astype(np.int64) + (rng.random(len(alloc)) < alloc - np.floor(alloc))
    return alloc, counts


def _priority_order(rng, proxy):
    """Processing order for a deadline: priority sampling on exp(proxy).

    Stopping after the first k vectors of this order leaves a priority sample
    (Duffield, Lund & Thorup): with tau the (k+1)-th priority, each kept
    vector stands for max(1, tau / exp(proxy)) copies, an unbiased estimate
    for any k; vectors with a large proxy are kept first.
    """
    logprio = proxy - np.log(1.0 - rng.random(len(proxy)))
    order = np.argsort(-logprio, kind="stable")
    return order, logprio


def _priority_adjust(proxy, logprio, order, kept):
    """Log multiplicity of each kept vector (aligned with order[:kept])."""
    if kept >= len(order):
        return np.zeros(kept)
    log_tau = logprio[order[kept]]
    return np.maximum(0.0, log_tau - proxy[order[:kept]])


def _rb_batched(ctx, vectors, pools, worlds, rng, rows_loglik):
    """Draw-then-weight combination without a size-average clue, batched."""
    stats = Counter(vectors=len(vectors), vectors_without_support=0, draws=0, matched_draws=0)
    weighted = _vector_table(ctx, vectors, pools, stats)
    if not weighted:
        return [], [], dict(stats)
    alloc, counts = _allocate(rng, np.asarray([w[3] for w in weighted]), worlds)
    recs, vids = [], []
    for (N, n, pq, base), x, draws in zip(weighted, alloc, counts):
        if draws > 0:
            vids.append(np.full(int(draws), len(recs)))
            recs.append((N, n, pq, base - math.log(x)))
    if not recs:
        return [], [], dict(stats)
    t, lw = _draw_worlds(ctx, recs, np.concatenate(vids), rng, rows_loglik, stats)
    return [t], [lw], dict(stats)


def _sa_k_dense(ctx, store, kmax, rb):
    """Dense cached clue probabilities for k in [0, kmax_q], and the pool's offset columns."""
    key = ("atom_idx", store["serial"])
    atom_idx = rb.get(key)
    if atom_idx is None:
        atom_idx = []
        for a in zip(rb["count"].tolist(), rb["value"].tolist()):
            if a not in store["atoms"]:
                store["atoms"][a] = len(store["atoms"])
            atom_idx.append(store["atoms"][a])
        atom_idx = rb[key] = np.asarray(atom_idx, np.int64)
    arr = store["arr"]
    shape = tuple(max(a, b + 1) for a, b in zip(arr.shape[:4], kmax)) + (max(arr.shape[4], len(store["atoms"])),)
    if shape != arr.shape:
        grown = np.full(shape, np.nan)
        grown[tuple(slice(0, d) for d in arr.shape)] = arr
        store["arr"] = arr = grown
    return arr, atom_idx


SA_SKIP_EPS = 1e-5
SA_SKIP_FLAG = 0.05
SOFT_SHARE = 0.7


def _time_now():
    import time as _time
    return _time.monotonic()


SA_SKIP_MIN_CELLS = 64
_THETA_UNITS = np.concatenate([-np.logspace(-9, -3, 48)[::-1], [0.0], np.logspace(-9, -3, 48)])


def _class_log_mgf(ctx, q):
    """log E[exp(theta V)] of one size-X identity of quality q on the theta grid (None: no identity)."""
    cache = ctx.__dict__.setdefault("sa_log_mgf", {})
    if q not in cache:
        dist = _class_distribution(ctx, q, ctx.size_avg[0], 35)
        if dist is None:
            cache[q] = None
        else:
            v, p = np.asarray(dist[0], float), np.asarray(dist[1], float)
            th = _THETA_UNITS[:, None]
            ref = np.where(th > 0, v.max(), v.min())
            with np.errstate(divide="ignore"):
                cache[q] = (th * ref + np.log((p[None, :] * np.exp(th * (v[None, :] - ref))).sum(1, keepdims=True)))[:, 0]
    return cache[q]


def _clue_bound(ctx, kes, lo, hi):
    """Chernoff upper bound on P(lo <= S <= hi), S the sum of k_q size-X values of qualities 0-3.

    exp(sum_q k_q log M_q(theta) - theta x) bounds P(S >= x) for theta >= 0
    and P(S <= x) for theta <= 0; any grid point gives a valid bound.
    """
    lam = np.zeros((4, len(_THETA_UNITS)))
    dead = np.zeros(len(kes), bool)
    for q in range(4):
        lq = _class_log_mgf(ctx, q)
        if lq is None:
            dead |= kes[:, q] > 0
        else:
            lam[q] = lq
    pos, neg = _THETA_UNITS >= 0, _THETA_UNITS <= 0
    out = np.empty(len(kes))
    for c0 in range(0, len(kes), 50_000):
        sl = slice(c0, c0 + 50_000)
        A = kes[sl].astype(float) @ lam
        up = (A[:, pos] - _THETA_UNITS[pos][None, :] * lo[sl, None]).min(1)
        down = (A[:, neg] - _THETA_UNITS[neg][None, :] * hi[sl, None]).min(1)
        out[sl] = np.exp(np.minimum(np.minimum(up, down), 0.0))
    out[dead] = 0.0
    return out


def _sa_k_fill(ctx, store, block, need, hull, rb, atom_idx, w, M, stats):
    """Fill the group's clue probabilities, skipping provably negligible cells.

    The cache holds NaN (unseen), an exact probability (>= 0) or minus a
    Chernoff upper bound (< 0; computed once per cell and offset). Open cells
    are computed exactly in decreasing order of an upper bound on their
    contribution (vector weight x size-X count probabilities x bound). Once
    the bound on everything left is below SA_SKIP_EPS of the group's computed
    mass, the rest counts as zero for this group: the group's mass, and any
    posterior probability, is off by at most that share. Skipped cells keep
    their bound in the cache, so a group where they matter computes them.
    """
    arr = store["arr"]
    open_ = (np.isnan(block) | (block < 0)) & need[..., None]
    cells = np.argwhere(open_.any(axis=4))
    if not len(cells):
        return block
    s, shown, _ = ctx.size_avg
    fc, fv = _sa_forced_total(ctx)
    kes = np.stack([hull[q][cells[:, q]] for q in range(4)], 1)
    sel = tuple(cells.T)
    sub = block[sel]
    is_open = open_[sel]
    remaining = None
    hard = getattr(ctx, "hard_deadline", None)
    if len(cells) >= SA_SKIP_MIN_CELLS:
        W = np.einsum("v,va,vb,vc,vd->abcd", w, *M, optimize=True)
        Wc = W[sel]
        # mass already known exactly
        exact = np.where(np.isnan(block) | (block < 0), 0.0, block) @ rb["mass"]
        mass = float((W * exact).sum())
        ub = np.zeros(len(cells))
        fresh = np.isnan(sub) & is_open
        if fresh.any():
            ci, ai = np.nonzero(fresh)
            total = kes[ci].sum(1) + fc + rb["count"][ai]
            lo = total * shown - rb["value"][ai] - fv
            hi = lo + total - 1
            # windows outside the reachable range: exactly zero
            vmin, vmax = np.zeros(4), np.zeros(4)
            for q in range(4):
                dist = _class_distribution(ctx, q, s, 35)
                if dist is not None:
                    vmin[q], vmax[q] = dist[0].min(), dist[0].max()
            kf = kes[ci].astype(float)
            ok = (total >= 1) & (hi >= kf @ vmin) & (lo <= kf @ vmax)
            if ctx.ev.size_count.get(s) is not None:
                ok &= total == ctx.ev.size_count[s]
            b = np.zeros(len(ci))
            live = np.flatnonzero(ok)
            live = live[np.argsort(-Wc[ci[live]], kind="stable")]
            reached = len(live)
            for c0 in range(0, len(live), 100_000):
                if hard is not None and c0 and _time_now() > hard:
                    reached = c0
                    break
                part = live[c0:c0 + 100_000]
                b[part] = _clue_bound(ctx, kes[ci[part]], lo[part].astype(float), hi[part].astype(float))
            known = np.ones(len(ci), bool)
            known[live[reached:]] = False
            sub[ci[known], ai[known]] = -b[known]
            arr[tuple(kes[ci[known]].T) + (atom_idx[ai[known]],)] = -b[known]
            stats["sa_cells_bounded"] += int(known.sum())
            # pairs not bounded in time count with the trivial bound 1
            rest = live[reached:]
            ub += np.bincount(ci[rest], weights=rb["mass"][ai[rest]], minlength=len(cells))
        pending = sub < 0
        ub += np.where(pending, -sub, 0.0) @ rb["mass"]
        bound = Wc * ub
        order = np.argsort(-bound, kind="stable")
        order = order[bound[order] > 0]
        remaining = float(bound[order].sum())
        work = order
    else:
        work = np.arange(len(cells))
    computed = 0
    for done, i in enumerate(work):
        if done % 32 == 0:
            _check_cancel(ctx)
        if remaining is not None and done % 32 == 0:
            late = hard is not None and _time_now() > hard
            if remaining <= SA_SKIP_EPS * mass or late:
                stats["sa_cells_skipped"] += len(work) - done
                if late:
                    # out of time: what is left is skipped, with its bound on record
                    stats["sa_late"] = 1
                    share = remaining / mass if mass > 0 else 1.0
                    stats["sa_skip_bound"] = max(stats.get("sa_skip_bound", 0.0), round(min(share, 1.0), 6))
                break
        if remaining is not None:
            remaining -= bound[i]
        cols = np.flatnonzero(np.isnan(sub[i]) | (sub[i] < 0))
        if not len(cols):
            continue
        ke = tuple(int(x) for x in kes[i])
        vals = _sa_k_psi(ctx, store, ke, rb["count"][cols], rb["value"][cols])
        arr[ke + (atom_idx[cols],)] = vals
        sub[i, cols] = vals
        computed += 1
        if remaining is not None:
            mass += float(Wc[i] * (vals @ rb["mass"][cols]))
    stats["sa_cells_computed"] += computed
    # pending bounds (skipped cells) count as zero for this group
    block[sel] = np.where(np.isnan(sub) | (sub < 0), 0.0, sub)
    return block


def _sa_k_group(ctx, store, members, stats):
    """Exact clue mass of the count vectors sharing one rainbow pool.

    With M_q[v] the vector's size-X count distribution of quality q (on the
    union of the members' counts), mr the rainbow offset masses and B the
    cached clue probabilities, z[v] = sum M0 M1 M2 M3 mr B. It is contracted
    one axis at a time; the partial sums are kept so pass 2 can draw cells
    from exact sequential conditionals.
    """
    rb = _pool_pairs(members[0][2][COLORED])
    if len(rb["count"]) == 0:
        return None
    V = len(members)
    kds = [[_pool_k(ctx, q, m[2][q]) for m in members] for q in range(4)]
    hull = []
    for q in range(4):
        ks = [d["k"] for d in kds[q] if len(d["k"])]
        if not ks:
            return None
        hull.append(np.unique(np.concatenate(ks)))
    M = []
    for q in range(4):
        m = np.zeros((V, len(hull[q])))
        for v, d in enumerate(kds[q]):
            if len(d["k"]):
                m[v, np.searchsorted(hull[q], d["k"])] = d["mass"]
        M.append(m)
    arr, atom_idx = _sa_k_dense(ctx, store, [int(h.max()) for h in hull], rb)
    block = arr[np.ix_(*hull, atom_idx)]
    # cells inside at least one member's box; the others get zero weight anyway
    need = np.einsum("va,vb,vc,vd->abcd", *[(m > 0).astype(np.float64) for m in M], optimize=True) > 0
    base = np.asarray([m[3] for m in members])
    block = _sa_k_fill(ctx, store, block, need, hull, rb, atom_idx, np.exp(base - base.max()), M, stats)
    block = np.where(need[..., None], block, 0.0)
    t = block @ rb["mass"]
    x1 = np.einsum("abcd,vd->vabc", t, M[3])
    x2 = np.einsum("vabc,vc->vab", x1, M[2])
    x3 = np.einsum("vab,vb->va", x2, M[1])
    z = np.einsum("va,va->v", x3, M[0])
    return {"rb": rb, "hull": hull, "M": M, "block": block, "t": t, "x1": x1, "x2": x2, "x3": x3, "z": z}


def _rowwise_choice(p, rng):
    """One index per row of p, in proportion to the row's (non-negative) entries."""
    cum = np.cumsum(p, 1)
    return np.minimum((cum < rng.random(len(p))[:, None] * cum[:, -1:]).sum(1), p.shape[1] - 1)


def _sa_k_cells(g, r, rng):
    """(k tuple, rainbow offset) cells for draws of members r, from the exact conditionals."""
    M = g["M"]
    a = _rowwise_choice(M[0][r] * g["x3"][r], rng)
    b = _rowwise_choice(M[1][r] * g["x2"][r, a], rng)
    c = _rowwise_choice(M[2][r] * g["x1"][r, a, b], rng)
    d = _rowwise_choice(M[3][r] * g["t"][a, b, c], rng)
    e = _rowwise_choice(g["rb"]["mass"][None, :] * g["block"][a, b, c, d], rng)
    ke = np.stack([g["hull"][0][a], g["hull"][1][b], g["hull"][2][c], g["hull"][3][d]], 1)
    return ke, e


def _sa_k_combine(ctx, vectors, pools, worlds, rng, rows_loglik, trace=None):
    """k-level clue-first combination (see the block comment above)."""
    import time as _time
    s, shown, _ = ctx.size_avg
    stats = Counter(vectors=len(vectors), sa_k_level=1, draws=0, matched_draws=0, vectors_clue_infeasible=0)
    store = _sa_k_store(ctx)
    deadline = getattr(ctx, "deadline", None)
    candidates = _vector_table(ctx, vectors, pools, stats)
    if not candidates:
        return [[], [], dict(stats)]
    # Pass 1: exact clue mass of every vector, rainbow pool by rainbow pool.
    # Groups go in priority order, so a deadline leaves an unbiased priority
    # sample of groups.
    proxy = np.asarray([c[3] for c in candidates])
    by_rb = {}
    for i, c in enumerate(candidates):
        by_rb.setdefault(id(c[2][COLORED]), []).append(i)
    groups = [np.asarray(m) for m in by_rb.values()]
    gproxy = np.asarray([np.logaddexp.reduce(proxy[m]) for m in groups])
    order, logprio = _priority_order(rng, gproxy)
    done = []
    for j in order:
        _check_cancel(ctx)
        if deadline is not None and len(done) >= 5 and _time.monotonic() > deadline:
            break
        done.append((int(j), _sa_k_group(ctx, store, [candidates[i] for i in groups[j]], stats)))
    adj = _priority_adjust(gproxy, logprio, order, len(done))
    stats["vectors_paired"] = int(sum(len(groups[j]) for j, _ in done))
    if len(done) < len(groups):
        stats["vectors_cut_by_deadline"] = len(candidates) - stats["vectors_paired"]
    refs, bases = [], []
    for (j, g), a in zip(done, adj):
        if g is None:
            stats["vectors_clue_infeasible"] += len(groups[j])
            continue
        live = np.flatnonzero(g["z"] > 0)
        stats["vectors_clue_infeasible"] += len(groups[j]) - len(live)
        for r in live:
            refs.append((g, int(r), int(groups[j][r])))
            bases.append(proxy[groups[j][r]] + math.log(g["z"][r]) + a)
    if not refs:
        return [[], [], dict(stats)]
    bases = np.asarray(bases)
    wv = np.exp(bases - bases.max())
    stats["vector_ess"] = round(float(wv.sum() ** 2 / (wv ** 2).sum()), 1)
    alloc, counts = _allocate(rng, bases, worlds)
    # Pass 2: cells of every draw, group by group, vectorised over draws.
    recs, vids, kes, atom = [], [], [], []
    per_group = {}
    for (g, r, i), base, x, draws in zip(refs, bases, alloc, counts):
        if draws <= 0:
            continue
        N, n, pq, _ = candidates[i]
        per_group.setdefault(id(g), (g, []))[1].append((r, int(draws), len(recs)))
        recs.append((N, n, pq, base - math.log(x)))
    if not recs:
        return [[], [], dict(stats)]
    rbs = []
    for g, items in per_group.values():
        r = np.repeat([it[0] for it in items], [it[1] for it in items])
        ke, e = _sa_k_cells(g, r, rng)
        vids.append(np.repeat([it[2] for it in items], [it[1] for it in items]))
        kes.append(ke)
        atom.append(e)
        rbs.append(g["rb"])
    vid = np.concatenate(vids)
    ke = np.concatenate(kes)
    c_all = np.concatenate([rb["count"][e] for rb, e in zip(rbs, atom)])
    f_all = np.concatenate([rb["value"][e] for rb, e in zip(rbs, atom)])
    g_all = np.concatenate([rb["group"][e] for rb, e in zip(rbs, atom)])
    # Class-sum values of the four qualities, jointly conditioned on the clue,
    # for all draws sharing a k tuple at once.
    vq = np.zeros((len(vid), 4), np.int64)
    width = int(ke.max()) + 1
    kkey = ((ke[:, 0] * width + ke[:, 1]) * width + ke[:, 2]) * width + ke[:, 3]
    fc, fv = _sa_forced_total(ctx)
    for _, rows_k in _groups(kkey):
        kt = tuple(int(v) for v in ke[rows_k[0]])
        left = _sa_k_half(ctx, store, "left", kt[:2])
        right = _sa_k_half(ctx, store, "right", kt[2:])
        step = max(1, 2_000_000 // len(left["value"]))
        for c0 in range(0, len(rows_k), step):
            rows = rows_k[c0:c0 + step]
            total = int(sum(kt)) + fc + c_all[rows]
            lo = total * shown - f_all[rows] - fv
            hi = total * shown + total - 1 - f_all[rows] - fv
            a = np.searchsorted(right["sorted"], lo[:, None] - left["value"][None, :], "left")
            b = np.searchsorted(right["sorted"], hi[:, None] - left["value"][None, :], "right")
            lm = left["prob"][None, :] * (right["prefix"][b] - right["prefix"][a])
            cum = np.cumsum(lm, 1)
            li = np.minimum((cum < rng.random(len(rows))[:, None] * cum[:, -1:]).sum(1), lm.shape[1] - 1)
            r_ = np.arange(len(rows))
            aa, bb = a[r_, li], b[r_, li]
            u = right["prefix"][aa] + rng.random(len(rows)) * (right["prefix"][bb] - right["prefix"][aa])
            ri = right["order"][np.clip(np.searchsorted(right["prefix"], u, "right") - 1, aa, np.maximum(bb - 1, aa))]
            for col, q in enumerate(_LEFT_Q):
                vq[rows, q] = _class_sum(ctx, q, 35, kt[col])[0][left["idx"][col][li]]
            for col, q in enumerate(_RIGHT_Q):
                vq[rows, q] = _class_sum(ctx, q, 35, kt[2 + col])[0][right["idx"][col][ri]]
    size_count = ke.sum(1) + fc + c_all
    size_value = vq.sum(1) + fv + f_all
    stats["clue_violations"] = int(np.sum((size_count < 1) | (size_value // np.maximum(size_count, 1) != shown)))
    group = {q: ke[:, q] for q in range(4)}
    group[COLORED] = g_all
    fv_q = [_sa_forced(ctx, q)[1] for q in range(4)]
    value_add = {q: fv_q[q] + vq[:, q] for q in range(4)}

    def group_of(q, pool, N):
        return _pool_k(ctx, q, pool)["group_of"] if q < COLORED else _pool_pairs(pool)["group_of"]

    t, lw = _draw_worlds(ctx, recs, vid, rng, rows_loglik, stats, group=group, group_of=group_of,
                         value_add=value_add)
    return [[t], [lw], dict(stats)]


def combine(ctx: Context, vectors, pools, worlds, rng, rows_loglik, trace=None):
    if ctx.size_avg is not None and SA_TUPLE:
        if _sa_k_applicable(ctx):
            return tuple(_sa_k_combine(ctx, vectors, pools, worlds, rng, rows_loglik, trace))
        return tuple(_sa_combine(ctx, vectors, pools, worlds, rng, rows_loglik, trace))
    return _rb_combine(ctx, vectors, pools, worlds, rng, rows_loglik, trace)


def _rb_combine(ctx: Context, vectors, pools, worlds, rng, rows_loglik, trace=None, bases=None):
    if ctx.size_avg is None and bases is None and trace is None:
        return _rb_batched(ctx, vectors, pools, worlds, rng, rows_loglik)
    return _rb_window_combine(ctx, vectors, pools, worlds, rng, rows_loglik, trace, bases)


def _rb_window_combine(ctx: Context, vectors, pools, worlds, rng, rows_loglik, trace=None, bases=None):
    """Draw-then-weight combination with size averages handled by window weights.

    Used for the clue-first path's fallback vectors. ``bases`` (aligned with ``vectors``) overrides each vector's log weight,
    as used by the two-stage subsample of the clue-first path.
    """
    prior, ev = ctx.prior, ctx.ev
    totals, logw = [], []
    nflags = len(ctx.flags)
    full_mask = (1 << nflags) - 1
    radix = 101
    stats = {"vectors": len(vectors), "vectors_without_support": 0, "draws": 0, "matched_draws": 0}
    pending = []
    # Pass 1: every count vector's exact n-level weight (prior share x
    # per-quality factors x reveal terms), so draws can follow the posterior.
    weighted = []
    for vi_, (N, n, share) in enumerate(vectors):
        pq = [pools(q, n[q], N) for q in range(NQ)]
        if any(p is None for p in pq):
            stats["vectors_without_support"] += 1
            continue
        if bases is not None:
            weighted.append((N, n, pq, bases[vi_]))
            continue
        base = math.log(share) + sum(p.log_f for p in pq)
        if ctx.reveal_total:
            base += sum(gammaln(n[q] - ctx.rho[q] + 1) - gammaln(ev.revealed_qualities.get(q, 0) + 1)
                        - gammaln(n[q] - ctx.rho[q] - ev.revealed_qualities.get(q, 0) + 1)
                        for q in range(NQ) if ev.revealed_qualities.get(q, 0))
            base -= gammaln(N + 1) - gammaln(ctx.reveal_total + 1) - gammaln(N - ctx.reveal_total + 1)
        if math.isfinite(base):
            weighted.append((N, n, pq, base))
    if not weighted:
        return totals, logw, stats
    bases = np.asarray([b for *_, b in weighted])
    rel = np.exp(bases - bases.max())
    alloc = worlds * rel / rel.sum()
    # Stochastic rounding keeps the estimator unbiased: E[draws] = alloc.
    counts = np.floor(alloc).astype(np.int64) + (rng.random(len(alloc)) < alloc - np.floor(alloc))
    for vi, ((N, n, pq, base), x, draws) in enumerate(zip(weighted, alloc, counts)):
        if draws <= 0:
            continue
        draws = int(draws)
        base = base - math.log(x) + math.log(draws)
        last = int(np.argmax(n))
        others = [q for q in range(NQ) if q != last]
        pick = {q: rng.integers(len(pq[q].cw), size=draws) for q in others}
        cells = sum(pq[q].cells[pick[q]] for q in others)
        sizes = sum(pq[q].size_counts[pick[q]] for q in others)
        bits = np.zeros(draws, np.int64)
        for q in others:
            bits |= pq[q].flag_bits[pick[q]]
        need = full_mask & ~bits
        deficit = np.zeros(draws, np.int64)
        if ev.total_cells is not None:
            deficit = ev.total_cells - cells
        for s in ctx.exact_sizes:
            deficit = deficit * radix + (ev.size_count[s] - sizes[:, s])
        lp = pq[last]
        bad = np.zeros(draws, bool)
        for s in ctx.exact_sizes:
            bad |= (ev.size_count[s] - sizes[:, s]) < 0
        if ev.total_cells is not None:
            bad |= deficit < 0
        chosen = np.full(draws, -1)
        mass = np.zeros(draws)
        for m in np.unique(need[~bad]):
            rows_m = np.flatnonzero(~bad & (need == m))
            keys, order, prefix = _pool_lookup(ctx, lp, int(m), radix)
            a = np.searchsorted(keys, deficit[rows_m], "left")
            b = np.searchsorted(keys, deficit[rows_m], "right")
            total = prefix[b] - prefix[a]
            hit = total > 0
            u = prefix[a] + rng.random(len(rows_m)) * total
            pos = np.clip(np.searchsorted(prefix, u, "right") - 1, a, np.maximum(b - 1, a))
            chosen[rows_m[hit]] = order[pos[hit]]
            mass[rows_m[hit]] = total[hit] / len(lp.cw)
        ok = chosen >= 0
        stats["draws"] += draws
        stats["matched_draws"] += int(ok.sum())
        if not ok.any():
            continue
        sel = {q: pick[q][ok] for q in others}
        sel[last] = chosen[ok]
        lw = base - math.log(draws) + np.log(mass[ok])
        for q in others:
            lw = lw + np.log(pq[q].cw[sel[q]])
        world_cells = sum(pq[q].cells[sel[q]] for q in range(NQ))
        total_value = sum(pq[q].value[sel[q]] for q in range(NQ)).astype(float)
        if rows_loglik is not None and ev.map_rows:
            lw = lw + rows_loglik(int(ev.map_rows), world_cells)
        if ctx.size_avg is not None:
            s, shown, _ = ctx.size_avg
            c_s = sum(pq[q].s_count[sel[q]] for q in range(NQ))
            fixed = sum(pq[q].s_fixed[sel[q]] for q in range(NQ))
            ks = np.stack([pq[q].s_res[sel[q]] for q in range(NQ)], 1)
            s_value = sum(pq[q].s_value[sel[q]] for q in range(NQ))
            integrated = ks.sum(1) > 0
            total_value = np.where(integrated, total_value - s_value + c_s * shown + (c_s - 1) / 2.0,
                                   total_value)
            # Window probabilities are computed once for all count vectors below.
            pending.append((ks, c_s * shown - fixed, c_s * shown + c_s - 1 - fixed, c_s))
            if trace is not None:
                trace.setdefault("vector", []).append(np.full(len(lw), vi))
                trace.setdefault("c_s", []).append(c_s)
                trace.setdefault("fixed", []).append(fixed)
                trace.setdefault("ks", []).append(ks)
                trace.setdefault("rainbow_s", []).append(pq[COLORED].s_count[sel[COLORED]])
                trace.setdefault("last", []).append(np.full(len(lw), last))
                trace.setdefault("base_lw", []).append(np.asarray(lw, float).copy())
        if prior.world_log_tilt is not None:
            lw = lw + prior.world_log_tilt(N, n[COLORED], pq[COLORED].value[sel[COLORED]], total_value)
        totals.append(total_value)
        logw.append(lw)
    if ctx.size_avg is not None and logw:
        ks, lo, hi, c_s = (np.concatenate(parts) for parts in zip(*pending))
        probs = _size_avg_probabilities(ctx, ks, lo, hi, c_s)
        if trace is not None:
            trace["window_prob"] = probs
            trace["vector_alloc"] = alloc
            trace["vector_base"] = bases
        with np.errstate(divide="ignore"):
            all_lw = np.concatenate(logw) + np.log(probs)
        totals, logw = [np.concatenate(totals)], [all_lw]
    keep = [np.isfinite(x) for x in logw]
    totals = [t[k] for t, k in zip(totals, keep)]
    logw = [x[k] for x, k in zip(logw, keep)]
    return totals, logw, stats


_POOL_CACHE: "OrderedDict" = None
POOL_CACHE_MAX = 400


def _pool_signature(ctx, q, n_q, N, K):
    """Everything a pool's content depends on, so later rounds can reuse it."""
    prior, ev = ctx.prior, ctx.ev
    idx = prior.by_quality[q]
    return (id(prior), q, n_q, N if q == COLORED else None, K,
            ctx.allowed[idx].tobytes(),
            tuple(ctx.forced.get(int(i), 0) for i in idx),
            tuple(ctx.revealed.get(int(i), 0) for i in idx),
            tuple(int(i) in ctx.atleast for i in idx),
            ev.total_value.get(q), ev.avg_area.get(q),
            tuple(m[idx].tobytes() for m in ctx.flags),
            (ctx.size_avg[0], ctx.size_avg[2][q]) if ctx.size_avg else None)


def sample_posterior(prior: Prior, ev: Evidence, rng, *, rows_loglik=None, n_vectors=10000,
                     pool_size=2000, worlds=25_000, rounds=5, min_ess=500.0, max_vectors=40_000,
                     min_ess_size_avg=1500.0, time_budget=None, cancel=None):
    """Adaptive: start small and double vectors/worlds until ESS >= target.

    Size-average weights are spikier (discrete window probabilities), so
    their decisions need a higher ESS for the same Monte Carlo error.
    With ``time_budget`` (seconds) escalation stops once the budget is spent;
    the result so far is returned and flagged low_ess if under target.
    """
    import time as _time
    started = _time.monotonic()
    global _POOL_CACHE
    from collections import OrderedDict
    if _POOL_CACHE is None:
        _POOL_CACHE = OrderedDict()
    ctx = Context(prior, ev)
    ctx.cancel = cancel
    if ctx.conflict:
        return Result("no_support", np.zeros(0), np.zeros(0), {"reason": ctx.conflict})
    if ctx.size_avg is not None:
        min_ess = max(min_ess, min_ess_size_avg)
    ctx.deadline = ctx.hard_deadline = None
    built = Counter()
    signatures = {}

    def pools(q, n_q, N):
        local = (q, n_q, N if q == COLORED else None)
        key = signatures.get(local)
        if key is None:
            key = signatures[local] = _pool_signature(ctx, q, n_q, N, pool_size)
        if key in _POOL_CACHE:
            _POOL_CACHE.move_to_end(key)
            built["reused"] += 1
        else:
            _check_cancel(ctx)
            _POOL_CACHE[key] = build_pool(ctx, q, n_q, N, pool_size, rng)
            built["built"] += 1
            while len(_POOL_CACHE) > POOL_CACHE_MAX:
                _POOL_CACHE.popitem(last=False)
        return _POOL_CACHE[key]

    rounds_out, diag = [], Counter()
    per_world = None
    for attempt in range(rounds):
        _check_cancel(ctx)
        scale = 1.0
        round_start = _time.monotonic()
        if time_budget is not None:
            remaining = started + time_budget - round_start
            if attempt and per_world is not None:
                # A doubled round that cannot finish in time is shrunk to fit
                # (rounds are pooled by size, so any size is unbiased), or
                # skipped when even a quarter of it does not fit.
                fits = 0.8 * remaining / (per_world * worlds * 2 ** attempt)
                if fits < 0.25:
                    diag["stopped_by_time_budget"] = 1
                    break
                scale = min(1.0, fits)
            # Within the round, the clue-first paths stop adding count
            # vectors (priority sample) and exact clue cells (recorded
            # bound) at this point; drawing the worlds takes the rest.
            ctx.deadline = ctx.hard_deadline = round_start + SOFT_SHARE * remaining
        size = max(1, int(worlds * 2 ** attempt * scale))
        vectors = sample_counts(ctx, max(1, int(min(max_vectors, n_vectors * 2 ** attempt) * scale)), rng)
        t, lw, stats = combine(ctx, vectors, pools, size, rng, rows_loglik)
        per_world = (_time.monotonic() - round_start) / size
        cut = stats.get("vectors_cut_by_deadline", 0) > 0 or stats.get("sa_late", 0) > 0
        if attempt and cut:
            # An extra round cut short is unbiased but can be far noisier
            # than the complete first round, so it is dropped, not pooled.
            diag["rounds_dropped"] += 1
            diag["stopped_by_time_budget"] = 1
            break
        if cut:
            diag["first_round_cut"] = 1
        diag.update(stats)
        if t and sum(len(x) for x in t):
            rounds_out.append((np.concatenate(t), np.concatenate(lw), size))
        if not rounds_out:
            continue
        # Each round is an unbiased estimate of the same posterior mass;
        # pool them in proportion to their size.
        total_size = sum(r[2] for r in rounds_out)
        allw = np.concatenate([lw_r + math.log(size_r / total_size) for _, lw_r, size_r in rounds_out])
        w = np.exp(allw - allw.max())
        if w.sum() ** 2 / (w ** 2).sum() >= min_ess:
            break
        if time_budget is not None and _time.monotonic() - started >= time_budget:
            diag["stopped_by_time_budget"] = 1
            break
    diag.update(rounds_used=attempt + 1, pools_built=built["built"], pools_reused=built["reused"])
    if not rounds_out:
        return Result("no_support", np.zeros(0), np.zeros(0), dict(diag, reason="zero posterior mass"))
    total_size = sum(r[2] for r in rounds_out)
    allw = np.concatenate([lw_r + math.log(size_r / total_size) for _, lw_r, size_r in rounds_out])
    w = np.exp(allw - allw.max())
    diag["worlds_requested"] = total_size
    result = Result("ok", np.concatenate([t_r for t_r, _, _ in rounds_out]), w, dict(diag))
    # Absolute log weights: their exp-sum estimates the posterior mass, so
    # independent runs (lanes) can be pooled.
    result.log_weights = allw
    if result.ess < min_ess or diag.get("sa_skip_bound", 0.0) > SA_SKIP_FLAG or diag.get("first_round_cut"):
        result.status = "low_ess"
    return result


def merge_results(results, min_ess):
    """Pool independent runs of sample_posterior (e.g. one per CPU lane).

    Each run's weights are an unbiased estimate of the same posterior mass;
    runs are weighted by their world counts, like rounds within a run. The
    spread of the runs' point estimates gives a direct Monte Carlo error.
    """
    ok = [r for r in results if r.status != "no_support" and getattr(r, "log_weights", None) is not None
          and len(r.totals)]
    if not ok:
        return results[0]
    sizes = np.asarray([r.diagnostics["worlds_requested"] for r in ok], float)
    allw = np.concatenate([r.log_weights + math.log(s / sizes.sum()) for r, s in zip(ok, sizes)])
    w = np.exp(allw - allw.max())
    diag = Counter()
    for r in ok:
        diag.update({k: v for k, v in r.diagnostics.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)})
    diag = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in diag.items()}
    diag["lanes"] = len(results)
    diag["lanes_with_support"] = len(ok)
    diag["sa_skip_bound"] = max(r.diagnostics.get("sa_skip_bound", 0.0) for r in ok)
    merged = Result("ok", np.concatenate([r.totals for r in ok]), w, diag)
    merged.log_weights = allw
    if len(ok) > 1:
        points = np.asarray([r.quantile(0.5, reweight=1.0 / r.totals) for r in ok])
        centre = merged.quantile(0.5, reweight=1.0 / merged.totals)
        merged.diagnostics["lane_rel_se"] = round(float(points.std(ddof=1) / math.sqrt(len(ok)) / centre), 4)
    if merged.ess < min_ess or diag["sa_skip_bound"] > SA_SKIP_FLAG or diag.get("first_round_cut"):
        merged.status = "low_ess"
    return merged
