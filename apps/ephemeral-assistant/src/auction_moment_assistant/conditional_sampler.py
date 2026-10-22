"""Bounded inference over identity counts AND a concrete generation order.

CP-SAT supplies feasible initial states, never probability weights. The target
is the frozen count prior times a uniform permutation, exact first-fit evidence
and an attributable random-reveal selection likelihood. No constraint kernels
are widened when evidence is rare. A time limit is not convergence evidence.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np

from .reveal_likelihood import build_reveal_observations, reveal_log_likelihood
from .world_layout import find_compatible_sequence, layout_compatible


@dataclass(frozen=True)
class ConditionalSamples:
    counts: np.ndarray
    effective_sample_size: float
    converged: bool
    diagnostics: dict


def ordered_log_probability(model, counts: Sequence[int]) -> float:
    """P(counts) / number of distinct permutations of that multiset."""
    score = float(model.log_probability(counts, rows=0, rows_exact=False).total)
    return (
        score
        - math.lgamma(sum(counts) + 1)
        + sum(math.lgamma(int(value) + 1) for value in counts)
    )


def chain_diagnostics(chains: np.ndarray) -> tuple[float, float]:
    """Split R-hat and initial-positive-pair autocorrelation ESS.

    Includes between-chain variance in the autocorrelation estimate; a chain
    stuck at a different constant cannot acquire an artificially large ESS.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2 or chains.shape[0] < 2 or chains.shape[1] < 16:
        return float("inf"), 0.0
    half = chains.shape[1] // 2
    split = np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)
    m, n = split.shape
    variances = np.var(split, axis=1, ddof=1)
    within = float(np.mean(variances))
    between = float(n * np.var(np.mean(split, axis=1), ddof=1))
    if within == 0:
        return (1.0, float(m * n)) if between == 0 else (float("inf"), 0.0)
    variance = (n - 1) / n * within + between / n
    rhat = math.sqrt(max(0.0, variance / within))
    centered = split - np.mean(split, axis=1, keepdims=True)
    rho = [1.0]
    for lag in range(1, n):
        covariance = float(
            np.mean(np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / n)
        )
        rho.append(1.0 - (within - covariance) / variance)
    pair_sum = 0.0
    previous = float("inf")
    for lag in range(0, n - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair < 0:
            break
        previous = min(previous, pair)
        pair_sum += previous
    tau = max(1.0, -1.0 + 2.0 * pair_sum)
    return rhat, min(float(m * n), float(m * n) / tau)


def enumerate_block(base_model, variables, counts, block, *, deadline, cap=256):
    """Return the COMPLETE conditional count support or None.

    Excluding count vectors (rather than solver auxiliary assignments) avoids
    double-counting. Partial enumerations are never used for Gibbs proposals.
    """
    from ortools.sat.python import cp_model

    model = base_model.clone()
    model.clear_objective()
    selected = tuple(sorted(block))
    outside = set(range(len(variables))) - set(selected)
    local = tuple(model.get_int_var_from_proto_index(v.index) for v in variables)
    for index in outside:
        model.add(local[index] == int(counts[index]))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    support = []
    while time.monotonic() < deadline:
        solver.parameters.max_time_in_seconds = max(0.001, deadline - time.monotonic())
        status = solver.solve(model)
        if status == cp_model.INFEASIBLE:
            return support
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            return None
        candidate = tuple(int(solver.value(variable)) for variable in local)
        support.append(candidate)
        if len(support) > cap:
            return None
        model.add_forbidden_assignments(
            [local[index] for index in selected],
            [[candidate[index] for index in selected]],
        )
    return None


class ConditionalWorldSampler:
    def __init__(
        self, model, catalog, *, budget_seconds=2.0, chains=4, independent_draws=1024
    ):
        self.model = model
        self.catalog = tuple(catalog)
        self.budget_seconds = float(budget_seconds)
        if not math.isfinite(self.budget_seconds) or self.budget_seconds <= 0:
            raise ValueError("budget_seconds must be finite and positive")
        self.chain_count = max(2, int(chains))
        self.independent_draws = max(0, int(independent_draws))
        self.values = np.asarray([item.value for item in catalog], dtype=np.int64)
        self.groups = tuple(
            tuple(i for i, item in enumerate(catalog) if item.quality == quality)
            for quality in sorted({item.quality for item in catalog})
        )
        self.shape_groups = {
            (item.width, item.height): tuple(
                i
                for i, other in enumerate(catalog)
                if (other.width, other.height) == (item.width, item.height)
            )
            for item in catalog
        }

    def sample(
        self,
        snapshot,
        *,
        rng,
        filter_population: Callable,
        prior_population: np.ndarray | None = None,
    ) -> ConditionalSamples:
        start = time.monotonic()
        deadline = start + self.budget_seconds
        observations, reveal_notes = build_reveal_observations(snapshot, self.catalog)
        notes = list(reveal_notes)
        diagnostics = {
            "target": "frozen_v2_counts_uniform_order_exact_first_fit",
            "budget_seconds": self.budget_seconds,
            "reveal_observations": len(observations),
            "notes": notes,
            "iid_draws": 0,
            "iid_accepted": 0,
            "seed_count": 0,
            "sweeps": 0,
            "accepted_moves": 0,
            "complete_blocks": 0,
            "incomplete_blocks_skipped": 0,
        }
        from .conditional_constraints import compile_hard_constraints
        from .conditional_count_moves import (
            ExactCountBlockEnumerator,
            CountSignatureIndex,
            build_symmetric_count_moves,
        )
        from ortools.sat.python import cp_model

        base, variables, constraint_notes = compile_hard_constraints(
            snapshot, self.catalog, self.model
        )
        diagnostics["initialization_seconds"] = time.monotonic() - start
        start = time.monotonic()
        deadline = start + self.budget_seconds
        notes.extend(constraint_notes)
        if any(note.startswith("unsupported:") for note in constraint_notes):
            diagnostics.update(
                method="unsupported_evidence", elapsed_seconds=time.monotonic() - start
            )
            return ConditionalSamples(
                np.empty((0, len(self.catalog)), dtype=np.int16), 0, False, diagnostics
            )
        enumerator = ExactCountBlockEnumerator(base, variables)
        signature_index = CountSignatureIndex(enumerator)
        needs_layout = bool(
            snapshot.map_height_exact
            or snapshot.map_items
            or (
                snapshot.completeness_confirmed
                and any(
                    (event.get("semantics") or {}).get("effect")
                    in {
                        "reveal_position_by_quality",
                        "reveal_outline_by_quality",
                        "reveal_outline_by_size",
                    }
                    and (event.get("semantics") or {}).get("reveal_outcome_available")
                    is not False
                    for event in snapshot.events
                    if int(event["round_number"]) <= snapshot.round_number
                )
            )
        )

        def spatially_valid(sequence):
            return not needs_layout or layout_compatible(
                sequence, self.catalog, snapshot
            )

        def reveal(counts):
            return reveal_log_likelihood(
                counts, observations, self.catalog, conditioned_on_positions=True
            )

        @lru_cache(maxsize=16384)
        def count_score(counts):
            return float(
                self.model.log_probability(counts, rows=0, rows_exact=False).total
            ) + reveal(counts)

        def ordered_score(counts):
            return (
                count_score(counts)
                - math.lgamma(sum(counts) + 1)
                + sum(math.lgamma(c + 1) for c in counts)
            )

        def sequence_for(counts):
            sequence = [i for i, count in enumerate(counts) for _ in range(int(count))]
            rng.shuffle(sequence)
            return tuple(sequence)

        # Pure prior rejection yields independent posterior samples. Empirical
        # settlement worlds are deliberately NOT mixed into this population.
        independent, seeds = [], []
        pool = prior_population if prior_population is not None else ()
        pool_index = 0
        # Fixed draw count prevents world-dependent computation cost from
        # selecting which retained values qualify as independent samples.
        fixed_draws = self.independent_draws
        iid_complete = True
        prior_deadline = start + self.budget_seconds * 0.25
        while diagnostics["iid_draws"] < fixed_draws:
            if time.monotonic() >= prior_deadline:
                iid_complete = False
                break
            if pool_index < len(pool):
                batch = np.asarray(
                    pool[
                        pool_index : pool_index
                        + min(256, fixed_draws - diagnostics["iid_draws"])
                    ],
                    dtype=np.int16,
                )
                pool_index += len(batch)
            else:
                batch = np.asarray(
                    [
                        self.model.sample_prior_counts(rng)
                        for _ in range(min(64, fixed_draws - diagnostics["iid_draws"]))
                    ],
                    dtype=np.int16,
                )
            diagnostics["iid_draws"] += len(batch)
            for counts in batch[filter_population(snapshot, batch)]:
                sequence = sequence_for(counts)
                if not spatially_valid(sequence):
                    continue
                likelihood = reveal(counts)
                if math.isfinite(likelihood):
                    if len(seeds) < self.chain_count * 4:
                        seeds.append((tuple(map(int, counts)), sequence))
                    if math.log(max(rng.random(), 1e-300)) <= likelihood:
                        independent.append(tuple(map(int, counts)))
        diagnostics["iid_accepted"] = len(independent)
        diagnostics["iid_fixed_batch_complete"] = iid_complete
        if iid_complete and len(independent) >= 64:
            diagnostics.update(
                method="independent_prior_rejection",
                elapsed_seconds=time.monotonic() - start,
            )
            return ConditionalSamples(
                np.asarray(independent, dtype=np.int16),
                float(len(independent)),
                True,
                diagnostics,
            )

        # Multiple starts close to independently drawn priors, not optimized
        # toward expensive worlds. Their distribution is NOT the posterior.
        seed_deadline = start + self.budget_seconds * 0.48
        attempts = 0
        while len(seeds) < self.chain_count and time.monotonic() < seed_deadline:
            attempts += 1
            search = base.clone()
            local = [search.get_int_var_from_proto_index(v.index) for v in variables]
            prior = self.model.sample_prior_counts(rng)
            deviations = []
            for i, variable in enumerate(local):
                deviation = search.new_int_var(
                    0, self.model.max_count, f"seed_delta_{i}"
                )
                search.add_abs_equality(deviation, variable - int(prior[i]))
                deviations.append(deviation)
            search.minimize(sum(deviations))
            # A constrained seed search is an initialization heuristic only.
            # Bound N to a fresh prior draw on alternate attempts so a solver's
            # first feasible (but extremely high-N) solution cannot seed every
            # chain. Unrestricted attempts still distinguish proven conflicts.
            restricted_n = attempts % 2 == 1
            if restricted_n:
                search.add(sum(local) == sum(prior))
            for variable, hint in zip(local, prior):
                search.add_hint(variable, int(hint))
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.max_time_in_seconds = min(
                0.1, max(0.001, seed_deadline - time.monotonic())
            )
            status = solver.solve(search)
            if status == cp_model.INFEASIBLE:
                if restricted_n:
                    continue
                diagnostics["numeric_infeasible"] = True
                break
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                continue
            counts = tuple(int(solver.value(v)) for v in local)
            if not math.isfinite(
                ordered_log_probability(self.model, counts) + reveal(counts)
            ):
                continue
            sequence = (
                find_compatible_sequence(
                    counts,
                    self.catalog,
                    snapshot,
                    rng,
                    maximum_nodes=1000,
                    deadline=seed_deadline,
                )
                if needs_layout
                else sequence_for(counts)
            )
            if sequence is not None:
                seeds.append((counts, sequence))
        diagnostics.update(seed_count=len(seeds), seed_attempts=attempts)
        if len(seeds) < self.chain_count:
            diagnostics.update(
                method="conditional_seed_search_failed",
                elapsed_seconds=time.monotonic() - start,
            )
            return ConditionalSamples(
                np.asarray(independent, dtype=np.int16).reshape(-1, len(self.catalog)),
                float(len(independent)),
                False,
                diagnostics,
            )
        states = seeds[: self.chain_count]
        traces = [[] for _ in states]
        reactions = build_symmetric_count_moves(
            enumerator,
            groups=self.groups,
            deadline=min(deadline, time.monotonic() + 0.15),
        )
        diagnostics.update(
            reaction_count=len(reactions),
            reaction_accepted=0,
            token_block_accepted=0,
            layout_required=needs_layout,
            block_enumerator="exact_integer_linear",
        )
        # Block selection depends only on catalog/evidence, never the current
        # state. Caching complete conditional supports is safe given outside.
        cache = {}
        sweep = 0
        while time.monotonic() < deadline:
            for chain_index, (counts, sequence) in enumerate(states):
                if time.monotonic() >= deadline:
                    break
                # A genuine prior independence proposal makes disconnected
                # numeric/quality modes reachable, including changes of N.
                # Its prior/order factors cancel in the MH ratio.
                if rng.random() < 0.2:
                    proposed_counts = tuple(self.model.sample_prior_counts(rng))
                    if filter_population(snapshot, np.asarray([proposed_counts]))[0]:
                        proposed_order = sequence_for(proposed_counts)
                        if spatially_valid(proposed_order):
                            ratio = reveal(proposed_counts) - reveal(counts)
                            if math.log(max(rng.random(), 1e-300)) <= ratio:
                                counts, sequence = proposed_counts, proposed_order
                                diagnostics["accepted_moves"] += 1
                # Symmetric position/identity moves maintain concrete order.
                for _ in range(6):
                    candidate = list(sequence)
                    candidate_counts = list(counts)
                    if rng.random() < 0.4 and len(sequence) > 1:
                        a, b = rng.sample(range(len(sequence)), 2)
                        candidate[a], candidate[b] = candidate[b], candidate[a]
                    elif sequence:
                        a = rng.randrange(len(sequence))
                        old = sequence[a]
                        item = self.catalog[old]
                        replacement = rng.choice(
                            self.shape_groups[(item.width, item.height)]
                        )
                        candidate[a] = replacement
                        candidate_counts[old] -= 1
                        candidate_counts[replacement] += 1
                    else:
                        continue
                    candidate_counts = tuple(candidate_counts)
                    candidate = tuple(candidate)
                    if candidate == sequence:
                        continue
                    if not filter_population(snapshot, np.asarray([candidate_counts]))[
                        0
                    ]:
                        continue
                    if not spatially_valid(candidate):
                        continue
                    logratio = ordered_score(candidate_counts) - ordered_score(counts)
                    if math.log(max(rng.random(), 1e-300)) <= logratio:
                        counts, sequence = candidate_counts, candidate
                        diagnostics["accepted_moves"] += 1
                # Select existing token positions within a fixed catalog
                # quality group. Replacement preserves that group's count,
                # so the position-selection probability is identical in the
                # reverse move. The multiset/permutation proposal correction
                # is required; these are not uniformly selected sequences.
                for _ in range(3):
                    group = rng.choice(self.groups)
                    positions = [
                        i for i, identity in enumerate(sequence) if identity in group
                    ]
                    if len(positions) < 2:
                        continue
                    positions = rng.sample(
                        positions, min(len(positions), rng.choice((2, 3)))
                    )
                    old_tokens = tuple(sorted(sequence[i] for i in positions))
                    choices = signature_index.lookup(group, old_tokens)
                    new_tokens = list(rng.choice(choices))
                    if tuple(new_tokens) == old_tokens:
                        continue
                    candidate_counts = list(counts)
                    for identity in old_tokens:
                        candidate_counts[identity] -= 1
                    for identity in new_tokens:
                        candidate_counts[identity] += 1
                    candidate_counts = tuple(candidate_counts)
                    if not filter_population(snapshot, np.asarray([candidate_counts]))[
                        0
                    ]:
                        continue
                    rng.shuffle(new_tokens)
                    candidate = list(sequence)
                    for position, identity in zip(positions, new_tokens):
                        candidate[position] = identity
                    candidate = tuple(candidate)
                    if not spatially_valid(candidate):
                        continue
                    log_multiplicity_ratio = sum(
                        math.lgamma(n + 1) for n in Counter(old_tokens).values()
                    ) - sum(math.lgamma(n + 1) for n in Counter(new_tokens).values())
                    ratio = (
                        ordered_score(candidate_counts)
                        - ordered_score(counts)
                        + log_multiplicity_ratio
                    )
                    if math.log(max(rng.random(), 1e-300)) <= ratio:
                        counts, sequence = candidate_counts, candidate
                        diagnostics["accepted_moves"] += 1
                        diagnostics["token_block_accepted"] += 1
                # Static integer reactions can change several identities and
                # N together while preserving every exact linear statistic.
                # Uniform sign symmetrizes the count proposal. Re-randomizing
                # the entire order cancels its multiset factor in the ratio.
                if reactions:
                    delta = rng.choice(reactions)
                    sign = rng.choice((-1, 1))
                    candidate_counts = tuple(
                        c + sign * d for c, d in zip(counts, delta)
                    )
                    if (
                        min(candidate_counts) >= 0
                        and filter_population(snapshot, np.asarray([candidate_counts]))[
                            0
                        ]
                    ):
                        score = count_score(candidate_counts)
                        if math.isfinite(score):
                            candidate = sequence_for(candidate_counts)
                            if spatially_valid(candidate) and math.log(
                                max(rng.random(), 1e-300)
                            ) <= score - count_score(counts):
                                counts, sequence = candidate_counts, candidate
                                diagnostics["accepted_moves"] += 1
                                diagnostics["reaction_accepted"] += 1
                # Complete block heat-bath for counts, then a uniform order.
                # Proposal marginal is P(counts | outside, numeric evidence),
                # so only the hard layout indicator remains in acceptance.
                group = (
                    tuple(range(len(self.catalog)))
                    if rng.random() < 0.35
                    else rng.choice(self.groups)
                )
                block = tuple(
                    sorted(rng.sample(group, min(len(group), rng.choice((2, 3)))))
                )
                key = (block, tuple(c for i, c in enumerate(counts) if i not in block))
                if key not in cache:
                    support = enumerator.enumerate(
                        counts,
                        block,
                        deadline=min(deadline, time.monotonic() + 0.025),
                    )
                    if support is not None:
                        cache[key] = support
                    if len(cache) > 128:
                        cache.pop(next(iter(cache)))
                else:
                    support = cache[key]
                if support is None:
                    diagnostics["incomplete_blocks_skipped"] += 1
                elif support:
                    diagnostics["complete_blocks"] += 1
                    logweights = [count_score(c) for c in support]
                    maximum = max(logweights)
                    if math.isfinite(maximum):
                        chosen = rng.choices(
                            support, weights=[math.exp(v - maximum) for v in logweights]
                        )[0]
                        proposed = sequence_for(chosen)
                        if spatially_valid(proposed):
                            if proposed != sequence:
                                diagnostics["accepted_moves"] += 1
                            counts, sequence = chosen, proposed
                states[chain_index] = counts, sequence
                traces[chain_index].append(counts)
            sweep += 1
        length = min(map(len, traces))
        burn = max(16, length // 3)
        retained = np.asarray([trace[burn:length] for trace in traces], dtype=np.int16)
        if retained.size == 0:
            retained = np.empty((len(states), 0, len(self.catalog)), dtype=np.int16)
        feature_traces = []
        if retained.shape[1]:
            feature_traces.extend((retained @ self.values, retained.sum(axis=2)))
            feature_traces.extend(
                retained[:, :, group].sum(axis=2) for group in self.groups
            )
            # Numeric events can make totals/qualities constant while identity
            # or shape modes remain stuck. Monitor every identity as well.
            feature_traces.extend(
                retained[:, :, index] for index in range(len(self.catalog))
            )
        checks = [chain_diagnostics(feature) for feature in feature_traces]
        max_rhat = max((r for r, _ in checks), default=float("inf"))
        ess = min((e for _, e in checks), default=0.0)
        flattened = retained.reshape(-1, len(self.catalog))
        unique = len({tuple(row) for row in flattened})
        # Diagnostics are a conservative finite-run gate, not a proof that all
        # posterior modes have been discovered or that intervals are calibrated.
        converged = (
            length - burn >= 32 and max_rhat <= 1.1 and ess >= 32 and unique >= 32
        )
        diagnostics.update(
            method="ordered_block_gibbs_mh",
            sweeps=sweep,
            retained_per_chain=max(0, length - burn),
            unique_count_states=unique,
            max_split_rhat=max_rhat if math.isfinite(max_rhat) else None,
            effective_sample_size=ess,
            elapsed_seconds=time.monotonic() - start,
        )
        return ConditionalSamples(flattened, ess, converged, diagnostics)
