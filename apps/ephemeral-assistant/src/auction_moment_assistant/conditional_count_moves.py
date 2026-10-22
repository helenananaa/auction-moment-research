"""Exact, small-block count supports without repeatedly invoking CP-SAT.

The adapter accepts the numeric model produced by ``compile_hard_constraints``.
Its nonconstant auxiliary variable is the total number of items.  Eliminating
that variable and fixed constants leaves bounded integer linear constraints.
Enumerating a two/three-identity block is then cheap integer arithmetic.  An
equality removes one enumeration dimension when possible.  No partial support
is returned, so callers can use the result in a conditional heat-bath kernel.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from itertools import combinations_with_replacement
from typing import Sequence

import numpy as np


class ExactCountBlockEnumerator:
    """Compile a supported CP-SAT model once, then enumerate complete blocks.

    Unsupported model features raise ``ValueError`` rather than dropping a
    constraint.  A deadline, oversized support, or excessive enumeration grid
    returns ``None``; an empty list means the complete support is empty.
    """

    def __init__(self, base_model, variables):
        proto = base_model.proto
        indices = tuple(int(variable.index) for variable in variables)
        if len(set(indices)) != len(indices):
            raise ValueError("count variables must be distinct")
        self.dimension = len(indices)
        count_index = {index: local for local, index in enumerate(indices)}
        auxiliary = set(range(len(proto.variables))) - set(indices)
        total_candidates = tuple(
            index for index in auxiliary if proto.variables[index].name == "total_count"
        )
        if len(total_candidates) != 1:
            raise ValueError("expected one total-count auxiliary")
        total_index = total_candidates[0]
        constants = {}
        for index in auxiliary - {total_index}:
            domain = tuple(map(int, proto.variables[index].domain))
            if len(domain) != 2 or domain[0] != domain[1]:
                raise ValueError("unsupported nonconstant auxiliary variable")
            constants[index] = domain[0]
        relation_verified = False
        for constraint in proto.constraints:
            if not constraint.has_linear() or constraint.enforcement_literal:
                continue
            linear = constraint.linear
            terms = dict(zip(map(int, linear.vars), map(int, linear.coeffs)))
            if (
                list(linear.domain) == [0, 0]
                and terms.get(total_index) in (-1, 1)
                and len(terms) == len(indices) + 1
                and all(terms.get(index) == -terms[total_index] for index in indices)
            ):
                relation_verified = True
                break
        if not relation_verified:
            raise ValueError("total-count elimination relation is missing")

        rows, domains, tables = [], [], []
        self.bounds = []
        maximum_absolute_count = 0

        def coefficients(raw_vars, raw_coefficients):
            row = [0] * self.dimension
            offset = 0
            for index, coefficient in zip(raw_vars, raw_coefficients):
                index, coefficient = int(index), int(coefficient)
                if index in count_index:
                    row[count_index[index]] += coefficient
                elif index == total_index:
                    row = [value + coefficient for value in row]
                elif index in constants:
                    offset += coefficient * constants[index]
                else:
                    raise ValueError("unsupported linear variable")
            return row, offset

        def shifted_domain(domain, offset):
            # The checked count linear forms are bounded by 2**61.  Clipping
            # distant CP-SAT sentinel bounds after subtracting constants is
            # exact on that range and avoids overflowing NumPy comparisons.
            domain = tuple(map(int, domain))
            result = []
            for low, high in zip(domain[::2], domain[1::2]):
                low, high = max(-2**62, int(low) - offset), min(2**62, int(high) - offset)
                if low <= high:
                    result.extend((low, high))
            return tuple(result)

        def literal_value(literal):
            literal = int(literal)
            index = literal if literal >= 0 else -literal - 1
            if index not in constants or constants[index] not in (0, 1):
                raise ValueError("unsupported nonconstant Boolean literal")
            return constants[index] == (1 if literal >= 0 else 0)

        for index, variable in enumerate(proto.variables):
            domain = tuple(map(int, variable.domain))
            if not domain or len(domain) % 2:
                raise ValueError("invalid variable domain")
            if index in count_index:
                local = count_index[index]
                if domain[0] < 0:
                    raise ValueError("counts must be nonnegative")
                self.bounds.append((local, domain[0], domain[-1]))
                maximum_absolute_count = max(maximum_absolute_count, domain[-1])
            row, offset = coefficients((index,), (1,))
            rows.append(row)
            domains.append(shifted_domain(domain, offset))

        self.infeasible = False
        for constraint in proto.constraints:
            if constraint.enforcement_literal:
                if not all(literal_value(literal) for literal in constraint.enforcement_literal):
                    continue
            if constraint.has_linear():
                row, offset = coefficients(constraint.linear.vars, constraint.linear.coeffs)
                rows.append(row)
                domains.append(shifted_domain(constraint.linear.domain, offset))
            elif constraint.has_table():
                table = constraint.table
                if table.negated or len(table.exprs) != 1:
                    raise ValueError("only positive one-expression tables supported")
                expression = table.exprs[0]
                row, offset = coefficients(expression.vars, expression.coeffs)
                allowed = np.asarray(
                    [
                        int(value) - int(expression.offset) - offset
                        for value in table.values
                        if -2**61 <= int(value) - int(expression.offset) - offset <= 2**61
                    ],
                    dtype=np.int64,
                )
                tables.append((np.asarray(row, dtype=np.int64), allowed))
            elif constraint.has_bool_or():
                self.infeasible |= not any(literal_value(literal) for literal in constraint.bool_or.literals)
            else:
                raise ValueError("unsupported numeric constraint")

        self.bounds.sort()
        self.bounds = tuple((lower, upper) for _, lower, upper in self.bounds)
        if len(self.bounds) != self.dimension:
            raise ValueError("count-variable domain missing")
        for row in rows + [row for row, _ in tables]:
            if sum(abs(value) for value in row) * maximum_absolute_count >= 2**61:
                raise ValueError("integer arithmetic would overflow")
        self.matrix = np.asarray(rows, dtype=np.int64).reshape(-1, self.dimension)
        self.domains = tuple(domains)
        self.tables = tuple(tables)
        self.equalities = tuple(
            (index, domain[0])
            for index, domain in enumerate(self.domains)
            if len(domain) == 2 and domain[0] == domain[1]
        )

    def enumerate(
        self,
        counts: Sequence[int],
        block: Sequence[int],
        *,
        deadline: float | None = None,
        cap: int = 256,
        maximum_grid: int = 300_000,
    ) -> list[tuple[int, ...]] | None:
        """Enumerate all counts conditional on the identities outside block."""
        if deadline is not None and time.monotonic() >= deadline:
            return None
        if self.infeasible:
            return []
        selected = tuple(sorted(set(map(int, block))))
        if not selected or len(selected) > 3:
            raise ValueError("block must contain one to three identities")
        if selected[0] < 0 or selected[-1] >= self.dimension:
            raise ValueError("block index outside catalog")
        if len(counts) != self.dimension:
            raise ValueError("count dimension mismatch")
        fixed = np.asarray(counts, dtype=np.int64).copy()
        fixed[list(selected)] = 0
        constant = self.matrix @ fixed
        local_matrix = self.matrix[:, selected]

        # Most rows constrain identities outside this tiny block.  Check their
        # fixed statistic once; do not allocate a grid-sized zero vector for
        # every outside identity's domain on every proposal.
        active = []
        for row, domain in enumerate(self.domains):
            if not domain:
                return []
            coefficients = tuple(map(int, local_matrix[row]))
            if any(coefficients):
                active.append((row, coefficients, domain))
            elif not any(low <= constant[row] <= high for low, high in zip(domain[::2], domain[1::2])):
                return []

        # Exact integer interval propagation includes the remaining total-N
        # capacity.  Domain holes are retained for the later full checks.
        lower = [self.bounds[index][0] for index in selected]
        upper = [self.bounds[index][1] for index in selected]

        def extremes(coefficients):
            minimum = sum(a * (low if a >= 0 else high) for a, low, high in zip(coefficients, lower, upper))
            maximum = sum(a * (high if a >= 0 else low) for a, low, high in zip(coefficients, lower, upper))
            return minimum, maximum

        for _ in range(4):
            changed = False
            for row, coefficients, domain in active:
                minimum, maximum = extremes(coefficients)
                low_stat, high_stat = int(domain[0]) - int(constant[row]), int(domain[-1]) - int(constant[row])
                if minimum > high_stat or maximum < low_stat:
                    return []
                for column, coefficient in enumerate(coefficients):
                    if not coefficient:
                        continue
                    own_min = coefficient * (lower[column] if coefficient > 0 else upper[column])
                    own_max = coefficient * (upper[column] if coefficient > 0 else lower[column])
                    required_low = low_stat - (maximum - own_max)
                    required_high = high_stat - (minimum - own_min)
                    if coefficient > 0:
                        new_lower = -((-required_low) // coefficient)
                        new_upper = required_high // coefficient
                    else:
                        new_lower = -((-required_high) // coefficient)
                        new_upper = required_low // coefficient
                    new_lower, new_upper = max(lower[column], new_lower), min(upper[column], new_upper)
                    if new_lower > new_upper:
                        return []
                    changed |= new_lower != lower[column] or new_upper != upper[column]
                    lower[column], upper[column] = new_lower, new_upper
            if not changed:
                break

        # An equality removes one Cartesian dimension.  Otherwise the grid is
        # visited in bounded chunks so the 25 ms proposal deadline is checked
        # throughout, including the large no-pivot case.
        pivot = None
        free = tuple(range(len(selected)))
        sizes = [high - low + 1 for low, high in zip(lower, upper)]
        best_grid = math.prod(sizes)
        for row, observed in self.equalities:
            for column, coefficient in enumerate(local_matrix[row]):
                grid_size = math.prod(size for i, size in enumerate(sizes) if i != column)
                if coefficient and grid_size < best_grid:
                    best_grid = grid_size
                    pivot = row, column, int(coefficient), int(observed)
        if pivot is not None:
            free = tuple(index for index in free if index != pivot[1])
        if best_grid > maximum_grid:
            return None
        checks = []
        for row, coefficients, domain in active:
            minimum, maximum = extremes(coefficients)
            minimum, maximum = minimum + int(constant[row]), maximum + int(constant[row])
            if not any(low <= minimum and maximum <= high for low, high in zip(domain[::2], domain[1::2])):
                checks.append((row, domain))
        table_checks = []
        for row, allowed in self.tables:
            coefficients, offset = row[list(selected)], int(row @ fixed)
            if not len(allowed):
                return []
            if not np.any(coefficients):
                if offset not in allowed:
                    return []
            else:
                table_checks.append((coefficients, offset, allowed))
        output = []
        shape = tuple(sizes[index] for index in free)
        for start in range(0, best_grid, 1024):
            if deadline is not None and time.monotonic() >= deadline:
                return None
            grid = (
                np.asarray(np.unravel_index(np.arange(start, min(start + 1024, best_grid)), shape), dtype=np.int64).T
                if free else np.empty((1, 0), dtype=np.int64)
            )
            if free:
                grid += np.asarray([lower[index] for index in free])
            candidates = np.empty((len(grid), len(selected)), dtype=np.int64)
            candidates[:, free] = grid
            if pivot is not None:
                row, column, coefficient, observed = pivot
                residual = observed - int(constant[row]) - grid @ local_matrix[row, free]
                candidates[:, column] = residual // coefficient
                candidates = candidates[
                    (residual % coefficient == 0)
                    & (candidates[:, column] >= lower[column])
                    & (candidates[:, column] <= upper[column])
                ]
            for row, domain in checks:
                if not len(candidates):
                    break
                totals = candidates @ local_matrix[row] + constant[row]
                valid = np.zeros(len(candidates), dtype=bool)
                for low, high in zip(domain[::2], domain[1::2]):
                    valid |= (totals >= low) & (totals <= high)
                candidates = candidates[valid]
            for coefficients, offset, allowed in table_checks:
                if not len(candidates):
                    break
                candidates = candidates[np.isin(candidates @ coefficients + offset, allowed)]
            if len(output) + len(candidates) > cap or (deadline is not None and time.monotonic() >= deadline):
                return None
            for local in candidates:
                candidate = list(map(int, counts))
                for index, count in zip(selected, local):
                    candidate[index] = int(count)
                output.append(tuple(candidate))
        return output


def _normalized_equalities(enumerator):
    normalized_rows = set()
    for row_index, _ in enumerator.equalities:
        row = tuple(map(int, enumerator.matrix[row_index]))
        divisor = math.gcd(*row)
        if divisor:
            row = tuple(value // divisor for value in row)
            if next(value for value in row if value) < 0:
                row = tuple(-value for value in row)
            normalized_rows.add(row)
    return tuple(sorted(normalized_rows))


class CountSignatureIndex:
    """Complete identity-multiset classes for fixed positions in a sequence.

    A lookup returns every size-k multiset within the supplied static catalog
    group with the same exact linear statistics as the old k tokens.  Classes
    are cached by group/k and depend only on fixed numeric constraints.  No
    current counts, feasibility filtering or posterior weights enter the cache.

    A sampler can choose a class member uniformly and uniformly permute it into
    the selected positions.  Its ordered-state MH log ratio includes
    ``log O(new_block) - log O(old_block)``, where ``O = k! / product(n_i!)``.
    Nonnegativity is automatic for replacement of existing tokens.  The caller
    must still check all inequalities, prior support, and layout evidence.
    """

    def __init__(self, enumerator: ExactCountBlockEnumerator):
        self.dimension = enumerator.dimension
        self.rows = _normalized_equalities(enumerator)
        self._cache = {}

    def lookup(
        self, group_indices: Sequence[int], old_tokens: Sequence[int]
    ) -> tuple[tuple[int, ...], ...]:
        group = tuple(sorted(set(map(int, group_indices))))
        old = tuple(sorted(map(int, old_tokens)))
        if not group or group[0] < 0 or group[-1] >= self.dimension:
            raise ValueError("group must contain catalog indices")
        if not 1 <= len(old) <= 3 or any(index not in group for index in old):
            raise ValueError("old block must contain one to three group identities")
        key = group, len(old)
        if key not in self._cache:
            # A constant row contributes k*a to every member of a fixed-k
            # group, so removing it changes no equivalence class.
            active = tuple(
                row for row in self.rows if len({row[index] for index in group}) > 1
            )
            columns = {
                index: tuple(row[index] for row in active) for index in group
            }

            def signature(tokens):
                return tuple(
                    sum(columns[index][feature] for index in tokens)
                    for feature in range(len(active))
                )

            classes = {}
            for multiset in combinations_with_replacement(group, len(old)):
                classes.setdefault(signature(multiset), []).append(multiset)
            self._cache[key] = (
                columns,
                {signature: tuple(members) for signature, members in classes.items()},
            )
        columns, classes = self._cache[key]
        signature = tuple(
            sum(columns[index][feature] for index in old)
            for feature in range(len(next(iter(columns.values()))))
        )
        return classes[signature]


def build_symmetric_count_moves(
    enumerator: ExactCountBlockEnumerator,
    *,
    groups: Sequence[Sequence[int]] = (),
    maximum_moves: int = 512,
    maximum_tokens: int = 3,
    maximum_delta: int = 2,
    deadline: float | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Find a static library of short integer reactions preserving equalities.

    Two multisets with the same vector of exact linear statistics have a
    difference ``d`` with ``A @ d == 0``.  A sampler may choose a library entry
    and a random sign independently of its current state.  Such a proposal is
    symmetric; it still needs the ordinary target MH ratio and checks of every
    inequality, allowed total, nonnegativity and spatial evidence.  This helper
    assigns no probability mass and does not claim its moves connect all modes.

    Single/pair/triple token multisets are matched within the supplied static
    catalog groups; a global single/pair pass also crosses group boundaries.
    Different multiset lengths can match when count is not an exact statistic,
    permitting changes in total item count.  No current world enters the API.
    """
    if maximum_tokens not in (1, 2, 3) or maximum_delta < 1:
        raise ValueError("expected at most three tokens and positive delta bound")
    if maximum_moves < 1 or enumerator.infeasible:
        return ()
    dimension = enumerator.dimension
    rows = _normalized_equalities(enumerator)
    columns = tuple(tuple(row[index] for row in rows) for index in range(dimension))
    pools = []
    seen_pools = set()
    for group in groups:
        group = tuple(sorted(set(map(int, group))))
        if group and (group[0] < 0 or group[-1] >= dimension):
            raise ValueError("group index outside catalog")
        if group and group not in seen_pools:
            pools.append((group, maximum_tokens))
            seen_pools.add(group)
    entire = tuple(range(dimension))
    if entire not in seen_pools:
        pools.append((entire, min(maximum_tokens, 2)))
    pools.sort(
        key=lambda pool: (
            -sum(any(row[index] for index in pool[0]) for row in rows),
            len(pool[0]),
        )
    )
    per_pool = []
    seen_moves = set()
    for group, tokens in pools:
        representatives = {}
        moves = []
        for length in range(1, tokens + 1):
            for identities in combinations_with_replacement(group, length):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                signature = tuple(
                    sum(columns[index][feature] for index in identities)
                    for feature in range(len(rows))
                )
                prior = representatives.setdefault(signature, [])
                for other in prior:
                    difference = Counter(identities)
                    difference.subtract(other)
                    reaction = tuple(difference[index] for index in range(dimension))
                    if not any(reaction) or max(map(abs, reaction)) > maximum_delta:
                        continue
                    if next(value for value in reaction if value) < 0:
                        reaction = tuple(-value for value in reaction)
                    if reaction not in seen_moves:
                        seen_moves.add(reaction)
                        moves.append(reaction)
                # A bounded representative set keeps uninformative zero-key
                # groups from causing quadratic enumeration. This truncation
                # affects available proposals only, never posterior weights.
                if len(prior) < 4:
                    prior.append(identities)
                if len(moves) >= maximum_moves:
                    break
            if len(moves) >= maximum_moves or (
                deadline is not None and time.monotonic() >= deadline
            ):
                break
        per_pool.append(moves)
        if deadline is not None and time.monotonic() >= deadline:
            break
    # Share capacity across groups so an unconstrained quality cannot consume
    # every proposal slot before a value-constrained quality is considered.
    output = []
    for index in range(max(map(len, per_pool), default=0)):
        for moves in per_pool:
            if index < len(moves):
                output.append(moves[index])
                if len(output) >= maximum_moves:
                    return tuple(output)
    return tuple(output)
