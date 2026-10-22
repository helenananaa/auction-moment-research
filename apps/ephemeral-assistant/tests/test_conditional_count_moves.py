from __future__ import annotations

import itertools
import math
import random
import time
import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np
from ortools.sat.python import cp_model

from auction_moment_assistant.conditional_constraints import compile_hard_constraints
from auction_moment_assistant.conditional_count_moves import (
    ExactCountBlockEnumerator,
    CountSignatureIndex,
    build_symmetric_count_moves,
)
from auction_moment_assistant.models import CatalogItem
from auction_moment_assistant.observations import ObservationSnapshot


CATALOG = (
    CatalogItem("a", "a", "白", 1, 1, 10),
    CatalogItem("b", "b", "蓝", 1, 1, 30),
    CatalogItem("c", "c", "蓝", 2, 1, 70),
    CatalogItem("d", "d", "金", 2, 2, 120),
)
PRIOR = SimpleNamespace(
    min_count=2, max_count=5, _count_probabilities={2: 0.5, 4: 0.5}, minimum_quotas={"白": 1}
)


def snapshot(*semantics):
    return ObservationSnapshot(
        revision=1,
        round_number=1,
        bankroll=None,
        events=tuple(
            {"round_number": 1, "semantics": {"parsed": True, **event}}
            for event in semantics
        ),
        map_items=(),
        completeness_confirmed=True,
        completeness_source="controlled",
        map_rows=None,
        map_height_exact=False,
        map_rows_source="unknown",
        pre_bid_confirmed=True,
        pre_bid_source="controlled",
    )


class Solutions(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        super().__init__()
        self.variables = variables
        self.solutions = set()

    def on_solution_callback(self):
        self.solutions.add(tuple(self.value(variable) for variable in self.variables))


def oracle(base, variables, counts, block):
    model = base.clone()
    local = [model.get_int_var_from_proto_index(variable.index) for variable in variables]
    for index, variable in enumerate(local):
        if index not in block:
            model.add(variable == counts[index])
    callback = Solutions(local)
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.solve(model, callback)
    return callback.solutions


class ExactCountBlockTests(unittest.TestCase):
    def test_matches_cp_solver_for_all_small_blocks_and_numeric_forms(self):
        evidence = (
            (),
            ({"effect": "total_item_count", "observed_count": 4},),
            ({"effect": "quality_total_value", "quality": "蓝", "observed_value": 100},),
            ({"effect": "quality_average_value", "quality": "蓝", "observed_value": 50},),
            ({"effect": "total_occupied_cells", "observed_cells": 5},),
            ({"effect": "average_item_area", "observed_average_cells": 1.5},),
            ({"effect": "max_item_value", "observed_value": 70},),
            ({"effect": "highest_quality", "observed_quality": "蓝"},),
            (
                {"effect": "total_item_count", "observed_count": 4},
                {"effect": "quality_total_value", "quality": "蓝", "observed_value": 100},
            ),
        )
        rng = random.Random(9)
        for semantics in evidence:
            base, variables, notes = compile_hard_constraints(snapshot(*semantics), CATALOG, PRIOR)
            self.assertFalse(notes)
            compiled = ExactCountBlockEnumerator(base, variables)
            for size in (1, 2, 3):
                for block in itertools.combinations(range(len(CATALOG)), size):
                    counts = tuple(rng.randrange(3) for _ in CATALOG)
                    with self.subTest(semantics=semantics, counts=counts, block=block):
                        self.assertEqual(
                            set(compiled.enumerate(counts, block)),
                            oracle(base, variables, counts, block),
                        )

    def test_deadline_cap_and_grid_never_return_partial_support(self):
        base, variables, _ = compile_hard_constraints(snapshot(), CATALOG, PRIOR)
        compiled = ExactCountBlockEnumerator(base, variables)
        support = compiled.enumerate((1, 1, 0, 0), (0, 1), cap=100)
        self.assertGreater(len(support), 1)
        self.assertIsNone(compiled.enumerate((1, 1, 0, 0), (0, 1), cap=len(support) - 1))
        self.assertIsNone(compiled.enumerate((1, 1, 0, 0), (0, 1), deadline=time.monotonic() - 1))
        self.assertIsNone(compiled.enumerate((1, 1, 0, 0), (0, 1), maximum_grid=1))

    def test_negative_pivot_and_disjoint_integer_domain(self):
        base, variables, _ = compile_hard_constraints(snapshot(), CATALOG, PRIOR)
        base.add(-3 * variables[0] + 2 * variables[1] - variables[2] == -1)
        base.add_linear_expression_in_domain(variables[1], cp_model.Domain.from_values([0, 2, 4]))
        compiled = ExactCountBlockEnumerator(base, variables)
        actual = compiled.enumerate((1, 0, 0, 0), (0, 1, 2))
        self.assertEqual(set(actual), oracle(base, variables, (1, 0, 0, 0), (0, 1, 2)))

    def test_explicit_infeasible_and_unsupported_features_fail_closed(self):
        base, variables, _ = compile_hard_constraints(snapshot(), CATALOG, PRIOR)
        base.add(False)
        self.assertEqual(ExactCountBlockEnumerator(base, variables).enumerate((1, 1, 0, 0), (0,)), [])
        other, variables, _ = compile_hard_constraints(snapshot(), CATALOG, PRIOR)
        other.add_all_different(variables)
        with self.assertRaisesRegex(ValueError, "unsupported numeric constraint"):
            ExactCountBlockEnumerator(other, variables)

    def test_highest_quality_empty_higher_set_accepts_fixed_true_auxiliary(self):
        base, variables, notes = compile_hard_constraints(
            snapshot({"effect": "highest_quality", "observed_quality": "金"}),
            CATALOG,
            PRIOR,
        )
        self.assertFalse(notes)
        # sum(empty higher-quality set)==0 creates CP-SAT's fixed True var.
        self.assertGreater(len(base.proto.variables), len(variables) + 1)
        compiled = ExactCountBlockEnumerator(base, variables)
        actual = compiled.enumerate((1, 0, 0, 1), (0, 1, 3))
        self.assertEqual(set(actual), oracle(base, variables, (1, 0, 0, 1), (0, 1, 3)))

    def test_fixed_constants_are_eliminated_from_linear_and_table_expressions(self):
        base, variables, _ = compile_hard_constraints(snapshot(), CATALOG, PRIOR)
        seven = base.new_constant(7)
        base.add(variables[0] + 2 * seven <= 16)
        base.add_allowed_assignments([variables[1] + seven], [(8,), (9,)])
        compiled = ExactCountBlockEnumerator(base, variables)
        actual = compiled.enumerate((1, 1, 0, 0), (0, 1, 2))
        self.assertEqual(set(actual), oracle(base, variables, (1, 1, 0, 0), (0, 1, 2)))
        self.assertTrue(actual)

    def test_fixed_false_and_negative_literals_are_evaluated_exactly(self):
        for constant, negate, feasible in ((0, False, False), (1, True, False), (0, True, True)):
            base, variables, _ = compile_hard_constraints(snapshot(), CATALOG, PRIOR)
            fixed = base.new_constant(constant)
            base.add_bool_or([fixed.Not() if negate else fixed])
            compiled = ExactCountBlockEnumerator(base, variables)
            actual = compiled.enumerate((1, 1, 0, 0), (0, 1))
            self.assertEqual(bool(actual), feasible)
            self.assertEqual(set(actual), oracle(base, variables, (1, 1, 0, 0), (0, 1)))

    def test_nonconstant_auxiliary_is_never_silently_eliminated(self):
        base, variables, _ = compile_hard_constraints(snapshot(), CATALOG, PRIOR)
        base.new_bool_var("unsupported")
        with self.assertRaisesRegex(ValueError, "nonconstant auxiliary"):
            ExactCountBlockEnumerator(base, variables)

    def test_remaining_total_capacity_shrinks_grid_before_enumeration(self):
        prior = SimpleNamespace(min_count=0, max_count=10, minimum_quotas={})
        base, variables, _ = compile_hard_constraints(snapshot(), CATALOG, prior)
        compiled = ExactCountBlockEnumerator(base, variables)
        # Outside count=7 leaves each selected variable <=3.  The original
        # 11**2 grid exceeds this limit; the propagated 4**2 grid fits.
        support = compiled.enumerate((0, 0, 3, 4), (0, 1), maximum_grid=16)
        self.assertIsNotNone(support)
        self.assertEqual(len(support), 10)
        self.assertEqual(set(support), oracle(base, variables, (0, 0, 3, 4), (0, 1)))

    def test_no_pivot_large_grid_checks_deadline_within_chunks(self):
        catalog = tuple(
            CatalogItem(str(i), str(i), "蓝", 1, 1, i + 1) for i in range(120)
        )
        base, variables, _ = compile_hard_constraints(
            snapshot(), catalog, SimpleNamespace(min_count=0, max_count=60, minimum_quotas={})
        )
        compiled = ExactCountBlockEnumerator(base, variables)
        self.assertFalse(any(np.any(compiled.matrix[row]) for row, _ in compiled.equalities))
        started = time.monotonic()
        result = compiled.enumerate(
            (0,) * len(catalog), (0, 1, 2), cap=1_000_000,
            deadline=started + 0.025,
        )
        elapsed = time.monotonic() - started
        # Large tolerance allows CI scheduling noise while catching the former
        # >0.5s monolithic 61**3 grid/120-row blocking operation.
        self.assertLess(elapsed, 0.2)
        self.assertIsNone(result)

    def test_adapter_is_faster_than_repeated_cp_support_solves(self):
        # This is a shape/work-count assertion, not a timing-dependent gate.
        # A 60-count bound plus an equality enumerates 61**2, not 61**3.
        prior = SimpleNamespace(
            min_count=10, max_count=60, _count_probabilities={n: 1 for n in range(10, 61)}, minimum_quotas={}
        )
        base, variables, _ = compile_hard_constraints(snapshot({"effect": "total_item_count", "observed_count": 25}), CATALOG, prior)
        compiled = ExactCountBlockEnumerator(base, variables)
        support = compiled.enumerate((5, 5, 5, 10), (0, 1, 2), cap=1000, maximum_grid=61**2)
        self.assertIsNotNone(support)
        self.assertEqual(len(support), math.comb(17, 2))


class SymmetricReactionTests(unittest.TestCase):
    def test_reactions_preserve_every_exact_statistic_and_allow_four_species(self):
        # Distinct 2-token sums 10+120=30+100 need all four identities.
        catalog = tuple(
            CatalogItem(str(index), str(index), "蓝", 1, 1, value)
            for index, value in enumerate((10, 30, 100, 120))
        )
        base, variables, _ = compile_hard_constraints(
            snapshot(
                {"effect": "total_item_count", "observed_count": 4},
                {"effect": "quality_total_value", "quality": "蓝", "observed_value": 260},
            ),
            catalog,
            SimpleNamespace(min_count=4, max_count=4, minimum_quotas={}),
        )
        compiled = ExactCountBlockEnumerator(base, variables)
        moves = build_symmetric_count_moves(compiled, groups=((0, 1, 2, 3),))
        self.assertIn((1, -1, -1, 1), moves)
        for move in moves:
            for index, _ in compiled.equalities:
                self.assertEqual(sum(int(a) * b for a, b in zip(compiled.matrix[index], move)), 0)

    def test_reactions_can_change_count_without_a_count_equality(self):
        catalog = tuple(
            CatalogItem(str(index), str(index), "蓝", 1, 1, value)
            for index, value in enumerate((10, 20, 30))
        )
        base, variables, _ = compile_hard_constraints(
            snapshot({"effect": "quality_total_value", "quality": "蓝", "observed_value": 60}),
            catalog,
            SimpleNamespace(min_count=1, max_count=6, minimum_quotas={}),
        )
        moves = build_symmetric_count_moves(ExactCountBlockEnumerator(base, variables))
        self.assertTrue(any(sum(move) for move in moves))
        self.assertIn((1, 1, -1), moves)

    def test_move_library_is_static_and_capped(self):
        base, variables, _ = compile_hard_constraints(snapshot(), CATALOG, PRIOR)
        compiled = ExactCountBlockEnumerator(base, variables)
        first = build_symmetric_count_moves(compiled, maximum_moves=5)
        second = build_symmetric_count_moves(compiled, maximum_moves=5)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(build_symmetric_count_moves(compiled, deadline=time.monotonic() - 1), ())


class CountSignatureIndexTests(unittest.TestCase):
    def test_classes_are_complete_against_direct_integer_statistics(self):
        base, variables, _ = compile_hard_constraints(
            snapshot(
                {"effect": "total_item_count", "observed_count": 4},
                {"effect": "quality_total_value", "quality": "蓝", "observed_value": 100},
            ),
            CATALOG,
            PRIOR,
        )
        enumerator = ExactCountBlockEnumerator(base, variables)
        index = CountSignatureIndex(enumerator)
        for group in ((0, 1, 2, 3), (1, 2), (3, 1, 0)):
            for size in (1, 2, 3):
                all_multisets = tuple(itertools.combinations_with_replacement(sorted(group), size))

                def statistics(tokens):
                    return tuple(
                        sum(int(enumerator.matrix[row, token]) for token in tokens)
                        for row, _ in enumerator.equalities
                    )

                for old in all_multisets:
                    expected = tuple(candidate for candidate in all_multisets if statistics(candidate) == statistics(old))
                    actual = index.lookup(group, old)
                    self.assertEqual(actual, expected)
                    for candidate in actual:
                        self.assertIs(index.lookup(tuple(reversed(group)), candidate), actual)

    def test_uniform_multiset_proposal_has_correct_order_multiplicity_ratio(self):
        catalog = tuple(
            CatalogItem(str(i), str(i), "蓝", 1, 1, i) for i in range(3)
        )
        base, variables, _ = compile_hard_constraints(
            snapshot({"effect": "quality_total_value", "quality": "蓝", "observed_value": 2}),
            catalog,
            SimpleNamespace(min_count=2, max_count=2, minimum_quotas={}),
        )
        index = CountSignatureIndex(ExactCountBlockEnumerator(base, variables))
        candidates = index.lookup((0, 1, 2), (1, 1))
        self.assertEqual(candidates, ((0, 2), (1, 1)))
        sequences = ((0, 2), (1, 1), (2, 0))
        probabilities = (0.2, 0.5, 0.3)

        def target(sequence):
            return math.prod(probabilities[token] for token in sequence)

        def orders(sequence):
            return math.factorial(len(sequence)) / math.prod(math.factorial(n) for n in Counter(sequence).values())

        for old, new in itertools.permutations(sequences, 2):
            forward_q = 1 / len(candidates) / orders(new)
            backward_q = 1 / len(candidates) / orders(old)
            ratio = target(new) / target(old) * orders(new) / orders(old)
            self.assertAlmostEqual(
                target(old) * forward_q * min(1, ratio),
                target(new) * backward_q * min(1, 1 / ratio),
            )

    def test_fixed_k_uninformative_classes_include_every_multiset(self):
        base, variables, _ = compile_hard_constraints(snapshot(), CATALOG, PRIOR)
        index = CountSignatureIndex(ExactCountBlockEnumerator(base, variables))
        # The only exact relation is N=sum(counts), which eliminates to zero.
        group = (0, 1, 2, 3)
        actual = index.lookup(group, (0, 0, 0))
        self.assertEqual(actual, tuple(itertools.combinations_with_replacement(group, 3)))
        self.assertEqual(len(actual), math.comb(6, 3))
        with self.assertRaises(ValueError):
            index.lookup((0, 1), (2, 2))
        with self.assertRaises(ValueError):
            index.lookup(group, (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
