"""Tests for the explicit ploidy-2/3 entry point: exact agreement with an
independent brute force, hand-computed normal/tie/missing-data cases,
boundaries, the input-error contract, and the analytic relation to the
diploid entry (which is left fully intact).
"""

from __future__ import annotations

import math
import unittest
from math import comb

from haplophase import (
    MAX_PLOIDY_LOCI,
    MAX_PLOIDY_OBSERVATIONS,
    InvalidPhasingInputError,
    Observation,
    phase_haplotypes,
    phase_polyploid,
)

from reference_polyploid import (
    brute_force_polyploid,
    random_polyploid_cases,
    reference_polyploid_assignment,
)


def to_obs(raw_reads):
    return [Observation(alleles=a, flip_cost=w) for a, w in raw_reads]


class ExactAgreementWithIndependentBruteForce(unittest.TestCase):
    def test_random_cases_match_reference(self):
        for loci, ploidy, raw in random_polyploid_cases():
            with self.subTest(loci=loci, ploidy=ploidy, raw=raw):
                result = phase_polyploid(loci, to_obs(raw), ploidy=ploidy)
                best, optima, second = brute_force_polyploid(loci, raw, ploidy)
                self.assertEqual(result.optimal_cost, best)
                self.assertEqual(result.optimal_solutions, optima)
                self.assertEqual(result.second_best_cost, second)
                self.assertEqual(result.gap, None if second is None else second - best)
                # canonical solution is the lexicographically smallest optimum
                self.assertEqual(result.haplotypes, optima[0])
                # canonical tie order: sorted copies inside sorted solutions
                self.assertEqual(list(result.optimal_solutions),
                                 sorted(result.optimal_solutions))
                for solution in result.optimal_solutions:
                    self.assertEqual(list(solution), sorted(solution))
                    self.assertEqual(len(solution), ploidy)
                # per-read reporting matches the independent formula
                for i, (alleles, w) in enumerate(raw):
                    a = result.assignments[i]
                    exp = reference_polyploid_assignment(
                        result.haplotypes, alleles, w
                    )
                    self.assertEqual(
                        (a.copy, a.tied, a.mismatches, a.costs, a.flips), exp
                    )
                    self.assertEqual(a.cost, w * a.mismatches[a.copy])
                # charged per-read costs sum to the reported optimum
                self.assertEqual(sum(a.cost for a in result.assignments), best)
                # enumeration visited exactly the multiset count
                self.assertEqual(
                    result.search.candidates_evaluated,
                    comb((1 << loci) + ploidy - 1, ploidy),
                )
                self.assertTrue(result.search.completed)
                self.assertEqual(result.search.ploidy, ploidy)

    def test_result_is_deterministic(self):
        loci, ploidy, raw = random_polyploid_cases(seed=11, count=1)[0]
        r1 = phase_polyploid(loci, to_obs(raw), ploidy=ploidy)
        r2 = phase_polyploid(loci, to_obs(raw), ploidy=ploidy)
        self.assertEqual(r1, r2)


class HandComputedCases(unittest.TestCase):
    def test_triploid_three_distinct_copies_unique_zero(self):
        # Each read is an exact copy string; with three distinct reads and
        # ploidy 3 the only zero-cost multiset is {000, 011, 101}.
        raw = [((0, 0, 0), 2), ((0, 1, 1), 1), ((1, 0, 1), 1)]
        result = phase_polyploid(3, to_obs(raw), ploidy=3)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(
            result.haplotypes, ((0, 0, 0), (0, 1, 1), (1, 0, 1))
        )
        self.assertEqual(result.optimal_solutions, (result.haplotypes,))
        self.assertEqual([a.copy for a in result.assignments], [0, 1, 2])
        self.assertEqual([a.tied for a in result.assignments], [False, False, False])
        self.assertEqual([a.cost for a in result.assignments], [0, 0, 0])

    def test_tied_optima_are_canonically_sorted(self):
        # One read (0,) on a single locus, ploidy 2: both {0,0} and {0,1}
        # cost 0; {1,1} costs 1.  Canonical order lists {0,0} first.
        result = phase_polyploid(1, [Observation((0,), 1)], ploidy=2)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(
            result.optimal_solutions, (((0,), (0,)), ((0,), (1,)))
        )
        self.assertEqual(result.haplotypes, ((0,), (0,)))
        self.assertEqual(result.second_best_cost, 1)
        self.assertEqual(result.gap, 1)
        a = result.assignments[0]
        self.assertTrue(a.tied)
        self.assertEqual(a.copy, 0)
        self.assertEqual(a.costs, (0, 0))

    def test_missing_sites_produce_five_hand_derived_optima(self):
        # Reads (0, -) w3 and (-, 1) w1 on 2 loci, ploidy 2.  Zero cost
        # needs one copy with locus0 = 0 and one with locus1 = 1; the five
        # qualifying multisets are exactly the ones below (hand-derived).
        raw = [((0, None), 3), ((None, 1), 1)]
        result = phase_polyploid(2, to_obs(raw), ploidy=2)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(
            result.optimal_solutions,
            (
                ((0, 0), (0, 1)),
                ((0, 0), (1, 1)),
                ((0, 1), (0, 1)),
                ((0, 1), (1, 0)),
                ((0, 1), (1, 1)),
            ),
        )
        self.assertEqual(result.haplotypes, ((0, 0), (0, 1)))
        self.assertEqual(result.second_best_cost, 1)
        self.assertEqual(result.gap, 1)
        first, second = result.assignments
        # read 0 sees only locus 0 = 0: both canonical copies start with 0
        self.assertTrue(first.tied)
        self.assertEqual((first.copy, first.cost, first.flips), (0, 0, ()))
        # read 1 sees only locus 1 = 1: only copy 1 (the (0,1)) matches
        self.assertFalse(second.tied)
        self.assertEqual((second.copy, second.cost, second.flips), (1, 0, ()))
        self.assertEqual(second.costs, (1, 0))

    def test_weights_decide_between_competing_copies(self):
        # Heavy read (0,0) w10 vs light read (1,1) w1: only {00, 11} is
        # free; the next best multisets ({00,01} and {00,10}) cost 1.
        raw = [((0, 0), 10), ((1, 1), 1)]
        result = phase_polyploid(2, to_obs(raw), ploidy=2)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(result.optimal_solutions, (((0, 0), (1, 1)),))
        self.assertEqual(result.second_best_cost, 1)
        self.assertEqual(result.gap, 1)
        self.assertEqual([a.copy for a in result.assignments], [0, 1])

    def test_triploid_tie_and_second_best(self):
        # One read (1,) w2, ploidy 3: every multiset with at least one
        # copy 1 is free; only {0,0,0} pays (2), so it is the second best.
        result = phase_polyploid(1, [Observation((1,), 2)], ploidy=3)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(
            result.optimal_solutions,
            (((0,), (0,), (1,)), ((0,), (1,), (1,)), ((1,), (1,), (1,))),
        )
        self.assertEqual(result.second_best_cost, 2)
        self.assertEqual(result.gap, 2)
        a = result.assignments[0]
        self.assertEqual((a.copy, a.tied, a.costs), (2, False, (2, 2, 0)))

    def test_all_missing_read_is_neutral_and_tied(self):
        # An all-missing read charges nothing; every multiset containing
        # (0,0) is optimal: 10 of the 20 size-3 multisets over 4 strings.
        raw = [((None, None), 7), ((0, 0), 1)]
        result = phase_polyploid(2, to_obs(raw), ploidy=3)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(len(result.optimal_solutions), 10)
        for solution in result.optimal_solutions:
            self.assertIn((0, 0), solution)
        self.assertEqual(result.haplotypes, ((0, 0), (0, 0), (0, 0)))
        miss = result.assignments[0]
        self.assertTrue(miss.tied)
        self.assertEqual((miss.copy, miss.cost, miss.flips), (0, 0, ()))
        self.assertEqual(miss.mismatches, (0, 0, 0))


class BoundarySizes(unittest.TestCase):
    def test_max_loci_max_reads_planted_triploid(self):
        copies = ((0, 0, 0, 0, 0, 0), (0, 1, 1, 0, 1, 1), (1, 1, 0, 1, 0, 1))
        planted = tuple(sorted(copies))
        raw = [
            ((None, 0, 0, 0, 0, 0), 1),          # copy A, locus 0 missing
            ((0, 0, 0, 0, 0, 0), 2),             # copy A
            ((0, 1, 1, None, 1, 1), 1),          # copy B, locus 3 missing
            ((0, 1, 1, 0, 1, 1), 3),             # copy B
            ((1, 1, 0, 1, 0, None), 1),          # copy C, locus 5 missing
            ((1, 1, 0, 1, 0, 1), 2),             # copy C
            ((None, None, None, None, None, None), 4),  # all missing
            ((0, 0, None, 0, None, 0), 1),       # copy A, two sites missing
        ]
        self.assertEqual(len(raw), MAX_PLOIDY_OBSERVATIONS)
        result = phase_polyploid(MAX_PLOIDY_LOCI, to_obs(raw), ploidy=3)
        self.assertEqual(result.optimal_cost, 0)
        self.assertIn(planted, result.optimal_solutions)
        self.assertEqual(result.search.candidates_evaluated, comb(64 + 2, 3))
        self.assertEqual(len(result.assignments), MAX_PLOIDY_OBSERVATIONS)
        self.assertEqual(sum(a.cost for a in result.assignments), 0)

    def test_no_reads_every_candidate_optimal(self):
        result = phase_polyploid(MAX_PLOIDY_LOCI, [], ploidy=2)
        self.assertEqual(result.optimal_cost, 0)
        self.assertIsNone(result.second_best_cost)
        self.assertIsNone(result.gap)
        self.assertEqual(len(result.optimal_solutions), comb(64 + 1, 2))
        self.assertEqual(result.haplotypes, ((0,) * 6, (0,) * 6))
        self.assertEqual(result.assignments, ())

    def test_single_locus_ploidy3(self):
        result = phase_polyploid(1, [Observation((0,), 1)], ploidy=3)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(result.haplotypes, ((0,), (0,), (0,)))
        self.assertEqual(result.second_best_cost, 1)


class DiploidRelation(unittest.TestCase):
    def test_ploidy2_never_costs_more_than_diploid_entry(self):
        # The diploid entry restricts the pair to complements {h, 1-h};
        # the ploidy-2 entry optimises over all multisets, a superset.
        for loci, ploidy, raw in random_polyploid_cases(seed=5, count=40):
            if ploidy != 2:
                continue
            with self.subTest(loci=loci, raw=raw):
                poly = phase_polyploid(loci, to_obs(raw), ploidy=2)
                diploid = phase_haplotypes(loci, to_obs(raw))
                self.assertLessEqual(poly.optimal_cost, diploid.optimal_cost)

    def test_ploidy2_can_beat_complementary_pair(self):
        # Reads (0,0) and (0,1): the diploid entry must keep the copies
        # complementary and pays 1 either way; the ploidy-2 entry may use
        # {00, 01} and pays 0.
        raw = [((0, 0), 1), ((0, 1), 1)]
        poly = phase_polyploid(2, to_obs(raw), ploidy=2)
        diploid = phase_haplotypes(2, to_obs(raw))
        self.assertEqual(poly.optimal_cost, 0)
        self.assertEqual(diploid.optimal_cost, 1)
        self.assertEqual(poly.haplotypes, ((0, 0), (0, 1)))


class InputValidation(unittest.TestCase):
    def test_ploidy_contract(self):
        for bad in (0, 1, 4, -2, True, False, 2.0, "2", None):
            with self.subTest(bad=bad):
                try:
                    phase_polyploid(2, [], ploidy=bad)
                except InvalidPhasingInputError as err:
                    self.assertIn("ploidy", str(err))
                else:
                    self.fail("expected InvalidPhasingInputError")

    def test_loci_contract_stricter_than_diploid(self):
        for bad in (0, -1, MAX_PLOIDY_LOCI + 1, 18, True, 1.5, "3", None):
            with self.subTest(bad=bad):
                try:
                    phase_polyploid(bad, [], ploidy=2)
                except InvalidPhasingInputError as err:
                    self.assertIn("loci", str(err))
                else:
                    self.fail("expected InvalidPhasingInputError")

    def test_observation_count_limit(self):
        obs = [Observation((0, 0), 1)] * (MAX_PLOIDY_OBSERVATIONS + 1)
        with self.assertRaises(InvalidPhasingInputError) as ctx:
            phase_polyploid(2, obs, ploidy=2)
        self.assertIn("observations", str(ctx.exception))
        # exactly at the limit is accepted
        ok = [Observation((0, 0), 1)] * MAX_PLOIDY_OBSERVATIONS
        self.assertEqual(phase_polyploid(2, ok, ploidy=2).optimal_cost, 0)

    def test_read_errors_are_locatable(self):
        cases = [
            [Observation((2, 0), 1)],
            [Observation((0, True), 1)],
            [Observation((0, math.nan), 1)],
            [Observation((math.inf, 0), 1)],
            [Observation((0, "0"), 1)],
        ]
        for obs in cases:
            with self.subTest(obs=obs):
                try:
                    phase_polyploid(2, obs, ploidy=2)
                except InvalidPhasingInputError as err:
                    self.assertIn("observations[0].alleles", str(err))
                else:
                    self.fail("expected InvalidPhasingInputError")
        for bad_cost in (0, -3, 1.0, True, math.nan, -math.inf, None, "1"):
            with self.subTest(bad_cost=bad_cost):
                try:
                    phase_polyploid(2, [Observation((0, 0), bad_cost)], ploidy=3)
                except InvalidPhasingInputError as err:
                    self.assertIn("observations[0].flip_cost", str(err))
                else:
                    self.fail("expected InvalidPhasingInputError")

    def test_container_and_length_errors(self):
        with self.assertRaises(InvalidPhasingInputError):
            phase_polyploid(2, "not-a-list", ploidy=2)  # type: ignore[arg-type]
        with self.assertRaises(InvalidPhasingInputError):
            phase_polyploid(2, [(0, 0)], ploidy=2)  # type: ignore[list-item]
        with self.assertRaises(InvalidPhasingInputError) as ctx:
            phase_polyploid(3, [Observation((0, 0), 1)], ploidy=2)
        self.assertIn("length 2", str(ctx.exception))

    def test_failed_call_mutates_nothing(self):
        data = [Observation((0, 0), 1), Observation((0, 1), 1)]
        snapshot = list(data)
        with self.assertRaises(ValueError):  # InvalidPhasingInputError is one
            phase_polyploid(2, data, ploidy=4)
        self.assertEqual(data, snapshot)


if __name__ == "__main__":
    unittest.main()
