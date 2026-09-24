"""Tests for phase_polyploid: explicit copy number 2 or 3 with missing sites.

Expected values come from hand-computed examples and from the independent
full-product reference in tests/reference.py (which never imports the
package under test). A supplementary consistency check against the diploid
branch-and-bound entry is included but is not the source of expected values.
"""

from __future__ import annotations

import math
import unittest

from haplophase import (
    POLYPLOID_MAX_LOCI,
    POLYPLOID_MAX_OBSERVATIONS,
    POLYPLOID_PLOIDIES,
    InvalidPhasingInputError,
    Observation,
    phase_haplotypes,
    phase_polyploid,
)

from reference import (
    brute_force_polyploid,
    random_polyploid_cases,
    reference_copy_assignment,
    reference_cost_copies,
)


def to_obs(raw_reads):
    return [Observation(alleles=a, flip_cost=w) for a, w in raw_reads]


class HandComputedExamples(unittest.TestCase):
    def test_triploid_three_error_free_clusters(self):
        # reads 00 (w2), 11 (w3), 01 (w1): zero cost needs copies 00, 01, 11
        # exactly, so the optimum is unique. Cheapest deviation (e.g. copies
        # 00/11/11) misplaces the w1 read once -> second-best cost 1.
        raw = [((0, 0), 2), ((1, 1), 3), ((0, 1), 1)]
        result = phase_polyploid(2, to_obs(raw), ploidy=3)
        self.assertEqual(result.ploidy, 3)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(result.haplotypes, ((0, 0), (0, 1), (1, 1)))
        self.assertEqual(result.optimal_solutions, (((0, 0), (0, 1), (1, 1)),))
        self.assertEqual(result.second_best_cost, 1)
        self.assertEqual(result.gap, 1)
        self.assertEqual(
            [a.assigned for a in result.assignments], [0, 2, 1]
        )
        self.assertEqual([a.tied for a in result.assignments], [False, False, False])
        self.assertEqual(result.assignments[0].costs, (0, 2, 4))
        self.assertEqual(result.assignments[2].costs, (1, 0, 1))

    def test_triploid_single_locus_tie_and_canonical_order(self):
        # reads 0 and 1 (both w1): optima are {0,0,1} and {0,1,1}; the
        # canonical choice is the lexicographically smaller sorted tuple.
        raw = [((0,), 1), ((1,), 1)]
        result = phase_polyploid(1, to_obs(raw), ploidy=3)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(
            result.optimal_solutions,
            (((0,), (0,), (1,)), ((0,), (1,), (1,))),
        )
        self.assertEqual(result.haplotypes, ((0,), (0,), (1,)))
        self.assertEqual(result.second_best_cost, 1)
        self.assertEqual(result.gap, 1)
        a0, a1 = result.assignments
        self.assertEqual((a0.assigned, a0.tied, a0.costs), (0, True, (0, 0, 1)))
        self.assertEqual((a1.assigned, a1.tied, a1.costs), (2, False, (1, 1, 0)))
        self.assertEqual(a0.flips, ((), (), (0,)))
        self.assertEqual(a1.flips_applied, ())

    def test_diploid_ploidy_weighted_tie_between_two_pairs(self):
        # reads 00 (w5), 01 (w1), 11 (w1): pairs {00,01} and {00,11} both
        # cost 1; {00,10} costs 2 and is the strict second best.
        raw = [((0, 0), 5), ((0, 1), 1), ((1, 1), 1)]
        result = phase_polyploid(2, to_obs(raw), ploidy=2)
        self.assertEqual(result.optimal_cost, 1)
        self.assertEqual(
            result.optimal_solutions,
            (((0, 0), (0, 1)), ((0, 0), (1, 1))),
        )
        self.assertEqual(result.haplotypes, ((0, 0), (0, 1)))
        self.assertEqual(result.second_best_cost, 2)
        self.assertEqual(result.gap, 1)
        heavy, mid, _ = result.assignments
        self.assertEqual((heavy.assigned, heavy.costs), (0, (0, 5)))
        self.assertEqual((mid.assigned, mid.tied, mid.costs), (1, False, (1, 0)))

    def test_missing_sites_and_unique_solution(self):
        # read (1,0,1) w2 forces a copy 101; reads (0,1,-) and (-,1,0)
        # jointly force the other copy to be 010. Unique zero-cost optimum.
        raw = [((0, 1, None), 1), ((None, 1, 0), 1), ((1, 0, 1), 2)]
        result = phase_polyploid(3, to_obs(raw), ploidy=2)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(result.optimal_solutions, (((0, 1, 0), (1, 0, 1)),))
        self.assertEqual(result.haplotypes, ((0, 1, 0), (1, 0, 1)))
        # cheapest deviation: copies {110, 101} cover reads 2 and 3 exactly
        # and charge read (0,1,-) a single flip -> strict second best is 1
        self.assertEqual(result.second_best_cost, 1)
        self.assertEqual(result.gap, 1)
        self.assertEqual([a.assigned for a in result.assignments], [0, 0, 1])
        # (1,0,1) mismatches all three sites of copy 010, none of copy 101
        self.assertEqual(result.assignments[2].flips, ((0, 1, 2), ()))
        self.assertEqual(result.assignments[2].flips_applied, ())

    def test_all_missing_read_is_neutral_and_third_copy_is_free(self):
        # copies 00 and 11 are forced; the third copy is unconstrained, so
        # there are exactly 4 optima; the all-missing read ties everywhere.
        raw = [((None, None), 5), ((0, 0), 1), ((1, 1), 1)]
        result = phase_polyploid(2, to_obs(raw), ploidy=3)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(len(result.optimal_solutions), 4)
        self.assertEqual(result.haplotypes, ((0, 0), (0, 0), (1, 1)))
        self.assertEqual(result.second_best_cost, 1)
        miss = result.assignments[0]
        self.assertEqual((miss.assigned, miss.tied, miss.costs), (0, True, (0, 0, 0)))
        self.assertEqual(miss.flips, ((), (), ()))
        self.assertEqual(miss.flips_applied, ())

    def test_no_reads_every_solution_optimal(self):
        result = phase_polyploid(2, [], ploidy=2)
        self.assertEqual(result.optimal_cost, 0)
        # C(2**2 + 2 - 1, 2) = 10 sorted pairs, all optimal
        self.assertEqual(len(result.optimal_solutions), 10)
        self.assertEqual(result.haplotypes, ((0, 0), (0, 0)))
        self.assertIsNone(result.second_best_cost)
        self.assertIsNone(result.gap)
        self.assertEqual(result.assignments, ())


class ExactAgreementWithIndependentReference(unittest.TestCase):
    def test_random_cases_match_full_product_reference(self):
        for loci, ploidy, raw in random_polyploid_cases():
            with self.subTest(loci=loci, ploidy=ploidy, raw=raw):
                result = phase_polyploid(loci, to_obs(raw), ploidy=ploidy)
                best, optima, second = brute_force_polyploid(loci, ploidy, raw)
                self.assertEqual(result.optimal_cost, best)
                self.assertEqual(result.optimal_solutions, optima)
                self.assertEqual(result.second_best_cost, second)
                self.assertEqual(
                    result.gap, None if second is None else second - best
                )
                self.assertEqual(result.haplotypes, optima[0])
                # every reported optimum really attains the minimum
                for sol in result.optimal_solutions:
                    self.assertEqual(tuple(sorted(sol)), sol)
                    self.assertEqual(reference_cost_copies(sol, raw), best)
                # per-read reporting matches the independent formula
                for i, (alleles, w) in enumerate(raw):
                    a = result.assignments[i]
                    exp = reference_copy_assignment(result.haplotypes, alleles, w)
                    self.assertEqual((a.assigned, a.tied, a.costs, a.flips), exp)
                    self.assertEqual(a.flips_applied, a.flips[a.assigned])
                # per-read minima sum to the reported optimum
                self.assertEqual(
                    sum(min(a.costs) for a in result.assignments), best
                )

    def test_candidate_space_size_matches_closed_form(self):
        # C(2**loci + ploidy - 1, ploidy) canonical candidates, all evaluated
        result = phase_polyploid(6, to_obs([((0,) * 6, 1)]), ploidy=3)
        self.assertEqual(result.candidates_evaluated, 45760)  # C(66, 3)
        result2 = phase_polyploid(6, to_obs([((0,) * 6, 1)]), ploidy=2)
        self.assertEqual(result2.candidates_evaluated, 2080)  # C(65, 2)

    def test_result_is_deterministic(self):
        loci, ploidy, raw = random_polyploid_cases(seed=5)[0]
        r1 = phase_polyploid(loci, to_obs(raw), ploidy=ploidy)
        r2 = phase_polyploid(loci, to_obs(raw), ploidy=ploidy)
        self.assertEqual(r1, r2)


class ConsistencyWithDiploidEntry(unittest.TestCase):
    """Supplementary round-trip check against the diploid B&B entry.

    The two models differ: the diploid entry forces the two copies to be a
    complementary pair {h, 1-h}, while phase_polyploid(ploidy=2) lets the two
    copies be arbitrary. The general model is therefore a relaxation: its
    optimum can be strictly lower (e.g. reads 00/00/01 cost 0 with copies
    00+01 but cost 1 under any complementary pair). The exact analytic
    relation checked here: the diploid optimal pair is always a *feasible*
    polyploid solution, so poly_optimal_cost <= diploid_optimal_cost, and
    whenever the two costs are equal the diploid pair must appear among the
    polyploid optimal solutions.
    """

    def test_ploidy_2_relaxes_diploid_entry(self):
        # analytic sanity case: strictly lower cost under the relaxed model
        raw = [((0, 0), 1), ((0, 1), 1)]
        obs = to_obs(raw)
        self.assertEqual(phase_polyploid(2, obs, ploidy=2).optimal_cost, 0)
        self.assertEqual(phase_haplotypes(2, obs).optimal_cost, 1)

    def test_ploidy_2_cost_bounded_by_diploid_entry(self):
        for loci, ploidy, raw in random_polyploid_cases(seed=77):
            if ploidy != 2:
                continue
            with self.subTest(loci=loci, raw=raw):
                obs = to_obs(raw)
                poly = phase_polyploid(loci, obs, ploidy=2)
                diploid = phase_haplotypes(loci, obs)
                self.assertLessEqual(poly.optimal_cost, diploid.optimal_cost)
                if poly.optimal_cost == diploid.optimal_cost:
                    pair = tuple(sorted((diploid.haplotype, diploid.complement)))
                    self.assertIn(pair, poly.optimal_solutions)


class BoundarySizes(unittest.TestCase):
    def test_interface_maximum_instance(self):
        raw = [
            ((0, 1, 0, 1, 0, 1), 2),
            ((1, 0, 1, 0, 1, 0), 3),
            ((0, 0, None, None, 1, 1), 1),
            ((None, None, 1, 1, None, None), 4),
            ((1, 1, 0, 0, None, None), 1),
            ((None, 0, None, 1, None, 0), 2),
            ((0, None, None, None, None, 1), 1),
            ((None, None, None, None, None, None), 9),
        ]
        self.assertEqual(len(raw), POLYPLOID_MAX_OBSERVATIONS)
        result = phase_polyploid(POLYPLOID_MAX_LOCI, to_obs(raw), ploidy=3)
        best, optima, second = brute_force_polyploid(
            POLYPLOID_MAX_LOCI, 3, raw
        )
        self.assertEqual(result.optimal_cost, best)
        self.assertEqual(result.optimal_solutions, optima)
        self.assertEqual(result.second_best_cost, second)
        self.assertEqual(result.candidates_evaluated, 45760)

    def test_max_observations_accepted_nine_rejected(self):
        raw = [((0, 0), 1)] * POLYPLOID_MAX_OBSERVATIONS
        result = phase_polyploid(2, to_obs(raw), ploidy=2)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(len(result.assignments), POLYPLOID_MAX_OBSERVATIONS)
        with self.assertRaises(InvalidPhasingInputError):
            phase_polyploid(
                2, to_obs(raw) + [Observation((0, 0), 1)], ploidy=2
            )


class InputValidation(unittest.TestCase):
    def test_ploidy_contract(self):
        for bad in (0, 1, 4, -2, True, False, 2.0, math.nan, "3", None):
            with self.subTest(bad=bad):
                try:
                    phase_polyploid(2, [], ploidy=bad)
                except InvalidPhasingInputError as err:
                    self.assertIn("ploidy", str(err))
                else:
                    self.fail("expected InvalidPhasingInputError")
        for ok in POLYPLOID_PLOIDIES:
            phase_polyploid(2, [], ploidy=ok)  # must not raise

    def test_tighter_loci_limit_than_diploid_entry(self):
        # 7 loci is legal for the diploid entry but beyond the enumerable
        # polyploid limit of 6
        phase_haplotypes(7, [])
        try:
            phase_polyploid(7, [], ploidy=2)
        except InvalidPhasingInputError as err:
            self.assertIn("loci", str(err))
        else:
            self.fail("expected InvalidPhasingInputError")

    def test_allele_and_cost_errors_are_located(self):
        cases = [
            [Observation((2, 0), 1)],
            [Observation((0, True), 1)],
            [Observation((0, math.nan), 1)],
            [Observation((0, "0"), 1)],
        ]
        for obs in cases:
            with self.subTest(obs=obs):
                try:
                    phase_polyploid(2, obs, ploidy=3)
                except InvalidPhasingInputError as err:
                    self.assertIn("observations[0].alleles", str(err))
                else:
                    self.fail("expected InvalidPhasingInputError")
        for bad_cost in (0, -1, 1.0, True, math.inf, None):
            with self.subTest(bad_cost=bad_cost):
                try:
                    phase_polyploid(2, [Observation((0, 0), bad_cost)], ploidy=2)
                except InvalidPhasingInputError as err:
                    self.assertIn("observations[0].flip_cost", str(err))
                else:
                    self.fail("expected InvalidPhasingInputError")

    def test_container_and_length_errors(self):
        with self.assertRaises(InvalidPhasingInputError):
            phase_polyploid(2, "not-a-list", ploidy=2)  # type: ignore[arg-type]
        with self.assertRaises(InvalidPhasingInputError) as ctx:
            phase_polyploid(3, [Observation((0, 0), 1)], ploidy=2)
        self.assertIn("length 2", str(ctx.exception))
        with self.assertRaises(InvalidPhasingInputError):
            phase_polyploid(2, [(0, 0)], ploidy=2)  # type: ignore[list-item]

    def test_error_is_value_error_and_inputs_unchanged(self):
        data = [Observation((0, 0), 1), Observation((0, 1), 1)]
        with self.assertRaises(ValueError):
            phase_polyploid(3, data, ploidy=2)
        self.assertEqual(len(data), 2)  # failed call mutates nothing


if __name__ == "__main__":
    unittest.main()
