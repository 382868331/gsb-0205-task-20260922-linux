"""Tests for phase_haplotypes_top_k: agreement with an independent full
enumeration over all h[0]=0 strings (loci <= 8), tie handling, the K=1
degenerate case, zero-information reads, budget semantics and validation.
"""

from __future__ import annotations

import random
import unittest

from haplophase import (
    MAX_TOP_K,
    InvalidPhasingInputError,
    Observation,
    phase_haplotypes,
    phase_haplotypes_top_k,
)

from reference import (
    brute_force_top_k,
    random_instance,
    reference_assignment,
    reference_cost,
)


def to_obs(raw_reads):
    return [Observation(alleles=a, flip_cost=w) for a, w in raw_reads]


def random_topk_cases(seed=20260925, count=60, max_loci=8):
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        loci = rng.randint(1, max_loci)
        k = rng.randint(1, MAX_TOP_K)
        cases.append((loci, random_instance(rng, loci), k))
    return cases


class ExactTopKAgreement(unittest.TestCase):
    def test_random_cases_match_independent_reference(self):
        for loci, raw, k in random_topk_cases():
            with self.subTest(loci=loci, k=k, raw=raw):
                result = phase_haplotypes_top_k(loci, to_obs(raw), k)
                expected, total = brute_force_top_k(loci, raw, k)
                self.assertTrue(result.complete)
                self.assertEqual(result.solution_count, total)
                got = [(c.cost, c.haplotype) for c in result.candidates]
                self.assertEqual(got, expected)
                # ranks are 1-based and dense, ordering is (cost, h)
                self.assertEqual(
                    [c.rank for c in result.candidates],
                    list(range(1, len(expected) + 1)),
                )
                for cand in result.candidates:
                    self.assertEqual(cand.haplotype[0], 0)
                    self.assertEqual(
                        cand.complement, tuple(1 - b for b in cand.haplotype)
                    )
                    self.assertEqual(
                        reference_cost(cand.haplotype, raw), cand.cost
                    )
                    # per-candidate read assignment matches the independent
                    # formula for that candidate's own haplotype
                    self.assertEqual(len(cand.assignments), len(raw))
                    for i, (alleles, w) in enumerate(raw):
                        a = cand.assignments[i]
                        exp = reference_assignment(cand.haplotype, alleles, w)
                        self.assertEqual(
                            (a.assigned, a.tied, a.cost_to_h0, a.cost_to_h1,
                             a.flips_to_h0, a.flips_to_h1),
                            exp,
                        )
                        self.assertEqual(
                            a.flips_applied,
                            a.flips_to_h0 if a.assigned == 0 else a.flips_to_h1,
                        )
                    self.assertEqual(
                        sum(min(a.cost_to_h0, a.cost_to_h1)
                            for a in cand.assignments),
                        cand.cost,
                    )
                self.assertEqual(
                    result.exhaustive, len(expected) == total
                )
                self.assertIsNone(result.remaining_lower_bound)

    def test_result_is_deterministic(self):
        loci, raw, k = random_topk_cases(seed=3, count=1)[0]
        r1 = phase_haplotypes_top_k(loci, to_obs(raw), k)
        r2 = phase_haplotypes_top_k(loci, to_obs(raw), k)
        self.assertEqual(r1, r2)


class StructuralTopKScenarios(unittest.TestCase):
    def test_zero_information_reads_all_cost_zero(self):
        # Fully missing reads carry no information: every h costs 0, so the
        # top-K is purely the lexicographic order and ties must survive.
        raw = [((None, None, None), 5), ((None, None, None), 1)]
        result = phase_haplotypes_top_k(3, to_obs(raw), 3)
        self.assertTrue(result.complete)
        self.assertEqual(
            [(c.cost, c.haplotype) for c in result.candidates],
            [(0, (0, 0, 0)), (0, (0, 0, 1)), (0, (0, 1, 0))],
        )
        self.assertFalse(result.exhaustive)
        self.assertEqual(result.solution_count, 4)
        for cand in result.candidates:
            for a in cand.assignments:
                self.assertTrue(a.tied)
                self.assertEqual(a.flips_applied, ())

    def test_equal_cost_candidates_not_pruned(self):
        # Two contradictory unit-weight fragments: h=000 and h=011 both cost
        # exactly 1; a non-strict (>=) prune would drop one of them.
        raw = [((0, 0, 0), 1), ((0, 1, 1), 1)]
        result = phase_haplotypes_top_k(3, to_obs(raw), 2)
        self.assertEqual(
            [(c.cost, c.haplotype) for c in result.candidates],
            [(1, (0, 0, 0)), (1, (0, 1, 1))],
        )

    def test_unique_optimum_then_strictly_priced_runners_up(self):
        raw = [((0, 0), 10), ((0, 1), 1)]
        result = phase_haplotypes_top_k(2, to_obs(raw), 2)
        self.assertEqual(
            [(c.cost, c.haplotype) for c in result.candidates],
            [(1, (0, 0)), (10, (0, 1))],
        )
        self.assertTrue(result.exhaustive)  # only 2 feasible h exist
        self.assertEqual(result.solution_count, 2)

    def test_k1_matches_single_optimum_interface(self):
        rng = random.Random(11)
        for _ in range(20):
            loci = rng.randint(1, 8)
            raw = random_instance(rng, loci)
            with self.subTest(loci=loci, raw=raw):
                top = phase_haplotypes_top_k(loci, to_obs(raw), 1)
                single = phase_haplotypes(loci, to_obs(raw))
                self.assertEqual(len(top.candidates), 1)
                self.assertEqual(top.candidates[0].cost, single.optimal_cost)
                self.assertEqual(top.candidates[0].haplotype, single.haplotype)

    def test_k_exceeding_solution_count_returns_all_and_marks_exhaustive(self):
        raw = [((0, 0), 1)]
        result = phase_haplotypes_top_k(2, to_obs(raw), MAX_TOP_K)
        self.assertTrue(result.complete)
        self.assertTrue(result.exhaustive)
        self.assertEqual(result.solution_count, 2)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(
            [c.haplotype for c in result.candidates], [(0, 0), (0, 1)]
        )

    def test_no_reads_every_h_listed_lexicographically(self):
        result = phase_haplotypes_top_k(3, [], 4)
        self.assertTrue(result.exhaustive)
        self.assertEqual(
            [c.haplotype for c in result.candidates],
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)],
        )
        self.assertTrue(all(c.cost == 0 for c in result.candidates))


class TopKBudgetSemantics(unittest.TestCase):
    def test_budget_exhaustion_returns_unproven_partial_listing(self):
        rng = random.Random(5)
        loci = 8
        raw = [
            (tuple(rng.choice((0, 1)) if rng.random() < 0.8 else None
                   for _ in range(loci)), rng.randint(1, 5))
            for _ in range(12)
        ]
        k = 4
        partial = phase_haplotypes_top_k(loci, to_obs(raw), k, max_nodes=3)
        self.assertFalse(partial.complete)
        self.assertFalse(partial.exhaustive)
        self.assertIsNone(partial.solution_count)
        self.assertIsNotNone(partial.remaining_lower_bound)
        self.assertFalse(partial.search.completed)
        # the partial listing is still feasible candidates with exact costs
        self.assertGreaterEqual(len(partial.candidates), 1)
        self.assertLessEqual(len(partial.candidates), k)
        for cand in partial.candidates:
            self.assertEqual(cand.haplotype[0], 0)
            self.assertEqual(reference_cost(cand.haplotype, raw), cand.cost)
        # the remaining bound never exceeds the true k-th cost: it bounds
        # what unexplored branches could still contribute
        proven = phase_haplotypes_top_k(loci, to_obs(raw), k)
        self.assertTrue(proven.complete)
        self.assertLessEqual(
            partial.remaining_lower_bound, proven.candidates[-1].cost
        )
        # the best partial candidate is a feasible upper bound on the
        # proven optimum; ranks beyond what was searched stay unproven
        self.assertGreaterEqual(
            min(c.cost for c in partial.candidates),
            proven.candidates[0].cost,
        )

    def test_generous_budget_completes(self):
        raw = [((0, 0, None), 1), ((None, 0, 1), 2)]
        result = phase_haplotypes_top_k(3, to_obs(raw), 3, max_nodes=10 ** 9)
        self.assertTrue(result.complete)
        self.assertTrue(result.search.completed)


class TopKValidation(unittest.TestCase):
    def test_k_contract(self):
        for bad in (0, -1, MAX_TOP_K + 1, 100, 1.0, 2.5, True, False, "3", None):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidPhasingInputError) as ctx:
                    phase_haplotypes_top_k(2, [], bad)
                self.assertIn("k:", str(ctx.exception))

    def test_observation_validation_still_applies(self):
        with self.assertRaises(InvalidPhasingInputError):
            phase_haplotypes_top_k(2, [Observation((0, 2), 1)], 2)
        with self.assertRaises(InvalidPhasingInputError):
            phase_haplotypes_top_k(2, [Observation((0, 0), 0)], 2)
        with self.assertRaises(InvalidPhasingInputError):
            phase_haplotypes_top_k(2, [Observation((0, 0), 1)], 2, max_nodes=0)

    def test_failed_call_mutates_nothing(self):
        data = [Observation((0, 0), 1)]
        with self.assertRaises(ValueError):
            phase_haplotypes_top_k(2, data, 0)
        self.assertEqual(len(data), 1)


if __name__ == "__main__":
    unittest.main()
