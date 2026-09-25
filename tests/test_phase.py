"""Tests for haplophase: exact agreement with an independent brute force,
disconnected/contradictory/weighted/missing-data scenarios, the B&B search
state, input-error contract, and the search-budget/infeasibility split.
"""

from __future__ import annotations

import math
import random
import unittest

from haplophase import (
    MAX_LOCI,
    MAX_OBSERVATIONS,
    MAX_TOP_K,
    Candidate,
    InvalidPhasingInputError,
    Observation,
    SearchBudgetExhaustedError,
    TopKResult,
    phase_haplotypes,
    top_k_haplotypes,
)

from reference import (
    brute_force,
    brute_top_k,
    random_instance,
    random_cases,
    reference_assignment,
    reference_cost,
)


def to_obs(raw_reads):
    return [Observation(alleles=a, flip_cost=w) for a, w in raw_reads]


class ExactAgreementWithBruteForce(unittest.TestCase):
    def test_random_cases_match_independent_reference(self):
        for loci, raw in random_cases():
            with self.subTest(loci=loci, raw=raw):
                result = phase_haplotypes(loci, to_obs(raw))
                best, optima, second = brute_force(loci, raw)
                self.assertEqual(result.optimal_cost, best)
                self.assertEqual(result.optimal_haplotypes, optima)
                self.assertEqual(result.second_best_cost, second)
                self.assertEqual(result.gap, None if second is None else second - best)
                self.assertEqual(result.haplotype, optima[0])
                self.assertEqual(
                    result.complement, tuple(1 - b for b in result.haplotype)
                )
                # every reported optimum really attains the minimum
                for h in result.optimal_haplotypes:
                    self.assertEqual(h[0], 0)
                    self.assertEqual(reference_cost(h, raw), best)
                # per-read reporting matches the independent assignment formula
                for i, (alleles, w) in enumerate(raw):
                    a = result.assignments[i]
                    exp = reference_assignment(result.haplotype, alleles, w)
                    self.assertEqual(
                        (a.assigned, a.tied, a.cost_to_h0, a.cost_to_h1,
                         a.flips_to_h0, a.flips_to_h1),
                        exp,
                    )
                    self.assertEqual(
                        a.flips_applied,
                        a.flips_to_h0 if a.assigned == 0 else a.flips_to_h1,
                    )
                # total of the applied flips equals the reported optimum
                self.assertEqual(
                    sum(
                        min(a.cost_to_h0, a.cost_to_h1)
                        for a in result.assignments
                    ),
                    best,
                )

    def test_result_is_deterministic(self):
        loci, raw = random_cases(seed=7, count=1)[0]
        r1 = phase_haplotypes(loci, to_obs(raw))
        r2 = phase_haplotypes(loci, to_obs(raw))
        self.assertEqual(r1, r2)


class TopKAgreementWithBruteForce(unittest.TestCase):
    def assert_matches_reference(self, loci, raw, k, **kwargs):
        result = top_k_haplotypes(loci, to_obs(raw), k, **kwargs)
        expected = brute_top_k(loci, raw, k)
        self.assertIsInstance(result, TopKResult)
        self.assertTrue(result.completed)
        self.assertTrue(result.ranking_proven)
        self.assertTrue(result.exhausted)
        self.assertIsNone(result.remaining_lower_bound)
        self.assertEqual(
            tuple((c.cost, c.haplotype) for c in result.candidates), expected
        )
        self.assertEqual(len(result.candidates), min(k, 2 ** (loci - 1)))
        self.assertEqual(
            result.candidates,
            tuple(sorted(result.candidates, key=lambda c: (c.cost, c.haplotype))),
        )
        for candidate, (cost, h) in zip(result.candidates, expected):
            self.assertIsInstance(candidate, Candidate)
            self.assertEqual(candidate.haplotype, h)
            self.assertEqual(candidate.complement, tuple(1 - b for b in h))
            self.assertEqual(candidate.cost, cost)
            self.assertEqual(reference_cost(h, raw), cost)
            self.assertEqual(
                sum(a.cost_to_h0 if a.assigned == 0 else a.cost_to_h1
                    for a in candidate.assignments),
                cost,
            )
            for i, (alleles, weight) in enumerate(raw):
                assignment = candidate.assignments[i]
                exp = reference_assignment(h, alleles, weight)
                self.assertEqual(
                    (
                        assignment.assigned,
                        assignment.tied,
                        assignment.cost_to_h0,
                        assignment.cost_to_h1,
                        assignment.flips_to_h0,
                        assignment.flips_to_h1,
                    ),
                    exp,
                )
                self.assertEqual(
                    assignment.flips_applied,
                    assignment.flips_to_h0
                    if assignment.assigned == 0
                    else assignment.flips_to_h1,
                )
        return result

    def test_random_cases_through_eight_loci(self):
        cases = []
        rng = random.Random(20260925)
        for loci in range(1, 9):
            for _ in range(8):
                cases.append((loci, random_instance(rng, loci)))
        for loci, raw in cases:
            for k in (1, 2, 5, 20):
                with self.subTest(loci=loci, k=k, raw=raw):
                    self.assert_matches_reference(loci, raw, k)

    def test_zero_information_reads_return_every_canonical_string(self):
        raw = [
            ((None, None, None), 1),
            ((None, None, None), 9),
        ]
        result = self.assert_matches_reference(3, raw, 20)
        self.assertEqual(
            [c.haplotype for c in result.candidates],
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)],
        )
        self.assertTrue(all(c.cost == 0 for c in result.candidates))
        self.assertTrue(all(a.tied for c in result.candidates for a in c.assignments))

    def test_equal_cost_candidates_are_lexicographically_ordered(self):
        raw = [
            ((0, 0, None, None), 1),
            ((None, None, 0, 1), 1),
        ]
        result = self.assert_matches_reference(4, raw, 5)
        self.assertEqual(
            [(c.cost, c.haplotype) for c in result.candidates[:2]],
            [(0, (0, 0, 0, 1)), (0, (0, 0, 1, 0))],
        )

    def test_unique_optimum_and_k_one(self):
        raw = [((0, 0), 10), ((0, 1), 1)]
        result = self.assert_matches_reference(2, raw, 1)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].haplotype, (0, 0))
        self.assertEqual(result.candidates[0].cost, 1)

    def test_k_greater_than_legal_count_reports_all_and_exhaustion(self):
        raw = [((0, 0, None), 1)]
        result = self.assert_matches_reference(3, raw, MAX_TOP_K)
        self.assertEqual(len(result.candidates), 4)

    def test_budget_returns_unproven_incumbents_with_remaining_bound(self):
        raw = [((0, 0, 0, 0), 1), ((0, 1, 0, 1), 1)]
        result = top_k_haplotypes(4, to_obs(raw), 3, max_nodes=1)
        self.assertFalse(result.completed)
        self.assertFalse(result.ranking_proven)
        self.assertFalse(result.exhausted)
        self.assertIsInstance(result.remaining_lower_bound, int)
        self.assertGreaterEqual(result.remaining_lower_bound, 0)
        self.assertLessEqual(len(result.candidates), 3)
        for candidate in result.candidates:
            self.assertEqual(reference_cost(candidate.haplotype, raw), candidate.cost)

    def test_top_k_validation(self):
        for bad in (0, -1, MAX_TOP_K + 1, 1.0, True, False, None, "3"):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidPhasingInputError):
                    top_k_haplotypes(2, [], bad)


class StructuralScenarios(unittest.TestCase):
    def test_disconnected_components_give_multiple_optima(self):
        # Two fragments share no locus: component {0,1} and free component {2,3}
        raw = [
            ((0, 0, None, None), 1),
            ((None, None, 0, 1), 1),
        ]
        result = phase_haplotypes(4, to_obs(raw))
        best, optima, _ = brute_force(4, raw)
        self.assertEqual(result.optimal_cost, best)
        self.assertEqual(result.optimal_haplotypes, optima)
        self.assertEqual(len(result.optimal_haplotypes), 2)
        self.assertEqual(result.free_components, ((2, 3),))
        self.assertEqual(result.gap, 1)  # nearest distinct h costs 1 more

    def test_single_unlinked_locus_is_free(self):
        raw = [((0, None), 1)]
        result = phase_haplotypes(2, to_obs(raw))
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(
            result.optimal_haplotypes, ((0, 0), (0, 1))
        )
        self.assertEqual(result.free_components, ((1,),))
        self.assertIsNone(result.second_best_cost)
        self.assertIsNone(result.gap)

    def test_no_reads_all_haplotypes_optimal(self):
        result = phase_haplotypes(3, [])
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(len(result.optimal_haplotypes), 4)
        self.assertIsNone(result.second_best_cost)
        self.assertEqual(result.free_components, ((1,), (2,)))

    def test_contradictory_fragments_force_a_flip_and_can_tie(self):
        # Two fragments agree at site 0 (both see 0) but contradict elsewhere:
        # no error-free phasing exists; two orientations are equally good.
        raw = [
            ((0, 0, 0), 1),
            ((0, 1, 1), 1),
        ]
        result = phase_haplotypes(3, to_obs(raw))
        best, optima, second = brute_force(3, raw)
        self.assertEqual(best, 1)
        self.assertEqual(result.optimal_cost, 1)
        self.assertEqual(result.optimal_haplotypes, optima)
        self.assertEqual(set(optima), {(0, 0, 0), (0, 1, 1)})
        self.assertEqual(result.second_best_cost, second)

    def test_non_uniform_weights_change_the_answer(self):
        # Heavy read (0,0) vs light contradictory read (0,1): uniform weights
        # tie at cost 1; with weights 10 vs 1 only h=00 stays optimal.
        raw = [((0, 0), 10), ((0, 1), 1)]
        result = phase_haplotypes(2, to_obs(raw))
        best, optima, second = brute_force(2, raw)
        self.assertEqual(best, 1)
        self.assertEqual(result.optimal_cost, 1)
        self.assertEqual(result.optimal_haplotypes, ((0, 0),))
        self.assertEqual(second, 10)
        self.assertEqual(result.gap, 9)
        # the heavy read is error-free on h=00; the light read is one flip
        # away and is internally tied (flipping either allele costs 1)...
        heavy, light = result.assignments
        self.assertEqual((heavy.assigned, heavy.cost_to_h0, heavy.tied), (0, 0, False))
        self.assertEqual((light.assigned, light.tied), (0, True))
        self.assertEqual((light.cost_to_h0, light.cost_to_h1), (1, 1))
        # ...but the two candidates are NOT globally tied: h=01 would charge
        # the heavy read 10, which is exactly what the gap reports.
        self.assertEqual(result.second_best_cost, 10)

    def test_all_missing_read_is_always_tied_and_neutral(self):
        raw = [
            ((None, None), 5),
            ((0, 0), 1),
        ]
        result = phase_haplotypes(2, to_obs(raw))
        best, optima, _ = brute_force(2, raw)
        self.assertEqual(best, 0)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(result.optimal_haplotypes, ((0, 0),))
        miss = result.assignments[0]
        self.assertTrue(miss.tied)
        self.assertEqual(miss.assigned, 0)
        self.assertEqual((miss.cost_to_h0, miss.cost_to_h1), (0, 0))
        self.assertEqual(miss.flips_to_h0, ())
        self.assertEqual(miss.flips_to_h1, ())
        self.assertEqual(miss.flips_applied, ())

    def test_per_site_majority_vote_would_be_wrong(self):
        # Independent site-wise majority votes (0,0) but the global minimum
        # is (0,1): the objective couples sites inside each read.
        raw = [
            ((0, 0), 1),
            ((0, 1), 1),
            ((1, 0), 1),
        ]
        result = phase_haplotypes(2, to_obs(raw))
        best, optima, _ = brute_force(2, raw)
        self.assertEqual(result.optimal_haplotypes, optima)
        self.assertEqual(result.optimal_cost, 1)
        self.assertNotIn((0, 0), result.optimal_haplotypes)
        self.assertEqual(best, 1)

    def test_single_locus_every_read_costs_zero(self):
        raw = [((0,), 3), ((1,), 2), ((None,), 9)]
        result = phase_haplotypes(1, to_obs(raw))
        self.assertEqual(result.haplotype, (0,))
        self.assertEqual(result.optimal_cost, 0)
        self.assertIsNone(result.second_best_cost)
        self.assertEqual([a.tied for a in result.assignments], [False, False, True])
        self.assertEqual(result.assignments[0].assigned, 0)
        self.assertEqual(result.assignments[1].assigned, 1)


class SearchStateTests(unittest.TestCase):
    def test_tree_counters_and_pruning(self):
        raw = [
            ((0, 0, 0, 0), 4),
            ((0, 1, 0, 1), 2),
            ((None, 1, 1, None), 3),
        ]
        result = phase_haplotypes(4, to_obs(raw))
        s = result.search
        self.assertTrue(s.completed)
        self.assertGreaterEqual(s.nodes_opened, 1)
        self.assertLessEqual(s.nodes_opened, 2 ** 4 - 1)  # full binary tree size
        self.assertLessEqual(s.leaves_evaluated, 2 ** 3)
        self.assertGreaterEqual(s.leaves_evaluated, 1)
        self.assertGreaterEqual(s.branches_pruned, 0)
        self.assertEqual(len(s.nodes_by_depth), 5)
        self.assertEqual(len(s.pruned_by_depth), 4)
        self.assertEqual(sum(s.nodes_by_depth), s.nodes_opened)
        self.assertEqual(sum(s.pruned_by_depth), s.branches_pruned)
        self.assertGreaterEqual(s.incumbent_updates, 1)
        # pruning + evaluation must not lose the independent optimum
        best, optima, _ = brute_force(4, raw)
        self.assertEqual(result.optimal_haplotypes, optima)
        self.assertEqual(result.optimal_cost, best)

    def test_pruning_actually_closes_nodes_on_hard_instance(self):
        # Dense overlapping error-free windows: connected overlap graph,
        # unique zero-cost phasing, the admissible bound prunes aggressively.
        truth = (0, 1, 0, 1, 0, 1, 0, 1)
        windows = ((0, 4), (2, 6), (4, 8))
        raw = []
        for lo, hi in windows:
            frag = (None,) * lo + truth[lo:hi] + (None,) * (8 - hi)
            raw.append((frag, 1))
        result = phase_haplotypes(8, to_obs(raw))
        full_tree = 2 ** 8 - 1
        self.assertLess(result.search.nodes_opened, full_tree)
        self.assertGreater(result.search.branches_pruned, 0)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(result.optimal_haplotypes, (truth,))

    def test_search_budget_exhaustion_is_separate_from_infeasibility(self):
        raw = [((0, 0, 0, 0), 1), ((0, 1, 0, 1), 1)]
        with self.assertRaises(SearchBudgetExhaustedError) as ctx:
            phase_haplotypes(4, to_obs(raw), max_nodes=1)
        err = ctx.exception
        self.assertIsInstance(err, RuntimeError)
        self.assertNotIsInstance(err, ValueError)
        self.assertEqual(err.budget, 1)
        self.assertLessEqual(err.nodes_opened, 2)
        # the seeded incumbent is a feasible candidate with an exact cost
        self.assertIsNotNone(err.best_cost)
        self.assertIsNotNone(err.best_haplotype)
        self.assertEqual(err.best_haplotype[0], 0)
        self.assertEqual(reference_cost(err.best_haplotype, raw), err.best_cost)

    def test_generous_budget_completes(self):
        raw = [((0, 0, None), 1), ((None, 0, 1), 2)]
        result = phase_haplotypes(3, to_obs(raw), max_nodes=10 ** 9)
        self.assertTrue(result.search.completed)


class BoundarySizes(unittest.TestCase):
    def test_max_loci_planted_solution(self):
        truth = (0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0)
        self.assertEqual(len(truth), MAX_LOCI)
        raw = []
        # error-free windows from h that overlap at one site, chaining all
        # 18 loci into one connected component (unique planted optimum)
        for lo in range(0, MAX_LOCI - 1, 3):
            hi = min(lo + 4, MAX_LOCI)
            frag = (None,) * lo + truth[lo:hi] + (None,) * (MAX_LOCI - hi)
            raw.append((frag, 1))
        # one fragment drawn from the complement 1-h; error-free on that side
        comp = tuple(1 - b for b in truth)
        frag = comp[0:3] + (None,) * (MAX_LOCI - 3)
        raw.append((frag, 2))
        raw.append((tuple(None for _ in range(MAX_LOCI)), 4))
        result = phase_haplotypes(MAX_LOCI, to_obs(raw))
        self.assertEqual(result.optimal_cost, 0)
        self.assertIn(truth, result.optimal_haplotypes)
        self.assertEqual(result.haplotype, truth)

    def test_max_observations_accepted(self):
        raw = [((0, 0), 1) for _ in range(MAX_OBSERVATIONS)]
        result = phase_haplotypes(2, to_obs(raw))
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(len(result.assignments), MAX_OBSERVATIONS)


class InputValidation(unittest.TestCase):
    def assertRejected(self, loci, obs, *, max_nodes=None):
        with self.assertRaises(InvalidPhasingInputError):
            phase_haplotypes(loci, obs, max_nodes=max_nodes)

    def test_loci_type_and_range(self):
        for bad in (0, MAX_LOCI + 1, -1, 1.5, True, False, "3", None):
            with self.subTest(bad=bad):
                self.assertRejected(bad, [])
        self.assertRejected(2, [], max_nodes=0)
        self.assertRejected(2, [], max_nodes=True)
        self.assertRejected(2, [], max_nodes=1.0)

    def test_allele_values_rejected_with_location(self):
        cases = [
            [Observation((2, 0), 1)],
            [Observation((0, -1), 1)],
            [Observation((0, True), 1)],
            [Observation((0, math.nan), 1)],
            [Observation((math.inf, 0), 1)],
            [Observation((0, "0"), 1)],
        ]
        for obs in cases:
            with self.subTest(obs=obs):
                try:
                    phase_haplotypes(2, obs)
                except InvalidPhasingInputError as err:
                    self.assertIn("observations[0].alleles", str(err))
                else:
                    self.fail("expected InvalidPhasingInputError")

    def test_flip_cost_contract(self):
        for bad in (0, -3, 1.0, True, math.nan, -math.inf, None, "1"):
            with self.subTest(bad=bad):
                try:
                    phase_haplotypes(2, [Observation((0, 0), bad)])
                except InvalidPhasingInputError as err:
                    self.assertIn("observations[0].flip_cost", str(err))
                else:
                    self.fail("expected InvalidPhasingInputError")

    def test_length_and_container_errors(self):
        with self.assertRaises(InvalidPhasingInputError) as ctx:
            phase_haplotypes(3, [Observation((0, 0), 1)])
        self.assertIn("length 2", str(ctx.exception))
        with self.assertRaises(InvalidPhasingInputError) as ctx:
            phase_haplotypes(2, [(0, 0)])  # type: ignore[list-item]
        self.assertIn("observations[0]", str(ctx.exception))
        with self.assertRaises(InvalidPhasingInputError):
            phase_haplotypes(2, "not-a-list")  # type: ignore[arg-type]
        with self.assertRaises(InvalidPhasingInputError):
            phase_haplotypes(2, [Observation((0, 0), 1)] * (MAX_OBSERVATIONS + 1))

    def test_error_is_value_error_and_inputs_unchanged(self):
        data = [Observation((0, 0), 1), Observation((0, 1), 1)]
        with self.assertRaises(ValueError):
            phase_haplotypes(3, data)
        self.assertEqual(len(data), 2)  # failed call mutates nothing

    def test_nan_inside_observation_constructor_is_caught_at_call(self):
        # The dataclass itself is just a container; validation happens at call.
        obs = Observation((math.nan, 0), 1)
        with self.assertRaises(InvalidPhasingInputError):
            phase_haplotypes(2, [obs])


if __name__ == "__main__":
    unittest.main()
