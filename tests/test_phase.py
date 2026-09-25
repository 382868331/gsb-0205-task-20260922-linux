"""Tests for haplophase: exact agreement with an independent brute force,
disconnected/contradictory/weighted/missing-data scenarios, the B&B search
state, input-error contract, and the search-budget/infeasibility split.
"""

from __future__ import annotations

import math
import unittest

from haplophase import (
    MAX_LOCI,
    MAX_OBSERVATIONS,
    BoltzmannPosterior,
    boltzmann_posterior,
    InvalidPhasingInputError,
    Observation,
    SearchBudgetExhaustedError,
    phase_haplotypes,
)

from reference import (
    reference_boltzmann,
    brute_force,
    random_cases,
    reference_assignment,
    reference_cost,
)


def assert_float_sequence_close(test_case, actual, expected, tol=1e-12):
    test_case.assertEqual(len(actual), len(expected))
    for a, b in zip(actual, expected):
        test_case.assertAlmostEqual(a, b, delta=tol)


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


class BoltzmannPosteriorTests(unittest.TestCase):
    def test_matches_independent_global_enumeration(self):
        for loci, raw in random_cases(seed=20260926, count=40):
            temperature = 0.37
            posterior = boltzmann_posterior(loci, to_obs(raw), temperature)
            marginals, log_z, entropy, count = reference_boltzmann(
                loci, raw, temperature
            )
            self.assertIsInstance(posterior, BoltzmannPosterior)
            self.assertEqual(posterior.temperature, temperature)
            self.assertEqual(posterior.canonical_configuration_count, count)
            assert_float_sequence_close(self, posterior.marginals, marginals)
            self.assertAlmostEqual(posterior.log_normalizer, log_z, places=10)
            self.assertAlmostEqual(posterior.entropy, entropy, places=10)
            self.assertTrue(all(0.0 <= p <= 1.0 for p in posterior.marginals))

    def test_all_costs_equal_gives_uniform_canonical_posterior(self):
        raw = []
        posterior = boltzmann_posterior(3, raw, temperature=2.0)
        self.assertEqual(posterior.canonical_configuration_count, 4)
        assert_float_sequence_close(
            self, posterior.marginals, (0.0, 0.5, 0.5)
        )
        self.assertAlmostEqual(posterior.log_normalizer, math.log(4.0), places=12)
        self.assertAlmostEqual(posterior.entropy, 2.0 * math.log(2.0), places=12)

    def test_disconnected_components_are_counted_without_extra_division(self):
        raw = [
            ((0, 0, None, None), 1),
            ((None, None, 0, 1), 1),
        ]
        posterior = boltzmann_posterior(4, to_obs(raw), temperature=1.0)
        self.assertEqual(
            [c.configuration_count for c in posterior.components], [2, 4]
        )
        self.assertEqual(posterior.canonical_configuration_count, 8)
        marginals, log_z, entropy, count = reference_boltzmann(4, raw, 1.0)
        assert_float_sequence_close(self, posterior.marginals, marginals)
        self.assertAlmostEqual(posterior.log_normalizer, log_z, places=12)
        self.assertAlmostEqual(posterior.entropy, entropy, places=12)

    def test_completely_unsupported_locus_is_coin_flip(self):
        raw = [((0, 1, None), 1)]
        posterior = boltzmann_posterior(3, to_obs(raw), temperature=0.5)
        marginals, log_z, entropy, _ = reference_boltzmann(3, raw, 0.5)
        assert_float_sequence_close(self, posterior.marginals, marginals)
        self.assertAlmostEqual(posterior.marginals[2], 0.5)
        self.assertAlmostEqual(posterior.log_normalizer, log_z, places=12)
        self.assertAlmostEqual(posterior.entropy, entropy, places=12)

    def test_extreme_low_temperature_uses_log_shifted_density_without_zero_divide(self):
        raw = [
            ((0, 0), 1),
            ((0, 1), 1),
            ((1, 0), 1),
        ]
        posterior = boltzmann_posterior(2, to_obs(raw), temperature=1e-100)
        assert_float_sequence_close(self, posterior.marginals, (0.0, 1.0))
        self.assertEqual(posterior.entropy, 0.0)
        self.assertTrue(math.isfinite(posterior.components[0].log_normalizer))

    def test_equal_optima_remain_uncertain_at_extreme_low_temperature(self):
        raw = [((0, 0, None, None), 1), ((None, None, 0, 0), 1)]
        posterior = boltzmann_posterior(4, to_obs(raw), temperature=1e-100)
        assert_float_sequence_close(
            self, posterior.marginals, (0.0, 0.0, 0.5, 0.5)
        )
        self.assertAlmostEqual(posterior.entropy, math.log(2.0), places=12)

    def test_same_optimum_can_have_different_posterior_uncertainty(self):
        sharp_raw = [
            ((0, 0, 0), 10),
            ((0, 0, 1), 10),
            ((0, 1, 0), 10),
        ]
        broad_raw = [
            ((0, 0, 0), 1),
            ((0, 0, 1), 1),
            ((0, 1, 0), 1),
        ]
        sharp = boltzmann_posterior(3, to_obs(sharp_raw), 0.5)
        broad = boltzmann_posterior(3, to_obs(broad_raw), 0.5)
        self.assertEqual(
            phase_haplotypes(3, to_obs(sharp_raw)).haplotype,
            phase_haplotypes(3, to_obs(broad_raw)).haplotype,
        )
        self.assertLess(sharp.entropy, broad.entropy)

    def test_temperature_must_be_positive_finite(self):
        obs = []
        for bad in (0, -1, True, False, math.nan, math.inf, -math.inf, "1"):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidPhasingInputError):
                    boltzmann_posterior(2, obs, bad)

    def test_posterior_reuses_original_input_contract(self):
        with self.assertRaises(InvalidPhasingInputError):
            boltzmann_posterior(0, [], 1.0)
        with self.assertRaises(InvalidPhasingInputError):
            boltzmann_posterior(2, [Observation((0, 2), 1)], 1.0)


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
