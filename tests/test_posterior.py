"""Tests for the Boltzmann posterior entry point: exact agreement with an
independent flat-enumeration reference, equal-cost / ultra-low-temperature
/ disconnected-component / no-evidence structure, and the error contract.
"""

from __future__ import annotations

import math
import random
import unittest

from haplophase import (
    InvalidPhasingInputError,
    Observation,
    phase_haplotypes,
    phase_posterior,
)

from reference import brute_force, brute_force_posterior, random_instance


def to_obs(raw_reads):
    return [Observation(alleles=a, flip_cost=w) for a, w in raw_reads]


class AgreementWithIndependentEnumeration(unittest.TestCase):
    def test_random_cases_match_reference(self):
        rng = random.Random(20260926)
        for case in range(60):
            loci = rng.randint(1, 8)
            raw = random_instance(rng, loci)
            temp = rng.choice([0.05, 0.5, 1, 3, 7, 100])
            with self.subTest(case=case, loci=loci, raw=raw, temp=temp):
                got = phase_posterior(loci, to_obs(raw), temperature=temp)
                marginals, log_z, entropy, mean_cost = brute_force_posterior(
                    loci, raw, temp
                )
                self.assertEqual(len(got.marginals), loci)
                self.assertEqual(got.marginals[0], 0.0)
                for g, e in zip(got.marginals, marginals):
                    self.assertAlmostEqual(g, e, places=12)
                self.assertAlmostEqual(got.log_partition, log_z, places=9)
                self.assertAlmostEqual(got.entropy, entropy, places=9)
                self.assertAlmostEqual(got.mean_cost, mean_cost, places=9)

    def test_candidate_multiplicity_matches_optima_counting(self):
        # disconnected instance: the flat canonical enumeration and the
        # component-wise sum must weight the same 2**(loci-1) candidates.
        raw = [
            ((0, 0, None, None), 2),
            ((1, 1, None, None), 1),
            ((None, None, 0, 1), 3),
        ]
        result = phase_haplotypes(4, to_obs(raw))
        self.assertEqual(len(result.optimal_haplotypes), 2)  # one free block
        got = phase_posterior(4, to_obs(raw), temperature=2.5)
        _, log_z, _, _ = brute_force_posterior(4, raw, 2.5)
        self.assertAlmostEqual(got.log_partition, log_z, places=12)


class EqualCosts(unittest.TestCase):
    def test_no_evidence_at_all_is_uniform(self):
        got = phase_posterior(5, [], temperature=1.5)
        self.assertEqual(got.marginals, (0.0, 0.5, 0.5, 0.5, 0.5))
        self.assertAlmostEqual(got.log_partition, 4 * math.log(2), places=12)
        self.assertAlmostEqual(got.entropy, 4 * math.log(2), places=12)
        self.assertEqual(got.mean_cost, 0.0)

    def test_all_missing_reads_are_uniform(self):
        raw = [((None, None, None), 4), ((None, None, None), 1)]
        got = phase_posterior(3, to_obs(raw), temperature=0.7)
        self.assertEqual(got.marginals, (0.0, 0.5, 0.5))
        self.assertAlmostEqual(got.log_partition, 2 * math.log(2), places=12)
        self.assertAlmostEqual(got.entropy, 2 * math.log(2), places=12)


class UltraLowTemperature(unittest.TestCase):
    def test_no_underflow_and_concentration_on_optima(self):
        raw = [
            ((0, 0, 0), 5),
            ((0, 0, 1), 1),
            ((1, 1, 0), 2),
        ]
        best, optima, _ = brute_force(3, raw)
        got = phase_posterior(3, to_obs(raw), temperature=1e-9)
        # posterior concentrates on the optima: marginals equal the fraction
        # of optimal haplotypes with a 1 at each locus, and entropy tends to
        # log(#optima); nothing underflowed into a 0/0.
        for k in range(3):
            expected = sum(h[k] for h in optima) / len(optima)
            self.assertAlmostEqual(got.marginals[k], expected, places=6)
        self.assertAlmostEqual(got.entropy, math.log(len(optima)), places=6)
        self.assertAlmostEqual(got.mean_cost, best, delta=1e-3)
        self.assertTrue(math.isfinite(got.log_partition))

    def test_unique_optimum_at_tiny_temperature(self):
        raw = [((0, 0), 3), ((0, 0), 1), ((1, 1), 2)]
        got = phase_posterior(2, to_obs(raw), temperature=1e-12)
        self.assertAlmostEqual(got.marginals[1], 0.0, places=9)
        self.assertAlmostEqual(got.entropy, 0.0, places=9)


class DisconnectedComponents(unittest.TestCase):
    def test_independent_blocks_factorize(self):
        block_a = [((0, 0, None, None), 2), ((1, 0, None, None), 1)]
        block_b = [((None, None, 1, 1), 3), ((None, None, 0, 1), 1)]
        raw = block_a + block_b
        temp = 1.3
        joint = phase_posterior(4, to_obs(raw), temperature=temp)
        ma, lza, ha, ea = brute_force_posterior(
            2, [((0, 0), 2), ((1, 0), 1)], temp
        )
        mb, lzb, hb, eb = brute_force_posterior(
            2, [((1, 1), 3), ((0, 1), 1)], temp
        )
        # block B is free: both global flips are distinct candidates, so its
        # partition sum doubles (adds log 2) and every marginal is 1/2.
        self.assertAlmostEqual(
            joint.log_partition, lza + lzb + math.log(2), places=12
        )
        self.assertAlmostEqual(joint.entropy, ha + hb + math.log(2), places=12)
        self.assertAlmostEqual(joint.mean_cost, ea + eb, places=12)
        # block A keeps the gauge (site 0 fixed to 0).
        self.assertAlmostEqual(joint.marginals[1], ma[1], places=12)
        self.assertEqual(joint.marginals[2], 0.5)
        self.assertEqual(joint.marginals[3], 0.5)
        self.assertEqual(joint.components, ((0, 1), (2, 3)))

    def test_free_block_flips_counted_as_distinct_candidates(self):
        # one free block with a unique internal optimum: at low temperature
        # the two global flips both survive, so the marginal tends to 1/2
        # and the block contributes log 2 to the entropy.
        raw = [((0, None, None), 1), ((None, 0, 1), 2), ((None, 1, 1), 1)]
        got = phase_posterior(3, to_obs(raw), temperature=1e-3)
        self.assertAlmostEqual(got.marginals[1], 0.5, places=5)
        self.assertAlmostEqual(got.marginals[2], 0.5, places=5)
        self.assertAlmostEqual(got.entropy, math.log(2), places=5)


class NoEvidenceLocus(unittest.TestCase):
    def test_uncovered_locus_is_exactly_one_half(self):
        raw = [((0, 0, None), 2), ((1, 0, None), 1)]
        got = phase_posterior(3, to_obs(raw), temperature=0.9)
        self.assertEqual(got.marginals[2], 0.5)
        # the uncovered locus is its own free component
        self.assertIn((2,), got.components)


class SameOptimumDifferentUncertainty(unittest.TestCase):
    def test_gap_controls_posterior_sharpness(self):
        confident = [((0, 0), 5), ((0, 0), 5), ((1, 1), 1)]
        ambiguous = [((0, 0), 1), ((0, 0), 1), ((1, 1), 1)]
        for raw in (confident, ambiguous):
            result = phase_haplotypes(2, to_obs(raw))
            self.assertEqual(result.haplotype, (0, 0))
        sharp = phase_posterior(2, to_obs(confident), temperature=1.0)
        flat = phase_posterior(2, to_obs(ambiguous), temperature=1.0)
        self.assertLess(sharp.marginals[1], flat.marginals[1])
        self.assertLess(sharp.entropy, flat.entropy)


class TemperatureContract(unittest.TestCase):
    def test_invalid_temperatures_rejected(self):
        obs = to_obs([((0, 1), 1)])
        for bad in (0, -1, -0.5, 0.0, float("nan"), float("inf"),
                    float("-inf"), True, "1", None):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidPhasingInputError):
                    phase_posterior(2, obs, temperature=bad)

    def test_input_validation_still_applies(self):
        with self.assertRaises(InvalidPhasingInputError):
            phase_posterior(2, to_obs([((0, 1, 1), 1)]), temperature=1.0)
        with self.assertRaises(InvalidPhasingInputError):
            phase_posterior(2, to_obs([((0, 2), 1)]), temperature=1.0)
        with self.assertRaises(InvalidPhasingInputError):
            phase_posterior(0, [], temperature=1.0)

    def test_temperature_accepts_positive_int_and_float(self):
        obs = to_obs([((0, 1), 1)])
        a = phase_posterior(2, obs, temperature=2)
        b = phase_posterior(2, obs, temperature=2.0)
        self.assertEqual(a.temperature, 2.0)
        self.assertEqual(a.marginals, b.marginals)


if __name__ == "__main__":
    unittest.main()
