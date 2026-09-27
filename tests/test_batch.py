"""Tests for the batched optimal-haplotype enumeration API.

Covers: exact agreement with the one-shot solver (no duplicates, no
omissions, stable lexicographic order, symmetry dedup), cursor
invalidation on any input/option change, rejection of malformed cursors
and bad batch sizes, the budget-exhaustion/incompleteness split, and
that huge optimal sets are streamed without being materialized.
"""

from __future__ import annotations

import unittest

from haplophase import (
    InvalidPhasingInputError,
    Observation,
    SearchBudgetExhaustedError,
    optimal_haplotype_batch,
    phase_haplotypes,
)

from reference import random_cases


def to_obs(raw_reads):
    return [Observation(alleles=a, flip_cost=w) for a, w in raw_reads]


def collect_all(loci, observations, batch_size, max_nodes=None):
    """Drain a batch sequence; return (cost, haplotypes, batch_count)."""
    cursor = None
    cost = None
    seen = []
    batches = 0
    while True:
        batch = optimal_haplotype_batch(
            loci,
            observations,
            batch_size=batch_size,
            max_nodes=max_nodes,
            cursor=cursor,
        )
        batches += 1
        if cost is None:
            cost = batch.optimal_cost
        self_cost = batch.optimal_cost
        assert self_cost == cost, "cost must be stable across a sequence"
        assert len(batch.haplotypes) <= batch_size
        seen.extend(batch.haplotypes)
        if batch.exhausted:
            assert batch.next_cursor is None
            assert batch.search.completed
            return cost, seen, batches
        assert batch.next_cursor is not None
        assert not batch.search.completed
        assert len(batch.haplotypes) == batch_size  # never a short non-final batch
        cursor = batch.next_cursor


class BatchedEnumerationMatchesOneShot(unittest.TestCase):
    def test_random_cases_all_batch_sizes(self):
        for loci, raw in random_cases(seed=20260927, count=40):
            observations = to_obs(raw)
            expected = phase_haplotypes(loci, observations)
            for batch_size in (1, 2, 3, 5, 1000):
                with self.subTest(loci=loci, raw=raw, batch_size=batch_size):
                    cost, seen, _ = collect_all(
                        loci, observations, batch_size
                    )
                    self.assertEqual(cost, expected.optimal_cost)
                    # identical content, identical order, no dup/omission
                    self.assertEqual(tuple(seen), expected.optimal_haplotypes)
                    # stable ascending lexicographic order, h[0] == 0
                    self.assertEqual(tuple(seen), tuple(sorted(seen)))
                    self.assertEqual(len(seen), len(set(seen)))
                    self.assertTrue(all(h[0] == 0 for h in seen))

    def test_batch_sequence_is_deterministic(self):
        loci, raw = random_cases(seed=11, count=1)[0]
        observations = to_obs(raw)
        first = collect_all(loci, observations, 2)
        second = collect_all(loci, observations, 2)
        self.assertEqual(first, second)

    def test_first_batch_establishes_cost_before_full_listing(self):
        # a case with several optima: cost is known after batch one, while
        # the enumeration is explicitly not finished yet
        observations = [
            Observation((0, 1, None, None), 2),
            Observation((None, None, 0, 1), 3),
        ]
        batch = optimal_haplotype_batch(4, observations, batch_size=1)
        self.assertFalse(batch.exhausted)
        self.assertIsNotNone(batch.next_cursor)
        self.assertEqual(
            batch.optimal_cost, phase_haplotypes(4, observations).optimal_cost
        )
        self.assertEqual(len(batch.haplotypes), 1)


class CursorContract(unittest.TestCase):
    def setUp(self):
        self.loci = 4
        self.observations = [
            Observation((0, 1, None, None), 2),
            Observation((None, None, 0, 1), 3),
        ]
        first = optimal_haplotype_batch(
            self.loci, self.observations, batch_size=1
        )
        self.assertFalse(first.exhausted)
        self.cursor = first.next_cursor

    def test_changed_observations_invalidate_cursor(self):
        changed = self.observations + [
            Observation((1, None, None, 0), 1)
        ]
        with self.assertRaises(InvalidPhasingInputError):
            optimal_haplotype_batch(
                self.loci, changed, batch_size=1, cursor=self.cursor
            )

    def test_changed_loci_invalidate_cursor(self):
        with self.assertRaises(InvalidPhasingInputError):
            optimal_haplotype_batch(
                self.loci + 1,
                self.observations + [Observation((0, 0, 0, 0, 0), 1)],
                batch_size=1,
                cursor=self.cursor,
            )

    def test_changed_options_invalidate_cursor(self):
        with self.assertRaises(InvalidPhasingInputError):
            optimal_haplotype_batch(
                self.loci, self.observations, batch_size=2, cursor=self.cursor
            )

    def test_budget_change_keeps_cursor_valid(self):
        # max_nodes is a per-call safety budget, not part of the query
        batch = optimal_haplotype_batch(
            self.loci,
            self.observations,
            batch_size=1,
            max_nodes=10**9,
            cursor=self.cursor,
        )
        self.assertEqual(len(batch.haplotypes), 1)

    def test_malformed_cursors_rejected(self):
        for bad in (
            123,
            None.__class__,  # not a str instance
            "",
            "hpb1",
            "hpb1:aa:bb",
            "hpb2:aa:0:00",
            "hpb1:aa:notanint:00",
            "hpb1:aa:0:1z",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidPhasingInputError):
                    optimal_haplotype_batch(
                        self.loci, self.observations, batch_size=1, cursor=bad
                    )

    def test_same_inputs_cursor_still_valid(self):
        # sanity: the untouched cursor continues the sequence correctly
        batch = optimal_haplotype_batch(
            self.loci, self.observations, batch_size=1, cursor=self.cursor
        )
        self.assertEqual(len(batch.haplotypes), 1)


class BatchSizeValidation(unittest.TestCase):
    def test_invalid_batch_sizes_rejected(self):
        for bad in (0, -3, True, 1.5, "2"):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidPhasingInputError):
                    optimal_haplotype_batch(3, [], batch_size=bad)


class BudgetAndCompleteness(unittest.TestCase):
    def test_budget_exhaustion_is_not_reported_as_complete(self):
        loci, raw = random_cases(seed=5, count=1)[0]
        observations = to_obs(raw)
        with self.assertRaises(SearchBudgetExhaustedError) as ctx:
            optimal_haplotype_batch(loci, observations, batch_size=2, max_nodes=1)
        # the attached incumbent is feasible but unproven: never below the
        # true optimum, and the error is distinct from a completed listing
        proven = phase_haplotypes(loci, observations)
        self.assertGreaterEqual(ctx.exception.best_cost, proven.optimal_cost)

    def test_budget_exhaustion_mid_sequence(self):
        # disconnected fragment -> several optima, so batch one cannot be last
        observations = [
            Observation((0, 1, None, None), 2),
            Observation((None, None, 0, 1), 3),
        ]
        full = phase_haplotypes(4, observations)
        self.assertGreater(len(full.optimal_haplotypes), 1)
        first = optimal_haplotype_batch(4, observations, batch_size=1)
        self.assertFalse(first.exhausted)
        with self.assertRaises(SearchBudgetExhaustedError):
            optimal_haplotype_batch(
                4, observations, batch_size=1, max_nodes=1,
                cursor=first.next_cursor,
            )

    def test_unfinished_sequence_never_claims_exhausted(self):
        loci, raw = random_cases(seed=8, count=1)[0]
        observations = to_obs(raw)
        total = len(phase_haplotypes(loci, observations).optimal_haplotypes)
        cursor = None
        delivered = 0
        while True:
            batch = optimal_haplotype_batch(
                loci, observations, batch_size=1, cursor=cursor
            )
            delivered += len(batch.haplotypes)
            if delivered < total:
                self.assertFalse(batch.exhausted)
                self.assertFalse(batch.search.completed)
                cursor = batch.next_cursor
            else:
                self.assertTrue(batch.exhausted)
                break


class StreamingLargeOptimalSets(unittest.TestCase):
    def test_huge_optimal_set_is_not_materialized(self):
        # 18 loci, no reads: every one of the 2**17 candidates is optimal.
        # The one-shot API would have to materialize all 131072 of them;
        # the batched API must stream them in tiny batches.
        batch = optimal_haplotype_batch(18, [], batch_size=4)
        self.assertEqual(batch.optimal_cost, 0)
        self.assertFalse(batch.exhausted)
        self.assertEqual(
            batch.haplotypes,
            (
                (0,) * 18,
                (0,) * 17 + (1,),
                (0,) * 16 + (1, 0),
                (0,) * 16 + (1, 1),
            ),
        )
        nxt = optimal_haplotype_batch(
            18, [], batch_size=4, cursor=batch.next_cursor
        )
        self.assertEqual(nxt.haplotypes[0], (0,) * 15 + (1, 0, 0))

    def test_full_drain_of_medium_optimal_set(self):
        # 8 loci, no reads -> 2**7 == 128 optima, drained 5 at a time
        cost, seen, batches = collect_all(8, [], 5)
        self.assertEqual(cost, 0)
        self.assertEqual(len(seen), 128)
        self.assertEqual(tuple(seen), tuple(sorted(set(seen))))
        self.assertEqual(batches, 26)  # 25 full batches + final batch of 3


if __name__ == "__main__":
    unittest.main()
