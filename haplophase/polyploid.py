"""Explicit ploidy-2/3 phasing by exact enumeration (weighted MEC).

Public API
----------
phase_polyploid(loci, observations, *, ploidy=2) -> PolyploidPhasingResult
    Exact minimum-cost phasing for an explicit ploidy of 2 or 3.

Model
-----
There are ``ploidy`` haplotype *copies*, each a binary string over the
``loci`` heterozygous sites.  Copies form an **unordered multiset**: two
solutions that differ only by permuting the copies are the same solution,
and (unlike the diploid entry point) the copies are *not* required to be
complementary.  Every fragment read is assigned to the single copy with
the fewest mismatched observed alleles; the integer mismatch cost of a
read is its ``Observation.flip_cost`` (cost of flipping one observed
allele), exactly as in the diploid model::

    cost(h_1..h_p) = sum_i  w_i * min_c  mismatches(read_i, h_c)

The search space is tiny by contract (loci <= 6, ploidy <= 3, reads <= 8),
so the solver enumerates every canonical candidate exactly; there is no
heuristic, no budget and no floating point anywhere.

Ties: every cost-optimal solution is returned in canonical order — copies
sorted lexicographically inside each solution, and the solutions
themselves sorted lexicographically.  ``haplotypes`` is the first
(lexicographically smallest) optimal solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Optional, Sequence

from .core import (
    InvalidPhasingInputError,
    Observation,
    _is_plain_int,
    _validate_inputs,
)

__all__ = [
    "MAX_PLOIDY_LOCI",
    "MAX_PLOIDY_OBSERVATIONS",
    "SUPPORTED_PLOIDIES",
    "EnumerationStats",
    "PolyploidReadAssignment",
    "PolyploidPhasingResult",
    "phase_polyploid",
]

MAX_PLOIDY_LOCI = 6
MAX_PLOIDY_OBSERVATIONS = 8
SUPPORTED_PLOIDIES = (2, 3)


@dataclass(frozen=True)
class EnumerationStats:
    """Observable state of the exhaustive enumeration."""

    ploidy: int
    candidates_evaluated: int
    completed: bool


@dataclass(frozen=True)
class PolyploidReadAssignment:
    """Assignment of one read to its cheapest haplotype copy.

    copy: index into ``PolyploidPhasingResult.haplotypes`` of the assigned
          copy.  When ``tied`` is True several copies attain the minimum
          cost and ``copy`` holds the deterministic tie-break (lowest
          index).
    tied: True iff more than one copy attains the minimum cost.
    mismatches: per-copy counts of mismatched observed alleles.
    costs: per-copy weighted costs (flip_cost * mismatches).
    cost: the minimum weighted cost actually charged for this read.
    flips: 0-based loci that would have to be flipped against the
           assigned copy.
    """

    index: int
    copy: int
    tied: bool
    mismatches: tuple[int, ...]
    costs: tuple[int, ...]
    cost: int
    flips: tuple[int, ...]


@dataclass(frozen=True)
class PolyploidPhasingResult:
    """Exact ploidy-2/3 phasing result.

    ploidy: the explicit ploidy the search ran with (2 or 3).
    haplotypes: canonical optimal solution — the lexicographically
        smallest sorted tuple of haplotype copies among all optima.
    optimal_solutions: every distinct optimal solution, each a sorted
        tuple of copies, the solutions themselves sorted
        lexicographically (canonical tie order).
    optimal_cost / second_best_cost: smallest cost and the smallest cost
        strictly greater than it (None if every candidate is optimal).
    gap: second_best_cost - optimal_cost, or None.  Costs are integers,
        so when present gap >= 1.
    assignments: per-read assignment for the canonical ``haplotypes``.
    search: enumeration state.
    """

    ploidy: int
    haplotypes: tuple[tuple[int, ...], ...]
    optimal_solutions: tuple[tuple[tuple[int, ...], ...], ...]
    optimal_cost: int
    second_best_cost: Optional[int]
    gap: Optional[int]
    assignments: tuple[PolyploidReadAssignment, ...]
    search: EnumerationStats


def _validate_polyploid_inputs(
    loci: object,
    observations: object,
    ploidy: object,
) -> tuple[int, list[Observation], int]:
    if not _is_plain_int(ploidy):
        kind = "bool" if isinstance(ploidy, bool) else type(ploidy).__name__
        raise InvalidPhasingInputError(
            f"ploidy: expected int, got {kind}; bool is not accepted"
        )
    if ploidy not in SUPPORTED_PLOIDIES:
        raise InvalidPhasingInputError(
            f"ploidy: must be one of {SUPPORTED_PLOIDIES}, got {ploidy}"
        )

    if not _is_plain_int(loci):
        kind = "bool" if isinstance(loci, bool) else type(loci).__name__
        raise InvalidPhasingInputError(
            f"loci: expected int, got {kind}; bool is not accepted"
        )
    if not 1 <= loci <= MAX_PLOIDY_LOCI:
        raise InvalidPhasingInputError(
            f"loci: must satisfy 1 <= loci <= {MAX_PLOIDY_LOCI} "
            f"for the polyploid entry, got {loci}"
        )

    if not isinstance(observations, (list, tuple)):
        raise InvalidPhasingInputError(
            "observations: expected a list or tuple of Observation, got "
            + type(observations).__name__
        )
    if len(observations) > MAX_PLOIDY_OBSERVATIONS:
        raise InvalidPhasingInputError(
            f"observations: at most {MAX_PLOIDY_OBSERVATIONS} reads "
            f"for the polyploid entry, got {len(observations)}"
        )

    # Per-read allele/flip_cost validation is identical to the diploid
    # contract; delegate so both entries reject the same bad reads with
    # the same locatable messages.  Validation precedes any computation,
    # so a rejected call leaves every passed object untouched.
    _, checked = _validate_inputs(loci, observations, None)
    return loci, checked, ploidy  # type: ignore[return-value]


def _to_haplotype(s: int, loci: int) -> tuple[int, ...]:
    """Bit k of the integer code is the allele at locus k."""
    return tuple((s >> k) & 1 for k in range(loci))


def phase_polyploid(
    loci: int,
    observations: Sequence[Observation],
    *,
    ploidy: int = 2,
) -> PolyploidPhasingResult:
    """Phase with an explicit ploidy of 2 or 3 by exact enumeration.

    Raises InvalidPhasingInputError for contract violations (with a
    message pinpointing the offending argument).  The candidate space is
    bounded by contract (loci <= 6, ploidy <= 3), so the enumeration
    always completes; there is no search budget and no budget exception.
    """
    L, obs_list, p = _validate_polyploid_inputs(loci, observations, ploidy)
    n = len(obs_list)
    weights = [obs.flip_cost for obs in obs_list]

    # Weighted mismatch cost of read i against every possible copy string.
    codes = range(1 << L)
    mis: list[list[int]] = []
    for i, obs in enumerate(obs_list):
        row = [0] * (1 << L)
        for s in codes:
            m = 0
            for k, allele in enumerate(obs.alleles):
                if allele is not None and allele != ((s >> k) & 1):
                    m += 1
            row[s] = weights[i] * m
        mis.append(row)

    best: Optional[int] = None
    second: Optional[int] = None
    optima: list[tuple[int, ...]] = []
    candidates_evaluated = 0

    # combinations_with_replacement enumerates each unordered multiset of
    # copies exactly once, so no solution is double counted.
    for combo in combinations_with_replacement(codes, p):
        candidates_evaluated += 1
        total = 0
        for i in range(n):
            row = mis[i]
            total += min(row[s] for s in combo)
        if best is None or total < best:
            if best is not None:
                second = best if second is None else min(second, best)
            best = total
            optima = [combo]
        elif total == best:
            optima.append(combo)
        elif second is None or total < second:
            second = total

    # Canonical tie order: copies sorted inside each solution, solutions
    # sorted lexicographically.  Sorting the decoded bit tuples (not the
    # integer codes) is what makes the order lexicographic on haplotypes.
    solutions = tuple(
        sorted(
            tuple(sorted(_to_haplotype(s, L) for s in combo)) for combo in optima
        )
    )
    haplotypes = solutions[0]

    assignments: list[PolyploidReadAssignment] = []
    for i, obs in enumerate(obs_list):
        counts = tuple(
            sum(
                1
                for k, allele in enumerate(obs.alleles)
                if allele is not None and allele != hap[k]
            )
            for hap in haplotypes
        )
        costs = tuple(weights[i] * c for c in counts)
        cheapest = min(costs)
        copy = costs.index(cheapest)
        flips = tuple(
            k
            for k, allele in enumerate(obs.alleles)
            if allele is not None and allele != haplotypes[copy][k]
        )
        assignments.append(
            PolyploidReadAssignment(
                index=i,
                copy=copy,
                tied=costs.count(cheapest) > 1,
                mismatches=counts,
                costs=costs,
                cost=cheapest,
                flips=flips,
            )
        )

    return PolyploidPhasingResult(
        ploidy=p,
        haplotypes=haplotypes,
        optimal_solutions=solutions,
        optimal_cost=best,  # type: ignore[arg-type]
        second_best_cost=second,
        gap=None if second is None else second - best,  # type: ignore[operator]
        assignments=tuple(assignments),
        search=EnumerationStats(
            ploidy=p,
            candidates_evaluated=candidates_evaluated,
            completed=True,
        ),
    )
