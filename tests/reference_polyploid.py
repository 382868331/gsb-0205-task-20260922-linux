"""Independent reference for the ploidy-2/3 phasing tests.

The brute force below deliberately operates on raw tuples and never
imports or calls the package under test to obtain expected values.  It
exhausts every ordered ploidy-tuple of copy strings with plain
itertools.product, canonicalises each to a sorted multiset, and scores it
with its own cost formula (per-read integer mismatch cost = flip_cost
times the number of mismatched observed alleles against the cheapest
copy).
"""

from __future__ import annotations

import random
from itertools import product


def decode(code, loci):
    """Bit k of the integer code is the allele at locus k."""
    return tuple((code >> k) & 1 for k in range(loci))


def reference_read_cost(alleles, weight, copies):
    """Cheapest weighted mismatch cost of one read against the copies."""
    cheapest = None
    for copy in copies:
        mismatches = 0
        for k, allele in enumerate(alleles):
            if allele is not None and allele != copy[k]:
                mismatches += 1
        cost = weight * mismatches
        if cheapest is None or cost < cheapest:
            cheapest = cost
    return cheapest


def brute_force_polyploid(loci, raw_reads, ploidy):
    """Return (best_cost, sorted tuple of all optimal canonical solutions,
    second_cost).  Each solution is a sorted tuple of haplotype tuples."""
    costs = {}
    for combo in product(range(1 << loci), repeat=ploidy):
        solution = tuple(sorted(decode(s, loci) for s in combo))
        if solution in costs:
            continue
        total = 0
        for alleles, weight in raw_reads:
            total += reference_read_cost(alleles, weight, solution)
        costs[solution] = total
    best = min(costs.values())
    optima = tuple(sorted(s for s, c in costs.items() if c == best))
    distinct = sorted({c for c in costs.values() if c > best})
    second = distinct[0] if distinct else None
    return best, optima, second


def reference_polyploid_assignment(haplotypes, alleles, weight):
    """Independent expected (copy, tied, mismatches, costs, flips)."""
    counts = tuple(
        sum(1 for k, a in enumerate(alleles) if a is not None and a != hap[k])
        for hap in haplotypes
    )
    costs = tuple(weight * c for c in counts)
    cheapest = min(costs)
    copy = costs.index(cheapest)
    tied = costs.count(cheapest) > 1
    flips = tuple(
        k for k, a in enumerate(alleles) if a is not None and a != haplotypes[copy][k]
    )
    return copy, tied, counts, costs, flips


def random_polyploid_instance(rng, loci, ploidy):
    """Deterministic random valid instance on raw tuples.

    Fully-missing reads are allowed (they must be neutral to the
    objective)."""
    n = rng.randint(0, 6)
    reads = []
    for _ in range(n):
        alleles = tuple(
            None if rng.random() < 0.3 else rng.randint(0, 1) for _ in range(loci)
        )
        reads.append((alleles, rng.randint(1, 5)))
    return reads


def random_polyploid_cases(seed=20260924, count=60, max_loci=4):
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        loci = rng.randint(1, max_loci)
        ploidy = rng.choice((2, 3))
        cases.append((loci, ploidy, random_polyploid_instance(rng, loci, ploidy)))
    return cases
