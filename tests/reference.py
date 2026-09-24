"""Independent reference and fixtures for the phasing tests.

The brute-force reference below deliberately operates on raw tuples and never
imports or calls the package under test to obtain expected values. It exhausts
all 2**(loci-1) canonical candidates (loci <= 10) with plain itertools.
"""

from __future__ import annotations

import random
from itertools import product

MAX_REFERENCE_LOCI = 10


def reference_cost(h, raw_reads):
    """Weighted MEC cost of a *complete* candidate h. Independent formula."""
    total = 0
    for alleles, weight in raw_reads:
        covered = 0
        mismatches = 0
        for allele, bit in zip(alleles, h):
            if allele is not None:
                covered += 1
                if allele != bit:
                    mismatches += 1
        total += weight * min(mismatches, covered - mismatches)
    return total


def brute_force(loci, raw_reads):
    """Return (best_cost, sorted tuple of all optimal h tuples, second_cost)."""
    costs = {}
    for bits in product((0, 1), repeat=loci):
        h = bits  # candidates with leading 0 are selected below
        if h[0] != 0:
            continue
        costs[h] = reference_cost(h, raw_reads)
    best = min(costs.values())
    optima = tuple(sorted(h for h, c in costs.items() if c == best))
    distinct = sorted({c for c in costs.values() if c > best})
    second = distinct[0] if distinct else None
    return best, optima, second


def reference_assignment(h, alleles, weight):
    """Independent expected (assigned, tied, c0, c1, flips0, flips1)."""
    flips0 = tuple(k for k, a in enumerate(alleles) if a is not None and a != h[k])
    flips1 = tuple(k for k, a in enumerate(alleles) if a is not None and a == h[k])
    c0 = weight * len(flips0)
    c1 = weight * len(flips1)
    tied = c0 == c1
    assigned = 0 if c0 <= c1 else 1
    return assigned, tied, c0, c1, flips0, flips1


def random_instance(rng, loci):
    """Deterministic random valid instance on raw tuples.

    Fully-missing reads are allowed (they must be neutral to the objective)."""
    n = rng.randint(1, 8)
    reads = []
    for _ in range(n):
        alleles = tuple(
            None if rng.random() < 0.35 else rng.randint(0, 1) for _ in range(loci)
        )
        reads.append((alleles, rng.randint(1, 7)))
    return reads


def random_cases(seed=20260923, count=120):
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        loci = rng.randint(1, MAX_REFERENCE_LOCI)
        cases.append((loci, random_instance(rng, loci)))
    return cases


# ---------------------------------------------------------------------------
# Polyploid reference (ploidy 2 or 3): independent of the package under test.
# It enumerates the FULL symmetric space of ordered copy tuples with plain
# itertools.product and normalises by sorting, instead of enumerating
# canonical combinations directly.
# ---------------------------------------------------------------------------


def reference_cost_copies(haplotypes, raw_reads):
    """Cost of one solution: each read pays weight * min mismatches to a copy."""
    total = 0
    for alleles, weight in raw_reads:
        cheapest = None
        for h in haplotypes:
            mismatches = sum(
                1 for a, b in zip(alleles, h) if a is not None and a != b
            )
            cheapest = mismatches if cheapest is None else min(cheapest, mismatches)
        total += weight * cheapest
    return total


def brute_force_polyploid(loci, ploidy, raw_reads):
    """Return (best_cost, all optimal sorted copy-tuples, second_cost).

    Enumerates every ordered ploidy-tuple of haplotypes, canonicalises by
    sorting, and deduplicates. Only feasible for small loci (the space is
    2**(loci*ploidy) ordered tuples before dedup).
    """
    all_h = list(product((0, 1), repeat=loci))
    costs = {}
    for copies in product(all_h, repeat=ploidy):
        key = tuple(sorted(copies))
        if key not in costs:
            costs[key] = reference_cost_copies(key, raw_reads)
    best = min(costs.values())
    optima = tuple(sorted(h for h, c in costs.items() if c == best))
    distinct = sorted({c for c in costs.values() if c > best})
    second = distinct[0] if distinct else None
    return best, optima, second


def reference_copy_assignment(haplotypes, alleles, weight):
    """Independent expected (assigned, tied, costs, flips) for one read."""
    costs = []
    flips = []
    for h in haplotypes:
        flip_loci = tuple(
            k for k, a in enumerate(alleles) if a is not None and a != h[k]
        )
        flips.append(flip_loci)
        costs.append(weight * len(flip_loci))
    cheapest = min(costs)
    assigned = costs.index(cheapest)
    tied = costs.count(cheapest) > 1
    return assigned, tied, tuple(costs), tuple(flips)


def random_polyploid_cases(seed=20260924):
    """Deterministic random instances sized so the full-product reference
    stays fast: ploidy 2 up to 6 loci, ploidy 3 up to 4 loci, plus a few
    larger triploid cases (one at the 6-locus interface maximum)."""
    rng = random.Random(seed)
    cases = []
    for _ in range(40):
        loci = rng.randint(1, 6)
        cases.append((loci, 2, random_instance(rng, loci)))
    for _ in range(40):
        loci = rng.randint(1, 4)
        cases.append((loci, 3, random_instance(rng, loci)))
    for loci in (5, 5, 6):
        cases.append((loci, 3, random_instance(rng, loci)))
    return cases
