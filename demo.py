#!/usr/bin/env python3
"""Deterministic offline demonstration for haplophase.

Run:  python demo.py

It performs real branch-and-bound computations (nothing is pre-recorded and
nothing sleeps) and prints:
  A. a small hand-built instance with a disconnected fragment, a tied read
     and a fully missing read;
  B. a larger deterministically generated instance with search-state counters;
  C. two boundary cases the contract actually triggers:
       - a locatable input rejection (NaN allele), plus a bool-vs-int one;
       - search-budget exhaustion, kept distinct from infeasibility.
"""

from __future__ import annotations

import math
import random
import time

from haplophase import (
    InvalidPhasingInputError,
    Observation,
    SearchBudgetExhaustedError,
    phase_haplotypes,
    phase_haplotypes_top_k,
)


def show(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show_result(result) -> None:
    h = "".join(map(str, result.haplotype))
    c = "".join(map(str, result.complement))
    print(f"optimal h           : {h}   (h[0] fixed to 0)")
    print(f"complement 1 - h    : {c}")
    print(f"optimal total cost  : {result.optimal_cost}")
    print(
        "second-best gap     : "
        + (
            f"next distinct h costs {result.second_best_cost} "
            f"(gap = {result.gap})"
            if result.gap is not None
            else "none - every admissible h is optimal"
        )
    )
    print(f"optimal h count     : {len(result.optimal_haplotypes)}")
    if len(result.optimal_haplotypes) > 1:
        print("all optimal h       : " + ", ".join(
            "".join(map(str, x)) for x in result.optimal_haplotypes
        ))
    if result.free_components:
        print(
            "ambiguous components: "
            + "; ".join(
                "locus "
                + "/".join(str(k + 1) for k in comp)
                + " (flipping it leaves the cost unchanged)"
                for comp in result.free_components
            )
        )
    print("per-read assignment (0 -> h, 1 -> 1-h, = marks a tie):")
    for a in result.assignments:
        mark = " =TIED" if a.tied else ""
        flips = (
            "no flip"
            if not a.flips_applied
            else "flip locus " + ", ".join(str(k + 1) for k in a.flips_applied)
        )
        print(
            f"  read {a.index}: -> {a.assigned}{mark}  "
            f"cost {a.cost_to_h0} vs {a.cost_to_h1}  ({flips})"
        )
    s = result.search
    print(
        "search state        : "
        f"nodes opened {s.nodes_opened}, leaves evaluated {s.leaves_evaluated}, "
        f"branches pruned {s.branches_pruned}, incumbent updates "
        f"{s.incumbent_updates}, seed costs {list(s.seed_costs)}"
    )
    print(f"nodes by depth      : {list(s.nodes_by_depth)}")
    print(f"pruned by depth     : {list(s.pruned_by_depth)}")


def main() -> None:
    # ------------------------------------------------------------------ A
    show("A. Small instance: weighted reads, a disconnected fragment, ambiguity")
    observations = [
        Observation((0, 1, 0, 1, None, None), 1),   # cheap read, one error
        Observation((0, 1, 1, 1, None, None), 5),   # heavy read anchors sites 1-4
        Observation((None, 1, 0, None, None, None), 2),  # 2-site read, will tie
        Observation((None, None, None, None, 0, 0), 3),  # disconnected fragment
        Observation((None, None, None, None, None, None), 4),  # all missing
    ]
    t0 = time.perf_counter()
    result = phase_haplotypes(6, observations)
    dt = time.perf_counter() - t0
    show_result(result)
    print(f"computed in {dt * 1000:.1f} ms (exact integer arithmetic)")
    print(
        "Note: the greedy seeds cost 9 and 12; branch and bound proves the "
        "true optimum is 3 - per-site voting cannot produce this result."
    )

    # ------------------------------------------------------------------ B
    show("B. Larger deterministically generated instance (15 loci, 40 reads)")
    rng = random.Random(20260923)
    truth = tuple(rng.randint(0, 1) for _ in range(15))
    truth = (0,) + truth[1:]
    reads = []
    for _ in range(40):
        lo = rng.randrange(0, 12)
        hi = min(15, lo + rng.randint(2, 6))
        allele_seq = []
        mutated = rng.random() < 0.25  # a quarter of reads carry one error
        for k in range(15):
            if lo <= k < hi:
                bit = truth[k]
                if mutated and k == lo:
                    bit = 1 - bit
                allele_seq.append(bit)
            else:
                allele_seq.append(None)
        reads.append(Observation(tuple(allele_seq), rng.randint(1, 9)))
    t0 = time.perf_counter()
    result_b = phase_haplotypes(15, reads)
    dt = time.perf_counter() - t0
    show_result(result_b)
    print(f"computed in {dt * 1000:.1f} ms over a space of 2**14 candidates")

    # ------------------------------------------------------------------ C
    show("C1. Rejected boundary: NaN allele (locatable error, no silent fix)")
    bad = [Observation((0, math.nan, 1), 1)]
    try:
        phase_haplotypes(3, bad)
    except InvalidPhasingInputError as err:
        print(f"raised {type(err).__name__}: {err}")

    show("C2. Rejected boundary: bool is not accepted as an integer parameter")
    try:
        phase_haplotypes(True, [])  # type: ignore[arg-type]
    except InvalidPhasingInputError as err:
        print(f"raised {type(err).__name__}: {err}")

    show("C3. Search-budget exhaustion is separate from 'proved impossible'")
    rng2 = random.Random(99)
    many = [
        Observation(
            tuple(rng2.choice((0, 1)) if rng2.random() < 0.8 else None
                  for _ in range(18)),
            rng2.randint(1, 9),
        )
        for _ in range(60)
    ]
    try:
        phase_haplotypes(18, many, max_nodes=500)
    except SearchBudgetExhaustedError as err:
        print(f"raised {type(err).__name__}: {err}")
        print(
            f"  opened {err.nodes_opened} nodes; feasible-but-unproven "
            f"incumbent h={err.best_haplotype} cost={err.best_cost}"
        )
        print("  (every validated instance is feasible, so this is not a "
              "'no solution' verdict: rerun without max_nodes to prove it.)")

    # ------------------------------------------------------------------ D
    show("D. Top-K listing: ranked distinct candidates (K=5, 8 loci)")
    rng3 = random.Random(20260925)
    truth3 = (0, 1, 1, 0, 0, 1, 0, 1)
    reads3 = []
    for _ in range(14):
        lo = rng3.randrange(0, 5)
        hi = min(8, lo + rng3.randint(2, 5))
        row = []
        for kpos in range(8):
            if lo <= kpos < hi:
                bit = truth3[kpos]
                if rng3.random() < 0.15:
                    bit = 1 - bit  # sequencing error
                row.append(bit)
            else:
                row.append(None)
        reads3.append(Observation(tuple(row), rng3.randint(1, 4)))
    t0 = time.perf_counter()
    top = phase_haplotypes_top_k(8, reads3, 5)
    dt = time.perf_counter() - t0
    print(f"complete={top.complete}  exhaustive={top.exhaustive}  "
          f"solution_count={top.solution_count}")
    for cand in top.candidates:
        hstr = "".join(map(str, cand.haplotype))
        moved = sorted(
            {site for a in cand.assignments for site in a.flips_applied}
        )
        flips = "none" if not moved else ",".join(str(s + 1) for s in moved)
        print(f"  rank {cand.rank}: h={hstr}  cost={cand.cost}  "
              f"flip loci: {flips}")
    s = top.search
    print(f"search: nodes {s.nodes_opened}, leaves {s.leaves_evaluated}, "
          f"pruned {s.branches_pruned}  ({dt * 1000:.1f} ms)")

    show("D2. Top-K with a tiny node budget: unproven ranks, not a gap")
    partial = phase_haplotypes_top_k(8, reads3, 5, max_nodes=6)
    print(f"complete={partial.complete}  "
          f"remaining_lower_bound={partial.remaining_lower_bound}")
    for cand in partial.candidates:
        print(f"  (unproven) rank {cand.rank}: h="
              f"{''.join(map(str, cand.haplotype))}  cost={cand.cost}")
    print("  -> budget stop returns the best candidates found so far; the "
          "lower bound covers every unexplored branch and is NOT the "
          "proven second-best gap of phase_haplotypes.")

    print()
    print("demo finished.")


if __name__ == "__main__":
    main()
