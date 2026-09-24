#!/usr/bin/env python3
"""Deterministic offline demonstration for haplophase.

Run:  python demo.py

It performs real branch-and-bound and exhaustive-enumeration computations
(nothing is pre-recorded and nothing sleeps) and prints:
  A. a small hand-built instance with a disconnected fragment, a tied read
     and a fully missing read;
  B. a larger deterministically generated instance with search-state counters;
  C. two boundary cases the contract actually triggers:
       - a locatable input rejection (NaN allele), plus a bool-vs-int one;
       - search-budget exhaustion, kept distinct from infeasibility;
  D. the explicit-ploidy entry: a triploid (ploidy=3) instance with
     missing sites, a tied read and canonically sorted tied optima;
  E. the polyploid boundary size (6 loci, 8 reads, ploidy 3);
  F. two real polyploid rejections (unsupported ploidy, too many reads).
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
    phase_polyploid,
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


def show_polyploid(result) -> None:
    print(f"ploidy              : {result.ploidy}")
    print(
        "canonical copies    : "
        + " | ".join("".join(map(str, h)) for h in result.haplotypes)
    )
    print(f"optimal total cost  : {result.optimal_cost}")
    print(
        "second-best gap     : "
        + (
            f"next distinct solution costs {result.second_best_cost} "
            f"(gap = {result.gap})"
            if result.gap is not None
            else "none - every candidate is optimal"
        )
    )
    print(f"optimal solutions   : {len(result.optimal_solutions)}")
    if 1 < len(result.optimal_solutions) <= 8:
        for sol in result.optimal_solutions:
            print("  " + " | ".join("".join(map(str, h)) for h in sol))
    print("per-read assignment (-> copy index, = marks a tie):")
    for a in result.assignments:
        mark = " =TIED" if a.tied else ""
        flips = (
            "no flip"
            if not a.flips
            else "flip locus " + ", ".join(str(k + 1) for k in a.flips)
        )
        print(
            f"  read {a.index}: -> copy {a.copy}{mark}  "
            f"costs per copy {list(a.costs)}  ({flips})"
        )
    s = result.search
    print(
        f"enumeration         : {s.candidates_evaluated} candidate "
        f"multisets evaluated, completed={s.completed}"
    )


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
    show("D. Explicit ploidy 3: missing sites, a tied read, tied optima")
    observations_d = [
        Observation((0, 0, 0, None), 2),        # matches copy A
        Observation((0, 1, 1, 0), 3),           # matches copy B
        Observation((1, 0, 1, 1), 1),           # matches copy C
        Observation((None, None, None, 0), 1),  # one-site read, will tie
        Observation((None, None, None, None), 5),  # all missing, neutral
    ]
    t0 = time.perf_counter()
    result_d = phase_polyploid(4, observations_d, ploidy=3)
    dt = time.perf_counter() - t0
    show_polyploid(result_d)
    print(f"computed in {dt * 1000:.1f} ms (exact enumeration, integer costs)")

    # ------------------------------------------------------------------ E
    show("E. Polyploid boundary size: 6 loci, 8 reads, ploidy 3")
    rng3 = random.Random(20260924)
    copies = [
        tuple(rng3.randint(0, 1) for _ in range(6)) for _ in range(3)
    ]
    reads_e = []
    for r in range(8):
        copy = copies[r % 3]
        reads_e.append(
            Observation(
                tuple(
                    None if rng3.random() < 0.25 else bit for bit in copy
                ),
                rng3.randint(1, 4),
            )
        )
    t0 = time.perf_counter()
    result_e = phase_polyploid(6, reads_e, ploidy=3)
    dt = time.perf_counter() - t0
    show_polyploid(result_e)
    print(f"computed in {dt * 1000:.1f} ms over "
          f"{result_e.search.candidates_evaluated} candidate multisets")

    # ------------------------------------------------------------------ F
    show("F1. Rejected boundary: ploidy 4 is outside the contract")
    try:
        phase_polyploid(4, observations_d, ploidy=4)
    except InvalidPhasingInputError as err:
        print(f"raised {type(err).__name__}: {err}")

    show("F2. Rejected boundary: 9 reads exceed the polyploid limit of 8")
    too_many = [Observation((0, 0), 1)] * 9
    try:
        phase_polyploid(2, too_many, ploidy=2)
    except InvalidPhasingInputError as err:
        print(f"raised {type(err).__name__}: {err}")

    print()
    print("demo finished.")


if __name__ == "__main__":
    main()
