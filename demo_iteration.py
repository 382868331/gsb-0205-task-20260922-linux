#!/usr/bin/env python3
"""Offline deterministic demo for the Boltzmann phasing posterior.

Run: python demo_iteration.py
"""

from __future__ import annotations

import random
import time

from haplophase import (
    InvalidPhasingInputError,
    Observation,
    boltzmann_posterior,
    phase_haplotypes,
)


def observations(raw):
    return [Observation(alleles, weight) for alleles, weight in raw]


def show_posterior(title, loci, raw, temperature):
    print(f"\n{title}")
    print("-" * len(title))
    t0 = time.perf_counter()
    optimal = phase_haplotypes(loci, observations(raw))
    posterior = boltzmann_posterior(loci, observations(raw), temperature)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    marginals = ", ".join(
        f"{site + 1}:{value:.4f}" for site, value in enumerate(posterior.marginals)
    )
    print(f"optimal h              : {''.join(map(str, optimal.haplotype))}")
    print(f"P(h[site] = 1)         : {marginals}")
    print(f"log Z                  : {posterior.log_normalizer:.8f}")
    print(f"entropy (nats)         : {posterior.entropy:.8f}")
    print(
        "canonical states       : "
        f"{posterior.canonical_configuration_count} ("
        + ", ".join(
            f"component {i}:{part.configuration_count}"
            for i, part in enumerate(posterior.components)
        )
        + ")"
    )
    print(f"computed in            : {elapsed_ms:.2f} ms")
    return posterior


def main():
    print("Boltzmann posterior over the existing weighted-MEC cost")
    print("Temperature is a user-selected exploration scale, not a calibrated model.")

    sharp_raw = [
        ((0, 0, 0, None), 10),
        ((0, 0, 1, None), 10),
        ((0, 1, 0, None), 10),
        ((0, 0, 0, None), 20),
    ]
    broad_raw = [
        ((0, 0, 0, None), 1),
    ]
    sharp = show_posterior(
        "A1. Same MAP optimum, strong evidence",
        4,
        sharp_raw,
        0.5,
    )
    broad = show_posterior(
        "A2. Same MAP optimum, weak evidence",
        4,
        broad_raw,
        0.5,
    )
    print(
        "\nComparison             : MAP h is identical, but entropy changes "
        f"{sharp.entropy:.4f} vs {broad.entropy:.4f}."
    )

    disconnected_raw = [
        ((0, 1, None, None, None), 1),
        ((None, None, 0, 1, None), 1),
    ]
    show_posterior(
        "B. Independent components plus one locus with no evidence",
        5,
        disconnected_raw,
        0.75,
    )
    print(
        "Counting rule          : component containing site 1 has 2 states; "
        "each disconnected component has 4 states."
    )
    print("The unsupported fifth locus remains a 0.5/0.5 coin flip.")

    low_raw = [
        ((0, 0, 0), 2),
        ((0, 0, 1), 2),
        ((0, 1, 0), 2),
    ]
    show_posterior(
        "C. Very low temperature without underflow division", 3, low_raw, 1e-100
    )

    rng = random.Random(20260926)
    random_raw = []
    for _ in range(24):
        lo = rng.randrange(0, 9)
        hi = min(10, lo + rng.randint(2, 5))
        alleles = tuple(
            rng.choice((0, 1)) if lo <= site < hi else None for site in range(10)
        )
        random_raw.append((alleles, rng.randint(1, 7)))
    show_posterior("D. Random 10-locus check, computed now", 10, random_raw, 1.25)

    print("\nE. Validation: non-positive temperature")
    try:
        boltzmann_posterior(2, [], 0)
    except InvalidPhasingInputError as error:
        print(f"raised {type(error).__name__}: {error}")

    print("\ndemo finished.")


if __name__ == "__main__":
    main()
