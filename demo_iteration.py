"""Fixed-input demo of the Boltzmann posterior entry point.

Everything is computed live (no sleeps, no prerecorded answers) and finishes
in well under 90 seconds -- the instances below run in milliseconds.

Scenario: two datasets whose *optimal* phasing is identical, yet whose
posterior uncertainty differs -- the optimum alone does not tell you how
sure the phasing is.  The temperature used here is a modelling choice; it
is NOT calibrated against real sequencing error rates.
"""

from __future__ import annotations

from haplophase import (
    InvalidPhasingInputError,
    Observation,
    phase_haplotypes,
    phase_posterior,
)

TEMPERATURE = 1.0


def show(name, loci, reads):
    obs = [Observation(alleles=a, flip_cost=w) for a, w in reads]
    result = phase_haplotypes(loci, obs)
    post = phase_posterior(loci, obs, temperature=TEMPERATURE)
    print(f"--- {name} ---")
    print(f"optimal haplotype h : {result.haplotype}  (cost {result.optimal_cost}, "
          f"gap {result.gap})")
    print(f"P(h[k]=1) per locus : "
          + " ".join(f"{p:.4f}" for p in post.marginals))
    print(f"log Z               : {post.log_partition:.6f}")
    print(f"entropy (nats)      : {post.entropy:.6f}")
    print(f"posterior mean cost : {post.mean_cost:.6f}")
    print()
    return post


def main():
    # Same optimum (0, 0, 0, 0): dataset A has strong, consistent evidence;
    # dataset B has weak, nearly tied evidence at loci 2..3.
    strong = [
        ((0, 0, 0, 0), 5), ((0, 0, 0, 0), 5), ((0, 0, 0, 0), 5),
        ((1, 1, 0, 0), 1), ((0, 0, 1, 1), 1),
    ]
    weak = [
        ((0, 0, 0, 0), 1), ((0, 0, 0, 0), 1), ((0, 0, 0, 1), 1),
        ((1, 1, 0, 0), 1), ((0, 0, 1, 1), 1), ((0, 0, 1, 0), 1),
        ((None, 0, 0, None), 1),
    ]
    a = show("dataset A: strong evidence", 4, strong)
    b = show("dataset B: weak evidence", 4, weak)
    print("same optimum, different certainty: "
          f"entropy A={a.entropy:.4f} nats vs B={b.entropy:.4f} nats")
    print("(temperature is a modelling knob here, not a calibrated error model)")
    print()

    # Error contract: temperature must be a positive finite number.
    for bad in (0.0, -1.0, float("nan"), True):
        try:
            phase_posterior(
                2, [Observation(alleles=(0, 1), flip_cost=1)], temperature=bad
            )
        except InvalidPhasingInputError as exc:
            print(f"rejected temperature={bad!r}: {exc}")


if __name__ == "__main__":
    main()
