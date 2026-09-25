"""Cross-fragment diploid haplotype phasing (weighted MEC) via branch and bound.

Public API
----------
Observation
    One fragment read: ``alleles`` is a sequence of length ``loci`` whose
    entries are 0, 1 or None (missing); ``flip_cost`` is the positive integer
    cost of flipping *one* observed allele in that read.
phase_haplotypes(loci, observations, *, max_nodes=None) -> PhasingResult
    Exact weighted minimum-error-correction phasing.

All arithmetic is exact integer arithmetic; the package never uses floating
point for costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "Observation",
    "PhasingResult",
    "PosteriorResult",
    "ReadAssignment",
    "SearchStats",
    "PhasingError",
    "InvalidPhasingInputError",
    "SearchBudgetExhaustedError",
    "phase_haplotypes",
    "phase_posterior",
    "MAX_LOCI",
    "MAX_OBSERVATIONS",
]

MAX_LOCI = 18
MAX_OBSERVATIONS = 200

_H0 = 0  # label of the reported haplotype h
_H1 = 1  # label of the complement 1-h


class PhasingError(ValueError):
    """Base class for all errors raised by this package."""


class InvalidPhasingInputError(PhasingError):
    """An input value violated the fixed public contract.

    The message always pinpoints the offending location
    (e.g. ``observations[3].alleles[7]``).
    """


class SearchBudgetExhaustedError(RuntimeError):
    """Raised when ``max_nodes`` is reached before optimality was proven.

    This state is deliberately distinct from input rejection and from
    infeasibility: every *validated* instance of this problem is feasible
    (the whole candidate space 2**(loci-1) is admissible), so a stopped
    search simply has an unproven incumbent, never a proven impossibility.
    """

    def __init__(
        self,
        message: str,
        *,
        budget: int,
        nodes_opened: int,
        leaves_evaluated: int,
        best_cost: Optional[int],
        best_haplotype: Optional[tuple[int, ...]],
    ) -> None:
        super().__init__(message)
        self.budget = budget
        self.nodes_opened = nodes_opened
        self.leaves_evaluated = leaves_evaluated
        self.best_cost = best_cost
        self.best_haplotype = best_haplotype


@dataclass(frozen=True)
class Observation:
    """A fragment read over the heterozygous sites.

    alleles: entry k is the observed binary allele at locus k, or None when
             the site is not covered by the read.
    flip_cost: positive integer cost of flipping one allele of this read.
    """

    alleles: tuple[Optional[int], ...]
    flip_cost: int


@dataclass(frozen=True)
class SearchStats:
    """Observable state of the branch-and-bound search."""

    depth: int
    budget: Optional[int]
    completed: bool
    nodes_opened: int
    leaves_evaluated: int
    branches_pruned: int
    incumbent_updates: int
    nodes_by_depth: tuple[int, ...]
    pruned_by_depth: tuple[int, ...]
    seed_costs: tuple[int, ...]


@dataclass(frozen=True)
class ReadAssignment:
    """Assignment of one read to the cheaper complementary haplotype.

    assigned: 0 (the reported h) or 1 (the complement 1-h). When ``tied`` is
              True the two costs are equal and ``assigned`` holds the
              deterministic tie-break 0; both flip lists are then valid.
    tied: True iff cost_to_h0 == cost_to_h1.
    cost_to_h0 / cost_to_h1: exact weighted flip costs to each haplotype.
    flips_to_h0 / flips_to_h1: 0-based loci that would have to be flipped.
    flips_applied: the cheaper flip set (equals flips_to_h0 on a tie).
    """

    index: int
    assigned: int
    tied: bool
    cost_to_h0: int
    cost_to_h1: int
    flips_to_h0: tuple[int, ...]
    flips_to_h1: tuple[int, ...]
    flips_applied: tuple[int, ...]


@dataclass(frozen=True)
class PhasingResult:
    """Exact phasing result.

    haplotype: canonical (lexicographically smallest) optimal h with h[0]=0.
    complement: 1-h.
    optimal_haplotypes: every distinct optimal h (more than one when the
        fragment-overlap graph is disconnected; all start with 0).
    optimal_cost / second_best_cost: smallest cost and the smallest cost
        strictly greater than it (None if every feasible h is optimal).
    gap: second_best_cost - optimal_cost, or None. Costs are integers, so
        when present gap >= 1.
    assignments: per-read assignment for the canonical ``haplotype``.
    free_components: connected components of the locus overlap graph not
        containing locus 0; flipping every bit of such a component leaves
        the total cost unchanged (there are 2**len(free_components) optima).
    search: branch-and-bound search state.
    """

    haplotype: tuple[int, ...]
    complement: tuple[int, ...]
    optimal_haplotypes: tuple[tuple[int, ...], ...]
    optimal_cost: int
    second_best_cost: Optional[int]
    gap: Optional[int]
    assignments: tuple[ReadAssignment, ...]
    free_components: tuple[tuple[int, ...], ...]
    search: SearchStats


@dataclass(frozen=True)
class PosteriorResult:
    """Boltzmann posterior over the gauge-fixed candidate space.

    The candidate space is the same one used by :func:phase_haplotypes:
    the component containing locus 0 is enumerated with h[0] = 0 fixed,
    and every free component contributes *both* of its global flips as
    distinct candidates (consistent with optimal_haplotypes counting
    2**len(free_components) optima).  Candidates are weighted by
    exp(-cost / temperature).

    marginals: marginals[k] is the posterior probability that the
        canonical haplotype takes allele 1 at locus k.  marginals[0]
        is exactly 0.0 because of the gauge.
    log_partition: natural log of the partition function Z (log-sum-exp
        accumulated, so arbitrarily low temperatures cannot underflow).
    entropy: posterior Shannon entropy in nats,
        H = log Z + mean_cost / temperature (additive over the
        independent connected components).
    mean_cost: posterior mean of the weighted-MEC cost.
    temperature: the temperature the posterior was evaluated at (float).
    components: the connected components of the locus overlap graph that
        were summed independently; the first one contains locus 0.
    """

    marginals: tuple[float, ...]
    log_partition: float
    entropy: float
    mean_cost: float
    temperature: float
    components: tuple[tuple[int, ...], ...]


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_inputs(
    loci: object,
    observations: object,
    max_nodes: object,
) -> tuple[int, list[Observation]]:
    if not _is_plain_int(loci):
        kind = "bool" if isinstance(loci, bool) else type(loci).__name__
        raise InvalidPhasingInputError(
            f"loci: expected int, got {kind}; bool is not accepted"
        )
    if not 1 <= loci <= MAX_LOCI:
        raise InvalidPhasingInputError(
            f"loci: must satisfy 1 <= loci <= {MAX_LOCI}, got {loci}"
        )

    if max_nodes is not None:
        if not _is_plain_int(max_nodes):
            kind = "bool" if isinstance(max_nodes, bool) else type(max_nodes).__name__
            raise InvalidPhasingInputError(
                f"max_nodes: expected int or None, got {kind}; bool is not accepted"
            )
        if max_nodes < 1:
            raise InvalidPhasingInputError(
                f"max_nodes: must be >= 1 when given, got {max_nodes}"
            )

    if not isinstance(observations, (list, tuple)):
        raise InvalidPhasingInputError(
            "observations: expected a list or tuple of Observation, got "
            + type(observations).__name__
        )
    if len(observations) > MAX_OBSERVATIONS:
        raise InvalidPhasingInputError(
            f"observations: at most {MAX_OBSERVATIONS} reads, got {len(observations)}"
        )

    checked: list[Observation] = []
    for i, obs in enumerate(observations):
        if not isinstance(obs, Observation):
            raise InvalidPhasingInputError(
                f"observations[{i}]: expected Observation, got {type(obs).__name__}"
            )
        alleles = obs.alleles
        if not isinstance(alleles, (list, tuple)):
            raise InvalidPhasingInputError(
                f"observations[{i}].alleles: expected tuple/list, got "
                + type(alleles).__name__
            )
        if len(alleles) != loci:
            raise InvalidPhasingInputError(
                f"observations[{i}].alleles: length {len(alleles)} "
                f"does not match loci={loci}"
            )
        for j, value in enumerate(alleles):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                if isinstance(value, float):
                    detail = (
                        "NaN/Infinity"
                        if not math.isfinite(value)
                        else "a float"
                    )
                    raise InvalidPhasingInputError(
                        f"observations[{i}].alleles[{j}]: expected 0/1/None, "
                        f"got {value!r} ({detail} is not accepted)"
                    )
                raise InvalidPhasingInputError(
                    f"observations[{i}].alleles[{j}]: expected 0/1/None, got "
                    f"{value!r} (type {type(value).__name__}); bool is not accepted"
                )
            if value not in (0, 1):
                raise InvalidPhasingInputError(
                    f"observations[{i}].alleles[{j}]: expected 0/1/None, got {value}"
                )

        cost = obs.flip_cost
        if isinstance(cost, bool) or not isinstance(cost, int):
            if isinstance(cost, float):
                detail = (
                    "NaN/Infinity" if not math.isfinite(cost) else "a float"
                )
                raise InvalidPhasingInputError(
                    f"observations[{i}].flip_cost: expected a positive int, got "
                    f"{cost!r} ({detail} is not accepted)"
                )
            raise InvalidPhasingInputError(
                f"observations[{i}].flip_cost: expected a positive int, got "
                f"{cost!r} (type {type(cost).__name__})"
            )
        if cost < 1:
            raise InvalidPhasingInputError(
                f"observations[{i}].flip_cost: must be a positive integer >= 1, "
                f"got {cost}"
            )
        checked.append(obs)
    return loci, checked


def _evaluate(h: tuple[int, ...], obs_list: Sequence[Observation]) -> int:
    """Exact weighted cost of a complete candidate h (independent of the B&B)."""
    total = 0
    for obs in obs_list:
        mismatches = 0
        covered = 0
        for k, allele in enumerate(obs.alleles):
            if allele is not None:
                covered += 1
                if allele != h[k]:
                    mismatches += 1
        total += obs.flip_cost * min(mismatches, covered - mismatches)
    return total


def _greedy_seed(
    loci: int,
    weights: Sequence[int],
    watch: Sequence[Sequence[tuple[int, int]]],
    *,
    flip_last: bool,
) -> tuple[int, ...]:
    """Deterministic incumbent heuristic: extend h bit by bit, each step
    choosing the bit with the smaller admissible prefix lower bound.
    This only seeds the incumbent; it never decides the returned optimum."""
    n = len(weights)
    mis = [0] * n
    seen = [0] * n
    bits = [0] * loci
    for k in range(1, loci):
        deltas: list[int] = []
        for bit in (0, 1):
            delta = 0
            for i, allele in watch[k]:
                b = seen[i] - mis[i]
                old = weights[i] * min(mis[i], b)
                new_mis = mis[i] + (1 if allele != bit else 0)
                new_b = b + (1 if allele == bit else 0)
                delta += weights[i] * min(new_mis, new_b) - old
            deltas.append(delta)
        bit = 0 if deltas[0] <= deltas[1] else 1
        bits[k] = bit
        for i, allele in watch[k]:
            seen[i] += 1
            if allele != bit:
                mis[i] += 1
    if flip_last and loci >= 2:
        bits[-1] ^= 1
    return tuple(bits)


def _free_components(
    loci: int, obs_list: Sequence[Observation]
) -> tuple[tuple[int, ...], ...]:
    parent = list(range(loci))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for obs in obs_list:
        sites = [k for k, allele in enumerate(obs.alleles) if allele is not None]
        for k in sites[1:]:
            union(sites[0], k)

    groups: dict[int, list[int]] = {}
    for k in range(loci):
        groups.setdefault(find(k), []).append(k)
    root0 = find(0)
    free = sorted(
        tuple(group) for root, group in groups.items() if root != root0
    )
    return tuple(free)


def phase_haplotypes(
    loci: int,
    observations: Sequence[Observation],
    *,
    max_nodes: Optional[int] = None,
) -> PhasingResult:
    """Phase diploid haplotypes by exact weighted-MEC branch and bound.

    Raises InvalidPhasingInputError for contract violations (with a message
    pinpointing the offending argument) and SearchBudgetExhaustedError if the
    optional node budget is exhausted before optimality is proven.
    """
    L, obs_list = _validate_inputs(loci, observations, max_nodes)
    n = len(obs_list)
    weights = [obs.flip_cost for obs in obs_list]
    covered_sites = [
        [k for k, allele in enumerate(obs.alleles) if allele is not None]
        for obs in obs_list
    ]
    watch: list[list[tuple[int, int]]] = [[] for _ in range(L)]
    for i, obs in enumerate(obs_list):
        for k in covered_sites[i]:
            watch[k].append((i, obs.alleles[k]))  # type: ignore[arg-type]

    # ---- incumbent seeds (feasible candidates; never prune against an
    # unproven bound that could hide an optimum: a seed cost is a true cost) --
    seed0 = _greedy_seed(L, weights, watch, flip_last=False)
    seed_costs = [_evaluate(seed0, obs_list)]
    if L >= 2:
        seed1 = _greedy_seed(L, weights, watch, flip_last=True)
        c1 = _evaluate(seed1, obs_list)
        seed_costs.append(c1)
    else:
        seed1, c1 = None, None

    best: Optional[int] = None
    second: Optional[int] = None
    incumbent_h: Optional[tuple[int, ...]] = None
    incumbent_updates = 0

    def absorb(cost: int, h_candidate: Optional[tuple[int, ...]] = None) -> None:
        nonlocal best, second, incumbent_h, incumbent_updates
        if best is None or cost < best:
            if best is not None:
                second = best if second is None else min(second, best)
            best = cost
            if h_candidate is not None:
                incumbent_h = h_candidate
            incumbent_updates += 1
        elif cost > best and (second is None or cost < second):
            second = cost

    seeds: list[tuple[int, tuple[int, ...]]] = [(seed_costs[0], seed0)]
    if L >= 2:
        seeds.append((c1, seed1))  # type: ignore[possibly-undefined]
    for cost, seed in seeds:
        absorb(cost, seed)

    # ---- branch-and-bound state -------------------------------------------
    mis = [0] * n       # mismatches against h among decided loci, per read
    seen = [0] * n      # observed sites among decided loci, per read
    bits = [0] * L
    optimal: list[tuple[int, ...]] = []
    nodes_by_depth = [0] * (L + 1)
    pruned_by_depth = [0] * L
    nodes_opened = 0
    leaves_evaluated = 0
    branches_pruned = 0
    stopped = False

    def child_delta(k: int, bit: int) -> int:
        """Admissible-prefix contribution change if h[k] is fixed to bit.

        Pure function of the current prefix state (does not mutate)."""
        delta = 0
        for i, allele in watch[k]:
            m, s, w = mis[i], seen[i], weights[i]
            old = w * min(m, s - m)
            new_m = m + (1 if allele != bit else 0)
            delta += w * min(new_m, s + 1 - new_m) - old
        return delta

    def commit(k: int, bit: int) -> None:
        for i, allele in watch[k]:
            seen[i] += 1
            if allele != bit:
                mis[i] += 1

    def rollback(k: int, bit: int) -> None:
        for i, allele in watch[k]:
            seen[i] -= 1
            if allele != bit:
                mis[i] -= 1

    def dfs(k: int, lb: int) -> None:
        nonlocal nodes_opened, leaves_evaluated, branches_pruned, stopped
        nodes_opened += 1
        nodes_by_depth[k] += 1
        if max_nodes is not None and nodes_opened > max_nodes:
            stopped = True
            return
        if k == L:
            leaves_evaluated += 1
            cost = lb  # all observed sites decided -> bound is exact
            prev_best = best
            candidate = tuple(bits)
            absorb(cost, candidate)
            if prev_best is None or cost < prev_best:
                optimal.clear()
                optimal.append(candidate)
            elif cost == best:
                optimal.append(candidate)
            return

        # admissible threshold: a child whose prefix LB already exceeds the
        # second distinct known cost cannot improve either optimum or the
        # exact second-best, so the whole subtree is safely discarded.
        threshold = second
        children = sorted(
            ((bit, lb + child_delta(k, bit)) for bit in (0, 1)),
            key=lambda item: item[1],
        )
        for bit, child_lb in children:
            if threshold is not None and child_lb > threshold:
                branches_pruned += 1
                pruned_by_depth[k] += 1
                continue
            commit(k, bit)
            bits[k] = bit
            dfs(k + 1, child_lb)
            rollback(k, bit)
            if stopped:
                return

    bits[0] = 0
    root_lb = child_delta(0, 0)
    commit(0, 0)
    dfs(1, root_lb)

    if stopped:
        raise SearchBudgetExhaustedError(
            f"search budget exhausted after {nodes_opened} nodes "
            f"(budget={max_nodes}); optimality not proven",
            budget=max_nodes,  # type: ignore[arg-type]
            nodes_opened=nodes_opened,
            leaves_evaluated=leaves_evaluated,
            best_cost=best,
            best_haplotype=incumbent_h,
        )

    # ---- deterministic canonical solution and per-read reporting ----------
    optimal_haplotypes = tuple(sorted(set(optimal)))
    h = optimal_haplotypes[0]
    complement = tuple(1 - bit for bit in h)

    assignments: list[ReadAssignment] = []
    for i, obs in enumerate(obs_list):
        flips0 = [k for k in covered_sites[i] if obs.alleles[k] != h[k]]
        flips1 = [k for k in covered_sites[i] if obs.alleles[k] == h[k]]
        c0 = weights[i] * len(flips0)
        c1 = weights[i] * len(flips1)
        tied = c0 == c1
        assigned = _H0 if c0 <= c1 else _H1
        flips_applied = flips0 if assigned == _H0 else flips1
        assignments.append(
            ReadAssignment(
                index=i,
                assigned=assigned,
                tied=tied,
                cost_to_h0=c0,
                cost_to_h1=c1,
                flips_to_h0=tuple(flips0),
                flips_to_h1=tuple(flips1),
                flips_applied=tuple(flips_applied),
            )
        )

    stats = SearchStats(
        depth=L,
        budget=max_nodes,
        completed=True,
        nodes_opened=nodes_opened,
        leaves_evaluated=leaves_evaluated,
        branches_pruned=branches_pruned,
        incumbent_updates=incumbent_updates,
        nodes_by_depth=tuple(nodes_by_depth),
        pruned_by_depth=tuple(pruned_by_depth),
        seed_costs=tuple(sorted(seed_costs)),
    )
    return PhasingResult(
        haplotype=h,
        complement=complement,
        optimal_haplotypes=optimal_haplotypes,
        optimal_cost=best,  # type: ignore[arg-type]
        second_best_cost=second,
        gap=None if second is None else second - best,  # type: ignore[operator]
        assignments=tuple(assignments),
        free_components=_free_components(L, obs_list),
        search=stats,
    )


def _validate_temperature(temperature: object) -> float:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        kind = "bool" if isinstance(temperature, bool) else type(temperature).__name__
        raise InvalidPhasingInputError(
            f"temperature: expected a positive number, got {kind}; "
            "bool is not accepted"
        )
    value = float(temperature)
    if not math.isfinite(value):
        raise InvalidPhasingInputError(
            f"temperature: NaN/Infinity is not accepted, got {temperature!r}"
        )
    if value <= 0.0:
        raise InvalidPhasingInputError(
            f"temperature: must be strictly positive, got {temperature!r}"
        )
    return value


def _components(
    loci: int, obs_list: Sequence[Observation]
) -> tuple[tuple[int, ...], ...]:
    """All connected components of the locus overlap graph (locus 0 first)."""
    parent = list(range(loci))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for obs in obs_list:
        sites = [k for k, allele in enumerate(obs.alleles) if allele is not None]
        for k in sites[1:]:
            ra, rb = find(sites[0]), find(k)
            if ra != rb:
                parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for k in range(loci):
        groups.setdefault(find(k), []).append(k)
    ordered = sorted(groups.values(), key=lambda g: (0 not in g, g[0]))
    return tuple(tuple(g) for g in ordered)


class _LogSumExp:
    """Running log(sum(exp(x_i))) with max-shift; never underflows to 0."""

    __slots__ = ("m", "s")

    def __init__(self) -> None:
        self.m = -math.inf
        self.s = 0.0

    def add(self, x: float) -> None:
        if x > self.m:
            self.s = self.s * math.exp(self.m - x) + 1.0
            self.m = x
        else:
            self.s += math.exp(x - self.m)

    @property
    def value(self) -> float:
        return self.m + math.log(self.s)


def _component_posterior(
    sites: tuple[int, ...],
    gauge_site: Optional[int],
    reads: list[tuple[list[int], list[int], int]],
    inv_temp: float,
) -> tuple[float, list[float], float, float]:
    """Exact posterior sums for one connected component.

    sites: loci of the component (sorted).
    gauge_site: locus fixed to 0 (locus 0's component) or None, in which
        case both global flips are enumerated as distinct candidates.
    reads: (covered local indices, alleles, weight) per read; every read
        covers only sites of this component.

    Returns (log Z_c, per-site P(bit=1), mean cost, entropy in nats).
    Enumerates assignments in binary-reflected Gray-code order so the
    running exact integer cost is updated incrementally.
    """
    s = len(sites)
    local_of = {site: j for j, site in enumerate(sites)}
    free_locals = [j for j in range(s) if sites[j] != gauge_site]
    n_free = len(free_locals)

    # reads touching each local site: (read_idx, allele)
    watch: list[list[tuple[int, int]]] = [[] for _ in range(s)]
    covered = [0] * len(reads)
    weights = [0] * len(reads)
    for i, (locs, alleles, w) in enumerate(reads):
        covered[i] = len(locs)
        weights[i] = w
        for loc, allele in zip(locs, alleles):
            watch[loc].append((i, allele))

    mis = [0] * len(reads)
    bits = [0] * s

    def reset_state() -> int:
        # state of the all-zero assignment: every allele 1 is a mismatch
        for i in range(len(reads)):
            mis[i] = 0
        for local in range(s):
            bits[local] = 0
        for local in range(s):
            for i, allele in watch[local]:
                if allele == 1:
                    mis[i] += 1
        return sum(read_cost(i) for i in range(len(reads)))

    def read_cost(i: int) -> int:
        m = mis[i]
        return weights[i] * min(m, covered[i] - m)

    def flip(local: int, cost: int) -> int:
        bit = bits[local] ^ 1
        bits[local] = bit
        for i, allele in watch[local]:
            before = read_cost(i)
            if allele != bit:
                mis[i] += 1
            else:
                mis[i] -= 1
            cost += read_cost(i) - before
        return cost

    cost = reset_state()
    lse = _LogSumExp()
    lse.add(-cost * inv_temp)
    n_assign = 1 << n_free
    for gray_step in range(1, n_assign):
        lowest = gray_step & (-gray_step)
        free_bit = lowest.bit_length() - 1
        cost = flip(free_locals[free_bit], cost)
        lse.add(-cost * inv_temp)
    log_z = lse.value

    # second Gray-code pass: normalized weights are <= 1, plain floats are
    # exact enough and cannot overflow; underflow only rounds a negligible
    # probability to 0.0.
    cost = reset_state()
    prob_one = [0.0] * s
    mean_cost = 0.0

    def absorb(cost: float) -> None:
        nonlocal mean_cost
        p = math.exp(-cost * inv_temp - log_z)
        mean_cost += p * cost
        for local in free_locals:
            if bits[local]:
                prob_one[local] += p

    absorb(cost)
    for gray_step in range(1, n_assign):
        lowest = gray_step & (-gray_step)
        free_bit = lowest.bit_length() - 1
        cost = flip(free_locals[free_bit], cost)
        absorb(cost)

    entropy = log_z + mean_cost * inv_temp
    return log_z, prob_one, mean_cost, entropy


def phase_posterior(
    loci: int,
    observations: Sequence[Observation],
    *,
    temperature: float,
) -> PosteriorResult:
    """Boltzmann posterior of the weighted-MEC objective.

    Every gauge-fixed candidate h is weighted by exp(-cost / temperature).
    Independent connected components of the locus overlap graph are summed
    separately and combined in the log domain; the component containing
    locus 0 keeps the h[0] = 0 gauge, and each free component contributes
    both of its global flips as distinct candidates (the same multiplicity
    used for optimal_haplotypes).  Raises InvalidPhasingInputError for
    contract violations, including a non-positive or non-finite
    temperature.
    """
    temp = _validate_temperature(temperature)
    L, obs_list = _validate_inputs(loci, observations, None)
    inv_temp = 1.0 / temp

    components = _components(L, obs_list)
    comp_of = [0] * L
    for c, sites in enumerate(components):
        for k in sites:
            comp_of[k] = c

    comp_reads: list[list[tuple[list[int], list[int], int]]] = [
        [] for _ in components
    ]
    for obs in obs_list:
        locs = [k for k, a in enumerate(obs.alleles) if a is not None]
        if not locs:
            continue  # zero covered sites: cost 0 for every candidate
        c = comp_of[locs[0]]
        local_of = {site: j for j, site in enumerate(components[c])}
        comp_reads[c].append(
            (
                [local_of[k] for k in locs],
                [obs.alleles[k] for k in locs],  # type: ignore[misc]
                obs.flip_cost,
            )
        )

    marginals = [0.0] * L
    log_partition = 0.0
    mean_cost = 0.0
    entropy = 0.0
    for c, sites in enumerate(components):
        gauge_site = 0 if 0 in sites else None
        log_z, prob_one, mean_c, entropy_c = _component_posterior(
            sites, gauge_site, comp_reads[c], inv_temp
        )
        log_partition += log_z
        mean_cost += mean_c
        entropy += entropy_c
        for site, p in zip(sites, prob_one):
            marginals[site] = p

    return PosteriorResult(
        marginals=tuple(marginals),
        log_partition=log_partition,
        entropy=entropy,
        mean_cost=mean_cost,
        temperature=temp,
        components=components,
    )
