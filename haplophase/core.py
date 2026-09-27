"""Cross-fragment diploid haplotype phasing (weighted MEC) via branch and bound.

Public API
----------
Observation
    One fragment read: ``alleles`` is a sequence of length ``loci`` whose
    entries are 0, 1 or None (missing); ``flip_cost`` is the positive integer
    cost of flipping *one* observed allele in that read.
phase_haplotypes(loci, observations, *, max_nodes=None) -> PhasingResult
    Exact weighted minimum-error-correction phasing.
optimal_haplotype_batch(loci, observations, *, batch_size, max_nodes=None,
                        cursor=None) -> OptimalBatch
    Batched enumeration of *all* optimal haplotypes in a stable
    lexicographic order: the first call proves the optimal cost, every call
    returns at most ``batch_size`` solutions plus an opaque continuation
    cursor, and no call materializes the whole optimal set internally.

All arithmetic is exact integer arithmetic; the package never uses floating
point for costs.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "Observation",
    "PhasingResult",
    "OptimalBatch",
    "ReadAssignment",
    "SearchStats",
    "PhasingError",
    "InvalidPhasingInputError",
    "SearchBudgetExhaustedError",
    "phase_haplotypes",
    "optimal_haplotype_batch",
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
class OptimalBatch:
    """One batch of optimal haplotypes from a batched enumeration.

    optimal_cost: the proven optimal total cost (established by the first
        call of a batch sequence and carried by the cursor afterwards).
    haplotypes: up to ``batch_size`` distinct optimal h (h[0] = 0), in
        stable ascending lexicographic order; complementary duplicates are
        already removed by the h[0] = 0 symmetry fixing, exactly as in
        ``PhasingResult.optimal_haplotypes``.
    next_cursor: opaque continuation token for the next call, or None when
        the enumeration is complete. A cursor is only valid for the exact
        same inputs and options (loci, observations, batch_size, max_nodes);
        anything else is rejected with InvalidPhasingInputError.
    exhausted: True iff the search ran to completion and every optimal
        haplotype has now been delivered. It is never set while the search
        is unfinished (a budget stop raises SearchBudgetExhaustedError
        instead of reporting a partial enumeration as complete).
    search: observable state of this call's search (proof pass plus
        enumeration pass on the first call, enumeration only afterwards).
    """

    optimal_cost: int
    haplotypes: tuple[tuple[int, ...], ...]
    next_cursor: Optional[str]
    exhausted: bool
    search: SearchStats


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


def _prepare_search_structures(
    L: int, obs_list: Sequence[Observation]
) -> tuple[list[int], list[list[int]], list[list[tuple[int, int]]]]:
    weights = [obs.flip_cost for obs in obs_list]
    covered_sites = [
        [k for k, allele in enumerate(obs.alleles) if allele is not None]
        for obs in obs_list
    ]
    watch: list[list[tuple[int, int]]] = [[] for _ in range(L)]
    for i, obs in enumerate(obs_list):
        for k in covered_sites[i]:
            watch[k].append((i, obs.alleles[k]))  # type: ignore[arg-type]
    return weights, covered_sites, watch


def _new_counters(L: int) -> dict:
    return {
        "nodes_opened": 0,
        "leaves_evaluated": 0,
        "branches_pruned": 0,
        "incumbent_updates": 0,
        "nodes_by_depth": [0] * (L + 1),
        "pruned_by_depth": [0] * L,
    }


def _run_branch_and_bound(
    L: int,
    obs_list: Sequence[Observation],
    weights: list[int],
    watch: list[list[tuple[int, int]]],
    max_nodes: Optional[int],
    collect: bool,
    counters: dict,
) -> tuple[
    Optional[int],
    Optional[int],
    list[tuple[int, ...]],
    Optional[tuple[int, ...]],
    tuple[int, ...],
    bool,
]:
    """Exact branch and bound over the h[0] = 0 candidate space.

    Proves the best and the strictly-second distinct cost. When ``collect``
    is true every optimal leaf is gathered; otherwise no solution list is
    materialized (the batched enumeration relies on this so it never
    pre-materializes all optima). ``counters`` is updated in place so
    several invocations can share one node budget.
    """
    n = len(weights)

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

    def absorb(cost: int, h_candidate: Optional[tuple[int, ...]] = None) -> None:
        nonlocal best, second, incumbent_h
        if best is None or cost < best:
            if best is not None:
                second = best if second is None else min(second, best)
            best = cost
            if h_candidate is not None:
                incumbent_h = h_candidate
            counters["incumbent_updates"] += 1
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
        nonlocal stopped
        counters["nodes_opened"] += 1
        counters["nodes_by_depth"][k] += 1
        if max_nodes is not None and counters["nodes_opened"] > max_nodes:
            stopped = True
            return
        if k == L:
            counters["leaves_evaluated"] += 1
            cost = lb  # all observed sites decided -> bound is exact
            prev_best = best
            candidate = tuple(bits)
            absorb(cost, candidate)
            if collect:
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
                counters["branches_pruned"] += 1
                counters["pruned_by_depth"][k] += 1
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
    return best, second, optimal, incumbent_h, tuple(sorted(seed_costs)), stopped


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
    weights, covered_sites, watch = _prepare_search_structures(L, obs_list)
    counters = _new_counters(L)
    best, second, optimal, incumbent_h, seed_costs, stopped = (
        _run_branch_and_bound(
            L, obs_list, weights, watch, max_nodes, True, counters
        )
    )
    if stopped:
        raise SearchBudgetExhaustedError(
            f"search budget exhausted after {counters['nodes_opened']} nodes "
            f"(budget={max_nodes}); optimality not proven",
            budget=max_nodes,  # type: ignore[arg-type]
            nodes_opened=counters["nodes_opened"],  # type: ignore[arg-type]
            leaves_evaluated=counters["leaves_evaluated"],  # type: ignore[arg-type]
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
        nodes_opened=counters["nodes_opened"],
        leaves_evaluated=counters["leaves_evaluated"],
        branches_pruned=counters["branches_pruned"],
        incumbent_updates=counters["incumbent_updates"],
        nodes_by_depth=tuple(counters["nodes_by_depth"]),
        pruned_by_depth=tuple(counters["pruned_by_depth"]),
        seed_costs=seed_costs,
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

_CURSOR_PREFIX = "hpb1"


def _batch_fingerprint(
    L: int, obs_list: Sequence[Observation], batch_size: int
) -> str:
    """Stable fingerprint of everything a continuation cursor depends on.

    Any change to the inputs (loci, observations) or to the result-shaping
    option batch_size yields a different fingerprint, which makes
    previously issued cursors invalid. max_nodes is deliberately excluded:
    it is a per-call safety budget, not part of the problem, so the same
    query may be continued with a different budget.
    """
    digest = hashlib.sha256()
    digest.update(repr((L, batch_size)).encode("utf-8"))
    for obs in obs_list:
        digest.update(repr((tuple(obs.alleles), obs.flip_cost)).encode("utf-8"))
    return digest.hexdigest()[:16]


def _encode_cursor(
    fingerprint: str, optimal_cost: int, last_haplotype: tuple[int, ...]
) -> str:
    bits = "".join(str(bit) for bit in last_haplotype)
    return f"{_CURSOR_PREFIX}:{fingerprint}:{optimal_cost}:{bits}"


def _decode_cursor(
    cursor: object, fingerprint: str, L: int
) -> tuple[int, tuple[int, ...]]:
    if not isinstance(cursor, str):
        raise InvalidPhasingInputError(
            f"cursor: expected a next_cursor string, got {type(cursor).__name__}"
        )
    parts = cursor.split(":")
    if len(parts) != 4 or parts[0] != _CURSOR_PREFIX:
        raise InvalidPhasingInputError(
            "cursor: malformed continuation token; pass the next_cursor value "
            "returned by a previous optimal_haplotype_batch call"
        )
    if parts[1] != fingerprint:
        raise InvalidPhasingInputError(
            "cursor: inputs or options changed since this cursor was issued; "
            "the continuation is invalid, start a new batch sequence"
        )
    try:
        optimal_cost = int(parts[2])
    except ValueError:
        raise InvalidPhasingInputError(
            "cursor: malformed continuation token (cost field)"
        ) from None
    bits = parts[3]
    if (
        len(bits) != L
        or any(ch not in "01" for ch in bits)
        or bits[0] != "0"
    ):
        raise InvalidPhasingInputError(
            "cursor: malformed continuation token (haplotype field)"
        )
    return optimal_cost, tuple(int(ch) for ch in bits)


def _enumerate_lex(
    L: int,
    weights: list[int],
    watch: list[list[tuple[int, int]]],
    optimal_cost: int,
    skip_below: Optional[tuple[int, ...]],
    fetch_limit: int,
    max_nodes: Optional[int],
    counters: dict,
) -> tuple[list[tuple[int, ...]], bool, bool]:
    """Lexicographic DFS collecting up to ``fetch_limit`` optimal leaves.

    Only subtrees whose admissible lower bound stays <= ``optimal_cost`` are
    entered; leaves are visited in ascending lexicographic order; leaves
    <= ``skip_below`` (already delivered) are skipped, whole subtrees at a
    time. At most ``fetch_limit`` leaves are kept, so the full optimal set
    is never materialized. Returns (results, stopped, hit_limit).
    """
    n = len(weights)
    mis = [0] * n
    seen = [0] * n
    bits = [0] * L
    results: list[tuple[int, ...]] = []
    stopped = False
    hit_limit = False

    def child_delta(k: int, bit: int) -> int:
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
        nonlocal stopped, hit_limit
        counters["nodes_opened"] += 1
        counters["nodes_by_depth"][k] += 1
        if max_nodes is not None and counters["nodes_opened"] > max_nodes:
            stopped = True
            return
        if k == L:
            counters["leaves_evaluated"] += 1
            candidate = tuple(bits)
            if lb == optimal_cost and (
                skip_below is None or candidate > skip_below
            ):
                results.append(candidate)
                if len(results) >= fetch_limit:
                    hit_limit = True
            return
        if skip_below is not None and tuple(bits[:k]) < skip_below[:k]:
            # every leaf below this prefix is lexicographically smaller than
            # the last delivered haplotype: already delivered, skip it all.
            counters["branches_pruned"] += 1
            counters["pruned_by_depth"][k] += 1
            return
        for bit in (0, 1):
            child_lb = lb + child_delta(k, bit)
            if child_lb > optimal_cost:
                counters["branches_pruned"] += 1
                counters["pruned_by_depth"][k] += 1
                continue
            commit(k, bit)
            bits[k] = bit
            dfs(k + 1, child_lb)
            rollback(k, bit)
            if stopped or hit_limit:
                return

    bits[0] = 0
    root_lb = child_delta(0, 0)
    commit(0, 0)
    dfs(1, root_lb)
    return results, stopped, hit_limit


def optimal_haplotype_batch(
    loci: int,
    observations: Sequence[Observation],
    *,
    batch_size: int,
    max_nodes: Optional[int] = None,
    cursor: Optional[str] = None,
) -> OptimalBatch:
    """Enumerate all optimal haplotypes in stable batches.

    The first call (``cursor=None``) proves the optimal cost with the exact
    branch and bound and returns the first batch; every later call passes
    the previous ``next_cursor`` and receives the next batch. Batches are
    delivered in ascending lexicographic order over h (h[0] = 0), so the
    concatenation of all batches contains every optimal haplotype exactly
    once - no duplicates, no omissions - with complementary symmetry
    removed exactly as in ``phase_haplotypes``.

    The full optimal set is never materialized internally: each call keeps
    at most ``batch_size + 1`` candidates in memory. ``exhausted`` (and
    ``search.completed``) is True only when the enumeration search ran to
    completion; an unfinished search is never reported as complete - if
    ``max_nodes`` runs out first, SearchBudgetExhaustedError is raised
    instead and no partial batch is delivered. The node budget covers the
    whole call (proof pass plus enumeration pass).

    A cursor is bound to the exact inputs and result-shaping options it was
    issued with (loci, observations, batch_size); using it after any of
    them changed raises InvalidPhasingInputError. max_nodes is a per-call
    safety budget and may be changed between calls without invalidating the
    continuation.
    """
    if not _is_plain_int(batch_size):
        kind = "bool" if isinstance(batch_size, bool) else type(batch_size).__name__
        raise InvalidPhasingInputError(
            f"batch_size: expected int, got {kind}; bool is not accepted"
        )
    if batch_size < 1:
        raise InvalidPhasingInputError(
            f"batch_size: must be a positive integer >= 1, got {batch_size}"
        )
    L, obs_list = _validate_inputs(loci, observations, max_nodes)
    fingerprint = _batch_fingerprint(L, obs_list, batch_size)
    if cursor is None:
        proven_cost: Optional[int] = None
        skip_below: Optional[tuple[int, ...]] = None
    else:
        proven_cost, skip_below = _decode_cursor(cursor, fingerprint, L)

    weights, _covered, watch = _prepare_search_structures(L, obs_list)
    counters = _new_counters(L)
    seed_costs: tuple[int, ...] = ()
    if proven_cost is None:
        best, _second, _opt, incumbent_h, seed_costs, stopped = (
            _run_branch_and_bound(
                L, obs_list, weights, watch, max_nodes, False, counters
            )
        )
        if stopped:
            raise SearchBudgetExhaustedError(
                f"search budget exhausted after {counters['nodes_opened']} "
                f"nodes (budget={max_nodes}); optimal cost not yet proven",
                budget=max_nodes,  # type: ignore[arg-type]
                nodes_opened=counters["nodes_opened"],  # type: ignore[arg-type]
                leaves_evaluated=counters["leaves_evaluated"],  # type: ignore[arg-type]
                best_cost=best,
                best_haplotype=incumbent_h,
            )
        proven_cost = best

    results, stopped, _hit_limit = _enumerate_lex(
        L,
        weights,
        watch,
        proven_cost,  # type: ignore[arg-type]
        skip_below,
        batch_size + 1,
        max_nodes,
        counters,
    )
    if stopped:
        raise SearchBudgetExhaustedError(
            f"search budget exhausted after {counters['nodes_opened']} nodes "
            f"(budget={max_nodes}); enumeration incomplete, no batch delivered",
            budget=max_nodes,  # type: ignore[arg-type]
            nodes_opened=counters["nodes_opened"],  # type: ignore[arg-type]
            leaves_evaluated=counters["leaves_evaluated"],  # type: ignore[arg-type]
            best_cost=proven_cost,
            best_haplotype=results[0] if results else None,
        )

    if len(results) > batch_size:
        delivered = tuple(results[:batch_size])
        exhausted = False
        next_cursor: Optional[str] = _encode_cursor(
            fingerprint, proven_cost, results[batch_size - 1]  # type: ignore[arg-type]
        )
    else:
        # the enumeration search ran to completion: nothing remains
        delivered = tuple(results)
        exhausted = True
        next_cursor = None

    stats = SearchStats(
        depth=L,
        budget=max_nodes,
        completed=exhausted,
        nodes_opened=counters["nodes_opened"],
        leaves_evaluated=counters["leaves_evaluated"],
        branches_pruned=counters["branches_pruned"],
        incumbent_updates=counters["incumbent_updates"],
        nodes_by_depth=tuple(counters["nodes_by_depth"]),
        pruned_by_depth=tuple(counters["pruned_by_depth"]),
        seed_costs=seed_costs,
    )
    return OptimalBatch(
        optimal_cost=proven_cost,  # type: ignore[arg-type]
        haplotypes=delivered,
        next_cursor=next_cursor,
        exhausted=exhausted,
        search=stats,
    )
