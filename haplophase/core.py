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

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "EnumerationCursor",
    "Observation",
    "OptimalBatch",
    "PhasingResult",
    "ReadAssignment",
    "SearchStats",
    "PhasingError",
    "InvalidPhasingInputError",
    "SearchBudgetExhaustedError",
    "phase_haplotypes",
    "phase_haplotypes_batched",
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


# ---------------------------------------------------------------------------
# Batched enumeration of optimal solutions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnumerationCursor:
    """Continuation token for phase_haplotypes_batched.

    Opaque: produced by a previous batched call, pass back unchanged as the
    cursor argument. Bound to a fingerprint of the exact inputs (loci, every
    observation); if the problem instance changes, the token is rejected
    instead of silently resuming the wrong enumeration. max_nodes is not
    part of the fingerprint: it only limits one call's search effort, so a
    budget-exhausted continuation can be retried without a budget. It never
    holds the solutions themselves, only the position of the suspended
    depth-first search (a path of bits plus, per depth, the next branch to
    try), so memory stays O(loci).
    """

    fingerprint: str
    next_bits: tuple[int, ...]
    path: tuple[int, ...]
    emitted: int


@dataclass(frozen=True)
class OptimalBatch:
    """One batch of optimal haplotypes from phase_haplotypes_batched.

    optimal_cost: the proven optimal total cost (re-proven on every call).
    haplotypes: this batch of distinct optimal h, in the same stable
        lexicographic order as PhasingResult.optimal_haplotypes; each
        starts with 0 (complement symmetry removed as in phase_haplotypes).
    exhausted: True only when the search has provably visited every
        remaining candidate; a non-final batch always has exhausted False.
    cursor: token for the next call, or None iff exhausted is True.
    emitted_total: cumulative number of optima emitted including this batch.
    search: search state of *this* call; search.completed mirrors
        exhausted and is never True while enumeration is unfinished.
    """

    optimal_cost: int
    haplotypes: tuple[tuple[int, ...], ...]
    exhausted: bool
    cursor: Optional[EnumerationCursor]
    emitted_total: int
    search: SearchStats


def _fingerprint(loci, obs_list) -> str:
    canonical = (
        loci,
        tuple((tuple(obs.alleles), obs.flip_cost) for obs in obs_list),
    )
    return hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()


class _Workspace:
    """Reusable prefix state (mis/seen/bits) for the batched search passes."""

    __slots__ = ("loci", "weights", "watch", "mis", "seen", "bits")

    def __init__(self, loci, weights, watch):
        self.loci = loci
        self.weights = weights
        self.watch = watch
        self.mis = [0] * len(weights)
        self.seen = [0] * len(weights)
        self.bits = [0] * loci

    def child_delta(self, k, bit):
        delta = 0
        for i, allele in self.watch[k]:
            m, s, w = self.mis[i], self.seen[i], self.weights[i]
            old = w * min(m, s - m)
            new_m = m + (1 if allele != bit else 0)
            delta += w * min(new_m, s + 1 - new_m) - old
        return delta

    def commit(self, k, bit):
        for i, allele in self.watch[k]:
            self.seen[i] += 1
            if allele != bit:
                self.mis[i] += 1

    def rollback(self, k, bit):
        for i, allele in self.watch[k]:
            self.seen[i] -= 1
            if allele != bit:
                self.mis[i] -= 1


class _Counters:
    """Mutable node/leaf/prune counters shared by the two search passes."""

    def __init__(self, loci, max_nodes):
        self.max_nodes = max_nodes
        self.nodes_opened = 0
        self.leaves_evaluated = 0
        self.branches_pruned = 0
        self.nodes_by_depth = [0] * (loci + 1)
        self.pruned_by_depth = [0] * loci
        self.stopped = False

    def open(self, k):
        self.nodes_opened += 1
        self.nodes_by_depth[k] += 1
        if self.max_nodes is not None and self.nodes_opened > self.max_nodes:
            self.stopped = True


def _prove_optimal_cost(ws, seed_costs, counters):
    """Branch-and-bound that proves the optimal cost without materializing
    any optimal solution list."""
    L = ws.loci
    best = None
    for cost in seed_costs:
        if best is None or cost < best:
            best = cost

    def dfs(k, lb):
        nonlocal best
        counters.open(k)
        if counters.stopped:
            return
        if k == L:
            counters.leaves_evaluated += 1
            if best is None or lb < best:
                best = lb
            return
        children = sorted(
            ((bit, lb + ws.child_delta(k, bit)) for bit in (0, 1)),
            key=lambda item: item[1],
        )
        for bit, child_lb in children:
            if best is not None and child_lb > best:
                counters.branches_pruned += 1
                counters.pruned_by_depth[k] += 1
                continue
            ws.commit(k, bit)
            dfs(k + 1, child_lb)
            ws.rollback(k, bit)
            if counters.stopped:
                return

    ws.commit(0, 0)
    dfs(1, ws.child_delta(0, 0))
    ws.rollback(0, 0)
    return best


def _enumerate_batch(ws, best, batch_size, cursor, counters):
    """Depth-first enumeration of every h with cost equal to best, in
    lexicographic order (bit 0 tried before bit 1 at every depth), emitting
    at most batch_size solutions. Only the suspended stack (O(loci)) is kept
    between calls; solutions are never buffered beyond one batch."""
    L = ws.loci
    batch = []
    # one entry per opened node at depths 1..L-1:
    # [next_branch_bit, lower_bound, entry_bit] where entry_bit is the bit
    # committed at depth k-1 to reach the node of depth k (0 for the root,
    # reached via the fixed h[0]=0 commit).
    frames = []

    ws.commit(0, 0)
    root_lb = ws.child_delta(0, 0)
    if cursor is None:
        counters.open(1)
        frames.append([0, root_lb, 0])
    else:
        lb = root_lb
        for depth in range(1, len(cursor.next_bits) + 1):
            counters.open(depth)
            if depth >= 2:
                bit = cursor.path[depth - 1]
                lb += ws.child_delta(depth - 1, bit)
                ws.commit(depth - 1, bit)
                ws.bits[depth - 1] = bit
            else:
                bit = 0
            frames.append([cursor.next_bits[depth - 1], lb, bit])

    while frames and len(batch) < batch_size and not counters.stopped:
        frame = frames[-1]
        k = len(frames)
        if frame[0] == 2:
            frames.pop()
            ws.rollback(k - 1, frame[2])
            continue
        bit = frame[0]
        frame[0] += 1
        child_lb = frame[1] + ws.child_delta(k, bit)
        if child_lb > best:
            counters.branches_pruned += 1
            counters.pruned_by_depth[k] += 1
            continue
        ws.commit(k, bit)
        ws.bits[k] = bit
        if k + 1 == L:
            counters.open(L)
            if not counters.stopped:
                counters.leaves_evaluated += 1
                if child_lb == best:
                    batch.append(tuple(ws.bits))
            ws.rollback(k, bit)
        else:
            counters.open(k + 1)
            frames.append([0, child_lb, bit])

    exhausted = not frames
    raw_cursor = None
    if not exhausted and not counters.stopped:
        raw_cursor = frames
    return batch, raw_cursor, exhausted


def _validate_cursor(cursor, loci, fingerprint):
    if not isinstance(cursor, EnumerationCursor):
        raise InvalidPhasingInputError(
            "cursor: expected an EnumerationCursor returned by a previous "
            f"phase_haplotypes_batched call, got {type(cursor).__name__}"
        )
    for name, values, allowed in (
        ("next_bits", cursor.next_bits, (0, 1, 2)),
        ("path", cursor.path, (0, 1)),
    ):
        if not isinstance(values, tuple):
            raise InvalidPhasingInputError(
                f"cursor.{name}: expected a tuple, got {type(values).__name__}"
            )
        for j, value in enumerate(values):
            if not _is_plain_int(value) or value not in allowed:
                raise InvalidPhasingInputError(
                    f"cursor.{name}[{j}]: expected one of {allowed}, got {value!r}"
                )
    if len(cursor.next_bits) != len(cursor.path):
        raise InvalidPhasingInputError(
            "cursor: malformed state (len(next_bits) must equal len(path))"
        )
    if not 1 <= len(cursor.next_bits) <= max(1, loci - 1):
        raise InvalidPhasingInputError(
            f"cursor: search depth {len(cursor.next_bits)} is out of range "
            f"for loci={loci}"
        )
    if not _is_plain_int(cursor.emitted) or cursor.emitted < 0:
        raise InvalidPhasingInputError(
            f"cursor.emitted: expected a non-negative int, got {cursor.emitted!r}"
        )
    if cursor.fingerprint != fingerprint:
        raise InvalidPhasingInputError(
            "cursor: inputs or options changed since this token was issued; "
            "start a new enumeration with cursor=None"
        )
    return cursor


def phase_haplotypes_batched(
    loci,
    observations,
    *,
    batch_size,
    cursor=None,
    max_nodes=None,
):
    """Enumerate all optimal haplotypes in stable batches.

    The first call (cursor=None) proves the optimal cost and returns the
    first batch; pass the returned OptimalBatch.cursor back to fetch the
    next batch. Batches are disjoint, in lexicographic order, and together
    cover exactly the same set as phase_haplotypes(...).optimal_haplotypes
    (complement symmetry removed by fixing h[0] = 0). At most one batch of
    solutions is ever held in memory; the cursor stores only the suspended
    search position. Changing loci or observations invalidates previously
    issued cursors; max_nodes only limits one call's search effort and does
    not invalidate a cursor (a budget-exhausted continuation can be retried
    without a budget). Budget exhaustion raises SearchBudgetExhaustedError
    and never produces a batch that claims completion.
    """
    if not _is_plain_int(batch_size):
        kind = "bool" if isinstance(batch_size, bool) else type(batch_size).__name__
        raise InvalidPhasingInputError(
            f"batch_size: expected int, got {kind}; bool is not accepted"
        )
    if batch_size < 1:
        raise InvalidPhasingInputError(
            f"batch_size: must be >= 1, got {batch_size}"
        )
    L, obs_list = _validate_inputs(loci, observations, max_nodes)
    fingerprint = _fingerprint(L, obs_list)
    if cursor is not None:
        cursor = _validate_cursor(cursor, L, fingerprint)

    weights = [obs.flip_cost for obs in obs_list]
    watch = [[] for _ in range(L)]
    for i, obs in enumerate(obs_list):
        for k, allele in enumerate(obs.alleles):
            if allele is not None:
                watch[k].append((i, allele))

    seed0 = _greedy_seed(L, weights, watch, flip_last=False)
    seed_costs = [_evaluate(seed0, obs_list)]
    if L >= 2:
        seed1 = _greedy_seed(L, weights, watch, flip_last=True)
        seed_costs.append(_evaluate(seed1, obs_list))

    counters = _Counters(L, max_nodes)
    ws = _Workspace(L, weights, watch)
    best = _prove_optimal_cost(ws, seed_costs, counters)
    incumbent_updates = len(seed_costs)
    if counters.stopped:
        raise SearchBudgetExhaustedError(
            f"search budget exhausted after {counters.nodes_opened} nodes "
            f"(budget={max_nodes}); optimality not proven",
            budget=max_nodes,
            nodes_opened=counters.nodes_opened,
            leaves_evaluated=counters.leaves_evaluated,
            best_cost=best,
            best_haplotype=None,
        )

    if L == 1:
        batch = [(0,)]
        exhausted = True
        next_cursor = None
    else:
        batch, raw_cursor, exhausted = _enumerate_batch(
            ws, best, batch_size, cursor, counters
        )
        if counters.stopped:
            raise SearchBudgetExhaustedError(
                f"search budget exhausted after {counters.nodes_opened} nodes "
                f"(budget={max_nodes}); enumeration not complete",
                budget=max_nodes,
                nodes_opened=counters.nodes_opened,
                leaves_evaluated=counters.leaves_evaluated,
                best_cost=best,
                best_haplotype=None,
            )
        next_cursor = None
        if raw_cursor is not None:
            frames = raw_cursor
            next_cursor = EnumerationCursor(
                fingerprint=fingerprint,
                next_bits=tuple(frame[0] for frame in frames),
                path=tuple(frame[2] for frame in frames),
                emitted=(cursor.emitted if cursor else 0) + len(batch),
            )

    emitted_total = (cursor.emitted if cursor else 0) + len(batch)
    stats = SearchStats(
        depth=L,
        budget=max_nodes,
        completed=exhausted,
        nodes_opened=counters.nodes_opened,
        leaves_evaluated=counters.leaves_evaluated,
        branches_pruned=counters.branches_pruned,
        incumbent_updates=incumbent_updates,
        nodes_by_depth=tuple(counters.nodes_by_depth),
        pruned_by_depth=tuple(counters.pruned_by_depth),
        seed_costs=tuple(sorted(seed_costs)),
    )
    return OptimalBatch(
        optimal_cost=best,
        haplotypes=tuple(batch),
        exhausted=exhausted,
        cursor=next_cursor,
        emitted_total=emitted_total,
        search=stats,
    )
