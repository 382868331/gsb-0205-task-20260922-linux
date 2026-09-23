"""haplophase — 跨片段二倍体单倍型定相（可复用、纯标准库）。

问题契约（见 README / TASK.md）：

* 二元杂合位点数 ``n_loci``（1..18，位点编号 1..n_loci）；
* 读段数 1..200，每条读段在若干位点上给出 0/1 观测，未观测即缺失；
* 每个观测带正整数翻转代价；
* 求互补单倍型对 (h, 1-h)，把每条读段分配给翻转代价较低的一方，
  最小化总翻转代价；固定 h 的第一位为 0 以消除互补对称。

核心算法是全局分支界限（B&B），不是逐位独立投票：一次完整 DFS 枚举
同时得出最优 h 集合与“次优不同 h”的代价。模块为无状态函数式接口：
全部输入校验先于任何搜索发生；搜索预算耗尽抛
:class:`SearchBudgetExhausted`，与非法输入（:class:`InvalidInputError`）
严格分开。全部运算使用整数，不存在浮点误差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

__all__ = [
    "MAX_LOCI",
    "MAX_READS",
    "AlleleCall",
    "ReadAssignment",
    "PhaseStats",
    "SearchState",
    "PhasingResult",
    "HaplophaseError",
    "InvalidInputError",
    "SearchBudgetExhausted",
    "phase_haplotypes",
]

MAX_LOCI = 18
MAX_READS = 200


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlleleCall:
    """一条读段在单个位点上的观测。

    ``locus`` 为 1 基位点编号（1..n_loci）；``allele`` 为 0/1；
    ``flip_cost`` 为该观测被翻转时的正整数代价。读段中未列出的位点
    一律视为缺失。
    """

    locus: int
    allele: int
    flip_cost: int


@dataclass(frozen=True)
class ReadAssignment:
    """一条读段相对于最优代表单倍型 h* 的最优分配。

    ``assigned``：0 表示分配给 h*，1 表示分配给 1-h*，``None`` 表示
    双方代价并列；``flipped_loci`` 为被分配一方需要翻转的位点（并列时
    两种分配等价，返回空元组）。
    """

    index: int
    assigned: Optional[int]
    cost_to_h0: int
    cost_to_h1: int
    flipped_loci: tuple[int, ...]


@dataclass(frozen=True)
class PhaseStats:
    """分支界限搜索的状态统计。"""

    visited_nodes: int
    leaves_evaluated: int
    pruned_nodes: int
    incumbent_updates: int
    completed: bool

    def __str__(self) -> str:
        return (
            f"visited={self.visited_nodes}, leaves={self.leaves_evaluated}, "
            f"pruned={self.pruned_nodes}, "
            f"incumbent_updates={self.incumbent_updates}, "
            f"completed={self.completed}"
        )


@dataclass(frozen=True)
class SearchState:
    """分支界限搜索状态（单次 DFS 同时产出最优与次优不同 h）。"""

    node_budget: int
    nodes_used: int
    locus_order: tuple[int, ...]
    stats: PhaseStats

    def __str__(self) -> str:
        return (
            f"node_budget={self.node_budget}, nodes_used={self.nodes_used}\n"
            f"locus_search_order={self.locus_order} (1-based; 第 1 位固定为 0)\n"
            f"{self.stats}"
        )


@dataclass(frozen=True)
class PhasingResult:
    """定相结果。

    ================ ======================================================
    字段             含义
    ================ ======================================================
    h                最优代表单倍型（h[0] 固定为 0），0/1 元组
    complement       1-h
    optimal_cost     最优总翻转代价（整数，精确值）
    is_unique        最优 h 是否唯一
    all_optimal_h    全部最优 h（按位掩码升序）；不唯一时有多个
    second_best_h    次优的“不同 h”：多最优时为另一个同代价最优
                     （此时 gap=0，直接刻画定相歧义）；唯一最优时为
                     严格下一层（gap>0）；候选唯一（n_loci=1）时为 None
    second_best_cost 次优不同 h 的总代价
    gap              second_best_cost - optimal_cost（非负整数）
    assignments      每条读段的分配 / 并列 / 翻转位点
    modified_positions  (读段下标, 位点) 列表，即所有非并列读段的翻转位置
    search           分支界限搜索状态
    ================ ======================================================

    当存在多个并列最优 h 时，``assignments`` 与 ``modified_positions``
    相对于位掩码最小的代表 h* 给出。
    """

    h: tuple[int, ...]
    complement: tuple[int, ...]
    optimal_cost: int
    is_unique: bool
    all_optimal_h: tuple[tuple[int, ...], ...]
    second_best_h: Optional[tuple[int, ...]]
    second_best_cost: Optional[int]
    gap: Optional[int]
    assignments: tuple[ReadAssignment, ...]
    modified_positions: tuple[tuple[int, int], ...]
    search: SearchState


# ---------------------------------------------------------------------------
# 错误语义
# ---------------------------------------------------------------------------


class HaplophaseError(Exception):
    """本模块所有错误的基类。"""


class InvalidInputError(HaplophaseError, ValueError):
    """非法输入。消息中带有读段下标 / 位点等可定位信息，不静默纠正。"""


class SearchBudgetExhausted(HaplophaseError):
    """搜索节点预算耗尽（区别于非法输入与“已证明无解”）。

    本问题在合法输入下可行解恒存在（枚举空间非空），因此预算耗尽绝不
    会被伪装成最优解返回；``best_known_*`` 只是搜索中断时的在任值
    （incumbent），不具最优性保证。
    """

    def __init__(
        self,
        message: str,
        *,
        nodes_used: int,
        node_budget: int,
        best_known_cost: Optional[int],
        best_known_h: Optional[tuple[int, ...]],
        phase_stats: PhaseStats,
    ) -> None:
        super().__init__(message)
        self.nodes_used = nodes_used
        self.node_budget = node_budget
        self.best_known_cost = best_known_cost
        self.best_known_h = best_known_h
        self.phase_stats = phase_stats


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _reject_bool_or_nonint(value, where: str) -> None:
    """整数参数拒绝 bool；浮点数按本契约（整数域）拒绝并先行拦截 NaN/Inf。"""
    if isinstance(value, bool):
        raise InvalidInputError(f"{where} 必须是整数，但收到了 bool: {value!r}")
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise InvalidInputError(
                f"{where} 必须是有限数，但收到了非有限值: {value!r}"
            )
        raise InvalidInputError(f"{where} 必须是整数，但收到了浮点数: {value!r}")
    if not isinstance(value, int):
        raise InvalidInputError(
            f"{where} 必须是整数，但收到了 {type(value).__name__}"
        )


def _validate_and_normalize(
    reads: Iterable[Iterable[AlleleCall]],
    n_loci: int,
) -> list[list[AlleleCall]]:
    _reject_bool_or_nonint(n_loci, "n_loci")
    if not 1 <= n_loci <= MAX_LOCI:
        raise InvalidInputError(f"n_loci 必须在 1..{MAX_LOCI} 之间，收到 {n_loci}")

    if isinstance(reads, (str, bytes)) or not isinstance(reads, Iterable):
        raise InvalidInputError("reads 必须是由读段组成的可迭代对象")

    normalized: list[list[AlleleCall]] = []
    for r_idx, raw_read in enumerate(reads):
        where_read = f"reads[{r_idx}]"
        if isinstance(raw_read, (str, bytes)) or not isinstance(raw_read, Iterable):
            raise InvalidInputError(
                f"{where_read} 必须是由 AlleleCall 组成的可迭代对象"
            )
        calls: list[AlleleCall] = []
        seen_loci: set[int] = set()
        for c_idx, call in enumerate(raw_read):
            where = f"{where_read}[{c_idx}]"
            if not isinstance(call, AlleleCall):
                raise InvalidInputError(
                    f"{where} 必须是 AlleleCall，收到 {type(call).__name__}"
                )
            _reject_bool_or_nonint(call.locus, f"{where}.locus")
            _reject_bool_or_nonint(call.allele, f"{where}.allele")
            _reject_bool_or_nonint(call.flip_cost, f"{where}.flip_cost")
            if not 1 <= call.locus <= n_loci:
                raise InvalidInputError(
                    f"{where}.locus={call.locus} 超出 1..{n_loci}"
                )
            if call.allele not in (0, 1):
                raise InvalidInputError(
                    f"{where}.allele 必须是 0 或 1，收到 {call.allele}"
                )
            if call.flip_cost <= 0:
                raise InvalidInputError(
                    f"{where}.flip_cost 必须是正整数，收到 {call.flip_cost}"
                )
            if call.locus in seen_loci:
                raise InvalidInputError(
                    f"{where_read} 在位点 {call.locus} 上存在重复观测"
                )
            seen_loci.add(call.locus)
            calls.append(call)
        normalized.append(calls)

    if not normalized:
        raise InvalidInputError(
            "reads 至少需要包含 1 条读段（全缺失读段可用空读段表示）"
        )
    if len(normalized) > MAX_READS:
        raise InvalidInputError(f"读段数 {len(normalized)} 超过上限 {MAX_READS}")
    return normalized


# ---------------------------------------------------------------------------
# 目标函数与贪心在任解
# ---------------------------------------------------------------------------


def _total_cost(mask: int, active_reads: list[list[AlleleCall]]) -> int:
    """f(h) 的直接精确计算（贪心初界用，不走 B&B）。"""
    total = 0
    for calls in active_reads:
        w = 0
        mismatch = 0
        for c in calls:
            w += c.flip_cost
            if c.allele != ((mask >> (c.locus - 1)) & 1):
                mismatch += c.flip_cost
        total += mismatch if mismatch < w - mismatch else w - mismatch
    return total


def _greedy_incumbent(
    active_reads: list[list[AlleleCall]], order: list[int]
) -> tuple[int, int]:
    """确定性贪心构造可行 h（第一位恒 0），返回 (mask, 代价) 作初始上界。

    两个起点各做坐标下降：(a) 全 0 起点；(b) 按搜索顺序逐位贪心。它只
    用于收紧 B&B 剪枝上界，绝不参与结果正确性；枚举仍完整，并列解不会
    因上界而丢失。
    """

    def coordinate_descent(start: int) -> tuple[int, int]:
        mask = start
        val = _total_cost(mask, active_reads)
        improved = True
        while improved:
            improved = False
            for L in order[1:]:
                cv = _total_cost(mask ^ (1 << L), active_reads)
                if cv < val:
                    mask ^= 1 << L
                    val = cv
                    improved = True
        return mask, val

    mask = 0
    for L in order[1:]:
        if _total_cost(mask | (1 << L), active_reads) < _total_cost(mask, active_reads):
            mask |= 1 << L
    cands = [coordinate_descent(0), coordinate_descent(mask)]
    return min(cands, key=lambda t: (t[1], t[0]))


# ---------------------------------------------------------------------------
# 分支界限核心
# ---------------------------------------------------------------------------


class _BudgetExhausted(Exception):
    """内部信号：节点预算耗尽（公开入口转换为 SearchBudgetExhausted）。"""


def _branch_and_bound(
    n_loci: int,
    active_reads: list[list[AlleleCall]],
    order: list[int],
    budget: list[int],
    counters: dict,
    seed_mask: int,
    seed_cost: int,
) -> tuple[int, list[int], Optional[int], Optional[int]]:
    """单次 DFS 分支界限，同时求最优集合与次优不同 h。

    目标 f(h) = Σ_r min(m_r, W_r - m_r)（m_r 为读段 r 与 h 的加权匹配，
    W_r 为其观测代价之和）。节点下界：部分前缀下 m_r ∈ [lo, hi]，凹函数
    min(m, W-m) 的区间最小值在端点取得，每读段贡献
    min(lo, W-lo, hi, W-hi)，叶子处恰为真实代价；严格大于在任值才剪枝，
    故最优与任何并列 / 次优候选都不会丢失。

    ``order`` 以位点 1（索引 0）开头，该位在深度 0 固定为 0，不分支。
    ``seed_mask`` / ``seed_cost`` 为可行在任解，只用于提前收紧剪枝；
    统计写入外部传入的 ``counters``。

    返回 ``(best_cost, tie_masks, strict_cost, strict_mask)``，其中
    strict_* 是代价 *严格* 大于最优的最小层（供唯一最优时作次优）；
    多最优情形的“另一个同代价最优 h”由调用方从 tie_masks 取。预算耗尽
    抛 :class:`_BudgetExhausted`。
    """

    R = len(active_reads)
    W = [0] * R
    m = [0] * R
    rem = [0] * R

    depth_at_locus = [-1] * n_loci
    for d, L in enumerate(order):
        depth_at_locus[L] = d

    # 每个深度三个平行数组：触及读段、观测代价、观测等位。
    r_by_d: list[list[int]] = [[] for _ in range(n_loci)]
    w_by_d: list[list[int]] = [[] for _ in range(n_loci)]
    a_by_d: list[list[int]] = [[] for _ in range(n_loci)]
    for r, calls in enumerate(active_reads):
        total = 0
        for c in calls:
            total += c.flip_cost
            d = depth_at_locus[c.locus - 1]
            r_by_d[d].append(r)
            w_by_d[d].append(c.flip_cost)
            a_by_d[d].append(c.allele)
        W[r] = total
        rem[r] = total

    best_cost = seed_cost
    tie_masks: list[int] = [seed_mask]
    # 次优不同 h：代价严格大于最优的最小叶候选（同代价取字典序最小 h）。
    second_cost: Optional[int] = None
    second_mask: Optional[int] = None

    def charge() -> None:
        budget[0] -= 1
        counters["visited"] += 1
        if budget[0] < 0:
            raise _BudgetExhausted()

    def apply(depth: int, bit: int) -> int:
        """固定 depth 位为 bit，返回下界增量；用逆操作回溯，无额外分配。

        内联 min4 以避免热路径上的函数调用（dense 数据每节点触及上百条
        读段，每条读段需前后两次四值取小）。
        """
        rs = r_by_d[depth]
        ws = w_by_d[depth]
        als = a_by_d[depth]
        delta = 0
        mm = m
        rr_arr = rem
        ww = W
        for j in range(len(rs)):
            r = rs[j]
            w = ws[j]
            mr = mm[r]
            rr = rr_arr[r]
            wr = ww[r]
            x1 = wr - mr
            x2 = mr + rr
            x3 = wr - x2
            before = mr
            if x1 < before:
                before = x1
            if x2 < before:
                before = x2
            if x3 < before:
                before = x3
            rr -= w
            if bit == als[j]:
                mr += w
            mm[r] = mr
            rr_arr[r] = rr
            y1 = wr - mr
            y2 = mr + rr
            y3 = wr - y2
            after = mr
            if y1 < after:
                after = y1
            if y2 < after:
                after = y2
            if y3 < after:
                after = y3
            delta += after - before
        return delta

    def undo(depth: int, bit: int) -> None:
        rs, ws, als = r_by_d[depth], w_by_d[depth], a_by_d[depth]
        for j in range(len(rs)):
            r = rs[j]
            rem[r] += ws[j]
            if bit == als[j]:
                m[r] -= ws[j]

    def _consider_second(cost: int, mask: int) -> None:
        nonlocal second_cost, second_mask
        if second_cost is None or cost < second_cost or (
            cost == second_cost
            and _mask_order_key(mask, n_loci) < _mask_order_key(second_mask, n_loci)
        ):
            second_cost, second_mask = cost, mask

    def leaf(prefix: int, cost: int) -> None:
        nonlocal best_cost
        counters["leaves"] += 1
        if cost < best_cost:
            # 旧最优层整体降级为次优候选：该层的 *每一个* 并列掩码都可能
            # 是同代价中字典序最小的次优 h，必须逐一登记，不能只登记代表。
            for mk in tie_masks:
                _consider_second(best_cost, mk)
            best_cost = cost
            tie_masks.clear()
            tie_masks.append(prefix)
            counters["updates"] += 1
        elif cost == best_cost:
            tie_masks.append(prefix)
        else:
            _consider_second(cost, prefix)

    def dfs(depth: int, prefix: int, lb: int) -> None:
        charge()
        # 前 2 个不同代价层的 B&B：只有当子树下界严格大于已知次优值 S
        # 时才能剪掉——该子树不可能改进最优 U，也不可能给出更小的 S。
        # S 尚未找到（=∞）时任何子树都要探索，因此次优是精确证明的，
        # 不是在任估计；代价等于 S 的子树也保留，以便登记同代价中掩码
        # 最小的次优 h。
        if second_cost is not None and lb > second_cost:
            counters["pruned"] += 1
            return
        if depth == n_loci:
            leaf(prefix, lb)
            return
        if depth == 0:
            # 位点 1 固定为 0（order[0] == 0），消除互补对称，不分支。
            assert order[0] == 0
            delta = apply(0, 0)
            dfs(1, 0, lb + delta)
            undo(0, 0)
            return

        locus_bit_index = order[depth]
        # 优先沿在任解取值分支，使上界尽早收紧；两位仍都完整探索。
        pref = (seed_mask >> locus_bit_index) & 1
        delta_pref = apply(depth, pref)
        dfs(depth + 1, prefix | (pref << locus_bit_index), lb + delta_pref)
        undo(depth, pref)
        other = 1 - pref
        delta_other = apply(depth, other)
        dfs(depth + 1, prefix | (other << locus_bit_index), lb + delta_other)
        undo(depth, other)

    dfs(0, 0, 0)
    return best_cost, tie_masks, second_cost, second_mask


def _mask_to_h(mask: int, n_loci: int) -> tuple[int, ...]:
    return tuple((mask >> i) & 1 for i in range(n_loci))


def _mask_order_key(mask: int, n_loci: int) -> tuple[int, ...]:
    """与 h 元组字典序一致的比较键（位点 1 最优先；与整数位序相反）。"""
    return tuple((mask >> i) & 1 for i in range(1, n_loci))


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------


def phase_haplotypes(
    reads: Iterable[Iterable[AlleleCall]],
    *,
    n_loci: int = MAX_LOCI,
    node_budget: int = 10_000_000,
) -> PhasingResult:
    """对跨片段读段做二倍体单倍型定相。

    :param reads: 1..200 条读段，每条为 :class:`AlleleCall` 的可迭代对象；
        空读段表示全缺失读段。
    :param n_loci: 位点数 1..18（位点编号 1..n_loci）。
    :param node_budget: B&B 节点预算（正整数）。耗尽时抛
        :class:`SearchBudgetExhausted`，不会返回未经证明的结果。
    :returns: :class:`PhasingResult`。
    :raises InvalidInputError: 任何非法输入（消息可定位到读段 / 位点）。
    :raises SearchBudgetExhausted: 节点预算耗尽。
    """

    normalized = _validate_and_normalize(reads, n_loci)
    _reject_bool_or_nonint(node_budget, "node_budget")
    if node_budget <= 0:
        raise InvalidInputError(f"node_budget 必须是正整数，收到 {node_budget}")

    # 单观测读段对目标函数恒贡献 0（总可分给匹配方），不参与搜索；
    # 全缺失读段同理。它们仍出现在最终 assignments 中。
    active_reads = [calls for calls in normalized if len(calls) >= 2]

    # 搜索顺序：第 1 位固定在前；其余位点按该位总观测代价降序，
    # 让约束强的位点先分支，尽快拿到紧的在任值与下界。
    weight_at_locus = [0] * n_loci
    for calls in active_reads:
        for c in calls:
            weight_at_locus[c.locus - 1] += c.flip_cost
    free_order = sorted(range(1, n_loci), key=lambda L: (-weight_at_locus[L], L))
    order = [0] + free_order

    seed_mask, seed_cost = _greedy_incumbent(active_reads, order)
    budget = [node_budget]
    counters = {"visited": 0, "leaves": 0, "pruned": 0, "updates": 0}

    try:
        if not active_reads:
            # 没有任何含 >=2 观测的读段：位点之间无耦合，所有 h 代价均 0。
            # 位 0（位点 1）恒为 0，自由位是位点 2..n => 掩码 m<<1。
            best_cost = 0
            ties = sorted(
                (m << 1 for m in range(1 << (n_loci - 1))),
                key=lambda mk: _mask_order_key(mk, n_loci),
            )
            strict_cost: Optional[int] = None
            strict_mask: Optional[int] = None
        else:
            best_cost, ties, strict_cost, strict_mask = _branch_and_bound(
                n_loci, active_reads, order, budget, counters, seed_mask, seed_cost
            )
    except _BudgetExhausted:
        used = node_budget - max(budget[0], 0)
        raise SearchBudgetExhausted(
            f"分支界限搜索预算耗尽：已访问节点达到预算 {node_budget}，"
            f"未能证明最优即中断（合法输入下本问题恒有可行解，这与"
            f"“证明无解”及非法输入均不同；在任值不保证最优）",
            nodes_used=used,
            node_budget=node_budget,
            best_known_cost=seed_cost,
            best_known_h=_mask_to_h(seed_mask, n_loci),
            phase_stats=PhaseStats(
                visited_nodes=counters["visited"],
                leaves_evaluated=counters["leaves"],
                pruned_nodes=counters["pruned"],
                incumbent_updates=counters["updates"],
                completed=False,
            ),
        ) from None

    nodes_used = node_budget - budget[0]
    state = SearchState(
        node_budget=node_budget,
        nodes_used=nodes_used,
        locus_order=tuple(L + 1 for L in order),
        stats=PhaseStats(
            visited_nodes=counters["visited"],
            leaves_evaluated=counters["leaves"],
            pruned_nodes=counters["pruned"],
            incumbent_updates=counters["updates"],
            completed=True,
        ),
    )

    # 种子叶会在 DFS 中被再次访问，tie 掩码需去重；按 h 元组字典序
    # （位点 1 优先）排序，保证确定输出且与独立参考一致。
    ties = sorted(set(ties), key=lambda mk: _mask_order_key(mk, n_loci))
    rep_mask = ties[0]
    h = _mask_to_h(rep_mask, n_loci)
    complement = tuple(1 - b for b in h)
    all_optimal = tuple(_mask_to_h(mk, n_loci) for mk in ties)

    # 次优“不同 h”：多最优时取字典序次小的另一个最优（gap=0，歧义）；
    # 唯一最优时取严格下一层（gap>0）。n_loci==1 时候选只有一个，无次优。
    if n_loci == 1:
        runner_cost: Optional[int] = None
        runner_mask: Optional[int] = None
    elif len(ties) > 1:
        runner_cost, runner_mask = best_cost, ties[1]
    else:
        runner_cost, runner_mask = strict_cost, strict_mask

    assignments: list[ReadAssignment] = []
    modified: list[tuple[int, int]] = []
    for r_idx, calls in enumerate(normalized):
        w_total = 0
        match = 0
        for c in calls:
            w_total += c.flip_cost
            if c.allele == h[c.locus - 1]:
                match += c.flip_cost
        cost_h0 = w_total - match
        cost_h1 = match
        if cost_h0 < cost_h1:
            assigned: Optional[int] = 0
        elif cost_h1 < cost_h0:
            assigned = 1
        else:
            assigned = None
        flips: list[int] = []
        if assigned == 0:
            flips = [c.locus for c in calls if c.allele != h[c.locus - 1]]
        elif assigned == 1:
            flips = [c.locus for c in calls if c.allele == h[c.locus - 1]]
        for L in flips:
            modified.append((r_idx, L))
        assignments.append(
            ReadAssignment(r_idx, assigned, cost_h0, cost_h1, tuple(flips))
        )

    gap = None if runner_cost is None else runner_cost - best_cost
    second_h = None if runner_mask is None else _mask_to_h(runner_mask, n_loci)

    return PhasingResult(
        h=h,
        complement=complement,
        optimal_cost=best_cost,
        is_unique=len(ties) == 1,
        all_optimal_h=all_optimal,
        second_best_h=second_h,
        second_best_cost=runner_cost,
        gap=gap,
        assignments=tuple(assignments),
        modified_positions=tuple(modified),
        search=state,
    )
