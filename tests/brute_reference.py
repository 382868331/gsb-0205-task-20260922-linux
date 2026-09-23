"""独立全枚举参考实现 —— 不导入、不调用被测核心 haplophase。

仅用于测试中作为期望值的金标准：n_loci <= 10 时直接枚举全部
2^(n_loci-1) 个代表单倍型（h[0] 固定为 0），按题面定义直接计算
每条读段分给较低翻转代价方后的总代价。

输入故意使用与被测库无关的原生结构：读段为 (locus 0基, allele, cost)
三元组的列表，缺失位点不出现在读段中。测试负责把核心输入转换成
该表示，因此本模块对 haplophase 零依赖。
"""

from __future__ import annotations

from typing import Optional


def read_costs(h: tuple[int, ...], read: list[tuple[int, int, int]]) -> tuple[int, int]:
    """返回 (分给 h 的翻转代价, 分给 1-h 的翻转代价)。"""
    to_h = 0
    to_comp = 0
    for locus0, allele, cost in read:
        if allele != h[locus0]:
            to_h += cost
        else:
            to_comp += cost
    return to_h, to_comp


def total_cost(h: tuple[int, ...], reads: list[list[tuple[int, int, int]]]) -> int:
    total = 0
    for read in reads:
        a, b = read_costs(h, read)
        total += a if a < b else b
    return total


def enumerate_all(
    n_loci: int, reads: list[list[tuple[int, int, int]]]
) -> dict:
    """全枚举金标准。

    候选按 (代价, h 字典序) 排序：第 1 个为代表最优 h*；第 2 个 *不同 h*
    即次优——多最优时它是另一个同代价最优（gap=0，刻画歧义），唯一最优
    时它属于严格更大的下一层（gap>0）。
    """
    first_bits: list[tuple[int, ...]] = [
        (0,) + tuple((mask >> i) & 1 for i in range(n_loci - 1))
        for mask in range(1 << (n_loci - 1))
    ]

    scored = sorted((total_cost(h, reads), h) for h in first_bits)
    best_cost, best_h = scored[0]
    opt_hs = tuple(h for c, h in scored if c == best_cost)

    if len(scored) >= 2:
        runner_cost, runner_h = scored[1]  # 必与 best_h 不同（h 两两不同）
    else:
        runner_cost, runner_h = None, None

    return {
        "optimal_cost": best_cost,
        "representative_h": best_h,
        "all_optimal_h": opt_hs,
        "is_unique": len(opt_hs) == 1,
        "second_best_cost": runner_cost,
        "second_best_h": runner_h,
        "gap": None if runner_cost is None else runner_cost - best_cost,
    }
