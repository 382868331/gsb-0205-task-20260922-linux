"""haplophase 固定输入演示：真实计算，不使用预录结果或 sleep。

运行：python demo.py

内容：
1. 正常定相：跨片段读段中存在一个与锚定位点断开的组件 => 多个并列最优，
   打印最优 h、互补单倍型、各读段归属 / 并列、修改位置与次优间隔（gap=0
   直接显示歧义），以及分支界限搜索状态。
2. 再给一个唯一最优例，显示 gap>0 的次优不同 h。
3. 实际触发一个拒绝边界（非法翻转代价 0），打印可定位错误；并演示
   搜索预算耗尽与“无解 / 非法输入”是不同异常。
"""

from __future__ import annotations

import time

from haplophase import (
    AlleleCall,
    InvalidInputError,
    SearchBudgetExhausted,
    phase_haplotypes,
)


def _fmt_h(h) -> str:
    return "".join(str(b) for b in h)


def _print_result(title: str, result) -> None:
    print("=" * 72)
    print(title)
    print("-" * 72)
    print(f"最优代表单倍型 h* : {_fmt_h(result.h)}   (首位固定为 0)")
    print(f"互补单倍型   1-h* : {_fmt_h(result.complement)}")
    print(f"最优总翻转代价    : {result.optimal_cost}（整数精确值，无浮点误差）")
    print(f"最优是否唯一      : {result.is_unique}")
    if not result.is_unique:
        print(f"并列最优 h 共 {len(result.all_optimal_h)} 个（定相歧义）:")
        for hh in result.all_optimal_h:
            print(f"    {_fmt_h(hh)}")
    print(
        f"次优不同 h        : {_fmt_h(result.second_best_h)}"
        f"  代价={result.second_best_cost}  代价间隔 gap={result.gap}"
    )
    print("各读段归属（0=h*，1=1-h*，tie=双方代价并列）:")
    for a in result.assignments:
        where = "h*" if a.assigned == 0 else ("1-h*" if a.assigned == 1 else "tie")
        flips = (
            ",".join(str(L) for L in a.flipped_loci) if a.flipped_loci else "-"
        )
        print(
            f"  read[{a.index}] -> {where:<4} "
            f"cost(to h*)={a.cost_to_h0:>2}, cost(to 1-h*)={a.cost_to_h1:>2}, "
            f"翻转位点=[{flips}]"
        )
    print(f"修改位置 (读段下标, 位点): {list(result.modified_positions)}")
    print("分支界限搜索状态:")
    for line in str(result.search).splitlines():
        print(f"  {line}")


def main() -> None:
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1) 正常结果 + 歧义：6 个位点，两条读段各覆盖互不相交的位点对，
    #    第三条读段锚定位点 1 并覆盖位点 5,6。位点 3-4 组件与锚点断开，
    #    可整体翻转 => 两个并列最优，gap=0。
    # ------------------------------------------------------------------
    ambiguous_reads = [
        [AlleleCall(1, 0, 3), AlleleCall(2, 1, 3)],
        [AlleleCall(3, 0, 2), AlleleCall(4, 1, 2)],   # 与位点 1 断开的组件
        [AlleleCall(1, 0, 1), AlleleCall(5, 1, 4), AlleleCall(6, 0, 4)],
        [],  # 一条全缺失读段：对任何 h 代价恒为 0，归属恒为 tie
    ]
    result_amb = phase_haplotypes(ambiguous_reads, n_loci=6)
    _print_result("情形 A：跨片段定相（含断开组件 -> 歧义）", result_amb)

    # ------------------------------------------------------------------
    # 2) 唯一最优 + 真实翻转：三条读段在共享位点上两两冲突，必有一方要
    #    翻转；非均匀权重（9/1）决定方向，是全局加权优化而非逐位投票。
    # ------------------------------------------------------------------
    unique_reads = [
        [AlleleCall(1, 0, 9), AlleleCall(2, 0, 1)],
        [AlleleCall(1, 1, 5), AlleleCall(2, 1, 5)],
        [AlleleCall(1, 0, 5), AlleleCall(2, 1, 6)],
        [AlleleCall(2, 0, 1), AlleleCall(3, 1, 1)],
    ]
    result_uniq = phase_haplotypes(unique_reads, n_loci=3)
    _print_result("情形 B：矛盾片段迫使翻转的唯一最优（非均匀权重）", result_uniq)

    # ------------------------------------------------------------------
    # 3) 实际触发的拒绝边界：翻转代价必须是正整数，0 被拒绝且可定位。
    # ------------------------------------------------------------------
    print("=" * 72)
    print("边界：非法输入必须被拒绝（不静默纠正）")
    print("-" * 72)
    try:
        phase_haplotypes(
            [[AlleleCall(1, 0, 9), AlleleCall(2, 1, 0)]], n_loci=3
        )
    except InvalidInputError as exc:
        print(f"捕获 InvalidInputError -> {exc}")

    # 数值输入拒绝 NaN / Infinity。
    try:
        phase_haplotypes([[AlleleCall(1, 0, float("nan"))]], n_loci=2)
    except InvalidInputError as exc:
        print(f"捕获 InvalidInputError -> {exc}")

    # 整数参数拒绝 bool。
    try:
        phase_haplotypes([[AlleleCall(1, 0, 1)]], n_loci=True)
    except InvalidInputError as exc:
        print(f"捕获 InvalidInputError -> {exc}")

    # ------------------------------------------------------------------
    # 4) 搜索预算耗尽：与非法输入不同的独立异常（合法输入恒有可行解）。
    # ------------------------------------------------------------------
    print("=" * 72)
    print("边界：搜索预算耗尽（区别于非法输入 / 证明无解）")
    print("-" * 72)
    import random

    rng = random.Random(20260923)
    big_reads = [
        [AlleleCall(L, rng.randint(0, 1), rng.randint(1, 5)) for L in range(1, 13)]
        for _ in range(24)
    ]
    try:
        phase_haplotypes(big_reads, n_loci=12, node_budget=7)
    except SearchBudgetExhausted as exc:
        print(f"捕获 SearchBudgetExhausted -> {exc}")
        print(
            f"  已用节点={exc.nodes_used}/{exc.node_budget}, "
            f"中断时在任代价={exc.best_known_cost}（不保证最优，仅诊断）"
        )
    # 给足预算则同一输入正常完成，证明这不是“无解”。
    ok = phase_haplotypes(big_reads, n_loci=12)
    print(
        f"同一输入给足预算后正常完成：最优代价={ok.optimal_cost}，"
        f"completed={ok.search.stats.completed}，实际节点={ok.search.nodes_used}"
    )

    elapsed = time.perf_counter() - t_start
    print("=" * 72)
    print(f"演示完成，实际耗时 {elapsed:.3f}s。")


if __name__ == "__main__":
    main()
