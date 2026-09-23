"""haplophase 的必要测试。

金标准是 tests/brute_reference.py 中的独立全枚举（n<=10），它不导入、
不调用被测核心；期望值全部来自该独立参考或按题面定义直接重算。
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import haplophase as hp  # noqa: E402
from haplophase import AlleleCall, phase_haplotypes  # noqa: E402
from brute_reference import enumerate_all, read_costs, total_cost  # noqa: E402


def to_raw(n_loci, reads):
    """核心输入 -> 独立参考用的 (locus 0基, allele, cost) 原生表示。"""
    raw = []
    for calls in reads:
        raw.append([(c.locus - 1, c.allele, c.flip_cost) for c in calls])
    return raw


def make_case(rng, n_loci, n_reads, p_empty=0.12, max_cost=7):
    reads = []
    for _ in range(n_reads):
        if rng.random() < p_empty:
            reads.append([])  # 全缺失读段
            continue
        length = rng.randint(1, n_loci)
        loci = rng.sample(range(1, n_loci + 1), length)
        reads.append(
            [AlleleCall(L, rng.randint(0, 1), rng.randint(1, max_cost)) for L in loci]
        )
    return reads


class CoreAgainstBruteForce(unittest.TestCase):
    """B&B 结果必须逐字段等于独立全枚举参考。"""

    def assert_matches_reference(self, reads, n_loci):
        result = phase_haplotypes(reads, n_loci=n_loci)
        ref = enumerate_all(n_loci, to_raw(n_loci, reads))

        self.assertEqual(result.optimal_cost, ref["optimal_cost"])
        self.assertEqual(result.is_unique, ref["is_unique"])
        self.assertEqual(result.all_optimal_h, ref["all_optimal_h"])
        self.assertIn(result.h, set(result.all_optimal_h))
        self.assertEqual(result.h, ref["all_optimal_h"][0])
        self.assertEqual(result.second_best_cost, ref["second_best_cost"])
        self.assertEqual(result.second_best_h, ref["second_best_h"])
        self.assertEqual(result.gap, ref["gap"])

        # 结构不变量：首位固定 0；complement 恰为 1-h。
        self.assertEqual(result.h[0], 0)
        self.assertEqual(result.complement, tuple(1 - b for b in result.h))

        # 目标值直接重算一致。
        self.assertEqual(total_cost(list(result.h), to_raw(n_loci, reads)),
                         result.optimal_cost)

        # 分配 / 翻转位置按题面定义逐读段重算。
        recomputed_total = 0
        for i, raw in enumerate(to_raw(n_loci, reads)):
            a = result.assignments[i]
            self.assertEqual(a.index, i)
            c0, c1 = read_costs(list(result.h), raw)
            self.assertEqual(a.cost_to_h0, c0)
            self.assertEqual(a.cost_to_h1, c1)
            if c0 < c1:
                self.assertEqual(a.assigned, 0)
                self.assertEqual(a.flipped_loci, tuple(L + 1 for L, al, _ in raw
                                                       if al != result.h[L]))
            elif c1 < c0:
                self.assertEqual(a.assigned, 1)
                self.assertEqual(a.flipped_loci, tuple(L + 1 for L, al, _ in raw
                                                       if al == result.h[L]))
            else:
                self.assertIsNone(a.assigned)
                self.assertEqual(a.flipped_loci, ())
            recomputed_total += min(c0, c1)
        self.assertEqual(recomputed_total, result.optimal_cost)

        # modified_positions = 所有非并列读段翻转位点（读段下标, 1基位点）。
        expect_mod = []
        for a in result.assignments:
            for L in a.flipped_loci:
                expect_mod.append((a.index, L))
        self.assertEqual(result.modified_positions, tuple(expect_mod))
        return result

    def test_handcrafted_disconnected_multiple_optima(self):
        # 三条读段覆盖两两不相交的位点集合：含位点 1 的组件方向被锚定，
        # 另外两个组件各自可整体翻转 => 2^2 = 4 个最优。
        reads = [
            [AlleleCall(1, 0, 1), AlleleCall(2, 1, 1)],
            [AlleleCall(3, 0, 2), AlleleCall(4, 1, 2)],
            [AlleleCall(5, 1, 3), AlleleCall(6, 0, 3)],
        ]
        result = self.assert_matches_reference(reads, 6)
        self.assertFalse(result.is_unique)
        self.assertEqual(len(result.all_optimal_h), 4)
        self.assertEqual(result.optimal_cost, 0)

    def test_two_components_gives_two_optima(self):
        # 两个不相交组件：一个含锚定位点 1（方向固定），另一个自由翻转。
        reads = [
            [AlleleCall(1, 0, 1), AlleleCall(2, 1, 1)],
            [AlleleCall(3, 0, 2), AlleleCall(4, 1, 2)],
        ]
        result = self.assert_matches_reference(reads, 4)
        self.assertEqual(len(result.all_optimal_h), 2)

    def test_disconnected_with_unobserved_locus(self):
        # 位点 2 完全无观测；其余位点也互不耦合 => 每位自由 => 8 个最优。
        reads = [
            [],
            [AlleleCall(1, 0, 1)],
            [AlleleCall(3, 1, 1)],
        ]
        result = self.assert_matches_reference(reads, 3)
        self.assertEqual(len(result.all_optimal_h), 4)  # 2^2（含未观测的位点2）

    def test_contradictory_fragments(self):
        # 两条读段在相同位点上支持相反相位；二倍体双方可分别承接，
        # 强权重方主导，弱权重方付出代价。
        reads = [
            [AlleleCall(1, 0, 5), AlleleCall(2, 0, 5), AlleleCall(3, 1, 5)],
            [AlleleCall(1, 1, 1), AlleleCall(2, 1, 1), AlleleCall(3, 0, 1)],
        ]
        result = self.assert_matches_reference(reads, 3)
        # 强读段整体给 h（零代价），弱读段整体给互补方（零代价）=> 0。
        self.assertEqual(result.optimal_cost, 0)
        self.assertTrue(result.is_unique)
        assigns = {a.index: a.assigned for a in result.assignments}
        self.assertEqual(assigns, {0: 0, 1: 1})

    def test_contradiction_forces_flips(self):
        # 三条互相冲突的读段覆盖同样位点，必有一方要翻转。
        reads = [
            [AlleleCall(1, 0, 2), AlleleCall(2, 0, 2)],
            [AlleleCall(1, 1, 2), AlleleCall(2, 1, 2)],
            [AlleleCall(1, 0, 2), AlleleCall(2, 1, 2)],
        ]
        result = self.assert_matches_reference(reads, 2)
        self.assertGreater(result.optimal_cost, 0)

    def test_non_uniform_weights(self):
        # 高代价观测主导定相方向：与均匀权重的逐位投票结论可能不同，
        # 这里只验证 B&B 与全枚举一致（全局加权优化而非独立投票）。
        reads = [
            [AlleleCall(1, 0, 100), AlleleCall(2, 1, 100)],
            [AlleleCall(1, 1, 1), AlleleCall(2, 0, 1)],
            [AlleleCall(2, 0, 3), AlleleCall(3, 1, 3)],
        ]
        result = self.assert_matches_reference(reads, 3)
        self.assertEqual(result.h[1], 1)  # 位点 2 由权重 100 的观测主导
        self.assertEqual(result.optimal_cost, 0)

    def test_all_missing_reads_only(self):
        # 只有全缺失读段：任何 h 代价均为 0，全部并列，分配全部为 tie。
        result = phase_haplotypes([[], [], []], n_loci=3)
        self.assertEqual(result.optimal_cost, 0)
        self.assertEqual(len(result.all_optimal_h), 4)
        self.assertTrue(all(a.assigned is None for a in result.assignments))
        self.assertEqual(result.modified_positions, ())
        # 4 个不同 h 存在 => 次优“不同 h”是另一个同代价最优，gap=0。
        self.assertEqual(result.second_best_h, (0, 0, 1))
        self.assertEqual(result.second_best_cost, 0)
        self.assertEqual(result.gap, 0)
        self.assertFalse(result.is_unique)

    def test_all_missing_single_locus_has_no_runner(self):
        result = phase_haplotypes([[], []], n_loci=1)
        self.assertEqual(result.h, (0,))
        self.assertIsNone(result.second_best_h)
        self.assertIsNone(result.gap)

    def test_single_locus(self):
        result = self.assert_matches_reference(
            [[AlleleCall(1, 1, 5)], [], [AlleleCall(1, 0, 2)]], 1
        )
        self.assertEqual(result.h, (0,))
        self.assertIsNone(result.second_best_h)
        self.assertIsNone(result.gap)

    def test_weighted_assignment_tie(self):
        # 构造在 *最优* h 下某读段分给两方阵价相等（c0==c1）的情形：
        # h=(0,0) 时 r2 恰好一个观测匹配、一个不匹配，且权重相等 => 并列。
        reads = [
            [AlleleCall(1, 0, 4), AlleleCall(2, 0, 4)],
            [AlleleCall(1, 1, 2), AlleleCall(2, 1, 2)],
            [AlleleCall(1, 0, 2), AlleleCall(2, 1, 2)],
        ]
        result = self.assert_matches_reference(reads, 2)
        self.assertEqual(result.h, (0, 0))
        self.assertEqual(result.optimal_cost, 2)
        tie_read = result.assignments[2]
        self.assertIsNone(tie_read.assigned)
        self.assertEqual(tie_read.cost_to_h0, tie_read.cost_to_h1)
        self.assertEqual(tie_read.flipped_loci, ())

    def test_randomized_against_bruteforce(self):
        # 固定种子的大规模（n<=10）随机交叉验证，覆盖各种断开 / 矛盾 /
        # 非均匀权重 / 缺失组合。
        for seed in range(300):
            rng = random.Random(1000 + seed)
            n_loci = rng.randint(1, 10)
            n_reads = rng.randint(1, 40)
            reads = make_case(rng, n_loci, n_reads)
            with self.subTest(seed=seed, n_loci=n_loci, n_reads=n_reads):
                self.assert_matches_reference(reads, n_loci)


class SearchStateTests(unittest.TestCase):
    def test_search_state_exposed_and_consistent(self):
        rng = random.Random(2024)
        reads = make_case(rng, 8, 20)
        result = phase_haplotypes(reads, n_loci=8)
        st = result.search
        self.assertTrue(st.stats.completed)
        self.assertGreaterEqual(st.stats.visited_nodes, st.stats.leaves_evaluated)
        self.assertGreater(st.stats.leaves_evaluated, 0)
        self.assertEqual(st.nodes_used, st.stats.visited_nodes)
        self.assertLessEqual(st.nodes_used, st.node_budget)
        self.assertEqual(st.locus_order[0], 1)  # 首位固定位点排最前
        self.assertEqual(len(st.locus_order), 8)
        # 搜索状态可打印。
        text = str(st)
        self.assertIn("visited", text)

    def test_search_order_high_weight_first(self):
        reads = [
            [AlleleCall(1, 0, 1), AlleleCall(4, 1, 50)],
            [AlleleCall(4, 0, 1), AlleleCall(2, 1, 2)],
        ]
        result = phase_haplotypes(reads, n_loci=4)
        # 位点 1 固定第一；自由位点中观测总代价最大的位点 4 紧随其后。
        self.assertEqual(result.search.locus_order[:2], (1, 4))

    def test_pruning_really_happens(self):
        # 强一致数据应当出现剪枝（证明不是裸全枚举），但叶子数仍覆盖
        # 全部并列 / 次优候选。
        rng = random.Random(77)
        reads = [
            [AlleleCall(L, (L + rng.randint(0, 1)) % 2, rng.randint(1, 9))
             for L in range(1, 11)]
            for _ in range(40)
        ]
        result = phase_haplotypes(reads, n_loci=10)
        self.assertTrue(result.search.stats.completed)
        self.assertGreater(result.search.stats.pruned_nodes, 0)
        # 与独立参考一致（10 位 <= 全枚举上限）。
        self.assert_matches_ref(reads, result, 10)

    def assert_matches_ref(self, reads, result, n_loci):
        ref = enumerate_all(n_loci, to_raw(n_loci, reads))
        self.assertEqual(result.optimal_cost, ref["optimal_cost"])
        self.assertEqual(result.all_optimal_h, ref["all_optimal_h"])
        self.assertEqual(result.second_best_cost, ref["second_best_cost"])
        self.assertEqual(result.second_best_h, ref["second_best_h"])
        self.assertEqual(result.gap, ref["gap"])


class BudgetTests(unittest.TestCase):
    def test_budget_exhaustion_is_distinct_and_raises(self):
        rng = random.Random(55)
        reads = [
            [AlleleCall(L, rng.randint(0, 1), rng.randint(1, 5))
             for L in range(1, 11)]
            for _ in range(30)
        ]
        with self.assertRaises(hp.SearchBudgetExhausted) as ctx:
            phase_haplotypes(reads, n_loci=10, node_budget=3)
        err = ctx.exception
        # 预算耗尽 ≠ 无解 ≠ 非法输入。
        self.assertNotIsInstance(err, hp.InvalidInputError)
        self.assertEqual(err.node_budget, 3)
        self.assertGreaterEqual(err.nodes_used, 1)
        self.assertFalse(err.phase_stats.completed)
        # 中断时在任值仅为诊断信息（可能为 None 之外的贪心解）。
        self.assertIsInstance(err.best_known_cost, int)

    def test_budget_one_allows_only_root(self):
        rng = random.Random(56)
        reads = make_case(rng, 6, 10)
        with self.assertRaises(hp.SearchBudgetExhausted):
            phase_haplotypes(reads, n_loci=6, node_budget=1)

    def test_legal_input_always_has_solution(self):
        # 合法输入下可行解恒存在：给足预算必定成功完成（“无解”情形不存在）。
        for seed in range(20):
            rng = random.Random(seed)
            reads = make_case(rng, rng.randint(1, 10), rng.randint(1, 30))
            result = phase_haplotypes(reads, n_loci=10)
            self.assertTrue(result.search.stats.completed)


class InputValidationTests(unittest.TestCase):
    def expect_invalid(self, reads, n_loci, fragment):
        with self.assertRaises(hp.InvalidInputError) as ctx:
            phase_haplotypes(reads, n_loci=n_loci)
        self.assertIn(fragment, str(ctx.exception))

    def test_n_loci_range(self):
        calls = [AlleleCall(1, 0, 1)]
        for bad in (0, 19, -1):
            with self.assertRaises(hp.InvalidInputError):
                phase_haplotypes(calls, n_loci=bad)

    def test_bool_rejected_as_integer(self):
        with self.assertRaises(hp.InvalidInputError):
            phase_haplotypes([[AlleleCall(True, 0, 1)]], n_loci=3)  # type: ignore[arg-type]
        with self.assertRaises(hp.InvalidInputError):
            phase_haplotypes([[AlleleCall(1, True, 1)]], n_loci=3)  # type: ignore[arg-type]
        with self.assertRaises(hp.InvalidInputError):
            phase_haplotypes([[AlleleCall(1, 0, True)]], n_loci=3)  # type: ignore[arg-type]
        with self.assertRaises(hp.InvalidInputError):
            phase_haplotypes([[AlleleCall(1, 0, 1)]], n_loci=True)  # type: ignore[arg-type]
        with self.assertRaises(hp.InvalidInputError):
            phase_haplotypes([[AlleleCall(1, 0, 1)]], n_loci=3, node_budget=True)

    def test_nan_infinity_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(hp.InvalidInputError):
                phase_haplotypes([[AlleleCall(1, 0, bad)]], n_loci=3)
            with self.assertRaises(hp.InvalidInputError):
                phase_haplotypes([[AlleleCall(1, bad, 1)]], n_loci=3)
        with self.assertRaises(hp.InvalidInputError):
            phase_haplotypes([[AlleleCall(math.nan, 0, 1)]], n_loci=3)
        with self.assertRaises(hp.InvalidInputError):
            phase_haplotypes([[AlleleCall(1, 0, 1)]], n_loci=math.nan)

    def test_locus_out_of_range_is_locatable(self):
        self.expect_invalid(
            [[AlleleCall(4, 0, 1)]], 3, "reads[0][0].locus"
        )

    def test_bad_allele_is_locatable(self):
        self.expect_invalid(
            [[AlleleCall(1, 2, 1)]], 3, "reads[0][0].allele"
        )

    def test_nonpositive_cost_is_locatable(self):
        self.expect_invalid(
            [[AlleleCall(1, 0, 0)]], 3, "reads[0][0].flip_cost"
        )
        self.expect_invalid(
            [[AlleleCall(1, 0, -5)]], 3, "reads[0][0].flip_cost"
        )

    def test_duplicate_locus_in_read(self):
        self.expect_invalid(
            [[AlleleCall(1, 0, 1), AlleleCall(1, 1, 2)]],
            3,
            "重复观测",
        )

    def test_empty_reads_iterable(self):
        with self.assertRaises(hp.InvalidInputError):
            phase_haplotypes([], n_loci=3)

    def test_too_many_reads(self):
        with self.assertRaises(hp.InvalidInputError):
            phase_haplotypes([[] for _ in range(hp.MAX_READS + 1)], n_loci=3)

    def test_boundary_200_reads_accepted(self):
        rng = random.Random(9)
        reads = [
            [AlleleCall(rng.randint(1, 4), rng.randint(0, 1), rng.randint(1, 5))]
            for _ in range(hp.MAX_READS)
        ]
        result = phase_haplotypes(reads, n_loci=4)
        self.assertEqual(len(result.assignments), hp.MAX_READS)

    def test_wrong_element_type(self):
        with self.assertRaises(hp.InvalidInputError):
            phase_haplotypes([[(1, 0, 1)]], n_loci=3)  # 元组而非 AlleleCall
        with self.assertRaises(hp.InvalidInputError):
            phase_haplotypes("not-iterable-of-reads", n_loci=3)  # type: ignore[arg-type]

    def test_bad_budget(self):
        calls = [[AlleleCall(1, 0, 1)]]
        for bad in (0, -1):
            with self.assertRaises(hp.InvalidInputError):
                phase_haplotypes(calls, n_loci=2, node_budget=bad)

    def test_validation_before_search_no_partial_state(self):
        # 非法输入直接抛 InvalidInputError，函数无状态、无部分变更可言。
        with self.assertRaises(hp.InvalidInputError):
            phase_haplotypes([[AlleleCall(1, 0, 1), AlleleCall(1, 1, -1)]], n_loci=3)


class InterfaceBoundaryTests(unittest.TestCase):
    def test_max_loci_small_fast_case(self):
        # 18 位是接口边界；用少量全长读段保证可快速完成。
        reads = [
            [AlleleCall(L, L % 2, 1) for L in range(1, 19)],
            [AlleleCall(L, 1 - (L % 2), 2) for L in range(1, 19)],
            [AlleleCall(1, 0, 1), AlleleCall(18, 1, 3)],
            [],
        ]
        result = phase_haplotypes(reads, n_loci=18)
        self.assertEqual(result.h[0], 0)
        self.assertEqual(len(result.h), 18)
        self.assertEqual(len(result.complement), 18)
        self.assertTrue(result.search.stats.completed)
        # 直接重算目标值。
        raw = to_raw(18, reads)
        self.assertEqual(total_cost(list(result.h), raw), result.optimal_cost)
        self.assertEqual(
            sum(min(*read_costs(list(result.h), r)) for r in raw),
            result.optimal_cost,
        )

    def test_determinism_across_calls(self):
        rng = random.Random(31337)
        reads = make_case(rng, 9, 25)
        r1 = phase_haplotypes(reads, n_loci=9)
        r2 = phase_haplotypes(reads, n_loci=9)
        self.assertEqual(r1.h, r2.h)
        self.assertEqual(r1.all_optimal_h, r2.all_optimal_h)
        self.assertEqual(r1.second_best_h, r2.second_best_h)
        self.assertEqual(
            tuple((a.assigned, a.flipped_loci) for a in r1.assignments),
            tuple((a.assigned, a.flipped_loci) for a in r2.assignments),
        )

    def test_result_is_readonly(self):
        result = phase_haplotypes(
            [[AlleleCall(1, 0, 1), AlleleCall(2, 1, 1)]], n_loci=2
        )
        with self.assertRaises(Exception):
            result.h = (1, 1)  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main(verbosity=2)
