# haplophase — 跨片段二倍体单倍型定相

离线、无第三方依赖的**加权最小纠错（weighted MEC）单倍型定相**库。对二元杂合位点
和带缺失的片段读段，用**分支界限（branch and bound）**精确求互补单倍型对
`h` 与 `1-h`，固定 `h[0] = 0` 消除互补对称；不是逐位独立投票。

- Python：**3.14.7**（任何 3.10+ 也可运行，开发与验收环境为 3.14.7）
- 依赖：**仅标准库**，安装 Python 后无需 `pip install` 任何东西
- 网络 / 账号 / 密钥 / 数据库 / Docker：全部不需要
- 全部计算为**精确整数运算，不使用任何浮点**（因此不存在浮点误差口径问题；
  唯一接受浮点的路径是输入校验——NaN/Infinity/普通 float 一律拒绝）

## 准确命令

```bash
# 必要测试（含 ≤10 位原接口全枚举；top-K 对 ≤8 位全枚举参照）
./.venv/bin/python -m unittest discover -s tests -v

# 固定输入演示：top-K、正常定相结果、输入拒绝和预算耗尽状态
./.venv/bin/python demo.py
```

演示真实执行分支界限并打印搜索计数，不 sleep、不预录结果，在普通笔记本上
远低于 90 秒（测试实测约 0.1 秒）。

## 问题定义

位点编号 0..L−1（接口为 0 基；演示打印时换算成 1..L）。每条读段 `i`：

- `alleles`：长度 L 的序列，元素为 `0` / `1` / `None`（缺失）；
- `flip_cost = wᵢ`：翻转该读段**一个**已观测等位的正整数代价。

对候选 `h`，读段分配给两条互补单倍型中翻转代价较小者：

```
aᵢ(h) = wᵢ · min( 对 h 的失配数, 对 1−h 的失配数 )
目标   = min_h Σᵢ aᵢ(h),   h[0] = 0
```

读段内部的 `min` 把多个位点耦合起来，所以逐位多数投票会答错（测试与 demo 均
有反例：投票结果 `00` 的总代价高于真正最优 `01`）。

## 接口

```python
from haplophase import Observation, phase_haplotypes

result = phase_haplotypes(
    loci,                       # int，1 <= loci <= 18，bool 不接受
    observations,               # Observation 的 list/tuple，长度 <= 200
    *,
    max_nodes=None,             # 可选正整数搜索节点预算
)

from haplophase import top_k_haplotypes

top = top_k_haplotypes(
    loci,                       # 同样的二倍体 0/1/缺失输入
    observations,
    5,                          # top_k：1..20，返回不同候选
    max_nodes=None,             # 可选正整数搜索节点预算
)
```

`Observation(alleles=(0, 1, None, ...), flip_cost=3)`：

- `alleles`：list/tuple，长度必须等于 `loci`；元素仅允许 `0`、`1`、`None`；
- `flip_cost`：正整数 `>= 1`；零、负数、float（含 `NaN`/`Infinity`）、`bool` 拒绝。

返回**冻结 dataclass** `PhasingResult`：

| 字段 | 含义 |
|---|---|
| `haplotype` | 规范最优 `h`（所有最优解中字典序最小，`h[0]=0`），int 元组 |
| `complement` | `1-h` |
| `optimal_haplotypes` | **全部**不同的最优 `h`（片段重叠图断开时多于一个） |
| `optimal_cost` | 最优总代价（整数） |
| `second_best_cost` | 严格次优的不同代价；若所有可行 `h` 都最优则为 `None` |
| `gap` | `second_best_cost - optimal_cost`，因整数性存在时 `gap >= 1`；否则 `None` |
| `assignments` | 每条读段的归属 `ReadAssignment`（见下） |
| `free_components` | 位点重叠图中不含位点 0 的连通分量；整分量翻转代价不变 |
| `search` | 分支界限搜索状态 `SearchStats` |

`ReadAssignment`（针对规范最优 `h`）：

- `assigned`：`0`=分配给 `h`，`1`=分配给 `1-h`；
- `tied`：两条链代价相等；并列时 `assigned` 确定性取 `0`，两套翻转都合法；
- `cost_to_h0` / `cost_to_h1`：到两条链的精确加权翻转代价；
- `flips_to_h0` / `flips_to_h1`：需要翻转的位点（0 基）；
- `flips_applied`：实际采用的较省翻转集合（`flips_to_h1` 即“修改位置”）。

`SearchStats`：`depth`、`budget`、`completed`、`nodes_opened`、
`leaves_evaluated`、`branches_pruned`、`incumbent_updates`、
`nodes_by_depth`、`pruned_by_depth`、`seed_costs`（两个确定性初始解代价）。

### top-K 接口

`top_k_haplotypes(loci, observations, top_k, *, max_nodes=None)` 返回冻结
dataclass `TopKResult`：

| 字段 | 含义 |
|---|---|
| `top_k` | 请求的 K；合法整数范围为 `1..20` |
| `candidates` | 前 K 个**不同**候选；不足 K 个时返回全部首位 0 合法串 |
| `completed` / `ranking_proven` | 是否已经证明返回列表排名；预算耗尽时均为 `False` |
| `exhausted` | 是否完整穷尽搜索；不足 K 个候选且搜索完成时为 `True` |
| `remaining_lower_bound` | 预算耗尽时，未证明剩余部分的可采纳下界；证完为 `None` |
| `search` | 与原接口相同口径的节点、叶子、剪枝计数 |

排序键固定为 `(总代价, haplotype 字典序)`。每个 `Candidate` 都包含
`haplotype`、`complement`、`cost` 和该候选自己的 `assignments`；其中每条读段
仍给出 `assigned`、两条链代价、两边修改位置和实际采用的 `flips_applied`。

top-K 分支界限只有在已找到 K 个完整候选，且某分支下界**严格大于**当前第 K 个
代价时才剪掉。下界等于第 K 个代价必须保留，因为同分候选可能按字典序进入列表。
预算耗尽时不抛异常、也不把该下界称为 `gap`：返回已找到的可行候选，同时把
`ranking_proven=False`、`exhausted=False` 和 `remaining_lower_bound` 明确列出。

### 错误语义

- `InvalidPhasingInputError`（继承 `ValueError`）：违反契约即拒绝，错误信息
  定位到参数位置，如 `observations[3].alleles[7]`；**不静默纠正**。
  数值输入拒绝 `NaN`/`Infinity` 与一切 float；整数参数拒绝 `bool`。
  校验先于计算，失败调用不修改任何传入对象（无部分变更）。
- `SearchBudgetExhaustedError`（继承 `RuntimeError`，**不是** `ValueError`）：
  设置 `max_nodes` 且预算耗尽、最优性尚未证明时抛出，附带已打开节点数和一个
  **可行但未证明最优**的在位解（其代价经精确计算）。
- **预算耗尽 ≠ 证明无解**：本问题任何通过校验的实例都可行（候选空间有限且全部
  容许），不存在“无解”；预算异常只表示“未证完”。去掉 `max_nodes` 重跑即可。

## 算法与设计取舍

1. **对称固定**：`h[0] = 0`，搜索空间 `2^(L−1)`（上限 2^17）。
2. **分支**：按位点 0..L−1 深度优先，每位点分支 `h[k]∈{0,1}`，代价较小的
   子节点先展开。
3. **可采纳下界**：已决定前缀上每条读段贡献
   `wᵢ·min(前缀失配数, 前缀非失配覆盖数)`。随位点增加单调不降，叶子处等于
   精确总代价，因此剪枝不会剪掉真正最优。
4. **双门限收集全部最优**：维护最优 `best` 与严格次优 `second`；子树下界
   `> second` 时整枝剪去（该子树不可能含最优，也不可能改进严格次优）；
   等于 `best` 的叶子全部保留。搜索开始前用两个确定性贪心初始解（含一个
   末位翻转变体）提供紧门限；初始解只用于门限，返回值一律来自被完整证明的
   枚举/剪枝过程。
   top-K 搜索沿用同一前缀下界，改维护有序的 K 个完整候选和第 K 个代价门限。
5. **歧义结构解释**：对“位点—被同一条读段共同覆盖”建并查集，不含位点 0 的
   连通分量是自由分量，最优解数量恰为 `2^自由分量数`（每个自由分量可整体
   取反）。
6. **精确性**：全部代价是 Python 任意精度整数，无浮点、无容差。
7. **独立验收**：`tests/reference.py` 用 `itertools.product` 对 ≤10 位实例做
   全枚举，公式独立编写，不导入被测核心来产生期望值；原接口核对 120 个固定种子
   随机实例。top-K 在 ≤8 位逐一枚举所有首位 0 串，覆盖零信息读段、同分候选、
   唯一最优、K=1、K 大于合法解数和预算未证明状态，并核对每个候选的读段归属与
   修改位置。

## 输入上限（接口边界）

- `loci`：1..**18**；
- 读段数：0..**200**；
- 每条读段 `alleles` 长度必须等于 `loci`；
- `flip_cost`：正整数（Python 大整数，无单独上限）。

规模为 18 位 / 200 读段的稠密随机“坏味道”实例实测约 5–7 秒完成（纯 CPython）。
短演示不穷举上限规模。

## 已知限制（按本题契约实现的数学子集）

- 只支持**二元**杂合位点与单条读段内无相位不确定性的 0/1/缺失模型；不实现
  多倍体、多等位基因、读段配对/插入长度分布或任何行业标准文件格式
  （不宣称 VCF/BAM/SAM 等兼容性）。
- 目标是本题定义的“每读段分配给较低翻转代价者”的加权 MEC；不含覆盖度概率
  模型、读段质量分或重组/距离惩罚。
- 分支界限为精确算法，最坏情况仍为指数级；上限 18 位是硬边界，超过即拒绝。
- 全部最优解均物化返回；若位点重叠图高度碎裂，其数量可达 `2^17`。
- top-K 最多物化 20 个候选；每个候选都带完整读段明细，因此读段很多时返回对象
  的大小约为 `K × 读段数`。

## 仓库结构

```
haplophase/__init__.py   # 公共导出
haplophase/core.py       # 校验、分支界限、连通分量、结果组装
tests/reference.py       # 独立全枚举参考（不调用被测核心求期望值）
tests/test_phase.py      # 必要测试
demo.py                  # 固定输入演示（正常结果 + 拒绝/异常边界）
README.md
```
