# haplophase — 跨片段单倍型定相（二倍体 + 显式倍性 2/3）

离线、无第三方依赖的**加权最小纠错（weighted MEC）单倍型定相**库，提供两个入口：

- `phase_haplotypes`：**二倍体**。对二元杂合位点和带缺失的片段读段，用
  **分支界限（branch and bound）**精确求互补单倍型对 `h` 与 `1-h`，固定
  `h[0] = 0` 消除互补对称；不是逐位独立投票。
- `phase_polyploid`：**显式倍性 2 或 3**（本次迭代新增）。求 `ploidy` 个拷贝
  单倍型，每条片段分配给错配最少的**某一个**拷贝，总整数错配成本最小；
  候选空间完全枚举（≤6 位点、≤8 片段），并列按单倍型规范排序。

- Python：**3.14.7**（任何 3.10+ 也可运行，开发与验收环境为 3.14.7）
- 依赖：**仅标准库**，安装 Python 后无需 `pip install` 任何东西
- 网络 / 账号 / 密钥 / 数据库 / Docker：全部不需要
- 全部计算为**精确整数运算，不使用任何浮点**（因此不存在浮点误差口径问题；
  唯一接受浮点的路径是输入校验——NaN/Infinity/普通 float 一律拒绝）

## 准确命令

```bash
# 必要测试（二倍体 ≤10 位独立全枚举参考 120 例 + 多倍体独立全积参考 83 例）
python -m unittest discover -s tests -v

# 固定输入演示：正常定相结果 + 多倍体正常/并列结果 + 实际触发的输入拒绝与预算异常
python demo.py
```

演示真实执行分支界限与完整枚举并打印搜索/枚举计数，不 sleep、不预录结果，
在普通笔记本上远低于 90 秒（实测约 0.1 秒）。

## 问题定义（二倍体入口）

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

## 问题定义（多倍体入口）

给定显式倍性 `p ∈ {2, 3}`，求 `p` 个拷贝单倍型 `H = (h₁, …, hₚ)`（每个是
长度 L 的 0/1 向量，拷贝可相同），每条读段分配给错配最少的一个拷贝：

```
aᵢ(H) = wᵢ · min_c ( 读段 i 对 h_c 的失配数 )     （缺失位点不计）
目标   = min_H Σᵢ aᵢ(H)
```

注意与二倍体入口的语义差别：这里 `p=2` 时两个拷贝**不必互补**，是二倍体
模型的松弛（例如读段 `00/00/01`：一般二拷贝模型取拷贝 `00`+`01` 代价 0，
互补模型 `h`/`1-h` 代价 1）。二倍体最优对 `{h, 1-h}` 恒为多倍体模型的
可行解，因此 `多倍体最优代价 ≤ 二倍体最优代价`（一致性测试逐例核对）。

**规范排序**：拷贝可互换，解一律表示为字典序升序的拷贝元组（每个无序
拷贝多重集只出现一次）；全部最优解按该规范序排列，`haplotypes` 取其中
字典序最小者。

## 接口

```python
from haplophase import Observation, phase_haplotypes, phase_polyploid

result = phase_haplotypes(
    loci,                       # int，1 <= loci <= 18，bool 不接受
    observations,               # Observation 的 list/tuple，长度 <= 200
    *,
    max_nodes=None,             # 可选正整数搜索节点预算
)

poly = phase_polyploid(
    loci,                       # int，1 <= loci <= 6，bool 不接受
    observations,               # Observation 的 list/tuple，长度 <= 8
    *,
    ploidy,                     # 必填关键字参数，int，仅 2 或 3
)
```

`Observation(alleles=(0, 1, None, ...), flip_cost=3)`（两个入口共用）：

- `alleles`：list/tuple，长度必须等于 `loci`；元素仅允许 `0`、`1`、`None`；
- `flip_cost`：正整数 `>= 1`；零、负数、float（含 `NaN`/`Infinity`）、`bool` 拒绝。

### 二倍体返回 `PhasingResult`（冻结 dataclass）

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

### 多倍体返回 `PolyploidPhasingResult`（冻结 dataclass，本次新增）

| 字段 | 含义 |
|---|---|
| `ploidy` | 本次调用的倍性（2 或 3） |
| `haplotypes` | 规范最优解：字典序最小的升序拷贝元组 |
| `optimal_solutions` | **全部**最优解，每个为升序拷贝元组，整体按规范序排列 |
| `optimal_cost` | 最优总代价（整数） |
| `second_best_cost` | 严格次优解的代价；所有候选都最优时为 `None` |
| `gap` | `second_best_cost - optimal_cost` 或 `None` |
| `assignments` | 每条读段的 `CopyAssignment`（针对规范解） |
| `candidates_evaluated` | 枚举的规范候选数，等于 `C(2**loci + ploidy - 1, ploidy)` |

`CopyAssignment`：

- `assigned`：分配到的拷贝下标（0 基）；并列时确定性取最小下标；
- `tied`：是否有多个拷贝同时达到最小代价；
- `costs`：该读段对每个拷贝的精确加权错配代价元组；
- `flips`：对每个拷贝需要翻转的位点（0 基）元组；
- `flips_applied`：实际采用拷贝的翻转集合。

多倍体入口是**完全枚举**（最大 `C(66,3) = 45760` 个候选），因此**没有**
`max_nodes` 预算参数，也不会抛出 `SearchBudgetExhaustedError`。

### 错误语义

- `InvalidPhasingInputError`（继承 `ValueError`）：违反契约即拒绝，错误信息
  定位到参数位置，如 `observations[3].alleles[7]`、`ploidy: must be one of
  (2, 3), got 4`；**不静默纠正**。
  数值输入拒绝 `NaN`/`Infinity` 与一切 float；整数参数拒绝 `bool`。
  校验先于计算，失败调用不修改任何传入对象（无部分变更）。
- `SearchBudgetExhaustedError`（继承 `RuntimeError`，**不是** `ValueError`）：
  仅二倍体入口在设置 `max_nodes` 且预算耗尽、最优性尚未证明时抛出，附带已打开
  节点数和一个**可行但未证明最优**的在位解（其代价经精确计算）。
- **预算耗尽 ≠ 证明无解**：本问题任何通过校验的实例都可行（候选空间有限且全部
  容许），不存在“无解”；预算异常只表示“未证完”。去掉 `max_nodes` 重跑即可。

## 算法与设计取舍

二倍体入口：

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
5. **歧义结构解释**：对“位点—被同一条读段共同覆盖”建并查集，不含位点 0 的
   连通分量是自由分量，最优解数量恰为 `2^自由分量数`（每个自由分量可整体
   取反）。
6. **精确性**：全部代价是 Python 任意精度整数，无浮点、无容差。

多倍体入口（本次新增）：

1. **候选编码**：长度 L 的 0/1 单倍型编码为整数 `0..2^L−1`（最高位为位点 0），
   整数序与位向量字典序一致。
2. **完全枚举**：用 `itertools.combinations_with_replacement` 只枚举升序拷贝
   元组（拷贝可互换、可相同），候选数 `C(2^L + p − 1, p)`，上限
   `C(66, 3) = 45760`；每个候选的代价由预计算的“读段 × 单倍型”失配表
   `O(读段数 × 倍性)` 查表得到。
3. **并列规范排序**：枚举顺序即规范序，等于最优代价的候选全部收集；
   `haplotypes` 为其中字典序最小者，`optimal_solutions` 为全部最优解。
4. **无搜索预算**：枚举有硬上界且规模小，不存在预算耗尽状态。
5. **独立验收**：`tests/reference.py` 的多倍体参考**不**使用上述规范枚举，
   而是用 `itertools.product` 枚举全部 `2^(L·p)` 个**有序**拷贝元组再排序
   去重，公式独立编写，不导入被测核心产生期望值；83 个固定种子随机实例
   （含 6 位点 3 倍体的接口上限实例）逐一核对最优值、全部最优解、严格次优、
   gap 与每读段归属。另有一致性测试核对 `ploidy=2` 与二倍体入口的松弛关系
   （仅作补充，不作为期望值来源）。

## 输入上限（接口边界）

- 二倍体入口 `phase_haplotypes`：`loci` 1..**18**，读段 0..**200**；
- 多倍体入口 `phase_polyploid`：`loci` 1..**6**，读段 0..**8**，
  `ploidy` 仅 **2 或 3**；
- 每条读段 `alleles` 长度必须等于 `loci`；
- `flip_cost`：正整数（Python 大整数，无单独上限）。

规模为 18 位 / 200 读段的稠密随机“坏味道”实例实测约 5–7 秒完成（纯 CPython）。
多倍体上限实例（6 位点 / 8 读段 / 3 倍体）实测约 0.1 秒。短演示不穷举上限规模。

## 已知限制（按本题契约实现的数学子集）

- 只支持**二元**杂合位点与单条读段内无相位不确定性的 0/1/缺失模型；不实现
  多等位基因、读段配对/插入长度分布或任何行业标准文件格式
  （不宣称 VCF/BAM/SAM 等兼容性）。
- 倍性仅支持显式 **2 或 3**（4 倍体及以上、拷贝数沿位点变化均不在本次
  交付范围）；多倍体入口不做可恢复搜索（完全枚举无需预算与恢复）。
- 目标是本题定义的“每读段分配给较低翻转代价者”的加权 MEC；不含覆盖度概率
  模型、读段质量分或重组/距离惩罚。
- 二倍体分支界限为精确算法，最坏情况仍为指数级；上限 18 位是硬边界，超过即拒绝。
- 全部最优解均物化返回；二倍体入口若位点重叠图高度碎裂，其数量可达 `2^17`。

## 仓库结构

```
haplophase/__init__.py   # 公共导出
haplophase/core.py       # 校验、分支界限（二倍体）、完全枚举（多倍体）、结果组装
tests/reference.py       # 独立全枚举参考（不调用被测核心求期望值）
tests/test_phase.py      # 二倍体必要测试
tests/test_polyploid.py  # 多倍体测试（手算例、独立参考对照、校验、边界）
demo.py                  # 固定输入演示（正常结果 + 多倍体 + 拒绝/异常边界）
README.md
```
