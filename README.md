# haplophase — 跨片段二倍体单倍型定相

离线、无第三方依赖的**加权最小纠错（weighted MEC）单倍型定相**库。对二元杂合位点
和带缺失的片段读段，用**分支界限（branch and bound）**精确求互补单倍型对
`h` 与 `1-h`，固定 `h[0] = 0` 消除互补对称；不是逐位独立投票。
另提供 **Boltzmann 后验入口**：候选 `h` 按 `exp(-cost/temperature)` 加权，
报告每个位点取 1 的边缘概率、归一常数对数与整体熵（见下文“后验接口”）。

- Python：**3.14.7**（任何 3.10+ 也可运行，开发与验收环境为 3.14.7）
- 依赖：**仅标准库**，安装 Python 后无需 `pip install` 任何东西
- 网络 / 账号 / 密钥 / 数据库 / Docker：全部不需要
- 最优定相（`phase_haplotypes`）全部计算为**精确整数运算，不使用任何浮点**；
  Boltzmann 后验（`phase_posterior`）是唯一返回浮点的路径，其误差口径见
  “后验接口”一节；输入校验对两条路径一致——NaN/Infinity/普通 float 代价一律拒绝

## 环境准备（venv）

只需要 Python 3.14.7 本体，无任何第三方依赖；venv 仅用于隔离解释器：

```bash
python3.14 -m venv .venv        # 或任何 python3 >= 3.10
source .venv/bin/activate
# 无需 pip install：requirements.txt 为空（仅标准库）
```

## 准确命令

```bash
# 必要测试（含 ≤10 位独立全枚举参考，120 个随机用例 + 后验独立枚举对照）
python -m unittest discover -s tests -v

# 固定输入演示：一个正常定相结果 + 实际触发的输入拒绝 + 预算耗尽异常
python demo.py

# 后验演示：最优解相同但后验不确定性不同的两组数据 + 非法温度拒绝
python demo_iteration.py
```

演示真实执行分支界限并打印搜索计数，不 sleep、不预录结果，在普通笔记本上
远低于 8 秒（实测约 0.1 秒）；后验演示现场枚举连通块并求和，实测远低于
90 秒（约 0.05 秒）。

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

## 后验接口（Boltzmann 加权）

```python
from haplophase import phase_posterior

post = phase_posterior(
    loci,
    observations,          # 与 phase_haplotypes 完全相同的契约与校验
    *,
    temperature=1.0,       # 必须为正数；int/float 均可
)
```

候选空间与最优定相完全一致：含位点 0 的连通块固定 `h[0] = 0`，每个自由块
（不含位点 0 的连通分量）的两种整体翻转都算作**不同候选**——与
`optimal_haplotypes` 数出 `2^自由块数` 个最优解是同一套计数，不存在
“枚举两份再随手除 2”。每个候选权重为 `exp(-cost/temperature)`。

返回冻结 dataclass `PosteriorResult`：

| 字段 | 含义 |
|---|---|
| `marginals` | `marginals[k]` = 后验下 `h[k]=1` 的概率；`marginals[0]` 恒为 0.0（规范固定） |
| `log_partition` | 归一常数 Z 的自然对数（log-sum-exp 累计，低温不下溢） |
| `entropy` | 后验香农熵（nats），`H = log Z + mean_cost/temperature` |
| `mean_cost` | 后验平均代价 |
| `temperature` | 实际使用的温度（float） |
| `components` | 参与独立求和的连通块（第一个含位点 0） |

实现要点：

- **分块独立求和**：读段覆盖的位点必落在同一连通块，总代价按块分解，
  `log Z`、熵、平均代价按块相加，边缘概率按块独立计算。
- **对数域累计**：配分函数用 max-shift 的 log-sum-exp；第二遍直接累计
  `exp(-cost/T - log Z) ≤ 1` 的归一化权重，任意低的温度都不会出现
  0/0 或下溢成零除。
- **Gray 码枚举**：每块按二进制反射 Gray 码枚举，翻一位只增量更新涉及
  该位点的读段，运行代价保持精确整数。
- **温度契约**：`temperature` 必须为正且有限；`0`、负数、`NaN`、
  `Infinity`、`bool`、非数值类型一律抛 `InvalidPhasingInputError`。
- **浮点口径**：后验是本库唯一返回浮点的路径。边缘概率与熵是 IEEE 双
  精度结果；极低温（代价间隔/温度 ≫ 1）下指数相消带来约
  `Δcost/T · 1e-16` 量级的指数误差，测试按此口径设定精度，不以宽容差
  掩盖边界错误。

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
5. **歧义结构解释**：对“位点—被同一条读段共同覆盖”建并查集，不含位点 0 的
   连通分量是自由分量，最优解数量恰为 `2^自由分量数`（每个自由分量可整体
   取反）。
6. **精确性**：全部代价是 Python 任意精度整数，无浮点、无容差。
7. **独立验收**：`tests/reference.py` 用 `itertools.product` 对 ≤10 位实例做
   全枚举，公式独立编写，不导入被测核心来产生期望值；120 个带固定种子的
   随机实例（含全缺失读段、非均匀权重）逐一核对最优值、全部最优解、严格次优、
   gap 与每读段归属。

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
- 后验温度是**建模旋钮，未经真实校准**：不声称对应任何真实测序错误率
  或覆盖度概率模型；它只把加权 MEC 目标转成 Boltzmann 权重。
- 后验按连通块精确枚举，单块最坏 `2^17` 个赋值（Gray 码增量求值，
  两遍）；与分支界限同为指数级最坏情况，18 位硬边界同样适用。
- 自由块的两种整体翻转计为不同候选（与最优解计数一致），因此自由块
  内每个位点的边缘概率恒为 0.5，每自由块向熵贡献至多 `log 2`。

## 仓库结构

```
haplophase/__init__.py   # 公共导出
haplophase/core.py       # 校验、分支界限、连通分量、后验分块求和、结果组装
tests/reference.py       # 独立全枚举参考（不调用被测核心求期望值）
tests/test_phase.py      # 最优定相必要测试
tests/test_posterior.py  # 后验必要测试（独立枚举对照 + 结构 + 异常路径）
demo.py                  # 固定输入演示（正常结果 + 拒绝/异常边界）
demo_iteration.py        # 后验演示（同最优解、不同不确定性 + 非法温度）
README.md
```
