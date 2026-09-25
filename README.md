# haplophase — 跨片段二倍体单倍型定相

离线、无第三方依赖的**加权最小纠错（weighted MEC）单倍型定相**库。对二元杂合位点
和带缺失的片段读段，用**分支界限（branch and bound）**精确求互补单倍型对
`h` 与 `1-h`，固定 `h[0] = 0` 消除互补对称；不是逐位独立投票。

- Python：**3.14.7**（任何 3.10+ 也可运行，开发与验收环境为 3.14.7）
- 依赖：**仅标准库**；下面的 venv 用于隔离环境，`pip` 不安装任何包
- 网络 / 账号 / 密钥 / 数据库 / Docker：全部不需要
- 原最优定相路径为**精确整数运算，不使用任何浮点**；新增 Boltzmann 后验按定义
  使用 Python `float`/`math` 在对数域与移位权重中计算，结果为自然对数单位。

## 准确命令

```bash
# Linux 原生 Python 3.14.7；若解释器名为 python3.14，可替换下面的 python
python -m venv .venv
source .venv/bin/activate

# 必要测试（原最优接口；≤10 位独立全枚举参考，120 个随机用例；Boltzmann 后验）
python -m unittest discover -s tests -v

# 本题增量演示：Boltzmann 边缘概率、log Z、熵、非连通块和低温处理
python demo_iteration.py

# 基线最优定相接口演示仍保留
python demo.py
```

两个演示都现场计算，不 sleep、不预录结果；`demo_iteration.py` 在普通笔记本上
远低于 90 秒（通常约 0.1 秒）。

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

### Boltzmann 后验入口

```python
from haplophase import Observation, boltzmann_posterior

posterior = boltzmann_posterior(
    loci,            # 同样接受 1..18 的普通 int
    observations,    # 同样的 Observation 列表
    temperature,     # 正且有限的 int/float；bool、0、负数、NaN/Infinity 拒绝
)
```

在**已经固定全局互补对称**的规范候选空间（`h[0] = 0`）上定义：

```
w(h) = exp(-cost(h) / T)
P(h) = w(h) / Z,      Z = Σ_(h[0]=0) w(h)
posterior.marginals[k] = P(h[k] = 1)
posterior.log_normalizer = log Z
posterior.entropy = -Σ_h P(h) log P(h)
```

`BoltzmannComponentResult` 分别给出每个片段重叠连通块的 `sites`、`anchored`、
`configuration_count`、局部位点边缘概率、局部 `log_normalizer` 与局部熵。
全局 `log Z` 与熵是各独立连通块之和。翻转自由度的计数是：

- 含位点 0 的锚定块：首位固定为 0，有 `2^(块大小-1)` 个规范配置；
- 其他块：保留完整局部位点翻转自由度，有 `2^块大小` 个配置；
- 总规范候选数是这些数量的乘积；不会把非锚定块先枚举一半再随手除 2。

所有读段都缺失的位点会形成单点自由块，后验边缘概率恰为 `0.5`。计算先减去整数
最小代价再做 `exp`，并对每个分量独立 `logsumexp`；极低温度下若普通浮点移位权重
全部下溢，则退化为“整数最优配置上的离散均匀极限”，不会执行零除。此时 `-C/T`
本身可能超出 `float` 范围而使 `log Z` 为 `-inf`，但边缘概率与熵仍有限；这是 IEEE
浮点数范围限制，不是概率下溢后重新归一化。

温度只用于构造题设要求的探索分布；本项目不依据真实测序错误率、覆盖度或重组过程
校准该温度，也不把 Boltzmann 概率解释成经过实证校准的基因型置信度。

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
5. **歧义结构解释**：对“位点—被同一条读段共同覆盖”建并查集，不含位点 0 的
   连通分量是自由分量，最优解数量恰为 `2^自由分量数`（每个自由分量可整体
   取反）。
6. **精确性**：全部代价是 Python 任意精度整数，无浮点、无容差。
7. **Boltzmann 分解**：用同一张“位点—被同一条读段共同覆盖”图拆分独立块；锚定块
   固定首位，其他块保留全部局部翻转，逐块枚举后把局部对数归一常数与熵相加。
8. **独立验收**：`tests/reference.py` 用 `itertools.product` 对 ≤10 位实例做
   全枚举，公式独立编写，不导入被测核心来产生期望值；120 个带固定种子的
   随机实例（含全缺失读段、非均匀权重）逐一核对最优值、全部最优解、严格次优、
   gap、每读段归属以及一组独立全局 Boltzmann 后验值；另有全代价相等、极低温、
   非连通块和完全无证据位点的专项测试。

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
  模型、读段质量分或重组/距离惩罚；Boltzmann 温度未经真实数据校准。
- 分支界限为精确算法，最坏情况仍为指数级；上限 18 位是硬边界，超过即拒绝。
- 全部最优解均物化返回；若位点重叠图高度碎裂，其数量可达 `2^17`。
- Boltzmann 后验按连通块枚举规范候选，最大也为 `2^17`；其边缘概率、`log Z`
  和熵使用双精度浮点，受约 1e-16 舍入误差约束；极低温且最优代价很大时，`log Z`
  可能因浮点范围成为 `-inf`，边缘概率与熵仍按整数最优极限给出。

## 仓库结构

```
haplophase/__init__.py   # 公共导出
haplophase/core.py       # 校验、分支界限、连通分量、结果组装
tests/reference.py       # 独立全枚举参考（不调用被测核心求期望值）
tests/test_phase.py      # 必要测试
demo_iteration.py        # 本题 Boltzmann 后验演示
demo.py                  # 基线最优定相固定输入演示
README.md
```
