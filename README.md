# haplophase — 跨片段单倍型定相（二倍体分支界限 + 倍性2/3精确枚举）

离线、无第三方依赖的**加权最小纠错（weighted MEC）单倍型定相**库，提供两个入口：

- `phase_haplotypes`：**二倍体**。对二元杂合位点和带缺失的片段读段，用
  **分支界限（branch and bound）**精确求互补单倍型对 `h` 与 `1-h`，固定
  `h[0] = 0` 消除互补对称；不是逐位独立投票。
- `phase_polyploid`：**显式倍性 2 或 3**。位点 ≤6、片段 ≤8 的小规模契约下
  **全枚举**所有拷贝多重集，片段各自分配给错配最少的一个拷贝，并列解按
  单倍型规范序（字典序）排列。

- Python：**3.14.7**（任何 3.10+ 也可运行，开发与验收环境为 3.14.7）
- 依赖：**仅标准库**，安装 Python 后无需 `pip install` 任何东西
- 网络 / 账号 / 密钥 / 数据库 / Docker：全部不需要
- 全部计算为**精确整数运算，不使用任何浮点**（因此不存在浮点误差口径问题；
  唯一接受浮点的路径是输入校验——NaN/Infinity/普通 float 一律拒绝）

## 准确命令

```bash
# 必要测试（二倍体含 ≤10 位独立全枚举参考、120 个随机用例；
# 倍性入口含 ≤4 位独立全枚举参考、60 个随机用例）
python -m unittest discover -s tests -v

# 固定输入演示：正常定相结果 + 倍性3枚举 + 实际触发的输入拒绝 + 预算耗尽异常
python demo.py
```

演示真实执行分支界限与全枚举并打印搜索/枚举计数，不 sleep、不预录结果，
在普通笔记本上远低于 8 秒（实测约 0.4 秒）。

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

## 接口（二倍体入口）

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

## 接口（显式倍性 2/3 入口）

```python
from haplophase import Observation, phase_polyploid

result = phase_polyploid(
    loci,                       # int，1 <= loci <= 6，bool 不接受
    observations,               # Observation 的 list/tuple，长度 <= 8
    *,
    ploidy=2,                   # int，仅接受 2 或 3，bool 不接受
)
```

模型：存在 `ploidy` 条单倍型**拷贝**（每条是长度 `loci` 的 0/1 串），拷贝组成
**无序多重集**——只差拷贝排列的解视为同一个解，且拷贝之间**不要求互补**
（这与二倍体入口不同，见下）。每条片段分配给错配最少的一个拷贝，错配成本
沿用 `Observation.flip_cost`（翻转该片段一个已观测等位的正整数代价）：

```
cost(h_1..h_p) = Σᵢ wᵢ · min_c 错配数(readᵢ, h_c)
```

契约规模很小（位点 ≤6、倍性 ≤3、片段 ≤8，候选多重集最多 C(64+2,3)=45760
个），因此求解是**精确全枚举**：没有启发式、没有搜索预算、不会抛
`SearchBudgetExhaustedError`。全部计算仍是精确整数运算。

返回**冻结 dataclass** `PolyploidPhasingResult`：

| 字段 | 含义 |
|---|---|
| `ploidy` | 本次搜索使用的显式倍性（2 或 3） |
| `haplotypes` | 规范最优解：所有最优解中字典序最小的拷贝元组（拷贝内部已按字典序排序） |
| `optimal_solutions` | **全部**不同的最优解；每个解是排序后的拷贝元组，解与解之间也按字典序排列（并列的规范排序） |
| `optimal_cost` | 最优总代价（整数） |
| `second_best_cost` | 严格次优的不同代价；若所有候选都最优则为 `None` |
| `gap` | `second_best_cost - optimal_cost`，存在时 `gap >= 1`；否则 `None` |
| `assignments` | 针对规范解 `haplotypes` 的每片段归属 `PolyploidReadAssignment` |
| `search` | 枚举状态 `EnumerationStats`（`ploidy`、`candidates_evaluated`、`completed`） |

`PolyploidReadAssignment`：

- `copy`：分配到的拷贝下标（指向 `haplotypes`）；并列时确定性取最小下标；
- `tied`：是否有多个拷贝同时达到最小代价；
- `mismatches` / `costs`：对每个拷贝的错配数 / 加权代价元组；
- `cost`：该片段实际计入的最小代价（`= flip_cost × mismatches[copy]`）；
- `flips`：相对所分配拷贝需要翻转的位点（0 基）。

与二倍体入口的关系：二倍体入口要求两条拷贝**互补**（`h` 与 `1-h`）并固定
`h[0]=0`；倍性入口的 `ploidy=2` 在**全部**无序拷贝对（含非互补对）上优化，
因此其最优代价**不大于**同实例的二倍体最优代价（测试中有严格更小的用例与
随机一致性校验）。两个入口都保留，原有二倍体调用方式不变。

### 错误语义（倍性入口）

- 与二倍体入口共用 `InvalidPhasingInputError`（继承 `ValueError`），错误信息
  定位到参数位置（如 `ploidy`、`observations[0].alleles[1]`）；**不静默纠正**。
- `ploidy` 只接受 int 2/3；`loci` 上限收紧为 6；片段数上限收紧为 8；
  逐片段的等位/代价校验与二倍体契约完全一致（复用同一校验）。
- 校验先于计算，失败调用不修改任何传入对象（无部分变更）。
- 枚举规模由契约硬性限制，总是完整跑完，不存在“预算耗尽”状态。

### 错误语义（二倍体入口）

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

二倍体入口（`phase_haplotypes`）：

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

倍性入口（`phase_polyploid`）：

1. **枚举而非搜索**：候选是 `ploidy` 条拷贝的无序多重集，用
   `itertools.combinations_with_replacement` 每个多重集恰好枚举一次；
   每读段对每个候选拷贝的加权错配数预先建表，枚举时只做整数 `min` 求和。
2. **规范并列排序**：拷贝解码为位元组后在每个解内部排序，解集整体再按
   字典序排序；`haplotypes` 取字典序最小者。注意排序键是**位元组**而不是
   整数编码（整数编码的位序与小端字典序不一致）。
3. **独立验收**：`tests/reference_polyploid.py` 用 `itertools.product` 枚举
   全部有序拷贝元组再规范化去重，代价公式独立编写；60 个带固定种子的随机
   实例（≤4 位、倍性 2/3、含全缺失读段）逐一核对最优值、全部最优解、严格
   次优、gap、每读段归属与枚举计数公式 `C(2^L + p − 1, p)`。另有一组手算
   用例（三倍体唯一零代价解、并列规范序、缺失位点导出的 5 个最优解、权重
   决定胜负、全缺失读段中性）与二倍体代价的解析关系校验
   （`ploidy=2` 最优 ≤ 二倍体最优）。

## 输入上限（接口边界）

二倍体入口 `phase_haplotypes`：

- `loci`：1..**18**；
- 读段数：0..**200**；
- 每条读段 `alleles` 长度必须等于 `loci`；
- `flip_cost`：正整数（Python 大整数，无单独上限）。

倍性入口 `phase_polyploid`：

- `loci`：1..**6**；
- 读段数：0..**8**；
- `ploidy`：**2 或 3**；
- 每条读段 `alleles` 长度必须等于 `loci`；`flip_cost` 契约同上。

规模为 18 位 / 200 读段的稠密随机“坏味道”实例实测约 5–7 秒完成（纯 CPython）。
倍性入口上限规模（6 位 / 8 读段 / 倍性 3，45760 个候选）实测约 0.3 秒。
短演示不穷举上限规模。

## 已知限制（按本题契约实现的数学子集）

- 只支持**二元**杂合位点与单条读段内无相位不确定性的 0/1/缺失模型；不实现
  多等位基因、读段配对/插入长度分布或任何行业标准文件格式
  （不宣称 VCF/BAM/SAM 等兼容性）。
- 倍性入口只支持**显式倍性 2 或 3**，且拷贝数沿位点不变；不实现 4 倍体及
  以上、不实现可恢复的中断搜索（枚举规模由契约限制，总是完整跑完）。
- 倍性入口的拷贝是**无序多重集且不要求互补**；需要互补二倍体语义时请用
  `phase_haplotypes`。
- 目标是本题定义的“每读段分配给较低翻转代价者”的加权 MEC；不含覆盖度概率
  模型、读段质量分或重组/距离惩罚。
- 二倍体分支界限为精确算法，最坏情况仍为指数级；上限 18 位是硬边界，超过
  即拒绝。倍性枚举同样精确，规模上限 6 位 / 8 片段 / 倍性 3 是硬边界。
- 全部最优解均物化返回；二倍体入口在最坏情况下数量可达 `2^17`，倍性入口
  最多 `C(2^L + p − 1, p)` 个（上限规模即 45760 个）。

## 仓库结构

```
haplophase/__init__.py        # 公共导出
haplophase/core.py            # 二倍体：校验、分支界限、连通分量、结果组装
haplophase/polyploid.py       # 倍性2/3：校验复用、精确枚举、规范排序、结果组装
tests/reference.py            # 二倍体独立全枚举参考（不调用被测核心求期望值）
tests/test_phase.py           # 二倍体必要测试
tests/reference_polyploid.py  # 倍性入口独立全枚举参考（独立代价公式）
tests/test_polyploid.py       # 倍性入口正常/边界/失败测试
demo.py                       # 固定输入演示（正常结果 + 倍性3枚举 + 拒绝/异常边界）
README.md
```
