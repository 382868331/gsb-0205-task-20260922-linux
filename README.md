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
# 必要测试（含 ≤10 位独立全枚举参考，120 个随机用例与分批领取测试）
python -m unittest discover -s tests -v

# 固定输入演示：一个正常定相结果 + 实际触发的输入拒绝 + 预算耗尽异常
python demo.py
```

演示真实执行分支界限并打印搜索计数，不 sleep、不预录结果，在普通笔记本上
远低于 8 秒（实测约 0.1 秒）。

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

分批领取接口 `phase_haplotypes_batched(loci, observations, *, batch_size, cursor=None, max_nodes=None)
-> OptimalBatch` 见下文「分批领取最优解」。

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

### 分批领取最优解（不一次性物化全部解）

并列最优解可能多达 `2^17`，`phase_haplotypes` 会一次性物化它们；需要分批
领取时使用有状态续取接口：

```python
from haplophase import phase_haplotypes_batched

cursor = None
while True:
    out = phase_haplotypes_batched(
        loci, observations,
        batch_size=64,     # 正整数；bool 不接受；可逐批改变
        cursor=cursor,     # 首轮传 None；之后传上一批返回的 cursor
        max_nodes=None,    # 可选；只限制本次调用，不改变实例指纹
    )
    for h in out.haplotypes:   # 本批最优解（字典序稳定、互不重复）
        ...
    cursor = out.cursor
    if out.exhausted:          # 仅当搜索真正走完最后一个候选时为 True
        assert out.cursor is None
        assert out.search.completed is True
        break
```

契约要点：

- **先证最优代价，再逐批列举**：每次调用都会独立重新证明 `optimal_cost`
  （可采纳下界的分支界限），随后只列举代价恰好等于最优值的叶子；因此每个批次
  返回的代价都是权威值，且内部从不预先保存全部最优解——续取令牌只保存挂起的
  深度优先搜索位置（每个已展开深度的“下一待试分支 + 进入位”），内存为
  `O(loci)`，任何时刻最多物化一个批次。
- **不重复、不遗漏、顺序稳定**：枚举按字典序深度优先（每位先试 `0` 再试 `1`），
  与 `PhasingResult.optimal_haplotypes` 的顺序完全一致；互补对称仍按原规则
  通过固定 `h[0] = 0` 去重。所有批次拼接起来恰好等于
  `phase_haplotypes(...).optimal_haplotypes`。`batch_size` 可以逐批改变。
- **实例变化即令牌失效**：令牌绑定输入指纹（`loci` 与每条读段的 `alleles`、
  `flip_cost` 的规范化表示）。输入变化后旧令牌被
  `InvalidPhasingInputError` 拒绝，不会串到别的实例；伪造/损坏的令牌同样被
  拒绝。`max_nodes` 是**每次调用**的工作量选项，不属于实例指纹：预算耗尽后
  可以用同一个令牌去掉预算重试。
- **预算语义**：预算在本次调用内覆盖“证优 + 列举”两段；耗尽时抛
  `SearchBudgetExhaustedError`，不返回批次、不声称列举完毕，旧令牌仍可复用。
  `OptimalBatch.search.completed` 与 `exhausted` 同义，列举未结束时永不为
  `True`；最后一批可能是空批（搜索在此批内恰好走完），此时 `exhausted=True`。

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
- `phase_haplotypes` 一次性物化全部最优解；若位点重叠图高度碎裂，其数量可达
  `2^17`。需要控制内存时用 `phase_haplotypes_batched` 分批领取（续取令牌
  只占 `O(loci)`，不预先保存全部最优解）。

## 仓库结构

```
haplophase/__init__.py   # 公共导出
haplophase/core.py       # 校验、分支界限、连通分量、结果组装、分批续取枚举
tests/reference.py       # 独立全枚举参考（不调用被测核心求期望值）
tests/test_phase.py      # 必要测试（含分批领取不重复/不遗漏/令牌失效测试）
demo.py                  # 固定输入演示（正常结果 + 分批领取 + 拒绝/异常边界）
README.md
```
