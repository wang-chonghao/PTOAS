# A5 Tile Fusion 接入 VF CostModel 方案

## 1. 目标

PTOAS 当前在 A5 场景下已经支持 tileop 层的前端融合规划。现有策略是启发式
cost model，只根据 tileop 数据流连接、循环域、live tile 数和 VF 参数数量来
决定是否融合。

本方案的目标是：在不改变 PTOAS 现有 correctness 限制的前提下，把 A5 tileop
fusion 的收益判断替换/增强为 VF costmodel 评估。

核心目标：

- 在 tileop 层接入 costmodel，服务融合策略搜索。
- 保留现有 fusion legality hard gate。
- 复用 PTOAS 现有 tileop 语义输入结构，避免重新定义融合分析输入。
- 参考 TileLang 模板的 pattern 结构，但不在 costmodel 内调用模板，也不先展开
  单 tileop VF 再做融合消除；costmodel 直接从 tileop group、DFG 和少量
  pattern/opcode 映射构造融合后的 VfSim 输入。
- 逐步从 Python VfSimulator 过渡到 PTOAS 内部 C++ costmodel 实现。

非目标：

- 第一阶段不改写完整 PTOAS lowering pipeline。
- 第一阶段不在每个候选上调用完整 VPTO lowering。
- 第一阶段不实现新的全局 fusion search 策略，先替换局部收益判断。

## 2. PTOAS 当前编译链路

PTOAS CLI 入口在：

```text
tools/ptoas/ptoas.cpp
```

典型 A5 EmitC 路径：

```text
.pto input
  -> parse MLIR module
  -> frontend pipe id / pipe op lowering
  -> pto-infer-validate-pipe-init
  -> pto-lowering-sync-to-pipe
  -> pto-infer-layout
  -> pto-a5-normalize-tmov
  -> pto-validate-int-to-ptr-uses
  -> optional A5 frontend fusion path
  -> pto-view-to-memref
  -> pto-plan-memory
  -> pto-resolve-reserved-buffers
  -> optional sync insertion
  -> pto-materialize-tile-handles
  -> EmitC lowering
  -> C++ output
```

当前 A5 frontend fusion path 由 `--enable-op-fusion` 打开，只在以下条件下启用：

```text
--pto-arch=a5
--pto-level=level2 或 level3
```

对应代码：

```text
tools/ptoas/ptoas.cpp
  if (enableA5FrontendFusionPath) {
    pto-fusion-plan
    pto-op-scheduling
    pto-mark-last-use
  }
```

### 2.1 当前 Fusion Pass 链路

当前 fusion pass 链路：

```text
PTO tile IR
  -> FusionPlanPass
  -> OpSchedulingPass
  -> PTOMarkLastUsePass
```

相关文件：

```text
include/PTO/Transforms/TileFusion/FusionAnalysis.h
include/PTO/Transforms/TileFusion/FusionOpSemantics.h
lib/PTO/Transforms/TileFusion/FusionAnalysis.cpp
lib/PTO/Transforms/TileFusion/FusionOpSemantics.cpp
lib/PTO/Transforms/TileFusion/PTOFusionPlan.cpp
lib/PTO/Transforms/TileFusion/PTOOpScheduling.cpp
lib/PTO/Transforms/TileFusion/PTOMarkLastUse.cpp
```

`FusionPlanPass` 负责分析 block-local tileop DAG，并给接受融合的 tileop 标注：

```text
pto.fusion.group_id
pto.fusion.order
```

`OpSchedulingPass` 根据这两个属性把同一 fusion group 内的 tileop 尽量排成一个
contiguous span。

`PTOMarkLastUsePass` 根据 scheduled span 标注：

```text
pto.last_use
```

当前实现没有生成 `pto.fusion_region`，也没有在这一层直接生成 fused VF IR。

## 3. 当前 Fusion 输入结构

当前 fusion 决策的核心输入来自 `PreFusionAnalysis`：

```cpp
struct FusionBlockAnalysis {
  Block *block;
  SmallVector<FusionComputeNode, 8> computeNodes;
  SmallVector<IterationDomainClass, 4> iterationDomainClasses;
  SmallVector<FusionDFGEdge, 8> edges;
  SmallVector<FusionValueLiveness, 8> liveness;
  SmallVector<FusionWriteInstanceLiveness, 8> writeInstances;
};
```

单个 tileop 节点：

```cpp
struct FusionComputeNode {
  unsigned id;
  unsigned blockOrder;
  Operation *op;
  FusionOpSemantics semantics;
  unsigned iterationDomainClass;
  SmallVector<unsigned, 4> incomingEdges;
  SmallVector<unsigned, 4> outgoingEdges;
};
```

tileop 语义：

```cpp
struct FusionOpSemantics {
  FusionOpKind kind;
  FusionComputeFamily computeFamily;
  Operation *op;
  std::string opName;
  SmallVector<Value, 4> tileInputs;
  SmallVector<Value, 2> tileOutputs;
  SmallVector<Value, 2> scalarInputs;
};
```

当前已识别的 compute family：

```text
Elementwise:
  tadd, tsub, tmul, tdiv, tmax, tmin
  tadds, tsubs, tmuls, tdivs, tmaxs, tmins
  texp

ScalarExpand:
  texpands

RowBroadcastBinary:
  trowexpandmul, trowexpanddiv

ReduceRow / ReduceCol:
  已在语义分析中识别，但当前 FusionPlan 的 plannable op 还没有全部开放。
```

## 4. 当前 CostModel

当前 cost model 在：

```text
lib/PTO/Transforms/TileFusion/PTOFusionPlan.cpp
```

核心接口：

```cpp
class CostModel {
public:
  virtual PlanningDecision evaluateSeed(
      const PlanningContext &ctx,
      const FusionComputeNode &candidate) const = 0;

  virtual PlanningDecision evaluateAppend(
      const PlanningContext &ctx,
      ArrayRef<const FusionComputeNode *> currentGroup,
      const FusionComputeNode &candidate) const = 0;
};
```

当前实际使用：

```cpp
ConservativeDAGGreedyCostModel costModel;
ConservativeDAGGreedyStrategyEngine strategyEngine;
```

现有 `ConservativeDAGGreedyCostModel::evaluateAppend()` 的 hard gate：

- candidate 必须是支持的 plannable compute op。
- iteration domain 必须 proven。
- candidate 必须与 group 的 iterationDomainClass 相同。
- candidate 不能跨 hard boundary。
- candidate 与 group 至少有一条 tile 数据流连接。

现有启发式 cost：

```text
dependencyBenefit = 4 * connectionCount
loopMergeBenefit = 4
liveTilePenalty = max(0, liveTileCount - 10)
vfParameterPenalty = max(0, vfParameterCount - 12)

accept = dependencyBenefit + loopMergeBenefit
         - liveTilePenalty - vfParameterPenalty > 0
```

这个公式只表达融合倾向，不是真实硬件时间模型。

## 5. VF CostModel 接入位置

接入位置保持在 tileop 层：

```text
ConservativeDAGGreedyCostModel::evaluateAppend(...)
```

推荐结构：

```text
evaluateAppend(ctx, currentGroup, candidate)
  -> 现有 legality hard gate
  -> 构造 unfused candidate cost
  -> 构造 fused candidate cost
  -> 调用 VF costmodel
  -> 早期根据 legality / supported pattern 返回 PlanningDecision
  -> 后期叠加 UB 容量约束
```

也就是说，VfSimulator 或后续 C++ VF costmodel 早期主要提供预测与校验信息。
`group_id` 决策先按“合法且支持则尽量融合”推进；到 UB 容量阶段，再用
`estimatedUbPeak <= ubCapacity` 约束“应融尽融”的边界。VfSimulator 的 latency
结果后续主要用于 unroll / loop form 等局部优化选择，不替换 correctness 约束。

保留的 hard gate：

- A5 + level2/level3 才启用。
- 只支持当前 plannable op 列表。
- dynamic shape 先拒绝。
- iteration domain 必须 proven。
- candidate 与 group 必须同 iterationDomainClass。
- candidate 与 group 之间必须有数据流连接。
- 不跨 hard boundary。
- scheduling / last_use 仍由现有 pass 处理。

## 6. 输入输出形式

### 6.1 CostModel 输入

直接沿用当前 PTOAS fusion 输入：

```cpp
struct VfCostInput {
  const FusionBlockAnalysis *blockAnalysis;
  ArrayRef<const FusionComputeNode *> currentGroup;
  const FusionComputeNode *candidate;
};
```

内部派生信息：

```text
group op list:
  opName
  computeFamily
  tileInputs
  tileOutputs
  scalarInputs
  iterationDomainInfo(vRow, vCol)
  DFG edges
  liveness / write instance escape class
```

### 6.2 直接构造 Fused VfSim 输入

costmodel 不调用 `ExpandTileOpPass`，也不调用 `lib/TileOps/*.py`。这些模板只作为
pattern 语义参考，例如：

```text
Binary pattern:
  VLDS src0
  VLDS src1
  vec::op
  VSTS dst

Unary pattern:
  VLDS src
  vec::op
  VSTS dst
```

PTOAS 当前 TileLang DSL 已经有类似机制：

```text
tilelang-dsl/examples/v1_template_slot_multiop_demo.py
  templates={
    "core": {
      "tadd": "vadd",
      "tsub": "vsub",
      "tmul": "vmul",
      "tdiv": "vdiv",
    }
  }
  out = pto.tpl("core", lhs, rhs, mask)
```

但 `lib/TileOps` 主实现当前仍主要是每个 tileop 一个 Python 模板，例如：

```text
lib/TileOps/tadd_template.py
lib/TileOps/tsub_template.py
lib/TileOps/tmul_template.py
```

第一版 costmodel 只维护轻量 pattern/opcode 映射：

```cpp
enum class TilePatternKind {
  BinaryElementwise,
  UnaryElementwise,
  ScaleElementwise,
  Cast,
  RowExpand,
};

enum class VfOpcode {
  VADD,
  VSUB,
  VMUL,
  VEXP,
};

struct TileOpPatternSpec {
  StringRef tileOpName;
  TilePatternKind pattern;
  VfOpcode vectorOpcode;
  unsigned tileInputCount;
  unsigned scalarInputCount;
  unsigned tileOutputCount;
  bool allowFlatten1D;
  bool allowLoopFusion;
};
```

首批支持：

```text
tadd -> BinaryElementwise + VADD
tsub -> BinaryElementwise + VSUB
tmul -> BinaryElementwise + VMUL
texp -> UnaryElementwise  + VEXP
```

这个映射不描述完整 lowering，不生成单 tileop 的完整 VF。它只回答：

```text
tileop 属于哪种 pattern
tileop 中间的 vector op 是哪条 VF 微指令
需要几个 tile/scalar 输入和几个 tile 输出
是否允许 flatten / loop fusion
```

真正的 VfSim 输入由 group builder 一次性构造：

```text
VfCostInput
  -> proposedGroup = currentGroup + candidate
  -> 查每个 tileop 的 TileOpPatternSpec
  -> 根据 DFG/liveness 识别 externalInputs / internalValues / finalOutputs
  -> 直接生成 fused VfSimProgram
```

### 6.3 VF CostModel 输出

建议输出：

```cpp
struct VfCostResult {
  bool supported;
  bool rejectedByPressure;
  int64_t fusedCycles;
  int64_t unfusedCycles;
  int64_t latencyGain;
  int64_t peakVRegEstimate;
  int64_t dependencyDepthEstimate;
};
```

`PlanningDecision` 映射需要分阶段处理。早期阶段保留 `fusedCycles` /
`unfusedCycles` 作为预测、校验和调试信息，但不把
`fusedCycles < unfusedCycles` 作为融合必要条件。早期融合策略是：

```text
accept = legality pass
         && supported pattern
         && same iteration domain
         && has dataflow connection
         && no hard boundary
```

引入 UB 容量约束后，融合策略变为：

```text
accept = legality pass
         && supported pattern
         && same iteration domain
         && has dataflow connection
         && no hard boundary
         && estimatedUbPeak <= ubCapacity
```

`latencyGain` 后续主要用于 unroll、loop form 等局部寻优，而不是第一版
`group_id` 是否融合的主判据。第一阶段可以保留现有 `PlanningCost` 字段，并新增
debug-only 字段或日志输出。

## 7. Tileop Group 到 Fused VfSimProgram 的直接构造规则

第一阶段不走“单 tileop 模板展开 -> 再做 VF fusion / loop fusion /
vlds-vsts 消除”的流程，而是从 tileop group 和依赖关系直接构造融合后的程序。

### 7.1 Group 分析

输入：

```cpp
struct VfCostInput {
  const FusionBlockAnalysis *blockAnalysis;
  ArrayRef<const FusionComputeNode *> currentGroup;
  const FusionComputeNode *candidate;
};
```

内部构造：

```text
proposedGroup = currentGroup + candidate
```

对 `proposedGroup` 做以下分析：

```text
producedValues:
  group 内所有 tileOutputs

externalInputs:
  group tileInputs - producedValues

internalValues:
  被 group 内后继消费的 producedValues

finalOutputs:
  没有 group 内后继消费的 tileOutputs
  或者虽然被 group 内消费，但仍有 group 外 user / escape 的 tileOutputs
```

### 7.2 Fused Loop 构造

第一版只支持简单 elementwise group：

```text
tadd / tsub / tmul / texp
```

要求：

```text
所有 tileop pattern 支持 flatten
iterationDomainClass 一致
iteration domain proven
dtype / shape 静态可证明
无 hard boundary
group 内有直接数据流连接
```

简单 elementwise group 直接生成 flattened loop：

```text
tripCount = ceil(vRow * vCol / lanes(dtype))

for iter in tripCount:
  fused body
```

其中 `lanes(dtype)` 需要与 TileLang 模板中的 `pto.get_lanes(dtype)` 保持同一语义。
当前 PTOAS 的 tileop 模板已经通过 `pto.get_lanes(dtype)` 表达 256B vector register
宽度，例如 fp32/i32 为 64 lanes、fp16/bf16 为 128 lanes、i64 为 32 lanes。C++ 侧
目前还没有统一的公共 lane helper，已有实现中存在局部 256B 常量。阶段 1 可暂时在
VF costmodel 中保留等价的 256B 计算，但后续需要抽到公共 C++ helper，例如
`PTOTypeUtils` 中的 `getPTOVectorLaneCount(Type)`，并逐步让 VF costmodel、
sync/memory 分析等 C++ 代码复用同一入口。

后续扩展时可以增加：

```text
Keep2D loop
Row-wise loop
Reduction loop
```

但第一阶段不处理复杂 loop pattern。

### 7.3 Pattern Emit

Binary pattern：

```text
lhs = getInputReg(src0)
rhs = getInputReg(src1)
out = newReg(dst)
emit vectorOpcode(out, lhs, rhs)
bind dst -> out
```

Unary pattern：

```text
src = getInputReg(src)
out = newReg(dst)
emit vectorOpcode(out, src)
bind dst -> out
```

`getInputReg()` 的规则：

```text
如果 input 是 group 内 producer 的 output:
  直接使用 producer 的 virtual reg，不生成 VLDS

否则:
  为外部输入生成 VLDS
```

output store 规则：

```text
如果 output 是 internalValues 且无 group 外 user:
  不生成 VSTS

如果 output 是 finalOutputs:
  生成 VSTS
```

因此中间 `VSTS tmp` / `VLDS tmp` 不需要先生成再消除，而是在 fused builder 中天然不生成。

### 7.4 示例

输入：

```text
tmp0 = TADD(src0, src1)
tmp1 = TMUL(tmp0, src2)
tmp2 = TEXP(tmp1)
tmp3 = TSUB(tmp2, src3)
```

分析：

```text
externalInputs:
  src0, src1, src2, src3

internalValues:
  tmp0, tmp1, tmp2

finalOutputs:
  tmp3
```

直接生成 fused VfSimProgram：

```text
loop iter:
  r0 = VLDS src0
  r1 = VLDS src1
  r2 = VADD r0, r1        // tmp0

  r3 = VLDS src2
  r4 = VMUL r2, r3        // tmp1

  r5 = VEXP r4            // tmp2

  r6 = VLDS src3
  r7 = VSUB r5, r6        // tmp3

  VSTS tmp3, r7
```

不会生成：

```text
VSTS tmp0 / VLDS tmp0
VSTS tmp1 / VLDS tmp1
VSTS tmp2 / VLDS tmp2
```

### 7.5 Unfused Baseline

为了比较 fused/unfused latency，同一套 builder 也需要生成 unfused baseline：

```text
buildUnfusedPrograms(group):
  每个 tileop 单独生成一个 VF program

buildFusedProgram(group):
  整个 group 一次性生成一个 VF program
```

unfused 的每个单 tileop program 可以用同样的 pattern 规则直接生成：

```text
外部输入 -> VLDS
vector op
输出 -> VSTS
```

### 7.6 压力与依赖保护

第一阶段保守估计：

- `peakVRegEstimate`：用 fused body 中同时 live 的 virtual vreg 数估算。
- `dependencyDepthEstimate`：用 group 内 tile DFG 最长路径估算。
- 如果超过阈值，拒绝融合或回退现有启发式。

## 8. 分阶段计划

整体分为五个阶段。前期先打通接口和预测能力，融合策略采用“合法且支持就尽量
融合”；后期再引入 UB 容量，在 UB/cache 允许范围内应融尽融；最后再做 unroll
和 loop 结构优化。

### 阶段 1：PTOAS 接口与外挂 VfSim 验证

目标：修改 PTOAS costmodel 接口，实现从 tileop group 直接构造融合后的
vector/VfSim 输入。VfSimulator 暂时作为外挂验证工具，不要求 PTOAS 端到端跑通。

已完成：

```text
1. 梳理并固化当前链路：
   .pto
   -> PreFusionAnalysis
   -> FusionPlanPass
   -> CostModel.evaluateSeed/evaluateAppend
   -> group_id/order metadata

2. 定义 costmodel 输入输出：
   FusionBlockAnalysis
   currentGroup
   candidate
   proposedGroup = currentGroup + candidate
   fusedCycles
   unfusedCycles
   supported
   rejectReason

3. 抽离/重构 PTOAS 当前 CostModel 接口：
   FusionCostModel.h/.cpp
   保留当前 ConservativeDAGGreedyCostModel 默认行为不变

4. 定义第一版 pattern/opcode 映射：
   tadd -> BinaryElementwise + VADD
   tsub -> BinaryElementwise + VSUB
   tmul -> BinaryElementwise + VMUL
   tdiv -> BinaryElementwise + VDIV
   texp -> UnaryElementwise  + VEXP
   可选 scale:
   tadds / tsubs / tmuls / tdivs

   说明：tdiv/tdivs 在真实 lowering 中可能走 soft div 或高精度 helper，未必等价于
   单条 vector 指令。阶段 1 暂时按简化模型接入，tdivs 视作单条 VDIVS；但 latency
   预测不得静默复用其他 opcode 的 timing。如果 VfSimulator 当前 ISA 配置没有对应
   微指令，例如 VDIVS/VSUBS，外挂验证脚本应直接报错，提示该 tileop 对应微指令在
   VfSimulator 中暂不支持。

5. 实现 group 分析：
   externalInputs
   internalValues
   finalOutputs
   group 内 producer-consumer 关系
   group 外 user / escape 判断

6. 实现 fused builder：
   外部输入 emit VLDS
   中间值使用 virtual reg
   最终输出 emit VSTS

7. 临时按 A5 256B vector register 计算 lanes(dtype)，与 Python DSL 的
   pto.get_lanes(dtype) 语义保持一致；后续抽出 C++ 公共 helper，避免多个
   C++ pass 各自维护 256B 常量。

8. 导出 fused VFProgram：
   --dump-vf-program 文本
   --dump-vf-program-json=<path> 结构化 JSON

9. 外挂验证：
   scripts/vfprogram_latency.py
   读取 VFProgram JSON / 文本 dump
   转换为当前 Python VfSimulator payload
   调用 CoreVfCostModel.run_payload()
   输出 vf_cycles
```

验收：

```text
TADD -> TMUL、GeLU_poly 风格链路、TEXP -> TDIV 能导出 fused VFProgram。
fused 版本中间 tmp 不生成 VSTS/VLDS。
外挂 VfSimulator 能对局部 case 给出 cycles。
unsupported micro-op 直接报错，不静默替换 opcode。
PTOAS 默认行为可暂时保持不变，不要求端到端跑通。
```

阶段 1 未做且后续按需补充：

```text
1. unfused baseline builder：
   每个 tileop 单独生成一个 VF program。
   由于当前阶段策略是 legal + supportedPattern 就尽量融合，unfused baseline
   不作为阶段 1 阻塞项。

2. latency 参与 group_id 决策：
   当前 fusion decision 仍由现有 legality / conservative greedy 策略决定。
   latency 只作为验证和后续优化信息。
```

阶段 1 外挂验证脚本：

```text
scripts/vfprogram_latency.py
```

当前推荐用法是先让 PTOAS 输出 `--dump-vf-program-json=<path>` 结构化 VFProgram，
再由脚本转换成 VfSimulator 现有 JSON trace payload，最后通过
`CoreVfCostModel.run_payload()` 得到 `vf_cycles`。脚本仍兼容旧的
`--dump-vf-program` 文本日志，用于调试和过渡。后续源码级接入时应替换为进程内对象
接口。

示例：

```bash
ptoas case.pto --pto-arch=a5 --pto-level=level2 --enable-op-fusion \
  --dump-vf-program-json=/tmp/case.vfprogram.json \
  --emit-pto-ir -o /tmp/case_out.pto

python3 scripts/vfprogram_latency.py /tmp/case.vfprogram.json \
  --vfsim-root /mnt/e/vfsimulator_structure \
  --dtype fp32 \
  --out-dir /tmp/ptoas_vfsim_case \
  --payload-out /tmp/case_vfsim_payload.json
```

### 阶段 2：源码级融合与 C++ VfSimulator 接入

目标：基于阶段 1 的 fused VFProgram 构造能力，定义 PTOAS 到 C++ 化 VfSimulator
的源码级接口，并逐步将 VfSimulator 的必要子集接入 PTOAS。源码级接入不应通过
JSON 文件传递数据；JSON 仅保留为 debug / 对比格式。

#### 阶段 2.1：定义源码级接口 IR

PTOAS 与 C++ VfSimulator 之间使用内存中的递归结构化 IR，命名为 `VfSimProgram`。
阶段 2 开始不再维护一套独立的 `VfProgram -> VfSimProgram` lowering 链路，而是让
TileFusion builder 直接生成 `VfSimProgram`。builder 内部仍可使用 MLIR `Value`
做 tile/scalar/register 映射和 liveness 判断，但对外产物不能携带 MLIR `Value`。

建议 C++ 结构：

```cpp
struct VfSimProgram {
  SmallVector<VfSimNode, 8> body;
};

using VfSimNode = std::variant<VfSimInst, VfSimLoop>;

struct VfSimLoop {
  int64_t tripCount;
  unsigned unroll;
  SmallVector<VfSimNode, 8> body;
};

struct VfSimInst {
  VfSimOpcode opcode;
  SmallVector<VfSimOperand, 4> dst;
  SmallVector<VfSimOperand, 4> src;
};

struct VfSimOperand {
  VfSimOperandKind kind;  // VReg / UB / Scalar
  unsigned id;
  VfSimDType dtype;      // fp32 / fp16 / bf16 / i32 / ...
};
```

接口约束：

```text
1. dtype 放在 operand 上，不放在 VfSimProgram 上。
   原因是 tcvt/cast/compare/mixed precision op 可能出现 src/dst dtype 不同。

2. VfSimProgram.body 支持 VfSimInst 和 VfSimLoop。
   顶层允许非 loop 指令。

3. VfSimLoop.body 同样支持 VfSimInst 和 VfSimLoop。
   这样可以表达嵌套 loop 和复杂 loop pattern。

4. opcode 必须是微指令 opcode，不是 tileop opcode。
   例如 tadd -> VADD，texp -> VEXP，tmuls -> VMULS。

5. unsupported micro-op 不能 silent fallback。
   C++ 接口应返回 supported=false 和明确 rejectReason。
```

#### 阶段 2.2：PTOAS adapter

工作：

```text
1. 用 VfSimProgram 替换当前阶段 1 的 VfProgram：
   buildFusedElementwiseVfProgram()
   -> buildFusedElementwiseVfSimProgram()

2. builder 直接生成递归 VfSimProgram：
   top-level body 可包含 inst/loop
   当前 elementwise group 生成一个 flattened VfSimLoop
   loop body 内包含 VLDS / compute / VSTS

3. builder 内部状态仍可使用 MLIR Value：
   Value -> UB operand
   Value -> scalar operand
   Value -> current virtual register
   这些 map 不进入 VfSimProgram 对外结构。

4. dtype 推导：
   tile operand 从 TileBufType element type 推导
   scalar operand 从 scalar MLIR type 推导
   virtual reg 在创建时使用绑定 tile output/input Value 的 element type
   当前无法推导时返回 failure，不能默认 fp32

5. JSON dump 从 VfSimProgram 导出：
   operand JSON 包含 kind / id / dtype
   body JSON 支持 inst / loop 递归结构
   JSON 只作为 debug / Python VfSimulator 对比路径，源码级接入不以 JSON 文件为接口。
```

#### 阶段 2.3：VfLatencyModel 接口骨架

```cpp
struct VfLatencyResult {
  bool supported;
  int64_t cycles;
  std::string rejectReason;
};

class VfLatencyModel {
public:
  VfLatencyResult predict(const VfSimProgram &program) const;
};
```

第一版 `predict()` 已经改成 native bridge：

```text
1. 递归遍历 VfSimProgram，转换为 native ProgramNode。
2. 推断 dtype，混合 dtype 直接 reject。
3. 用 native ParamDB + ProgramFlatten + IFU + IDU + OOO 主线计算 latency。
4. 成功时返回 supported=true 和 vf_end_cycle。
5. 预测路径默认不落盘，debug 时可再接日志目录。
```

当前状态：`include/PTO/VFcostmodel/VfLatencyModel.h` 已新增抽象接口定义，
当前 `VfLatencyModel` 实现已切到 native VfSimulator 主线桥接。
TileFusion / VPTO 两侧继续复用这一层入口。

#### 阶段 2.4：C++ 化 VfSimulator 子集

```text
1. C++ 化 VfSimulator 子集：
   VLDS / VSTS
   VADD / VSUB / VMUL / VDIV / VEXP
   scale op 所需 vector/scalar 指令
   简单 flattened loop
   基础 dependency / latency / II / forwarding 子集

2. 在 PTOAS costmodel 内部接入 C++ VfLatencyModel：
   proposedGroup
   -> fused VfSimProgram
   -> predict cycles
   -> VfLatencyResult

3. 早期 group 决策仍采用合法性/支持范围驱动：
   legal + supportedPattern 就尽量融合
   cycles 作为评估和校验信息，不作为 group_id 主判据

4. 输出：
   pto.fusion.group_id
   pto.fusion.order
```

验收：

```text
binary / unary / scale 简单 case 能端到端跑通。
PTOAS 内部可以构造 VfSimProgram 并调用 VfLatencyModel。
JSON / 外挂 VfSimulator 结果可作为对照。
unsupported micro-op 在源码接口中返回明确 rejectReason。
现有 hard gate 仍有效。
lit 测试覆盖简单链、unsupported、domain mismatch、hard boundary。
```

当前阶段 2.1 / 2.2 的落地状态：

```text
1. PTOAS 已不再额外维护阶段 1 的 VfProgram 结构。
   TileFusion builder 直接生成 VfSimProgram。

2. VfSimProgram 使用递归 body：
   VfSimProgram.body: inst 或 loop
   VfSimLoop.body: inst 或 loop
   当前 elementwise group 先生成一个 flattened loop，后续复杂 tileop 可以扩展为嵌套 loop。

3. operand 已携带 dtype：
   UB tile operand 从 TileBufType element type 推导
   scalar operand 从 scalar MLIR type 推导
   virtual register operand 从绑定的 tile input/output element type 推导
   无法推导时 buildFusedElementwiseVfSimProgram() 返回 failure

4. JSON dump 仍保留为 debug / 外挂 Python VfSimulator 对比路径。

5. 现阶段已验证：
   tadd + tmul fused case
   GeLU_poly 简化 case
   texp + tdiv supported case
   tdivs -> VDIVS unsupported reject case
```

### 阶段 3：扩展 TileOp Pattern 覆盖

目标：扩大 tileop 到 vector op 的映射范围。这一阶段工作量最大，复杂度来自不同
tileop pattern 的 loop 结构、mask、dtype 和多指令实现差异。

工作：

```text
1. 扩展 BinaryElementwise：
   tmax / tmin / tdiv

2. 扩展 ScaleElementwise：
   tadds / tsubs / tmuls / tmaxs / tmins

3. 扩展 UnaryElementwise：
   tabs / tneg / trelu / tsqrt / tlog
   具体开放顺序取决于 ISA 和 VfSim 支持情况。

4. 扩展复杂 pattern：
   cast / tcvt
   rowexpand / colexpand
   其他 broadcast 类 op

5. 每个新增 op 都定义：
   pattern kind
   vector opcode 或 vector op sequence
   loop form
   是否允许 flatten
   是否允许 loop fusion
   特殊 mask / scalar / dtype 约束
```

验收：

```text
新增 op 能生成 fused/unfused VfSimProgram。
不支持的 op 明确 reject 或切 group，不能 silent fallback 成错误预测。
```

### 阶段 4：UB 容量约束与应融尽融

目标：从早期的“无限资源假设下合法就尽量融合”变成“UB/cache 容量允许范围内
应融尽融”。

工作：

```text
1. 定义 UB 容量来源：
   arch 固定参数
   命令行 / 配置项
   或后续与 memory planning 信息对接

2. 实现 fused group UB peak 估算：
   external input tile
   final output tile
   group 外仍使用的中间 tile
   group 内唯一消费的中间 tile 不生成 VSTS/VLDS，可不计或少计

3. 修改 evaluateAppend：
   legal + supportedPattern + estimatedUbPeak <= ubCapacity 时继续融合
   超过 UB 容量时停止或切 group

4. 保持 produce-consumer 优先：
   UB 足够时尽量长链融合
   UB 不足时以容量约束切分
```

验收：

```text
UB 足够时长链尽量同 group。
UB 超限时能稳定切 group。
切分结果可解释，并能输出 UB peak 估算信息。
```

### 阶段 5：Unroll 寻优与 Loop 优化 Metadata

目标：在 group 已确定后，对 loop fusion、loop form 和 unroll 做优化建议，输出
后端可消费的 loop 级 metadata。

工作：

```text
1. 定义 metadata contract：
   pto.fusion.loop_id
   pto.fusion.loop_order
   pto.fusion.loop_unroll

2. 判断 loop fusion legality：
   same iteration domain
   pattern loop 兼容
   无 reduction-carried dependency
   mask/step 兼容

3. 对可 loop fusion 的 group 打 loop_id/order。

4. 支持 unroll 候选：
   1 / 2 / 4

5. 对不同 unroll / loop form 构造 VfSimProgram 并预测 cycles。

6. 选择资源合法且 latency 较优的 loop_unroll / loop form。
```

验收：

```text
简单 elementwise group 能打 loop_id / loop_order。
能输出 loop_unroll。
不同 unroll 能得到不同 predicted cycles。
```

## 9. 预计结果

短期结果：

- PTOAS A5 fusion pass 可以从 tileop group 直接构造 fused/unfused VfSimProgram，
  并获得 VF-aware 预测结果。
- 输入结构与当前 `FusionBlockAnalysis/FusionComputeNode` 保持一致。
- 默认 correctness 限制不变，风险可控。
- 编译时不会因为每个候选都跑完整 lowering 而显著增加。

中期结果：

- 可以在 UB 容量允许范围内实现 VF 应融尽融。
- 能识别过大 fusion group 的寄存器压力和依赖链风险。
- 能更准确判断 `vsts/vlds` 消除、loop 合并带来的收益。
- 可以用 latency 结果进行 unroll / loop form 寻优。

长期结果：

- 形成 PTOAS 内部 C++ VF costmodel。
- Python VfSimulator 作为验证 oracle 或离线分析工具。
- 后续可以在同一输入结构上开发更复杂的 fusion search 策略。

## 10. 风险与开放问题

- 轻量 pattern/opcode 映射与真实 TileLang template 可能偏离，需要定期用 VPTO
  lowering 校验。
- 动态 shape 当前继续拒绝，后续如果要支持，需要引入 symbolic cost。
- `tdiv/tcvt/soft div/mod` 等复杂模板可能包含 helper call 和多条微指令，第一阶段应谨慎开放。
- vreg pressure 的轻量估计可能和真实 rename/OoO 行为有偏差，需要用 Python VfSimulator 校准。
- 完整 C++ 化 VfSimulator 难度中高，应分 op 子集逐步推进。

## 11. VfSimulator C++ 化后续计划

这一节只记录 VfSimulator 主线的 C++ 化计划，和上面的 PTOAS 接入设计分开。
当前约束是：

- 只接主线 `queue_level4`，暂不迁理论极限分支。
- 继续把现有 `configs/*.json` 作为单一事实来源。
- C++ 侧先做行为对齐，再做性能优化。
- `VfSimProgram` 是统一输入边界，PTOAS 与 VPTO 侧都只负责构造它。

### 11.1 当前已完成的准备

- 主线模型已经稳定在 `queue_level4`。
- 主线解释链路已经收敛为：
  `JSON/CCE input -> api -> flatten -> IFU -> IDU -> OOO -> VF end cycle`
- 当前关键魔法值已经逐步整理进 `configs/uarch.json`，包括：
  - `idu_to_ooo_delay`
  - `ooo_to_shq_delay`
  - `ooo_to_lsq_delay`
  - `vloop_to_dispatch_delay`
  - `initial_top_block_vloop_start_cycle`
  - `nested_vloop_initial_start_gap`
  - `loop1_min_feedback_gap`
  - `innermost_iter_dispatch_stride`
  - `consumer_release_start_offset`
- `isa.json`、`forwarding.json`、`InitiationInterval.json` 的结构已经明确，后续可以直接映射到 C++ 查询对象。

### 11.2 Python -> C++ 映射表

| Python 模块 | 作用 | C++ 目标 |
|---|---|---|
| `api/simulator_costmodel.py` | 入口编排 | `native/VfLatencyModel.*` |
| `core/param_db.py` | 参数加载与查询 | `native/ParamDB.*` |
| `core/isa_traits.py` | ISA class / load-store 判定 | `native/ISATraits.*` |
| `core/program_analysis.py` | loop / bounds / 结构分析 | `native/ProgramAnalysis.*` |
| `core/vreg_live_range_normalization.py` | vreg 规范化 | `native/VRegLiveRangeNormalization.*` |
| `core/flatten.py` | 程序展平 | `native/ProgramFlatten.*` |
| `core/ifu.py` | 动态指令展开 | `native/IFU.*` |
| `core/idu.py` | 发射 / credit gate / VLOOP 可见性 | `native/IDU.*` |
| `core/ooo.py` | OoO 核心 | `native/OOO.*` |
| `core/ooo_factory.py` | 模型/参数组装 | `native/OOOFactory.*` |
| `core/simulator_runner.py` | 主循环与日志 | `native/SimulatorRunner.*` |

### 11.3 参数 C++ 化方案

参数层不改 JSON 文件格式，继续把下面这些文件作为单一事实来源：

- `configs/isa.json`
- `configs/uarch.json`
- `configs/forwarding.json`
- `configs/InitiationInterval.json`

第一阶段的 C++ 化方式是：

1. C++ 在启动时读取 JSON 配置。
2. 把配置解析到内存对象中缓存。
3. 对外提供和当前 Python `ParamDB` 等价的查询接口。

这样做的好处是：

- 不改配置格式
- 不引入双份配置
- 不改变回归数据
- 后面如果启动成本变高，再考虑生成编译期快照

### 11.4 第一阶段 C++ 化范围

第一步只接主线，不碰理论极限分支：

1. `ParamDB`
2. `ISATraits`
3. `ProgramAnalysis`

这三者先完成后，再继续：

4. `flatten`
5. `IFU`
6. `IDU`
7. `OOO`
8. `SimulatorRunner`

### 11.5 设计原则

- C++ 版先做行为对齐，不先做性能优化。
- Python 版继续保留，作为行为 oracle。
- 理论极限分支先不迁移，先把主线稳定住。
- `VfSimProgram` 是新的输入边界，PTOAS 和 VPTO 两边都只负责构造它，不把仿真细节散到上层。
