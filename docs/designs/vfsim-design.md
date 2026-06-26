# VF CostModel External Fusion

本文记录 PTOAS 与 vfsimulator 解耦后的推荐接口方案。目标是让 PTOAS 只负责 tileop 合法融合分组，vfsimulator 负责融合策略生成、costmodel 评估和策略选择。

## PTO TileOp Fusion Pass

在确定接口协议前，需要先明确 PTOAS 中与 tileop fusion 相关的现有 pass，便于后续确定协议范围与职责边界。

当前前端 tileop fusion 链路可以概括为：

```text
PreFusionAnalysis
  -> FusionPlan
  -> OpScheduling
  -> MarkLastUse
  -> FusionRegionGen
  -> VPTO backend
```

| 阶段 | Pass | 主要职责 | 是否修改 IR |
| --- | --- | --- | --- |
| 1 | `pto-pre-fusion-analysis` | 分析 tileop、依赖、iteration domain、liveness 和边界信息 | 否 |
| 2 | `pto-fusion-plan` | 生成 fusion group，并给 tileop 打 fusion 标签 | 是 |
| 3 | `pto-op-scheduling` | 将同一 fusion group 调整为 block 内连续 span | 是 |
| 4 | `pto-mark-last-use` | 标记 fusion span 内 tile 输入的 last-use 信息 | 是 |
| 5 | `pto-fusion-region-gen` | 将连续 fusion span 包装成 `pto.fusion_region` | 是 |

## PreFusionAnalysis

`PreFusionAnalysis` 对应 pass：

```text
pto-pre-fusion-analysis
```

它是 tileop fusion 最前面的分析层，只提供合法性基础，不生成 fusion group，也不打 `pto.fusion.group_id` / `pto.fusion.order` 标签。

| 项目 | 内容 |
| --- | --- |
| 输入 | tile-native PTO IR |
| 输出 | 按 block 组织的 `FusionBlockAnalysis` |
| 核心作用 | 告诉后续 pass：有哪些 tileop 可以分析、它们之间有什么依赖、哪些地方不能跨 |
| 代码位置 | `include/PTO/Transforms/TileFusion/FusionAnalysis.h` |
|  | `lib/PTO/Transforms/TileFusion/FusionAnalysis.cpp` |
|  | `lib/PTO/Transforms/TileFusion/PTOPreFusionAnalysis.cpp` |
|  | `lib/PTO/Transforms/TileFusion/FusionOpSemantics.cpp` |

`FusionBlockAnalysis` 的主要内容：

| 字段 | 含义 |
| --- | --- |
| compute nodes | block 内可参与 fusion 分析的 tile compute op |
| dataflow edges | tileop 之间的 SSA 数据依赖 |
| iteration domain classes | 判断 tileop 是否在同一个循环空间 |
| liveness | 判断 value 的生命周期和是否被外部使用 |
| write instance escape class | 判断某一次 tile buffer 写入能不能作为 fusion 内部临时结果 |
| local/hard boundary info | 判断 fusion 能不能跨过某些 op |

## FusionPlan

`FusionPlan` 对应 pass：

```text
pto-fusion-plan
```

它消费 `PreFusionAnalysis` 的结果，并对每个 block 的 `FusionBlockAnalysis` 做规划。当前主要做两件事：

1. 决定 fusion group；
2. 给接受融合的 tileop 打 `pto.fusion.group_id` / `pto.fusion.order` 标签。

代码位置：

| 文件 | 作用 |
| --- | --- |
| `include/PTO/Transforms/TileFusion/FusionCostModel.h` | 定义 planner/cost model 接口 |
| `lib/PTO/Transforms/TileFusion/FusionCostModel.cpp` | 实现 seed/candidate 合法性与启发式收益判断 |
| `lib/PTO/Transforms/TileFusion/PTOFusionPlan.cpp` | 组装 fusion group，并将结果写回 IR metadata |

### Fusion Group

Fusion group 在当前代码中对应 `PlannedFusionGroup`，本质上是一个 `FusionComputeNode` 数组：

```cpp
struct PlannedFusionGroup {
  SmallVector<const pto::FusionComputeNode *, 8> members;
};
```

因此，fusion group 可以理解为：

```text
同一个 block 内，一组被 planner 接受融合的 tile compute op。
```

当前实际使用的 group 生成逻辑是：

```text
ConservativeDAGGreedyStrategyEngine
```

基本流程：

```text
遍历 computeNodes
  -> 选择未分配 node 作为 seed
  -> costModel.evaluateSeed(ctx, seed)
  -> 创建临时 group
  -> 反复扫描 candidate
  -> costModel.evaluateAppend(ctx, groupMembers, candidate)
  -> accept 后加入 group
  -> 直到没有新 candidate
  -> group size >= 2 时保留
```

### 合法性和收益判断

当前 `ConservativeDAGGreedyCostModel` 同时包含合法性判断和启发式收益判断。

| 对象 | 条件 |
| --- | --- |
| seed | 是当前支持规划的 tileop |
| seed | iteration domain 必须 proven |
| candidate | candidate 自己也能作为合法 seed |
| candidate | candidate 和 group 第一个成员在同一个 iteration domain |
| candidate | candidate 和 group 之间不能有 hard boundary |
| candidate | candidate 和 group 至少有一个直接数据流连接 |
| candidate | 启发式分数大于 0 |

启发式分数：

```text
total = dependencyBenefit + loopMergeBenefit
        - liveTilePenalty
        - vfParameterPenalty
```

两个核心类的职责区别：

| 类 | 职责 |
| --- | --- |
| `ConservativeDAGGreedyStrategyEngine` | 搜索和组装 group，决定以哪个 node 为 seed、按什么顺序扫描 candidate、何时加入临时 group |
| `ConservativeDAGGreedyCostModel` | 判断 seed/candidate 是否 accept，包括合法性判断和启发式收益分数计算 |

因此，当前 `FusionPlan` 层并不只是判断“能否融合”，也包含“融合是否有收益”的保守判断。后续如果要与 vfsimulator 解耦，可以考虑将这两部分拆开：PTOAS 保留合法 group 生成，收益判断和策略搜索迁移到 vfsimulator。

### 打标签

打标签在 `PTOFusionPlan.cpp` 中完成，供后续 pass 和 VPTO 后端识别 fusion group。

标签定义位置：

```cpp
static constexpr llvm::StringLiteral kFusionGroupIdAttr =
    "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr =
    "pto.fusion.order";
```

标签含义：

| 标签 | 含义 |
| --- | --- |
| `pto.fusion.group_id` | tileop 所属的 fusion group id |
| `pto.fusion.order` | tileop 在 group 内的顺序，从 0 开始 |

示例：

```mlir
pto.tadd ... {pto.fusion.group_id = 0 : i64, pto.fusion.order = 0 : i64}
pto.tadd ... {pto.fusion.group_id = 0 : i64, pto.fusion.order = 1 : i64}
pto.tadd ... {pto.fusion.group_id = 0 : i64, pto.fusion.order = 2 : i64}
```

## 标签后处理

在标签被 VPTO 后端正式消费前，还需要经过三个后处理 pass：

```text
pto-op-scheduling
  -> pto-mark-last-use
  -> pto-fusion-region-gen
```

| Pass | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| `pto-op-scheduling` | `pto.fusion.group_id` / `pto.fusion.order` | 重排后的 IR | 将同一个 fusion group 的 op 调整成 block 内连续 span |
| `pto-mark-last-use` | 连续 fusion span | `pto.last_use` | 给 fusion span 内每个 op 的 tile 输入标记 last-use 信息 |
| `pto-fusion-region-gen` | 连续 fusion span | `pto.fusion_region` | 将已经排成连续 span 的 fusion group 包装成 `pto.fusion_region` |

这三步的关系可以理解为：

```text
FusionPlan metadata
  -> compact into contiguous span
  -> annotate last-use
  -> materialize pto.fusion_region
```

## VF CostModel 接入方案

外接 VF costmodel 的接入点建议放在 `FusionPlan` 层。原因是这一层已经拥有 `PreFusionAnalysis` 产出的 tileop graph、iteration domain、boundary 和 liveness 信息，并且当前本来就负责生成 fusion group 和打 fusion metadata。

### 接入目标

当前 `FusionPlan` 默认路径是：

```text
PreFusionAnalysis
  -> ConservativeDAGGreedyStrategyEngine
  -> ConservativeDAGGreedyCostModel
  -> fusion group
  -> pto.fusion.group_id / pto.fusion.order
```

其中 `ConservativeDAGGreedyCostModel` 同时包含：

| 部分 | 当前职责 | 接入后的处理 |
| --- | --- | --- |
| 合法性判断 | 判断 tileop 是否可放入同一个 group | 保留在 PTOAS |
| 启发式收益判断 | 用 benefit/penalty 判断融合是否有收益 | 默认模式保留；costmodel 模式由 vfsimulator 接管 |

因此，接入目标不是删除现有启发式策略，而是增加一个可选路径：

```text
默认模式：
  继续使用 ConservativeDAGGreedyCostModel

costmodel 模式：
  PTOAS 只生成合法可融合 group
  vfsimulator 负责策略生成、收益评估和最优策略选择
```

### 双路径设计

建议 `FusionPlan` 支持两种模式：

```text
default mode
  -> ConservativeDAGGreedyStrategyEngine
  -> ConservativeDAGGreedyCostModel
  -> 现有启发式收益判断

costmodel mode
  -> ConservativeDAGGreedyStrategyEngine
  -> LegalityOnlyFusionCostModel
  -> 合法可融合 group
  -> external vfsimulator planner
  -> selected strategy / unroll
```

也就是说，现有 `ConservativeDAGGreedyCostModel` 保留为默认选项。只有在编译器显式使能 VF costmodel 优化时，才切换到 costmodel mode。

### Legality-only Group

costmodel mode 下，PTOAS 仍然需要保留合法性判断。也就是说，不是无条件融合所有 tileop，而是在 PTOAS 能证明合法的范围内尽可能形成 group。

legality-only 判断应保留当前 `ConservativeDAGGreedyCostModel` 中与合法性相关的条件：

| 条件 | 说明 |
| --- | --- |
| supported tileop | tileop 必须是当前 fusion planner 支持分析的 op |
| proven iteration domain | iteration domain 必须可证明 |
| same domain class | candidate 和 group 在同一个 iteration domain |
| no hard boundary | candidate 和 group 之间不能跨 hard boundary |
| dataflow connection | candidate 和 group 至少有一个直接数据流连接 |

costmodel mode 下应移除的是当前启发式收益判断：

```text
total = dependencyBenefit + loopMergeBenefit
        - liveTilePenalty
        - vfParameterPenalty
```

第一阶段可以复用现有 `ConservativeDAGGreedyStrategyEngine`，只将 `CostModel` 替换为 legality-only 版本。这样得到的是 greedy 意义下的最大合法 group。后续如果需要更完整的最大合法范围，可以再引入专门的 legality group builder。

### External vfsimulator Planner

PTOAS 在 costmodel mode 下将 legality-only group 交给外接 vfsimulator planner：

```text
legal tileop group
  -> vfsimulator planner
  -> strategy generation
  -> VF costmodel evaluation
  -> selected fusion plan
```

vfsimulator 侧负责：

| 职责 | 说明 |
| --- | --- |
| 解析 tileop group | 接收 PTOAS 给出的合法 group 信息 |
| 生成融合策略 | 基于 tileop 模板生成候选融合策略 |
| costmodel 评估 | 将候选策略喂给 VF costmodel |
| 策略选择 | 选择最优 strategy / unroll |
| 返回 plan | 将结果返回 PTOAS，供后续打标签和 VPTO 消费 |

第一阶段策略可以先保持简单：

```text
fuse_all + unroll search
```

即在合法 group 内采用应融尽融策略，然后枚举 unroll candidates，通过 VF costmodel 选择最优 unroll。

### 输入结构

PTOAS 内部生成的 group 当前是：

```cpp
struct PlannedFusionGroup {
  SmallVector<const pto::FusionComputeNode *, 8> members;
};
```

该结构不能直接作为外接 vfsimulator 的输入，因为其中包含 PTOAS/MLIR 内部对象，例如 `FusionComputeNode *`、`Operation *`、`Value` 等。vfsimulator 不应该依赖这些内部定义。

因此，PTOAS 侧需要提供一个 adapter，将 `PlannedFusionGroup` 转换为与 PTOAS 内部 IR 无关的中立结构。该中立结构暂定名为：

```cpp
TileOpGroupInfo
```

转换关系：

```text
PlannedFusionGroup
  -> PTOAS adapter
  -> TileOpGroupInfo
  -> vfsimulator planner
```

#### TileOpGroupInfo

`TileOpGroupInfo` 描述一个 legality-only tileop group：

```cpp
struct TileOpGroupInfo {
  std::string groupId;
  std::vector<TileOpInfo> ops;
  ShapeInfo iterationDomain;
  std::vector<std::string> externalOutputValueIds;
  std::vector<DataEdgeInfo> edges; // optional
};
```

| 字段 | 含义 |
| --- | --- |
| `groupId` | PTOAS 为本次请求分配的 group 标识，用于关联 vfsimulator 返回的 plan |
| `ops` | group 内 tileop 列表，必须按 PTOAS 原始 block 顺序排列 |
| `iterationDomain` | group 对应的迭代空间 |
| `externalOutputValueIds` | group 对外可见的输出 value，用于判断哪些结果必须保留/store |
| `edges` | 可选字段，显式描述 tileop 之间的数据依赖 |

第一阶段必需字段：

```text
groupId
ops
iterationDomain
externalOutputValueIds
```

第一阶段可选字段：

```text
edges
```

对于简单 elementwise group，vfsimulator 可以通过 `src` / `dst` 中的 `valueId` 推导数据依赖，因此 `edges` 可以暂时不作为必需输入。后续如果需要支持 fanout、join 或更复杂策略搜索，可以由 PTOAS adapter 显式填充 `edges`。

#### TileOpInfo

`TileOpInfo` 描述 group 内的一个 tileop：

```cpp
struct TileOpInfo {
  std::string opId;
  std::string opName;
  std::vector<ValueInfo> src;
  std::vector<ValueInfo> dst;
};
```

| 字段 | 含义 |
| --- | --- |
| `opId` | PTOAS adapter 生成的稳定 op 标识，用于输出 plan 时映射回原始 op |
| `opName` | tileop 名称，例如 `tadd`、`tmul`、`texp` |
| `src` | tileop 输入 operand 信息 |
| `dst` | tileop 输出 operand 信息 |

`TileOpGroupInfo::ops` 本身承载原始顺序：`ops[0]` 表示 group 中原始 block 顺序最早的 tileop，`ops[1]` 表示下一个 tileop。vfsimulator 如果需要生成 `fusionOrder`，可以基于该数组顺序生成。

#### ValueInfo

`ValueInfo` 描述 tileop 的输入或输出：

```cpp
enum class OperandKind {
  Tile,
  Scalar,
  Immediate,
  Unknown,
};

enum class DType {
  FP32,
  FP16,
  INT32,
  UINT32,
  INT16,
  UINT16,
  INT8,
  UINT8,
  BOOL,
  UNKNOWN,
};

struct ValueInfo {
  std::string valueId;
  OperandKind kind;
  DType dtype;
  ShapeInfo shape;
};
```

| 字段 | 含义 |
| --- | --- |
| `valueId` | PTOAS adapter 生成的稳定 value 标识 |
| `kind` | operand 类型，例如 tile、scalar、immediate |
| `dtype` | value 数据类型。dtype 跟随 value，而不是只放在 group 顶层 |
| `shape` | value 的 shape / valid shape 信息 |

该字段保持在 tileop 语义层，不表达 `UB`、`REG`、predicate register 等 VPTO/micro-op 层概念。第一阶段主要支持 `Tile` 和 `Scalar`；`Immediate` 用于后续常量 operand 或 attribute 形式的输入。

#### ShapeInfo

```cpp
struct ShapeInfo {
  std::vector<int64_t> dims;
  std::vector<int64_t> validDims;
  std::string layout;
};
```

| 字段 | 含义 |
| --- | --- |
| `dims` | 静态 shape |
| `validDims` | valid row / valid col 等有效区域信息 |
| `layout` | tile layout 信息，第一版可选 |

#### DataEdgeInfo

```cpp
struct DataEdgeInfo {
  std::string producerOpId;
  std::string producerValueId;
  std::string consumerOpId;
  std::string consumerValueId;
};
```

`DataEdgeInfo` 描述 tileop 之间的数据流依赖。它来自 `FusionBlockAnalysis::edges`，用于 vfsimulator 理解 group 内 producer/consumer 关系。第一阶段该字段可以作为 optional，不要求 vfsimulator 依赖它。

#### externalOutputValueIds

`externalOutputValueIds` 是 PTOAS adapter 从 `FusionBlockAnalysis::liveness` 和 `FusionBlockAnalysis::writeInstances` 中提取出的中立信息。

它表达的是：

```text
哪些 tileop 输出会逃逸出当前 fusion group，必须作为 group 的对外输出保留。
```

这样 vfsimulator 不需要理解 PTOAS 内部的 `FusionValueLiveness` / `FusionWriteInstanceLiveness` 结构，只需要根据 `externalOutputValueIds` 判断哪些结果需要在生成策略时保留或 store。

#### Adapter 位置

该转换应放在 PTOAS 侧，因为只有 PTOAS 能解释 `FusionComputeNode`、`Operation`、`Value` 和 MLIR type。建议后续单独实现一个 adapter，例如：

```text
lib/PTO/Transforms/TileFusion/VfsimFusionPlannerClient.cpp
include/PTO/Transforms/TileFusion/VfsimFusionPlannerClient.h
```

adapter 的核心职责：

```text
FusionComputeNode / Value / Type
  -> TileOpInfo / ValueInfo / ShapeInfo

FusionBlockAnalysis::edges
  -> DataEdgeInfo(optional)

FusionBlockAnalysis::liveness / writeInstances
  -> externalOutputValueIds

IterationDomainClass
  -> ShapeInfo iterationDomain
```

### 输出结构

vfsimulator planner 的输出不是直接修改后的 PTOAS IR，而是一个中立的 fusion plan。PTOAS adapter 收到该 plan 后，再将其 materialize 成 `pto.fusion.*` metadata。

输出结构暂定名为：

```cpp
FusionPlanInfo
```

转换关系：

```text
vfsimulator planner
  -> FusionPlanInfo
  -> PTOAS adapter
  -> pto.fusion.group_id / pto.fusion.order / pto.fusion.unroll
```

#### FusionPlanInfo

`FusionPlanInfo` 描述 vfsimulator 为一个 `TileOpGroupInfo` 选择出的融合计划：

```cpp
struct FusionPlanInfo {
  std::string groupId;
  int64_t predictedCycles; // optional/debug
  std::vector<TileOpPlanInfo> ops;
};
```

| 字段 | 含义 |
| --- | --- |
| `groupId` | 对应输入 `TileOpGroupInfo::groupId`，用于 PTOAS 将 plan 映射回原 group |
| `predictedCycles` | vfsimulator 预测时间，第一阶段主要用于 dump/debug，不一定写入 IR |
| `ops` | vfsimulator 选择参与融合的 tileop 及其融合顺序 |

第一阶段采用应融尽融时，`ops` 应包含输入 `TileOpGroupInfo::ops` 中的全部 tileop。后续如果 vfsimulator 支持更复杂策略，也可以只返回被选择融合的 tileop 子集。

#### TileOpPlanInfo

`TileOpPlanInfo` 描述单个 tileop 在 fusion plan 中的 metadata：

```cpp
struct TileOpPlanInfo {
  std::string opId;
  int64_t fusionOrder;
  int64_t unroll;
};
```

| 字段 | 含义 |
| --- | --- |
| `opId` | 对应输入 `TileOpInfo::opId`，用于 PTOAS 映射回原始 tileop |
| `fusionOrder` | tileop 在融合后的 group 内顺序，对应后续 `pto.fusion.order` |
| `unroll` | 该 tileop 对应 loop 的 unroll，对应后续 `pto.fusion.unroll` |

第一阶段如果 vfsimulator 不改变 tileop 顺序，可以直接按 `TileOpGroupInfo::ops` 的数组顺序生成 `fusionOrder`：

```text
ops[0] -> fusionOrder = 0
ops[1] -> fusionOrder = 1
ops[2] -> fusionOrder = 2
```

第一阶段如果采用 group 内统一 unroll，可以让所有 `TileOpPlanInfo::unroll` 填同一个值。后续如果区分 VF fusion 和 loop fusion，例如只做 VF fusion、不做 loop fusion，则不同 tileop 可以返回不同的 `unroll`。更完整的 `loopId` / `LoopPlanInfo` 可以作为后续扩展。

#### 输出到 IR Metadata 的映射

PTOAS adapter 根据 `FusionPlanInfo` 回写 IR metadata：

| FusionPlanInfo 字段 | PTOAS metadata |
| --- | --- |
| `groupId` | `pto.fusion.group_id` |
| `TileOpPlanInfo::fusionOrder` | `pto.fusion.order` |
| `TileOpPlanInfo::unroll` | `pto.fusion.unroll` |
| `predictedCycles` | 可选 dump/debug 信息 |

PTOAS adapter 需要维护：

```text
opId -> Operation *
```

这样才能将 vfsimulator 返回的 `TileOpPlanInfo` 映射回具体 tileop 并写入 metadata。

### Metadata 扩展

当前已有标签：

| 标签 | 含义 |
| --- | --- |
| `pto.fusion.group_id` | tileop 所属 fusion group |
| `pto.fusion.order` | tileop 在 group 内顺序 |

costmodel mode 下需要额外表达 vfsimulator 返回的策略信息。第一阶段建议至少增加：

| 标签 | 含义 |
| --- | --- |
| `pto.fusion.unroll` | vfsimulator 选择的最优 unroll |

第一阶段可以先将 `pto.fusion.unroll` 打在 group 内每个 tileop 上；在 `pto-fusion-region-gen` 生成 `pto.fusion_region` 后，再转移或汇总到 `pto.fusion_region` 上。长期建议 VPTO 后端主要消费 region-level metadata。

### Fallback 策略

外接 costmodel 可能出现不可用、unsupported、超时或返回 invalid plan 的情况，因此需要明确 fallback 行为。

建议默认采用 fail-open：

```text
costmodel 不可用
  -> 回退到 ConservativeDAGGreedyCostModel
  -> 保持当前 PTOAS 行为
```

可选策略：

| 策略 | 行为 | 适用场景 |
| --- | --- | --- |
| fail-open | costmodel 失败时回退现有启发式策略 | 默认推荐 |
| no-fusion | costmodel 失败时禁用当前 group fusion | 调试 costmodel 正确性 |
| fail-closed | costmodel 失败时报错终止编译 | 严格验证模式 |

### 推荐演进路径

第一阶段：

```text
复用 ConservativeDAGGreedyStrategyEngine
新增 LegalityOnlyFusionCostModel
生成 legality-only group
vfsimulator 执行 fuse_all + unroll search
返回 pto.fusion.unroll
```

第二阶段：

```text
vfsimulator 支持更多融合策略
PTOAS 消费更完整的 strategy metadata
VPTO 后端基于 region-level metadata 生成融合 VF
```

第三阶段：

```text
如 greedy 最大合法 group 不够，再引入 dedicated legality-group builder
进一步减少 PTOAS 中的策略判断逻辑
```

## 外挂集成方式

基于前面定义的输入输出形式，推荐采用：

```text
PTOAS 内部 adapter + 外部 vfsimulator planner 动态库插件
```

整体链路：

```text
PTOAS
  PlannedFusionGroup
  -> TileOpGroupInfo
  -> libvfsim_fusion_planner.so
  -> FusionPlanInfo
  -> pto.fusion.* metadata
```

### 推荐方式

正式集成建议采用：

```text
dynamic library plugin + stable C ABI
```

也就是 vfsimulator 编译出一个动态库，例如：

```text
libvfsim_fusion_planner.so
```

PTOAS 在 `FusionPlan` 层通过一个 client 加载该插件，并调用稳定 C ABI。

### 职责划分

| 组件 | 职责 |
| --- | --- |
| `FusionPlanPass` | 生成 legality-only group，调用 vfsimulator planner client，回写 metadata |
| PTOAS adapter | `PlannedFusionGroup` -> `TileOpGroupInfo`；`FusionPlanInfo` -> `pto.fusion.*` |
| vfsimulator plugin | `TileOpGroupInfo` -> strategy generation -> VF costmodel evaluation -> `FusionPlanInfo` |
| fallback path | 插件不可用或失败时回退现有 `ConservativeDAGGreedyCostModel` |

建议 PTOAS 侧新增独立 client 文件：

```text
lib/PTO/Transforms/TileFusion/VfsimFusionPlannerClient.cpp
include/PTO/Transforms/TileFusion/VfsimFusionPlannerClient.h
include/PTO/Transforms/TileFusion/VfsimFusionPlannerCAPI.h
```

### C ABI 形式

PTOAS 和 vfsimulator 共同维护一个稳定 C ABI header。示意接口：

```cpp
extern "C" const VfsimPlannerApi *vfsimGetPlannerApi(uint32_t apiVersion);
```

API table 示例：

```cpp
struct VfsimPlannerApi {
  uint32_t apiVersion;

  VfsimStatus (*planFusionGroup)(
      const VfsimTileOpGroupInfo *input,
      VfsimFusionPlanInfo *output);

  void (*freeFusionPlanInfo)(VfsimFusionPlanInfo *output);
};
```

PTOAS 调用流程：

```text
build TileOpGroupInfo
  -> api->planFusionGroup(input, output)
  -> materialize FusionPlanInfo into IR metadata
  -> api->freeFusionPlanInfo(output)
```

### 为什么选择动态库插件

| 方案 | 评价 |
| --- | --- |
| 源码级接入 | 耦合较强，vfsimulator 内部修改容易影响 PTOAS |
| CLI/JSON | 原型和 debug 方便，但频繁进程启动和解析开销不适合编译器在线优化主路径 |
| C++ API | 调用方便，但 ABI 稳定性弱，容易受编译器/标准库/LLVM 版本影响 |
| 动态库插件 + C ABI | 解耦程度较高，适合在线调用，接口边界清晰 |

因此，推荐：

```text
主路径：动态库插件 + 稳定 C ABI
调试路径：保留可 dump 的文本格式
代码管理：可选 submodule，但 submodule 不作为接口协议本身
```

### 短期落地方式

第一阶段可以先不直接接真实 vfsimulator 动态库，而是在 PTOAS 侧实现一个 mock planner：

```text
VfsimFusionPlannerClient
  -> mock planner
  -> fuse_all
  -> 固定或简单搜索 unroll
  -> 生成 FusionPlanInfo
```

这样可以先验证：

1. `PlannedFusionGroup` 到 `TileOpGroupInfo` 的转换；
2. `FusionPlanInfo` 到 `pto.fusion.*` metadata 的回写；
3. `pto-op-scheduling` / `pto-mark-last-use` / `pto-fusion-region-gen` 是否能继续消费这些 metadata；
4. VPTO 后端是否能看到并使用新增的 `pto.fusion.unroll`。

之后再将 mock planner 替换为真正的：

```text
libvfsim_fusion_planner.so
```
