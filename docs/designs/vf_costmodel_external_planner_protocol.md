# VF CostModel External Fusion Planner Protocol

本文记录 PTOAS 与 vfsimulator 解耦后的推荐接口方案。目标是让 PTOAS 只负责 tileop 合法融合分组，vfsimulator 负责融合策略生成、costmodel 评估和策略选择。

## 1. PTOAS 当前 TileOp Fusion 分层

当前 PTOAS tileop fusion 路径可以理解为以下几层：

```text
PreFusionAnalysis
  -> FusionPlan / StrategyEngine / CostModel
  -> PTOOpScheduling / PTOMarkLastUse / PTOFusionRegionGen
  -> VPTO backend fusion-region local optimization and lowering
```

### 1.1 PreFusionAnalysis

对应 pass：

```text
pto-pre-fusion-analysis
```

职责：

- 扫描 tile-native IR 中的 tileop。
- 抽取 `FusionComputeNode`、tile 输入输出、SSA 数据依赖、block order、iteration domain 等信息。
- 判断每个 tileop 是否具备可分析的 fusion semantics。
- 给后续 planner 提供 block-local analysis result。

需要注意：当前 `PreFusionAnalysis` 主要生成可复用分析状态和候选图，不直接生成最终 fusion group。

### 1.2 FusionPlan / Strategy Decision

对应 pass：

```text
pto-fusion-plan
```

当前实现：

```text
ConservativeDAGGreedyStrategyEngine
ConservativeDAGGreedyCostModel
```

职责：

- 基于 `PreFusionAnalysis` 的 compute graph 形成 fusion group。
- 当前 `ConservativeDAGGreedyCostModel` 同时承担两类判断：
  - 合法性/保守约束：op 是否支持、iteration domain 是否 proven、是否有 hard boundary、是否有数据流连接。
  - 收益启发式：dependency benefit、loop merge benefit、live tile penalty、vf parameter penalty。
- 对被接受的 group 打 metadata：

```text
pto.fusion.group_id
pto.fusion.order
```

因此，当前 PTOAS 第二层并不只是“能否融合”，也包含了“是否值得融合”的启发式收益判断。

### 1.3 Scheduling / Region Materialization

对应 pass：

```text
pto-op-scheduling
pto-mark-last-use
pto-fusion-region-gen
```

职责：

- 根据 `pto.fusion.group_id` / `pto.fusion.order` 调整 group 内 op 顺序。
- 标注 last-use 信息。
- 将连续 fusion span 包装成 `pto.fusion_region`。

### 1.4 VPTO Backend Consumption

当前 VPTO 相关 fusion-region 后端 pass 包括：

```text
pto-low-level-loop-fusion
pto-fusion-predicate-elision
pto-fusion-load-store-elision
pto-flatten-fusion-region
```

职责：

- 在 `pto.fusion_region` 内做 VPTO post-lowering loop fusion。
- 做 fusion-local predicate/load/store cleanup。
- 最后 flatten fusion region。

需要注意：从当前代码看，VPTO 后端已经会消费 `pto.fusion_region` 这一层结构做局部优化；但“由前端 metadata 直接驱动生成完整融合 VF micro-op”的能力仍需要继续明确接口和实现边界。

## 2. 目标分层

解耦后希望调整为：

```text
PTOAS
  1. 判断 tileop 是否具备合法融合条件
  2. 形成 tileop fusion candidate group
  3. 将 group 交给 vfsimulator external planner
  4. 消费 planner 返回的 fusion plan
  5. 打 metadata / 生成 fusion_region / 进入 VPTO 后端

vfsimulator
  1. 接收 tileop group
  2. 基于 tileop 模板生成融合策略
  3. 第一阶段采用 fuse-all policy
  4. 对 unroll candidates 做寻优
  5. 调用 VF costmodel 评估
  6. 返回 selected fusion plan
```

核心变化：

- PTOAS 不再做融合收益判断。
- PTOAS 不维护 costmodel 强相关的策略生成逻辑。
- vfsimulator 成为 fusion strategy 和 costmodel 的 owner。
- PTOAS 与 vfsimulator 之间通过稳定 planner protocol 交互。

## 3. 推荐接口形态

推荐正式集成形态：

```text
dynamic library plugin + stable C ABI
```

PTOAS 运行时加载 vfsimulator planner plugin：

```text
libvfsim_fusion_planner.so
```

插件暴露一个稳定 C ABI 入口：

```c
extern "C" const VfsimPluginApi *vfsimGetPluginApi(uint32_t requestedApiVersion);
```

PTOAS 只依赖一个很薄的 public header，例如：

```text
VfsimFusionPlannerCAPI.h
```

vfsimulator 内部可以用 Python、C++ 或其他实现，但对 PTOAS 暴露的 ABI 保持稳定。

## 4. Plugin API

### 4.1 API Table

```c
typedef struct VfsimPluginApi {
  uint32_t apiVersion;

  VfsimStatus (*planFusionGroups)(
      const VfsimFusionPlanRequest *request,
      VfsimFusionPlanResponse *response);

  void (*freeFusionPlanResponse)(VfsimFusionPlanResponse *response);

  const char *(*getVersionString)(void);
} VfsimPluginApi;
```

### 4.2 Status

```c
typedef enum VfsimStatusCode {
  VFSIM_STATUS_OK = 0,
  VFSIM_STATUS_UNSUPPORTED_API_VERSION = 1,
  VFSIM_STATUS_INVALID_REQUEST = 2,
  VFSIM_STATUS_UNSUPPORTED_GROUP = 3,
  VFSIM_STATUS_MODEL_ERROR = 4,
  VFSIM_STATUS_INTERNAL_ERROR = 5,
} VfsimStatusCode;

typedef struct VfsimStatus {
  VfsimStatusCode code;
  const char *message;
} VfsimStatus;
```

Top-level `VfsimStatus` 表示整个调用是否完成。单个 group 是否 supported 由 response 内的 per-group result 表达。

## 5. Request Schema

### 5.1 FusionPlanRequest

```c
typedef struct VfsimFusionPlanRequest {
  uint32_t schemaVersion;
  VfsimTargetDesc target;
  VfsimPlanningOptions options;

  const VfsimTileOpGroup *groups;
  uint64_t numGroups;
} VfsimFusionPlanRequest;
```

### 5.2 TargetDesc

```c
typedef struct VfsimTargetDesc {
  const char *arch;        // example: "a5"
  uint32_t vectorBytes;    // example: 256
  const char *profile;     // optional, example: "default"
} VfsimTargetDesc;
```

### 5.3 PlanningOptions

```c
typedef enum VfsimFusionPolicy {
  VFSIM_FUSION_POLICY_FUSE_ALL = 0,
  VFSIM_FUSION_POLICY_SEARCH = 1,
} VfsimFusionPolicy;

typedef struct VfsimPlanningOptions {
  VfsimFusionPolicy policy;

  const uint32_t *unrollCandidates;
  uint64_t numUnrollCandidates;

  bool requestAlternatives;
  bool requestDiagnostics;
} VfsimPlanningOptions;
```

第一阶段建议：

```text
policy = VFSIM_FUSION_POLICY_FUSE_ALL
unrollCandidates = [1, 2, 4, 8]
```

### 5.4 TileOpGroup

```c
typedef struct VfsimTileOpGroup {
  const char *groupId;
  VfsimIterationDomain iterationDomain;

  const VfsimTileOpDesc *ops;
  uint64_t numOps;

  const VfsimDataEdge *edges;
  uint64_t numEdges;

  const char *const *externalOutputValueIds;
  uint64_t numExternalOutputs;
} VfsimTileOpGroup;
```

PTOAS 应保证传入 group 已经满足前端合法性约束，例如：

- group 内 tileop 位于同一个可融合范围。
- 没有 call、region、terminator 等 hard boundary。
- iteration domain 已 proven 或能被协议明确表达。
- side effect 和 memory visibility 满足融合要求。

### 5.5 TileOpDesc

```c
typedef struct VfsimTileOpDesc {
  const char *opId;
  const char *opName;      // example: "tadd", "texp", "trowexpandmul"
  uint64_t originalOrder;  // stable order in PTOAS block

  const VfsimValueDesc *inputs;
  uint64_t numInputs;

  const VfsimValueDesc *outputs;
  uint64_t numOutputs;

  const VfsimAttribute *attributes;
  uint64_t numAttributes;
} VfsimTileOpDesc;
```

### 5.6 ValueDesc

```c
typedef enum VfsimValueRole {
  VFSIM_VALUE_TILE = 0,
  VFSIM_VALUE_SCALAR = 1,
  VFSIM_VALUE_MEMORY = 2,
  VFSIM_VALUE_PREDICATE = 3,
  VFSIM_VALUE_IMMEDIATE = 4,
} VfsimValueRole;

typedef enum VfsimDType {
  VFSIM_DTYPE_FP32 = 0,
  VFSIM_DTYPE_FP16 = 1,
  VFSIM_DTYPE_INT32 = 2,
  VFSIM_DTYPE_UINT32 = 3,
  VFSIM_DTYPE_INT16 = 4,
  VFSIM_DTYPE_UINT16 = 5,
  VFSIM_DTYPE_INT8 = 6,
  VFSIM_DTYPE_UINT8 = 7,
  VFSIM_DTYPE_BOOL = 8,
  VFSIM_DTYPE_UNKNOWN = 255,
} VfsimDType;

typedef struct VfsimValueDesc {
  const char *valueId;
  VfsimValueRole role;
  VfsimDType dtype;
  VfsimShapeDesc shape;
} VfsimValueDesc;
```

`dtype` 跟随 value，而不是只放在 group 顶层。这样可以支持 `vcvt` 等输入输出精度不同的 tileop。

### 5.7 ShapeDesc

```c
typedef struct VfsimShapeDesc {
  const int64_t *dims;
  uint64_t rank;

  const int64_t *validDims;
  uint64_t validRank;

  const char *layout;
} VfsimShapeDesc;
```

### 5.8 DataEdge

```c
typedef struct VfsimDataEdge {
  const char *producerOpId;
  const char *producerValueId;
  const char *consumerOpId;
  const char *consumerValueId;
} VfsimDataEdge;
```

## 6. Response Schema

### 6.1 FusionPlanResponse

```c
typedef struct VfsimFusionPlanResponse {
  uint32_t schemaVersion;

  VfsimFusionPlanResult *results;
  uint64_t numResults;
} VfsimFusionPlanResponse;
```

Response 内存由 vfsimulator plugin 分配，由 PTOAS 调用：

```c
freeFusionPlanResponse(response)
```

释放。

### 6.2 FusionPlanResult

```c
typedef enum VfsimPlanRecommendation {
  VFSIM_PLAN_ACCEPT = 0,
  VFSIM_PLAN_REJECT = 1,
  VFSIM_PLAN_FALLBACK = 2,
} VfsimPlanRecommendation;

typedef struct VfsimFusionPlanResult {
  const char *groupId;
  bool supported;

  VfsimStatusCode statusCode;
  const char *reason;

  VfsimPlanRecommendation recommendation;
  VfsimSelectedStrategy selectedStrategy;

  VfsimStrategyCandidate *alternatives;
  uint64_t numAlternatives;

  VfsimFusionPlanMetadata metadata;
} VfsimFusionPlanResult;
```

### 6.3 SelectedStrategy

```c
typedef struct VfsimSelectedStrategy {
  const char *strategyName;    // example: "fuse_all"
  uint32_t selectedUnroll;
  int64_t estimatedCycles;
  const char *cycleMetric;     // example: "vf_end_cycle"
} VfsimSelectedStrategy;
```

### 6.4 StrategyCandidate

```c
typedef struct VfsimStrategyCandidate {
  const char *strategyName;
  uint32_t unroll;
  int64_t estimatedCycles;
  bool valid;
  const char *rejectReason;
} VfsimStrategyCandidate;
```

### 6.5 FusionPlanMetadata

```c
typedef struct VfsimFusionPlanMetadata {
  int64_t assignedGroupId;
  uint32_t selectedUnroll;

  const VfsimOpPlan *opPlans;
  uint64_t numOpPlans;
} VfsimFusionPlanMetadata;
```

`assignedGroupId` 可以由 PTOAS 分配，也可以由 vfsimulator 返回逻辑 group id。第一阶段建议 PTOAS 保持最终 metadata id 分配权，vfsimulator 返回 group-local plan 信息即可。

### 6.6 OpPlan

```c
typedef struct VfsimOpPlan {
  const char *opId;
  uint64_t fusionOrder;
} VfsimOpPlan;
```

第一阶段 `OpPlan` 可以只返回 group 内顺序。后续可以扩展：

- strategy-specific lowering hint。
- tile template choice。
- materialization hint。
- load/store elision hint。
- predicate handling hint。

## 7. 第一阶段行为

第一阶段只要求支持 elementwise group：

```text
tadd / tadds
tsub / tsubs
tmul / tmuls
tdiv / tdivs
tmax / tmaxs
tmin / tmins
texp
```

vfsimulator 行为：

1. 接收 PTOAS 传入的合法 tileop group。
2. 检查 group 是否属于当前支持的 pattern。
3. 采用 `fuse_all` policy。
4. 对 unroll candidates 逐个生成内部融合策略。
5. 每个策略生成内部 `VfSimProgram` 或等价 micro-op program。
6. 调用 VF costmodel 预测 cycles。
7. 返回 cycles 最小的策略。

PTOAS 行为：

1. 如果 result 为 `VFSIM_PLAN_ACCEPT`，使用返回的 op order / unroll / metadata hint 打标。
2. 如果 result 为 `VFSIM_PLAN_REJECT`，不融合该 group。
3. 如果 result 为 `VFSIM_PLAN_FALLBACK` 或 plugin 调用失败，回退到当前 PTOAS 保守策略，或按编译选项选择 fail-open / fail-closed。

## 8. 与当前 PTOAS 实现的改动点

当前 `ConservativeDAGGreedyCostModel` 的职责需要拆分：

```text
当前：
  legality check + benefit heuristic

目标：
  legality check 留在 PTOAS
  benefit heuristic / strategy search 迁到 vfsimulator
```

建议新增或重命名一个 PTOAS 侧组件：

```text
TileFusionLegalityPlanner
```

职责：

- 基于 `PreFusionAnalysis` 形成合法 candidate group。
- 不计算收益。
- 不根据 dependencyBenefit / penalty 做 accept/reject。
- 将合法 group 发给 vfsimulator planner。

当前 `FusionPlanPass` 可以演进为：

```text
PreFusionAnalysis
  -> TileFusionLegalityPlanner
  -> VfsimExternalFusionPlannerClient
  -> assign pto.fusion.group_id / pto.fusion.order / optional attrs
```

## 9. Open Questions

仍需 PTOAS 与 vfsimulator 双方确认的问题：

1. PTOAS 侧最终是否保留 group id 分配权。
2. vfsimulator 返回的 plan 是否只包含 metadata，还是需要返回更具体的 lowering hint。
3. 第一阶段 fail-open 策略：
   - plugin 不可用时继续用 ConservativeDAGGreedyCostModel。
   - plugin 不可用时禁用 fusion。
   - plugin 不可用时报错。
4. `trowexpand*`、reduction、selection、conversion 等复杂 tileop 何时进入协议第一版。
5. VPTO 后端消费的是 `pto.fusion_region`，还是未来直接消费 vfsimulator 返回的更具体 fusion plan。
6. vfsimulator 是否需要返回用于离线复现的 opaque debug artifact id。

## 10. 推荐结论

推荐第一版协议边界：

```text
PTOAS -> legal tileop group
vfsimulator -> selected fusion plan
```

推荐第一版集成方式：

```text
dynamic library plugin + stable C ABI
```

推荐第一版策略：

```text
fuse_all + unroll search
```

推荐第一版 PTOAS 改造：

```text
将 ConservativeDAGGreedyCostModel 拆成 legality planner 和收益 planner。
PTOAS 主路径只保留 legality planner。
收益判断、策略生成和 unroll 寻优迁到 vfsimulator。
```
