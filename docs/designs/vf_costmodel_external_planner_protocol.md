# VF CostModel External Planner Protocol

本文档定义 PTOAS 与外接 VfSim CostModel 的规划接口。目标是让 PTOAS 保留融合合法性判断，让 VfSim 负责生成融合策略、进行 unroll 搜索并返回最终 fusion plan。

## Data Flow

```text
PTOAS PreFusionAnalysis / FusionPlan legality
  -> legal fusion group
  -> PTOAS adapter
  -> FusionGroupInfo
  -> VfSim planner
  -> template-based micro-op program generation
  -> costmodel evaluation
  -> FusionPlanInfo
  -> PTOAS writes fusion attrs
  -> scheduling / fusion region generation / VPTO backend
```

## Responsibility Boundary

| Component | Responsibility |
| --- | --- |
| PTOAS FusionPlan | 判断 tileop 是否可以融合，生成合法 fusion group |
| PTOAS Adapter | 将 PTOAS 内部对象转换为不含 PTOAS 指针的 `FusionGroupInfo` |
| VfSim Planner | 基于 group 生成候选 fusion 策略，扫描 unroll，调用 costmodel |
| VfSim Template Registry | 维护和 PTOAS tileop lowering 语义一致的性能模板 |
| PTOAS Result Consumer | 根据 `FusionPlanInfo` 给 tileop 写 metadata |
| VPTO Backend | 消费 metadata，生成融合后的 VF / loop IR |

## Input Abstraction

PTOAS 传给 VfSim 的输入是一个经过合法性验证的 fusion group。该结构只描述 tileop 语义、输入输出、shape、dtype 和模板选择参数，不暴露 PTOAS 内部指针。

```cpp
struct FusionGroupInfo {
  std::string groupId;
  std::vector<TileOpInfo> ops;
};

struct TileOpInfo {
  std::string opId;
  std::string opName;
  uint64_t originalOrder;

  std::vector<ValueInfo> inputs;
  std::vector<ValueInfo> outputs;

  std::vector<TemplateParamInfo> templateParams;
};

struct ValueInfo {
  std::string valueId;
  ValueKind kind;
  ShapeInfo shape;
  DType dtype;
  bool externalOutput;
};

struct ShapeInfo {
  std::vector<int64_t> shape;
  std::vector<int64_t> validShape;
  LayoutKind layout;
};

struct TemplateParamInfo {
  std::string key;
  std::string value;
};
```

### FusionGroupInfo

| Field | Meaning |
| --- | --- |
| `groupId` | PTOAS 侧生成的 group 标识，用于结果回写和日志关联 |
| `ops` | group 内 tileop 列表，顺序为合法性规划后的初始顺序 |

### TileOpInfo

| Field | Meaning |
| --- | --- |
| `opId` | tileop 的稳定标识，VfSim 输出结果通过它回写到 PTOAS |
| `opName` | tileop 类型，例如 `tadd`、`tcvt`、`tdivs` |
| `originalOrder` | tileop 在原 block 中的顺序，用于稳定排序和调试 |
| `inputs` | tileop 输入 value 列表 |
| `outputs` | tileop 输出 value 列表 |
| `templateParams` | 影响模板选择或 micro-op 结构的参数 |

### ValueInfo

| Field | Meaning |
| --- | --- |
| `valueId` | value 的稳定标识，用于重建 tileop 之间的数据依赖 |
| `kind` | value 类型，例如 vector、scalar、constant |
| `shape` | shape、valid shape 和 layout 信息 |
| `dtype` | 当前 value 的数据类型，支持输入输出 dtype 不同的 op，例如 `tcvt` |
| `externalOutput` | 仅对 output 有意义，表示该输出需要保留到 fusion group 外部 |

### ShapeInfo

| Field | Meaning |
| --- | --- |
| `shape` | 逻辑 shape |
| `validShape` | 有效计算 shape，用于推导实际循环次数和 mask |
| `layout` | 数据布局，用于选择模板和推导访存模式 |

### TemplateParamInfo

`templateParams` 只传递 VfSim 生成模板需要的信息。典型字段包括：

| Key | Meaning |
| --- | --- |
| `round_mode` | 类型转换或近似计算的 rounding 模式 |
| `convert_mode` | `tcvt` 类 op 的转换模式 |
| `scalar_operand_kind` | scalar/broadcast operand 的结构信息 |
| `mask_mode` | mask 或 valid shape 无法直接表达时的补充信息 |
| `approx_mode` | 近似计算模板选择信息 |

## Dependency And Loop Inference

VfSim 通过 `valueId` 建立 tileop 之间的数据依赖：

```text
producer.outputs[i].valueId == consumer.inputs[j].valueId
```

VfSim 通过以下信息推导模板和循环结构：

```text
opName + inputs/outputs dtype + shape/validShape/layout + templateParams
  -> VfSim template registry
  -> loop count / micro-op sequence / live-out stores
```

unroll 合法性由 VfSim 根据推导出的循环次数判断。第一阶段扫描不超过 8 的循环次数因子，例如 `1, 2, 3, 4, 6, 8`。

## Template Registry

VfSim 维护独立的性能模板库。模板语义与 PTOAS tileop lowering 保持一致，用于预测融合后 VF 的 micro-op 形式。

| TileOp | Template Output |
| --- | --- |
| `tadd` / `tadds` | elementwise add micro-op |
| `tmul` / `tmuls` | elementwise multiply micro-op |
| `tcvt` | conversion micro-op，输入输出 dtype 可不同 |
| `tdivs` | scalar/vector divide 模板，可展开为 `vbr` + `vdiv` 语义 |
| `texp` | exp micro-op |

模板输出需要标记哪些中间值只在 group 内使用，哪些输出需要通过 `externalOutput` materialize 到 group 外。

## Output Abstraction

VfSim 返回一个 fusion plan。PTOAS 根据该结果写 metadata，后续 pass 继续消费这些 metadata。

```cpp
struct FusionPlanInfo {
  std::string groupId;
  std::vector<TileOpPlanInfo> ops;
  std::vector<StrategyCandidate> candidates;
};

struct TileOpPlanInfo {
  std::string opId;

  uint64_t order;
  int64_t unroll;

  std::string loopFusionId;
  bool innerLoopExpand;
};

struct StrategyCandidate {
  std::string strategyName;
  std::vector<TemplateParamInfo> params;
  bool valid;
  int64_t estimatedCycles;
  std::string rejectReason;
};
```

### FusionPlanInfo

| Field | Meaning |
| --- | --- |
| `groupId` | 对应输入 `FusionGroupInfo::groupId` |
| `ops` | 每个 tileop 的规划结果 |
| `candidates` | 候选策略的耗时和诊断信息，用于 debug dump |

### TileOpPlanInfo

| Field | Meaning |
| --- | --- |
| `opId` | 对应输入 tileop |
| `order` | VF fusion 内 tileop 的执行顺序 |
| `unroll` | 该 tileop 主 inner loop 的 unroll 值 |
| `loopFusionId` | 相同 ID 表示这些 tileop 融合到同一个 loop |
| `innerLoopExpand` | 是否将 tileop 内部嵌套 inner loop 展开成单层 loop |

### StrategyCandidate

| Field | Meaning |
| --- | --- |
| `strategyName` | 候选策略名，例如 `fuse_all` |
| `params` | 候选参数，例如 `unroll=1/2/4/8` |
| `valid` | 该候选是否可评估 |
| `estimatedCycles` | costmodel 预测 cycle |
| `rejectReason` | 候选无效时的原因 |

## Metadata Written Back To PTOAS

第一阶段 PTOAS 主要写回以下 metadata：

| Metadata | Meaning |
| --- | --- |
| `pto.fusion.group_id` | 哪些 tileop 属于同一个 fusion group |
| `pto.fusion.order` | group 内 tileop 顺序 |
| `pto.fusion.unroll` | VfSim 选出的 loop unroll 值 |
| `pto.fusion.loop_id` | 哪些 tileop 融合到同一个 loop |
| `pto.fusion.inner_loop_expand` | 是否展开 tileop 内部 inner loop |

当前阶段 elementwise case 可以让同一个 `group_id` 下的 tileop 使用相同 `loop_id`。复杂 case 中，`group_id` 表示 VF fusion 范围，`loop_id` 表示 loop fusion 范围。

## First Phase Scope

| Category | Scope |
| --- | --- |
| Supported | elementwise、scalar-elementwise、simple conversion |
| Examples | `tadd`、`tadds`、`tmul`、`tmuls`、`tcvt`、`tdivs`、`texp` |
| Planning | legal group 输入，VfSim 采用 fuse-all + unroll search |
| Unroll Search | 扫描不超过 8 的循环次数因子 |
| Diagnostics | dump 每个候选 unroll 的预测 cycle |
| Fallback | VfSim 返回 `fallback` 时，PTOAS 使用默认策略 |

## Implementation Form

源码级穿刺阶段可以直接使用 C++ `struct + std::vector + std::string`。

长期外挂阶段采用同一抽象模型，并机械转换为 C ABI：

```text
std::vector<T>      -> const T *data + uint64_t size
std::string        -> const char *
enum class         -> stable integer enum
```

这样 PTOAS 只依赖稳定接口，VfSim 可以独立维护 costmodel、模板库和策略搜索逻辑。

## Integration Form

长期接入建议采用 `submodule + dynamic library` 的形式：

```text
PTOAS repo
  -> third_party/vfsimulator      # git submodule
  -> include/vfsim_planner_api.h  # stable protocol header
  -> link libvfsim_planner.so
```

| Part | Role |
| --- | --- |
| `third_party/vfsimulator` | 以 submodule 形式固定 VfSim 版本 |
| `vfsim_planner_api.h` | PTOAS 与 VfSim 共享的稳定接口定义 |
| `libvfsim_planner.so` | VfSim 编译出的动态库，提供 planner 和 costmodel 实现 |
| PTOAS adapter | 将 PTOAS 内部 fusion group 转换为接口结构 |
| VfSim planner | 接收接口结构，生成 `FusionPlanInfo` |

推荐的调用方向：

```text
PTOAS build
  -> build third_party/vfsimulator
  -> generate libvfsim_planner.so
  -> PTOAS links libvfsim_planner.so

PTOAS compile pipeline
  -> construct FusionGroupInfo
  -> call vfsimPlanFusion(...)
  -> receive FusionPlanInfo
  -> write PTOAS fusion attrs
```

接口头文件只放协议结构和入口函数。VfSim 内部的模板库、costmodel 参数、搜索策略都留在 VfSim 仓库中维护。

示例入口：

```cpp
extern "C" bool vfsimPlanFusion(
    const FusionGroupInfo *input,
    FusionPlanInfo *output,
    VfsimDiagnosticInfo *diagnostics);
```

`submodule` 用来管理源码版本，`dynamic library` 用来隔离实现细节。PTOAS 侧只需要适配协议，不直接依赖 VfSim 内部代码结构。
