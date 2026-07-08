# VF CostModel Planner IR Protocol

本文档定义 PTOAS 与 VfSim CostModel 的长期接入方案。当前推荐形式是：

```text
submodule source-level integration + IR protocol
```

PTOAS 保留 tileop 融合合法性判断，VfSim 负责策略生成、costmodel 评估和优化建议。双方通过 IR 交换信息，避免额外维护一套与 MLIR 重复的结构体 ABI。

## Integration Form

VfSim 以 submodule 方式放在 PTOAS 仓库中，源码级参与 PTOAS 编译：

```text
PTOAS/
  third_party/
    vfsimulator/        # git submodule, points to a fixed VfSimulator commit
```

| Part | Role |
| --- | --- |
| `third_party/vfsimulator` | 独立 VfSimulator 仓库，PTOAS 只记录 commit hash |
| PTOAS CMake | `add_subdirectory(third_party/vfsimulator)` |
| VfSim planner library | 编译进 PTOAS，提供 C++ API |
| IR protocol | PTOAS 与 VfSim 之间的输入输出载体 |

submodule 的隔离方式：

```text
VfSim 源码修改
  -> commit in VfSimulator repo
  -> PTOAS updates third_party/vfsimulator submodule pointer
```

PTOAS 主仓不直接接管 VfSim 源码历史，只记录所依赖的 VfSim 版本。

## Two Compiler Routes

PTOAS 需要支持两条 costmodel 使用路线：

| Route | Input | Output | Purpose |
| --- | --- | --- | --- |
| 自动 tileop 融合 | legal tileop fusion candidate IR | planned tileop IR with fusion attrs | 由 VfSim 决定 fusion plan |
| 手写 VF 分析 | developer-written VF IR | annotated VF IR with cost/advice attrs | 评估已有 VF 并给优化建议 |

整体链路：

```text
Route A: PTOAS automatic fusion

.pto source
  -> PTOAS tileop IR
  -> PreFusionAnalysis / legality planning
  -> costmodel-facing tileop candidate IR
  -> VfSim planner
  -> tileop IR with fusion attrs
  -> PTOAS scheduling / fusion_region / VPTO backend
```

```text
Route B: developer-written VF

developer VF IR
  -> VfSim analyzer
  -> VF IR with cycle / bottleneck / advice attrs
  -> PTOAS VPTO backend or report
```

## Responsibility Boundary

| Component | Responsibility |
| --- | --- |
| PTOAS legality planner | 判断 tileop 是否可以进入 candidate group |
| PTOAS IR adapter | 生成 costmodel-facing IR，并保留必要 attrs |
| VfSim planner | 从 IR 中读取 tileop、shape、dtype、依赖和外部输出信息 |
| VfSim template registry | 维护与 PTOAS lowering 语义一致的性能模板 |
| VfSim costmodel | 生成候选策略，扫描 unroll，预测 cycle |
| PTOAS consumer | 消费 VfSim 写回的 IR attrs，继续后端 lowering |

## Route A Input IR

自动融合路线的输入是一段已经通过 PTOAS 合法性判断的 tileop-level IR。IR 中需要表达以下信息：

| Information | Source In IR | Use In VfSim |
| --- | --- | --- |
| group boundary | group/candidate attrs 或 wrapper op | 确定规划范围 |
| tileop identity | op name、stable op id、block order | 模板匹配和结果回写 |
| data dependency | SSA use-def | 重建 tileop DAG |
| inputs/outputs | op operands/results | 识别外部输入和内部中间值 |
| dtype | operand/result type | 支持 `tcvt` 等输入输出 dtype 不同的 op |
| shape | tile type、valid shape attrs、layout attrs | 推导 loop count 和 mask |
| template params | op attrs | 选择模板变体 |
| external output | liveness / yield / explicit attrs | 判断是否需要 materialize 到 group 外 |

输入 IR 的抽象形态：

```mlir
pto.vfsim_candidate_group @G0 {
  %a = pto.tadd ... {
    pto.vfsim.op_id = "op0"
  }
  %b = pto.tcvt ... {
    pto.vfsim.op_id = "op1",
    rmode = ...
  }
  pto.vfsim.yield %b
}
```

实际实现可以复用现有 tileop IR 和 attrs；`pto.vfsim_candidate_group` 只是协议层抽象，不要求第一版必须新增 op。

## Route A Output IR

VfSim 输出仍是 tileop-level IR，只是在 tileop 或 group 上写回 fusion plan attrs。

第一阶段主要 attrs：

| Attr | Meaning |
| --- | --- |
| `pto.fusion.group_id` | VF fusion group 标识 |
| `pto.fusion.order` | group 内 tileop 执行顺序 |
| `pto.fusion.unroll` | 当前 tileop 主 inner loop 的 unroll 值 |
| `pto.fusion.loop_id` | 相同 ID 的 tileop 融合到同一个 loop |
| `pto.fusion.inner_loop_expand` | 是否将 tileop 内部嵌套 inner loop 展开成单层 loop |

输出 IR 抽象形态：

```mlir
%a = pto.tadd ... {
  pto.fusion.group_id = 0,
  pto.fusion.order = 0,
  pto.fusion.loop_id = "L0",
  pto.fusion.unroll = 8,
  pto.fusion.inner_loop_expand = false
}

%b = pto.tcvt ... {
  pto.fusion.group_id = 0,
  pto.fusion.order = 1,
  pto.fusion.loop_id = "L0",
  pto.fusion.unroll = 8,
  pto.fusion.inner_loop_expand = false
}
```

`group_id` 表示 VF fusion 范围，`loop_id` 表示 loop fusion 范围。对于全 elementwise case，两者通常一致；对于 reduce、rowmax、嵌套 loop 等复杂 case，一个 VF group 内可以有多个 `loop_id`。

## Route B VF Analysis IR

手写 VF 路线的输入是开发者已经写好的 VF/micro-op IR。VfSim 不再决定哪些 tileop 融合，而是分析已有实现。

输入信息：

| Information | Use |
| --- | --- |
| VF micro-op sequence | 构建 VfSimProgram |
| loop structure | 计算 trip count 和 unroll 效果 |
| dtype / mask / layout | 选择 latency 参数 |
| memory/register use | 分析 forwarding、store/load、barrier |
| existing attrs | 识别开发者指定的 unroll、schedule、layout |

输出信息：

| Attr / Report | Meaning |
| --- | --- |
| `vfsim.estimated_cycles` | 预测 cycle |
| `vfsim.bottleneck` | 主要瓶颈，例如 latency、barrier、store/load |
| `vfsim.advice` | 优化建议 |
| `vfsim.suggested_unroll` | 建议 unroll 值 |
| `vfsim.candidate_cycles` | 候选策略耗时，用于 debug |

抽象链路：

```text
VF IR
  -> VfSim analyzer
  -> VF IR + vfsim diagnostic attrs
```

## VfSim Template Registry

VfSim 维护独立的性能模板库。模板语义与 PTOAS tileop lowering 保持一致，用于从 tileop IR 预测融合后的 VF micro-op 形式。

| TileOp | Template Output |
| --- | --- |
| `tadd` / `tadds` | elementwise add |
| `tmul` / `tmuls` | elementwise multiply |
| `tcvt` | conversion，输入输出 dtype 可不同 |
| `tdivs` | scalar/vector divide，可展开为 `vbr + vdiv` 语义 |
| `texp` | exp |
| row/reduce ops | 后续阶段补充复杂 loop 模板 |

模板输入来自 IR：

```text
op name + operand/result dtype + shape/valid shape/layout + op attrs
  -> template registry
  -> loop structure + micro-op sequence + external output materialization
```

## First Phase Scope

| Category | Scope |
| --- | --- |
| Input | legal tileop candidate IR |
| Supported ops | elementwise、scalar-elementwise、simple conversion |
| Examples | `tadd`、`tadds`、`tmul`、`tmuls`、`tcvt`、`tdivs`、`texp` |
| Planning policy | fuse-all within legal group |
| Unroll search | 扫描不超过 8 的循环次数因子 |
| Output | `group_id`、`order`、`loop_id`、`unroll`、`inner_loop_expand` attrs |
| Debug | dump 每个候选 unroll 的预测 cycle |

## API Shape

源码级接入下，API 可以直接接收 MLIR operation/module，不需要跨进程序列化。

```cpp
namespace vfsim {

struct PlannerOptions {
  bool dumpCandidates = false;
  unsigned maxUnroll = 8;
};

mlir::LogicalResult planTileFusionIR(mlir::Operation *candidateIR,
                                     const PlannerOptions &options);

mlir::LogicalResult analyzeVfIR(mlir::Operation *vfIR,
                                const PlannerOptions &options);

} // namespace vfsim
```

`planTileFusionIR` 在输入 tileop IR 上写回 fusion attrs。`analyzeVfIR` 在 VF IR 上写回 cost/advice attrs，或生成诊断报告。
