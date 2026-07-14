# VF CostModel Planner Interface Design

本文档定义 PTOAS 与 VfSimulator CostModel 的接口设计。PTOAS 负责生成
tileop fusion group 并提供 MLIR IR；VfSim 基于该 IR 生成 costmodel 优化策略，
并将策略结果写回同一份 IR。

## 1. 接入形式

VfSimulator 作为 PTOAS 的 git submodule 引入，并以源码级方式参与 PTOAS 编译。

```text
PTOAS/
  3rdparty/
    vfsimulator/        # git submodule, fixed by PTOAS commit
```

PTOAS 侧通过 CMake 控制是否启用该接口：

```cmake
-DPTO_ENABLE_VFSIM_COSTMODEL=ON
```

控制选项分为编译期和运行期两层：

| 选项 | 类型 | 作用 | 默认行为 |
| --- | --- | --- | --- |
| `-DPTO_ENABLE_VFSIM_COSTMODEL=ON` | CMake 编译期选项 | 决定 PTOAS 是否编译、链接 VfSim planner 能力 | `OFF` 时 PTOAS 不包含 VfSim planner 代码 |
| `--enable-vfsim-fusion-planner` | ptoas 运行期选项 | 决定当前编译任务是否调用 VfSim planner | 不加时保持 PTOAS 原有 FusionPlan 行为 |

启用后，PTOAS 编译并链接 VfSimulator 提供的 planner target：

```text
vfsim::native_core
vfsim::ir_planner
```

VfSimulator 源码仍由 VfSimulator 仓库维护。PTOAS 主仓只记录 submodule commit，
避免直接接管 VfSim 源码历史。

## 2. 接入位置

接口接在 PTOAS `FusionPlan` pass 末尾。

```text
PreFusionAnalysis
  -> FusionPlan
       - PTOAS 生成合法 fusion group
       - PTOAS 写入 pto.fusion.group_id / pto.fusion.order
       - PTOAS 调用 VfSim planner
  -> OpScheduling / FusionRegionGen / downstream backend
```

PTOAS 侧调用形式：

```cpp
vfsim::PlannerOptions options;
return vfsim::planTileFusionIR(func.getOperation(), options);
```

VfSim planner 在运行时由 driver 选项打开：

```bash
--enable-vfsim-fusion-planner
```

该选项默认关闭。关闭时 PTOAS 保持原有 FusionPlan 行为。

## 3. C++ API

VfSimulator 向 PTOAS 暴露源码级 C++ API：

```cpp
namespace vfsim {

struct PlannerOptions {
  bool dumpCandidates = false;
  unsigned maxUnroll = 8;
};

mlir::LogicalResult planTileFusionIR(
    mlir::Operation *candidateIR,
    const PlannerOptions &options = {});

} // namespace vfsim
```

接口约定：

| Item | Contract |
| --- | --- |
| `candidateIR` | FusionPlan 后的 MLIR operation；自动 tileop fusion 路线使用 `func::FuncOp` |
| Return value | `success()` 表示 planner 正常完成或无可处理 group；`failure()` 表示接口级错误 |
| Mutation | VfSim 允许在传入 IR 上写回 `pto.fusion.*` attrs |
| Serialization | 不使用 JSON/YAML，不跨进程序列化 |

## 4. 输入 IR 形式

VfSim 接收的是 FusionPlan 后的 tileop-level IR。PTOAS 必须在可融合 tileop 上写入：

| Attr | Meaning |
| --- | --- |
| `pto.fusion.group_id` | PTOAS 已选择的 fusion group ID |
| `pto.fusion.order` | group 内 tileop 的执行顺序 |

VfSim 从 IR 中读取：

| Information | IR Source |
| --- | --- |
| group boundary | `pto.fusion.group_id` |
| group order | `pto.fusion.order` |
| tileop kind | MLIR op name, e.g. `pto.tadd` |
| data dependency | SSA operands / use-def |
| input/output values | tileop operands |
| dtype | operand / output type |
| shape | tile buffer type and shape-related attrs |
| template parameters | tileop attrs |

输入示例：

```mlir
func.func @kernel(%a: !pto.tile_buf<vec, 32x32xf32>,
                  %b: !pto.tile_buf<vec, 32x32xf32>,
                  %c: !pto.tile_buf<vec, 32x32xf32>) {
  pto.tadd ins(%a, %b : !pto.tile_buf<vec, 32x32xf32>,
                        !pto.tile_buf<vec, 32x32xf32>)
           outs(%c : !pto.tile_buf<vec, 32x32xf32>)
           {pto.fusion.group_id = 0 : i64,
            pto.fusion.order = 0 : i64}

  pto.tmul ins(%c, %b : !pto.tile_buf<vec, 32x32xf32>,
                        !pto.tile_buf<vec, 32x32xf32>)
           outs(%c : !pto.tile_buf<vec, 32x32xf32>)
           {pto.fusion.group_id = 0 : i64,
            pto.fusion.order = 1 : i64}

  return
}
```

上例中，PTOAS 已经选择 `tadd -> tmul` 为同一个 fusion group。VfSim 不重新判断
这两个 tileop 是否可以融合，只基于该 group 生成优化策略。

## 5. 输出 IR 形式

VfSim 输出仍然是同一份 tileop-level IR，通过写回 attrs 表示 planner 决策。

基础输出 attrs：

| Attr | Meaning |
| --- | --- |
| `pto.fusion.unroll` | VfSim 为该 fusion group 选择的 inner loop unroll 值 |

输出示例：

```mlir
func.func @kernel(%a: !pto.tile_buf<vec, 32x32xf32>,
                  %b: !pto.tile_buf<vec, 32x32xf32>,
                  %c: !pto.tile_buf<vec, 32x32xf32>) {
  pto.tadd ins(%a, %b : !pto.tile_buf<vec, 32x32xf32>,
                        !pto.tile_buf<vec, 32x32xf32>)
           outs(%c : !pto.tile_buf<vec, 32x32xf32>)
           {pto.fusion.group_id = 0 : i64,
            pto.fusion.order = 0 : i64,
            pto.fusion.unroll = 2 : i64}

  pto.tmul ins(%c, %b : !pto.tile_buf<vec, 32x32xf32>,
                        !pto.tile_buf<vec, 32x32xf32>)
           outs(%c : !pto.tile_buf<vec, 32x32xf32>)
           {pto.fusion.group_id = 0 : i64,
            pto.fusion.order = 1 : i64,
            pto.fusion.unroll = 2 : i64}

  return
}
```

## 6. 手写 VF IR 接口

除自动 tileop fusion 外，接口也支持开发者直接提供 VF/micro-op IR。
该路线复用同一源码级接入形式，但输入不再是 tileop group，而是开发者已经写好的 VF IR。

```text
developer VF IR
  -> VfSim analyzer
  -> VF IR with cost/advice attrs or diagnostic report
```

Route B 的输出不决定 fusion group，而是给出已有 VF 实现的评估结果：

| Attr / Report | Meaning |
| --- | --- |
| `vfsim.estimated_cycles` | 预测 cycle |
| `vfsim.bottleneck` | 主要瓶颈 |
| `vfsim.advice` | 优化建议 |
| `vfsim.suggested_unroll` | 建议 unroll 值 |
