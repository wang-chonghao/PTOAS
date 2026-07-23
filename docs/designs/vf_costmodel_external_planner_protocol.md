# PTOAS + VfSim Unroll 接入设计

本文合并记录 PTOAS 前端 tile fusion 接入 VfSimulator cost model 的当前实现和
接口约定。当前方案是源码级接入：PTOAS 通过 git submodule 引入 VfSimulator，
在 FusionPlan pass 后调用 VfSim IR planner，由 VfSim 基于已选 fusion group
搜索 unroll，并将结果写回同一份 MLIR IR。

范围：

- 本文档覆盖 PTOAS 前端接入、IR 接口、VfSim planner 入口和当前 unroll 输出形式。
- 本文档不展开说明 VPTO 后端具体如何消费 unroll 属性。

## 整体流程

```text
PTOAS FusionPlan
  -> 由 PTOAS 策略生成合法 fusion group
  -> 给 tileop 标记 pto.fusion.group_id / pto.fusion.order
  -> 当使能 --enable-vfsim-fusion-planner 时调用 VfSim IR planner
  -> VfSim 从 MLIR IR 中读取 group/order 属性
  -> VfSim 将支持的 tileop lower 成 VfInfo micro-op program
  -> VfSim 扫描 unroll 候选值
  -> VfSim 写回 pto.fusion.row_unroll_factor / pto.fusion.col_unroll_factor
  -> 后续 PTOAS/VPTO pass 消费这些属性
```

## 接入形式

VfSimulator 作为 PTOAS 的 git submodule 引入，并以源码级方式参与 PTOAS 编译。

```text
PTOAS/
  3rdparty/
    VfSimulator/        # git submodule，由 PTOAS commit 固定版本
```

这种方式下，PTOAS 仓只记录 submodule commit，不直接接管 VfSimulator 的源码历史。
VfSimulator 的 Python 版本和 C++ native 版本仍由 VfSimulator 仓库维护。

控制选项分为两层：

| 选项 | 类型 | 作用 | 默认行为 |
|---|---|---|---|
| `-DPTO_ENABLE_VFSIM_COSTMODEL=ON` | CMake 编译期选项 | 决定 PTOAS 是否编译、链接 VfSim planner | `OFF` 时不包含 VfSim planner 代码 |
| `--enable-vfsim-fusion-planner` | ptoas 运行期选项 | 决定当前编译任务是否调用 VfSim planner | 不加时保持 PTOAS 原有 FusionPlan 行为 |

## 构建接入

| 文件 | 行号 | 作用 |
|---|---:|---|
| `CMakeLists.txt` | 240 | 引入 `cmake/VfSimulator.cmake`。 |
| `cmake/VfSimulator.cmake` | 9 | 定义 `PTO_ENABLE_VFSIM_COSTMODEL`。 |
| `cmake/VfSimulator.cmake` | 12 | 默认 VfSim 源码路径为 `3rdparty/VfSimulator`。 |
| `cmake/VfSimulator.cmake` | 31 | 嵌入 PTOAS 时关闭 VfSim 自身测试。 |
| `cmake/VfSimulator.cmake` | 33 | 打开 `VFSIM_ENABLE_MLIR_PLANNER`。 |
| `cmake/VfSimulator.cmake` | 35 | 将 `3rdparty/VfSimulator/native` 加入 PTOAS 构建。 |
| `lib/PTO/Transforms/CMakeLists.txt` | 153 | 将 `vfsim::native_core` 链接进 `PTOTransforms`。 |
| `lib/PTO/Transforms/CMakeLists.txt` | 162 | 将 `vfsim::ir_planner` 链接进 `PTOTransforms`。 |

启用构建：

```bash
-DPTO_ENABLE_VFSIM_COSTMODEL=ON
```

启用后，PTOAS 会编译并链接 VfSimulator 提供的两个 native target：

```text
vfsim::native_core
vfsim::ir_planner
```

## 用户侧开关

| 文件 | 行号 | 作用 |
|---|---:|---|
| `tools/ptoas/ptoas.cpp` | 534 | 定义 `--enable-vfsim-fusion-planner`。 |
| `include/PTO/Transforms/Passes.td` | 284 | 在 `pto-fusion-plan` 中加入 `enableVfSimFusionPlanner` 选项。 |
| `tools/ptoas/ptoas.cpp` | 3012 | 创建 `FusionPlanOptions`。 |
| `tools/ptoas/ptoas.cpp` | 3014 | 将命令行选项传入 `FusionPlanOptions`。 |
| `tools/ptoas/ptoas.cpp` | 3021 | 在 VPTO fusion 链路中插入 `FusionPlanPass(fusionPlanOpts)`。 |

运行 PTOAS 时启用 planner：

```bash
--enable-vfsim-fusion-planner
```

## 接入位置

VfSim 接口接在 PTOAS `FusionPlan` pass 末尾。

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

关键点：

- PTOAS 仍然负责 fusion 合法性判断和 fusion group 生成。
- 当前分支中，VfSim 不替换 PTOAS 的 group 生成逻辑。
- VfSim 接收已经带有 group/order 属性的 MLIR IR，并补充 unroll 规划属性。

## PTOAS FusionPlan 调用点

接入点在 `lib/PTO/Transforms/TileFusion/PTOFusionPlan.cpp`。

| 行号 | 逻辑 |
|---:|---|
| 25 | 当定义 `PTO_ENABLE_VFSIM_IR_PLANNER` 时 include `native/IRPlanner.h`。 |
| 531 | 定义 `runVfSimFusionPlanner(func)`。 |
| 533 | 创建默认 `vfsim::PlannerOptions`。 |
| 534 | 调用 `vfsim::planTileFusionIR(func.getOperation(), options)`。 |
| 573 | 创建 PTOAS 原有 FusionPlan 状态。 |
| 580 | 调用 `strategyEngine.planBlock(...)` 生成 fusion group。 |
| 582 | 调用 `assignStableGroupMetadata(...)` 写入 `group_id/order`。 |
| 585 | 只有 `enableVfSimFusionPlanner` 为 true 时才调用 VfSim planner。 |

## C++ API

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

| 项 | 约定 |
|---|---|
| `candidateIR` | FusionPlan 后的 MLIR operation；自动 tileop fusion 路线当前传入 `func::FuncOp`。 |
| 返回值 | `success()` 表示 planner 正常完成或没有可处理 group；`failure()` 表示接口级错误。 |
| 修改方式 | VfSim 在传入 IR 上原地写回 `pto.fusion.*` 属性。 |


## 输入 IR 形式

VfSim 接收 FusionPlan 后的 tileop-level MLIR IR。PTOAS 必须先在可融合 tileop 上写入：

| 属性 | 含义 |
|---|---|
| `pto.fusion.group_id` | PTOAS 已选择的 fusion group ID。 |
| `pto.fusion.order` | group 内 tileop 的执行顺序。 |

VfSim 从 IR 中读取的信息：

| 信息 | IR 来源 |
|---|---|
| group 边界 | `pto.fusion.group_id` |
| group 内顺序 | `pto.fusion.order` |
| tileop 类型 | MLIR op name，例如 `pto.tadd` |
| 数据依赖 | SSA operand / use-def |
| 输入输出 value | tileop operands |
| dtype | operand / output type |
| shape | tile buffer type |
| 模板参数 | tileop attrs |

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

## 输出 IR 形式

VfSim 输出仍然是同一份 tileop-level IR，通过写回 attrs 表示 planner 决策。

当前基础输出 attrs：

| 属性 | 含义 |
|---|---|
| `pto.fusion.row_unroll_factor` | 当 row loop 是实际最内层 loop 时使用的 unroll factor。 |
| `pto.fusion.col_unroll_factor` | 当 col loop 是实际最内层 loop 时使用的 unroll factor。 |

当前策略：

- 如果 col trip count 为 1，VfSim 按 VPTO 后端行为将其建模为展开后的 row loop，
  写入 `row_unroll_factor > 1`，`col_unroll_factor = 1`。
- 如果 col trip count 大于 1，VfSim 认为最内层是 col loop，
  写入 `row_unroll_factor = 1`，`col_unroll_factor > 1`。

## RegionGen 中的属性提升

VfSim planner 在 `FusionPlan` 层先把两个 unroll attrs 写到同一个 fusion group
内的每个 tileop 上。随后 `FusionRegionGen` 会把这些 tileop 包成
`pto.fusion_region`，并将同组一致的 unroll attrs 提升到和 `group_id` 同一层。

示意：

```mlir
%0 = pto.fusion_region {
  // region body: lowered/fused tile operations
} {
  pto.fusion.group_id = 0 : i64,
  pto.fusion.row_unroll_factor = 8 : i64,
  pto.fusion.col_unroll_factor = 1 : i64
} : !pto.tile_buf<vec, 96x64xf32>
```

也就是说，后续 pass 不需要再从每个 tileop 上读取 unroll 信息，而是直接从
`pto.fusion_region` 的属性上读取 `group_id`、`row_unroll_factor` 和
`col_unroll_factor`。

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
            pto.fusion.row_unroll_factor = 1 : i64,
            pto.fusion.col_unroll_factor = 2 : i64}

  pto.tmul ins(%c, %b : !pto.tile_buf<vec, 32x32xf32>,
                        !pto.tile_buf<vec, 32x32xf32>)
           outs(%c : !pto.tile_buf<vec, 32x32xf32>)
           {pto.fusion.group_id = 0 : i64,
            pto.fusion.order = 1 : i64,
            pto.fusion.row_unroll_factor = 1 : i64,
            pto.fusion.col_unroll_factor = 2 : i64}

  return
}
```

## VfSim IR Planner 入口

VfSim 代码位于 `3rdparty/VfSimulator/native`。

| 文件 | 行号 | 作用 |
|---|---:|---|
| `IRPlanner.h` | 20 | 定义 `PlannerOptions`，其中默认 `maxUnroll = 8`。 |
| `IRPlanner.h` | 25 | 对 PTOAS 暴露 `planTileFusionIR(mlir::Operation *)`。 |
| `IRPlanner.cpp` | 32 | 定义 planner 读写的属性名。 |
| `IRPlanner.cpp` | 90 | 通过 `runVfInfo` 仿真一个 unroll 候选。 |
| `IRPlanner.cpp` | 305 | 选择最优 unroll 候选。 |
| `IRPlanner.cpp` | 341 | `planTileFusionIR` 主入口。 |
| `IRPlanner.cpp` | 346 | 遍历 MLIR IR，按 `pto.fusion.group_id` 收集 tileop。 |
| `IRPlanner.cpp` | 381 | 将一个 planned group lower 成 VfSim program。 |
| `IRPlanner.cpp` | 386 | 搜索候选 unroll factor。 |
| `IRPlanner.cpp` | 408 | 将 row/col unroll 属性写回 tileop。 |

候选枚举规则：

- 搜索范围是 `1..maxUnroll`。
- 当前默认 `maxUnroll = 8`。
- 只考虑能够整除目标 loop trip count 的 factor。

## TileOp 模板 Lowering

当前源码级 planner 只覆盖 elementwise 风格的 tileop。

| 文件 | 行号 | 作用 |
|---|---:|---|
| `TileOpTemplates.h` | 25 | `PlannedTileOpIR`：保存 MLIR op 指针和稳定顺序。 |
| `TileOpTemplates.h` | 36 | `LoweredTileGroupProgram`：保存 lower 后的 `VfInfo`、trip count 和 unroll 维度。 |
| `TileOpTemplates.cpp` | 47 | 将支持的 tileop 映射到 micro-op。 |
| `TileOpTemplates.cpp` | 96 | 从 PTOAS tile type 中解析静态 shape 和 dtype。 |
| `TileOpTemplates.cpp` | 225 | `lowerTileGroupWithPerformanceTemplates` 主入口。 |
| `TileOpTemplates.cpp` | 301 | 推导 loop shape 和 unroll 作用维度。 |
| `TileOpTemplates.cpp` | 356 | 当 col trip count 为 1 时构造展开后的 row-loop 形式。 |
| `TileOpTemplates.cpp` | 364 | 当 col trip count 大于 1 时构造 row-loop + col-loop 形式。 |

当前支持的模板映射：

| TileOp | Micro-op |
|---|---|
| `tadd` | `VADD` |
| `tsub` | `VSUB` |
| `tmul` | `VMUL` |
| `tdiv` | `VDIV` |
| `tmax` | `VMAX` |
| `tmin` | `VMIN` |
| `tabs` | `VABS` |
| `texp` | `VEXP` |
| `tadds` | `VADDS` |
| `tsubs` | `VSUB`，标量通过 `VBR` 转成向量 |
| `tmuls` | `VMULS` |
| `tdivs` | `VDIV`，标量通过 `VBR` 转成向量 |
| `tmaxs` | `VMAXS` |
| `tmins` | `VMINS` |
| `tcvt` | `VCVT_F16_TO_F32` 或 `VCVT_F32_TO_F16` |

## 模板选择原则

VfSim 根据输入 IR 中的 tileop 信息选择 performance template。模板选择规则应与
VPTO 后端 tileop template 选择规则保持语义一致。

| 信息 | 用途 |
|---|---|
| op name | 选择 tileop 模板族，例如 `tadd`、`tmul`、`trowmax`。 |
| dtype | 选择 micro-op 参数和 vector lanes。 |
| shape / valid shape | 推导 row/col loop 结构和 mask。 |
| layout | 区分 row/col 展开方式。 |
| attrs | 选择模板变体。 |

VfSim 不重新调用 VPTO 后端展开 pass，而是在自身源码中维护同语义的 performance
template。template 可以包含 prelude、body、epilogue、单层 loop 或嵌套 loop。

## 属性输出示例

对于 96x64 的 GeLU 类 elementwise group，col trip count 为 1，所以 VfSim 写入：

```mlir
} {
  pto.fusion.col_unroll_factor = 1 : i64,
  pto.fusion.group_id = 0 : i64,
  pto.fusion.row_unroll_factor = 8 : i64
} : !pto.tile_buf<vec, 96x64xf32>
```

对于 32x128 的 elementwise group，col trip count 为 2，所以 VfSim 写入：

```mlir
} {
  pto.fusion.col_unroll_factor = 2 : i64,
  pto.fusion.group_id = 0 : i64,
  pto.fusion.row_unroll_factor = 1 : i64
} : !pto.tile_buf<vec, 32x128xf32>
```

## 手写 VF IR 扩展方向

除自动 tileop fusion 路线外，后续也可以支持开发者直接提供 VF/micro-op IR。
该路线仍可复用同一源码级接入形式，但输入不再是 tileop group，而是开发者已经写好的
VF IR。

```text
developer VF IR
  -> VfSim analyzer
  -> VF IR with cost/advice attrs or diagnostic report
```

该路线不决定 fusion group，而是评估已有 VF 实现，并给出预测和建议：

| 输出 | 含义 |
|---|---|
| `vfsim.estimated_cycles` | 预测 cycle。 |
| `vfsim.bottleneck` | 主要瓶颈。 |
| `vfsim.advice` | 优化建议。 |
| `vfsim.suggested_unroll` | 建议 unroll 值。 |
