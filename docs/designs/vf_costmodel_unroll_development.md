# VF CostModel Unroll 开发方案

本文记录 VF costmodel 生成 unroll 决策后，PTOAS 后端如何传递并消费该决策。

## 1. 目标

当前已经完成：

```text
pto-fusion-plan
  -> costmodel mode 下只保留 fusion legality
  -> 形成合法 fusion group
  -> 扫描 unroll=1..8 中能整除 trip count 的候选
  -> 调用 C++ VfSimulator 预测 latency
  -> 选择最优 unroll
  -> 给 tileop 写入 pto.fusion.unroll
```

下一步目标：

```text
tileop 上的 pto.fusion.unroll
  -> 传递到 tileop expand 后生成的 unroll target scf.for
  -> PTOLowLevelLoopFusion 消费 loop 上的 unroll 标签
  -> 生成带 unroll 的融合 loop
```

## 2. 编译链路

costmodel unroll 链路：

```text
PTO tile IR
  -> pto-fusion-plan
       生成 tileop 级 fusion metadata
  -> pto-op-scheduling
       根据 group/order 调整 tileop 为连续 span
  -> pto-fusion-region-gen
       将连续 span 包装成 pto.fusion_region
  -> pto-expand-tile-op
       tileop 替换为 TileLang 模板 func.call
       将 tileop fusion metadata 复制到 func.call
  -> pto-inline-lib-call
       inline TileLang 模板函数
  -> pto-fold-tile-buf-intrinsics(shape-only)
       折叠 shape/valid shape 信息
  -> pto-propagate-fusion-loop-attrs
       将 inline 后的 fusion metadata 传播到模板生成的 unroll target scf.for
  -> pto-low-level-loop-fusion
       读取 loop 上的 fusion metadata
       完成 stage loop fusion
       根据 pto.fusion.loop_unroll 复制 loop body
  -> 后续 predicate/load-store elision、flatten、VPTO/EmitC lowering
```

默认编译路径不生成 unroll，也不触发 unroll 行为。只有显式启用 costmodel planner 时，才会生成并消费 unroll 标签。

### 2.1 TileOp 到 Loop 的标签生命周期

`tileop` 在 `pto-expand-tile-op` 中不会一步直接变成最终 micro-op，而是先替换为一个临时的 TileLang 模板 `func.call`。随后 `pto-inline-lib-call` 将该 call inline 成模板体 IR，才出现 `scf.for` 和 `pto.vlds/vadd/vsts/...` 等低层结构。

| 阶段 | IR 里是什么 | 标签在哪里 |
| --- | --- | --- |
| `pto-fusion-plan` 后 | 原始 `tileop`，例如 `pto.tadd` | `tileop` 上有 `pto.fusion.group_id/order/unroll` |
| `pto-expand-tile-op` 后 | `tileop` 被 erase，原位置生成 `func.call @__pto_tilelang_*` | 标签从 `tileop` 复制到 `func.call` |
| `pto-inline-lib-call` 后 | `func.call` 消失，TileLang 模板体被 inline 成 `scf.for + micro-op` | 标签从 `func.call` 复制到 clone 出来的模板 IR |
| `pto-fold-tile-buf-intrinsics(shape-only)` 后 | shape/valid shape 信息被折叠 | 标签继续保留 |
| `pto-propagate-fusion-loop-attrs` 后 | 目标 `scf.for` 被标记为 fusion loop | loop 上有 `pto.fusion.loop_unroll` 等标签 |
| `pto-low-level-loop-fusion` 后 | 相邻 stage loop 被融合，并按 unroll 展开 body | loop 上的 unroll 标签被消费 |

`func.call` 只是中间 IR 的临时承载点，用来避免 `tileop` 被 erase 后丢失 fusion metadata。它不会出现在最终 C++ 输出里。

示例 dump 命令：

```bash
build-vf-costmodel/tools/ptoas/ptoas \
  /tmp/ptoas_a5_e2e_probe/tadd3_32x32_e2e.pto \
  --pto-backend=vpto \
  --pto-level=level2 \
  --pto-arch=a5 \
  --enable-op-fusion \
  --use-vfsim-fusion-planner \
  --mlir-print-ir-after=pto-expand-tile-op \
  -o /dev/null
```

该阶段可看到：

```mlir
func.call @__pto_tilelang_a5_tadd_...(...)
  {pto.fusion.unroll = 4 : i64}
```

继续 dump `pto-inline-lib-call` 后，`func.call` 消失，标签已经转移到 inline 后的模板 IR：

```bash
build-vf-costmodel/tools/ptoas/ptoas \
  /tmp/ptoas_a5_e2e_probe/tadd3_32x32_e2e.pto \
  --pto-backend=vpto \
  --pto-level=level2 \
  --pto-arch=a5 \
  --enable-op-fusion \
  --use-vfsim-fusion-planner \
  --mlir-print-ir-after=pto-inline-libcall \
  -o /dev/null
```

```mlir
scf.for ... {
  ...
  pto.vlds ...
  pto.vadd ...
  pto.vsts ...
} {pto.fusion.unroll = 4 : i64}
```

## 3. Metadata

### 3.1 TileOp 级标签

`pto-fusion-plan` 生成：

| 标签 | 含义 |
| --- | --- |
| `pto.fusion.group_id` | fusion group id |
| `pto.fusion.order` | tileop 在 group 内的顺序 |
| `pto.fusion.unroll` | costmodel 选择的 unroll factor |

第一阶段 `pto.fusion.unroll` 使用整数：

```mlir
pto.fusion.unroll = 4 : i64
```

后续可扩展为数组，表达一个 tileop 展开出多个并列 loop nest 时，每个 loop nest 的 inner loop unroll：

```mlir
pto.fusion.unroll = array<i64: 2, 2, 4>
```

### 3.2 Loop 级标签

新增传播逻辑后，unroll target `scf.for` 上携带：

| 标签 | 含义 |
| --- | --- |
| `pto.fusion.group_id` | 来源 fusion group id |
| `pto.fusion.order` | 来源 tileop 在 group 内的顺序 |
| `pto.fusion.loop_index` | 该 loop nest 在 tileop 模板展开结果中的序号 |
| `pto.fusion.loop_unroll` | 该 loop 的 unroll factor |

第一阶段：

```mlir
pto.fusion.loop_index = 0 : i64
pto.fusion.loop_unroll = 4 : i64
```

## 4. 新增和修改点

### 4.1 FusionPlan

文件：

```text
lib/PTO/Transforms/TileFusion/PTOFusionPlan.cpp
```

当前职责：

- costmodel mode 下生成 `pto.fusion.unroll`。
- 第一阶段只选择 `1..8` 中能整除 trip count 的 unroll。
- `--dump-vfsim-unroll-test` 打印每个候选的预测 latency。

### 4.2 ExpandTileOp

文件：

```text
lib/PTO/Transforms/ExpandTileOp.cpp
```

需要新增：

- 原 tileop 被替换为 `func.call` 前，将 tileop 上的 fusion metadata 复制到 call。
- 需要复制的标签：

```text
pto.fusion.group_id
pto.fusion.order
pto.fusion.unroll
```

功能：

```text
tileop metadata
  -> template func.call metadata
```

### 4.3 PropagateFusionLoopAttrs

新增 pass：

```text
pto-propagate-fusion-loop-attrs
```

建议文件：

```text
lib/PTO/Transforms/TileFusion/PTOPropagateFusionLoopAttrs.cpp
```

插入位置：

```text
pto-inline-lib-call
pto-fold-tile-buf-intrinsics(shape-only)
pto-propagate-fusion-loop-attrs
pto-low-level-loop-fusion
```

功能：

- 找到 inline 后仍携带 fusion metadata 的模板展开范围。
- 在该范围内识别 loop nest。
- 将 tileop/call 级 unroll plan 写到每个 loop nest 的 unroll 目标 loop。
- 第一版 unroll 目标 loop 选择规则：从外到内扫描单链 loop nest，选择最内层的无 loop-carried values 的 `scf.for`。

第一阶段传播规则：

```text
pto.fusion.unroll = N
  -> 第 0 个 loop nest 的 unroll target scf.for:
       pto.fusion.loop_index = 0
       pto.fusion.loop_unroll = N
```

后续数组形式：

```text
pto.fusion.unroll = [u0, u1, u2]
  -> 第 0 个 loop nest unroll target: loop_unroll = u0
  -> 第 1 个 loop nest unroll target: loop_unroll = u1
  -> 第 2 个 loop nest unroll target: loop_unroll = u2
```

### 4.4 LowLevelLoopFusion

文件：

```text
lib/PTO/Transforms/TileFusion/PTOLowLevelLoopFusion.cpp
```

需要新增：

- `analyzeStage()` 读取 loop 上的：

```text
pto.fusion.group_id
pto.fusion.order
pto.fusion.loop_index
pto.fusion.loop_unroll
```

- `fuseStageRun()` 校验同一 stage run 的 loop metadata：

```text
group_id 一致
loop_index 一致
loop_unroll 一致
loop shape 一致
trip_count % loop_unroll == 0
```

- 生成 fused loop 时，根据 `loop_unroll` 复制 leaf ops。

示例：

```text
for { A }
for { B }
for { C }
```

当 `loop_unroll = 2` 时融合为：

```text
for {
  A(i)
  A(i + 1)
  B(i)
  B(i + 1)
  C(i)
  C(i + 1)
}
```

第一阶段约束：

- 只消费能整除 trip count 的 unroll。
- 不生成 tail loop。
- 不在带 loop-carried values 的 loop 本身上做 unroll。
- 不满足条件时跳过 unroll，保留普通 loop fusion 或跳过该 stage run。

### 4.5 Pass 注册和 Pipeline

需要修改：

```text
include/PTO/Transforms/Passes.td
include/PTO/Transforms/Passes.h
lib/PTO/Transforms/CMakeLists.txt
tools/ptoas/ptoas.cpp
```

在 VPTO tileop lowering 链路中插入：

```text
pto-propagate-fusion-loop-attrs
```

建议只在 costmodel planner 模式下启用 unroll 行为：

```text
--use-vfsim-fusion-planner
```

如果没有 `pto.fusion.unroll`，新增 pass 和 `PTOLowLevelLoopFusion` 都应 no-op。

## 5. 功能验证

### 5.1 IR 验证

新增或更新 lit case：

```text
test/lit/tile_fusion/fusion_plan_tcvt_vfsim.pto
test/lit/tile_fusion/fusion_loop_unroll_tadd3.pto
test/lit/tile_fusion/fusion_loop_unroll_tadd3_vfsim.pto
```

检查：

- `pto-fusion-plan` 后 tileop 有 `pto.fusion.unroll`。
- `pto-expand-tile-op + inline + propagate` 后 unroll target `scf.for` 有：

```text
pto.fusion.loop_unroll
pto.fusion.loop_index
```

- `pto-low-level-loop-fusion` 后 fused loop body 出现按 unroll 复制的 micro-op。

### 5.2 Lowering 输出验证

生成 C++：

```bash
ptoas tadd3_32x32_e2e.pto \
  --pto-level=level2 \
  --pto-arch=a5 \
  --enable-op-fusion \
  --use-vfsim-fusion-planner \
  --dump-vfsim-unroll-test \
  --enable-insert-sync \
  -o tadd3_unroll.a5.cpp
```

检查：

- `--dump-vfsim-unroll-test` 的 selected unroll 与 loop 展开结果一致。
- C++ 中对应 loop body 体现 unroll 后的重复 micro-op。
