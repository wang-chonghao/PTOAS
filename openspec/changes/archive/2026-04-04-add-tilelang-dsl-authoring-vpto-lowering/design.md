## Context

### 范围

本 design 只覆盖 TileLang DSL v1 的 lowering 主线：

- 输入：`tilelang-dsl/` 中定义的 v1 surface
- 输出：authoring-form VPTO IR（`func.func + arith/scf + pto.*`）
- 验证：通过 repo 当前 `ptoas --pto-backend=vpto` 的 authoring-stage legality contract

它不覆盖：

- matcher / registry
- implicit vecscope inference
- A5 text / LLVM emission
- advanced vector family

### 当前状态

当前仓库里与本 change 直接相关的事实有：

1. 真实 authoring-form VPTO carrier 是 dedicated `pto.vecscope/pto.strict_vecscope`

- `lib/PTO/Transforms/PTOValidateVPTOIR.cpp`
- `test/vpto_validate/vpto_validate_authoring_legacy_scope_negative.mlir`

都已经说明 legacy `scf.for {llvm.loop.aivector_scope}` 已被拒绝。

2. `docs/tilelang-dsl-guide.md` 的 surface 比 v1 计划范围更大

- 文档包含 matcher、implicit vecscope inference、advanced family、低层 DMA programming 等。
- 本 change 必须明确压缩到 elementwise 套餐，否则 lowering 无法在短期内闭合。

3. 现有实验 `python/pto/dialects/pto.py` parser 不可直接作为实现基线

- 它的 surface 接近 hand-written VPTO，并不等于 TileLang DSL guide。
- 用户要求本特性工作集中在 `tilelang-dsl/`，并明确不以现有其他 Python binding 实现为前提。

### 实现约束

- lowering 输出必须符合当前真实的 authoring-form VPTO legality contract。
- v1 只支持 `strict_vecscope` 显式 vector region，不做 implicit inference。
- support matrix 必须固定为 elementwise 套餐，避免“实现到哪算哪”。
- `verify()` 需要给出稳定行为；在无法访问 `ptoas` binary 时不能静默成功。

## Goals / Non-Goals

**Goals:**

- 定义并实现 TileLang DSL v1 的 fixed support matrix lowering。
- 让 `dma_load/dma_store`、`make_mask`、`vlds/vsts`、elementwise unary/binary/vector-scalar family、`for`/`if`、基础 sync 都有明确 VPTO lowering 目标。
- 保证输出 IR 能通过当前 `ptoas --pto-backend=vpto` authoring-stage legality。
- 为 `verify()` 定义一个明确、可落地、与 repo 当前验证路径一致的契约。

**Non-Goals:**

- 不在本 change 中扩展 matcher surface。
- 不支持 implicit vecscope inference。
- 不支持 compare/select/reduction/rearrangement/carry/UB-to-UB copy 等 advanced family。
- 不把 generated IR 直接送进 A5 text / LLVM emission 作为本 change 的完成标准。

## Decisions

### 1. v1 只接受显式 `strict_vecscope` 作为 Python surface 的 vector carrier

决策：

- 用户写 vector op 时，必须显式使用 `with pto.strict_vecscope(...) as (...):`
- frontend 直接 lower 为 dedicated `pto.strict_vecscope`
- v1 不做 implicit `pto.vecscope` inference

原因：

- 这与当前真实 authoring contract 一致。
- 显式 `strict_vecscope` 能明确 capture 边界、block 参数和类型来源，降低 v1 lowering 复杂度。

备选方案：

- 直接实现 implicit vecscope inference
  - 放弃原因：需要引入 CFG 分析、scope boundary 规则、与 scalar/control-flow 边界的交互，超出 v1。

### 2. v1 lowering support matrix 固定为 elementwise 套餐

决策：

- 支持：
  - 2D `TensorView`
  - 1D/2D `Tile`
  - `dma_load`
  - `dma_store`
  - `make_mask(dtype, PAT.*)` / `make_mask(dtype, remaining)`
  - `vlds` / `vsts`
  - unary：`vabs`, `vrelu`, `vexp`, `vnot`
  - binary：`vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vand`, `vor`, `vxor`
  - vector-scalar：`vadds`, `vsubs`, `vmuls`, `vdivs`, `vmaxs`, `vmins`
  - `for range(lb, ub, step)`
  - `if/else`
  - `set_flag`, `wait_flag`, `pipe_barrier`
- 其余 family 在 frontend reject

原因：

- 这套矩阵足以覆盖 `docs/tilelang-dsl-guide.md` 的代表性 elementwise kernel。
- 它避免把 advanced families 和 low-level authoring surface 混入 v1。

### 3. `dma_load/dma_store` 在 frontend 直接展开为 VPTO copy programming + copy op

决策：

- `dma_load` lower 到必要的 `set_loop*_stride_outtoub` / `set_loop_size_outtoub` + `copy_gm_to_ubuf`
- `dma_store` lower 到必要的 `set_loop*_stride_ubtoout` / `set_loop_size_ubtoout` + `copy_ubuf_to_gm`
- 参数由 TensorView slice、Tile shape/config、padding mode 推导

原因：

- 当前 authoring-form VPTO 已经以这些 op 作为合法 surface。
- 对 v1 来说，直接在 frontend materialize 这些 op 比再引入一层 TileLang-specific DMA IR 更简单。

### 4. shape profile 固定为“静态 physical Tile + 动态 view/bound”

决策：

- Tile physical shape 必须是静态编译期常量
- TensorView 的 shape、slice 边界、loop bound 可以包含 runtime value
- `valid_shape` 仅支持：
  - 静态值
  - 由 TensorView partition 直接推导

原因：

- 这能覆盖 guide 中动态 TensorView / tail handling 的主要场景。
- 同时避免在 v1 引入 fully-dynamic tile allocation 语义。

### 5. `verify()` 通过 `ptoas` subprocess 复用 repo 当前 legality contract

决策：

- `descriptor.verify()` 以临时文件或等价 stdin 方式调用 repo 中可用的 `ptoas` binary
- 命令路径按以下顺序解析：
  - 显式传入或环境变量覆盖
  - `build/tools/ptoas/ptoas`
- 验证命令以 `--pto-backend=vpto --emit-vpto` 或等价 authoring-stage legality 路径运行
- binary 缺失或不可执行时，返回结构化 `verifier unavailable` 结果，而不是静默成功

原因：

- 当前 repo 没有直接暴露 custom VPTO legality pass 的 Python binding。
- 复用 `ptoas` binary 是最直接且与现有回归一致的验证方式。

备选方案：

- 在 Python 中直接调用 verifier pass
  - 放弃原因：当前不存在稳定的 Python 入口，短期内实现成本高于收益。

## Risks / Trade-offs

- [Risk] 直接展开 `dma_load/dma_store` 会把 DMA parameter inference 复杂度推到 frontend  
  Mitigation：v1 只支持固定 profile 的 TensorView slice / Tile layout，超出矩阵的场景一律 reject。

- [Risk] `verify()` 依赖 `ptoas` binary，环境不完整时可能影响开发体验  
  Mitigation：定义清晰的 binary 查找顺序和结构化 unavailable 结果，避免模糊失败。

- [Risk] support matrix 过窄可能与 guide 的完整愿景存在落差  
  Mitigation：在 change3 中单独扩展 matcher 和 advanced surface，并在 v1 diagnostics 中明确延期边界。

- [Risk] 显式 `strict_vecscope` 可能让首版示例看起来较啰嗦  
  Mitigation：把 implicit vecscope inference 明确放入 follow-up change，不在 v1 做半成品推断。
