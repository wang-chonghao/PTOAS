# Proposal: 实现 TileLang DSL v1 到 authoring-form VPTO IR 的 lowering

## 概述

在 `add-tilelang-dsl-core-foundation` 固定 package、surface 和 diagnostics 之后，本 change 负责把 v1 核心子集真正 lower 到 authoring-form VPTO IR。  
目标不是直接产出 A5 text/LLVM 发射结果，而是在 `tilelang-dsl/` 下建立一条稳定的 `TileLang DSL -> func/arith/scf/pto` authoring pipeline，并要求生成结果能够通过当前 `ptoas --pto-backend=vpto` 的 authoring-stage legality contract。

## 背景与动机

当前仓库里虽然已经有：

- `docs/tilelang-dsl-guide.md` 对高层 DSL 的表述
- `docs/vpto-spec.md` / `docs/vpto-verify.md` 对 VPTO IR 的表述
- 一个实验性的 `python/pto/dialects/pto.py` parser

但还没有一条与 guide 对齐、且以 `tilelang-dsl/` 为承载目录的正式 lowering 路径。  
如果没有这条路径，TileLang DSL 只能停留在文档层，后续 sample、regression 和 capability 都无法真正收敛。

同时，v1 必须先压缩到固定 support matrix：

- 只做 `a5`
- 只做 elementwise 套餐
- 只做显式 `strict_vecscope`
- 只做到 authoring-form VPTO

否则会把 matcher、implicit vecscope inference、advanced family 一起引入，导致 v1 无法闭合。

## 目标

- 在 `tilelang-dsl/` 下实现 TileLang DSL v1 到 authoring-form VPTO IR 的 lowering。
- 固定 v1 lowering 仅支持 elementwise 套餐：
  - 2D `TensorView`
  - 1D/2D `Tile`
  - `dma_load` / `dma_store`
  - `make_mask`
  - `vlds` / `vsts`
  - 常用 unary / binary / vector-scalar family
  - `for` / `if`
  - `set_flag` / `wait_flag` / `pipe_barrier`
- 明确 vector surface 必须位于显式 `strict_vecscope` 内，v1 不做 implicit vecscope inference。
- 提供 `verify()` 契约，使 generated IR 能按当前 repo 的 VPTO authoring-stage legality 路径进行验证。

## 非目标

- 不在本 change 中实现 kernel matcher、`constraints`、`priority`、`Any*`、`TypeVar`。
- 不在本 change 中实现 implicit vecscope inference。
- 不在本 change 中扩展到 compare/select/reduction/rearrangement 等 advanced family。
- 不在本 change 中直接产出 A5 text/LLVM emission 结果。
- 不在本 change 中把实现回填到现有 `python/pto/dialects/pto.py` 实验 parser。

## 变更内容

- 新增 `tilelang-dsl-vpto-lowering` capability，定义 v1 fixed support matrix 的 lowering 行为。
- 固定 `strict_vecscope` 是 v1 唯一合法的 vector-surface Python carrier；vector op 出现在显式 scope 外必须由 frontend 拒绝。
- 固定 `dma_load` / `dma_store` 到 `copy_gm_to_ubuf` / `copy_ubuf_to_gm` 以及必需 DMA programming op 的 lowering 规则。
- 固定 `verify()` 通过 `ptoas --pto-backend=vpto` authoring-stage legality 契约校验 generated module；当环境缺少 `ptoas` binary 时返回结构化 “verifier unavailable” 结果。

## Capabilities

### New Capabilities

- `tilelang-dsl-vpto-lowering`: 定义 TileLang DSL v1 从高层 Tile/TensorView surface 到 authoring-form VPTO IR 的 lowering 目标、support matrix、dynamic-bound 轮廓与验证接口。

### Modified Capabilities

- 无

## 预期结果

- `tilelang-dsl/` 下的 v1 kernel 能产出稳定的 authoring-form VPTO IR 文本/模块。
- 生成的 IR 使用当前真实 contract：显式 `pto.strict_vecscope`、typed mask、authoring-form buffer-like address。
- v1 support matrix 外的 surface 在 frontend 直接 reject，不让未实现的 family 混入 lowering。
- `verify()` 能复用 repo 当前 `ptoas` legality 路径，对 generated IR 给出 pass/fail 结果。

## 成功标准

- 新增 `openspec/changes/add-tilelang-dsl-authoring-vpto-lowering/`，包含 proposal、design、tasks。
- 新增 `specs/tilelang-dsl-vpto-lowering/spec.md`。
- proposal/design/tasks 明确写清：
  - v1 只支持显式 `strict_vecscope`
  - v1 support matrix 的 family 列表
  - `dma_load/dma_store` 的 lowering 目标
  - `verify()` 必须走与 `ptoas --pto-backend=vpto` 一致的 authoring-stage legality contract

## 影响

- 受影响目录：
  - `tilelang-dsl/python/`
  - `tilelang-dsl/tests/`
  - `tilelang-dsl/examples/`
  - `tilelang-dsl/docs/`
- 受影响 public API：
  - `descriptor.mlir_text()`
  - `descriptor.mlir_module()`
  - `descriptor.verify()`
- 受影响验证路径：
  - 生成物必须兼容 `ptoas --pto-backend=vpto` 的 authoring-stage legality
