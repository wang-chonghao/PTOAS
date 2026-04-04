## 1. OpenSpec 契约落定

- [x] 1.1 新增 `openspec/changes/add-tilelang-dsl-authoring-vpto-lowering/specs/tilelang-dsl-vpto-lowering/spec.md`，固定 v1 lowering support matrix、dynamic-bound 轮廓和 `verify()` 契约。
- [x] 1.2 在 `proposal.md` 和 `design.md` 中明确 v1 只支持显式 `strict_vecscope`，不做 implicit vecscope inference。

## 2. Frontend lowering 骨架

- [x] 2.1 在 `tilelang-dsl/python/` 中建立独立的 AST/语义/lowering pipeline，把 core-foundation 的 descriptor 接到 authoring-form VPTO builder。
- [x] 2.2 实现 `TensorView`、`Tile`、标量、loop bound 和 `strict_vecscope` block argument 的类型绑定与 SSA 环境管理。
- [x] 2.3 让 lowering 输出稳定的 `func.func + arith/scf + pto.*` authoring-form VPTO module。

## 3. Elementwise support matrix

- [x] 3.1 实现 `dma_load` / `dma_store` 的 TensorView slice 到 DMA programming + copy op lowering。
- [x] 3.2 实现 `make_mask`、`vlds`、`vsts` 以及 v1 unary/binary/vector-scalar family 的 lowering。
- [x] 3.3 实现 `for range(lb, ub, step)`、`if/else`、`set_flag`、`wait_flag`、`pipe_barrier` 的 lowering。
- [x] 3.4 对 support matrix 外的 family 保持 fail-fast reject，不允许 silent fallback。

## 4. Dynamic-bound 与合法性验证

- [x] 4.1 实现“静态 physical Tile + 动态 TensorView slice/loop bound”的 shape profile，拒绝 dynamic physical tile shape。
- [x] 4.2 实现 tail `make_mask(dtype, remaining)` 的 typed-mask lowering，确保输出满足当前 VPTO legality contract。
- [x] 4.3 实现 `descriptor.verify()`，通过 `ptoas` binary 运行与 `--pto-backend=vpto` 一致的 authoring-stage legality 验证，并对 binary 缺失返回结构化 unavailable 结果。

## 5. 测试、样例与文档

- [x] 5.1 在 `tilelang-dsl/tests/` 增加 elementwise kernel 的 positive regression，覆盖 `dma_load/store`、`strict_vecscope`、typed-mask、dynamic loop bound。
- [x] 5.2 增加 negative regression，覆盖 vector op 出 scope、unsupported family、非法 shape profile、verifier unavailable。
- [x] 5.3 在 `tilelang-dsl/examples/` 和 `tilelang-dsl/docs/` 提供与 guide 对齐的 v1 示例，并明确记录 support matrix 与延期 feature。
- [x] 5.4 运行并记录最小验证命令，确认生成的 IR 能通过 `build/tools/ptoas/ptoas --pto-backend=vpto` 的 authoring-stage legality 路径。
