# tilelang-dsl-vpto-lowering Specification

## Purpose
TBD - created by archiving change support-tilelang-cross-file-inline-proc-imports. Update Purpose after archive.
## Requirements
### Requirement: imported inline-proc helpers MUST materialize like local inline-proc helpers

跨文件导入的 `@pto.inline_proc` helper 一旦被 `@pto.vkernel` body 调用，frontend、semantic 和 lowering MUST 将其视为当前 kernel 的 reachable inline helper。  
`.mlir_text()` materialization MUST 为该 helper 生成 private helper function，并带有 `pto.tilelang.inline_proc` marker。  
该 helper 的 call MUST 使用与本文件 `@pto.inline_proc` helper 相同的 call/specialization contract。

#### Scenario: imported helper appears in materialized MLIR

- **WHEN** 模板调用从共享文件直接导入的 `@pto.inline_proc` helper
- **THEN** `.mlir_text()` 输出 MUST 包含对该 helper specialization 的 `func.call`
- **AND** 输出 MUST 包含对应的 private `func.func`
- **AND** helper function MUST 标记 `pto.tilelang.inline_proc`

#### Scenario: imported helper can call another helper from the same shared file

- **WHEN** 模板只导入并调用共享文件中的入口 helper
- **AND** 该入口 helper 调用同共享文件中的另一个 `@pto.inline_proc`
- **THEN** lowering MUST materialize reachable helper call graph 中需要的 helper specializations
- **AND** helper 互调 MUST 继续遵守现有递归检测规则

### Requirement: imported inline-proc helpers MUST be compatible with backend-inline cleanup

跨文件导入 helper 生成的 `pto.tilelang.inline_proc` private function MUST 与现有 backend-inline 主线兼容。  
经过 inline cleanup pipeline 后，导入 helper 的 `func.call` 和 private helper function MUST 被消除，除非该 pipeline 已按既有规则明确跳过对应 helper。

#### Scenario: backend inline removes imported helper boundary

- **WHEN** materialized MLIR 中包含跨文件导入 helper 的 private inline-proc function
- **AND** 该 MLIR 经过现有 `pto-inline-libcall` 或等价 backend-inline cleanup
- **THEN** helper call SHOULD 被内联
- **AND** private helper function SHOULD 不再残留为最终用户可见边界

### Requirement: cross-file helper import MUST not require algorithm-specific lowering

跨文件 helper import 的 lowering 验证 MUST 只依赖 generic inline-proc materialization 行为。  
实现和测试 MAY 使用简单 arithmetic helper 验证该能力，不得要求 high precision divide 或其它复杂算法先完成。

#### Scenario: simple imported helper proves the lowering path

- **WHEN** 共享 helper 只执行简单 vector operation 或 pass-through operation
- **THEN** frontend/lowering/backend-inline 测试仍 MUST 覆盖跨文件 helper 的完整接线
- **AND** 复杂 high precision 算法缺失 MUST NOT 阻塞该能力验收

