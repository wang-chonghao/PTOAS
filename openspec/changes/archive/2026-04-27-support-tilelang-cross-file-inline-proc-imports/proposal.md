# Proposal: 支持 TileLang DSL 模板跨文件复用 `@pto.inline_proc`

## 概述

当前 TileLang DSL 模板已经支持在单个模板文件内定义并调用 `@pto.inline_proc` helper。  
Issue #190 提出的新需求是：多个模板文件需要复用同一份可被 DSL 前端分析和 lowering 的公共 helper，例如未来的 high-precision divide 逻辑应该只维护一份，而不是在 `trowexpanddiv`、`tcolexpanddiv`、`trsqrt` 等模板里重复实现。

本 change 只落定并验证“跨文件引用函数生效”这条能力：

- 公共文件中定义 `@pto.inline_proc` helper。
- 模板文件通过 Python import 引入该 helper。
- DSL frontend / semantic / lowering 能把导入的 helper 纳入当前 `vkernel` 实例化。
- 输出 MLIR 中出现 private inline-proc helper，并继续通过既有 backend-inline 主线收敛。

本 change 不实现 high precision divide 算法本身；测试可以使用简单 helper 验证跨文件接线。

## 背景与动机

DIV/EXP 类模板后续会有公共近似或高精度算法需求。如果每个 TileOp 模板各自复制一份 helper，会带来：

1. 模板文件膨胀，review 成本变高。
2. 同一算法多处维护，修复容易遗漏。
3. DSL 现有 `inline_proc` 能力被限制在单文件使用，无法成为模板库级公共机制。

当前实现中已经有可复用基础：

- `@pto.inline_proc` 可以描述可分析 helper。
- `vkernel` descriptor 会收集当前模块可见的 inline proc。
- frontend AST、semantic 和 lowering 已能 materialize inline helper。
- 后端已有 `pto.tilelang.inline_proc` 的 inline 主线。

缺口主要在模板加载和 import contract：`expand_helper` 扫描模板目录时，需要让模板文件可以稳定 import 同目录或模板包内的共享 helper，并把导入的 `InlineProcDescriptor` 纳入当前 kernel。

## 目标

- 支持模板文件直接导入同一 template-dir 中定义的 `@pto.inline_proc` helper。
- 支持公共 helper 内部继续调用同文件中其它 `@pto.inline_proc` helper。
- 保持跨文件 helper 与单文件 helper 相同的 frontend diagnostics、semantic lowering 和 backend-inline 行为。
- 增加针对跨文件 import 的单测和定向 end-to-end 回归。
- 在文档中说明共享 helper 的支持形式和限制。

## 非目标

- 不实现 high precision divide / exp / rsqrt 等具体算法。
- 不支持任意普通 Python 函数作为 DSL helper。
- 不承诺 `import shared; shared.helper(...)` 形式；首期以 `from shared import helper` 的直接导入调用为正式支持形式。
- 不设计新的 public helper namespace。
- 不改变现有 `@pto.inline_proc` 的语法限制、递归限制、capture 规则或 backend-inline pass。

## What Changes

- `tilelang-dsl-surface`：
  - 明确 TileLang DSL 模板 MAY 从同一模板目录或模板包导入 `@pto.inline_proc` helper。
  - 明确首期支持直接导入后的简单名字调用，例如 `from shared_div import high_precision_div` 后调用 `high_precision_div(...)`。
  - 明确普通 Python 函数 import 不属于 DSL 可分析 helper surface。
- `tilelang-dsl-diagnostics`：
  - 明确跨文件导入 helper 仍使用现有 `inline_proc` 诊断规则。
  - 明确普通外部函数、不可见 helper、递归 helper、非法 capture、重复 helper 名等错误需要 fail fast。
- `tilelang-dsl-vpto-lowering`：
  - 明确跨文件导入的 helper MUST materialize 为 private inline-proc helper function。
  - 明确 helper call MUST 继续通过现有 backend-inline 主线消除。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `tilelang-dsl-surface`: 新增模板跨文件直接导入 `@pto.inline_proc` helper 的 public authoring contract。
- `tilelang-dsl-diagnostics`: 新增跨文件 helper 导入场景下的 fail-fast 诊断契约。
- `tilelang-dsl-vpto-lowering`: 新增跨文件导入 helper 的 frontend-to-MLIR materialization 与 backend-inline 收敛契约。

## 预期结果

- 模板库可以把公共可分析算法写在共享 Python 文件中。
- 多个 TileOp 模板可以通过 import 复用该 helper，而不复制 helper body。
- `.mlir_text()` 阶段可以看到来自共享文件的 private inline-proc helper。
- 后续 `pto-inline-libcall` 仍能消除 helper 边界，不把共享 helper 作为最终 IR contract 暴露。

## 成功标准

- 新增 `openspec/changes/support-tilelang-cross-file-inline-proc-imports/`，包含 `proposal.md`、`design.md`、`tasks.md`。
- 新增 spec delta：
  - `specs/tilelang-dsl-surface/spec.md`
  - `specs/tilelang-dsl-diagnostics/spec.md`
  - `specs/tilelang-dsl-vpto-lowering/spec.md`
- 测试中使用轻量 helper 验证：
  - `shared_helper.py` 定义 `@pto.inline_proc`
  - `*_template.py` 通过 `from shared_helper import helper` 调用
  - frontend 能收集 helper
  - lowering 输出 inline-proc helper
  - backend inline 后不残留 helper call/function
- 不要求新增或完成 high precision divide 实现。

## Impact

- 受影响目录：
  - `tilelang-dsl/python/tilelang_dsl/`
  - `tilelang-dsl/tests/`
  - `tilelang-dsl/docs/user_guide/`
  - `lib/TileOps/`
  - `test/basic/`
  - `openspec/changes/support-tilelang-cross-file-inline-proc-imports/`
- 受影响 public authoring behavior：
  - `from shared_helper import helper` 形式的跨文件 `@pto.inline_proc` 复用。
- 受影响 lowering 行为：
  - 导入 helper 与当前文件 helper 一样生成 inline-proc helper function，并走既有 backend inline。
