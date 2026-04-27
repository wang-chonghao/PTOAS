## 1. OpenSpec 契约落定

- [x] 1.1 完成 `proposal.md`，明确本 change 只验证跨文件引用 helper 生效，不实现 high precision 算法。
- [x] 1.2 完成 `design.md`，固定直接导入 `@pto.inline_proc` helper 的设计边界。
- [x] 1.3 完成 `specs/tilelang-dsl-surface/spec.md`，定义跨文件共享 helper authoring surface。
- [x] 1.4 完成 `specs/tilelang-dsl-diagnostics/spec.md`，定义导入 helper 的负向诊断。
- [x] 1.5 完成 `specs/tilelang-dsl-vpto-lowering/spec.md`，定义导入 helper 的 materialization 和 backend-inline 契约。

## 2. Frontend / template import

- [x] 2.1 在 `tilelang-dsl/python/tilelang_dsl/expand_helper.py` 中为模板扫描增加 scoped import context，至少包含 `template_dir` 与 `template_dir.parent`。
- [x] 2.2 确保 `from shared_helper import helper` 导入的 `InlineProcDescriptor` 会被当前 `vkernel` descriptor 收集。
- [x] 2.3 确保共享 helper 文件不需要包含 `@pto.vkernel` 也能被模板导入使用。
- [x] 2.4 对同名导入 helper 冲突增加 fail-fast 诊断，避免静默依赖 import 顺序。

## 3. Lowering / backend path

- [x] 3.1 确保跨文件导入 helper 与本文件 helper 一样生成 `FrontendInlineProcNode`。
- [x] 3.2 确保导入 helper 在 `mlir_text()` 中 materialize 为 private `pto.tilelang.inline_proc` helper function。
- [x] 3.3 确保现有 `pto-inline-libcall` 能消除导入 helper 的 call 和 private helper function。

## 4. 回归测试与文档

- [x] 4.1 在 TileLang DSL Python 单测中新增“共享文件定义 helper，模板直接导入并调用”的正向测试。
- [x] 4.2 新增共享 helper 互调测试：模板只导入入口 helper，被入口 helper 调用的同文件 helper 也能被 materialize。
- [x] 4.3 新增负向测试：普通 Python 函数 import 后调用仍被拒绝。
- [x] 4.4 新增负向测试：导入 helper 的递归、非法 capture、同名冲突均给出明确诊断。
- [x] 4.5 新增或更新 lit 回归，验证导入 helper 经 backend-inline 后不残留 helper call/function。
- [x] 4.6 更新 TileLang DSL 用户文档，说明共享 `@pto.inline_proc` helper 的推荐写法和限制。

## 5. 验证

- [x] 5.1 执行针对跨文件 helper import 的 TileLang DSL 单测。
- [x] 5.2 执行覆盖 helper + backend-inline 收敛路径的定向回归。
- [x] 5.3 执行 `openspec validate support-tilelang-cross-file-inline-proc-imports --type change --strict --json --no-interactive`。
