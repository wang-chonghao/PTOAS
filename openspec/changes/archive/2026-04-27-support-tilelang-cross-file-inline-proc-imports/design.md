## Context

### 范围

本 design 覆盖 TileLang DSL 模板库中的跨文件 `@pto.inline_proc` helper 复用：

- 模板目录扫描和 Python import 搜索路径。
- `from shared_helper import helper` 形式导入的 `InlineProcDescriptor` 收集。
- 导入 helper 的 frontend AST 构建、semantic 分析和 lowering。
- 导入 helper 的 backend-inline 收敛验证。

它不覆盖：

- high precision divide 的算法实现。
- 普通 Python 函数的静态分析或自动内联。
- 模块限定调用 `shared_helper.helper(...)`。
- 任意第三方 Python 包作为 DSL helper 源。

### 当前状态

当前仓库已有以下事实：

1. `@pto.inline_proc` 可以注册 source-visible helper，并由 `vkernel` 收集。
2. `vkernel` 当前会收集当前模块内可见的 `InlineProcDescriptor`，包括从其它模块导入后绑定到当前模块 globals 的 descriptor。
3. `frontend_ast` 已能把 reachable inline proc 构造成 `FrontendInlineProcNode`，并处理 helper 互调、递归检测与 capture 校验。
4. `lowering` 已能把 inline helper 合并到输出 module，并标记 `pto.tilelang.inline_proc`。
5. `expand_helper` 负责扫描 `--template-dir` 下的 `.py` 文件，但模板 import 搜索路径和跨文件 helper 契约尚未正式冻结。

因此，本 change 优先补齐模板 import 环境、诊断和测试契约，而不是新建一套 helper lowering 机制。

## Goals / Non-Goals

**Goals:**

- 让模板文件可以直接导入同目录共享 helper 文件中的 `@pto.inline_proc`。
- 让导入 helper 与当前文件 helper 走同一套 frontend/semantic/lowering/backend-inline 主线。
- 保持 DSL 静态可分析边界：只有 `@pto.inline_proc` helper 被当作可调用 DSL helper。
- 用轻量 helper 建立回归测试，不引入 high precision 算法复杂度。

**Non-Goals:**

- 不支持普通 Python 函数作为 DSL helper。
- 不支持用户在 DSL body 中调用 arbitrary external function。
- 不支持 `**kwargs` unpacking、varargs、递归等现有 `inline_proc` 禁止行为。
- 不要求公共 helper 文件本身包含 `@pto.vkernel`。

## Decisions

### 1. 首期支持直接导入后的简单名字调用

决策：

- 支持如下形式：

```python
from shared_helper import shared_op

@pto.vkernel(target="a5", op="pto.some_op")
def template(src: pto.Tile, dst: pto.Tile):
    ...
    out = shared_op(lhs, rhs, mask)
```

- 暂不把 `import shared_helper; shared_helper.shared_op(...)` 纳入首期 contract。

原因：

- 当前 frontend call model 以简单名字匹配 inline proc，直接导入可以复用既有实现。
- 模块限定调用需要扩展 `ast.Attribute` 解析、helper identity 和同名冲突处理，适合作为后续增量。

### 2. 共享 helper 必须由 `@pto.inline_proc` 装饰

决策：

- 公共文件中的可复用 helper MUST 是 source-visible top-level `@pto.inline_proc`。
- 普通 Python 函数即便被 import，也 MUST 继续按 unsupported external call 处理。

原因：

- DSL lowering 需要读取 helper AST 并生成受控 IR。
- 放开普通 Python 函数会破坏当前受限 Python 子集和静态分析边界。

### 3. `expand_helper` 提供模板目录级 import context

决策：

- 在扫描 `--template-dir` 时，helper 进程 SHOULD 临时把以下路径加入 import 搜索：
  - `template_dir`
  - `template_dir.parent`
- 这样同时支持：
  - `from shared_helper import helper`
  - `from TileOps.shared_helper import helper`

原因：

- issue 示例使用裸模块名导入。
- 当前 `lib/TileOps` 下已有 `__init__.py`，保留 package import 形式有利于兼容现有布局。

### 4. 导入 helper 与本文件 helper 使用同一 lowering contract

决策：

- frontend MUST 将导入 helper 纳入当前 kernel 的 reachable inline proc 集合。
- semantic MUST 对导入 helper 按参数类型和静态值生成必要 specializations。
- lowering MUST emit private `func.func ... attributes { pto.tilelang.inline_proc }`。
- 后续 backend-inline MUST 能消除 helper call 和 helper function。

原因：

- 用户只需要关心 helper 能否复用，不应观察到“本文件 helper”和“跨文件 helper”在 IR contract 上有差异。

### 5. 同名 helper 冲突应 fail fast

决策：

- 如果当前模板模块通过多个来源暴露了同名 `@pto.inline_proc` helper，frontend SHOULD 报出明确诊断，而不是静默选择一个。

原因：

- 当前 call surface 以简单名字调用，静默覆盖会让模板行为依赖 import 顺序。
- 在没有引入 qualified helper identity 前，重复名字最好被显式拒绝。

## 测试策略

- Python/frontend 单测：
  - 临时模板目录中创建 `shared_helper.py` 和 `import_user_template.py`。
  - `shared_helper.py` 定义简单 `@pto.inline_proc`，例如返回 `pto.vadd(lhs, rhs, mask)` 或简单 pass-through。
  - 模板通过 `from shared_helper import helper` 调用。
  - 断言 descriptor 能被 `expand_helper` 找到，`build_frontend_kernel_node()` 包含导入 helper。
  - 断言 `mlir_text()` 中出现 `pto.tilelang.inline_proc` 和对应 helper call。
- Helper 互调单测：
  - 共享文件中 `helper_a()` 调用 `helper_b()`。
  - 模板只 import `helper_a()`。
  - 断言 `helper_b()` 也被纳入 reachable helper materialization。
- 负向单测：
  - import 普通 Python 函数后调用，仍按 arbitrary external call 报错。
  - 导入 helper 递归或互递归，仍报 recursive inline_proc。
  - 导入 helper 非法捕获动态值，仍报 implicit capture。
  - 同名 helper 冲突报明确错误。
- lit / end-to-end 回归：
  - 新增一个小型 TileOp/template case，只验证跨文件 helper 被调用和内联。
  - FileCheck `pto-inline-libcall` 后不残留 inline helper call/function。

## Risks / Trade-offs

- [Risk] 把 `template_dir` 加入 `sys.path` 可能与标准库或第三方模块同名冲突  
  Mitigation：只在 `expand_helper` 扫描模板期间使用 scoped import context，并在文档中建议共享 helper 文件避免使用标准库同名文件名。

- [Risk] 简单名字调用可能遇到同名 helper 冲突  
  Mitigation：首期 fail fast；后续若需要再支持 qualified call。

- [Risk] 测试若直接使用 high precision div 会把算法风险混入 import 能力验证  
  Mitigation：本 change 明确只使用轻量 helper 验证跨文件引用，不实现 high precision 算法。

## Migration Plan

1. 先冻结 OpenSpec contract，明确跨文件 helper 的支持形式和非目标。
2. 在 `expand_helper` 中补模板目录 scoped import context。
3. 补 frontend/diagnostic 对导入 helper、冲突和负向场景的测试。
4. 补一个端到端回归，验证导入 helper materialize 并被 backend inline 消除。
5. 更新 TileLang DSL 用户文档，说明共享 helper 写法和限制。

## Open Questions

- 后续是否需要支持 `import shared_helper; shared_helper.helper(...)` qualified call。  
  本 change 暂不承诺，优先完成直接导入形式。

- 是否需要为共享 helper 建立专门的 `lib/TileOps/shared/` 子目录。  
  本 change 不强制目录结构，只要求 template-dir import contract 稳定。
