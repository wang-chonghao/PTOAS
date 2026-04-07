## Context

### 范围

本 design 覆盖两个相互配合的能力：

1. 新增 `tilelang-dsl-template-slots`
2. 修改 `tilelang-dsl-kernel-matcher`

目标是让多个同 family 的具体 PTO op 共享一份 TileLang DSL kernel body，同时保持 frontend 仍然是静态、受限、deterministic 的 DSL，而不是变成任意 Python 解释执行器。

### 当前状态

当前 `tilelang-dsl-kernel-matcher` 已提供：

- `KernelRegistry`
- `pto.select_kernel(target, op, operand_types, context_attrs, registry=None)`
- 多 signature `dtypes`
- `constraints`
- `priority`

但 descriptor 仍以单个 concrete `op` 为中心。  
与此同时，kernel body 中的 vector op surface 也仍要求直接书写真实 `pto.vadd`、`pto.vsub`、`pto.vmul`、`pto.vdiv` 等调用。

这使得 `tadd/tsub/tmul/tdiv` 这类共享同一 loop/mask/load-store 骨架、只在核心计算 op 上不同的 kernel 很难复用实现。

### 关键约束

- `tilelang-dsl/` 继续作为源码、测试、样例和局部文档的 source of truth。
- `pto.select_kernel(...)` 的外部查询形态继续保持 concrete `op` 查询，不新增 family 级公共查询接口。
- frontend 继续只接受受限 Python 子集和固定 DSL call surface；不能靠支持任意 dict/callable 来实现模板分发。
- 新能力必须复用现有 semantic/type-check/lowering 路径，最终仍收敛到当前 authoring-form VPTO legality contract。
- 多-op / template 相关行为必须 deterministic，不依赖注册顺序、定义顺序或运行时值。

### 设计拆分

本 change 明确拆成两层：

1. matcher 层

- 负责让一个 descriptor 匹配多个 concrete PTO op，并在 `select_kernel(...)` 后绑定唯一 `selected_op`

2. template slot 层

- 负责让 kernel body 使用统一的 `pto.tpl("slot", ...)` 占位调用
- 在 frontend 编译阶段按 `selected_op` 把占位调用静态替换成真实 `pto.*` op

这两层的拆分保证：

- descriptor 选择仍由 matcher capability 管理
- kernel body 模板化能力独立成一个新的 authoring capability
- semantic/type-check/lowering 不需要接入“动态模板调用”的新运行时概念

## Goals / Non-Goals

**Goals:**

- 允许 `@pto.vkernel` 用 `ops=[...]` 描述一个 descriptor 覆盖多个 concrete PTO op。
- 提供统一模板入口 `pto.tpl("slot", ...)`，支持在 kernel body 的任意合法 DSL 位置表达“按 concrete op 替换”的核心计算。
- 在 frontend 阶段完成模板替换，让后续 semantic/type-check/lowering 继续只面对真实 `pto.*` 调用。
- 保证多-op descriptor 在未绑定 concrete `op` 时不会提前 materialize。
- 保持现有单-op kernel、现有显式真实 `pto.*` 调用写法、现有 selector API 与现有 legality contract 全部兼容。

**Non-Goals:**

- 不支持在 kernel body 中执行 Python dict lookup、callable value、lambda、闭包或其他 higher-order call。
- 不把 `pto.tpl(...)` 设计成运行时 dispatch；它不是运行时 helper，而是编译期 placeholder。
- 不把 family 变成 `select_kernel(...)` 的新公共查询轴。
- 不在本 change 中一次性覆盖所有 family 的模板化 authoring；首版只提供通用 slot 机制和明确的合法性边界。

## Decisions

### 1. 新增独立 capability `tilelang-dsl-template-slots`，而不是把模板语义塞进 `advanced` 或 `matcher`

决策：

- `pto.tpl("slot", ...)` 与 `templates={...}` 作为独立 capability 定义
- `tilelang-dsl-kernel-matcher` 只负责 multi-op descriptor 与 concrete `selected_op` 绑定
- `tilelang-dsl-advanced-surface` 不承载本次模板语义

原因：

- 模板槽位本质上是 authoring sugar，不等于 advanced-family lowering。
- 它既可以服务 `advanced=True` 的 kernel，也可以服务显式 `strict_vecscope` 的非-advanced kernel。
- 把它独立成 capability 更容易保持语义聚焦，避免把 advanced surface 继续做胖。

备选方案：

- 方案 A：把 `pto.tpl(...)` 放进 `tilelang-dsl-advanced-surface`
  - 未采用，因为模板槽位不天然依赖 advanced mode。
- 方案 B：直接修改 matcher spec，不新增 capability
  - 未采用，因为 matcher 只负责“选哪个 descriptor”，不负责“kernel body 如何模板化表达”。

### 2. 模板映射放在 decorator 元数据中，而不是开放 kernel body 里的 Python dict/callable

决策：

- `@pto.vkernel` 新增 `templates={...}` 静态元数据
- kernel body 只允许写 `pto.tpl("slot", ...)`
- 模板映射值只接受真实 `pto.*` op 名字符串，不接受 Python callable

原因：

- 当前 frontend 明确拒绝 arbitrary external call，也没有把 Python dict 作为正式 DSL 表达式收进 AST/semantic 契约。
- 如果允许 `ops["core"](...)` 或 `table[name](...)` 这类写法，就必须给 DSL 增加 dict、callable value、索引后调用、甚至作用域捕获等一整套新语义，复杂度和风险都明显超出这次 change 目标。
- 把映射放在 decorator 元数据里，可以把模板分发收敛为 compile-time static metadata，保持 deterministic。

备选方案：

- 方案 A：在 kernel body 支持 Python dict literal + indexed callable
  - 未采用，因为会显著扩张 DSL 的 Python 子集和 frontend 复杂度。
- 方案 B：用 family-specific placeholder op，如 `pto.vbinary`
  - 未采用，因为会引入一批专用 placeholder API，扩展到其他 family 时容易持续膨胀。

### 3. 公开 surface 采用单一通用入口 `pto.tpl("slot", ...)`

决策：

- 新公共 API 只有一个模板入口：`pto.tpl("slot", *args)`
- `slot` 必须是字符串字面量
- `slot` 对应的真实 op 映射由 `templates={...}` 给出

原因：

- 单一入口能避免 placeholder API 数量随 family 数量膨胀。
- `slot` 命名由 kernel author 控制，能表达“core”“cmp”“select”“postprocess”等局部语义，而不是把 DSL 绑死到某一组预定义 family 名称。
- 这仍然允许一份 kernel body 在多个位置复用同一 slot，或在同一 kernel 里声明多个 slot。

备选方案：

- 方案 A：`pto.tpl.core(...)`
  - 未采用，因为会额外引入 attribute-based placeholder namespace，收益不高。
- 方案 B：`pto.tpl["core"](...)`
  - 未采用，因为需要 DSL 支持 subscripted callable surface。

### 4. matcher 扩展为 `op` / `ops`，并在 selection 之后绑定 `selected_op`

决策：

- `@pto.vkernel` 接受：
  - `op="tadd"`，保持现状
  - `ops=["tadd", "tsub", "tmul", "tdiv"]`，作为新增能力
- `op` 与 `ops` 必须互斥，且至少提供其一
- descriptor 内部统一保存 `match_ops`
- `pto.select_kernel(...)` 保持现有公共签名不变，仍使用 concrete `op` 查询
- selector 命中 multi-op descriptor 时，必须把 query `op` 绑定成唯一 `selected_op`

原因：

- 对外仍按 concrete `op` 查询，能兼容现有上层集成和调用路径。
- `selected_op` 把模板替换需要的 concrete 上下文显式传入后续 materialization 阶段，避免“后面再猜当前 op 是什么”。
- `op/ops` 互斥可以避免 descriptor 元数据出现双重来源和优先级歧义。

备选方案：

- 方案 A：把 `select_kernel(...)` 改成 family 查询
  - 未采用，因为会改动公共 API 语义，也与用户当前实际关心的 concrete PTO op 查询不一致。
- 方案 B：`op` 和 `ops` 同时允许出现，由实现隐式 merge
  - 未采用，因为容易制造含糊的匹配集合和隐藏优先级。

### 5. 模板替换发生在 frontend AST 构建阶段，先替换后做 semantic/type-check

决策：

- `build_frontend_kernel_node(...)` 在把 Python AST 投影成 frontend AST 时识别 `pto.tpl("slot", ...)`
- 若 descriptor 已绑定 `selected_op`，则直接把该调用重写成真实 `FrontendCallExpr(namespace="pto", name="<resolved-op>", args=...)`
- semantic analyzer 和 lowering renderer 继续只消费真实 `pto.*` op

原因：

- 这样 semantic/type-check/lowering 不需要知道“模板调用”这个新概念，只需要沿用已有的真实 op 检查逻辑。
- 错误可以更早暴露在 frontend，而不是等到后续阶段才发现某个模板槽位无法解析。
- 替换点足够早，能让模板调用出现在循环、分支、inferred vecscope、strict_vecscope 内等任意合法位置，而不改变后续编译形态。

备选方案：

- 方案 A：在 semantic 阶段再解析 `pto.tpl`
  - 未采用，因为 semantic 需要额外承载 unresolved template call，增加中间状态复杂度。
- 方案 B：在 lowering 阶段才解析
  - 未采用，因为会把本该 fail-fast 的模板错误延后到更晚阶段。

### 6. 多-op descriptor 的 materialization gate 与 polymorphic dtypes 一致

决策：

- 若 descriptor 同时满足以下任一条件，则不得直接 `mlir_text()` / `verify()` / `emit()`：
  - 还未绑定 concrete `dtype_signature`
  - 还未绑定 concrete `selected_op`
- 只有经过 `select_kernel(...)` 绑定后，才能进入 `specialize()` 与 materialization 流程

原因：

- template slot 替换依赖 `selected_op`；未绑定 concrete `op` 时无法确定最终真实 `pto.*` 调用。
- 这与当前 polymorphic `dtypes` 的 gate 规则一致，用户心智模型也更统一。

备选方案：

- 方案 A：对 multi-op descriptor 默认取 `ops[0]` 作为 materialization op
  - 未采用，因为会引入不透明的隐式默认值，破坏 deterministic 语义。

### 7. 模板槽位必须做静态合法性约束，而不是“能替换就替换”

决策：

- 注册时必须校验：
  - `templates` 是静态 mapping
  - slot 名是非空字符串
  - 映射 key 是 descriptor 可匹配的 concrete op 子集
  - 映射 value 是受支持的真实 `pto.*` op 名字符串
- frontend 替换时必须校验：
  - `slot` 参数是字符串字面量
  - 当前 `selected_op` 在该 slot 下存在映射
  - 模板展开后的真实 op 属于当前 DSL 支持 surface

原因：

- 模板能力本质上是“静态展开”，因此错误必须在 frontend 明确、可重复地暴露。
- 如果允许模糊或延迟解析，后续会把用户错误混淆成底层 type/lowering 错误。

备选方案：

- 方案 A：只在模板实际被执行到时懒解析
  - 未采用，因为 kernel DSL 不是运行时解释器，不应存在执行路径依赖的语义。

## 测试策略

- matcher 正例：
  - `ops=[...]` 的 descriptor 可被不同 concrete `op` 查询命中
  - single-op 与 multi-op descriptor 并存时仍按 `priority` / tie error 决策
- materialization 正例/负例：
  - multi-op descriptor 在未绑定 `selected_op` 前拒绝 materialization
  - 绑定后可继续 `specialize()` / `mlir_text()`
- template slot 正例：
  - 同一份 kernel body 在 `tadd/tsub/tmul/tdiv` 下分别展开成正确真实 op
  - `pto.tpl("slot", ...)` 可位于 loop、if、strict_vecscope、inferred vecscope 中
- template slot 负例：
  - 未定义 slot
  - slot 不覆盖当前 `selected_op`
  - `slot` 不是字符串字面量
  - 映射到未知或不受支持的 `pto.*` op
- 文档/样例：
  - 增加一个共享 `tadd/tsub/tmul/tdiv` 的 template-slot 示例
  - 在 guide 中明确“为什么不支持 kernel body Python dict/callable”

## Risks / Trade-offs

- [Risk] `op` / `ops` 双入口会增加 decorator 心智负担  
  Mitigation：要求两者互斥，并保持 `op=` 旧写法完全兼容。

- [Risk] `pto.tpl("slot", ...)` 可能被误解为运行时 helper  
  Mitigation：spec、文档和诊断都明确它是 compile-time placeholder，不是 runtime dispatch。

- [Risk] 如果模板映射值过于自由，容易把不兼容调用形态混到一个 slot 里  
  Mitigation：要求映射值必须是当前已支持的真实 `pto.*` op 名，并在 frontend 做静态合法性检查。

- [Risk] 多-op descriptor 若允许隐式默认 concrete `op`，会导致 materialization 结果不稳定  
  Mitigation：未绑定 `selected_op` 时一律拒绝 materialization。

## Migration Plan

- 该 change 为增量能力，不需要仓库级迁移或兼容层清理。
- 现有单-op kernel 与显式真实 `pto.*` 调用保持原样可用。
- 新功能按以下顺序接入：
  1. 扩展 descriptor/matcher 元数据为 `op` / `ops` + `selected_op`
  2. 引入 `templates={...}` 与 `pto.tpl("slot", ...)`
  3. 在 frontend AST 构建阶段接入模板替换
  4. 补齐 unittest、example、guide 与 migration 文档
- 若实现中发现模板替换无法在 frontend 阶段稳定落地，回退策略是保留 `ops=[...]` matcher 扩展，而暂不开放 `pto.tpl(...)` public surface。

## Open Questions

- 当前没有必须阻塞实现的开放问题。
- 若后续需要让模板槽位覆盖 compare/select 或 vector-scalar family，应沿用同一 `pto.tpl("slot", ...)` 机制扩展映射表，而不是新增另一套 placeholder API。
