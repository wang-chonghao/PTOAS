# TileLang DSL Diagnostics Specification

## Purpose
TBD - created by archiving change add-tilelang-dsl-core-foundation. Update Purpose after archive.
## Requirements
### Requirement: v1 MUST fail fast on unsupported matcher and decorator features

TileLang DSL v1 frontend 对以下 surface MUST fail-fast，而不是静默忽略或拖到 lowering 阶段：

- 多个 `dtypes` signature
- `constraints`
- `priority`
- `AnyFloat` / `AnyInt` / `AnyType` / `AnyMask`
- `TypeVar`

#### Scenario: unsupported matcher feature is rejected at decorator parse time

- **WHEN** 用户在 v1 kernel decorator 中写入 `constraints`、`priority`、多 signature `dtypes`、`Any*` 或 `TypeVar`
- **THEN** frontend MUST 直接报错
- **AND** 诊断 MUST 明确指出该 feature 不属于 v1 范围
- **AND** 诊断 SHOULD 指向 follow-up change `extend-tilelang-dsl-matcher-and-advanced-surface`，而不是伪装成底层 type error

### Requirement: v1 MUST reject unsupported Python syntax and unsupported DSL calls before IR generation

TileLang DSL v1 frontend MUST 只接受受限 Python 子集。  
`while`、list/dict/set comprehension、arbitrary external function call、未注册 DSL op、以及其他超出 v1 surface 的 Python 结构 MUST 在 frontend 被拒绝。

#### Scenario: unsupported Python construct is rejected before lowering

- **WHEN** kernel body 使用 `while`、comprehension、任意非 `pto.*` function call 或未纳入 v1 support matrix 的 DSL call
- **THEN** frontend MUST 在生成任何 VPTO IR 之前报错
- **AND** 诊断 MUST 指明违规的 Python construct 或 DSL call 名称

### Requirement: Tile specialization and shape-profile errors MUST be diagnosed in the frontend

TileLang DSL v1 frontend MUST 把以下错误归类为前端错误：

- bare `Tile` 参数未完成 specialization
- Tile physical shape 不是静态编译期常量
- Tile profile 与 v1 支持的 rank / memory-space 约束不匹配

#### Scenario: unspecialized or dynamically-shaped tile fails before materialization

- **WHEN** kernel 含 bare `Tile` 参数但调用方未完成 `specialize()`，或 specialization 试图给出 dynamic physical tile shape
- **THEN** frontend MUST 在 `mlir_text()` / `mlir_module()` / `verify()` 之前直接报错
- **AND** MUST NOT 继续尝试生成不完整的 authoring-form VPTO IR

### Requirement: frontend diagnostics MUST include source location and semantic cause

TileLang DSL v1 的 frontend diagnostics MUST 包含 DSL 源位置和语义原因。  
错误消息 MUST 能区分：

- decorator surface 不支持
- Python 语法子集不支持
- 参数定型失败
- Tile specialization/profile 非法

#### Scenario: user sees actionable diagnostic with source location

- **WHEN** frontend 因 unsupported feature、unsupported syntax、type binding failure 或 specialization error 拒绝一个 kernel
- **THEN** 诊断 MUST 至少包含 DSL 源文件位置、行列号或等价的 source span
- **AND** MUST 明确指出失败原因属于哪一层 frontend 语义，而不是只给出底层 verifier 或 parser 的通用报错

### Requirement: selector diagnostics MUST identify the failing stage and failing constraint for each reported candidate

当调用方启用 `select_kernel(...)` 的 report/metadata 模式时，TileLang DSL diagnostics MUST 为每个候选明确指出其失败或胜出的阶段。  
对于 constraint 阶段失败的候选，诊断 MUST 至少包含：

- 失败的 constraint 索引
- 若可解析则包含 callable 名称或 `qualname`
- 区分“predicate 返回 `False`”与“constraint 执行抛异常”
- 可读的失败原因文本

对于 `dtype` 不匹配、priority 落败和 materialization 失败，diagnostics 也 MUST 使用不同的 kind/status 表达，而不是统一折叠成同一种通用错误。  
selector diagnostics MUST 让调用方能仅凭 report 判定候选究竟挂在 `dtype`、`constraints`、`priority` 还是 materialization。

#### Scenario: false-returning constraint is reported with index and callable identity

- **WHEN** 某个候选在 constraint evaluation 中命中第 `N` 个 constraint，且该 callable 返回 `False`
- **THEN** selection diagnostics MUST 报告该候选失败在第 `N` 个 constraint
- **AND** diagnostics MUST 在可解析时包含该 constraint 的 callable 名称或 `qualname`

#### Scenario: raising constraint is reported as a constraint error

- **WHEN** 某个 constraint 在执行时抛出异常
- **THEN** selection diagnostics MUST 将该候选标记为 `constraint_error`
- **AND** diagnostics MUST 包含异常类型与消息摘要，而不是只给通用失败

#### Scenario: materialization failure remains distinguishable from matcher failure

- **WHEN** 某个候选通过 `dtype` 与 `constraints`，但在 `mlir_text()` materialization 时失败
- **THEN** selection diagnostics MUST 将该候选标记为 materialization 相关失败
- **AND** MUST NOT 把该失败重新表述为 `dtype` mismatch 或 constraint failure

### Requirement: imported helper diagnostics MUST preserve `@pto.inline_proc` restrictions

跨文件导入的 helper MUST 继续遵守现有 `@pto.inline_proc` 诊断规则。  
frontend MUST 在 IR 生成之前拒绝以下情况：

- 被调用对象不是 `InlineProcDescriptor`
- helper 使用 unsupported Python syntax
- helper 使用 positional-only、varargs、kwargs 或 keyword-only 参数
- helper 发生递归或互递归
- helper 隐式捕获非允许的动态值

#### Scenario: imported ordinary Python function is rejected

- **WHEN** 模板从其它文件导入普通 Python 函数
- **AND** `@pto.vkernel` body 调用该函数
- **THEN** frontend MUST 把该调用视为 unsupported external call
- **AND** 诊断 MUST NOT 暗示普通 Python 函数会被自动 lowering

#### Scenario: imported inline proc recursion is rejected

- **WHEN** 跨文件导入的 `@pto.inline_proc` helper 直接或间接递归调用自身
- **THEN** frontend MUST 在 materialization 前报错
- **AND** 诊断 MUST 明确指出 recursive inline_proc call 不受支持

#### Scenario: imported inline proc illegal capture is rejected

- **WHEN** 跨文件导入的 `@pto.inline_proc` helper 隐式捕获非字面量或非允许的动态值
- **THEN** frontend MUST 报出 inline_proc capture 相关诊断
- **AND** 诊断位置 SHOULD 指向共享 helper 源文件中的相关 helper

### Requirement: duplicate imported inline-proc names MUST fail fast

当一个模板模块中可见多个同名 `@pto.inline_proc` helper，且 DSL body 只能通过简单名字调用该 helper 时，frontend MUST fail fast。  
实现 MUST NOT 静默依赖 import 顺序选择其中一个 helper。

#### Scenario: duplicate helper names are visible in one template module

- **WHEN** 模板通过多个 import 来源暴露同名 `@pto.inline_proc` helper
- **AND** kernel body 调用该简单名字
- **THEN** frontend MUST 报出重复 helper 名或 ambiguous inline_proc call 诊断
- **AND** MUST NOT 静默选择任意一个实现

### Requirement: Qualified imported helper calls SHALL be rejected until specified

The frontend SHALL treat `import shared_helper; shared_helper.helper(...)` as outside this change surface.  
Until qualified helper calls are specified by a follow-up change, the frontend SHALL reject this call form as an unsupported external or attribute call.

#### Scenario: qualified imported helper call is not part of this change

- **WHEN** 用户写 `import shared_helper` 并在 DSL body 中调用 `shared_helper.helper(...)`
- **THEN** frontend SHALL reject the call
- **AND** the rejection SHALL be treated as this change's controlled boundary

