## ADDED Requirements

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
