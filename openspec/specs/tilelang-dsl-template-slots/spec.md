# tilelang-dsl-template-slots Specification

## ADDED Requirements

### Requirement: TileLang DSL MUST provide static template-slot metadata and a compile-time placeholder call

TileLang DSL MUST 提供基于 descriptor 元数据的模板槽位机制，用于让多个 concrete PTO op 共享同一份 kernel body。  
`@pto.vkernel` MAY 声明 `templates={...}` 静态映射；kernel body MAY 使用统一模板入口 `pto.tpl("slot", ...)`。  
`templates` MUST 是静态 mapping，slot 名 MUST 是非空字符串，映射 value MUST 是真实 `pto.*` op 名字符串。  
系统 MUST NOT 要求用户在 kernel body 中执行 Python dict lookup、callable value 调用或其他 runtime dispatch 来实现该能力。

#### Scenario: kernel declares a template slot and uses the placeholder call

- **WHEN** 一个 kernel descriptor 声明 `templates={"core": {"tadd": "vadd", "tsub": "vsub"}}`，并在 kernel body 中使用 `pto.tpl("core", lhs, rhs, mask)`
- **THEN** frontend MUST 接受该模板槽位写法
- **AND** 该模板调用 MUST 被视为 compile-time placeholder，而不是 runtime helper

### Requirement: template-slot substitution MUST resolve from the selected concrete op before semantic checking and lowering

对使用模板槽位的 kernel，frontend MUST 在 semantic checking 和 lowering 之前，根据 descriptor 已绑定的 concrete `selected_op` 把 `pto.tpl("slot", ...)` 静态替换成真实 `pto.<resolved-op>(...)` 调用。  
替换后的真实调用 MUST 继续沿用现有 semantic/type-check/lowering 路径，并满足当前 authoring-form VPTO legality contract。  
模板调用 MAY 出现在 loop、`if`、显式 `strict_vecscope` 或 inferred `pto.vecscope` 等任意合法 DSL 位置；其替换结果 MUST 与用户直接书写真实 `pto.*` 调用等价。

#### Scenario: one shared kernel body expands to different real ops for different selected op values

- **WHEN** 同一个 descriptor 通过不同的 concrete query `op` 分别绑定到 `selected_op="tadd"` 和 `selected_op="tsub"`
- **THEN** `pto.tpl("core", lhs, rhs, mask)` MUST 分别静态展开成 `pto.vadd(lhs, rhs, mask)` 和 `pto.vsub(lhs, rhs, mask)`
- **AND** 后续 semantic/type-check/lowering MUST 只看到展开后的真实 `pto.*` 调用

#### Scenario: template placeholder remains valid inside legal control-flow and vecscope contexts

- **WHEN** `pto.tpl("core", ...)` 出现在合法的 `for`、`if`、`strict_vecscope` 或 inferred `pto.vecscope` 上下文中
- **THEN** frontend MUST 先完成模板替换
- **AND** 替换后的编译结果 MUST 与同位置直接书写真实 `pto.*` 调用保持等价

### Requirement: template slots MUST fail fast on unresolved or invalid static mappings

模板槽位是 compile-time static surface，因此 frontend MUST 对以下情况 fail-fast：

- `pto.tpl(...)` 的 slot 不是字符串字面量
- 使用了未声明的 slot
- 当前 `selected_op` 在该 slot 下没有映射
- 映射 value 不是已支持的真实 `pto.*` op 名
- descriptor 尚未绑定 concrete `selected_op` 就尝试解析模板槽位

这些错误 MUST 在生成任何 VPTO IR 之前报出，且诊断 MUST 明确指出失败的 slot 或 concrete `op` 绑定原因。

#### Scenario: unknown slot or missing op mapping is rejected before IR generation

- **WHEN** kernel body 中使用 `pto.tpl("core", ...)`，但 descriptor 没有声明 `core` slot，或该 slot 未覆盖当前 `selected_op`
- **THEN** frontend MUST 在生成任何 VPTO IR 之前报错
- **AND** 诊断 MUST 指出缺失的 slot 或缺失的 concrete `op` 映射

#### Scenario: non-literal slot name is rejected as unsupported template syntax

- **WHEN** 用户写出 `pto.tpl(slot_name, lhs, rhs, mask)`，其中 `slot_name` 不是字符串字面量
- **THEN** frontend MUST 直接报错
- **AND** MUST NOT 把该写法当作运行时字符串分发处理

### Requirement: template slots MUST NOT introduce arbitrary Python callable semantics into the DSL

TileLang DSL MUST 继续保持受限 Python 子集。  
实现 MUST NOT 因为模板槽位能力而接受 kernel body 中的 dict-lookup callable、lambda、闭包函数对象调用或其他 higher-order dispatch。  
模板化 authoring 的正式路径 MUST 是 descriptor 元数据中的 `templates={...}` 加上 kernel body 中的 `pto.tpl("slot", ...)`。

#### Scenario: callable-based runtime template dispatch remains rejected

- **WHEN** 用户尝试在 kernel body 中通过 `table["core"](lhs, rhs, mask)`、`resolver(lhs, rhs, mask)` 或等价 callable-dispatch 写法实现模板分发
- **THEN** frontend MUST 继续按 unsupported Python / unsupported call surface 拒绝该写法
- **AND** MUST NOT 把它解释成合法的 template-slot surface
