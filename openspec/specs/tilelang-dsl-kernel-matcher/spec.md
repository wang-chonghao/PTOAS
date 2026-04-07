# tilelang-dsl-kernel-matcher Specification

## MODIFIED Requirements

### Requirement: TileLang DSL MUST provide an explicit kernel registry and selection API

当同一 `target/op` 下存在多个 `@pto.vkernel` descriptor 时，TileLang DSL MUST 将它们注册到显式、可查询的 `KernelRegistry`。  
默认 registry MUST 是 module-level 对象；调用方 MAY 传入自定义 registry 以获得隔离的候选集合。  
系统 MUST 提供显式 selection API `pto.select_kernel(target, op, operand_types, context_attrs, registry=None)`，用于在给定 `target`、concrete `op`、operand type 信息和上下文属性时选择唯一 kernel。  
descriptor MUST 支持两种互斥的 matcher 元数据：

- `op="<concrete-op>"`
- `ops=["<op0>", "<op1>", ...]`

descriptor MUST 至少提供其中一种，且实现 MUST NOT 同时接受两者。  
当 selector 命中一个 `ops=[...]` descriptor 时，返回结果 MUST 绑定当前 query 对应的唯一 concrete `selected_op`，再进入后续 `specialize()` / `mlir_text()` / `verify()` 流程。  
实现 MUST NOT 依赖扫描 Python globals、locals 或导入顺序来隐式发现候选。

#### Scenario: selector returns the unique best kernel

- **WHEN** registry 中存在多个针对同一 `target/op` 的 kernel descriptor，且其中一个在全部匹配步骤后成为唯一最佳候选
- **THEN** `pto.select_kernel(...)` MUST 返回该 descriptor
- **AND** 返回结果 MUST 可继续走 `specialize()` / `mlir_text()` / `verify()` 流程

#### Scenario: custom registry restricts the candidate set explicitly

- **WHEN** 调用方显式传入一个只含局部 kernel 的 `KernelRegistry`
- **THEN** selector MUST 只在该 registry 的候选集合内做匹配和决策
- **AND** MUST NOT 回退去查询 module-level 默认 registry

#### Scenario: selector binds the concrete op for a multi-op descriptor

- **WHEN** 一个 descriptor 通过 `ops=["tadd", "tsub", "tmul", "tdiv"]` 注册，且调用方以 `pto.select_kernel(..., op="tmul", ...)` 查询命中该 descriptor
- **THEN** selector MUST 返回已经绑定 `selected_op="tmul"` 的 descriptor
- **AND** 后续 materialization MUST 基于该 concrete `selected_op` 而不是未绑定的原始 matcher 集合

### Requirement: matcher MUST support concrete types, `Any*`, and `TypeVar` across multiple signatures

matcher MUST 支持：

- 多个 `dtypes` signature
- `AnyFloat`
- `AnyInt`
- `AnyType`
- `AnyMask`
- `TypeVar`

`TypeVar` 在单个 signature 内 MUST 约束所有同名位置绑定到同一最终类型。  
多个 `dtypes` signature MUST 逐个独立求值；某个 signature 的 `TypeVar` 绑定状态 MUST NOT 泄漏到另一个 signature。

#### Scenario: wildcard and type-variable signatures match deterministically

- **WHEN** 某个 kernel 使用多个 `dtypes` signature，并在其中混用 concrete type、`Any*` 与 `TypeVar`
- **THEN** matcher MUST 对每个 signature 独立求值
- **AND** 只有满足所有 `TypeVar` 一致性约束的 signature 才能视为匹配成功

### Requirement: selection order MUST be target -> op -> dtype signature -> constraints -> priority -> tie error

对一个 registry 中的候选集合，selector MUST 按以下固定顺序求值：

1. `target`
2. `op`
3. `dtypes` signature 的 concrete / wildcard / type-variable 匹配
4. `constraints`
5. `priority`
6. highest-priority tie error

其中第 2 步的 `op` 匹配 MUST 使用调用方给出的 concrete query `op`，并按以下规则求值：

- 对 `op="<concrete-op>"` descriptor，要求 exact match
- 对 `ops=[...]` descriptor，要求 query `op` 属于该 matcher 集合

实现 MUST 保持该顺序 deterministic。  
系统 MUST NOT 依赖注册顺序、定义顺序、导入顺序或其他隐式规则来打破同一阶段的歧义。  
系统 MUST NOT 因为候选是 single-op descriptor 或 multi-op descriptor 而引入额外隐式优先级。

#### Scenario: type match happens before constraints and priority

- **WHEN** 一个候选在 `target/op` 上匹配，但没有任何 `dtypes` signature 能通过 concrete / wildcard / `TypeVar` 规则
- **THEN** 该候选 MUST 在进入 `constraints` 评估前被移除
- **AND** 其 `priority` MUST NOT 参与后续决策

#### Scenario: multi-op descriptor participates in selection without hidden specificity bonus

- **WHEN** 同一个 concrete query `op` 同时命中 single-op descriptor 与 multi-op descriptor
- **THEN** selector MUST 继续按既有的 `dtypes -> constraints -> priority -> tie error` 顺序求值
- **AND** MUST NOT 仅因为 single-op descriptor 更“具体”就隐式优先选择它

### Requirement: constraint evaluation MUST happen after type matching and before priority resolution

对同一 `target/op` 的候选集合，matcher MUST 先完成 dtype matching，再评估 `constraints`。  
只有通过 constraint evaluation 的候选，才允许进入 `priority` 比较阶段。

#### Scenario: higher-priority kernel with failing constraint does not win

- **WHEN** 一个更高 `priority` 的 kernel 在 target/op/type 层面匹配成功，但 `constraints` 评估失败
- **THEN** 该 kernel MUST 从候选集合中移除
- **AND** selector MUST 继续在剩余候选中选择合法 kernel

### Requirement: priority ties MUST raise an explicit selection error

若在 target/op/type/constraint 全部通过后，最高 `priority` 仍对应多个候选，matcher MUST 报显式选择错误。  
系统 MUST NOT 依赖定义顺序、导入顺序或其他隐式规则做 tiebreak。

#### Scenario: equal-priority winners cause deterministic tie error

- **WHEN** 多个 kernel 在 target/op/type/constraint 匹配后拥有相同的最高 `priority`
- **THEN** selector MUST 报错
- **AND** 错误消息 MUST 指出发生 tie 的 kernel 集合
- **AND** MUST NOT 静默选择第一个已注册 kernel

## ADDED Requirements

### Requirement: multi-op descriptors MUST require concrete op binding before IR materialization

当 descriptor 使用 `ops=[...]` 覆盖多个 concrete PTO op 时，系统 MUST 在 materialization 前先绑定唯一 `selected_op`。  
在未绑定 concrete `selected_op` 之前，descriptor MUST NOT 允许执行 `mlir_text()`、`mlir_module()`、`verify()` 或 `emit(path)`。  
一旦 selector 已绑定 concrete `selected_op`，该 descriptor MUST 与已绑定 concrete dtype signature 的其他 descriptor 一样继续参与 specialization 和 materialization。

#### Scenario: unresolved multi-op descriptor is rejected before materialization

- **WHEN** 用户直接对一个通过 `ops=[...]` 注册、但尚未经过 `pto.select_kernel(...)` 绑定 concrete `selected_op` 的 descriptor 调用 `mlir_text()`
- **THEN** frontend MUST 直接报错
- **AND** 诊断 MUST 明确指出该 descriptor 需要先绑定 concrete `op`

#### Scenario: selected multi-op descriptor can materialize normally

- **WHEN** 一个 `ops=[...]` descriptor 已经通过 `pto.select_kernel(...)` 绑定了 concrete `selected_op`
- **THEN** 调用方 MUST 可以继续执行 `specialize()`、`mlir_text()`、`verify()` 和 `emit(path)`
- **AND** materialization 结果 MUST 使用已绑定的 concrete `selected_op`
