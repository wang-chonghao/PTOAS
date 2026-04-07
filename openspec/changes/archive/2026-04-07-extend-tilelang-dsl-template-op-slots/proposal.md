# Proposal: 扩展 TileLang DSL 的模板槽位与多-op matcher

## 概述

当前 TileLang DSL 的 matcher 仍以单个 `op` 为中心，kernel body 里也只能直接写具体的 `pto.vadd`、`pto.vsub`、`pto.vmul` 等真实 op。  
这让 `tadd/tsub/tmul/tdiv` 这类同一 family op 很难共享一份实现；一旦核心计算只在一两个 vector op 上不同，作者就只能复制多份几乎相同的 kernel body。

## 背景与动机

现有 `tilelang-dsl-kernel-matcher` capability 已经支持多 signature `dtypes`、`constraints` 和 `priority`，但 descriptor 仍然只匹配一个具体 `op`。  
与此同时，DSL authoring surface 仍要求用户在 kernel body 中显式写出真实 `pto.*` 调用；当前 frontend 既不支持在 kernel body 中执行任意 Python dict/callable，也没有“模板 op -> concrete op”的编译期替换能力。

这会带来两个直接问题：

1. 同一 family op 的实现复用成本高

- 像 `tadd/tsub/tmul/tdiv` 这种共享同一 loop / mask / load-store 骨架的 kernel，作者必须复制多份函数体，只替换中间一条 `pto.v*` 调用。

2. 现有 DSL 没有正式的“模板化 authoring”契约

- 如果让用户直接在 kernel body 里写 Python 字典、索引和 callable 分发，就会把当前受限 DSL 推向“任意 Python 解释执行器”，破坏现有 frontend 的 deterministic 边界。

因此，需要一个新的正式 capability，把“同一份 kernel body 可被多个具体 op 复用”收敛为静态、可验证、可测试的 OpenSpec 契约。

## 目标

- 为 TileLang DSL 增加模板槽位 authoring capability，使用户可以在 kernel body 任意合法位置使用统一的模板调用，并在编译期被替换成当前 concrete `op` 对应的真实 `pto.*` op。
- 扩展 `tilelang-dsl-kernel-matcher` capability，使一个 descriptor 可以通过 `ops=[...]` 覆盖多个具体 PTO op，同时在 `select_kernel(...)` 之后绑定唯一 concrete `op`。
- 保持 TileLang DSL 继续是静态、受限、deterministic 的 frontend，不把 kernel body 扩展为任意 Python dict/callable 执行环境。

## 非目标

- 不支持在 kernel body 中直接执行 Python dict lookup、lambda、函数对象调用或其他 higher-order runtime dispatch。
- 不新增运行时模板分发机制；模板替换只发生在编译期。
- 不改变 `pto.select_kernel(target, op, operand_types, context_attrs, registry=None)` 的公共查询形态。
- 不在本 change 中一次性引入一批 family-specific placeholder op，如 `pto.vbinary`、`pto.vbinarys`、`pto.vcmp_template`。

## 变更内容

- 新增 `tilelang-dsl-template-slots` capability，定义：
  - `@pto.vkernel(..., templates={...})`
  - 通用模板入口 `pto.tpl("slot", ...)`
  - 基于当前 concrete `op` 的编译期静态替换
  - 模板槽位的 frontend diagnostics 与合法性边界
- 修改 `tilelang-dsl-kernel-matcher` capability，允许 descriptor 通过 `ops=[...]` 匹配多个具体 op，并要求 selector 在返回 descriptor 前绑定 concrete `op`。
- 要求模板替换继续复用现有 semantic/type-check/lowering 路径，最终仍输出当前合法的 authoring-form VPTO，而不是发明新的公开中间 IR。

## Capabilities

### New Capabilities

- `tilelang-dsl-template-slots`: 定义 `pto.tpl("slot", ...)`、`templates={...}` 静态映射、模板槽位的编译期替换规则，以及相关 frontend diagnostics 与 materialization 边界。

### Modified Capabilities

- `tilelang-dsl-kernel-matcher`: 从单 `op` descriptor 扩展到 `op` / `ops` matcher 元数据，并要求 `select_kernel(...)` 为多-op descriptor 绑定唯一 concrete `op` 后再进入 materialization。

## 预期结果

- 用户可以为 `tadd/tsub/tmul/tdiv` 这类同 family op 注册一份共享 kernel body，而不再复制多份只差一条核心 vector op 的实现。
- kernel body 里的模板调用在 frontend 阶段被静态展开成真实 `pto.*` 调用，后续 semantic/type-check/lowering 继续走现有路径。
- TileLang DSL 仍然保持“受限 Python 子集 + 固定 DSL call surface”的边界，不因为引入模板能力而变成任意 Python 执行器。

## 成功标准

- OpenSpec 中新增 `tilelang-dsl-template-slots` capability，并明确：
  - `templates={...}` 的静态结构
  - `pto.tpl("slot", ...)` 的编译期替换语义
  - 非法模板映射、未绑定 concrete `op`、未知 slot 等 frontend 失败路径
- OpenSpec 中修改 `tilelang-dsl-kernel-matcher` capability，并明确：
  - `ops=[...]` 与现有 `op=` 的关系
  - selector 对多-op descriptor 的 concrete `op` 绑定语义
  - 多-op descriptor 在未绑定 concrete `op` 时不得 materialize
- change 落地后，`tilelang-dsl/tests/` 能新增覆盖：
  - 多-op matcher 正例
  - `pto.tpl("slot", ...)` 对 `tadd/tsub/tmul/tdiv` 的展开正例
  - 模板映射与 materialization 的负例诊断

## 影响

- 受影响目录：
  - `tilelang-dsl/python/`
  - `tilelang-dsl/tests/`
  - `tilelang-dsl/examples/`
  - `tilelang-dsl/docs/`
  - `openspec/specs/`
- 受影响 public API：
  - `@pto.vkernel(..., op=..., ops=..., templates=...)`
  - `pto.select_kernel(...)`
  - `pto.tpl("slot", ...)`
