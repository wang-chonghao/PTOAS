# TileLang DSL Surface Specification

## Purpose
TBD - created by archiving change add-tilelang-dsl-core-foundation. Update Purpose after archive.
## Requirements
### Requirement: TileLang DSL v1 MUST live under `tilelang-dsl/` and expose a dedicated `tilelang_dsl` package

TileLang DSL v1 的实现、样例、测试和局部文档 MUST 集中在 `tilelang-dsl/`。  
对外 import 入口 MUST 是独立 package `tilelang_dsl`，不得继续把本特性的核心逻辑建立在现有 `python/pto/dialects/pto.py` 的实验 DSL 上。  
根目录其他路径若有改动，MUST 仅限最小 build/install/test wiring。

#### Scenario: TileLang DSL source stays isolated from existing Python binding code

- **WHEN** 仓库为 TileLang DSL v1 新增源码、样例、测试和局部文档
- **THEN** 这些工件 MUST 放在 `tilelang-dsl/` 下
- **AND** repo root 或 `python/` 现有目录树的改动 MUST 只承担最小接线职责
- **AND** `python/pto/dialects/pto.py` MUST NOT 继续作为 TileLang DSL v1 的 source of truth

### Requirement: v1 `@pto.vkernel` surface MUST be limited to the monomorphic `a5` profile

TileLang DSL v1 的 `@pto.vkernel` MUST 只接受 `target="a5"`。  
`op` MUST 作为必填 metadata 保留。  
`dtypes` MUST 只包含一个 monomorphic signature tuple。  
`name` 和 `verify` MAY 保留为可选字段。  
v1 不在 public surface 中支持多 signature `dtypes`、`constraints`、`priority`、`Any*` 或 `TypeVar`。

#### Scenario: monomorphic a5 kernel descriptor is accepted

- **WHEN** 用户定义 `@pto.vkernel(target="a5", op="scale", dtypes=[(pto.f32, pto.f32, pto.f32)])`
- **THEN** frontend MUST 接受该 decorator surface
- **AND** descriptor MUST 保留 `target/op/dtypes/name/verify` metadata 用于后续编译和调试

### Requirement: bare `TensorView` and `Tile` annotations MUST bind element types through the single `dtypes` signature

在 v1 中，`TensorView` 和 `Tile` 参数 MUST 允许使用 bare annotation。  
其元素类型 MUST 由 decorator 的单个 `dtypes` signature 按参数位置绑定。  
标量参数 MUST 继续使用显式标量注解，并与 `dtypes` 中对应位置的标量类型保持一致。

#### Scenario: `dtypes` binds operand element types positionally

- **WHEN** kernel 参数按位置写成 `TensorView, TensorView, Tile, pto.f32`
- **THEN** 单个 `dtypes` signature MUST 按同样的位置顺序提供两个 GM operand 的元素类型、一个 Tile operand 的元素类型和一个标量类型
- **AND** frontend MUST 使用该 signature 作为参数定型的唯一来源

### Requirement: bare `Tile` parameters MUST require explicit specialization before IR materialization

对 bare `Tile` 参数，frontend MUST 在 descriptor 上提供显式 specialization 入口。  
Tile 的 physical shape、memory space 和配置 MUST 在 specialization 阶段补全。  
在所有 bare `Tile` 参数完成 specialization 之前，descriptor MUST NOT 允许执行 `mlir_text()`, `mlir_module()`, `verify()` 或 `emit(path)`。

#### Scenario: specialized tile kernel can materialize IR

- **WHEN** kernel 含 bare `Tile` 参数，且调用方通过 `descriptor.specialize(**bindings)` 为所有 bare `Tile` 参数补齐静态 shape / space / config
- **THEN** 返回的 specialized descriptor MUST 允许调用 `mlir_text()`, `mlir_module()`, `verify()` 和 `emit(path)`
- **AND** specialization 之后的 Tile physical shape MUST 作为编译期静态契约固定下来

### Requirement: TileLang DSL v1 MUST expose fixed-width vector type construction as `pto.vreg(dtype)`

TileLang DSL v1 MUST 提供 public type constructor `pto.vreg(dtype)`。  
该 surface MUST 只接受元素类型，不接受显式 lanes 参数。  
frontend MUST 依据固定 256-byte vector register 宽度自动推导 lanes。  
当前 v1 若元素类型不在已支持的 vector lowering 子集内，frontend MUST fail fast。

#### Scenario: `pto.vreg(dtype)` returns the inferred fixed-width vector type

- **WHEN** 用户在 DSL 中书写 `pto.vreg(pto.f32)` 或 `pto.vreg(dst.element_type)`
- **THEN** frontend MUST 将其识别为 vector type expression
- **AND** `pto.f32` MUST 对应 `!pto.vreg<64xf32>`，`pto.f16` MUST 对应 `!pto.vreg<128xf16>`
- **AND** MUST NOT 要求用户显式提供 lanes 参数

### Requirement: TileLang DSL v1 MUST expose typed mask markers and MUST NOT expose `pto.memref(...)`

TileLang DSL v1 MUST 提供 public typed-mask marker：`pto.mask_b8`、`pto.mask_b16`、`pto.mask_b32`。  
frontend MUST 将这些 surface 识别为对应 `!pto.mask<b8>`、`!pto.mask<b16>`、`!pto.mask<b32>` 的 type expression。  
与此同时，DSL public surface MUST NOT 暴露 `pto.memref(...)` constructor；memref 只允许作为内部 IR / lowering 表达出现，不得作为 DSL authoring type surface。

#### Scenario: typed mask marker matches `make_mask` result type

- **WHEN** 用户书写 `mask: pto.mask_b32 = pto.make_mask(pto.f32, pto.PAT.ALL)`
- **THEN** frontend MUST 接受该注解
- **AND** 若注解 granularity 与 `make_mask` 推导结果不一致，frontend MUST fail fast

#### Scenario: `pto.memref(...)` is not part of the DSL public type surface

- **WHEN** 用户查看或使用 TileLang DSL v1 public type surface
- **THEN** 文档和 package surface MUST 只暴露 `TensorView`、`Tile`、typed pointer、`pto.vreg(...)`、typed mask 等 authoring type
- **AND** MUST NOT 将 `pto.memref(...)` 描述为 DSL 侧可用 constructor

### Requirement: TileLang DSL templates MUST support directly imported `@pto.inline_proc` helpers

TileLang DSL 模板文件 MUST 支持从同一模板目录或模板包中直接导入 source-visible `@pto.inline_proc` helper，并在 `@pto.vkernel` body 中以简单名字调用该 helper。  
支持形式 MUST 至少包含：

- `from shared_helper import helper`
- `from TileOps.shared_helper import helper`

导入 helper MUST 与当前模板文件内定义的 `@pto.inline_proc` helper 使用同一 authoring contract。  
普通 Python 函数即便被 import，也 MUST NOT 因跨文件 import 而变成 TileLang DSL 可分析 helper。

#### Scenario: template imports a shared inline proc from the same template directory

- **WHEN** `shared_helper.py` 定义 `@pto.inline_proc def helper(...)`
- **AND** `some_template.py` 通过 `from shared_helper import helper` 导入它
- **AND** `some_template.py` 的 `@pto.vkernel` body 调用 `helper(...)`
- **THEN** frontend MUST 接受该调用作为合法 inline-proc call
- **AND** 用户 MUST NOT 需要在 `some_template.py` 中复制 helper body

#### Scenario: template imports a shared inline proc through the template package

- **WHEN** 模板目录可作为 Python package 导入
- **AND** 模板通过 `from TileOps.shared_helper import helper` 导入 `@pto.inline_proc`
- **THEN** frontend MUST 接受该 helper 的简单名字调用
- **AND** 该行为 MUST 与同目录裸模块导入保持一致

### Requirement: shared helper files MUST be usable without vkernel descriptors

共享 helper 文件 MUST 允许只包含 `@pto.inline_proc` helper，而不包含任何 `@pto.vkernel` descriptor。  
模板扫描和导入流程 MUST NOT 要求共享 helper 文件本身匹配 TileOp 或注册 kernel descriptor。

#### Scenario: shared helper file has no vkernel descriptor

- **WHEN** `shared_helper.py` 只定义一个或多个 `@pto.inline_proc`
- **AND** 没有定义 `@pto.vkernel`
- **THEN** 该文件仍 MAY 被其它模板导入复用
- **AND** 模板扫描 MUST NOT 因该共享文件没有 descriptor 而拒绝使用它

### Requirement: cross-file helper support MUST NOT imply high precision algorithm implementation

跨文件 helper import 能力 MUST 独立于具体算法实现。  
验证该能力时 MAY 使用简单 helper，例如 pass-through、vector add 或 small arithmetic helper。  
本 change MUST NOT 要求实现 high precision divide、exp、rsqrt 或其它复杂近似算法。

#### Scenario: lightweight helper validates the import contract

- **WHEN** 测试使用一个简单 `@pto.inline_proc` 验证跨文件调用
- **THEN** 该测试 MUST 足以证明模板 import、frontend collection、lowering 和 inline path 生效
- **AND** 测试 MUST NOT 依赖 high precision divide 的具体算法正确性

