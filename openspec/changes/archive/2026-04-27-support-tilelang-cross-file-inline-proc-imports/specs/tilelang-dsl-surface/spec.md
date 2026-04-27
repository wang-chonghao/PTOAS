## ADDED Requirements

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
