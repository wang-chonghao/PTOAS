## 1. Descriptor 与 matcher 元数据扩展

- [x] 1.1 扩展 `tilelang-dsl/python/tilelang_dsl/kernel.py` 的 `@pto.vkernel` / descriptor 校验逻辑，支持互斥的 `op=` 与 `ops=[...]`，并为 multi-op descriptor 保存 `match_ops` 与 `selected_op`
- [x] 1.2 更新 `pto.select_kernel(...)` 的候选过滤与绑定逻辑，使其能匹配 `ops=[...]` descriptor，并在返回前绑定 concrete `selected_op`
- [x] 1.3 为 multi-op descriptor 增加 materialization gate，确保未绑定 concrete `selected_op` 时拒绝 `mlir_text()`、`mlir_module()`、`verify()` 和 `emit(path)`

## 2. 模板槽位 frontend 能力

- [x] 2.1 在 descriptor 元数据中增加 `templates={...}` 静态映射的解析与校验，限制 slot 名、映射 key/value 和与 `match_ops` 的一致性
- [x] 2.2 在 frontend AST 构建路径中增加 `pto.tpl("slot", ...)` 识别与 compile-time 替换，把模板调用重写成真实 `pto.*` call
- [x] 2.3 补齐模板槽位的 fail-fast diagnostics，覆盖非字面量 slot、未知 slot、缺失 op 映射、非法真实 op 名以及未绑定 `selected_op` 的错误路径

## 3. 回归测试与验证

- [x] 3.1 在 `tilelang-dsl/tests/test_tilelang_dsl_v1.py` 增加 multi-op matcher 回归，覆盖 `ops=[...]` 命中、single-op/multi-op 竞争、priority 与 tie error 行为
- [x] 3.2 在 `tilelang-dsl/tests/test_tilelang_dsl_v1.py` 增加 template-slot 正例回归，验证同一份 kernel body 在 `tadd/tsub/tmul/tdiv` 下分别展开成正确真实 `pto.*` op
- [x] 3.3 在 `tilelang-dsl/tests/test_tilelang_dsl_v1.py` 增加 template-slot 负例回归，覆盖未绑定 `selected_op`、未知 slot、非字面量 slot、非法映射值和 callable-based runtime dispatch reject
- [x] 3.4 运行最小验证集，至少包括 `PYTHONPATH=$PWD/tilelang-dsl/python python3 -m unittest $PWD/tilelang-dsl/tests/test_tilelang_dsl_v1.py`

## 4. 样例与文档

- [x] 4.1 新增或更新 `tilelang-dsl/examples/` 中的共享 kernel body 样例，展示 `ops=[...] + templates={...} + pto.tpl("slot", ...)` 的推荐写法
- [x] 4.2 更新 `docs/tilelang-dsl-guide.md`，补充 template-slot surface、`op`/`ops` 语义、编译期替换模型和不支持 kernel body Python dict/callable 的原因
- [x] 4.3 更新 `tilelang-dsl/docs/matcher-and-advanced-surface-migration.md` 或相邻文档，说明从显式真实 `pto.*` 调用迁移到模板槽位写法的适用场景与边界
