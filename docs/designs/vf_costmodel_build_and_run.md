# PTOAS VF CostModel 改动与构建运行说明

本文记录当前分支中 VF CostModel 接入 PTOAS 的主要改动范围，以及从干净环境开始编译并运行最小例子的步骤。

## 1. 改动目标

当前改动的目标是把 VF CostModel 接入 PTOAS 的 A5 frontend tile fusion path。

当前已经实现的链路是：

```text
.pto tileop IR
  -> pto-fusion-plan
  -> 构造融合后的 VfSimProgram
  -> 调用 PTOAS 内置 C++ VfSimulator
  -> 得到 VF latency cycles
  -> 给 tileop 标注 fusion metadata
```

当前主要服务的是第一阶段和第二阶段目标：

- 在 tileop 层接入 VF latency 预测。
- 支持从 fusion group 直接构造融合后的 vector micro-op program。
- 支持基础 elementwise op 的映射，包括 `tadd/tadds/tmul/tmuls/tmax/tmaxs/tmin/tmins` 等。
- 将 C++ 化后的 VfSimulator 放入 PTOAS 内部，避免依赖外部 VfSimulator 仓库。
- 保留 JSON dump 路径，方便和 Python VfSimulator 对齐调试。

## 2. 主要改动目录

### 2.1 VF CostModel 公共接口

```text
include/PTO/VFcostmodel/
```

主要文件：

```text
include/PTO/VFcostmodel/VfSimProgram.h
include/PTO/VFcostmodel/VfLatencyModel.h
include/PTO/VFcostmodel/VfCostModel.h
```

作用：

- 定义 PTOAS 内部使用的 `VfSimProgram`、`VfSimLoop`、`VfSimInst`。
- 定义 latency model 接口。
- 定义 tile fusion 层调用 costmodel 的接口。

`VfSimProgram` 是当前 PTOAS 给 C++ VfSimulator 的核心输入结构。它支持：

- 嵌套 loop。
- loop 内指令。
- loop 外指令。
- 指令 operand 上携带 dtype 信息。

### 2.2 C++ VfSimulator

```text
include/PTO/VFcostmodel/VfSimulator/
lib/PTO/VFcostmodel/VfSimulator/
```

主要文件：

```text
IDU.h / IDU.cpp
IFU.h / IFU.cpp
OOO.h / OOO.cpp
ISATraits.h / ISATraits.cpp
ParamDB.h / ParamDB.cpp
ProgramFlatten.h / ProgramFlatten.cpp
ProgramAnalysis.h / ProgramAnalysis.cpp
SimulatorRunner.h / SimulatorRunner.cpp
Json.h / Json.cpp
```

作用：

- 将原 Python VfSimulator 主线模型 C++ 化。
- 当前只接入主线工程可达模型，理论极限模型暂未接入。
- 支持从 `VfSimProgram` 直接进入 IFU/IDU/OoO/LSU 仿真。
- 支持输出调试日志，例如 `start_by_cycle.json`、`done_by_cycle.json`、`idu_to_ooo.json`、`vloop_trace.json`。

### 2.3 VF CostModel 实现

```text
lib/PTO/VFcostmodel/
```

主要文件：

```text
lib/PTO/VFcostmodel/VfSimProgram.cpp
lib/PTO/VFcostmodel/VfLatencyModel.cpp
lib/PTO/VFcostmodel/VfCostModel.cpp
```

作用：

- 实现 `VfSimProgram` 打印和 JSON dump。
- 实现 `VfSimProgram -> C++ VfSimulator -> latency cycles`。
- 实现 tile fusion costmodel 的调用入口。

### 2.4 Tile Fusion Pass 接入

```text
include/PTO/Transforms/TileFusion/
lib/PTO/Transforms/TileFusion/
```

主要文件：

```text
include/PTO/Transforms/TileFusion/FusionCostModel.h
include/PTO/Transforms/TileFusion/FusionOpSemantics.h
lib/PTO/Transforms/TileFusion/PTOFusionPlan.cpp
lib/PTO/Transforms/TileFusion/FusionCostModel.cpp
lib/PTO/Transforms/TileFusion/FusionOpSemantics.cpp
lib/PTO/Transforms/TileFusion/FusionAnalysis.cpp
lib/PTO/Transforms/TileFusion/PTOOpScheduling.cpp
lib/PTO/Transforms/TileFusion/PTOMarkLastUse.cpp
```

作用：

- `FusionPlanPass` 中调用 VF costmodel。
- 从 `FusionComputeNode` group 直接构造融合后的 `VfSimProgram`。
- 给接受融合的 tileop 标注：

```text
pto.fusion.group_id
pto.fusion.order
```

- 后续 `PTOOpScheduling` 根据 group/order 调整 tileop 顺序。
- `PTOMarkLastUse` 标注：

```text
pto.last_use
```

当前还没有输出 `loop_id`、`loop_unroll` 等优化 metadata。

### 2.5 Pass 参数

```text
include/PTO/Transforms/Passes.td
```

给 `pto-fusion-plan` 增加了两个调试参数：

```text
--dump-vf-program
--dump-vf-program-json=<path>
```

用途：

- `--dump-vf-program`：把构造出的 `VfSimProgram` 打印到 stderr。
- `--dump-vf-program-json=<path>`：把构造出的 `VfSimProgram` 写成 JSON，便于和外部 Python VfSimulator 对齐。

### 2.6 内置配置

```text
configs/
```

作用：

- 放置 C++ VfSimulator 需要的 uarch/isa 参数。
- PTOAS 内部 C++ VfSimulator 从 PTOAS 工程目录读取配置。
- 不再依赖外部 VfSimulator 仓库路径。

### 2.7 测试用例

```text
test/lit/tile_fusion/
```

当前新增或使用的关键例子：

```text
test/lit/tile_fusion/min_tadd_tmul.pto
```

该例子是最小 TADD -> TMUL produce-consumer fusion case。

## 3. 当前支持范围

当前重点支持基础 elementwise tileop。

已接入的典型映射：

```text
tadd   -> vadd
tadds  -> vadds
tmul   -> vmul
tmuls  -> vmuls
tmax   -> vmax
tmaxs  -> vmaxs
tmin   -> vmin
tmins  -> vmins
```

当前默认构造的 VF program 形式是：

```text
loop trip_count = tile element count / vector lane element count
  vlds input tile
  vector compute op
  vsts output tile
```

对于 produce-consumer chain，会尽量构造成融合后的 VF program，例如：

```text
tadd -> tmul
```

会生成类似：

```text
loop 0 trip_count=16 unroll=1
  reg1 = vlds tile0
  reg3 = vlds tile2
  reg4 = vadd reg1, reg3
  reg6 = vlds tile5
  reg7 = vmul reg4, reg6
  vsts tile8, reg7
```

当前限制：

- 只接入 C++ VfSimulator 主线模型。
- 理论极限模型暂未接入。
- 复杂 tileop pattern 尚未完整覆盖，例如 rowreduce、rowexpand、复杂 cast 等。
- 动态 shape 当前不作为第一阶段支持目标。
- vector lane 相关参数后续应继续和 PTOAS/VPTO 后端配置统一，目前还有进一步收敛空间。

## 4. 从头构建环境

### 4.1 拉取代码

```bash
git clone https://github.com/wang-chonghao/PTOAS.git
cd PTOAS
git checkout vf-costmodel-phase1
```

如果是在已有目录中：

```bash
cd /mnt/e/vfsimulator_structure/PTOAS
git checkout vf-costmodel-phase1
```

### 4.2 准备编译环境

如果使用已有 A5/PTO 环境脚本：

```bash
source /mnt/e/PTO/tools/wsl_env_a5_davc310.sh
```

然后检查关键工具：

```bash
which cmake
which ninja
which clang++
```

如果本地 LLVM/MLIR 环境没有配置好，CMake 可能找不到 MLIR/LLVM package。此时需要先按照 PTOAS README 或本地环境脚本配置 LLVM/MLIR 路径。

### 4.3 配置 CMake

建议使用独立构建目录：

```bash
cmake -S . -B build-vf-costmodel -G Ninja
```

如果 CMake 找不到 LLVM/MLIR，可以检查：

```bash
echo $LLVM_DIR
echo $MLIR_DIR
```

必要时显式传入：

```bash
cmake -S . -B build-vf-costmodel -G Ninja \
  -DLLVM_DIR=<path-to-llvm-cmake> \
  -DMLIR_DIR=<path-to-mlir-cmake>
```

具体路径取决于本机 LLVM/MLIR 安装位置。

### 4.4 编译 ptoas

```bash
cmake --build build-vf-costmodel --target ptoas -j2
```

建议先用 `-j2`。如果 WSL 内存或进程资源不稳定，不建议直接开很高并行度。

编译完成后，二进制位置通常是：

```text
build-vf-costmodel/tools/ptoas/ptoas
```

检查：

```bash
./build-vf-costmodel/tools/ptoas/ptoas --help
```

应能看到和 VF costmodel 相关的参数：

```text
--dump-vf-program
--dump-vf-program-json=<string>
```

## 5. 运行最小例子

最小例子路径：

```text
test/lit/tile_fusion/min_tadd_tmul.pto
```

内容是一个 `tadd -> tmul` chain：

```text
%tmp0 = tadd(%a, %b)
%tmp1 = tmul(%tmp0, %c)
```

运行：

```bash
./build-vf-costmodel/tools/ptoas/ptoas \
  --pto-arch=a5 \
  --pto-level=level2 \
  --enable-op-fusion \
  --emit-pto-ir \
  --dump-vf-program \
  test/lit/tile_fusion/min_tadd_tmul.pto \
  -o /dev/null
```

预期能看到类似输出：

```text
[pto-fusion-plan] VF program for fusion group:
loop 0 trip_count=16 unroll=1
  reg1 = vlds tile0
  reg3 = vlds tile2
  reg4 = vadd reg1, reg3
  reg6 = vlds tile5
  reg7 = vmul reg4, reg6
  vsts tile8, reg7
[pto-fusion-plan] VF latency cycles=82
```

这里的关键点：

- `trip_count=16` 来自 `32x32xf32` tile，元素数为 `1024`。
- 当前按 `256B` vector 宽度理解，`fp32` 每次处理 `64` 个元素。
- 因此循环次数是 `1024 / 64 = 16`。
- `VF latency cycles=82` 是 C++ VfSimulator 主线模型给出的预测结果。

## 6. 输出 JSON 方便调试

可以把 `VfSimProgram` dump 成 JSON：

```bash
./build-vf-costmodel/tools/ptoas/ptoas \
  --pto-arch=a5 \
  --pto-level=level2 \
  --enable-op-fusion \
  --emit-pto-ir \
  --dump-vf-program-json=/tmp/min_tadd_tmul_vfprogram.json \
  test/lit/tile_fusion/min_tadd_tmul.pto \
  -o /dev/null
```

输出文件：

```text
/tmp/min_tadd_tmul_vfprogram.json
```

这个 JSON 用于：

- 检查 PTOAS 构造出的融合后 VF program。
- 和外部 Python VfSimulator 做结果对比。
- 定位 C++ VfSimulator 和 Python VfSimulator 的语义差异。

## 7. 查看 fusion metadata

如果想看 `pto-fusion-plan` 后的 IR：

```bash
./build-vf-costmodel/tools/ptoas/ptoas \
  --pto-arch=a5 \
  --pto-level=level2 \
  --enable-op-fusion \
  --emit-pto-ir \
  --mlir-print-ir-after=pto-fusion-plan \
  test/lit/tile_fusion/min_tadd_tmul.pto \
  -o /dev/null
```

重点关注：

```text
pto.fusion.group_id
pto.fusion.order
```

如果继续看调度和 last use：

```bash
./build-vf-costmodel/tools/ptoas/ptoas \
  --pto-arch=a5 \
  --pto-level=level2 \
  --enable-op-fusion \
  --emit-pto-ir \
  --mlir-print-ir-after=pto-op-scheduling \
  --mlir-print-ir-after=pto-mark-last-use \
  test/lit/tile_fusion/min_tadd_tmul.pto \
  -o /dev/null
```

重点关注：

```text
pto.last_use
```

`pto.last_use` 表示该 tile operand 在当前 fusion span 中是否是最后一次使用，可用于后端判断中间结果是否还需要保留。

## 8. 常见问题

### 8.1 看不到 VF program 输出

确认命令里有：

```text
--enable-op-fusion
--dump-vf-program
```

同时需要：

```text
--pto-arch=a5
--pto-level=level2 或 level3
```

否则 A5 frontend fusion path 不会启用。

### 8.2 没有触发 fusion

常见原因：

- tileop 不在当前支持的 elementwise 范围内。
- tileop 之间不是当前支持的 produce-consumer 关系。
- shape 是动态 shape。
- op 中间存在当前 legality 不允许跨越的边界。

### 8.3 CMake 找不到 MLIR/LLVM

先确认环境脚本是否 source：

```bash
source /mnt/e/PTO/tools/wsl_env_a5_davc310.sh
```

再检查：

```bash
echo $LLVM_DIR
echo $MLIR_DIR
```

如果仍然为空，需要按本机 LLVM/MLIR 安装路径显式传入 CMake。

### 8.4 WSL 编译中途退出

优先降低并行度：

```bash
cmake --build build-vf-costmodel --target ptoas -j2
```

不要直接使用很大的 `-j` 数。

## 9. 后续计划

后续主要工作：

- 扩展 tileop 到 vector op 的映射覆盖范围。
- 支持更多 pattern，例如 cast、rowexpand、rowreduce 等。
- 接入更多 metadata，例如 `loop_id`、`loop_unroll`。
- 引入 UB 容量约束，在容量允许时尽量融合。
- 加入 unroll 寻优和 loop 结构优化。
- 继续对齐 C++ VfSimulator 和 Python VfSimulator 的回归测试。
