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

## 4. 新机器从零配置环境

本节按 Ubuntu/WSL 环境写。目标是从一台干净机器开始，最终跑出：

```text
[pto-fusion-plan] VF latency cycles=82
```

PTOAS 依赖 LLVM/MLIR `llvmorg-19.1.7`。如果机器上没有可用的 LLVM/MLIR 19.1.7，需要先编译 LLVM。

### 4.1 推荐目录结构

先建一个统一工作目录：

```bash
export WORKSPACE_DIR=$HOME/ptoas-workspace
export LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
export PTO_BUILD_DIR=$PTO_SOURCE_DIR/build-vf-costmodel
export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install-vf-costmodel

mkdir -p "$WORKSPACE_DIR"
```

后续命令默认这些变量已经设置。换 shell 后需要重新 export，或者写入自己的环境脚本。

### 4.2 安装系统依赖

Ubuntu/WSL：

```bash
sudo apt update
sudo apt install -y \
  git \
  cmake \
  ninja-build \
  build-essential \
  clang \
  lld \
  python3 \
  python3-pip \
  python3-venv \
  zlib1g-dev \
  libzstd-dev
```

检查版本：

```bash
cmake --version
ninja --version
python3 --version
clang++ --version
```

基本要求：

- CMake >= 3.20。
- Python >= 3.8。
- C++ 编译器支持 C++17。

### 4.3 创建 Python 虚拟环境

建议新机器上使用 venv，避免污染系统 Python：

```bash
cd "$WORKSPACE_DIR"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pybind11==2.12.0 numpy lit
```

说明：

- `pybind11==2.12.0` 是为了兼容当前 LLVM/MLIR Python binding。
- `lit` 用于后续跑 LLVM/MLIR 风格测试，不跑测试时不是必须，但建议安装。

确认：

```bash
python -m pybind11 --cmakedir
```

### 4.4 下载并编译 LLVM/MLIR 19.1.7

如果机器上已经有 LLVM/MLIR 19.1.7，可以跳到 4.5。

下载源码：

```bash
cd "$WORKSPACE_DIR"
git clone https://github.com/llvm/llvm-project.git
cd "$LLVM_SOURCE_DIR"
git checkout llvmorg-19.1.7
```

配置 LLVM/MLIR：

```bash
cmake -G Ninja -S llvm -B "$LLVM_BUILD_DIR" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DBUILD_SHARED_LIBS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$(which python)" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="host"
```

编译：

```bash
cmake --build "$LLVM_BUILD_DIR" -j2
```

新机器第一次编译 LLVM 会比较久。WSL 内存不稳定时先用 `-j2`，确认稳定后再提高并行度。

验证 LLVM：

```bash
"$LLVM_BUILD_DIR/bin/llvm-config" --version
```

预期输出：

```text
19.1.7
```

### 4.5 如果已有 LLVM/MLIR 19.1.7

如果另一台机器已经安装了 LLVM/MLIR 19.1.7，例如在 `/opt/llvm`，可以不编译 LLVM，直接设置：

```bash
export LLVM_BUILD_DIR=/opt/llvm
```

然后确认：

```bash
"$LLVM_BUILD_DIR/bin/llvm-config" --version
ls "$LLVM_BUILD_DIR/lib/cmake/llvm"
ls "$LLVM_BUILD_DIR/lib/cmake/mlir"
```

如果 `llvm-config --version` 不是 `19.1.7`，不建议继续构建 PTOAS。

### 4.6 下载 PTOAS VF CostModel 分支

从 fork 拉取：

```bash
cd "$WORKSPACE_DIR"
git clone https://github.com/wang-chonghao/PTOAS.git
cd "$PTO_SOURCE_DIR"
git fetch origin
git checkout vf-costmodel-phase1
```

如果分支还没有推到远程，先在有代码的机器上执行：

```bash
cd /mnt/e/vfsimulator_structure/PTOAS
git push -u myfork vf-costmodel-phase1
```

确认当前分支：

```bash
git branch --show-current
```

预期：

```text
vf-costmodel-phase1
```

### 4.7 配置 PTOAS CMake

进入 PTOAS：

```bash
cd "$PTO_SOURCE_DIR"
export PYBIND11_CMAKE_DIR=$(python -m pybind11 --cmakedir)
```

配置：

```bash
cmake -G Ninja \
  -S . \
  -B "$PTO_BUILD_DIR" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython3_FIND_STRATEGY=LOCATION \
  -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_PYTHON_PACKAGE_DIR="$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core" \
  -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"
```

如果使用的是安装版 LLVM，例如 `/opt/llvm`，`MLIR_PYTHON_PACKAGE_DIR` 可能是：

```text
$LLVM_BUILD_DIR/python_packages/mlir_core
```

此时配置命令改成：

```bash
cmake -G Ninja \
  -S . \
  -B "$PTO_BUILD_DIR" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython3_FIND_STRATEGY=LOCATION \
  -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_PYTHON_PACKAGE_DIR="$LLVM_BUILD_DIR/python_packages/mlir_core" \
  -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"
```

如何判断用哪个 `MLIR_PYTHON_PACKAGE_DIR`：

```bash
ls "$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core" 2>/dev/null
ls "$LLVM_BUILD_DIR/python_packages/mlir_core" 2>/dev/null
```

哪个存在就用哪个。

### 4.8 编译 PTOAS

只编译 `ptoas`：

```bash
cmake --build "$PTO_BUILD_DIR" --target ptoas -j2
```

也可以完整编译：

```bash
cmake --build "$PTO_BUILD_DIR" -j2
```

新机器建议先只编译 `ptoas`，能更快验证 VF CostModel 链路。

编译产物：

```text
$PTO_BUILD_DIR/tools/ptoas/ptoas
```

验证：

```bash
"$PTO_BUILD_DIR/tools/ptoas/ptoas" --help | grep -E "dump-vf-program|enable-op-fusion"
```

预期能看到：

```text
--dump-vf-program
--dump-vf-program-json=<string>
--enable-op-fusion
```

### 4.9 配置运行时环境

使用 build 目录运行 `ptoas` 时，设置：

```bash
export PATH="$PTO_BUILD_DIR/tools/ptoas:$PTO_BUILD_DIR/tools/ptobc:$PATH"
export LD_LIBRARY_PATH="$LLVM_BUILD_DIR/lib:$PTO_BUILD_DIR/lib:$LD_LIBRARY_PATH"
```

如果需要 Python binding，再补：

```bash
if [ -d "$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core" ]; then
  export MLIR_PYTHON_ROOT="$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core"
else
  export MLIR_PYTHON_ROOT="$LLVM_BUILD_DIR/python_packages/mlir_core"
fi

export PYTHONPATH="$MLIR_PYTHON_ROOT:$PTO_BUILD_DIR/python:$PYTHONPATH"
```

本 VF CostModel 最小验证只需要 CLI，不强制使用 Python binding。

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

### 5.1 新环境 smoke test 顺序

新机器上建议按下面顺序确认，每一步成功后再继续：

```bash
# 1. 确认 LLVM 版本
"$LLVM_BUILD_DIR/bin/llvm-config" --version

# 2. 确认 ptoas 可以启动
"$PTO_BUILD_DIR/tools/ptoas/ptoas" --version

# 3. 确认 VF CostModel 参数已经编进 ptoas
"$PTO_BUILD_DIR/tools/ptoas/ptoas" --help | grep -E "dump-vf-program|enable-op-fusion"

# 4. 跑最小 VF CostModel case
"$PTO_BUILD_DIR/tools/ptoas/ptoas" \
  --pto-arch=a5 \
  --pto-level=level2 \
  --enable-op-fusion \
  --emit-pto-ir \
  --dump-vf-program \
  "$PTO_SOURCE_DIR/test/lit/tile_fusion/min_tadd_tmul.pto" \
  -o /dev/null
```

第 4 步至少应该看到：

```text
[pto-fusion-plan] VF program for fusion group:
[pto-fusion-plan] VF latency cycles=82
```

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

先确认 `LLVM_BUILD_DIR` 是否指向正确位置：

```bash
echo "$LLVM_BUILD_DIR"
"$LLVM_BUILD_DIR/bin/llvm-config" --version
ls "$LLVM_BUILD_DIR/lib/cmake/llvm"
ls "$LLVM_BUILD_DIR/lib/cmake/mlir"
```

如果 `llvm-config --version` 不是 `19.1.7`，需要切换到正确 LLVM。

如果 `lib/cmake/llvm` 或 `lib/cmake/mlir` 不存在，说明当前路径不是可供 PTOAS 使用的 LLVM build/install root。

重新配置 PTOAS 时显式传入：

```bash
cmake -G Ninja \
  -S . \
  -B "$PTO_BUILD_DIR" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython3_FIND_STRATEGY=LOCATION \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_PYTHON_PACKAGE_DIR="$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core" \
  -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"
```

如果是安装版 LLVM，把 `MLIR_PYTHON_PACKAGE_DIR` 改成实际存在的路径。

### 8.4 WSL 编译中途退出

优先降低并行度：

```bash
cmake --build "$PTO_BUILD_DIR" --target ptoas -j2
```

不要直接使用很大的 `-j` 数。

### 8.5 运行时找不到 libMLIR 或 libLLVM

如果执行 `ptoas` 时出现类似：

```text
error while loading shared libraries: libMLIR*.so
```

设置：

```bash
export LD_LIBRARY_PATH="$LLVM_BUILD_DIR/lib:$PTO_BUILD_DIR/lib:$LD_LIBRARY_PATH"
```

然后重试：

```bash
"$PTO_BUILD_DIR/tools/ptoas/ptoas" --version
```

### 8.6 `git checkout vf-costmodel-phase1` 失败

如果提示分支不存在，先确认远程分支：

```bash
git branch -a | grep vf-costmodel
```

如果远程确实没有该分支，说明本地修改还没有 push 到 fork。需要先在原机器执行：

```bash
cd /mnt/e/vfsimulator_structure/PTOAS
git push -u myfork vf-costmodel-phase1
```

### 8.7 `MLIR_PYTHON_PACKAGE_DIR` 路径不存在

源码 build 版 LLVM 常见路径：

```text
$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
```

安装版 LLVM 常见路径：

```text
$LLVM_BUILD_DIR/python_packages/mlir_core
```

用下面命令确认：

```bash
find "$LLVM_BUILD_DIR" -path '*python_packages/mlir_core' -type d
```

找到实际路径后重新配置 PTOAS。

## 9. 后续计划

后续主要工作：

- 扩展 tileop 到 vector op 的映射覆盖范围。
- 支持更多 pattern，例如 cast、rowexpand、rowreduce 等。
- 接入更多 metadata，例如 `loop_id`、`loop_unroll`。
- 引入 UB 容量约束，在容量允许时尽量融合。
- 加入 unroll 寻优和 loop 结构优化。
- 继续对齐 C++ VfSimulator 和 Python VfSimulator 的回归测试。
