# VF CostModel A5 E2E 验证流程

本文记录当前分支上 `tadd -> tadd -> tadd` 例子从 PTOAS 编译到 A5 camodel 运行的端到端验证流程。

## 1. 验证目标

验证链路：

```text
.pto
  -> PTOAS 生成 A5 CCE C++
  -> npu_validation 生成 host runner
  -> BiSheng 编译 kernel so
  -> msprof op simulator / camodel 执行
  -> 生成 dump 日志
  -> 输出结果 compare passed
```

目标平台：

```text
SOC_VERSION=Ascend950PR_9599
AICORE_ARCH=dav-c310-vec
PTOAS arch=a5
```

## 2. 关键原则

A5 camodel 验证时使用干净环境：

- 在 `/home/<user>/...` 下创建运行目录，不建议从 `/tmp` 或 `/mnt/*` 直接跑。
- 不手动设置 `LD_PRELOAD`。
- 不使用旧 shim。
- 不混用多套 CANN 路径。
- 使用 `msprof op simulator` 跑 simulator，不直接运行 native sim runner。
- `LD_LIBRARY_PATH` 中显式包含同一套 CANN 的 runtime、simulator 和 camodel 目录。

本次使用：

```bash
ASC=/home/lenovo/Ascend/ascend-toolkit/cann-9.0.0-beta.1
ARCH=x86_64-linux
PTO_ISA_ROOT=/mnt/e/PTO/PTOISA
RUN=/home/lenovo/ptoas_a5_run_tadd3
```

## 3. 生成 A5 CCE C++

示例 `.pto`：

```text
/tmp/ptoas_a5_e2e_probe/tadd3_32x32_e2e.pto
```

PTOAS 命令：

```bash
/mnt/e/vfsimulator_structure/PTOAS/build-vf-costmodel/tools/ptoas/ptoas \
  /tmp/ptoas_a5_e2e_probe/tadd3_32x32_e2e.pto \
  --pto-level=level2 \
  --pto-arch=a5 \
  --enable-op-fusion \
  --use-vfsim-fusion-planner \
  --enable-insert-sync \
  -o /tmp/ptoas_a5_e2e_probe/tadd3_32x32_sync_fusion.a5.cpp
```

注意：

- `--enable-insert-sync` 需要打开；否则 BiSheng 后端可能报 `Do not know how to split this operator's operand!`。

## 4. 生成 npu_validation testcase

```bash
python3 /mnt/e/vfsimulator_structure/PTOAS/test/npu_validation/scripts/generate_testcase.py \
  --input /tmp/ptoas_a5_e2e_probe/tadd3_32x32_sync_fusion.a5.cpp \
  --testcase tadd3_32x32_sync_fusion \
  --output-root /tmp/ptoas_a5_e2e_probe/npu_validation \
  --run-mode sim \
  --soc-version Ascend950PR_9599 \
  --aicore-arch dav-c310-vec
```

生成目录：

```text
/tmp/ptoas_a5_e2e_probe/npu_validation/ptoas_a5_e2e_probe/tadd3_32x32_sync_fusion
```

## 5. 准备干净运行目录

把 testcase 移到 `/home` 下运行：

```bash
SRC=/tmp/ptoas_a5_e2e_probe/npu_validation/ptoas_a5_e2e_probe/tadd3_32x32_sync_fusion
RUN=/home/lenovo/ptoas_a5_run_tadd3
ASC=/home/lenovo/Ascend/ascend-toolkit/cann-9.0.0-beta.1

rm -rf "$RUN"
mkdir -p "$RUN"
cp -a "$SRC"/. "$RUN"/
mkdir -p "$RUN/etc"
cp -f "$ASC/x86_64-linux/simulator/dav_3510/lib/1982_cloud_config.toml" \
  "$RUN/etc/1982_cloud_config.toml"
```

生成输入：

```bash
cd "$RUN"
python3 golden.py
```

## 6. 重新构建 sim runner

复制 testcase 后需要在 `/home` 目录重新构建，避免 runner 仍链接到 `/tmp` 下旧的 kernel so。

```bash
cd "$RUN"
rm -rf build

export ASCEND_HOME_PATH="$ASC"
export PATH="$ASC/bin:$PATH"
export LD_LIBRARY_PATH="$ASC/lib64:${LD_LIBRARY_PATH:-}"

cmake -S . -B build \
  -DSOC_VERSION=Ascend950PR_9599 \
  -DPTO_ISA_ROOT=/mnt/e/PTO/PTOISA

cmake --build build --target tadd3_32x32_sync_fusion_sim -j 2
```

检查 runner 是否链接到当前运行目录的 kernel so：

```bash
ldd build/tadd3_32x32_sync_fusion_sim | grep libtadd3
```

期望包含：

```text
/home/lenovo/ptoas_a5_run_tadd3/build/libtadd3_32x32_sync_fusion_kernel.so
```

## 7. 运行 msprof op simulator

```bash
RUN=/home/lenovo/ptoas_a5_run_tadd3
ASC=/home/lenovo/Ascend/ascend-toolkit/cann-9.0.0-beta.1
ARCH=x86_64-linux
OUTDIR="$RUN/msprof_clean"

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
cd "$RUN"

unset LD_PRELOAD
unset CAMODEL_CONFIG_DIR
unset CAMODEL_CONFIG_PATH

export ASCEND_HOME_PATH="$ASC"
export ASCEND_TOOLKIT_HOME="$ASC"
export SOC_VERSION=Ascend950PR_9599
export RUN_MODE=sim
export LD_LIBRARY_PATH="$ASC/$ARCH/fwkacllib/lib64:$ASC/runtime/lib64:$ASC/$ARCH/lib64/device/lib64:$ASC/$ARCH/lib64:$ASC/$ARCH/devlib/device:$ASC/$ARCH/devlib:$ASC/$ARCH/simulator/dav_3510/camodel:$ASC/$ARCH/simulator/dav_3510/lib:$ASC/$ARCH/simulator/Ascend950PR_9599/lib:$ASC/lib64:$RUN/build"

"$ASC/bin/msprof" op simulator \
  --application="$RUN/build/tadd3_32x32_sync_fusion_sim" \
  --kernel-name=tadd3_32x32_e2e \
  --launch-count=1 \
  --soc-version=Ascend950PR_9599 \
  --timeout=120 \
  --output="$OUTDIR/out" \
  > "$OUTDIR/msprof_collect.log" 2>&1
```

## 8. 成功判据

查看：

```bash
sed -n '1,220p' "$RUN/msprof_clean/msprof_collect.log"
```

成功日志包含：

```text
PEM MODEL Init Success
[block_start]           : AIV, task_id=0, core_id=0, block_id=0
[block_end]             : AIV, task_id=0, core_id=0, block_id=0
Total tick: 3560
Profiling running finished. All task success.
Core operator results run in simulator as follow:
core0.veccore0      1.26                1.26
```

## 9. 输出校验

当前 `generate_testcase.py` 生成的 `golden.py` 只初始化输出文件，不会自动生成该例子的真实 golden。

本例语义是：

```text
v5 = v1 + v2 + v3 + v4
```

手动生成 `golden_v5.bin`：

```bash
cd "$RUN"
python3 - <<'PY'
import numpy as np
v1 = np.fromfile("v1.bin", dtype=np.float32)
v2 = np.fromfile("v2.bin", dtype=np.float32)
v3 = np.fromfile("v3.bin", dtype=np.float32)
v4 = np.fromfile("v4.bin", dtype=np.float32)
(v1 + v2 + v3 + v4).astype(np.float32).tofile("golden_v5.bin")
PY

COMPARE_STRICT=1 python3 compare.py
```

成功输出：

```text
[INFO] compare passed
```

## 10. Dump 日志位置

本次 profiling 输出目录：

```text
/home/lenovo/ptoas_a5_run_tadd3/msprof_clean/out/OPPROF_20260627153924_IAIKHNYIPKPFIBMM
```

主日志：

```text
/home/lenovo/ptoas_a5_run_tadd3/msprof_clean/msprof_collect.log
```

指令 popped 日志：

```text
/home/lenovo/ptoas_a5_run_tadd3/msprof_clean/out/OPPROF_20260627153924_IAIKHNYIPKPFIBMM/dump/core0.veccore0.instr_popped_log.dump
```

常用 dump：

```text
dump/core0.veccore0.instr_log.dump
dump/core0.veccore0.instr_popped_log.dump
dump/core0.veccore0.rvec.simd.idu.TRACE.dump
```

## 11. 已排除的错误路径

以下做法容易导致 camodel 启动失败：

- 直接运行 `build/*_sim`，不通过 `msprof op simulator`。
- 手动维护长 `LD_PRELOAD` 列表。
- 使用旧 shim 绕过 camodel 动态库问题。
- 在 `/tmp` 或 `/mnt/*` 下直接启动 profiling。
- 复制 testcase 后不重新 CMake 构建，导致 runner 仍链接旧目录下的 kernel so。
- 混入其他 CANN 版本的 runtime 或 simulator 库。

典型错误包括：

```text
libCuberWrapper.so: undefined symbol: core_wrapper::get_core()
libpem_davinci.so: undefined symbol: hard_code_toml_cfg_map
libstars_wrapper.so: undefined symbol: STARS_TOP::ext_write_ffts_plus_context
TMultiRing.cpp: Assertion `insert.second' failed
Config file is invalid. Path: ./etc/1982_cloud_config.toml
```

这些问题的最终处理方式不是继续补 `LD_PRELOAD`，而是回到干净 `/home` 运行目录、同一套 CANN `LD_LIBRARY_PATH`、`msprof op simulator`。
