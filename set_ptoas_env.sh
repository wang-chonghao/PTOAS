#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# after `quick_install.sh`, run `source set_ptoas_env.sh` in a new shell to find the lib
export PTO_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTO_INSTALL_DIR="${PTO_INSTALL_DIR:-${PTO_SOURCE_DIR}/install}"
PTOAS_ENV_PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
LLVM_RUNTIME_ROOT="${LLVM_BUILD_DIR:-${LLVM_DIR:-}}"
export PATH="${PTO_SOURCE_DIR}/build/tools/ptoas:${PATH}"
if [[ -n "${LLVM_RUNTIME_ROOT}" ]]; then
  export LD_LIBRARY_PATH="${LLVM_RUNTIME_ROOT}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -z "${PTOAS_ENV_PYTHON_BIN}" ]]; then
  echo "error: unable to locate python3 or python for set_ptoas_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

PTOAS_ENV_TMP="${PTO_SOURCE_DIR}/tmp/set_ptoas_env"
mkdir -p "${PTOAS_ENV_TMP}/MatMul" "${PTOAS_ENV_TMP}/Abs"
(cd "${PTO_SOURCE_DIR}/test/samples/MatMul" && "${PTOAS_ENV_PYTHON_BIN}" ./tmatmulk.py > "${PTOAS_ENV_TMP}/MatMul/tmatmulk.pto" && ptoas "${PTOAS_ENV_TMP}/MatMul/tmatmulk.pto" -o "${PTOAS_ENV_TMP}/MatMul/tmatmulk.cpp")
(cd "${PTO_SOURCE_DIR}/test/samples/Abs" && "${PTOAS_ENV_PYTHON_BIN}" ./abs.py > "${PTOAS_ENV_TMP}/Abs/abs.pto" && ptoas --enable-insert-sync "${PTOAS_ENV_TMP}/Abs/abs.pto" -o "${PTOAS_ENV_TMP}/Abs/abs.cpp")

echo "test set_env: OK"
