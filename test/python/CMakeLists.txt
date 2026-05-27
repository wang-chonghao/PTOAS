# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

get_filename_component(_llvm_root "${LLVM_DIR}/../../.." ABSOLUTE)
set(_ptodsl_test_env
  "PYTHONPATH=${MLIR_PYTHON_PACKAGE_DIR}:${CMAKE_INSTALL_PREFIX}:$ENV{PYTHONPATH}"
  "LD_LIBRARY_PATH=${_llvm_root}/lib:${CMAKE_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}"
  "PATH=${CMAKE_BINARY_DIR}/tools/ptoas:${CMAKE_INSTALL_PREFIX}/bin:$ENV{PATH}"
)

function(add_ptodsl_python_test test_name script_name)
  add_test(
    NAME ${test_name}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${script_name}"
  )
  set_tests_properties(${test_name} PROPERTIES
    LABELS "PTODSL"
    ENVIRONMENT "${_ptodsl_test_env}"
  )
endfunction()

add_ptodsl_python_test(ptodsl_compact_tile_buf_asm compact_tile_buf_asm.py)
add_ptodsl_python_test(ptodsl_low_precision_types low_precision_types.py)
add_ptodsl_python_test(ptodsl_docs_as_test ptodsl_docs_as_test.py)
add_ptodsl_python_test(ptodsl_flash_attention_demo_compile ptodsl_flash_attention_demo_compile.py)
add_ptodsl_python_test(ptodsl_jit_compile ptodsl_jit_compile.py)
add_ptodsl_python_test(ptodsl_jit_diagnostics ptodsl_jit_diagnostics.py)
add_ptodsl_python_test(ptodsl_ptoas_frontend_verify ptodsl_ptoas_frontend_verify.py)
add_ptodsl_python_test(ptodsl_subkernel_diagnostics ptodsl_subkernel_diagnostics.py)
