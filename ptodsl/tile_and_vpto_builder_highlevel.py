# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
High-level builder for the TADD vPTO kernel.

Reconstructs the same IR as expand_tileop_to_vpto_result.pto using the
thin wrappers in ptodsl_utils instead of raw MLIR Python binding calls.

Compare with tile_and_vpto_builder_lowlevel.py to see what the utils hide:
  • No manual InsertionPoint management
  • No Operation.create("builtin.module", ...) boilerplate
  • No Type.parse() / arith.ConstantOp(...).result calls in the kernel body
  • vecscope and scf.for become ordinary Python context managers
"""

from mlir.ir import F32Type

from ptodsl_utils import (
    # types
    ptr_type, vreg_type,
    # constants
    c_idx, c_i32, c_i64,
    # arithmetic
    muli,
    # vector / pointer ops
    castptr, addptr, vlds, vadd, vsts, plt_b32,
    # scope helpers
    vecscope, for_range,
    # module builders
    pto_context, vpto_kernel,
)


def build():
    with pto_context():
        # ── Types used in this kernel ─────────────────────────────────────
        f32 = F32Type.get()
        ptr_f32_ub  = ptr_type(f32, "ub")    # !pto.ptr<f32, ub>
        vreg_64f32  = vreg_type(64, f32)     # !pto.vreg<64xf32>

        # ── Build the nested module shell and the @TADD function body ─────
        with vpto_kernel("TADD", arch="a5") as mod:

            # Integer-address constants for the two input buffers
            c0_i64    = c_i64(0)
            c4096_i64 = c_i64(4096)

            # Loop-control constants
            c0  = c_idx(0)
            c1  = c_idx(1)
            c16 = c_idx(16)   # 1024-element array / 64-wide vreg = 16 tiles

            # Scalar used to generate the per-iteration mask
            c64_i32 = c_i32(64)
            c64     = c_idx(64)

            with vecscope():
                # Materialise typed pointers from the raw integer addresses
                ptr_src = castptr(c4096_i64, ptr_f32_ub)  # source buffer
                ptr_dst = castptr(c0_i64,    ptr_f32_ub)  # destination buffer

                with for_range(c0, c16, c1) as tile_idx:
                    # Build a 64-lane all-true mask for this iteration
                    mask, _ = plt_b32(c64_i32)

                    # Byte offset for the current 64-element tile
                    tile_off = muli(tile_idx, c64)

                    # Load source tile, add to destination tile, store result
                    va = vlds(addptr(ptr_src, tile_off), c0, vreg_64f32)
                    ptr_dst_tile = addptr(ptr_dst, tile_off)
                    vb = vlds(ptr_dst_tile, c0, vreg_64f32)
                    vc = vadd(va, vb, mask, vreg_64f32)
                    vsts(vc, ptr_dst_tile, c0, mask)

        return mod


if __name__ == "__main__":
    print(build())
