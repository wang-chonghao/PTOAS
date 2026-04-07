# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`pto.tload` 的 TileLang DSL 模板"""

import tilelang_dsl as pto


def _tload_preconditions(src, dst) -> bool:
    logical_rows = src.shape[0] * src.shape[1] * src.shape[2] * src.shape[3]
    logical_cols = src.shape[4]
    return (
        src.rank == 5
        and src.strides[4] == 1
        and dst.valid_shape[0] <= logical_rows
        and dst.valid_shape[1] <= logical_cols
        and logical_rows <= dst.shape[0]
        and logical_cols <= dst.shape[1]
        and dst.valid_shape[0] <= dst.shape[0]
        and dst.valid_shape[1] <= dst.shape[1]
    )


@pto.vkernel(
    target="a5",
    op="pto.tload",
    advanced=True,
    constraints=[_tload_preconditions],
)
def template_tload(src: pto.PartitionTensorView, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    g0, g1, g2, g3, g4 = src.shape
    s0, s1, s2, s3, s4 = src.strides

    valid_rows, valid_cols = dst.valid_shape
    ub_rows, ub_cols = dst.shape

    # These preconditions are expressed through the descriptor-level constraint
    # callable above, using direct `src.shape[i]` / `dst.shape[i]` syntax.

    n_burst = g3
    len_burst = g4 * elem_bytes
    gm_stride = s3 * elem_bytes
    ub_stride = ub_cols * elem_bytes

    dst_stride2 = g3 * ub_cols
    dst_stride1 = g2 * dst_stride2
    dst_stride0 = g1 * dst_stride1

    loop1 = g2
    loop2 = g1
    loop1_src_stride = s2 * elem_bytes
    loop1_dst_stride = dst_stride2 * elem_bytes
    loop2_src_stride = s1 * elem_bytes
    loop2_dst_stride = dst_stride1 * elem_bytes

    gm_ptr = src.as_ptr()
    ub_ptr = dst.as_ptr()

    if loop1 != 1 or loop2 != 1:
        pto.set_loop2_stride_outtoub(
            src_stride=loop2_src_stride, dst_stride=loop2_dst_stride
        )
        pto.set_loop1_stride_outtoub(
            src_stride=loop1_src_stride, dst_stride=loop1_dst_stride
        )
        pto.set_loop_size_outtoub(loop1=loop1, loop2=loop2)

    for i in range(0, g0, 1):
        src_i = pto.addptr(gm_ptr, i * s0 * elem_bytes)
        dst_i = pto.addptr(ub_ptr, i * dst_stride0 * elem_bytes)
        pto.copy_gm_to_ubuf(
            dst=dst_i,
            src=src_i,
            n_burst=n_burst,
            len_burst=len_burst,
            gm_stride=gm_stride,
            ub_stride=ub_stride,
            enable_ub_pad=False,
        )

    if loop1 != 1 or loop2 != 1:
        pto.set_loop_size_outtoub(loop1=1, loop2=1)
    return
