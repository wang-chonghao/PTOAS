# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`pto.tstore` 的 TileLang DSL 模板"""

import tilelang_dsl as pto


def _tstore_preconditions(src, dst) -> bool:
    logical_rows = dst.shape[0] * dst.shape[1] * dst.shape[2] * dst.shape[3]
    logical_cols = dst.shape[4]
    return (
        dst.rank == 5
        and dst.strides[4] == 1
        and src.valid_shape[0] == logical_rows
        and src.valid_shape[1] == logical_cols
        and src.valid_shape[0] <= src.shape[0]
        and src.valid_shape[1] <= src.shape[1]
    )


@pto.vkernel(
    target="a5",
    op="pto.tstore",
    advanced=True,
    constraints=[_tstore_preconditions],
)
def template_tstore(src: pto.Tile, dst: pto.PartitionTensorView):
    dtype = src.element_type
    elem_bytes = pto.bytewidth(dtype)

    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides

    valid_rows, valid_cols = src.valid_shape
    ub_rows, ub_cols = src.shape

    # These preconditions are expressed through the descriptor-level constraint
    # callable above, using direct `src.*` / `dst.*` metadata syntax.

    n_burst = g3
    len_burst = valid_cols * elem_bytes
    ub_stride = ub_cols * elem_bytes
    gm_stride = s3 * elem_bytes

    src_stride2 = g3 * ub_cols
    src_stride1 = g2 * src_stride2
    src_stride0 = g1 * src_stride1

    loop1 = g2
    loop2 = g1
    loop1_src_stride = src_stride2 * elem_bytes
    loop1_dst_stride = s2 * elem_bytes
    loop2_src_stride = src_stride1 * elem_bytes
    loop2_dst_stride = s1 * elem_bytes

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()

    if loop1 != 1 or loop2 != 1:
        pto.set_loop2_stride_ubtoout(
            src_stride=loop2_src_stride, dst_stride=loop2_dst_stride
        )
        pto.set_loop1_stride_ubtoout(
            src_stride=loop1_src_stride, dst_stride=loop1_dst_stride
        )
        pto.set_loop_size_ubtoout(loop1=loop1, loop2=loop2)

    for i in range(0, g0, 1):
        src_i = pto.addptr(ub_ptr, i * src_stride0 * elem_bytes)
        dst_i = pto.addptr(gm_ptr, i * s0 * elem_bytes)
        pto.copy_ubuf_to_gm(
            dst=dst_i,
            src=src_i,
            n_burst=n_burst,
            len_burst=len_burst,
            gm_stride=gm_stride,
            ub_stride=ub_stride,
        )

    if loop1 != 1 or loop2 != 1:
        pto.set_loop_size_ubtoout(loop1=1, loop2=1)
    return
