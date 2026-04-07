"""TileLang DSL template for pto.tadds"""

import sys
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tadds",
)
def template_tadds(src: pto.Tile, scalar: pto.AnyType, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            vec = pto.vlds(src[row, col:])
            result = pto.vadds(vec, scalar, mask)
            pto.vsts(result, dst[row, col:], mask)
    return
