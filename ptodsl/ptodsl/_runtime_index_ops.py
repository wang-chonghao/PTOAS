# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tracing-time helpers for coercing authored runtime values to MLIR index."""

from __future__ import annotations

from mlir.dialects import arith
from mlir.ir import IndexType, IntegerType


def coerce_runtime_index(value, *, context: str):
    """Normalize one authored loop/slice bound to an MLIR index SSA value."""
    if isinstance(value, bool):
        raise TypeError(f"{context} does not accept bool values")

    if isinstance(value, int):
        return arith.ConstantOp(IndexType.get(), value).result

    if not hasattr(value, "type"):
        raise TypeError(
            f"{context} expects a Python int, an index value, or an integer runtime scalar; "
            f"got {value!r}"
        )

    value_type = value.type
    if IndexType.isinstance(value_type):
        return value
    if IntegerType.isinstance(value_type):
        return arith.IndexCastOp(IndexType.get(), value).result

    raise TypeError(
        f"{context} expects an index or integer runtime scalar, got {value_type}"
    )


__all__ = [
    "coerce_runtime_index",
]
