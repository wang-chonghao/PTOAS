# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared authored scalar type-adaptation helpers for PTODSL surface lowering."""

from __future__ import annotations

from ._runtime_scalar_ops import classify_runtime_scalar_type
from ._surface_values import unwrap_surface_value
from ._types import (
    _integer_signedness,
    _materialize_integer_literal,
    _restore_integer_signedness,
    _signless_integer_type,
    _strip_integer_signedness,
)

from mlir.dialects import arith
from mlir.ir import BF16Type, F16Type, F32Type, FloatAttr, IndexType, IntegerType


def coerce_scalar_to_type(value, target_type, *, context: str):
    """Normalize one authored scalar value/literal to *target_type*."""
    raw_value = unwrap_surface_value(value)
    if not hasattr(raw_value, "type"):
        return materialize_scalar_literal(raw_value, target_type, context=context)

    if raw_value.type == target_type:
        return raw_value

    source_kind = classify_runtime_scalar_type(raw_value.type)
    target_kind = classify_runtime_scalar_type(target_type)

    if source_kind == "index" and target_kind == "integer":
        return _coerce_integer_like(raw_value, target_type)
    if source_kind == "integer" and target_kind == "index":
        return arith.IndexCastOp(target_type, _strip_integer_signedness(raw_value)).result
    if source_kind == "integer" and target_kind == "integer":
        return _coerce_integer_like(raw_value, target_type)
    if source_kind == "float" and target_kind == "float":
        return _coerce_float_like(raw_value, target_type)

    raise TypeError(
        f"{context} cannot coerce the authored value to the expected scalar type: "
        f"got {raw_value.type}, expected {target_type}"
    )


def materialize_scalar_literal(value, target_type, *, context: str):
    """Materialize one Python literal as an MLIR scalar constant of *target_type*."""
    if isinstance(value, bool):
        raise TypeError(f"{context} does not accept bool literals")

    target_kind = classify_runtime_scalar_type(target_type)
    if target_kind == "float":
        return arith.ConstantOp(target_type, FloatAttr.get(target_type, float(value))).result
    if target_kind == "index":
        return arith.ConstantOp(target_type, int(value)).result

    if isinstance(value, float):
        raise TypeError(
            f"{context} cannot materialize a floating-point literal against non-floating "
            f"target type {target_type}"
        )

    return _materialize_integer_literal(target_type, value)


def _coerce_integer_like(raw_value, target_type):
    if IndexType.isinstance(raw_value.type):
        signless_target = _signless_integer_type(target_type)
        adapted = arith.IndexCastOp(signless_target, raw_value).result
        return _restore_integer_signedness(adapted, target_type)

    source_type = raw_value.type
    source_width = IntegerType(source_type).width
    target_width = IntegerType(target_type).width
    signless_source = _strip_integer_signedness(raw_value)
    signless_target = _signless_integer_type(target_type)

    if source_width < target_width:
        source_signedness = _integer_signedness(source_type)
        if source_signedness == "unsigned":
            widened = arith.ExtUIOp(signless_target, signless_source).result
        else:
            widened = arith.ExtSIOp(signless_target, signless_source).result
        return _restore_integer_signedness(widened, target_type)
    if source_width > target_width:
        truncated = arith.TruncIOp(signless_target, signless_source).result
        return _restore_integer_signedness(truncated, target_type)
    return _restore_integer_signedness(signless_source, target_type)


def _coerce_float_like(raw_value, target_type):
    source_width = _float_bytewidth(raw_value.type)
    target_width = _float_bytewidth(target_type)
    if source_width < target_width:
        return arith.ExtFOp(target_type, raw_value).result
    if source_width > target_width:
        return arith.TruncFOp(target_type, raw_value).result
    return raw_value


def _float_bytewidth(type_obj):
    if BF16Type.isinstance(type_obj) or F16Type.isinstance(type_obj):
        return 2
    if F32Type.isinstance(type_obj):
        return 4
    raise TypeError(f"unsupported floating-point type {type_obj}")


__all__ = [
    "coerce_scalar_to_type",
    "materialize_scalar_literal",
]
