# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Lightweight wrappers around the low-level MLIR Python bindings for the PTO
dialect.  The goal is to eliminate boilerplate so that a vPTO kernel body can
be written in plain-looking Python without manual InsertionPoint management,
verbose type constructors, or raw Operation.create() calls.

Design rules
────────────
• Every helper is a plain function or a contextlib.contextmanager – no classes.
• All helpers work with the *current* MLIR context / location / insertion-point
  (set by `pto_context` and `vpto_kernel`); no context parameter is threaded.
• The module is self-contained: only mlir.* imports are allowed.
"""

from contextlib import contextmanager

from mlir.ir import (
    Attribute,
    Context,
    IntegerType,
    IndexType,
    InsertionPoint,
    Location,
    Module,
    Operation,
    StringAttr,
    Type,
)
from mlir.dialects import arith, func, pto, scf


# ─── Type constructors ────────────────────────────────────────────────────────

def i32_type():
    """Signless 32-bit integer type."""
    return IntegerType.get_signless(32)


def i64_type():
    """Signless 64-bit integer type."""
    return IntegerType.get_signless(64)


def idx_type():
    """MLIR index type."""
    return IndexType.get()


def ptr_type(elem_type, space="ub"):
    """PTO pointer type: !pto.ptr<{elem_type}, {space}>."""
    return Type.parse(f"!pto.ptr<{elem_type}, {space}>")


def vreg_type(lanes, elem_type):
    """PTO vector-register type: !pto.vreg<{lanes}x{elem_type}>."""
    return Type.parse(f"!pto.vreg<{lanes}x{elem_type}>")


def mask_type(bits="b32"):
    """PTO mask/predicate type: !pto.mask<{bits}>  (b8 | b16 | b32)."""
    return Type.parse(f"!pto.mask<{bits}>")


# ─── Constant builders ───────────────────────────────────────────────────────

def c_idx(value):
    """Emit an index constant."""
    return arith.ConstantOp(IndexType.get(), value).result


def c_i32(value):
    """Emit a 32-bit integer constant."""
    return arith.ConstantOp(IntegerType.get_signless(32), value).result


def c_i64(value):
    """Emit a 64-bit integer constant."""
    return arith.ConstantOp(IntegerType.get_signless(64), value).result


# ─── Arithmetic shorthands ───────────────────────────────────────────────────

def muli(lhs, rhs):
    """arith.muli"""
    return arith.MulIOp(lhs, rhs).result


def addi(lhs, rhs):
    """arith.addi"""
    return arith.AddIOp(lhs, rhs).result


def subi(lhs, rhs):
    """arith.subi"""
    return arith.SubIOp(lhs, rhs).result


# ─── PTO vector / pointer operations ────────────────────────────────────────

def castptr(int_addr, result_ptr_type):
    """Cast an integer address to a typed PTO pointer (pto.castptr)."""
    return pto.CastPtrOp(result_ptr_type, int_addr).result


def addptr(base_ptr, index_offset):
    """Advance a PTO pointer by an index offset (pto.addptr)."""
    return pto.AddPtrOp(base_ptr, index_offset).result


def vlds(src_ptr, offset, result_vreg_type):
    """Vector load from a PTO pointer at *offset* (pto.vlds)."""
    return pto.VldsOp(result_vreg_type, src_ptr, offset).result


def vadd(lhs, rhs, mask, result_vreg_type):
    """Element-wise vector add under a predicate mask (pto.vadd)."""
    return pto.VaddOp(result_vreg_type, lhs, rhs, mask).result


def vsts(val, dst_ptr, offset, mask):
    """Vector store to a PTO pointer at *offset* under a mask (pto.vsts)."""
    pto.VstsOp(val, dst_ptr, offset, mask)


def plt_b32(scalar):
    """
    Predicate-load from a 32-bit scalar value (pto.plt_b32).

    Returns (mask_value, scalar_out) – the mask is typically the only value
    used downstream; scalar_out can be discarded with ``_``.
    """
    plt_op = pto.PltB32Op(mask_type("b32"), i32_type(), scalar)
    return plt_op.mask, plt_op.scalar_out


# ─── Scope context managers ──────────────────────────────────────────────────

@contextmanager
def vecscope():
    """
    Emit a ``pto.vecscope { ... }`` region.

    Usage::

        with vecscope():
            ptr = castptr(addr, ptr_f32)
            ...
    """
    op = pto.VecScopeOp()
    block = op.body.blocks.append()
    with InsertionPoint(block):
        yield


@contextmanager
def for_range(start, stop, step):
    """
    Emit an ``scf.for`` loop; yield the induction variable.
    The mandatory ``scf.yield`` terminator is inserted automatically on exit.

    Usage::

        with for_range(c0, c16, c1) as i:
            off = muli(i, c64)
            ...
    """
    for_op = scf.ForOp(start, stop, step)
    with InsertionPoint(for_op.body):
        yield for_op.induction_variable
        scf.YieldOp([])


# ─── Top-level module / kernel builder ───────────────────────────────────────

@contextmanager
def pto_context():
    """
    Activate an MLIR context with the PTO dialect registered.
    Must wrap all other utility calls.

    Usage::

        with pto_context():
            f32 = F32Type.get()
            with vpto_kernel("MyKernel", arch="a5") as mod:
                ...
    """
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown():
            yield ctx


@contextmanager
def vpto_kernel(func_name, *, arch="a5"):
    """
    Build the standard two-level nested-module + no-arg ``func.func`` shell
    for a vPTO vector kernel, then yield the outer ``Module`` as the context
    variable.  ``func.ReturnOp`` and ``module.verify()`` are inserted/called
    automatically on context exit.

    The emitted skeleton is::

        module attributes {pto.target_arch = arch} {
          module attributes {pto.kernel_kind = #pto.kernel_kind<vector>,
                             pto.target_arch = arch} {
            func.func @func_name() {
              <your code here>
              return
            }
          }
        }

    Usage::

        with vpto_kernel("TADD", arch="a5") as mod:
            c0 = c_idx(0)
            ...
        return mod
    """
    arch_attr = StringAttr.get(arch)
    kind_attr = Attribute.parse("#pto.kernel_kind<vector>")

    outer_mod = Module.create()
    outer_mod.operation.attributes["pto.target_arch"] = arch_attr

    with InsertionPoint(outer_mod.body):
        # Module.create() ignores the active InsertionPoint, so use
        # Operation.create("builtin.module") to insert the inner module.
        inner_op = Operation.create("builtin.module", regions=1)
        inner_op.attributes["pto.target_arch"] = arch_attr
        inner_op.attributes["pto.kernel_kind"] = kind_attr
        inner_body = inner_op.regions[0].blocks.append()

        with InsertionPoint(inner_body):
            fn = func.FuncOp(func_name, func.FunctionType.get([], []))
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            yield outer_mod
            func.ReturnOp([])

    outer_mod.operation.verify()
