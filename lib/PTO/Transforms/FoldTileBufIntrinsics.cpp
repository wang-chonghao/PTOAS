// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- FoldTileBufIntrinsics.cpp ------------------------------------------===//
//
// After TileLang DSL template functions are inlined, the IR contains:
//   - pto.tile_buf_addr   → extract memref address from tile_buf
//   - pto.tile_valid_rows → extract valid row count
//   - pto.tile_valid_cols → extract valid column count
//
// This pass resolves them against the concrete tile_buf values at the
// call site.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace pto {
  #define GEN_PASS_DEF_FOLDTILEBUFINTRINSICS
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

/// Locate the `pto.bind_tile` op that produced `tileBuf`, expecting the
/// strict pattern emitted by MemrefToTileBuf:
///
///   %bound  = pto.bind_tile %src, %vrow, %vcol : memref -> memref
///   %tile   = builtin.unrealized_conversion_cast %bound : memref -> !pto.tile_buf
///
/// Returns nullptr (with an error emitted on `loc`) if the pattern does not
/// hold — the caller is expected to signal pass failure.
static pto::BindTileOp findBindTileForTileBuf(Value tileBuf, Operation *user) {
  auto cast = tileBuf.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getNumOperands() != 1) {
    user->emitError(
        "FoldTileBufIntrinsics: expected tile_buf to be defined by a "
        "single-operand builtin.unrealized_conversion_cast");
    return nullptr;
  }
  auto bindOp = cast.getOperand(0).getDefiningOp<pto::BindTileOp>();
  if (!bindOp) {
    user->emitError(
        "FoldTileBufIntrinsics: expected unrealized_conversion_cast operand "
        "to be defined by pto.bind_tile");
    return nullptr;
  }
  return bindOp;
}

struct FoldTileBufIntrinsicsPass
    : public pto::impl::FoldTileBufIntrinsicsBase<FoldTileBufIntrinsicsPass> {
  using FoldTileBufIntrinsicsBase::FoldTileBufIntrinsicsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // Leftover TileLang template instances (private, uncalled after
    // PTOInlineLibCall) still contain pto.tile_buf_addr / tile_valid_*
    // ops on tile_buf function arguments — they have no bind_tile to
    // fold against and will be removed by later DCE.  Skip them.
    if (func->hasAttr("pto.tilelang.instance"))
      return;

    SmallVector<pto::TileBufAddrOp, 8> addrOps;
    SmallVector<pto::TileValidRowsOp, 8> rowsOps;
    SmallVector<pto::TileValidColsOp, 8> colsOps;

    func.walk([&](Operation *op) {
      if (auto addr = dyn_cast<pto::TileBufAddrOp>(op))
        addrOps.push_back(addr);
      else if (auto rows = dyn_cast<pto::TileValidRowsOp>(op))
        rowsOps.push_back(rows);
      else if (auto cols = dyn_cast<pto::TileValidColsOp>(op))
        colsOps.push_back(cols);
    });

    // Fold pto.tile_buf_addr → bind_tile's source memref (the static-layout
    // pto.pointer_cast result), or further to pto.castptr when the requested
    // result type is already !pto.ptr<...>. This bypasses the dynamic-offset
    // memref produced by bind_tile itself, so downstream vlds/vsts
    // canonicalization sees a clean strided<[..],offset:0> layout.
    for (auto addrOp : addrOps) {
      pto::BindTileOp bindOp = findBindTileForTileBuf(addrOp.getSrc(), addrOp);
      if (!bindOp)
        return signalPassFailure();

      Value srcMemref = bindOp.getSource();
      if (!isa<MemRefType>(srcMemref.getType())) {
        addrOp.emitError(
            "FoldTileBufIntrinsics: pto.bind_tile source is not a memref");
        return signalPassFailure();
      }

      if (auto resultMemrefType = dyn_cast<MemRefType>(addrOp.getDst().getType())) {
        // The declared tile_buf_addr result type may differ from the actual
        // bind_tile source layout (e.g. plain shape vs. strided layout) — the
        // downstream vector ops are polymorphic over strided layouts of the
        // same element type and shape, so retype the result in place.
        if (srcMemref.getType() != resultMemrefType)
          addrOp.getDst().setType(cast<MemRefType>(srcMemref.getType()));
        addrOp.getDst().replaceAllUsesWith(srcMemref);
        addrOp.erase();
        continue;
      }

      auto resultPtrType = dyn_cast<pto::PtrType>(addrOp.getDst().getType());
      if (!resultPtrType) {
        addrOp.emitError(
            "FoldTileBufIntrinsics: tile_buf_addr result must be memref or !pto.ptr");
        return signalPassFailure();
      }

      builder.setInsertionPoint(addrOp);
      Value replacement =
          builder.create<pto::CastPtrOp>(addrOp.getLoc(), resultPtrType, srcMemref);
      addrOp.getDst().replaceAllUsesWith(replacement);
      addrOp.erase();
    }

    // Fold pto.tile_valid_rows → arith.constant (static) or bind_tile's
    // valid_row operand (dynamic).
    for (auto rowsOp : rowsOps) {
      builder.setInsertionPoint(rowsOp);
      auto tbTy = dyn_cast<pto::TileBufType>(rowsOp.getSrc().getType());
      if (!tbTy || tbTy.getValidShape().empty()) {
        rowsOp.emitError("tile_valid_rows: invalid tile_buf type");
        return signalPassFailure();
      }

      int64_t vRow = tbTy.getValidShape()[0];
      Value replacement;
      if (vRow != ShapedType::kDynamic) {
        replacement =
            builder.create<arith::ConstantIndexOp>(rowsOp.getLoc(), vRow);
      } else {
        pto::BindTileOp bindOp =
            findBindTileForTileBuf(rowsOp.getSrc(), rowsOp);
        if (!bindOp)
          return signalPassFailure();
        replacement = bindOp.getValidRow();
        if (!replacement) {
          rowsOp.emitError(
              "tile_valid_rows: dynamic v_row but bind_tile has no "
              "valid_row operand");
          return signalPassFailure();
        }
        // bind_tile's valid_row is `index` (matches tile_valid_rows result),
        // so no type adaptation is required.
        assert(replacement.getType() == rowsOp.getResult().getType() &&
               "tile_valid_rows fold: type mismatch with bind_tile valid_row");
      }
      rowsOp.getResult().replaceAllUsesWith(replacement);
      rowsOp.erase();
    }

    // Fold pto.tile_valid_cols → arith.constant (static) or bind_tile's
    // valid_col operand (dynamic).
    for (auto colsOp : colsOps) {
      builder.setInsertionPoint(colsOp);
      auto tbTy = dyn_cast<pto::TileBufType>(colsOp.getSrc().getType());
      if (!tbTy || tbTy.getValidShape().size() < 2) {
        colsOp.emitError("tile_valid_cols: invalid tile_buf type");
        return signalPassFailure();
      }

      int64_t vCol = tbTy.getValidShape()[1];
      Value replacement;
      if (vCol != ShapedType::kDynamic) {
        replacement =
            builder.create<arith::ConstantIndexOp>(colsOp.getLoc(), vCol);
      } else {
        pto::BindTileOp bindOp =
            findBindTileForTileBuf(colsOp.getSrc(), colsOp);
        if (!bindOp)
          return signalPassFailure();
        replacement = bindOp.getValidCol();
        if (!replacement) {
          colsOp.emitError(
              "tile_valid_cols: dynamic v_col but bind_tile has no "
              "valid_col operand");
          return signalPassFailure();
        }
        assert(replacement.getType() == colsOp.getResult().getType() &&
               "tile_valid_cols fold: type mismatch with bind_tile valid_col");
      }
      colsOp.getResult().replaceAllUsesWith(replacement);
      colsOp.erase();
    }
  }
};

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createFoldTileBufIntrinsicsPass() {
  return std::make_unique<FoldTileBufIntrinsicsPass>();
}

} // namespace pto
} // namespace mlir
