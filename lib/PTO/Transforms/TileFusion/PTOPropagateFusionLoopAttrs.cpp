// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOPROPAGATEFUSIONLOOPATTRS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr =
    "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";
static constexpr llvm::StringLiteral kFusionUnrollAttr = "pto.fusion.unroll";
static constexpr llvm::StringLiteral kFusionLoopIndexAttr =
    "pto.fusion.loop_index";
static constexpr llvm::StringLiteral kFusionLoopUnrollAttr =
    "pto.fusion.loop_unroll";

static scf::ForOp findUnrollTargetLoop(scf::ForOp root) {
  scf::ForOp current = root;
  scf::ForOp candidate;
  while (current) {
    if (current.getInitArgs().empty())
      candidate = current;

    scf::ForOp child;
    for (Operation &op : current.getBody()->without_terminator()) {
      auto nested = dyn_cast<scf::ForOp>(op);
      if (!nested)
        continue;
      if (child)
        return {};
      child = nested;
    }
    if (!child)
      return candidate;
    current = child;
  }
  return candidate;
}

static bool propagateLoopAttrs(scf::ForOp loop, Attribute fallbackGroupId,
                               IntegerAttr fallbackOrder, Builder &builder) {
  Attribute unrollAttr = loop->getAttr(kFusionUnrollAttr);
  if (!unrollAttr)
    return false;

  scf::ForOp target = findUnrollTargetLoop(loop);
  if (!target)
    return false;

  if (Attribute groupId = loop->getAttr(kFusionGroupIdAttr))
    target->setAttr(kFusionGroupIdAttr, groupId);
  else if (fallbackGroupId)
    target->setAttr(kFusionGroupIdAttr, fallbackGroupId);

  if (Attribute order = loop->getAttr(kFusionOrderAttr))
    target->setAttr(kFusionOrderAttr, order);
  else if (fallbackOrder)
    target->setAttr(kFusionOrderAttr, fallbackOrder);

  target->setAttr(kFusionLoopIndexAttr, builder.getI64IntegerAttr(0));
  target->setAttr(kFusionLoopUnrollAttr, unrollAttr);
  return true;
}

struct PTOPropagateFusionLoopAttrsPass
    : public pto::impl::PTOPropagateFusionLoopAttrsBase<
          PTOPropagateFusionLoopAttrsPass> {
  using pto::impl::PTOPropagateFusionLoopAttrsBase<
      PTOPropagateFusionLoopAttrsPass>::PTOPropagateFusionLoopAttrsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    Builder builder(ctx);

    module.walk([&](pto::FusionRegionOp region) {
      Attribute groupId = region->getAttr(kFusionGroupIdAttr);
      int64_t order = 0;
      for (Operation &op : region.getBody().front().without_terminator()) {
        auto loop = dyn_cast<scf::ForOp>(op);
        if (!loop)
          continue;
        if (propagateLoopAttrs(loop, groupId, builder.getI64IntegerAttr(order),
                               builder))
          ++order;
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOPropagateFusionLoopAttrsPass() {
  return std::make_unique<PTOPropagateFusionLoopAttrsPass>();
}
