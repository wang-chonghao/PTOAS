// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOInferVPTOVecScope.cpp ------------------------------------------===//
//
// VPTO automatic vecscope inference.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINFERVPTOVECSCOPE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

enum class VPTOInferenceOpClass {
  Vector,
  SafeScalar,
  Boundary,
};

struct NestedRegionSummary {
  bool hasVectorOperation = false;
  bool hasBoundaryOperation = false;
};

struct EscapingVectorScopeValue {
  Value value;
  Operation *producer = nullptr;
  Operation *user = nullptr;
};

static VPTOInferenceOpClass classifyOperationForInference(Operation *op);
static LogicalResult inferVecScopesInRegion(Region &region,
                                            MLIRContext *context);

static bool isVecScopeType(Type type) {
  return isa<pto::VRegType, pto::MaskType, pto::AlignType>(type);
}

static bool isPTOOperation(Operation *op) {
  return op && op->getName().getStringRef().starts_with("pto.");
}

static bool isExplicitVectorScopeCarrier(Operation *op) {
  return isa<pto::VecScopeOp, pto::StrictVecScopeOp>(op);
}

static bool isForbiddenInsideInferredVectorScope(Operation *op) {
  return isa<pto::VbitsortOp, pto::Vmrgsort4Op>(op);
}

static bool isVectorScopeBoundaryOperation(Operation *op) {
  return isa<pto::BarrierOp, pto::BarrierSyncOp>(op);
}

static bool hasVecScopeTypedOperandOrResult(Operation *op) {
  for (Type type : op->getOperandTypes()) {
    if (isVecScopeType(type))
      return true;
  }
  for (Type type : op->getResultTypes()) {
    if (isVecScopeType(type))
      return true;
  }
  return false;
}

static bool requiresVectorScope(Operation *op) {
  if (!isPTOOperation(op))
    return false;

  return hasVecScopeTypedOperandOrResult(op) ||
         isa<pto::MemBarOp, pto::SprclrOp>(op);
}

static bool isAtomicControlFlowCandidate(Operation *op) {
  return isa<scf::IfOp, scf::ForOp>(op);
}

static bool isSafeScalarOperation(Operation *op) {
  if (op->getNumRegions() != 0)
    return false;
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isa<func::CallOp>(op))
    return false;
  if (isPTOOperation(op) && !isMemoryEffectFree(op))
    return false;
  return isMemoryEffectFree(op);
}

static void summarizeNestedRegionForAtomicCluster(
    Region &region, NestedRegionSummary &summary) {
  for (Block &block : region) {
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      switch (classifyOperationForInference(&op)) {
      case VPTOInferenceOpClass::Vector:
        summary.hasVectorOperation = true;
        break;
      case VPTOInferenceOpClass::SafeScalar:
        break;
      case VPTOInferenceOpClass::Boundary:
        summary.hasBoundaryOperation = true;
        return;
      }
    }
  }
}

static bool canTreatAsAtomicControlFlow(Operation *op) {
  if (!isAtomicControlFlowCandidate(op))
    return false;

  NestedRegionSummary summary;
  for (Region &region : op->getRegions()) {
    summarizeNestedRegionForAtomicCluster(region, summary);
    if (summary.hasBoundaryOperation)
      return false;
  }
  return summary.hasVectorOperation;
}

static VPTOInferenceOpClass classifyOperationForInference(Operation *op) {
  if (!op)
    return VPTOInferenceOpClass::Boundary;

  if (isExplicitVectorScopeCarrier(op))
    return VPTOInferenceOpClass::Boundary;
  if (op->hasTrait<OpTrait::IsTerminator>())
    return VPTOInferenceOpClass::Boundary;
  if (isa<func::CallOp>(op))
    return VPTOInferenceOpClass::Boundary;
  if (isVectorScopeBoundaryOperation(op))
    return VPTOInferenceOpClass::Boundary;
  if (isForbiddenInsideInferredVectorScope(op))
    return VPTOInferenceOpClass::Boundary;

  if (requiresVectorScope(op))
    return VPTOInferenceOpClass::Vector;

  if (canTreatAsAtomicControlFlow(op))
    return VPTOInferenceOpClass::Vector;

  if (isSafeScalarOperation(op))
    return VPTOInferenceOpClass::SafeScalar;

  return VPTOInferenceOpClass::Boundary;
}

static bool hasVectorOperation(ArrayRef<Operation *> ops) {
  return llvm::any_of(ops, [](Operation *op) {
    return classifyOperationForInference(op) == VPTOInferenceOpClass::Vector;
  });
}

static bool isUserInsideCluster(Operation *user,
                                const llvm::SmallPtrSetImpl<Operation *> &ops) {
  for (Operation *cur = user; cur; cur = cur->getParentOp()) {
    if (ops.contains(cur))
      return true;
  }
  return false;
}

static bool canMoveIntoResultlessScope(ArrayRef<Operation *> ops) {
  llvm::SmallPtrSet<Operation *, 16> opSet;
  for (Operation *op : ops)
    opSet.insert(op);

  for (Operation *op : ops) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!isUserInsideCluster(user, opSet))
          return false;
      }
    }
  }
  return true;
}

static bool
findEscapingVectorScopeResult(ArrayRef<Operation *> ops,
                              EscapingVectorScopeValue &escapingValue) {
  llvm::SmallPtrSet<Operation *, 16> opSet;
  for (Operation *op : ops)
    opSet.insert(op);

  for (Operation *op : ops) {
    for (Value result : op->getResults()) {
      if (!isVecScopeType(result.getType()))
        continue;

      for (Operation *user : result.getUsers()) {
        if (isUserInsideCluster(user, opSet))
          continue;

        escapingValue.value = result;
        escapingValue.producer = op;
        escapingValue.user = user;
        return true;
      }
    }
  }
  return false;
}

static LogicalResult
emitEscapingVectorScopeValueError(const EscapingVectorScopeValue &escapingValue) {
  Operation *producer = escapingValue.producer;
  if (!producer)
    return failure();

  InFlightDiagnostic diag = producer->emitOpError()
                            << "cannot infer resultless pto.vecscope because "
                               "VPTO vector-scope data cannot have external "
                               "users";
  if (escapingValue.value)
    diag << "; escaping value type is " << escapingValue.value.getType();
  if (escapingValue.user)
    diag.attachNote(escapingValue.user->getLoc())
        << "external user is here";
  return failure();
}

static void wrapCluster(ArrayRef<Operation *> ops, MLIRContext *context) {
  if (ops.empty() || !hasVectorOperation(ops))
    return;

  Operation *first = ops.front();
  Operation *last = ops.back();
  Block *parentBlock = first->getBlock();

  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(first);
  auto scope = rewriter.create<pto::VecScopeOp>(first->getLoc());
  scope.getBody().push_back(new Block());

  Block &scopeBody = scope.getBody().front();
  scopeBody.getOperations().splice(scopeBody.end(), parentBlock->getOperations(),
                                   Block::iterator(first),
                                   std::next(Block::iterator(last)));
}

static LogicalResult wrapGreedySubclusters(ArrayRef<Operation *> ops,
                                           MLIRContext *context) {
  for (size_t begin = 0; begin < ops.size();) {
    size_t bestEnd = begin;
    EscapingVectorScopeValue escapingValue;
    bool sawEscapingVectorScopeResult = false;

    for (size_t end = ops.size(); end > begin; --end) {
      ArrayRef<Operation *> candidate = ops.slice(begin, end - begin);
      if (!hasVectorOperation(candidate))
        continue;

      // Prefer the largest suffix-preserving candidate that actually needs a
      // vecscope and can be moved into today's resultless pto.vecscope form.
      if (canMoveIntoResultlessScope(candidate)) {
        bestEnd = end;
        break;
      }

      if (!sawEscapingVectorScopeResult &&
          findEscapingVectorScopeResult(candidate, escapingValue))
        sawEscapingVectorScopeResult = true;
    }

    if (bestEnd == begin) {
      if (classifyOperationForInference(ops[begin]) ==
              VPTOInferenceOpClass::Vector &&
          sawEscapingVectorScopeResult)
        return emitEscapingVectorScopeValueError(escapingValue);
      ++begin;
      continue;
    }

    wrapCluster(ops.slice(begin, bestEnd - begin), context);
    begin = bestEnd;
  }
  return success();
}

static LogicalResult inferVecScopesInBlock(Block &block, MLIRContext *context) {
  SmallVector<Operation *, 16> pending;

  auto flush = [&]() -> LogicalResult {
    if (failed(wrapGreedySubclusters(pending, context)))
      return failure();
    pending.clear();
    return success();
  };

  SmallVector<Operation *, 32> ops;
  for (Operation &op : block)
    ops.push_back(&op);

  for (Operation *op : ops) {
    switch (classifyOperationForInference(op)) {
    case VPTOInferenceOpClass::Vector:
    case VPTOInferenceOpClass::SafeScalar:
      pending.push_back(op);
      continue;
    case VPTOInferenceOpClass::Boundary:
      if (failed(flush()))
        return failure();
      continue;
    }
  }
  if (failed(flush()))
    return failure();

  SmallVector<Operation *, 32> remainingOps;
  for (Operation &op : block)
    remainingOps.push_back(&op);

  for (Operation *op : remainingOps) {
    if (isExplicitVectorScopeCarrier(op))
      continue;
    for (Region &nested : op->getRegions()) {
      if (failed(inferVecScopesInRegion(nested, context)))
        return failure();
    }
  }
  return success();
}

static LogicalResult inferVecScopesInRegion(Region &region,
                                            MLIRContext *context) {
  for (Block &block : region) {
    if (failed(inferVecScopesInBlock(block, context)))
      return failure();
  }
  return success();
}

struct PTOInferVPTOVecScopePass
    : public pto::impl::PTOInferVPTOVecScopeBase<
          PTOInferVPTOVecScopePass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (failed(inferVecScopesInRegion(func.getBody(), &getContext())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInferVPTOVecScopePass() {
  return std::make_unique<PTOInferVPTOVecScopePass>();
}
