#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOPTRCASTCLEANUP
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct CollapsePtrMemRefPtrBridgePattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();

    auto resultPtrType = dyn_cast<pto::PtrType>(op.getResult(0).getType());
    if (!resultPtrType)
      return failure();

    auto castOp = op.getOperand(0).getDefiningOp<memref::CastOp>();
    if (!castOp || castOp->getNumOperands() != 1)
      return failure();

    auto innerCast =
        castOp.getSource().getDefiningOp<UnrealizedConversionCastOp>();
    if (!innerCast || innerCast->getNumOperands() != 1 ||
        innerCast->getNumResults() != 1)
      return failure();

    Value basePtr = innerCast.getOperand(0);
    if (basePtr.getType() != resultPtrType)
      return failure();

    rewriter.replaceOp(op, basePtr);
    if (castOp->use_empty())
      rewriter.eraseOp(castOp);
    if (innerCast->use_empty())
      rewriter.eraseOp(innerCast);
    return success();
  }
};

struct VPTOPtrCastCleanupPass
    : public pto::impl::VPTOPtrCastCleanupBase<VPTOPtrCastCleanupPass> {
  using pto::impl::VPTOPtrCastCleanupBase<
      VPTOPtrCastCleanupPass>::VPTOPtrCastCleanupBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CollapsePtrMemRefPtrBridgePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOPtrCastCleanupPass() {
  return std::make_unique<VPTOPtrCastCleanupPass>();
}
