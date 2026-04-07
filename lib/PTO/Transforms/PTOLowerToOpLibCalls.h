#ifndef PTO_TRANSFORMS_PTOLOWERTOOPLIBCALLS_H
#define PTO_TRANSFORMS_PTOLOWERTOOPLIBCALLS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace pto {

FailureOr<bool> tryCloneOpLibInlineBridgeOp(OpBuilder &builder, Operation &op,
                                            IRMapping &mapping);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_PTOLOWERTOOPLIBCALLS_H
