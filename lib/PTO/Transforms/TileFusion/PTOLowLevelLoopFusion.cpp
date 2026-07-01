// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdlib>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOLOWLEVELLOOPFUSION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr =
    "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";
static constexpr llvm::StringLiteral kFusionLoopIndexAttr =
    "pto.fusion.loop_index";
static constexpr llvm::StringLiteral kFusionLoopUnrollAttr =
    "pto.fusion.loop_unroll";

struct LoopLevelInfo {
  scf::ForOp loop;
  SmallVector<Operation *, 4> preludeOps;
};

struct StageInfo {
  SmallVector<Operation *, 4> setupOps;
  SmallVector<LoopLevelInfo, 4> levels;
  SmallVector<Operation *, 8> leafOps;

  scf::ForOp getOuterLoop() const { return levels.front().loop; }
  unsigned getDepth() const { return levels.size(); }
};

static bool areEquivalentValues(Value lhs, Value rhs);

static bool isFusionControlAttr(StringRef name) {
  return name == kFusionGroupIdAttr || name == kFusionOrderAttr ||
         name == kFusionLoopIndexAttr || name == kFusionLoopUnrollAttr ||
         name == "pto.fusion.unroll";
}

static bool sameNonFusionAttrs(Operation *lhs, Operation *rhs) {
  for (NamedAttribute attr : lhs->getAttrs()) {
    if (isFusionControlAttr(attr.getName().getValue()))
      continue;
    if (rhs->getAttr(attr.getName()) != attr.getValue())
      return false;
  }
  for (NamedAttribute attr : rhs->getAttrs()) {
    if (isFusionControlAttr(attr.getName().getValue()))
      continue;
    if (!lhs->hasAttr(attr.getName()))
      return false;
  }
  return true;
}

static Value mapValueOrSelf(Value value, IRMapping &mapping) {
  return mapping.lookupOrDefault(value);
}

static bool sameForHeader(scf::ForOp lhs, scf::ForOp rhs) {
  return areEquivalentValues(lhs.getLowerBound(), rhs.getLowerBound()) &&
         areEquivalentValues(lhs.getUpperBound(), rhs.getUpperBound()) &&
         areEquivalentValues(lhs.getStep(), rhs.getStep()) &&
         sameNonFusionAttrs(lhs.getOperation(), rhs.getOperation());
}

static bool isPureNoRegionOp(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op);
}

static bool isMovableMemoryPreludeOp(Operation *op) {
  return op->getNumRegions() == 0 && isa<MemoryEffectOpInterface>(op);
}

static bool isSupportedPreludeOp(Operation *op) {
  return isPureNoRegionOp(op) || isMovableMemoryPreludeOp(op);
}

static bool isSupportedLeafOp(Operation *op) { return op->getNumRegions() == 0; }

static bool isInterstageSetupOp(Operation *op) {
  if (isPureNoRegionOp(op))
    return true;

  // Tile-native buffers lower to backend memrefs through pto.pointer_cast
  // between adjacent stage loops. Treat these address materializations as
  // stage-boundary-transparent so loop-run collection can keep walking.
  return isa<pto::PointerCastOp>(op);
}

static bool areEquivalentOperations(Operation *lhs, Operation *rhs) {
  if (!lhs || !rhs)
    return false;
  if (lhs->getName() != rhs->getName())
    return false;
  if (lhs->getNumRegions() != 0 || rhs->getNumRegions() != 0)
    return false;
  if (lhs->getNumResults() != rhs->getNumResults())
    return false;
  if (lhs->getNumOperands() != rhs->getNumOperands())
    return false;
  if (lhs->getAttrDictionary() != rhs->getAttrDictionary())
    return false;
  if (!llvm::equal(lhs->getResultTypes(), rhs->getResultTypes()))
    return false;

  if (auto lhsDim = dyn_cast<memref::DimOp>(lhs)) {
    auto rhsDim = cast<memref::DimOp>(rhs);
    return lhsDim.getSource().getType() == rhsDim.getSource().getType() &&
           areEquivalentValues(lhsDim.getIndex(), rhsDim.getIndex());
  }

  for (auto [lhsOperand, rhsOperand] :
       llvm::zip(lhs->getOperands(), rhs->getOperands())) {
    if (!areEquivalentValues(lhsOperand, rhsOperand))
      return false;
  }
  return true;
}

static bool areEquivalentValues(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs)
    return false;
  if (lhs.getType() != rhs.getType())
    return false;

  auto lhsArg = dyn_cast<BlockArgument>(lhs);
  auto rhsArg = dyn_cast<BlockArgument>(rhs);
  if (lhsArg || rhsArg) {
    return lhsArg && rhsArg && lhsArg.getOwner() == rhsArg.getOwner() &&
           lhsArg.getArgNumber() == rhsArg.getArgNumber();
  }

  return areEquivalentOperations(lhs.getDefiningOp(), rhs.getDefiningOp());
}

static Value traceAliasRootOneStep(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto *parentOp = arg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(parentOp)) {
      if (arg.getArgNumber() > 0 &&
          forOp.getInitArgs().size() >= arg.getArgNumber())
        return forOp.getInitArgs()[arg.getArgNumber() - 1];
    }
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return {};

  if (auto subview = dyn_cast<memref::SubViewOp>(def))
    return subview.getSource();
  if (auto cast = dyn_cast<memref::CastOp>(def))
    return cast.getSource();
  if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def))
    return cast.getSource();
  if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(def))
    return cast.getSource();
  if (auto transpose = dyn_cast<memref::TransposeOp>(def))
    return transpose.getIn();
  if (auto bind = dyn_cast<pto::BindTileOp>(def))
    return bind.getSource();
  if (auto subview = dyn_cast<pto::SubViewOp>(def))
    return subview.getSource();
  if (auto bitcast = dyn_cast<pto::BitcastOp>(def))
    return bitcast.getSrc();
  if (auto reshape = dyn_cast<pto::TReshapeOp>(def))
    return reshape.getSrc();
  if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
    if (cast.getInputs().empty())
      return {};
    if (auto result = dyn_cast<OpResult>(value)) {
      unsigned resultNumber = result.getResultNumber();
      if (resultNumber < cast.getInputs().size())
        return cast.getInputs()[resultNumber];
    }
    if (cast.getInputs().size() == 1)
      return cast.getInputs().front();
    return {};
  }
  if (auto forOp = dyn_cast<scf::ForOp>(def)) {
    if (auto result = dyn_cast<OpResult>(value)) {
      unsigned resultNumber = result.getResultNumber();
      if (resultNumber < forOp.getInitArgs().size())
        return forOp.getInitArgs()[resultNumber];
    }
  }

  return {};
}

static Value traceAliasRoot(Value value) {
  int loopBound = 256;
  while (value) {
    Value upward = traceAliasRootOneStep(value);
    if (!upward)
      break;
    value = upward;
    if (loopBound-- <= 0)
      break;
  }
  return value;
}

static LogicalResult collectAliasRelevantRoots(
    Operation *op, SmallVectorImpl<Value> &roots) {
  if (isMemoryEffectFree(op))
    return success();

  auto effectsOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectsOp)
    return failure();

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  effectsOp.getEffects(effects);
  for (const auto &effect : effects) {
    Value effectValue = effect.getValue();
    if (!effectValue)
      return failure();

    Type effectType = effectValue.getType();
    if (!isa<BaseMemRefType, pto::PtrType>(effectType)) {
      if (isa<MemoryEffects::Write>(effect.getEffect()))
        return failure();
      continue;
    }

    Value root = traceAliasRoot(effectValue);
    if (!root)
      return failure();
    roots.push_back(root);
  }
  return success();
}

static bool containsEquivalentRoot(ArrayRef<Value> roots, Value candidate) {
  return llvm::any_of(roots, [&](Value root) {
    return areEquivalentValues(root, candidate);
  });
}

static bool canMovePreludeAcrossPriorStages(Operation *preludeOp,
                                            ArrayRef<StageInfo> priorStages,
                                            llvm::raw_ostream *debugOS) {
  SmallVector<Value, 4> preludeRoots;
  if (failed(collectAliasRelevantRoots(preludeOp, preludeRoots))) {
    if (debugOS)
      *debugOS << "[op-fusion] reject prelude op " << preludeOp->getName()
               << " at " << preludeOp->getLoc()
               << ": touched roots are not alias-analyzable\n";
    return false;
  }
  for (const StageInfo &priorStage : priorStages) {
    for (Operation *leafOp : priorStage.leafOps) {
      SmallVector<Value, 4> leafRoots;
      if (failed(collectAliasRelevantRoots(leafOp, leafRoots))) {
        if (debugOS)
          *debugOS << "[op-fusion] reject prelude op " << preludeOp->getName()
                   << " at " << preludeOp->getLoc()
                   << ": crossed effects of " << leafOp->getName()
                   << " are not alias-analyzable\n";
        return false;
      }
      for (Value preludeRoot : preludeRoots) {
        if (containsEquivalentRoot(leafRoots, preludeRoot)) {
          if (debugOS)
            *debugOS << "[op-fusion] reject prelude op "
                     << preludeOp->getName() << " at " << preludeOp->getLoc()
                     << ": touched root may alias a prior stage memory op\n";
          return false;
        }
      }
    }
  }

  return true;
}

static bool arePreludeReordersLegal(ArrayRef<StageInfo> stages,
                                    llvm::raw_ostream *debugOS) {
  for (size_t stageIndex = 1; stageIndex < stages.size(); ++stageIndex) {
    ArrayRef<StageInfo> priorStages(stages.data(), stageIndex);
    for (const LoopLevelInfo &level : stages[stageIndex].levels) {
      for (Operation *op : level.preludeOps) {
        if (!canMovePreludeAcrossPriorStages(op, priorStages, debugOS))
          return false;
      }
    }
  }
  return true;
}

static LogicalResult analyzeStage(scf::ForOp outerLoop, StageInfo &stage) {
  scf::ForOp currentLoop = outerLoop;
  while (currentLoop) {
    stage.levels.push_back(LoopLevelInfo{currentLoop, {}});
    LoopLevelInfo &currentLevel = stage.levels.back();

    SmallVector<Operation *, 8> bodyOps;
    scf::ForOp childLoop;
    for (Operation &op : currentLoop.getBody()->without_terminator()) {
      bodyOps.push_back(&op);
      if (auto nestedLoop = dyn_cast<scf::ForOp>(op)) {
        if (childLoop)
          return failure();
        childLoop = nestedLoop;
      }
    }

    if (!childLoop) {
      for (Operation *op : bodyOps) {
        if (!isSupportedLeafOp(op))
          return failure();
        stage.leafOps.push_back(op);
      }
      return failure(stage.leafOps.empty());
    }

    bool seenChildLoop = false;
    for (Operation *op : bodyOps) {
      if (op == childLoop.getOperation()) {
        seenChildLoop = true;
        continue;
      }
      if (seenChildLoop || !isSupportedPreludeOp(op))
        return failure();
      currentLevel.preludeOps.push_back(op);
    }

    currentLoop = childLoop;
  }

  return failure();
}

static SmallVector<StageInfo, 8> collectStageRunFrom(scf::ForOp firstLoop,
                                                     llvm::raw_ostream *debugOS) {
  SmallVector<StageInfo, 8> stages;

  StageInfo firstStage;
  if (failed(analyzeStage(firstLoop, firstStage))) {
    if (debugOS)
      *debugOS << "[op-fusion] reject loop stage at " << firstLoop.getLoc()
               << ": stage analysis failed\n";
    return stages;
  }
  stages.push_back(std::move(firstStage));

  SmallVector<Operation *, 4> pendingSetup;
  for (Operation *op = firstLoop->getNextNode(); op; op = op->getNextNode()) {
    if (auto nextLoop = dyn_cast<scf::ForOp>(op)) {
      StageInfo nextStage;
      nextStage.setupOps = pendingSetup;
      pendingSetup.clear();
      if (failed(analyzeStage(nextLoop, nextStage))) {
        if (debugOS)
          *debugOS << "[op-fusion] stop stage run before " << nextLoop.getLoc()
                   << ": next stage analysis failed\n";
        break;
      }
      stages.push_back(std::move(nextStage));
      continue;
    }

    if (!isInterstageSetupOp(op)) {
      if (debugOS)
        *debugOS << "[op-fusion] stop stage run at op " << op->getName()
                 << "\n";
      break;
    }
    pendingSetup.push_back(op);
  }

  return stages;
}

static bool sameLoopNestShape(const StageInfo &lhs, const StageInfo &rhs) {
  if (lhs.getDepth() != rhs.getDepth())
    return false;
  return llvm::all_of(llvm::zip(lhs.levels, rhs.levels), [](auto pair) {
    return sameForHeader(std::get<0>(pair).loop, std::get<1>(pair).loop);
  });
}

static std::optional<int64_t> getConstantIntValue(Value value) {
  APInt intValue;
  if (!matchPattern(value, m_ConstantInt(&intValue)))
    return std::nullopt;
  return intValue.getSExtValue();
}

static std::optional<int64_t> getStaticTripCount(scf::ForOp loop) {
  std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || *upper < *lower)
    return std::nullopt;
  int64_t distance = *upper - *lower;
  if (distance % *step != 0)
    return std::nullopt;
  return distance / *step;
}

static std::optional<int64_t> getLoopUnrollFactor(scf::ForOp loop) {
  Attribute attr = loop->getAttr(kFusionLoopUnrollAttr);
  if (!attr)
    return 1;
  auto intAttr = dyn_cast<IntegerAttr>(attr);
  if (!intAttr)
    return std::nullopt;
  int64_t value = intAttr.getInt();
  if (value < 1)
    return std::nullopt;
  return value;
}

static bool validateFusionLoopAttrs(ArrayRef<StageInfo> stages,
                                    unsigned levelIndex,
                                    llvm::raw_ostream *debugOS) {
  bool hasAnyUnroll = false;
  for (const StageInfo &stage : stages)
    hasAnyUnroll |= stage.levels[levelIndex].loop->hasAttr(kFusionLoopUnrollAttr);
  if (!hasAnyUnroll)
    return true;

  scf::ForOp firstLoop = stages.front().levels[levelIndex].loop;
  Attribute firstGroup = firstLoop->getAttr(kFusionGroupIdAttr);
  Attribute firstLoopIndex = firstLoop->getAttr(kFusionLoopIndexAttr);
  Attribute firstUnroll = firstLoop->getAttr(kFusionLoopUnrollAttr);
  if (!firstGroup || !firstLoopIndex || !firstUnroll) {
    if (debugOS)
      *debugOS << "[op-fusion] reject loop run: incomplete fusion loop attrs\n";
    return false;
  }

  std::optional<int64_t> unroll = getLoopUnrollFactor(firstLoop);
  if (!unroll) {
    if (debugOS)
      *debugOS << "[op-fusion] reject loop run: invalid loop unroll attr\n";
    return false;
  }

  for (const StageInfo &stage : stages) {
    scf::ForOp loop = stage.levels[levelIndex].loop;
    if (loop->getAttr(kFusionGroupIdAttr) != firstGroup ||
        loop->getAttr(kFusionLoopIndexAttr) != firstLoopIndex ||
        loop->getAttr(kFusionLoopUnrollAttr) != firstUnroll) {
      if (debugOS)
        *debugOS << "[op-fusion] reject loop run: fusion loop attr mismatch\n";
      return false;
    }
    if (*unroll > 1 && !loop.getInitArgs().empty()) {
      if (debugOS)
        *debugOS << "[op-fusion] reject loop run: unroll with loop-carried "
                    "values is not supported yet\n";
      return false;
    }
    std::optional<int64_t> tripCount = getStaticTripCount(loop);
    if (*unroll > 1 && (!tripCount || *tripCount % *unroll != 0)) {
      if (debugOS)
        *debugOS << "[op-fusion] reject loop run: loop trip count is not "
                    "statically divisible by unroll\n";
      return false;
    }
  }

  return true;
}

static int64_t getCommonLoopUnrollFactor(ArrayRef<StageInfo> stages,
                                         unsigned levelIndex) {
  std::optional<int64_t> unroll =
      getLoopUnrollFactor(stages.front().levels[levelIndex].loop);
  return unroll.value_or(1);
}

static void cloneOpAndMapResults(OpBuilder &builder, Operation *op,
                                 IRMapping &mapping) {
  Operation *cloned = builder.clone(*op, mapping);
  for (auto [oldRes, newRes] :
       llvm::zip(op->getResults(), cloned->getResults()))
    mapping.map(oldRes, newRes);
}

static void appendMappedValues(ValueRange values, IRMapping &mapping,
                               SmallVectorImpl<Value> &mappedValues) {
  for (Value value : values)
    mappedValues.push_back(mapValueOrSelf(value, mapping));
}

static Value buildOffsetInductionVar(OpBuilder &builder, Location loc,
                                     Value baseIv, Value originalStep,
                                     int64_t offset) {
  if (offset == 0)
    return baseIv;

  Value offsetValue = builder.create<arith::ConstantIndexOp>(loc, offset);
  Value delta = offsetValue;
  if (std::optional<int64_t> step = getConstantIntValue(originalStep)) {
    delta = builder.create<arith::ConstantIndexOp>(loc, *step * offset);
  } else {
    delta = builder.create<arith::MulIOp>(loc, originalStep, offsetValue);
  }
  return builder.create<arith::AddIOp>(loc, baseIv, delta);
}

static scf::ForOp buildFusedLoopNestAtLevel(OpBuilder &builder,
                                            MutableArrayRef<StageInfo> stages,
                                            MutableArrayRef<IRMapping> mappings,
                                            unsigned levelIndex) {
  scf::ForOp firstLoop = stages.front().levels[levelIndex].loop;
  int64_t unroll = getCommonLoopUnrollFactor(stages, levelIndex);

  SmallVector<Value, 8> fusedInitArgs;
  for (auto [stageIndex, stage] : llvm::enumerate(stages))
    appendMappedValues(ValueRange(stage.levels[levelIndex].loop.getInitArgs()),
                       mappings[stageIndex], fusedInitArgs);

  Value originalStep = mapValueOrSelf(firstLoop.getStep(), mappings.front());
  Value fusedStep = originalStep;
  if (unroll > 1) {
    Value factor =
        builder.create<arith::ConstantIndexOp>(firstLoop.getLoc(), unroll);
    fusedStep = builder.create<arith::MulIOp>(firstLoop.getLoc(), originalStep,
                                              factor);
  }

  auto fusedLoop = builder.create<scf::ForOp>(
      firstLoop.getLoc(),
      mapValueOrSelf(firstLoop.getLowerBound(), mappings.front()),
      mapValueOrSelf(firstLoop.getUpperBound(), mappings.front()),
      fusedStep, fusedInitArgs);
  fusedLoop->setAttrs(firstLoop->getAttrs());

  unsigned iterArgOffset = 0;
  for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
    scf::ForOp originalLoop = stage.levels[levelIndex].loop;
    mappings[stageIndex].map(originalLoop.getInductionVar(),
                             fusedLoop.getInductionVar());
    for (auto [argIndex, originalArg] :
         llvm::enumerate(originalLoop.getRegionIterArgs()))
      mappings[stageIndex].map(
          originalArg, fusedLoop.getRegionIterArgs()[iterArgOffset + argIndex]);
    iterArgOffset += originalLoop.getRegionIterArgs().size();
  }

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(fusedLoop.getBody());
  auto emitLevelBody = [&](MutableArrayRef<IRMapping> bodyMappings) {
    for (auto [stageIndex, stage] : llvm::enumerate(stages))
      for (Operation *op : stage.levels[levelIndex].preludeOps)
        cloneOpAndMapResults(bodyBuilder, op, bodyMappings[stageIndex]);

    if (levelIndex + 1 < stages.front().getDepth()) {
      (void)buildFusedLoopNestAtLevel(bodyBuilder, stages, bodyMappings,
                                      levelIndex + 1);
      return;
    }

    for (auto [stageIndex, stage] : llvm::enumerate(stages))
      for (Operation *op : stage.leafOps)
        cloneOpAndMapResults(bodyBuilder, op, bodyMappings[stageIndex]);
  };

  if (unroll == 1) {
    emitLevelBody(mappings);
  } else {
    for (int64_t offset = 0; offset < unroll; ++offset) {
      SmallVector<IRMapping, 8> offsetMappings;
      offsetMappings.reserve(stages.size());
      for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
        scf::ForOp originalLoop = stage.levels[levelIndex].loop;
        IRMapping offsetMapping = mappings[stageIndex];
        Value offsetIv = buildOffsetInductionVar(
            bodyBuilder, originalLoop.getLoc(), fusedLoop.getInductionVar(),
            mapValueOrSelf(originalLoop.getStep(), mappings[stageIndex]),
            offset);
        offsetMapping.map(originalLoop.getInductionVar(), offsetIv);
        offsetMappings.push_back(std::move(offsetMapping));
      }
      emitLevelBody(offsetMappings);
    }
  }

  SmallVector<Value, 8> fusedYieldOperands;
  for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
    auto originalYield = cast<scf::YieldOp>(
        stage.levels[levelIndex].loop.getBody()->getTerminator());
    appendMappedValues(ValueRange(originalYield.getOperands()),
                       mappings[stageIndex], fusedYieldOperands);
  }

  Block *fusedBody = fusedLoop.getBody();
  Operation *fusedTerminator = nullptr;
  if (!fusedBody->empty() &&
      fusedBody->back().hasTrait<OpTrait::IsTerminator>())
    fusedTerminator = &fusedBody->back();

  if (auto fusedYield = dyn_cast_or_null<scf::YieldOp>(fusedTerminator)) {
    fusedYield->setOperands(fusedYieldOperands);
  } else {
    OpBuilder yieldBuilder = OpBuilder::atBlockEnd(fusedBody);
    yieldBuilder.create<scf::YieldOp>(firstLoop.getLoc(), fusedYieldOperands);
  }

  unsigned resultOffset = 0;
  for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
    scf::ForOp originalLoop = stage.levels[levelIndex].loop;
    for (Value originalResult : originalLoop.getResults())
      mappings[stageIndex].map(originalResult,
                               fusedLoop.getResults()[resultOffset++]);
  }

  return fusedLoop;
}

static bool fuseStageRun(SmallVectorImpl<StageInfo> &stages,
                         llvm::raw_ostream *debugOS) {
  if (stages.size() < 2) {
    if (debugOS)
      *debugOS << "[op-fusion] reject loop run: need at least 2 stages, got "
               << stages.size() << "\n";
    return false;
  }

  StageInfo &first = stages.front();
  for (StageInfo &stage : llvm::drop_begin(stages)) {
    if (!sameLoopNestShape(first, stage)) {
      if (debugOS)
        *debugOS << "[op-fusion] reject loop run: loop nest shape mismatch\n";
      return false;
    }
  }
  if (!arePreludeReordersLegal(stages, debugOS))
    return false;
  for (unsigned levelIndex = 0; levelIndex < first.getDepth(); ++levelIndex)
    if (!validateFusionLoopAttrs(stages, levelIndex, debugOS))
      return false;

  OpBuilder blockBuilder(first.getOuterLoop());
  SmallVector<IRMapping, 8> stageMappings(stages.size());
  auto fusedOuterLoop =
      buildFusedLoopNestAtLevel(blockBuilder, stages, stageMappings, 0);

  for (StageInfo &stage : llvm::drop_begin(stages))
    for (Operation *setupOp : stage.setupOps)
      setupOp->moveBefore(fusedOuterLoop);

  for (StageInfo &stage : llvm::reverse(stages))
    stage.getOuterLoop().erase();

  return true;
}

static bool fuseStageRunsInBlock(Block &block, llvm::raw_ostream *debugOS) {
  bool changed = false;
  bool localChange = true;

  while (localChange) {
    localChange = false;
    for (Operation &op : block) {
      auto firstLoop = dyn_cast<scf::ForOp>(op);
      if (!firstLoop)
        continue;

      SmallVector<StageInfo, 8> stages =
          collectStageRunFrom(firstLoop, debugOS);
      if (!fuseStageRun(stages, debugOS))
        continue;

      changed = true;
      localChange = true;
      break;
    }
  }

  return changed;
}

struct PTOLowLevelLoopFusionPass
    : public pto::impl::PTOLowLevelLoopFusionBase<
          PTOLowLevelLoopFusionPass> {
  using pto::impl::PTOLowLevelLoopFusionBase<
      PTOLowLevelLoopFusionPass>::PTOLowLevelLoopFusionBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const bool traceEnabled =
        debug || (std::getenv("PTO_LL_LOOP_FUSION_TRACE") != nullptr);
    llvm::raw_ostream *traceOS = traceEnabled ? &llvm::errs() : nullptr;

    int fusedFuncs = 0;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (func.getSymName().starts_with("__pto_oplib_"))
        continue;
      if (func.empty())
        continue;

      bool changed = false;
      func.walk([&](pto::FusionRegionOp fusionRegion) {
        changed |= fuseStageRunsInBlock(fusionRegion.getBody().front(), traceOS);
      });
      if (changed)
        ++fusedFuncs;
    }

    if (traceEnabled) {
      llvm::errs() << "[op-fusion] low-level loop fusion changed " << fusedFuncs
                   << " function(s)\n";
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowLevelLoopFusionPass(
    const PTOLowLevelLoopFusionOptions &options) {
  return std::make_unique<PTOLowLevelLoopFusionPass>(options);
}
