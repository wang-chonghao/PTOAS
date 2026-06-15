// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/TileFusion/VfCostModel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace mlir {
namespace pto {
namespace {

struct VfProgramBuilder {
  unsigned nextOperandId = 0;
  DenseMap<Value, VfOperand> tileValueOperands;
  DenseMap<Value, VfOperand> scalarValueOperands;

  VfOperand makeOperand(VfOperandKind kind, Value value = {}) {
    return VfOperand{nextOperandId++, kind, value};
  }

  VfOperand getTileValueOperand(Value value) {
    auto [it, inserted] = tileValueOperands.try_emplace(value);
    if (inserted)
      it->second = makeOperand(VfOperandKind::TileValue, value);
    return it->second;
  }

  VfOperand getScalarValueOperand(Value value) {
    auto [it, inserted] = scalarValueOperands.try_emplace(value);
    if (inserted)
      it->second = makeOperand(VfOperandKind::ScalarValue, value);
    return it->second;
  }

  VfOperand makeVirtualReg(Value value = {}) {
    return makeOperand(VfOperandKind::VirtualReg, value);
  }
};

static SmallVector<const FusionComputeNode *, 8>
buildStableNodeOrder(ArrayRef<const FusionComputeNode *> nodes) {
  SmallVector<const FusionComputeNode *, 8> ordered(nodes.begin(), nodes.end());
  llvm::stable_sort(ordered, [](const FusionComputeNode *lhs,
                                const FusionComputeNode *rhs) {
    if (lhs->blockOrder != rhs->blockOrder)
      return lhs->blockOrder < rhs->blockOrder;
    return lhs->id < rhs->id;
  });
  return ordered;
}

static const FusionValueLiveness *
findLivenessForValue(const FusionBlockAnalysis &blockAnalysis, Value value) {
  for (const FusionValueLiveness &live : blockAnalysis.liveness)
    if (live.value == value)
      return &live;
  return nullptr;
}

static bool hasConsumerInGroup(const FusionValueLiveness &live,
                               const DenseSet<unsigned> &groupNodeIds) {
  for (unsigned consumerId : live.consumerNodes)
    if (groupNodeIds.contains(consumerId))
      return true;
  return false;
}

static bool hasConsumerOutsideGroup(const FusionValueLiveness &live,
                                    const DenseSet<unsigned> &groupNodeIds) {
  for (unsigned consumerId : live.consumerNodes)
    if (!groupNodeIds.contains(consumerId))
      return true;
  return false;
}

static bool mustStoreTileOutput(const FusionBlockAnalysis &blockAnalysis,
                                const DenseSet<unsigned> &groupNodeIds,
                                Value value) {
  const FusionValueLiveness *live = findLivenessForValue(blockAnalysis, value);
  if (!live)
    return true;

  if (live->hasExternalUsers || live->escapesBlock ||
      live->hasLocalBoundaryUsers || live->hasLocalHardBoundaryUsers)
    return true;

  if (hasConsumerOutsideGroup(*live, groupNodeIds))
    return true;

  return !hasConsumerInGroup(*live, groupNodeIds);
}

static FailureOr<int64_t>
computeFlattenTripCount(const FusionBlockAnalysis &blockAnalysis,
                        ArrayRef<const FusionComputeNode *> group) {
  if (group.empty())
    return failure();

  const FusionComputeNode *first = group.front();
  if (first->iterationDomainClass >= blockAnalysis.iterationDomainClasses.size())
    return failure();

  const IterationDomainInfo &info =
      blockAnalysis.iterationDomainClasses[first->iterationDomainClass].info;
  if (info.proof != IterationDomainProof::Proven ||
      info.vRow == ShapedType::kDynamic || info.vCol == ShapedType::kDynamic)
    return failure();

  return info.vRow * info.vCol;
}

static void appendLoadIfNeeded(VfProgramBuilder &builder, VfLoopProgram &loop,
                               DenseMap<Value, VfOperand> &valueToReg,
                               Value input) {
  if (valueToReg.contains(input))
    return;

  VfOperand source = builder.getTileValueOperand(input);
  VfOperand reg = builder.makeVirtualReg(input);
  loop.instructions.push_back(VfInstruction{VfOpcode::VLDS, {source}, reg});
  valueToReg.try_emplace(input, reg);
}

static FailureOr<VfOperand>
getTileInputReg(VfProgramBuilder &builder, VfLoopProgram &loop,
                DenseMap<Value, VfOperand> &valueToReg, Value input) {
  appendLoadIfNeeded(builder, loop, valueToReg, input);
  auto it = valueToReg.find(input);
  if (it == valueToReg.end())
    return failure();
  return it->second;
}

static FailureOr<VfOperand>
emitComputeNode(VfProgramBuilder &builder, VfLoopProgram &loop,
                DenseMap<Value, VfOperand> &valueToReg,
                const FusionComputeNode &node) {
  std::optional<TileOpPatternSpec> spec =
      lookupTileOpPatternSpec(node.semantics.opName);
  if (!spec)
    return failure();

  SmallVector<VfOperand, 4> operands;
  for (Value input : node.semantics.tileInputs) {
    FailureOr<VfOperand> reg =
        getTileInputReg(builder, loop, valueToReg, input);
    if (failed(reg))
      return failure();
    operands.push_back(*reg);
  }
  for (Value scalar : node.semantics.scalarInputs)
    operands.push_back(builder.getScalarValueOperand(scalar));

  VfOperand result = builder.makeVirtualReg(node.semantics.tileOutputs.front());
  loop.instructions.push_back(
      VfInstruction{spec->vectorOpcode, std::move(operands), result});
  valueToReg[node.semantics.tileOutputs.front()] = result;
  return result;
}

} // namespace

std::optional<TileOpPatternSpec> lookupTileOpPatternSpec(StringRef opName) {
  struct PatternKey {
    TilePatternKind pattern;
    VfOpcode opcode;
    unsigned tileInputs;
    unsigned scalarInputs;
  };

  std::optional<PatternKey> key =
      llvm::StringSwitch<std::optional<PatternKey>>(opName)
          .Case("tadd", PatternKey{TilePatternKind::BinaryElementwise,
                                    VfOpcode::VADD, 2, 0})
          .Case("tsub", PatternKey{TilePatternKind::BinaryElementwise,
                                    VfOpcode::VSUB, 2, 0})
          .Case("tmul", PatternKey{TilePatternKind::BinaryElementwise,
                                    VfOpcode::VMUL, 2, 0})
          .Case("texp", PatternKey{TilePatternKind::UnaryElementwise,
                                    VfOpcode::VEXP, 1, 0})
          .Case("tadds", PatternKey{TilePatternKind::ScaleElementwise,
                                     VfOpcode::VADD, 1, 1})
          .Case("tsubs", PatternKey{TilePatternKind::ScaleElementwise,
                                     VfOpcode::VSUB, 1, 1})
          .Case("tmuls", PatternKey{TilePatternKind::ScaleElementwise,
                                     VfOpcode::VMUL, 1, 1})
          .Default(std::nullopt);
  if (!key)
    return std::nullopt;

  return TileOpPatternSpec{
      opName,
      key->pattern,
      key->opcode,
      key->tileInputs,
      key->scalarInputs,
      1,
      true,
      true,
  };
}

bool isSupportedVfCostTileOp(const FusionComputeNode &node) {
  if (node.semantics.kind != FusionOpKind::Compute)
    return false;

  std::optional<TileOpPatternSpec> spec =
      lookupTileOpPatternSpec(node.semantics.opName);
  if (!spec)
    return false;

  return node.semantics.tileInputs.size() == spec->tileInputCount &&
         node.semantics.scalarInputs.size() == spec->scalarInputCount &&
         node.semantics.tileOutputs.size() == spec->tileOutputCount;
}

FailureOr<VfProgram> buildFusedElementwiseVfProgram(const VfCostInput &input) {
  if (!input.blockAnalysis || !input.candidate)
    return failure();

  SmallVector<const FusionComputeNode *, 8> proposedGroup(
      input.currentGroup.begin(), input.currentGroup.end());
  proposedGroup.push_back(input.candidate);
  proposedGroup = buildStableNodeOrder(proposedGroup);

  DenseSet<unsigned> groupNodeIds;
  for (const FusionComputeNode *node : proposedGroup) {
    if (!node || !isSupportedVfCostTileOp(*node))
      return failure();
    groupNodeIds.insert(node->id);
    if (node->iterationDomainClass >=
        input.blockAnalysis->iterationDomainClasses.size())
      return failure();
    const IterationDomainClass &domain =
        input.blockAnalysis->iterationDomainClasses[node->iterationDomainClass];
    if (domain.info.proof != IterationDomainProof::Proven)
      return failure();
    if (node->iterationDomainClass != proposedGroup.front()->iterationDomainClass)
      return failure();
  }

  FailureOr<int64_t> tripCount =
      computeFlattenTripCount(*input.blockAnalysis, proposedGroup);
  if (failed(tripCount))
    return failure();

  VfProgramBuilder builder;
  VfProgram program;
  VfLoopProgram loop;
  loop.tripCount = *tripCount;

  DenseMap<Value, VfOperand> valueToReg;
  for (const FusionComputeNode *node : proposedGroup) {
    if (failed(emitComputeNode(builder, loop, valueToReg, *node)))
      return failure();
  }

  for (const FusionComputeNode *node : proposedGroup) {
    for (Value output : node->semantics.tileOutputs) {
      if (!mustStoreTileOutput(*input.blockAnalysis, groupNodeIds, output))
        continue;

      auto regIt = valueToReg.find(output);
      if (regIt == valueToReg.end())
        return failure();

      VfOperand destination = builder.getTileValueOperand(output);
      loop.instructions.push_back(
          VfInstruction{VfOpcode::VSTS, {destination, regIt->second},
                        std::nullopt});
    }
  }

  program.loops.push_back(std::move(loop));
  return program;
}

} // namespace pto
} // namespace mlir
