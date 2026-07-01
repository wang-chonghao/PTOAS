// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/TileFusion/FusionCostModel.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>

using namespace mlir;

namespace mlir {
namespace pto {

bool isCurrentlyPlannableOp(StringRef opName) {
  return llvm::StringSwitch<bool>(opName)
      .Cases("tmul", "tdiv", "tadd", "tsub", "tmax", "tmin", true)
      .Cases("tmuls", "tdivs", "tadds", "tsubs", "tmaxs", "tmins", true)
      .Cases("texp", "tcvt", true)
      .Case("texpands", true)
      .Cases("trowexpandmul", "trowexpanddiv", true)
      .Default(false);
}

bool isProvenIterationDomain(const FusionBlockAnalysis &blockAnalysis,
                             const FusionComputeNode &node) {
  if (node.iterationDomainClass >= blockAnalysis.iterationDomainClasses.size())
    return false;
  return blockAnalysis.iterationDomainClasses[node.iterationDomainClass]
             .info.proof == IterationDomainProof::Proven;
}

static bool hasHardBoundaryBetween(const FusionComputeNode &a,
                                   const FusionComputeNode &b) {
  const FusionComputeNode &earlier = a.blockOrder < b.blockOrder ? a : b;
  const FusionComputeNode &later = a.blockOrder < b.blockOrder ? b : a;

  Operation *cursor = earlier.op->getNextNode();
  while (cursor && cursor != later.op) {
    if (cursor->hasTrait<OpTrait::IsTerminator>() ||
        !cursor->getRegions().empty() || isa<CallOpInterface>(cursor))
      return true;
    cursor = cursor->getNextNode();
  }
  return false;
}

static bool hasHardBoundaryToGroup(ArrayRef<const FusionComputeNode *> group,
                                   const FusionComputeNode &candidate) {
  for (const FusionComputeNode *member : group)
    if (hasHardBoundaryBetween(*member, candidate))
      return true;
  return false;
}

static bool dependsOnPreviousNode(const FusionBlockAnalysis &blockAnalysis,
                                  const FusionComputeNode &previous,
                                  const FusionComputeNode &current) {
  for (unsigned edgeId : current.incomingEdges) {
    if (edgeId >= blockAnalysis.edges.size())
      continue;
    if (blockAnalysis.edges[edgeId].producerNode == previous.id)
      return true;
  }

  for (Value output : previous.semantics.tileOutputs)
    if (llvm::is_contained(current.semantics.tileInputs, output))
      return true;

  return false;
}

static bool isSupportedPlanningNode(const FusionComputeNode &node) {
  return node.semantics.kind == FusionOpKind::Compute &&
         isCurrentlyPlannableOp(node.semantics.opName);
}

static unsigned countEdgesFromGroup(const FusionBlockAnalysis &blockAnalysis,
                                    ArrayRef<const FusionComputeNode *> group,
                                    const FusionComputeNode &candidate) {
  DenseSet<unsigned> producerIds;
  for (const FusionComputeNode *member : group)
    producerIds.insert(member->id);

  unsigned count = 0;
  for (unsigned edgeId : candidate.incomingEdges) {
    if (edgeId >= blockAnalysis.edges.size())
      continue;
    if (producerIds.contains(blockAnalysis.edges[edgeId].producerNode))
      ++count;
  }
  return count;
}

struct GroupFootprint {
  unsigned liveTileCount = 0;
  unsigned vfParameterCount = 0;
};

static bool nodesHaveDirectDataFlowConnection(
    const FusionBlockAnalysis &blockAnalysis, const FusionComputeNode &lhs,
    const FusionComputeNode &rhs) {
  for (unsigned edgeId : lhs.outgoingEdges) {
    if (edgeId >= blockAnalysis.edges.size())
      continue;
    if (blockAnalysis.edges[edgeId].consumerNode == rhs.id)
      return true;
  }

  for (unsigned edgeId : lhs.incomingEdges) {
    if (edgeId >= blockAnalysis.edges.size())
      continue;
    if (blockAnalysis.edges[edgeId].producerNode == rhs.id)
      return true;
  }

  for (Value output : lhs.semantics.tileOutputs)
    if (llvm::is_contained(rhs.semantics.tileInputs, output))
      return true;

  for (Value output : rhs.semantics.tileOutputs)
    if (llvm::is_contained(lhs.semantics.tileInputs, output))
      return true;

  return false;
}

static unsigned countConnectionsToGroup(
    const FusionBlockAnalysis &blockAnalysis,
    ArrayRef<const FusionComputeNode *> group,
    const FusionComputeNode &candidate) {
  unsigned connections = 0;
  for (const FusionComputeNode *member : group)
    if (nodesHaveDirectDataFlowConnection(blockAnalysis, *member, candidate))
      ++connections;
  return connections;
}

static GroupFootprint
computeGroupFootprint(ArrayRef<const FusionComputeNode *> members) {
  DenseSet<Value> producedTiles;
  DenseSet<Value> touchedTiles;
  DenseSet<Value> externalInputs;

  for (const FusionComputeNode *member : members) {
    for (Value output : member->semantics.tileOutputs) {
      producedTiles.insert(output);
      touchedTiles.insert(output);
    }
  }

  for (const FusionComputeNode *member : members) {
    for (Value input : member->semantics.tileInputs) {
      touchedTiles.insert(input);
      if (!producedTiles.contains(input))
        externalInputs.insert(input);
    }
  }

  GroupFootprint footprint;
  footprint.liveTileCount = touchedTiles.size();
  footprint.vfParameterCount = externalInputs.size() + producedTiles.size();
  return footprint;
}

PlanningDecision ConservativeGreedyCostModel::evaluateSeed(
    const PlanningContext &ctx, const FusionComputeNode &candidate) const {
  PlanningDecision decision;
  if (!isSupportedPlanningNode(candidate))
    return decision;

  if (!isProvenIterationDomain(ctx.blockAnalysis, candidate)) {
    decision.cost.rejectedForDynamicShape = true;
    return decision;
  }

  decision.accept = true;
  return decision;
}

PlanningDecision ConservativeGreedyCostModel::evaluateAppend(
    const PlanningContext &ctx, ArrayRef<const FusionComputeNode *> currentGroup,
    const FusionComputeNode &candidate) const {
  PlanningDecision seedDecision = evaluateSeed(ctx, candidate);
  if (!seedDecision.accept)
    return seedDecision;

  PlanningDecision decision;
  if (currentGroup.empty()) {
    decision.accept = true;
    return decision;
  }

  const FusionComputeNode &previous = *currentGroup.back();
  const bool sameDomainClass =
      previous.iterationDomainClass == candidate.iterationDomainClass;
  const bool contiguousInBlock = candidate.blockOrder == previous.blockOrder + 1;
  const bool directlyDependent =
      dependsOnPreviousNode(ctx.blockAnalysis, previous, candidate);
  if (!sameDomainClass || !contiguousInBlock || !directlyDependent)
    return decision;

  SmallVector<const FusionComputeNode *, 8> proposedGroup(currentGroup.begin(),
                                                          currentGroup.end());
  proposedGroup.push_back(&candidate);
  GroupFootprint footprint = computeGroupFootprint(proposedGroup);

  decision.cost.dependencyBenefit =
      4 * static_cast<int64_t>(
              countEdgesFromGroup(ctx.blockAnalysis, currentGroup, candidate));
  decision.cost.loopMergeBenefit = 2;
  decision.cost.liveTilePenalty =
      std::max<int64_t>(0, static_cast<int64_t>(footprint.liveTileCount) - 4);
  decision.cost.vfParameterPenalty =
      std::max<int64_t>(0,
                        static_cast<int64_t>(footprint.vfParameterCount) - 6);
  decision.accept = decision.cost.total() > 0;
  return decision;
}

PlanningDecision ConservativeDAGGreedyCostModel::evaluateSeed(
    const PlanningContext &ctx, const FusionComputeNode &candidate) const {
  PlanningDecision decision;
  if (!isSupportedPlanningNode(candidate))
    return decision;

  if (!isProvenIterationDomain(ctx.blockAnalysis, candidate)) {
    decision.cost.rejectedForDynamicShape = true;
    return decision;
  }

  decision.accept = true;
  return decision;
}

PlanningDecision ConservativeDAGGreedyCostModel::evaluateAppend(
    const PlanningContext &ctx, ArrayRef<const FusionComputeNode *> currentGroup,
    const FusionComputeNode &candidate) const {
  PlanningDecision seedDecision = evaluateSeed(ctx, candidate);
  if (!seedDecision.accept)
    return seedDecision;

  PlanningDecision decision;
  if (currentGroup.empty()) {
    decision.accept = true;
    return decision;
  }

  if (currentGroup.front()->iterationDomainClass !=
      candidate.iterationDomainClass)
    return decision;

  if (hasHardBoundaryToGroup(currentGroup, candidate))
    return decision;

  const unsigned connectionCount =
      countConnectionsToGroup(ctx.blockAnalysis, currentGroup, candidate);
  if (connectionCount == 0)
    return decision;

  SmallVector<const FusionComputeNode *, 8> proposedGroup(currentGroup.begin(),
                                                          currentGroup.end());
  proposedGroup.push_back(&candidate);
  GroupFootprint footprint = computeGroupFootprint(proposedGroup);

  decision.cost.dependencyBenefit = 4 * static_cast<int64_t>(connectionCount);
  decision.cost.loopMergeBenefit = 4;
  decision.cost.liveTilePenalty =
      std::max<int64_t>(0, static_cast<int64_t>(footprint.liveTileCount) - 10);
  decision.cost.vfParameterPenalty =
      std::max<int64_t>(0,
                        static_cast<int64_t>(footprint.vfParameterCount) - 12);
  decision.accept = decision.cost.total() > 0;
  return decision;
}

PlanningDecision LegalityOnlyDAGGreedyCostModel::evaluateSeed(
    const PlanningContext &ctx, const FusionComputeNode &candidate) const {
  PlanningDecision decision;
  if (!isSupportedPlanningNode(candidate))
    return decision;

  if (!isProvenIterationDomain(ctx.blockAnalysis, candidate)) {
    decision.cost.rejectedForDynamicShape = true;
    return decision;
  }

  decision.accept = true;
  return decision;
}

PlanningDecision LegalityOnlyDAGGreedyCostModel::evaluateAppend(
    const PlanningContext &ctx, ArrayRef<const FusionComputeNode *> currentGroup,
    const FusionComputeNode &candidate) const {
  PlanningDecision seedDecision = evaluateSeed(ctx, candidate);
  if (!seedDecision.accept)
    return seedDecision;

  PlanningDecision decision;
  if (currentGroup.empty()) {
    decision.accept = true;
    return decision;
  }

  if (currentGroup.front()->iterationDomainClass !=
      candidate.iterationDomainClass)
    return decision;

  if (hasHardBoundaryToGroup(currentGroup, candidate))
    return decision;

  const unsigned connectionCount =
      countConnectionsToGroup(ctx.blockAnalysis, currentGroup, candidate);
  if (connectionCount == 0)
    return decision;

  decision.cost.dependencyBenefit = 4 * static_cast<int64_t>(connectionCount);
  decision.cost.loopMergeBenefit = 4;
  decision.accept = true;
  return decision;
}

} // namespace pto
} // namespace mlir
