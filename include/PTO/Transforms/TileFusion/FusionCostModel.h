// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_TILEFUSION_FUSIONCOSTMODEL_H
#define PTO_TRANSFORMS_TILEFUSION_FUSIONCOSTMODEL_H

#include "PTO/Transforms/TileFusion/FusionAnalysis.h"

#include "mlir/Support/LLVM.h"

namespace mlir {
namespace pto {

struct PlanningContext {
  const FusionBlockAnalysis &blockAnalysis;
};

struct PlanningCost {
  int64_t dependencyBenefit = 0;
  int64_t loopMergeBenefit = 0;
  int64_t liveTilePenalty = 0;
  int64_t vfParameterPenalty = 0;
  bool rejectedForDynamicShape = false;

  int64_t total() const {
    return dependencyBenefit + loopMergeBenefit - liveTilePenalty -
           vfParameterPenalty;
  }
};

struct PlanningDecision {
  bool accept = false;
  PlanningCost cost;
};

bool isCurrentlyPlannableOp(StringRef opName);
bool isProvenIterationDomain(const FusionBlockAnalysis &blockAnalysis,
                             const FusionComputeNode &node);

class CostModel {
public:
  virtual ~CostModel() = default;

  virtual PlanningDecision evaluateSeed(const PlanningContext &ctx,
                                        const FusionComputeNode &candidate)
      const = 0;

  virtual PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const FusionComputeNode *> currentGroup,
                 const FusionComputeNode &candidate) const = 0;
};

class ConservativeGreedyCostModel final : public CostModel {
public:
  PlanningDecision evaluateSeed(const PlanningContext &ctx,
                                const FusionComputeNode &candidate) const override;

  PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const FusionComputeNode *> currentGroup,
                 const FusionComputeNode &candidate) const override;
};

class ConservativeDAGGreedyCostModel final : public CostModel {
public:
  PlanningDecision evaluateSeed(const PlanningContext &ctx,
                                const FusionComputeNode &candidate) const override;

  PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const FusionComputeNode *> currentGroup,
                 const FusionComputeNode &candidate) const override;
};

class LegalityOnlyDAGGreedyCostModel final : public CostModel {
public:
  PlanningDecision evaluateSeed(const PlanningContext &ctx,
                                const FusionComputeNode &candidate) const override;

  PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const FusionComputeNode *> currentGroup,
                 const FusionComputeNode &candidate) const override;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_TILEFUSION_FUSIONCOSTMODEL_H
