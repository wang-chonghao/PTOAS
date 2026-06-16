// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/TileFusion/FusionCostModel.h"
#include "PTO/Transforms/TileFusion/FusionAnalysis.h"
#include "PTO/Transforms/TileFusion/FusionOpSemantics.h"
#include "PTO/VFcostmodel/VfCostModel.h"
#include "PTO/VFcostmodel/VfLatencyModel.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_FUSIONPLAN
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

using pto::ConservativeDAGGreedyCostModel;
using pto::CostModel;
using pto::PlanningContext;
using pto::PlanningDecision;

static constexpr llvm::StringLiteral kFusionGroupIdAttr =
    "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";

struct PlannedFusionGroup {
  SmallVector<const pto::FusionComputeNode *, 8> members;
};

struct PlannedVfSimProgram {
  int64_t groupId = -1;
  pto::VfSimProgram program;
  pto::VfLatencyResult latency;
};

static pto::VfLatencyResult
predictVfLatency(const pto::VfSimProgram &program) {
  static const std::unique_ptr<pto::VfLatencyModel> model =
      pto::createVfLatencyModel();
  return model->predict(program);
}

static SmallVector<const pto::FusionComputeNode *, 8>
buildStableInGroupOrder(ArrayRef<const pto::FusionComputeNode *> members) {
  SmallVector<const pto::FusionComputeNode *, 8> ordered(members.begin(),
                                                         members.end());
  llvm::stable_sort(ordered, [](const pto::FusionComputeNode *lhs,
                                const pto::FusionComputeNode *rhs) {
    if (lhs->blockOrder != rhs->blockOrder)
      return lhs->blockOrder < rhs->blockOrder;
    return lhs->id < rhs->id;
  });
  return ordered;
}

static void assignStableGroupMetadata(ArrayRef<PlannedFusionGroup> groups,
                                      MLIRContext *ctx,
                                      int64_t &nextGroupId) {
  SmallVector<const PlannedFusionGroup *, 8> orderedGroups;
  orderedGroups.reserve(groups.size());
  for (const PlannedFusionGroup &group : groups)
    orderedGroups.push_back(&group);

  llvm::stable_sort(orderedGroups, [](const PlannedFusionGroup *lhs,
                                      const PlannedFusionGroup *rhs) {
    const pto::FusionComputeNode *lhsFirst = lhs->members.front();
    const pto::FusionComputeNode *rhsFirst = rhs->members.front();
    if (lhsFirst->blockOrder != rhsFirst->blockOrder)
      return lhsFirst->blockOrder < rhsFirst->blockOrder;
    return lhsFirst->id < rhsFirst->id;
  });

  for (const PlannedFusionGroup *group : orderedGroups) {
    const int64_t groupId = nextGroupId++;
    SmallVector<const pto::FusionComputeNode *, 8> stableOrder =
        buildStableInGroupOrder(group->members);
    for (auto [order, node] : llvm::enumerate(stableOrder)) {
      node->op->setAttr(kFusionGroupIdAttr,
                        IntegerAttr::get(IntegerType::get(ctx, 64), groupId));
      node->op->setAttr(
          kFusionOrderAttr,
          IntegerAttr::get(IntegerType::get(ctx, 64),
                           static_cast<int64_t>(order)));
    }
  }
}

class StrategyEngine {
public:
  virtual ~StrategyEngine() = default;

  virtual SmallVector<PlannedFusionGroup, 8>
  planBlock(const PlanningContext &ctx, const CostModel &costModel) const = 0;
};

class ConservativeGreedyStrategyEngine final : public StrategyEngine {
public:
  SmallVector<PlannedFusionGroup, 8>
  planBlock(const PlanningContext &ctx,
            const CostModel &costModel) const override {
    SmallVector<PlannedFusionGroup, 8> groups;
    SmallVector<const pto::FusionComputeNode *, 8> chain;

    auto flushChain = [&]() {
      if (chain.size() < 2) {
        chain.clear();
        return;
      }

      PlannedFusionGroup group;
      group.members = chain;
      groups.push_back(std::move(group));
      chain.clear();
    };

    for (const pto::FusionComputeNode &node : ctx.blockAnalysis.computeNodes) {
      PlanningDecision seedDecision = costModel.evaluateSeed(ctx, node);
      if (!seedDecision.accept) {
        flushChain();
        continue;
      }

      if (chain.empty()) {
        chain.push_back(&node);
        continue;
      }

      PlanningDecision appendDecision =
          costModel.evaluateAppend(ctx, chain, node);
      if (!appendDecision.accept) {
        flushChain();
        chain.push_back(&node);
        continue;
      }

      chain.push_back(&node);
    }

    flushChain();
    return groups;
  }
};

class ConservativeDAGGreedyStrategyEngine final : public StrategyEngine {
public:
  SmallVector<PlannedFusionGroup, 8>
  planBlock(const PlanningContext &ctx,
            const CostModel &costModel) const override {
    SmallVector<PlannedFusionGroup, 8> groups;
    DenseSet<unsigned> assignedNodes;

    for (const pto::FusionComputeNode &seed : ctx.blockAnalysis.computeNodes) {
      if (assignedNodes.contains(seed.id))
        continue;

      PlanningDecision seedDecision = costModel.evaluateSeed(ctx, seed);
      if (!seedDecision.accept)
        continue;

      SmallVector<const pto::FusionComputeNode *, 8> groupMembers;
      DenseSet<unsigned> groupNodeIds;
      groupMembers.push_back(&seed);
      groupNodeIds.insert(seed.id);

      bool changed = true;
      while (changed) {
        changed = false;
        for (const pto::FusionComputeNode &candidate :
             ctx.blockAnalysis.computeNodes) {
          if (assignedNodes.contains(candidate.id) ||
              groupNodeIds.contains(candidate.id))
            continue;

          PlanningDecision appendDecision =
              costModel.evaluateAppend(ctx, groupMembers, candidate);
          if (!appendDecision.accept)
            continue;

          groupMembers.push_back(&candidate);
          groupNodeIds.insert(candidate.id);
          changed = true;
        }
      }

      if (groupMembers.size() < 2)
        continue;

      PlannedFusionGroup group;
      group.members = buildStableInGroupOrder(groupMembers);
      groups.push_back(group);
      for (const pto::FusionComputeNode *member : group.members)
        assignedNodes.insert(member->id);
    }

    return groups;
  }
};

static void clearPlanningAttrs(func::FuncOp func) {
  func.walk([](Operation *op) {
    op->removeAttr(kFusionGroupIdAttr);
    op->removeAttr(kFusionOrderAttr);
  });
}

static void dumpVfProgramsForGroupsText(
    ArrayRef<PlannedFusionGroup> groups,
    const pto::FusionBlockAnalysis &blockAnalysis) {
  for (const PlannedFusionGroup &group : groups) {
    if (group.members.size() < 2)
      continue;

    ArrayRef<const pto::FusionComputeNode *> prefix(group.members.data(),
                                                    group.members.size() - 1);
    pto::VfCostInput input{&blockAnalysis, prefix, group.members.back()};
    FailureOr<pto::VfSimProgram> program =
        pto::buildFusedElementwiseVfSimProgram(input);
    if (failed(program)) {
      llvm::errs() << "[pto-fusion-plan] failed to build VF program for group\n";
      continue;
    }

    llvm::errs() << "[pto-fusion-plan] VF program for fusion group:\n";
    pto::printVfSimProgram(*program, llvm::errs());
    pto::VfLatencyResult latency = predictVfLatency(*program);
    if (latency.supported) {
      llvm::errs() << "[pto-fusion-plan] VF latency cycles="
                   << latency.cycles << "\n";
    } else {
      llvm::errs() << "[pto-fusion-plan] VF latency rejected: "
                   << latency.rejectReason << "\n";
    }
  }
}

static void collectVfProgramsForGroups(
    ArrayRef<PlannedFusionGroup> groups,
    const pto::FusionBlockAnalysis &blockAnalysis,
    SmallVectorImpl<PlannedVfSimProgram> &programs) {
  for (auto [groupIndex, group] : llvm::enumerate(groups)) {
    if (group.members.size() < 2)
      continue;

    ArrayRef<const pto::FusionComputeNode *> prefix(group.members.data(),
                                                    group.members.size() - 1);
    pto::VfCostInput input{&blockAnalysis, prefix, group.members.back()};
    FailureOr<pto::VfSimProgram> program =
        pto::buildFusedElementwiseVfSimProgram(input);
    if (failed(program))
      continue;

    pto::VfLatencyResult latency = predictVfLatency(*program);
    programs.push_back(PlannedVfSimProgram{
        static_cast<int64_t>(groupIndex), std::move(*program), latency});
  }
}

static LogicalResult writeVfProgramsJson(ArrayRef<PlannedVfSimProgram> programs,
                                         StringRef path) {
  if (path.empty())
    return success();

  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "[pto-fusion-plan] failed to open VF program JSON dump '"
                 << path << "': " << ec.message() << "\n";
    return failure();
  }

  os << "{\n";
  os << "  \"programs\": [\n";
  for (auto [index, planned] : llvm::enumerate(programs)) {
    os << "    {\n";
    os << "      \"group_id\": " << planned.groupId << ",\n";
    os << "      \"vf_latency\": {\"supported\": "
       << (planned.latency.supported ? "true" : "false")
       << ", \"cycles\": " << planned.latency.cycles
       << ", \"reject_reason\": ";
    os << "\"";
    for (char c : planned.latency.rejectReason) {
      if (c == '\\' || c == '"')
        os << '\\';
      os << c;
    }
    os << "\"},\n";
    os << "      \"vf_program\":\n";
    pto::printVfSimProgramJson(planned.program, os, 6);
    os << "\n";
    os << "    }";
    if (index + 1 != programs.size())
      os << ",";
    os << "\n";
  }
  os << "  ]\n";
  os << "}\n";
  return success();
}

struct FusionPlanPass : public pto::impl::FusionPlanBase<FusionPlanPass> {
  using pto::impl::FusionPlanBase<FusionPlanPass>::FusionPlanBase;

  explicit FusionPlanPass(bool dumpVfProgram) {
    this->dumpVfProgram = dumpVfProgram;
  }

  FusionPlanPass(bool dumpVfProgram, StringRef dumpVfProgramJson) {
    this->dumpVfProgram = dumpVfProgram;
    this->dumpVfProgramJson = dumpVfProgramJson.str();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    clearPlanningAttrs(func);

    const auto &analysis = getAnalysis<pto::PreFusionAnalysis>();
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = &getContext();
    int64_t nextGroupId = 0;
    ConservativeDAGGreedyCostModel costModel;
    ConservativeDAGGreedyStrategyEngine strategyEngine;
    SmallVector<PlannedVfSimProgram, 8> plannedVfSimPrograms;

    for (const pto::FusionBlockAnalysis &blockAnalysis :
         analysis.getResult().blocks) {
      PlanningContext planningCtx{blockAnalysis};
      SmallVector<PlannedFusionGroup, 8> groups =
          strategyEngine.planBlock(planningCtx, costModel);
      if (dumpVfProgram)
        dumpVfProgramsForGroupsText(groups, blockAnalysis);
      if (!dumpVfProgramJson.empty())
        collectVfProgramsForGroups(groups, blockAnalysis, plannedVfSimPrograms);
      assignStableGroupMetadata(groups, ctx, nextGroupId);
    }

    if (failed(writeVfProgramsJson(plannedVfSimPrograms, dumpVfProgramJson))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createFusionPlanPass() {
  return std::make_unique<FusionPlanPass>();
}

std::unique_ptr<Pass> mlir::pto::createFusionPlanPass(bool dumpVfProgram) {
  return std::make_unique<FusionPlanPass>(dumpVfProgram);
}

std::unique_ptr<Pass>
mlir::pto::createFusionPlanPass(bool dumpVfProgram,
                                StringRef dumpVfProgramJson) {
  return std::make_unique<FusionPlanPass>(dumpVfProgram, dumpVfProgramJson);
}
