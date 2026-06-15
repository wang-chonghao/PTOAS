// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/TileFusion/VfCostModel.h"

#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace mlir {
namespace pto {

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

  for (const FusionComputeNode *node : proposedGroup) {
    if (!node || !isSupportedVfCostTileOp(*node))
      return failure();
    if (node->iterationDomainClass >=
        input.blockAnalysis->iterationDomainClasses.size())
      return failure();
    const IterationDomainClass &domain =
        input.blockAnalysis->iterationDomainClasses[node->iterationDomainClass];
    if (domain.info.proof != IterationDomainProof::Proven)
      return failure();
  }

  VfProgram program;
  program.loops.emplace_back();
  return program;
}

} // namespace pto
} // namespace mlir
