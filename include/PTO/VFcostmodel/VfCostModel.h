// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_VFCOSTMODEL_VFCOSTMODEL_H
#define PTO_VFCOSTMODEL_VFCOSTMODEL_H

#include "PTO/Transforms/TileFusion/FusionAnalysis.h"
#include "PTO/VFcostmodel/VfSimProgram.h"

#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir {
namespace pto {

enum class TilePatternKind {
  BinaryElementwise,
  UnaryElementwise,
  ScaleElementwise,
};

struct TileOpPatternSpec {
  StringRef tileOpName;
  TilePatternKind pattern;
  VfOpcode vectorOpcode;
  unsigned tileInputCount = 0;
  unsigned scalarInputCount = 0;
  unsigned tileOutputCount = 0;
  bool allowFlatten1D = true;
  bool allowLoopFusion = true;
};

struct VfCostInput {
  const FusionBlockAnalysis *blockAnalysis = nullptr;
  ArrayRef<const FusionComputeNode *> currentGroup;
  const FusionComputeNode *candidate = nullptr;
};

std::optional<TileOpPatternSpec> lookupTileOpPatternSpec(StringRef opName);
bool isSupportedVfCostTileOp(const FusionComputeNode &node);
FailureOr<VfSimProgram> buildFusedElementwiseVfSimProgram(
    const VfCostInput &input);

} // namespace pto
} // namespace mlir

#endif // PTO_VFCOSTMODEL_VFCOSTMODEL_H
