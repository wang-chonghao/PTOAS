// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_TILEFUSION_VFCOSTMODEL_H
#define PTO_TRANSFORMS_TILEFUSION_VFCOSTMODEL_H

#include "PTO/Transforms/TileFusion/FusionAnalysis.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace pto {

enum class VfOpcode {
  VLDS,
  VSTS,
  VADD,
  VSUB,
  VMUL,
  VEXP,
};

enum class TilePatternKind {
  BinaryElementwise,
  UnaryElementwise,
  ScaleElementwise,
};

enum class VfOperandKind {
  VirtualReg,
  TileValue,
  ScalarValue,
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

struct VfOperand {
  unsigned id = 0;
  VfOperandKind kind = VfOperandKind::VirtualReg;
  Value value;
};

struct VfInstruction {
  VfOpcode opcode;
  SmallVector<VfOperand, 4> operands;
  std::optional<VfOperand> result;
};

struct VfLoopProgram {
  int64_t tripCount = ShapedType::kDynamic;
  unsigned unroll = 1;
  SmallVector<VfInstruction, 16> instructions;
};

struct VfProgram {
  SmallVector<VfLoopProgram, 2> loops;
};

struct VfCostInput {
  const FusionBlockAnalysis *blockAnalysis = nullptr;
  ArrayRef<const FusionComputeNode *> currentGroup;
  const FusionComputeNode *candidate = nullptr;
};

std::optional<TileOpPatternSpec> lookupTileOpPatternSpec(StringRef opName);
bool isSupportedVfCostTileOp(const FusionComputeNode &node);
FailureOr<VfProgram> buildFusedElementwiseVfProgram(const VfCostInput &input);
StringRef getVfOpcodeName(VfOpcode opcode);
StringRef getVfOperandKindName(VfOperandKind kind);
void printVfProgram(const VfProgram &program, raw_ostream &os);
std::string formatVfProgram(const VfProgram &program);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_TILEFUSION_VFCOSTMODEL_H
