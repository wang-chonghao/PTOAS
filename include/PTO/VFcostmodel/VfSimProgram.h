// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_VFCOSTMODEL_VFSIMPROGRAM_H
#define PTO_VFCOSTMODEL_VFSIMPROGRAM_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace pto {

enum class VfOpcode {
  VLDS,
  VSTS,
  VADD,
  VADDS,
  VSUB,
  VSUBS,
  VMUL,
  VMULS,
  VDIV,
  VDIVS,
  VMAX,
  VMAXS,
  VMIN,
  VMINS,
  VEXP,
  VCVT_F16_TO_F32,
  VCVT_F32_TO_F16,
  VCVT_F32_TO_S32,
  VCVT_S32_TO_F32,
};

enum class VfOperandKind {
  VReg,
  UB,
  Scalar,
};

enum class VfDType {
  Unknown,
  F32,
  F16,
  BF16,
  I64,
  I32,
  I16,
  I8,
  UI64,
  UI32,
  UI16,
  UI8,
};

struct VfSimOperand {
  unsigned id = 0;
  VfOperandKind kind = VfOperandKind::VReg;
  VfDType dtype = VfDType::Unknown;
};

struct VfSimInst {
  VfOpcode opcode;
  std::string form;
  SmallVector<VfSimOperand, 4> dst;
  SmallVector<VfSimOperand, 4> src;
};

struct VfSimNode {
  enum class Kind {
    Inst,
    Loop,
  };

  Kind kind = Kind::Inst;
  VfSimInst inst;
  int64_t tripCount = ShapedType::kDynamic;
  unsigned unroll = 1;
  std::vector<VfSimNode> body;
};

struct VfSimProgram {
  std::vector<VfSimNode> body;
};

StringRef getVfOpcodeName(VfOpcode opcode);
StringRef getVfOperandKindName(VfOperandKind kind);
StringRef getVfDTypeName(VfDType dtype);
void printVfSimProgram(const VfSimProgram &program, raw_ostream &os);
std::string formatVfSimProgram(const VfSimProgram &program);
void printVfSimProgramJson(const VfSimProgram &program, raw_ostream &os,
                           unsigned indent = 0);
std::string formatVfSimProgramJson(const VfSimProgram &program);

} // namespace pto
} // namespace mlir

#endif // PTO_VFCOSTMODEL_VFSIMPROGRAM_H
