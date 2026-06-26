// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/VFcostmodel/VfSimProgram.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir {
namespace pto {
namespace {

static void printVfOperand(const VfSimOperand &operand, raw_ostream &os) {
  os << getVfOperandKindName(operand.kind) << operand.id;
}

static void printIndent(raw_ostream &os, unsigned indent) {
  for (unsigned i = 0; i < indent; ++i)
    os << ' ';
}

static void printJsonString(StringRef value, raw_ostream &os) {
  os << '"';
  for (char c : value) {
    switch (c) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << c;
      break;
    }
  }
  os << '"';
}

static void printVfOperandJson(const VfSimOperand &operand, raw_ostream &os) {
  os << "{";
  os << "\"kind\": ";
  printJsonString(getVfOperandKindName(operand.kind), os);
  os << ", \"id\": " << operand.id;
  os << ", \"dtype\": ";
  printJsonString(getVfDTypeName(operand.dtype), os);
  os << "}";
}

static void printVfOperandArrayJson(ArrayRef<VfSimOperand> operands,
                                    raw_ostream &os) {
  os << "[";
  llvm::interleaveComma(operands, os, [&](const VfSimOperand &operand) {
    printVfOperandJson(operand, os);
  });
  os << "]";
}

static void printVfSimNode(const VfSimNode &node, raw_ostream &os,
                           unsigned indent, unsigned &loopIndex) {
  if (node.kind == VfSimNode::Kind::Loop) {
    printIndent(os, indent);
    os << "loop " << loopIndex++ << " trip_count=" << node.tripCount
       << " unroll=" << node.unroll << "\n";
    for (const VfSimNode &child : node.body)
      printVfSimNode(child, os, indent + 2, loopIndex);
    return;
  }

  printIndent(os, indent);
  const VfSimInst &inst = node.inst;
  if (!inst.dst.empty() && inst.opcode != VfOpcode::VSTS) {
    printVfOperand(inst.dst.front(), os);
    os << " = ";
  }
  os << getVfOpcodeName(inst.opcode);
  SmallVector<VfSimOperand, 4> operands;
  if (inst.opcode == VfOpcode::VSTS) {
    operands.append(inst.dst.begin(), inst.dst.end());
    operands.append(inst.src.begin(), inst.src.end());
  } else {
    operands.append(inst.src.begin(), inst.src.end());
  }
  if (!operands.empty())
    os << " ";
  llvm::interleaveComma(operands, os, [&](const VfSimOperand &operand) {
    printVfOperand(operand, os);
  });
  os << "\n";
}

static void printVfSimNodeJson(const VfSimNode &node, raw_ostream &os,
                               unsigned indent) {
  printIndent(os, indent);
  if (node.kind == VfSimNode::Kind::Loop) {
    os << "{\n";
    printIndent(os, indent + 2);
    os << "\"type\": \"loop\",\n";
    printIndent(os, indent + 2);
    os << "\"trip_count\": " << node.tripCount << ",\n";
    printIndent(os, indent + 2);
    os << "\"unroll\": " << node.unroll << ",\n";
    printIndent(os, indent + 2);
    os << "\"body\": [\n";
    for (auto [index, child] : llvm::enumerate(node.body)) {
      printVfSimNodeJson(child, os, indent + 4);
      if (index + 1 != node.body.size())
        os << ",";
      os << "\n";
    }
    printIndent(os, indent + 2);
    os << "]\n";
    printIndent(os, indent);
    os << "}";
    return;
  }

  const VfSimInst &inst = node.inst;
  os << "{";
  os << "\"type\": \"inst\", \"op\": ";
  printJsonString(getVfOpcodeName(inst.opcode).upper(), os);
  os << ", \"form\": ";
  printJsonString(inst.form, os);
  os << ", \"dst\": ";
  printVfOperandArrayJson(inst.dst, os);
  os << ", \"src\": ";
  printVfOperandArrayJson(inst.src, os);
  os << "}";
}

} // namespace

StringRef getVfOpcodeName(VfOpcode opcode) {
  switch (opcode) {
  case VfOpcode::VLDS:
    return "vlds";
  case VfOpcode::VSTS:
    return "vsts";
  case VfOpcode::VADD:
    return "vadd";
  case VfOpcode::VADDS:
    return "vadds";
  case VfOpcode::VSUB:
    return "vsub";
  case VfOpcode::VSUBS:
    return "vsubs";
  case VfOpcode::VMUL:
    return "vmul";
  case VfOpcode::VMULS:
    return "vmuls";
  case VfOpcode::VDIV:
    return "vdiv";
  case VfOpcode::VDIVS:
    return "vdivs";
  case VfOpcode::VMAX:
    return "vmax";
  case VfOpcode::VMAXS:
    return "vmaxs";
  case VfOpcode::VMIN:
    return "vmin";
  case VfOpcode::VMINS:
    return "vmins";
  case VfOpcode::VEXP:
    return "vexp";
  case VfOpcode::VCVT_F16_TO_F32:
    return "vcvt_f16_to_f32";
  case VfOpcode::VCVT_F32_TO_F16:
    return "vcvt_f32_to_f16";
  case VfOpcode::VCVT_F32_TO_S32:
    return "vcvt_f32_to_s32";
  case VfOpcode::VCVT_S32_TO_F32:
    return "vcvt_s32_to_f32";
  }
  llvm_unreachable("unknown VF opcode");
}

StringRef getVfOperandKindName(VfOperandKind kind) {
  switch (kind) {
  case VfOperandKind::VReg:
    return "reg";
  case VfOperandKind::UB:
    return "tile";
  case VfOperandKind::Scalar:
    return "scalar";
  }
  llvm_unreachable("unknown VF operand kind");
}

StringRef getVfDTypeName(VfDType dtype) {
  switch (dtype) {
  case VfDType::Unknown:
    return "unknown";
  case VfDType::F32:
    return "fp32";
  case VfDType::F16:
    return "fp16";
  case VfDType::BF16:
    return "bf16";
  case VfDType::I64:
    return "i64";
  case VfDType::I32:
    return "i32";
  case VfDType::I16:
    return "i16";
  case VfDType::I8:
    return "i8";
  case VfDType::UI64:
    return "ui64";
  case VfDType::UI32:
    return "ui32";
  case VfDType::UI16:
    return "ui16";
  case VfDType::UI8:
    return "ui8";
  }
  llvm_unreachable("unknown VF dtype");
}

void printVfSimProgram(const VfSimProgram &program, raw_ostream &os) {
  unsigned loopIndex = 0;
  for (const VfSimNode &node : program.body)
    printVfSimNode(node, os, 0, loopIndex);
}

std::string formatVfSimProgram(const VfSimProgram &program) {
  std::string text;
  llvm::raw_string_ostream os(text);
  printVfSimProgram(program, os);
  return os.str();
}

void printVfSimProgramJson(const VfSimProgram &program, raw_ostream &os,
                           unsigned indent) {
  printIndent(os, indent);
  os << "{\n";
  printIndent(os, indent + 2);
  os << "\"body\": [\n";
  for (auto [index, node] : llvm::enumerate(program.body)) {
    printVfSimNodeJson(node, os, indent + 4);
    if (index + 1 != program.body.size())
      os << ",";
    os << "\n";
  }
  printIndent(os, indent + 2);
  os << "]\n";
  printIndent(os, indent);
  os << "}";
}

std::string formatVfSimProgramJson(const VfSimProgram &program) {
  std::string text;
  llvm::raw_string_ostream os(text);
  printVfSimProgramJson(program, os);
  return os.str();
}

} // namespace pto
} // namespace mlir
