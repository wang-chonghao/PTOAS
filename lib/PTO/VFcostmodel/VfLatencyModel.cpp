// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. Please read the License for details.
// THIS SOFTWARE IS PROVIDED ON "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/VFcostmodel/VfLatencyModel.h"

#include "PTO/VFcostmodel/VfSimulator/IDU.h"
#include "PTO/VFcostmodel/VfSimulator/OOO.h"
#include "PTO/VFcostmodel/VfSimulator/ProgramAnalysis.h"
#include "PTO/VFcostmodel/VfSimulator/ProgramFlatten.h"
#include "PTO/VFcostmodel/VfSimulator/SimulatorRunner.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace mlir {
namespace pto {
namespace {

std::string toUpper(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return text;
}

std::string makeVregName(unsigned id) {
  return "v" + std::to_string(id);
}

std::string makeMemName(unsigned id) {
  return "mem" + std::to_string(id);
}

std::string makeScalarName(unsigned id) {
  return "s" + std::to_string(id);
}

std::string operandToName(const VfSimOperand &operand) {
  switch (operand.kind) {
  case VfOperandKind::VReg:
    return makeVregName(operand.id);
  case VfOperandKind::UB:
    return makeMemName(operand.id);
  case VfOperandKind::Scalar:
    return makeScalarName(operand.id);
  }
  return makeVregName(operand.id);
}

vfsim::ProgramInstNode convertInst(const VfSimInst &inst) {
  vfsim::ProgramInstNode out;
  out.op = toUpper(std::string(getVfOpcodeName(inst.opcode)));
  for (const auto &src : inst.src)
    out.src.push_back(operandToName(src));
  for (const auto &dst : inst.dst)
    out.dst.push_back(operandToName(dst));
  return out;
}

vfsim::ProgramNode convertNode(const VfSimNode &node) {
  if (node.kind == VfSimNode::Kind::Inst)
    return vfsim::ProgramNode::makeInst(convertInst(node.inst));

  vfsim::ProgramLoopNode loop;
  loop.iters = std::to_string(node.tripCount);
  loop.unroll = std::to_string(node.unroll);
  for (const auto &child : node.body)
    loop.body.push_back(convertNode(child));
  return vfsim::ProgramNode::makeLoop(std::move(loop));
}

std::vector<vfsim::ProgramNode> convertProgram(const VfSimProgram &program) {
  std::vector<vfsim::ProgramNode> out;
  out.reserve(program.body.size());
  for (const auto &node : program.body)
    out.push_back(convertNode(node));
  return out;
}

std::string dtypeToString(VfDType dtype) {
  switch (dtype) {
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
  case VfDType::Unknown:
    break;
  }
  return {};
}

std::string inferDType(const VfSimProgram &program) {
  std::string dtype;
  auto visitNode = [&](const auto &self, const VfSimNode &node) -> void {
    if (node.kind == VfSimNode::Kind::Inst) {
      auto checkOperand = [&](const VfSimOperand &op) {
        const std::string cur = dtypeToString(op.dtype);
        if (cur.empty())
          return;
        if (dtype.empty()) {
          dtype = cur;
          return;
        }
        if (dtype != cur)
          throw std::runtime_error("mixed VF operand dtypes are not supported in native VfLatencyModel");
      };
      for (const auto &dst : node.inst.dst)
        checkOperand(dst);
      for (const auto &src : node.inst.src)
        checkOperand(src);
      return;
    }
    for (const auto &child : node.body)
      self(self, child);
  };

  for (const auto &node : program.body)
    visitNode(visitNode, node);

  return dtype.empty() ? "fp32" : dtype;
}

std::filesystem::path nativeRootPath() {
#ifdef PTOAS_VFSIM_ROOT
  return std::filesystem::path(PTOAS_VFSIM_ROOT);
#else
  std::filesystem::path cur = std::filesystem::absolute(std::filesystem::current_path());
  for (int depth = 0; depth < 6; ++depth) {
    if (std::filesystem::exists(cur / "configs" / "isa.json"))
      return cur;
    if (!cur.has_parent_path())
      break;
    auto parent = cur.parent_path();
    if (parent == cur)
      break;
    cur = std::move(parent);
  }
  return std::filesystem::absolute(std::filesystem::current_path());
#endif
}

class NativeVfLatencyModel final : public VfLatencyModel {
public:
  VfLatencyResult predict(const VfSimProgram &program) const override {
    try {
      const std::filesystem::path baseDir = nativeRootPath();
      vfsim::ParamDB pdb(baseDir);
      const auto nativeProgram = convertProgram(program);
      const std::string dtype = inferDType(program);

      vfsim::ProgramAnalysis analysis;
      const auto topBlockLoopBounds = analysis.inferTopBlockLoopBounds(nativeProgram);
      int totalTopBlocks = 0;
      for (const auto &node : nativeProgram) {
        if (node.kind == vfsim::ProgramNode::Kind::Loop)
          ++totalTopBlocks;
      }

      vfsim::ProgramFlatten flattener;
      const auto &linear = flattener.flatten(nativeProgram);

      vfsim::IFU ifu(linear, {}, &pdb, topBlockLoopBounds, totalTopBlocks, dtype);
      vfsim::IDU idu(pdb.uarch(), pdb, {}, {}, totalTopBlocks, topBlockLoopBounds, dtype);
      vfsim::OoOCoreMainline ooo(pdb.uarch(), pdb, dtype);

      std::string resultsDir;
      if (const char *env = std::getenv("PTOAS_VFSIM_OUT_DIR"))
        resultsDir = env;

      const auto result = vfsim::runSimulation(
          ifu, idu, ooo, pdb.uarch(), {}, resultsDir);
      return VfLatencyResult{
          /*supported=*/true,
          /*cycles=*/result.vfEndCycle,
          /*rejectReason=*/{},
      };
    } catch (const std::exception &ex) {
      return VfLatencyResult{
          /*supported=*/false,
          /*cycles=*/0,
          /*rejectReason=*/ex.what(),
      };
    }
  }
};

} // namespace

std::unique_ptr<VfLatencyModel> createVfLatencyModel() {
  return std::make_unique<NativeVfLatencyModel>();
}

} // namespace pto
} // namespace mlir
