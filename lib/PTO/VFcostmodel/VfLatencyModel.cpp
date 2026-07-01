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
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
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

std::string dtypeToString(VfDType dtype);

std::string defaultFormForInst(const VfSimInst &inst) {
  if (!inst.form.empty())
    return inst.form;
  for (const auto &dst : inst.dst) {
    const std::string dtype = dtypeToString(dst.dtype);
    if (!dtype.empty() && dtype != "unknown")
      return dtype;
  }
  for (const auto &src : inst.src) {
    const std::string dtype = dtypeToString(src.dtype);
    if (!dtype.empty() && dtype != "unknown")
      return dtype;
  }
  return "fp32";
}

vfsim::ProgramInstNode convertInst(const VfSimInst &inst) {
  vfsim::ProgramInstNode out;
  out.op = toUpper(std::string(getVfOpcodeName(inst.opcode)));
  out.form = defaultFormForInst(inst);
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

struct VregVersion {
  std::string name;
  int64_t generation = 0;
};

std::string makeVersionKey(const VregVersion &version) {
  return version.name + "#" + std::to_string(version.generation);
}

std::pair<int64_t, std::string> vregSortKey(const std::string &name) {
  if (name.size() <= 1)
    return {std::numeric_limits<int64_t>::max(), name};
  int64_t value = 0;
  for (size_t i = 1; i < name.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(name[i])))
      return {std::numeric_limits<int64_t>::max(), name};
    value = value * 10 + static_cast<int64_t>(name[i] - '0');
  }
  return {value, name};
}

std::string nextFreshVreg(ArrayRef<std::string> slotPool) {
  std::unordered_set<std::string> used(slotPool.begin(), slotPool.end());
  int64_t maxIndex = -1;
  for (const std::string &name : slotPool) {
    auto key = vregSortKey(name);
    if (key.first != std::numeric_limits<int64_t>::max())
      maxIndex = std::max(maxIndex, key.first);
  }

  for (int64_t candidate = maxIndex + 1;; ++candidate) {
    std::string name = "v" + std::to_string(candidate);
    if (used.find(name) == used.end())
      return name;
  }
}

bool containsSlot(ArrayRef<std::string> slots, StringRef slot) {
  return llvm::is_contained(slots, slot);
}

void normalizeFlatLoopVregs(std::vector<vfsim::ProgramNode> &body) {
  std::unordered_map<std::string, VregVersion> currentVersionByVreg;
  std::unordered_map<std::string, int64_t> versionCounter;
  std::vector<std::vector<std::optional<std::string>>> srcVersions(body.size());
  std::vector<std::vector<std::optional<std::string>>> dstVersions(body.size());
  std::unordered_map<std::string, int64_t> lastUse;

  for (size_t idx = 0; idx < body.size(); ++idx) {
    vfsim::ProgramInstNode &inst = body[idx].inst;
    srcVersions[idx].reserve(inst.src.size());
    for (const std::string &src : inst.src) {
      if (!vfsim::ProgramAnalysis::isVregName(src)) {
        srcVersions[idx].push_back(std::nullopt);
        continue;
      }
      auto it = currentVersionByVreg.find(src);
      if (it == currentVersionByVreg.end()) {
        srcVersions[idx].push_back(std::nullopt);
        continue;
      }
      std::string key = makeVersionKey(it->second);
      srcVersions[idx].push_back(key);
      lastUse[key] = static_cast<int64_t>(idx);
    }

    dstVersions[idx].reserve(inst.dst.size());
    for (const std::string &dst : inst.dst) {
      if (!vfsim::ProgramAnalysis::isVregName(dst)) {
        dstVersions[idx].push_back(std::nullopt);
        continue;
      }
      int64_t &generation = versionCounter[dst];
      ++generation;
      VregVersion version{dst, generation};
      currentVersionByVreg[dst] = version;
      dstVersions[idx].push_back(makeVersionKey(version));
    }
  }

  std::unordered_map<std::string, std::string> currentSlotByVreg;
  std::unordered_map<std::string, std::string> slotOfVersion;
  std::unordered_map<std::string, std::optional<std::string>> slotOccupant;
  std::vector<std::string> slotPool;

  for (size_t idx = 0; idx < body.size(); ++idx) {
    vfsim::ProgramInstNode &inst = body[idx].inst;
    std::vector<std::string> newSrcs = inst.src;
    std::vector<std::string> srcSlotsInUse;

    for (size_t pos = 0; pos < inst.src.size(); ++pos) {
      const std::string &src = inst.src[pos];
      if (!vfsim::ProgramAnalysis::isVregName(src))
        continue;

      std::string slot = src;
      const std::optional<std::string> &version =
          pos < srcVersions[idx].size() ? srcVersions[idx][pos] : std::nullopt;
      if (version) {
        auto slotIt = slotOfVersion.find(*version);
        if (slotIt != slotOfVersion.end())
          slot = slotIt->second;
        else {
          size_t hash = version->find('#');
          std::string versionName = hash == std::string::npos
                                        ? src
                                        : version->substr(0, hash);
          auto curIt = currentSlotByVreg.find(versionName);
          slot = curIt == currentSlotByVreg.end() ? versionName : curIt->second;
        }
      } else {
        auto curIt = currentSlotByVreg.find(src);
        if (curIt != currentSlotByVreg.end())
          slot = curIt->second;
      }

      newSrcs[pos] = slot;
      srcSlotsInUse.push_back(slot);
    }

    std::vector<std::string> newDsts = inst.dst;
    if (inst.dst.size() == 1 &&
        vfsim::ProgramAnalysis::isVregName(inst.dst.front())) {
      const std::string &dstName = inst.dst.front();
      const std::optional<std::string> &dstVersion =
          dstVersions[idx].empty() ? std::nullopt : dstVersions[idx].front();
      if (dstVersion) {
        std::vector<std::string> candidateSlots;
        for (const std::string &slot : slotPool) {
          auto occIt = slotOccupant.find(slot);
          bool reusable = occIt == slotOccupant.end() || !occIt->second;
          if (!reusable) {
            auto lastIt = lastUse.find(*occIt->second);
            int64_t last = lastIt == lastUse.end() ? -1 : lastIt->second;
            reusable = last < static_cast<int64_t>(idx);
          }
          if (reusable)
            candidateSlots.push_back(slot);
        }

        for (size_t pos = 0; pos < srcVersions[idx].size(); ++pos) {
          const std::optional<std::string> &version = srcVersions[idx][pos];
          if (!version)
            continue;
          auto lastIt = lastUse.find(*version);
          if (lastIt == lastUse.end() ||
              lastIt->second != static_cast<int64_t>(idx))
            continue;
          if (pos < newSrcs.size() &&
              !containsSlot(candidateSlots, newSrcs[pos]))
            candidateSlots.push_back(newSrcs[pos]);
        }

        std::string chosenSlot;
        if (newSrcs.size() == 1 && containsSlot(candidateSlots, newSrcs[0])) {
          chosenSlot = newSrcs[0];
        } else if (!candidateSlots.empty()) {
          llvm::sort(candidateSlots, [](const std::string &lhs,
                                        const std::string &rhs) {
            return vregSortKey(lhs) < vregSortKey(rhs);
          });
          chosenSlot = candidateSlots.front();
        } else if (!containsSlot(slotPool, dstName)) {
          chosenSlot = dstName;
          slotPool.push_back(chosenSlot);
        } else {
          chosenSlot = nextFreshVreg(slotPool);
          slotPool.push_back(chosenSlot);
        }

        if (!containsSlot(slotPool, chosenSlot))
          slotPool.push_back(chosenSlot);
        slotOfVersion[*dstVersion] = chosenSlot;
        currentSlotByVreg[dstName] = chosenSlot;
        slotOccupant[chosenSlot] = *dstVersion;
        newDsts[0] = chosenSlot;
      }
    }

    inst.src = std::move(newSrcs);
    inst.dst = std::move(newDsts);
  }
}

void normalizeVregLiveRanges(std::vector<vfsim::ProgramNode> &program) {
  for (vfsim::ProgramNode &node : program) {
    if (node.kind != vfsim::ProgramNode::Kind::Loop || !node.loop)
      continue;

    const bool flatInstBody = llvm::all_of(node.loop->body, [](const auto &op) {
      return op.kind == vfsim::ProgramNode::Kind::Inst;
    });
    if (flatInstBody) {
      normalizeFlatLoopVregs(node.loop->body);
      continue;
    }
    normalizeVregLiveRanges(node.loop->body);
  }
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
      auto nativeProgram = convertProgram(program);
      normalizeVregLiveRanges(nativeProgram);

      vfsim::ProgramAnalysis analysis;
      const auto topBlockLoopBounds = analysis.inferTopBlockLoopBounds(nativeProgram);
      int totalTopBlocks = 0;
      for (const auto &node : nativeProgram) {
        if (node.kind == vfsim::ProgramNode::Kind::Loop)
          ++totalTopBlocks;
      }

      vfsim::ProgramFlatten flattener;
      const auto &linear = flattener.flatten(nativeProgram);

      vfsim::IFU ifu(linear, {}, &pdb, topBlockLoopBounds, totalTopBlocks);
      vfsim::IDU idu(pdb.uarch(), pdb, {}, {}, totalTopBlocks, topBlockLoopBounds);
      vfsim::OoOCoreMainline ooo(pdb.uarch(), pdb);

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
