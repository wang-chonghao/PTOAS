// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/VFcostmodel/VfCostModel.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
namespace pto {
namespace {

constexpr int64_t kA5VectorBytes = 256;

struct VfSimProgramBuilder {
  unsigned nextOperandId = 0;
  DenseMap<Value, VfSimOperand> tileValueOperands;
  DenseMap<Value, VfSimOperand> scalarValueOperands;

  FailureOr<VfDType> inferDType(Type type) {
    if (auto tileType = dyn_cast<pto::TileBufType>(type))
      type = tileType.getElementType();
    if (type.isF32())
      return VfDType::F32;
    if (type.isF16())
      return VfDType::F16;
    if (type.isBF16())
      return VfDType::BF16;
    if (type.isUnsignedInteger(64))
      return VfDType::UI64;
    if (type.isUnsignedInteger(32))
      return VfDType::UI32;
    if (type.isUnsignedInteger(16))
      return VfDType::UI16;
    if (type.isUnsignedInteger(8))
      return VfDType::UI8;
    if (type.isInteger(64))
      return VfDType::I64;
    if (type.isInteger(32) || type.isIndex())
      return VfDType::I32;
    if (type.isInteger(16))
      return VfDType::I16;
    if (type.isInteger(8))
      return VfDType::I8;
    return failure();
  }

  FailureOr<VfSimOperand> makeOperand(VfOperandKind kind, Value value) {
    FailureOr<VfDType> dtype = inferDType(value.getType());
    if (failed(dtype))
      return failure();
    return VfSimOperand{nextOperandId++, kind, *dtype};
  }

  FailureOr<VfSimOperand> getTileValueOperand(Value value) {
    auto [it, inserted] = tileValueOperands.try_emplace(value);
    if (inserted) {
      FailureOr<VfSimOperand> operand = makeOperand(VfOperandKind::UB, value);
      if (failed(operand))
        return failure();
      it->second = *operand;
    }
    return it->second;
  }

  FailureOr<VfSimOperand> getScalarValueOperand(Value value) {
    auto [it, inserted] = scalarValueOperands.try_emplace(value);
    if (inserted) {
      FailureOr<VfSimOperand> operand =
          makeOperand(VfOperandKind::Scalar, value);
      if (failed(operand))
        return failure();
      it->second = *operand;
    }
    return it->second;
  }

  FailureOr<VfSimOperand> makeVirtualReg(Value value) {
    return makeOperand(VfOperandKind::VReg, value);
  }
};

static VfSimNode makeInstNode(VfSimInst inst) {
  VfSimNode node;
  node.kind = VfSimNode::Kind::Inst;
  node.inst = std::move(inst);
  return node;
}

static VfSimNode makeLoopNode(int64_t tripCount, unsigned unroll,
                              std::vector<VfSimNode> body) {
  VfSimNode node;
  node.kind = VfSimNode::Kind::Loop;
  node.tripCount = tripCount;
  node.unroll = unroll;
  node.body = std::move(body);
  return node;
}

static SmallVector<const FusionComputeNode *, 8>
buildStableNodeOrder(ArrayRef<const FusionComputeNode *> nodes) {
  SmallVector<const FusionComputeNode *, 8> ordered(nodes.begin(), nodes.end());
  llvm::stable_sort(ordered, [](const FusionComputeNode *lhs,
                                const FusionComputeNode *rhs) {
    if (lhs->blockOrder != rhs->blockOrder)
      return lhs->blockOrder < rhs->blockOrder;
    return lhs->id < rhs->id;
  });
  return ordered;
}

static const FusionValueLiveness *
findLivenessForValue(const FusionBlockAnalysis &blockAnalysis, Value value) {
  for (const FusionValueLiveness &live : blockAnalysis.liveness)
    if (live.value == value)
      return &live;
  return nullptr;
}

static bool hasConsumerInGroup(const FusionValueLiveness &live,
                               const DenseSet<unsigned> &groupNodeIds) {
  for (unsigned consumerId : live.consumerNodes)
    if (groupNodeIds.contains(consumerId))
      return true;
  return false;
}

static bool hasConsumerOutsideGroup(const FusionValueLiveness &live,
                                    const DenseSet<unsigned> &groupNodeIds) {
  for (unsigned consumerId : live.consumerNodes)
    if (!groupNodeIds.contains(consumerId))
      return true;
  return false;
}

static const FusionWriteInstanceLiveness *
findWriteInstanceForOutput(const FusionBlockAnalysis &blockAnalysis,
                           unsigned producerNode, Value value) {
  for (const FusionWriteInstanceLiveness &write : blockAnalysis.writeInstances) {
    if (!write.producerNode || *write.producerNode != producerNode)
      continue;
    if (write.value == value)
      return &write;
  }
  return nullptr;
}

static bool hasConsumerInGroup(const FusionWriteInstanceLiveness &live,
                               const DenseSet<unsigned> &groupNodeIds) {
  for (unsigned consumerId : live.consumerNodes)
    if (groupNodeIds.contains(consumerId))
      return true;
  return false;
}

static bool hasConsumerOutsideGroup(const FusionWriteInstanceLiveness &live,
                                    const DenseSet<unsigned> &groupNodeIds) {
  for (unsigned consumerId : live.consumerNodes)
    if (!groupNodeIds.contains(consumerId))
      return true;
  return false;
}

static bool mustStoreWriteInstance(const FusionWriteInstanceLiveness &live,
                                   const DenseSet<unsigned> &groupNodeIds) {
  if (live.hasExternalUsers || live.escapesBlock ||
      live.hasLocalBoundaryUsers || live.hasLocalHardBoundaryUsers)
    return true;

  if (hasConsumerOutsideGroup(live, groupNodeIds))
    return true;

  return !hasConsumerInGroup(live, groupNodeIds);
}

static bool mustStoreTileOutput(const FusionBlockAnalysis &blockAnalysis,
                                const DenseSet<unsigned> &groupNodeIds,
                                const FusionComputeNode &producer,
                                Value value) {
  if (const FusionWriteInstanceLiveness *write =
          findWriteInstanceForOutput(blockAnalysis, producer.id, value))
    return mustStoreWriteInstance(*write, groupNodeIds);

  const FusionValueLiveness *live = findLivenessForValue(blockAnalysis, value);
  if (!live)
    return true;

  if (live->hasExternalUsers || live->escapesBlock ||
      live->hasLocalBoundaryUsers || live->hasLocalHardBoundaryUsers)
    return true;

  if (hasConsumerOutsideGroup(*live, groupNodeIds))
    return true;

  return !hasConsumerInGroup(*live, groupNodeIds);
}

static FailureOr<int64_t>
computeFlattenTripCount(const FusionBlockAnalysis &blockAnalysis,
                        ArrayRef<const FusionComputeNode *> group) {
  if (group.empty())
    return failure();

  const FusionComputeNode *first = group.front();
  if (first->iterationDomainClass >= blockAnalysis.iterationDomainClasses.size())
    return failure();

  const IterationDomainInfo &info =
      blockAnalysis.iterationDomainClasses[first->iterationDomainClass].info;
  if (info.proof != IterationDomainProof::Proven ||
      info.vRow == ShapedType::kDynamic || info.vCol == ShapedType::kDynamic)
    return failure();

  Type elementType;
  for (const FusionComputeNode *node : group) {
    for (Value output : node->semantics.tileOutputs) {
      if (auto tileType = dyn_cast<pto::TileBufType>(output.getType())) {
        elementType = tileType.getElementType();
        break;
      }
    }
    if (elementType)
      break;
    for (Value input : node->semantics.tileInputs) {
      if (auto tileType = dyn_cast<pto::TileBufType>(input.getType())) {
        elementType = tileType.getElementType();
        break;
      }
    }
    if (elementType)
      break;
  }

  if (!elementType)
    return failure();

  unsigned elemBytes = pto::getPTOStorageElemByteSize(elementType);
  if (elemBytes == 0 || elemBytes > kA5VectorBytes)
    return failure();

  int64_t elemsPerVector = kA5VectorBytes / elemBytes;
  if (elemsPerVector <= 0)
    return failure();

  int64_t elementCount = info.vRow * info.vCol;
  return (elementCount + elemsPerVector - 1) / elemsPerVector;
}

static LogicalResult appendLoadIfNeeded(VfSimProgramBuilder &builder,
                                        std::vector<VfSimNode> &body,
                                        DenseMap<Value, VfSimOperand> &valueToReg,
                                        Value input) {
  if (valueToReg.contains(input))
    return success();

  FailureOr<VfSimOperand> source = builder.getTileValueOperand(input);
  FailureOr<VfSimOperand> reg = builder.makeVirtualReg(input);
  if (failed(source) || failed(reg))
    return failure();
  body.push_back(
      makeInstNode(VfSimInst{VfOpcode::VLDS, {}, {*reg}, {*source}}));
  valueToReg.try_emplace(input, *reg);
  return success();
}

static FailureOr<VfSimOperand>
getTileInputReg(VfSimProgramBuilder &builder, std::vector<VfSimNode> &body,
                DenseMap<Value, VfSimOperand> &valueToReg, Value input) {
  if (failed(appendLoadIfNeeded(builder, body, valueToReg, input)))
    return failure();
  auto it = valueToReg.find(input);
  if (it == valueToReg.end())
    return failure();
  return it->second;
}

static Type getTileElementType(Value value) {
  if (auto tileType = dyn_cast<pto::TileBufType>(value.getType()))
    return tileType.getElementType();
  return {};
}

static std::optional<std::pair<VfOpcode, std::string>>
selectConvertOpcodeAndForm(Type srcType, Type dstType) {
  if (!srcType || !dstType)
    return std::nullopt;
  if (srcType.isF16() && dstType.isF32())
    return std::make_pair(VfOpcode::VCVT_F16_TO_F32, "f16_to_f32");
  if (srcType.isF32() && dstType.isF16())
    return std::make_pair(VfOpcode::VCVT_F32_TO_F16, "f32_to_f16");
  if (srcType.isF32() && dstType.isInteger(32))
    return std::make_pair(VfOpcode::VCVT_F32_TO_S32, "f32_to_s32");
  if (srcType.isInteger(32) && dstType.isF32())
    return std::make_pair(VfOpcode::VCVT_S32_TO_F32, "s32_to_f32");
  return std::nullopt;
}

static FailureOr<VfSimOperand>
emitComputeNode(VfSimProgramBuilder &builder, std::vector<VfSimNode> &body,
                DenseMap<Value, VfSimOperand> &valueToReg,
                const FusionComputeNode &node) {
  std::optional<TileOpPatternSpec> spec =
      lookupTileOpPatternSpec(node.semantics.opName);
  if (!spec)
    return failure();

  SmallVector<VfSimOperand, 4> src;
  for (Value input : node.semantics.tileInputs) {
    FailureOr<VfSimOperand> reg =
        getTileInputReg(builder, body, valueToReg, input);
    if (failed(reg))
      return failure();
    src.push_back(*reg);
  }
  for (Value scalar : node.semantics.scalarInputs) {
    FailureOr<VfSimOperand> operand = builder.getScalarValueOperand(scalar);
    if (failed(operand))
      return failure();
    src.push_back(*operand);
  }

  FailureOr<VfSimOperand> result =
      builder.makeVirtualReg(node.semantics.tileOutputs.front());
  if (failed(result))
    return failure();

  VfOpcode opcode = spec->vectorOpcode;
  std::string form;
  if (node.semantics.opName == "tcvt") {
    if (node.semantics.tileInputs.size() != 1 ||
        node.semantics.tileOutputs.size() != 1)
      return failure();
    std::optional<std::pair<VfOpcode, std::string>> selected =
        selectConvertOpcodeAndForm(
            getTileElementType(node.semantics.tileInputs.front()),
            getTileElementType(node.semantics.tileOutputs.front()));
    if (!selected)
      return failure();
    opcode = selected->first;
    form = selected->second;
  }

  body.push_back(makeInstNode(
      VfSimInst{opcode, form, {*result}, std::move(src)}));
  valueToReg[node.semantics.tileOutputs.front()] = *result;
  return *result;
}

} // namespace

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
          .Case("tdiv", PatternKey{TilePatternKind::BinaryElementwise,
                                    VfOpcode::VDIV, 2, 0})
          .Case("tmax", PatternKey{TilePatternKind::BinaryElementwise,
                                    VfOpcode::VMAX, 2, 0})
          .Case("tmin", PatternKey{TilePatternKind::BinaryElementwise,
                                    VfOpcode::VMIN, 2, 0})
          .Case("texp", PatternKey{TilePatternKind::UnaryElementwise,
                                    VfOpcode::VEXP, 1, 0})
          .Case("tcvt", PatternKey{TilePatternKind::UnaryElementwise,
                                    VfOpcode::VCVT_F32_TO_F16, 1, 0})
          .Case("tadds", PatternKey{TilePatternKind::ScaleElementwise,
                                     VfOpcode::VADDS, 1, 1})
          .Case("tsubs", PatternKey{TilePatternKind::ScaleElementwise,
                                     VfOpcode::VSUBS, 1, 1})
          .Case("tmuls", PatternKey{TilePatternKind::ScaleElementwise,
                                     VfOpcode::VMULS, 1, 1})
          .Case("tdivs", PatternKey{TilePatternKind::ScaleElementwise,
                                     VfOpcode::VDIVS, 1, 1})
          .Case("tmaxs", PatternKey{TilePatternKind::ScaleElementwise,
                                     VfOpcode::VMAXS, 1, 1})
          .Case("tmins", PatternKey{TilePatternKind::ScaleElementwise,
                                     VfOpcode::VMINS, 1, 1})
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

FailureOr<VfSimProgram>
buildFusedElementwiseVfSimProgram(const VfCostInput &input) {
  if (!input.blockAnalysis || !input.candidate)
    return failure();

  SmallVector<const FusionComputeNode *, 8> proposedGroup(
      input.currentGroup.begin(), input.currentGroup.end());
  proposedGroup.push_back(input.candidate);
  proposedGroup = buildStableNodeOrder(proposedGroup);

  DenseSet<unsigned> groupNodeIds;
  for (const FusionComputeNode *node : proposedGroup) {
    if (!node || !isSupportedVfCostTileOp(*node))
      return failure();
    groupNodeIds.insert(node->id);
    if (node->iterationDomainClass >=
        input.blockAnalysis->iterationDomainClasses.size())
      return failure();
    const IterationDomainClass &domain =
        input.blockAnalysis->iterationDomainClasses[node->iterationDomainClass];
    if (domain.info.proof != IterationDomainProof::Proven)
      return failure();
    if (node->iterationDomainClass != proposedGroup.front()->iterationDomainClass)
      return failure();
  }

  FailureOr<int64_t> tripCount =
      computeFlattenTripCount(*input.blockAnalysis, proposedGroup);
  if (failed(tripCount))
    return failure();

  VfSimProgramBuilder builder;
  VfSimProgram program;
  std::vector<VfSimNode> loopBody;

  DenseMap<Value, VfSimOperand> valueToReg;
  for (const FusionComputeNode *node : proposedGroup) {
    if (failed(emitComputeNode(builder, loopBody, valueToReg, *node)))
      return failure();
  }

  for (const FusionComputeNode *node : proposedGroup) {
    for (Value output : node->semantics.tileOutputs) {
      if (!mustStoreTileOutput(*input.blockAnalysis, groupNodeIds, *node,
                               output))
        continue;

      auto regIt = valueToReg.find(output);
      if (regIt == valueToReg.end())
        return failure();

      FailureOr<VfSimOperand> destination = builder.getTileValueOperand(output);
      if (failed(destination))
        return failure();
      loopBody.push_back(makeInstNode(
          VfSimInst{VfOpcode::VSTS, {}, {*destination}, {regIt->second}}));
    }
  }

  program.body.push_back(makeLoopNode(*tripCount, 1, std::move(loopBody)));
  return program;
}

} // namespace pto
} // namespace mlir
