// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/VFcostmodel/VfLatencyModel.h"

namespace mlir {
namespace pto {
namespace {

class StubVfLatencyModel final : public VfLatencyModel {
public:
  VfLatencyResult predict(const VfSimProgram & /*program*/) const override {
    return VfLatencyResult{
        /*supported=*/false,
        /*cycles=*/0,
        /*rejectReason=*/"C++ VfSimulator is not implemented yet",
    };
  }
};

} // namespace

std::unique_ptr<VfLatencyModel> createVfLatencyModel() {
  return std::make_unique<StubVfLatencyModel>();
}

} // namespace pto
} // namespace mlir
