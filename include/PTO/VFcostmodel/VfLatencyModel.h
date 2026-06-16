// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_VFCOSTMODEL_VFLATENCYMODEL_H
#define PTO_VFCOSTMODEL_VFLATENCYMODEL_H

#include "PTO/VFcostmodel/VfSimProgram.h"

#include <cstdint>
#include <memory>
#include <string>

namespace mlir {
namespace pto {

struct VfLatencyResult {
  bool supported = false;
  int64_t cycles = 0;
  std::string rejectReason;
};

class VfLatencyModel {
public:
  virtual ~VfLatencyModel() = default;

  virtual VfLatencyResult predict(const VfSimProgram &program) const = 0;
};

std::unique_ptr<VfLatencyModel> createVfLatencyModel();

} // namespace pto
} // namespace mlir

#endif // PTO_VFCOSTMODEL_VFLATENCYMODEL_H
