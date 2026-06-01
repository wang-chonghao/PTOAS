# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared tracing runtime building blocks for PTODSL frontends."""

from .active import (
    activate_runtime,
    activate_session,
    current_runtime,
    current_session,
    require_active_runtime,
    require_active_session,
)
from .artifacts import ModuleArtifact
from .module_builder import KernelModuleSpec, ModuleStyle, create_kernel_module
from .runtime import CallbackTracingRuntime, SignatureTracingRuntime, TracingRuntime
from .session import HelperFunctionSpec, SubkernelTraceFrame, TraceSession

__all__ = [
    "activate_runtime",
    "activate_session",
    "current_runtime",
    "current_session",
    "require_active_runtime",
    "require_active_session",
    "ModuleArtifact",
    "KernelModuleSpec",
    "ModuleStyle",
    "create_kernel_module",
    "CallbackTracingRuntime",
    "SignatureTracingRuntime",
    "TracingRuntime",
    "HelperFunctionSpec",
    "SubkernelTraceFrame",
    "TraceSession",
]
