# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Reusable module-backed artifacts for PTODSL tracing frontends."""

from __future__ import annotations

from pathlib import Path


class ModuleArtifact:
    """
    Cached module-backed artifact.

    Subclasses may either pass an eager ``module`` or a lazy ``module_factory``.
    """

    def __init__(self, py_name: str, *, module=None, module_factory=None):
        self._py_name = py_name
        self._cached_module = module
        self._module_factory = module_factory

    def build(self):
        """Return the cached ``mlir.ir.Module``."""
        if self._cached_module is None:
            if self._module_factory is None:
                raise RuntimeError(f"{self._py_name} has no module factory")
            self._cached_module = self._module_factory()
        return self._cached_module

    def mlir_module(self):
        """Return the cached ``mlir.ir.Module``."""
        return self.build()

    def mlir_text(self) -> str:
        """Return the textual MLIR form."""
        return str(self.build())

    def verify(self) -> None:
        """Verify the cached module operation."""
        self.build().operation.verify()

    def emit(self, path: str | Path) -> None:
        """Write the textual MLIR form to *path*."""
        Path(path).write_text(self.mlir_text(), encoding="utf-8")

    def __str__(self):
        return self.mlir_text()

    def __repr__(self):
        return self.mlir_text()


__all__ = ["ModuleArtifact"]
