"""CLI helper invoked by ExpandTileOp to instantiate a tilelang DSL template.

Usage:
    python3 -m tilelang_dsl.expand_helper \
        --template-dir /path/to/templates \
        --op pto.tadd \
        --dtype f32 \
        --shape 16,64 \
        --memory-space ub

Scans --template-dir for .py files, finds a @vkernel whose `op` matches,
specializes every Tile parameter with the given shape/memory_space, and
prints the materialized MLIR module to stdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .kernel import VKernelDescriptor, _match_descriptor_dtype_signature
from .types import MemorySpace, ScalarType, TileSpecialization


_DTYPE_MAP: dict[str, ScalarType] = {}


def _populate_dtype_map() -> None:
    from . import types as _t

    for name in ("f16", "bf16", "f32", "i8", "i16", "i32", "i64"):
        obj = getattr(_t, name, None)
        if isinstance(obj, ScalarType):
            _DTYPE_MAP[name] = obj


_populate_dtype_map()

_MEMSPACE_MAP = {
    "ub": MemorySpace.UB,
    "gm": MemorySpace.GM,
}


def _find_descriptors(module) -> list[VKernelDescriptor]:
    """Return all VKernelDescriptor instances found as module-level attributes."""
    result = []
    for attr_name in dir(module):
        obj = getattr(module, attr_name, None)
        if isinstance(obj, VKernelDescriptor):
            result.append(obj)
    return result


def _import_py_file(path: Path):
    """Import a .py file as a module and return it."""
    spec = importlib.util.spec_from_file_location(f"_tl_template_{path.stem}", str(path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"expand_helper: warning: failed to import {path}: {exc}", file=sys.stderr)
        return None
    return mod


def _match_descriptor(
    descriptors: list[VKernelDescriptor],
    op_name: str,
    operand_types: tuple[ScalarType, ...],
) -> VKernelDescriptor | None:
    """Find and bind the first descriptor matching (op, dtype)."""
    for desc in descriptors:
        if op_name not in desc.match_ops:
            continue
        op_bound = desc._bind_selected_op(op_name)
        matched_signature = _match_descriptor_dtype_signature(op_bound, operand_types)
        if matched_signature is None:
            continue
        if op_bound._selected_dtype_signature == matched_signature:
            return op_bound
        return op_bound._bind_selected_dtype_signature(matched_signature)
    return None


def _parse_operand_specs(spec_text: str) -> list[dict]:
    try:
        raw_specs = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid operand-specs JSON: {exc}") from exc

    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("operand-specs must be a non-empty JSON array")

    specs: list[dict] = []
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            raise ValueError(f"operand-specs[{index}] must be an object")
        kind = raw.get("kind")
        dtype_name = raw.get("dtype")
        dtype = _DTYPE_MAP.get(dtype_name)
        if dtype is None:
            raise ValueError(f"operand-specs[{index}] has unsupported dtype {dtype_name!r}")
        if kind == "scalar":
            specs.append({"kind": "scalar", "dtype": dtype})
            continue
        if kind == "tile":
            shape = raw.get("shape")
            if not isinstance(shape, list) or not shape:
                raise ValueError(f"operand-specs[{index}] tile shape must be a non-empty list")
            memory_space = _MEMSPACE_MAP.get(raw.get("memory_space"))
            if memory_space is None:
                raise ValueError(
                    f"operand-specs[{index}] has unknown memory-space {raw.get('memory_space')!r}"
                )
            specs.append(
                {
                    "kind": "tile",
                    "dtype": dtype,
                    "shape": tuple(int(dim) for dim in shape),
                    "memory_space": memory_space,
                }
            )
            continue
        raise ValueError(f"operand-specs[{index}] has unknown kind {kind!r}")
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TileLang DSL expand helper")
    parser.add_argument("--template-dir", required=True, help="Directory of .py templates")
    parser.add_argument("--op", required=True, help="Tile op name, e.g. pto.tadd")
    parser.add_argument("--dtype", help="Element dtype, e.g. f32")
    parser.add_argument("--shape", help="Tile shape, e.g. 16,64")
    parser.add_argument("--memory-space", default="ub", help="Memory space (ub or gm)")
    parser.add_argument(
        "--operand-specs",
        help="JSON array describing each operand (tile/scalar schema)",
    )
    args = parser.parse_args(argv)

    template_dir = Path(args.template_dir)
    if not template_dir.is_dir():
        print(f"expand_helper: error: {template_dir} is not a directory", file=sys.stderr)
        return 1

    operand_specs: list[dict] | None = None
    if args.operand_specs:
      try:
          operand_specs = _parse_operand_specs(args.operand_specs)
      except ValueError as exc:
          print(f"expand_helper: error: {exc}", file=sys.stderr)
          return 1
    else:
        if args.dtype is None or args.shape is None:
            print(
                "expand_helper: error: either --operand-specs or both --dtype/--shape are required",
                file=sys.stderr,
            )
            return 1
        shape = tuple(int(d) for d in args.shape.split(","))
        mem_space = _MEMSPACE_MAP.get(args.memory_space)
        if mem_space is None:
            print(f"expand_helper: error: unknown memory-space '{args.memory_space}'", file=sys.stderr)
            return 1
        target_dtype = _DTYPE_MAP.get(args.dtype)
        if target_dtype is None:
            print(f"expand_helper: error: unknown dtype '{args.dtype}'", file=sys.stderr)
            return 1
        operand_specs = [
            {"kind": "tile", "dtype": target_dtype, "shape": shape, "memory_space": mem_space}
        ]

    # Scan all .py files for descriptors.
    all_descriptors: list[VKernelDescriptor] = []
    for py_path in sorted(template_dir.glob("*.py")):
        mod = _import_py_file(py_path)
        if mod is None:
            continue
        all_descriptors.extend(_find_descriptors(mod))

    if not all_descriptors:
        print(f"expand_helper: error: no @vkernel descriptors found in {template_dir}", file=sys.stderr)
        return 1

    # Match.
    operand_types = tuple(spec["dtype"] for spec in operand_specs)
    desc = _match_descriptor(all_descriptors, args.op, operand_types)
    if desc is None:
        print(
            f"expand_helper: error: no template matches op={args.op} operand_types={operand_types!r}",
            file=sys.stderr,
        )
        return 1

    if len(desc.parameters) != len(operand_specs):
        print(
            "expand_helper: error: descriptor parameter count does not match operand-specs",
            file=sys.stderr,
        )
        return 1

    # Specialize Tile parameters positionally from operand-specs.
    tile_specs = {}
    for param, operand_spec in zip(desc.parameters, operand_specs):
        if param.kind == "tile":
            if operand_spec["kind"] != "tile":
                print(
                    "expand_helper: error: descriptor tile parameter does not match operand-specs",
                    file=sys.stderr,
                )
                return 1
            tile_specs[param.name] = TileSpecialization(
                shape=operand_spec["shape"],
                memory_space=operand_spec["memory_space"],
            )
            continue
        if param.kind == "scalar" and operand_spec["kind"] != "scalar":
            print(
                "expand_helper: error: descriptor scalar parameter does not match operand-specs",
                file=sys.stderr,
            )
            return 1

    specialized = desc.specialize(**tile_specs)

    # Emit MLIR to stdout.
    try:
        mlir_text = specialized.mlir_text()
    except Exception as exc:
        print(f"expand_helper: error: materialization failed: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(mlir_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
