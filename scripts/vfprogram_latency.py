#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


LOOP_RE = re.compile(r"^loop\s+\d+\s+trip_count=(\d+)\s+unroll=(\d+)\s*$")
RESULT_INST_RE = re.compile(r"^(reg\d+)\s*=\s*([a-zA-Z0-9_]+)(?:\s+(.*))?$")
VOID_INST_RE = re.compile(r"^([a-zA-Z0-9_]+)(?:\s+(.*))?$")


def _split_operands(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _operand_name(operand: str) -> str:
    if operand.startswith("reg"):
        return "V" + operand[3:]
    if operand.startswith("tile"):
        return "mem" + operand[4:]
    if operand.startswith("scalar"):
        return ""
    raise ValueError(f"unsupported VFProgram operand: {operand}")


def _operand_obj_name(operand: dict[str, Any]) -> str:
    kind = str(operand.get("kind", "")).lower()
    operand_id = int(operand.get("id"))
    if kind == "reg":
        return f"V{operand_id}"
    if kind == "tile":
        return f"mem{operand_id}"
    if kind == "scalar":
        return ""
    raise ValueError(f"unsupported VFProgram operand kind: {kind}")


def _operand_obj_dtype(operand: Any) -> str:
    if not isinstance(operand, dict):
        return ""
    return str(operand.get("dtype", "") or "")


def _infer_inst_form(op: str, dst: list[Any], src: list[Any], explicit_form: str) -> str:
    if explicit_form:
        return explicit_form
    op = op.upper()
    if op == "VLDS":
        return _operand_obj_dtype(dst[0]) if dst else ""
    if op == "VSTS":
        return _operand_obj_dtype(src[-1]) if src else ""
    if dst:
        return _operand_obj_dtype(dst[0])
    for operand in src:
        dtype = _operand_obj_dtype(operand)
        if dtype:
            return dtype
    return ""


def _operand_names(operands: list[str], *, drop_scalars: bool = False) -> list[str]:
    names = [_operand_name(operand) for operand in operands]
    if drop_scalars:
        names = [name for name in names if name]
    return names


def parse_vfprogram_text(text: str) -> dict[str, Any]:
    loops: list[dict[str, Any]] = []
    current_loop: dict[str, Any] | None = None
    in_dump = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[pto-fusion-plan] VF program"):
            in_dump = True
            current_loop = None
            continue
        if not in_dump:
            continue

        loop_match = LOOP_RE.match(line)
        if loop_match:
            current_loop = {
                "type": "loop",
                "iters": int(loop_match.group(1)),
                "unroll": int(loop_match.group(2)),
                "body": [],
            }
            loops.append(current_loop)
            continue

        if current_loop is None:
            continue

        result_match = RESULT_INST_RE.match(line)
        if result_match:
            dst = _operand_name(result_match.group(1))
            op = result_match.group(2).upper()
            src = _operand_names(
                _split_operands(result_match.group(3)),
                drop_scalars=op.endswith("S"),
            )
            current_loop["body"].append({
                "type": "inst",
                "op": op,
                "dst": [dst],
                "src": src,
            })
            continue

        void_match = VOID_INST_RE.match(line)
        if void_match:
            op = void_match.group(1).upper()
            operands = [_operand_name(operand) for operand in _split_operands(void_match.group(2))]
            if op == "VSTS":
                if len(operands) != 2:
                    raise ValueError(f"VSTS expects destination and source: {line}")
                dst = [operands[0]]
                src = [operands[1]]
            else:
                dst = []
                src = operands
            current_loop["body"].append({
                "type": "inst",
                "op": op,
                "dst": dst,
                "src": src,
            })
            continue

        raise ValueError(f"failed to parse VFProgram line: {raw_line}")

    if not loops:
        raise ValueError("no VFProgram dump found")

    return {
        "dtype": "fp32",
        "params": {},
        "program": loops,
    }


def _convert_vfprogram_json_program(vf_program: dict[str, Any]) -> dict[str, Any]:
    if "body" in vf_program:
        return {
            "dtype": "fp32",
            "params": {},
            "program": [_convert_vfsim_json_node(node) for node in vf_program["body"]],
        }

    loops = []
    for loop in vf_program.get("loops", []):
        body = []
        for inst in loop.get("instructions", []):
            op = str(inst.get("op", "")).upper()
            dst = _operand_names([str(x) for x in inst.get("dst", [])])
            src = _operand_names(
                [str(x) for x in inst.get("src", [])],
                drop_scalars=op.endswith("S"),
            )
            body.append({
                "type": "inst",
                "op": op,
                "dst": dst,
                "src": src,
            })
        loops.append({
            "type": "loop",
            "iters": int(loop.get("trip_count", 1)),
            "unroll": int(loop.get("unroll", 1)),
            "body": body,
        })
    if not loops:
        raise ValueError("VFProgram JSON contains no loops")
    return {
        "dtype": "fp32",
        "params": {},
        "program": loops,
    }


def _convert_vfsim_json_operands(operands: list[Any], *, drop_scalars: bool) -> list[str]:
    names = []
    for operand in operands:
        if isinstance(operand, dict):
            names.append(_operand_obj_name(operand))
        else:
            names.append(_operand_name(str(operand)))
    if drop_scalars:
        names = [name for name in names if name]
    return names


def _convert_vfsim_json_node(node: dict[str, Any]) -> dict[str, Any]:
    node_type = str(node.get("type"))
    if node_type == "loop":
        return {
            "type": "loop",
            "iters": int(node.get("trip_count", 1)),
            "unroll": int(node.get("unroll", 1)),
            "body": [_convert_vfsim_json_node(child) for child in node.get("body", [])],
        }
    if node_type == "inst":
        op = str(node.get("op", "")).upper()
        dst_operands = node.get("dst", [])
        src_operands = node.get("src", [])
        inst = {
            "type": "inst",
            "op": op,
            "dst": _convert_vfsim_json_operands(
                dst_operands,
                drop_scalars=False,
            ),
            "src": _convert_vfsim_json_operands(
                src_operands,
                drop_scalars=op.endswith("S"),
            ),
        }
        form = _infer_inst_form(op, dst_operands, src_operands, str(node.get("form", "") or ""))
        if form:
            inst["form"] = form
        return inst
    raise ValueError(f"unsupported VFProgram JSON node type: {node_type}")


def load_vfprogram_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return parse_vfprogram_text(text)

    root = json.loads(text)
    if "program" in root:
        return root
    programs = root.get("programs")
    if isinstance(programs, list) and programs:
        return _convert_vfprogram_json_program(programs[0]["vf_program"])
    if "loops" in root:
        return _convert_vfprogram_json_program(root)
    raise ValueError("unsupported VFProgram JSON format")


def predict_latency(payload: dict[str, Any], vfsim_root: Path, out_dir: Path) -> int:
    sys.path.insert(0, str(vfsim_root))
    from api.simulator_costmodel import CoreVfCostModel

    _validate_supported_opcodes(payload, vfsim_root)
    model = CoreVfCostModel(base_dir=vfsim_root, out_dir=out_dir, dtype=payload["dtype"])
    return int(model.run_payload(payload)["vf_end_cycle"])


def _iter_instructions(nodes: list[dict[str, Any]]):
    for node in nodes:
        if node.get("type") == "inst":
            yield node
        elif node.get("type") == "loop":
            yield from _iter_instructions(node.get("body", []))


def _validate_supported_opcodes(payload: dict[str, Any], vfsim_root: Path) -> None:
    isa_path = vfsim_root / "configs" / "isa.json"
    isa = json.loads(isa_path.read_text(encoding="utf-8"))
    supported = set((isa.get("instructions") or {}).keys())
    unsupported = sorted({
        str(inst.get("op"))
        for inst in _iter_instructions(payload.get("program", []))
        if str(inst.get("op")) not in supported
    })
    if unsupported:
        ops = ", ".join(unsupported)
        raise RuntimeError(
            "VFProgram contains micro-op(s) not supported by the current "
            f"VfSimulator ISA config: {ops}. The corresponding tileop-to-micro-op "
            "mapping exists in PTOAS, but latency prediction is unavailable until "
            "VfSimulator adds these instruction entries."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PTOAS --dump-vf-program text to a VfSimulator payload and report latency.",
    )
    parser.add_argument("dump", type=Path, help="File containing PTOAS --dump-vf-program stderr text.")
    parser.add_argument("--vfsim-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/ptoas_vfsim_latency"))
    parser.add_argument("--payload-out", type=Path, default=None)
    args = parser.parse_args()

    payload = load_vfprogram_payload(args.dump)
    payload["dtype"] = args.dtype

    if args.payload_out:
        args.payload_out.parent.mkdir(parents=True, exist_ok=True)
        args.payload_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        cycles = predict_latency(payload, args.vfsim_root.resolve(), args.out_dir)
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(1)
    print(f"vf_cycles={cycles}")


if __name__ == "__main__":
    main()
