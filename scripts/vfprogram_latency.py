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

    payload = parse_vfprogram_text(args.dump.read_text(encoding="utf-8"))
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
