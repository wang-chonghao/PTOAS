#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


ROWS = 16
COLS = 64
SEED = 23


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 5.0, size=(ROWS, COLS)).astype(np.float32)
    x_clamp = np.minimum(np.maximum(x, np.float32(-4.0)), np.float32(4.0)).astype(np.float32)
    x2 = (x_clamp * x_clamp).astype(np.float32)
    x3 = (x2 * x_clamp).astype(np.float32)
    poly = (x3 * np.float32(0.0447)).astype(np.float32)
    poly = (poly + np.float32(0.3989)).astype(np.float32)
    gate = (poly * x_clamp).astype(np.float32)
    gate = (gate + np.float32(0.5)).astype(np.float32)
    out = (x_clamp * gate).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    x.reshape(-1).tofile(output_dir / "v1.bin")
    np.zeros_like(x).reshape(-1).tofile(output_dir / "v2.bin")
    out.reshape(-1).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
