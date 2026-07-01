#!/usr/bin/env python3

import os
import sys

import numpy as np


def main() -> None:
    if not os.path.exists("v2.bin"):
        print("[ERROR] Output missing: v2.bin")
        sys.exit(2)
    if not os.path.exists("golden_v2.bin"):
        print("[ERROR] Golden missing: golden_v2.bin")
        sys.exit(2)

    golden = np.fromfile("golden_v2.bin", dtype=np.float32)
    output = np.fromfile("v2.bin", dtype=np.float32)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden.shape} vs {output.shape}")
        sys.exit(2)
    if not np.allclose(golden, output, atol=1e-4, rtol=1e-4, equal_nan=True):
        diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
        idx = int(np.argmax(diff))
        print(
            f"[ERROR] Mismatch: max diff={float(diff[idx])} at idx={idx} "
            f"(golden={float(golden[idx])}, out={float(output[idx])})"
        )
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
