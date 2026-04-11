#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import sys
import os
import numpy as np

ANSI_RESET = "\033[0m"
ANSI_BOLD_GREEN = "\033[1;32m"
ANSI_BOLD_RED = "\033[1;31m"


CASES = [
    {"name": "f32_16x64", "dtype": np.float32, "eps": 1e-6},
    {"name": "f32_32x32", "dtype": np.float32, "eps": 1e-6},
]


def supports_color():
    return sys.stdout.isatty() and os.environ.get("TERM") not in (None, "", "dumb")


def style_pass(text):
    if not supports_color():
        return text
    return f"{ANSI_BOLD_GREEN}{text}{ANSI_RESET}"


def style_fail(text):
    if not supports_color():
        return text
    return f"{ANSI_BOLD_RED}{text}{ANSI_RESET}"


def compare_bin(golden_path, output_path, dtype, eps):
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    if golden.shape != output.shape:
        print(style_fail(f"[ERROR] Shape mismatch: golden {golden.shape} vs output {output.shape}"))
        return False
    if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        g = golden.astype(np.float64, copy=False)
        o = output.astype(np.float64, copy=False)
        abs_diff = np.abs(g - o)
        idx = int(np.argmax(abs_diff))
        print(style_fail(f"[ERROR] Mismatch: max diff={float(abs_diff[idx])} at idx={idx} "
                         f"(golden={g[idx]}, output={o[idx]})"))
        return False
    return True


if __name__ == "__main__":
    # Optional filter: python compare.py [case_name]
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue
        case_dir = case["name"]
        golden_path = os.path.join(case_dir, "golden.bin")
        output_path = os.path.join(case_dir, "output.bin")
        ok = compare_bin(golden_path, output_path, case["dtype"], case["eps"])
        if ok:
            print(style_pass(f"[INFO] {case['name']}: compare passed"))
        else:
            print(style_fail(f"[ERROR] {case['name']}: compare failed"))
            all_passed = False

    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed"))
