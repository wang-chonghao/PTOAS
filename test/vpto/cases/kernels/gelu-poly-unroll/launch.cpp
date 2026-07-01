// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to LICENSE in the root of the software repository for the full text of the License.

#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t;
typedef struct { unsigned char v; } float8_e5m2_t;
typedef struct { unsigned char v; } float8_e8m0_t;
typedef struct { unsigned char v; } float4_e1m2x2_t;
typedef struct { unsigned char v; } float4_e2m1x2_t;
#endif

#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void fusion_loop_unroll_gelu_poly_vfsim(
    __gm__ float *x, __gm__ float *out);

void LaunchFusionLoopUnrollGeluPoly(float *x, float *out, void *stream) {
  fusion_loop_unroll_gelu_poly_vfsim<<<1, nullptr, stream>>>(
      (__gm__ float *)x, (__gm__ float *)out);
}
