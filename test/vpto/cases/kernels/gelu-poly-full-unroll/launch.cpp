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

extern "C" __global__ [aicore] void gelu_poly_full_unroll_vfsim(
    __gm__ float *x, __gm__ float *out);

void LaunchGeluPolyFull(float *x, float *out, void *stream) {
  gelu_poly_full_unroll_vfsim<<<1, nullptr, stream>>>(
      (__gm__ float *)x, (__gm__ float *)out);
}
