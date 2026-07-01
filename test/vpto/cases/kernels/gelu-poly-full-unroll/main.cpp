#include "acl/acl.h"
#include "test_common.h"
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                                          \
  do {                                                                                           \
    const aclError ret = (expr);                                                                 \
    if (ret != ACL_SUCCESS) {                                                                    \
      std::fprintf(stderr, "[ERROR] %s failed: %d\n", #expr, (int)ret);                         \
      rc = 1;                                                                                    \
      goto cleanup;                                                                              \
    }                                                                                            \
  } while (0)

void LaunchGeluPolyFull(float *x, float *out, void *stream);

int main() {
  constexpr size_t elemCount = 16 * 64;
  constexpr size_t fileSize = elemCount * sizeof(float);
  float *xHost = nullptr;
  float *outHost = nullptr;
  float *xDevice = nullptr;
  float *outDevice = nullptr;
  size_t inputSize = fileSize;
  int rc = 0;
  int deviceId = 0;
  bool aclInited = false;
  bool deviceSet = false;
  aclrtStream stream = nullptr;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost((void **)&xHost, fileSize));
  ACL_CHECK(aclrtMallocHost((void **)&outHost, fileSize));
  ACL_CHECK(aclrtMalloc((void **)&xDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));

  if (!ReadFile("./v1.bin", inputSize, xHost, fileSize)) {
    std::fprintf(stderr, "[ERROR] failed to read v1.bin\n");
    rc = 1;
    goto cleanup;
  }
  for (size_t i = 0; i < elemCount; ++i)
    outHost[i] = 0.0f;

  ACL_CHECK(aclrtMemcpy(xDevice, fileSize, xHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outDevice, fileSize, outHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchGeluPolyFull(xDevice, outDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(outHost, fileSize, outDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST));

  if (!WriteFile("./v2.bin", outHost, fileSize)) {
    std::fprintf(stderr, "[ERROR] failed to write v2.bin\n");
    rc = 1;
  }

cleanup:
  if (xDevice)
    aclrtFree(xDevice);
  if (outDevice)
    aclrtFree(outDevice);
  if (xHost)
    aclrtFreeHost(xHost);
  if (outHost)
    aclrtFreeHost(outHost);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
