/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn_engine.h"
#include <cuda.h>

extern "C" int ifn_ar_test_context(ifn_session *session) {
    CUcontext old = nullptr, owned = nullptr, current = nullptr;
    CUdevice device;
    CUdeviceptr sentinel = 0;
    unsigned char bytes[32] = {};
    int failed = 0;
    if (cuCtxGetCurrent(&old) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&owned, nullptr, 0, device) != CUDA_SUCCESS)
        return 1;
    if (cuMemAlloc(&sentinel, sizeof bytes) != CUDA_SUCCESS ||
        cuMemsetD8(sentinel, 0xa5, sizeof bytes) != CUDA_SUCCESS)
        failed = 1;
    if (!failed && ifn_session_reset(session) != IFN_OK)
        failed = 1;
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != owned ||
        cuMemcpyDtoH(bytes, sentinel, sizeof bytes) != CUDA_SUCCESS)
        failed = 1;
    for (unsigned char b : bytes)
        if (b != 0xa5)
            failed = 1;
    if (cuCtxSetCurrent(owned) != CUDA_SUCCESS)
        failed = 1;
    if (sentinel && cuMemFree(sentinel) != CUDA_SUCCESS)
        failed = 1;
    if (cuCtxSetCurrent(old) != CUDA_SUCCESS || cuCtxDestroy(owned) != CUDA_SUCCESS)
        failed = 1;
    return failed;
}
