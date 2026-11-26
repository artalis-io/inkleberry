/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Test-owned driver context: library calls must restore it and its allocation. */
#include "ifn_gpu.h"
#include <cuda.h>
#include <cstdio>

extern "C" int ifn_test_context(ifn_gpu_weights **image, int destroy) {
    CUcontext previous = nullptr, owned = nullptr, current = nullptr;
    CUdevice device;
    CUdeviceptr sentinel = 0;
    unsigned char bytes[64] = {};
    int failed = 0;
    if (cuCtxGetCurrent(&previous) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&owned, nullptr, 0, device) != CUDA_SUCCESS)
        return 1;
    if (cuMemAlloc(&sentinel, sizeof bytes) != CUDA_SUCCESS ||
        cuMemsetD8(sentinel, 0xa5, sizeof bytes) != CUDA_SUCCESS)
        failed = 1;
    if (!failed && (ifn_gpu_weights_read(*image, 0, 0, bytes, sizeof bytes) != IFN_OK ||
                    (destroy && ifn_gpu_weights_destroy(image) != IFN_OK)))
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
    if (cuCtxSetCurrent(previous) != CUDA_SUCCESS || cuCtxDestroy(owned) != CUDA_SUCCESS)
        failed = 1;
    if (failed)
        std::fputs("caller CUDA context/sentinel preservation failed\n", stderr);
    return failed;
}
