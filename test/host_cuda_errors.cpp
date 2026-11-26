/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "config.hpp"
#include "../src/checked_cuda.h"
#include <thread>
#include <cstdio>

int main() {
    if (!ifn_cuda_capability_supported(12, 0) || !ifn_cuda_capability_supported(12, 1) ||
        ifn_cuda_capability_supported(12, 2) || ifn_cuda_capability_supported(13, 0) ||
        ifn_cuda_capability_supported(11, 0) || ifn_cuda_capability_supported(12, -1))
        return 3;
    IFN_CUDA_CUDA_CHECK(0);
    if (ifn_cuda::ifn_cuda_take_error())
        return 1;
    IFN_CUDA_CUDA_CHECK(801);
    IFN_CUDA_CUDA_CHECK(2);
    int child = -1;
    std::thread t([&] {
        if (ifn_cuda::ifn_cuda_take_error())
            return;
        IFN_CUDA_CUDA_CHECK(700);
        child = ifn_cuda::ifn_cuda_take_error();
    });
    t.join();
    if (child != 700 || ifn_cuda::ifn_cuda_take_error() != 801 || ifn_cuda::ifn_cuda_take_error())
        return 2;
    puts("CUDA host macro returned safely; first error preserved; threads isolated");
    return 0;
}
