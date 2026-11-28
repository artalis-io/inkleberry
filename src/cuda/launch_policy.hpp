/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Host launch helper extracted from the pinned kernels_common.cuh; adds an
 * explicit experimental target policy without changing kernel arguments. */
#ifndef IFN_CUDA_LAUNCH_POLICY_HPP
#define IFN_CUDA_LAUNCH_POLICY_HPP
#include "target_profile.h"

/* Included inside namespace ifn_cuda after CUDA runtime types and the checked
 * error macro. The Blackwell launch configuration is intentionally unchanged. */
template <class... KArgs, class... Args>
inline void ifn_cuda_launch_pdl(void (*kernel)(KArgs...), dim3 grid, int block, cudaStream_t stream,
                                Args &&...args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(block);
    cfg.stream = stream;
#if IFN_CUDA_USE_PDL
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
#endif
    IFN_CUDA_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}
#endif
