/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_CUDA_PREFILL_RESOURCES_HPP
#define IFN_CUDA_PREFILL_RESOURCES_HPP
#include "target_profile.h"
#include "resource_policy.hpp"
#define IFN_CUDA_PREFILL_RESOURCE_TRY(call)                                                        \
    do {                                                                                           \
        cudaError_t ifn_error__ = (call);                                                          \
        if (ifn_error__ != cudaSuccess)                                                            \
            return (int)ifn_error__;                                                               \
    } while (0)
#if !IFN_CUDA_USE_PDL
template <int FMT, ifn_cuda::ifn_cuda_wgemm_mode MODE, int K, int TM>
inline cudaError_t ifn_cuda_prepare_wgemm() {
    return ifn_cuda_prepare_shared(ifn_cuda::ifn_cuda_p_wgemm_kernel<FMT, MODE, K, TM>,
                                   ifn_cuda::ifn_cuda_wg_cfg<FMT, K, TM>::SMEM,
                                   ifn_cuda::IFN_CUDA_WG_NT);
}

inline int ifn_cuda_prepare_prefill_resources() {
    IFN_CUDA_PREFILL_RESOURCE_TRY(
        (ifn_cuda_prepare_wgemm<0, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_UP,
                                ifn_cuda::IFN_CUDA_HIDDEN, 32>()));
    IFN_CUDA_PREFILL_RESOURCE_TRY(
        (ifn_cuda_prepare_wgemm<0, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_UP,
                                ifn_cuda::IFN_CUDA_HIDDEN, 128>()));
    IFN_CUDA_PREFILL_RESOURCE_TRY(
        (ifn_cuda_prepare_wgemm<0, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_DOWN,
                                ifn_cuda::IFN_CUDA_INTERMEDIATE, 32>()));
    IFN_CUDA_PREFILL_RESOURCE_TRY(
        (ifn_cuda_prepare_wgemm<0, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_DOWN,
                                ifn_cuda::IFN_CUDA_INTERMEDIATE, 128>()));
    IFN_CUDA_PREFILL_RESOURCE_TRY(
        (ifn_cuda_prepare_wgemm<1, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_UP,
                                ifn_cuda::IFN_CUDA_HIDDEN, 32>()));
    IFN_CUDA_PREFILL_RESOURCE_TRY(
        (ifn_cuda_prepare_wgemm<1, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_UP,
                                ifn_cuda::IFN_CUDA_HIDDEN, 128>()));
    IFN_CUDA_PREFILL_RESOURCE_TRY(
        (ifn_cuda_prepare_wgemm<1, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_DOWN,
                                ifn_cuda::IFN_CUDA_INTERMEDIATE, 32>()));
    IFN_CUDA_PREFILL_RESOURCE_TRY(
        (ifn_cuda_prepare_wgemm<1, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_DOWN,
                                ifn_cuda::IFN_CUDA_INTERMEDIATE, 128>()));
    return 0;
}
#endif

#undef IFN_CUDA_PREFILL_RESOURCE_TRY
#endif
