/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "config.hpp"
#include "../src/cuda/geometry_contract.hpp"
#include "../src/cuda/capability.h"
#include "../src/cuda/resource_policy.hpp"
#include <cstdio>
#include <cstdlib>

namespace ifn_cuda {
#include "../src/cuda/launch_policy.hpp"
}

#if !IFN_CUDA_USE_PDL
// Resource-size oracles from the audited dispatched specializations, not the
// production sizing formula. No numerical or device implementation is copied.
namespace ifn_cuda {
enum class ifn_cuda_wgemm_mode { IFN_CUDA_UP, IFN_CUDA_DOWN };
constexpr int IFN_CUDA_WG_NT = 256;

template <int FMT, int K, int TM> struct ifn_cuda_wg_cfg {
    static constexpr int SMEM =
        FMT == 0 ? (K == 2304 ? (TM == 32 ? 92160 : 98304) : (TM == 32 ? 82944 : 89088))
                 : (K == 2304 ? (TM == 32 ? 93184 : 99328) : (TM == 32 ? 88576 : 94720));
};

template <int FMT, ifn_cuda_wgemm_mode MODE, int K, int TM> void ifn_cuda_p_wgemm_kernel() {
    ifn_cuda_stub.set_bytes = FMT + static_cast<int>(MODE) * 2 + K * 4 + TM * 10000;
}
} // namespace ifn_cuda

#endif
#include "../src/cuda/prefill_resources.hpp"
#define IFN_POLICY_CHECK(x)                                                                        \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            std::fprintf(stderr, "policy:%d: %s\n", __LINE__, #x);                                 \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

static void ifn_policy_kernel(int value, int *out) {
    *out = value;
}

int main(int argc, char **argv) {
    if (argc != 4)
        return 2;
    const int major = std::atoi(argv[1]), minor = std::atoi(argv[2]);
    const bool pdl = std::atoi(argv[3]) != 0;
    for (int m = 0; m < 15; ++m)
        for (int n = -1; n < 10; ++n)
            IFN_POLICY_CHECK(ifn_cuda_capability_supported(m, n) == (m == major && n == minor));
    int result = 0, stream_marker = 0;
    ifn_cuda::ifn_cuda_launch_pdl(ifn_policy_kernel, dim3(3, 2, 1), 128, &stream_marker,
                                  static_cast<short>(37), &result);
    IFN_POLICY_CHECK(
        result == 37 && ifn_cuda_stub.launches == 1 && ifn_cuda_stub.programmatic == pdl &&
        ifn_cuda_stub.launch.numAttrs == (pdl ? 1u : 0u) && (pdl || !ifn_cuda_stub.launch.attrs) &&
        ifn_cuda_stub.launch.stream == &stream_marker && ifn_cuda_stub.launch.gridDim.x == 3 &&
        ifn_cuda_stub.launch.gridDim.y == 2 && ifn_cuda_stub.launch.gridDim.z == 1 &&
        ifn_cuda_stub.launch.blockDim.x == 128 && ifn_cuda_stub.launch.blockDim.y == 1 &&
        ifn_cuda_stub.launch.blockDim.z == 1 && ifn_cuda_stub.launch.dynamicSmemBytes == 0);
    ifn_cuda_stub.launch_error = 700;
    ifn_cuda::ifn_cuda_launch_pdl(ifn_policy_kernel, dim3(1), 32, nullptr, 99, &result);
    ifn_cuda_stub.launch_error = 2;
    ifn_cuda::ifn_cuda_launch_pdl(ifn_policy_kernel, dim3(1), 32, nullptr, 99, &result);
    IFN_POLICY_CHECK(ifn_cuda::ifn_cuda_take_error() == 700 && !ifn_cuda::ifn_cuda_take_error() &&
                     result == 37);
    ifn_cuda_stub = {};
    IFN_POLICY_CHECK(ifn_cuda_prepare_shared(ifn_policy_kernel, 99328, 256) == cudaSuccess &&
                     ifn_cuda_stub.sets == 1 && ifn_cuda_stub.set_bytes == 99328);
    // Recheck another context/device rather than inheriting successful opt-in.
    ifn_cuda_stub.device = 1;
    IFN_POLICY_CHECK(ifn_cuda_prepare_shared(ifn_policy_kernel, 99328, 256) ==
                         cudaErrorInvalidConfiguration &&
                     ifn_cuda_stub.sets == 1 && ifn_cuda_stub.queries == 2);
    ifn_cuda_stub = {};
    ifn_cuda_stub.function.sharedSizeBytes = 3000;
    IFN_POLICY_CHECK(ifn_cuda_prepare_shared(ifn_policy_kernel, 99328, 256) ==
                         cudaErrorInvalidConfiguration &&
                     !ifn_cuda_stub.sets);
    ifn_cuda_stub.function.sharedSizeBytes = static_cast<size_t>(-1);
    IFN_POLICY_CHECK(ifn_cuda_prepare_shared(ifn_policy_kernel, 1, 256) ==
                     cudaErrorInvalidConfiguration);
    ifn_cuda_stub = {};
    ifn_cuda_stub.function.sharedSizeBytes = 1024;
    IFN_POLICY_CHECK(ifn_cuda_prepare_shared(ifn_policy_kernel, 100352, 256) == cudaSuccess);
    IFN_POLICY_CHECK(ifn_cuda_prepare_shared(ifn_policy_kernel, 100353, 256) ==
                         cudaErrorInvalidConfiguration &&
                     ifn_cuda_stub.sets == 1);
    ifn_cuda_stub = {};
    ifn_cuda_stub.limits[0] = -1;
    IFN_POLICY_CHECK(ifn_cuda_prepare_shared(ifn_policy_kernel, 0, 32) ==
                     cudaErrorInvalidConfiguration);
    IFN_POLICY_CHECK(ifn_cuda_prepare_shared(static_cast<void (*)(int, int *)>(nullptr), 0, 32) ==
                     cudaErrorInvalidValue);
    ifn_cuda_stub = {};
    ifn_cuda_stub.function.maxThreadsPerBlock = 128;
    IFN_POLICY_CHECK(ifn_cuda_prepare_shared(ifn_policy_kernel, 4096, 256) ==
                     cudaErrorInvalidConfiguration);
    IFN_POLICY_CHECK(ifn_cuda_prepare_shared(ifn_policy_kernel, -1, 32) == cudaErrorInvalidValue);
    IFN_POLICY_CHECK(ifn_cuda_prepare_shared(ifn_policy_kernel, 1, 0) == cudaErrorInvalidValue);
    for (int stage = 0; stage < 4; ++stage) {
        ifn_cuda_stub = {};
        int *errors[] = {&ifn_cuda_stub.get_error, &ifn_cuda_stub.limit_error,
                         &ifn_cuda_stub.function_error, &ifn_cuda_stub.set_error};
        *errors[stage] = 700 + stage;
        IFN_POLICY_CHECK(ifn_cuda_prepare_shared(ifn_policy_kernel, 4096, 256) == 700 + stage);
        IFN_POLICY_CHECK(ifn_cuda_stub.sets == (stage == 3 ? 1 : 0));
    }
#if !IFN_CUDA_USE_PDL
    ifn_cuda_stub = {};
    IFN_POLICY_CHECK(ifn_cuda_prepare_prefill_resources() == 0 && ifn_cuda_stub.sets == 8);
    const int sizes[] = {92160, 98304, 82944, 89088, 93184, 99328, 88576, 94720};
    const std::uintptr_t kernels[] = {
        reinterpret_cast<std::uintptr_t>(
            ifn_cuda::ifn_cuda_p_wgemm_kernel<0, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_UP, 2304,
                                              32>),
        reinterpret_cast<std::uintptr_t>(
            ifn_cuda::ifn_cuda_p_wgemm_kernel<0, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_UP, 2304,
                                              128>),
        reinterpret_cast<std::uintptr_t>(
            ifn_cuda::ifn_cuda_p_wgemm_kernel<0, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_DOWN, 1152,
                                              32>),
        reinterpret_cast<std::uintptr_t>(
            ifn_cuda::ifn_cuda_p_wgemm_kernel<0, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_DOWN, 1152,
                                              128>),
        reinterpret_cast<std::uintptr_t>(
            ifn_cuda::ifn_cuda_p_wgemm_kernel<1, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_UP, 2304,
                                              32>),
        reinterpret_cast<std::uintptr_t>(
            ifn_cuda::ifn_cuda_p_wgemm_kernel<1, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_UP, 2304,
                                              128>),
        reinterpret_cast<std::uintptr_t>(
            ifn_cuda::ifn_cuda_p_wgemm_kernel<1, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_DOWN, 1152,
                                              32>),
        reinterpret_cast<std::uintptr_t>(
            ifn_cuda::ifn_cuda_p_wgemm_kernel<1, ifn_cuda::ifn_cuda_wgemm_mode::IFN_CUDA_DOWN, 1152,
                                              128>),
    };
    for (int i = 0; i < 8; ++i)
        IFN_POLICY_CHECK(ifn_cuda_stub.prepared_bytes[i] == sizes[i] &&
                         ifn_cuda_stub.prepared_kernels[i] == kernels[i]);
    for (int failure = 1; failure <= 8; ++failure) {
        ifn_cuda_stub = {};
        ifn_cuda_stub.fail_set_at = failure;
        IFN_POLICY_CHECK(ifn_cuda_prepare_prefill_resources() == 703 &&
                         ifn_cuda_stub.sets == failure);
    }
#endif
    std::puts("target admission, launch arguments/order policy, first errors and resource/context "
              "failures passed (CPU stubs)");
    return 0;
}
