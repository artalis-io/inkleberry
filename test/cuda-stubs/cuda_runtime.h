/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * CPU-only simulation of the CUDA host API. Never include in production. */
#pragma once
#include <cstddef>
#include <cstdint>
using cudaError_t = int;
using cudaStream_t = void *;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidValue = 1;
constexpr cudaError_t cudaErrorInvalidConfiguration = 9;
constexpr int cudaLaunchAttributeProgrammaticStreamSerialization = 6;
constexpr int cudaDevAttrMaxSharedMemoryPerBlockOptin = 97;
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize = 8;

struct ifn_cuda_stub_dim3 {
    unsigned x, y, z;

    ifn_cuda_stub_dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {
    }
};

using dim3 = ifn_cuda_stub_dim3;

struct ifn_cuda_stub_launch_attribute {
    int id;

    struct {
        int programmaticStreamSerializationAllowed;
    } val;
};

using cudaLaunchAttribute = ifn_cuda_stub_launch_attribute;

struct ifn_cuda_stub_launch_config {
    dim3 gridDim, blockDim;
    size_t dynamicSmemBytes;
    cudaStream_t stream;
    cudaLaunchAttribute *attrs;
    unsigned numAttrs;
};

using cudaLaunchConfig_t = ifn_cuda_stub_launch_config;

struct ifn_cuda_stub_func_attributes {
    size_t sharedSizeBytes = 0;
    int maxThreadsPerBlock = 1024;
};

using cudaFuncAttributes = ifn_cuda_stub_func_attributes;

struct ifn_cuda_stub_state {
    int device = 0, limits[2] = {101376, 49152};
    int get_error = 0, limit_error = 0, function_error = 0, set_error = 0, launch_error = 0;
    int queries = 0, sets = 0, launches = 0, set_bytes = 0, set_device = -1;
    int fail_set_at = 0, prepared_bytes[16] = {};
    std::uintptr_t prepared_kernels[16] = {};
    bool programmatic = false;
    cudaLaunchConfig_t launch = {};
    cudaFuncAttributes function = {};
};

inline ifn_cuda_stub_state ifn_cuda_stub;

inline cudaError_t ifn_cuda_stub_get_device(int *device) {
    ++ifn_cuda_stub.queries;
    *device = ifn_cuda_stub.device;
    return ifn_cuda_stub.get_error;
}

inline cudaError_t ifn_cuda_stub_device_attribute(int *value, int attribute, int device) {
    if (attribute != cudaDevAttrMaxSharedMemoryPerBlockOptin || device < 0 || device > 1)
        return cudaErrorInvalidValue;
    *value = ifn_cuda_stub.limits[device];
    return ifn_cuda_stub.limit_error;
}

inline constexpr auto cudaGetDevice = ifn_cuda_stub_get_device;
inline constexpr auto cudaDeviceGetAttribute = ifn_cuda_stub_device_attribute;

struct ifn_cuda_stub_get_attributes {
    template <class Kernel> cudaError_t operator()(cudaFuncAttributes *out, Kernel) const {
        *out = ifn_cuda_stub.function;
        return ifn_cuda_stub.function_error;
    }
};

struct ifn_cuda_stub_set_attribute {
    template <class Kernel> cudaError_t operator()(Kernel kernel, int attribute, int bytes) const {
        if (attribute != cudaFuncAttributeMaxDynamicSharedMemorySize)
            return cudaErrorInvalidValue;
        if (ifn_cuda_stub.sets < 16) {
            ifn_cuda_stub.prepared_bytes[ifn_cuda_stub.sets] = bytes;
            ifn_cuda_stub.prepared_kernels[ifn_cuda_stub.sets] =
                reinterpret_cast<std::uintptr_t>(kernel);
        }
        ++ifn_cuda_stub.sets;
        ifn_cuda_stub.set_bytes = bytes;
        ifn_cuda_stub.set_device = ifn_cuda_stub.device;
        return ifn_cuda_stub.fail_set_at && ifn_cuda_stub.sets == ifn_cuda_stub.fail_set_at
                   ? 703
                   : ifn_cuda_stub.set_error;
    }
};

struct ifn_cuda_stub_launch_kernel {
    template <class Kernel, class... Args>
    cudaError_t operator()(const cudaLaunchConfig_t *config, Kernel kernel, Args... args) const {
        ++ifn_cuda_stub.launches;
        ifn_cuda_stub.launch = *config;
        ifn_cuda_stub.programmatic =
            config->numAttrs == 1 && config->attrs &&
            config->attrs[0].id == cudaLaunchAttributeProgrammaticStreamSerialization &&
            config->attrs[0].val.programmaticStreamSerializationAllowed == 1;
        if (ifn_cuda_stub.launch_error)
            return ifn_cuda_stub.launch_error;
        kernel(args...); // A CPU fixture, never a GPU launch.
        return cudaSuccess;
    }
};

inline constexpr ifn_cuda_stub_get_attributes cudaFuncGetAttributes{};
inline constexpr ifn_cuda_stub_set_attribute cudaFuncSetAttribute{};
inline constexpr ifn_cuda_stub_launch_kernel cudaLaunchKernelEx{};
