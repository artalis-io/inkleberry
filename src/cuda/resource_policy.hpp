/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_CUDA_RESOURCE_POLICY_HPP
#define IFN_CUDA_RESOURCE_POLICY_HPP

/* Call only during preparation, in the session's selected context, before
 * capture. No global cache: another device/context must be checked afresh. */
template <class Kernel>
inline cudaError_t ifn_cuda_prepare_shared(Kernel kernel, int bytes, int threads) {
    if (!kernel || bytes < 0 || threads <= 0)
        return cudaErrorInvalidValue;
    int device = 0, limit = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        return error;
    error = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (error != cudaSuccess)
        return error;
    cudaFuncAttributes attributes = {};
    error = cudaFuncGetAttributes(&attributes, kernel);
    if (error != cudaSuccess)
        return error;
    if (limit < 0 || attributes.sharedSizeBytes > static_cast<size_t>(limit) ||
        static_cast<size_t>(bytes) > static_cast<size_t>(limit) - attributes.sharedSizeBytes ||
        threads > attributes.maxThreadsPerBlock)
        return cudaErrorInvalidConfiguration;
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}
#endif
