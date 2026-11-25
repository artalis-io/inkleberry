/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Additive checked trampolines around unchanged upstream helper kernels.
 * Original bridge is compiled in this TU, not also linked as a second object.
 * Same launch geometry/math; dedicated stream and recoverable error returns. */
#include "../../cuda/cuda_abi.cu"
#include "../c/checked_cuda.h"
#include <cuda.h>
#define TRY(call)                                                                                  \
    do {                                                                                           \
        cudaError_t e = (call);                                                                    \
        if (e != cudaSuccess)                                                                      \
            return (int)e;                                                                         \
    } while (0)

extern "C" int sq_cuda_enter(int device, sq_device_scope *old) {
    CUcontext context = nullptr;
    CUresult before = cuCtxGetCurrent(&context);
    if (before != CUDA_SUCCESS && before != CUDA_ERROR_NOT_INITIALIZED)
        return (int)before;
    old->context = context;
    TRY(cudaGetDevice(&old->device));
    TRY(cudaSetDevice(device));
    int major = 0, minor = 0;
    TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    return major == 12 && minor == 0 ? 0 : (int)cudaErrorNotSupported;
}

extern "C" int sq_cuda_leave(const sq_device_scope *old) {
    cudaError_t runtime = cudaSetDevice(old->device);
    CUresult driver = cuCtxSetCurrent(static_cast<CUcontext>(old->context));
    return runtime != cudaSuccess ? (int)runtime : (int)driver;
}

extern "C" int sq_cuda_available(size_t *bytes) {
    size_t total;
    return (int)cudaMemGetInfo(bytes, &total);
}

extern "C" int sq_cuda_pack(const void *src, void *dst, void *scales, void *amax, int rows,
                            int cols, int kind, float *scale, void *raw_stream) {
    cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    if (kind == 1) {
        z_quant_bf16_to_fp8_kernel<<<rows, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(src), static_cast<uint8_t *>(dst),
            static_cast<float *>(scales), rows, cols);
        return (int)cudaGetLastError();
    }
    TRY(cudaMemsetAsync(amax, 0, 4, stream));
    size_t n = (size_t)rows * cols;
    int grid = (int)std::min<size_t>((n + 255) / 256, 2048);
    z_bf16_amax_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16 *>(src), n,
                                                 static_cast<float *>(amax));
    TRY(cudaGetLastError());
    float h_amax = 0;
    TRY(cudaMemcpyAsync(&h_amax, amax, 4, cudaMemcpyDeviceToHost, stream));
    TRY(cudaStreamSynchronize(stream));
    *scale = std::max(h_amax / (448.0f * 6.0f), 1e-12f);
    size_t blocks = (size_t)rows * (cols / 16);
    z_quant_bf16_to_nvfp4_kernel<<<(int)((blocks + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(src), static_cast<uint8_t *>(dst),
        static_cast<uint8_t *>(scales), rows, cols, *scale);
    return (int)cudaGetLastError();
}

extern "C" int sq_cuda_expert(const void *a, const void *b, void *out, void *scales, void *outer,
                              int e, int rows, int cols, void *raw_stream) {
    cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    z_mxfp8_expert_amax_kernel<<<e, 256, 0, stream>>>(static_cast<const uint8_t *>(a),
                                                      static_cast<const uint8_t *>(b),
                                                      static_cast<float *>(outer), rows, cols);
    TRY(cudaGetLastError());
    size_t blocks = (size_t)e * rows * (cols / 16);
    z_mxfp8_to_nvfp4_expert_kernel<<<(int)((blocks + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint8_t *>(a), static_cast<const uint8_t *>(b),
        static_cast<float *>(outer), static_cast<uint8_t *>(out), static_cast<uint8_t *>(scales), e,
        rows, cols);
    return (int)cudaGetLastError();
}

extern "C" int sq_cuda_rnn(const void *src, void *a, void *b, void *c, void *raw_stream) {
    z_split_dspark_rnn_kernel<<<768, 256, 0, static_cast<cudaStream_t>(raw_stream)>>>(
        static_cast<const __nv_bfloat16 *>(src), static_cast<__nv_bfloat16 *>(a),
        static_cast<__nv_bfloat16 *>(b), static_cast<__nv_bfloat16 *>(c));
    return (int)cudaGetLastError();
}

extern "C" int sq_cuda_fold(const void *u, const void *h, const void *b, void *fb, void *fw,
                            int vocab, void *raw_stream) {
    if (vocab < 1 || vocab > 200058)
        return (int)cudaErrorInvalidValue;
    cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    dim3 block(32, 8);
    z_compute_dspark_fold_bias_kernel<<<(vocab + 7) / 8, block, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(u), static_cast<const __nv_bfloat16 *>(b),
        static_cast<float *>(fb), vocab, 2304);
    TRY(cudaGetLastError());
    z_compute_dspark_fold_w_kernel<<<vocab, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(u), static_cast<const __nv_bfloat16 *>(h),
        static_cast<__nv_bfloat16 *>(fw), vocab, 2304, 256);
    return (int)cudaGetLastError();
}
