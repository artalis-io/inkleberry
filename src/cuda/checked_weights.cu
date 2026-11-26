/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Additive checked trampolines around unchanged upstream helper kernels.
 * Original bridge is compiled in this TU, not also linked as a second object.
 * Same launch geometry/math; dedicated stream and recoverable error returns. */
#include "../../cuda/cuda_abi.cu"
#include "../checked_cuda.h"
#include <cuda.h>
#define IFN_CHECKED_WEIGHTS_LOCAL_TRY(call)                                                        \
    do {                                                                                           \
        cudaError_t e = (call);                                                                    \
        if (e != cudaSuccess)                                                                      \
            return (int)e;                                                                         \
    } while (0)

extern "C" int ifn_cuda_enter(int device, ifn_device_scope *old) {
    CUcontext context = nullptr;
    CUresult before = cuCtxGetCurrent(&context);
    if (before != CUDA_SUCCESS && before != CUDA_ERROR_NOT_INITIALIZED)
        return (int)before;
    old->context = context;
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(cudaGetDevice(&old->device));
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(cudaSetDevice(device));
    int major = 0, minor = 0;
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    return major == 12 && minor == 0 ? 0 : (int)cudaErrorNotSupported;
}

extern "C" int ifn_cuda_leave(const ifn_device_scope *old) {
    cudaError_t runtime = cudaSetDevice(old->device);
    CUresult driver = cuCtxSetCurrent(static_cast<CUcontext>(old->context));
    return runtime != cudaSuccess ? (int)runtime : (int)driver;
}

extern "C" int ifn_cuda_available(size_t *bytes) {
    size_t total;
    return (int)cudaMemGetInfo(bytes, &total);
}

extern "C" int ifn_cuda_pack(const void *src, void *dst, void *scales, void *amax, int rows,
                             int cols, int kind, float *scale, void *raw_stream) {
    cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    if (kind == 1) {
        ifn_cuda_z_quant_bf16_to_fp8_kernel<<<rows, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(src), static_cast<uint8_t *>(dst),
            static_cast<float *>(scales), rows, cols);
        return (int)cudaGetLastError();
    }
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(cudaMemsetAsync(amax, 0, 4, stream));
    size_t n = (size_t)rows * cols;
    int grid = (int)std::min<size_t>((n + 255) / 256, 2048);
    ifn_cuda_z_bf16_amax_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16 *>(src),
                                                          n, static_cast<float *>(amax));
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(cudaGetLastError());
    float h_amax = 0;
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(
        cudaMemcpyAsync(&h_amax, amax, 4, cudaMemcpyDeviceToHost, stream));
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(cudaStreamSynchronize(stream));
    *scale = std::max(h_amax / (448.0f * 6.0f), 1e-12f);
    size_t blocks = (size_t)rows * (cols / 16);
    ifn_cuda_z_quant_bf16_to_nvfp4_kernel<<<(int)((blocks + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(src), static_cast<uint8_t *>(dst),
        static_cast<uint8_t *>(scales), rows, cols, *scale);
    return (int)cudaGetLastError();
}

extern "C" int ifn_cuda_expert(const void *a, const void *b, void *out, void *scales, void *outer,
                               int e, int rows, int cols, void *raw_stream) {
    cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    ifn_cuda_z_mxfp8_expert_amax_kernel<<<e, 256, 0, stream>>>(
        static_cast<const uint8_t *>(a), static_cast<const uint8_t *>(b),
        static_cast<float *>(outer), rows, cols);
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(cudaGetLastError());
    size_t blocks = (size_t)e * rows * (cols / 16);
    ifn_cuda_z_mxfp8_to_nvfp4_expert_kernel<<<(int)((blocks + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint8_t *>(a), static_cast<const uint8_t *>(b),
        static_cast<float *>(outer), static_cast<uint8_t *>(out), static_cast<uint8_t *>(scales), e,
        rows, cols);
    return (int)cudaGetLastError();
}

extern "C" int ifn_cuda_rnn(const void *src, void *a, void *b, void *c, void *raw_stream) {
    ifn_cuda_z_split_dspark_rnn_kernel<<<768, 256, 0, static_cast<cudaStream_t>(raw_stream)>>>(
        static_cast<const __nv_bfloat16 *>(src), static_cast<__nv_bfloat16 *>(a),
        static_cast<__nv_bfloat16 *>(b), static_cast<__nv_bfloat16 *>(c));
    return (int)cudaGetLastError();
}

extern "C" int ifn_cuda_fold(const void *u, const void *h, const void *b, void *fb, void *fw,
                             int vocab, void *raw_stream) {
    if (vocab < 1 || vocab > 200058)
        return (int)cudaErrorInvalidValue;
    cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    dim3 block(32, 8);
    ifn_cuda_z_compute_dspark_fold_bias_kernel<<<(vocab + 7) / 8, block, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(u), static_cast<const __nv_bfloat16 *>(b),
        static_cast<float *>(fb), vocab, 2304);
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(cudaGetLastError());
    ifn_cuda_z_compute_dspark_fold_w_kernel<<<vocab, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(u), static_cast<const __nv_bfloat16 *>(h),
        static_cast<__nv_bfloat16 *>(fw), vocab, 2304, 256);
    return (int)cudaGetLastError();
}

extern "C" int ifn_cuda_launch_status(void) {
    return (int)cudaGetLastError();
}

/* Caller-allocated graph state: retain every acquired handle on failure.
 * Reuse the original selector kernel; no C++ heap allocation or deletion. */
#include <new>

struct ifn_adaptive_owner {
    cudaGraph_t root = nullptr, extra = nullptr, active_graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphConditionalHandle condition{};
    cudaGraph_t bodies[3] = {nullptr, nullptr, nullptr};
    cudaStream_t capture_stream = nullptr;
    bool capturing = false;
};

extern "C" size_t ifn_adaptive_bytes(void) {
    return sizeof(ifn_adaptive_owner);
}

extern "C" void ifn_adaptive_init(void *p) {
    new (p) ifn_adaptive_owner();
}

static int ifn_end_adaptive_capture(ifn_adaptive_owner *a, int original) {
    cudaGraph_t ended = nullptr;
    cudaError_t status = cudaStreamEndCapture(a->capture_stream, &ended);
    if (status == cudaSuccess)
        a->capturing = false;
    else {
        cudaStreamCaptureStatus capture;
        if (cudaStreamIsCapturing(a->capture_stream, &capture) == cudaSuccess &&
            capture == cudaStreamCaptureStatusNone)
            a->capturing = false;
    }
    if (ended && ended != a->root && ended != a->active_graph)
        a->extra = ended;
    return original ? original : (int)status;
}

extern "C" int ifn_adaptive_prepare(void *state, const int32_t *anchor, const int32_t *drafts,
                                    const float *confidence, float threshold, int32_t *tokens,
                                    int32_t *width, const int32_t *accepted, int32_t *host_accepted,
                                    const int32_t *count, int32_t *host_count, int32_t *host_width,
                                    void *raw_stream) {
    auto *a = static_cast<ifn_adaptive_owner *>(state);
    if (a->root || a->exec || a->extra || a->capturing)
        return (int)cudaErrorInvalidResourceHandle;
    cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(cudaGraphCreate(&a->root, 0));
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(cudaGraphConditionalHandleCreate(&a->condition, a->root, 0, 0));
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(cudaStreamBeginCaptureToGraph(
        stream, a->root, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    a->capturing = true;
    a->capture_stream = stream;
    a->active_graph = a->root;
#define IFN_CHECKED_WEIGHTS_LOCAL_CAPTURE_TRY(call)                                                \
    do {                                                                                           \
        cudaError_t e = (call);                                                                    \
        if (e != cudaSuccess)                                                                      \
            return ifn_end_adaptive_capture(a, (int)e);                                            \
    } while (0)
    ifn_cuda_z_adaptive_select_width_kernel<<<1, 1, 0, stream>>>(
        anchor, drafts, confidence, threshold, tokens, width, a->condition);
    IFN_CHECKED_WEIGHTS_LOCAL_CAPTURE_TRY(cudaGetLastError());
    cudaStreamCaptureStatus status;
    cudaGraph_t graph = nullptr;
    const cudaGraphNode_t *deps = nullptr;
    const cudaGraphEdgeData *edges = nullptr;
    size_t n = 0;
    IFN_CHECKED_WEIGHTS_LOCAL_CAPTURE_TRY(
        cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, &deps, &edges, &n));
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = a->condition;
    params.conditional.type = cudaGraphCondTypeSwitch;
    params.conditional.size = 3;
    cudaGraphNode_t node = nullptr;
    IFN_CHECKED_WEIGHTS_LOCAL_CAPTURE_TRY(cudaGraphAddNode(&node, graph, deps, edges, n, &params));
    for (int i = 0; i < 3; i++)
        a->bodies[i] = params.conditional.phGraph_out[i];
    IFN_CHECKED_WEIGHTS_LOCAL_CAPTURE_TRY(cudaStreamUpdateCaptureDependencies(
        stream, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
    IFN_CHECKED_WEIGHTS_LOCAL_CAPTURE_TRY(cudaMemcpyAsync(
        host_accepted, accepted, 8 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    IFN_CHECKED_WEIGHTS_LOCAL_CAPTURE_TRY(
        cudaMemcpyAsync(host_count, count, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    IFN_CHECKED_WEIGHTS_LOCAL_CAPTURE_TRY(
        cudaMemcpyAsync(host_width, width, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
#undef IFN_CHECKED_WEIGHTS_LOCAL_CAPTURE_TRY
    return ifn_end_adaptive_capture(a, 0);
}

extern "C" int ifn_adaptive_branch_begin(void *state, int branch, void *raw_stream) {
    auto *a = static_cast<ifn_adaptive_owner *>(state);
    if (branch < 0 || branch > 2 || a->capturing || !a->root || !a->bodies[branch])
        return (int)cudaErrorInvalidValue;
    a->capture_stream = static_cast<cudaStream_t>(raw_stream);
    a->active_graph = a->bodies[branch];
    IFN_CHECKED_WEIGHTS_LOCAL_TRY(cudaStreamBeginCaptureToGraph(
        a->capture_stream, a->active_graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    a->capturing = true;
    return 0;
}

extern "C" int ifn_adaptive_branch_end(void *state) {
    auto *a = static_cast<ifn_adaptive_owner *>(state);
    return a->capturing ? ifn_end_adaptive_capture(a, 0) : (int)cudaErrorInvalidResourceHandle;
}

extern "C" int ifn_adaptive_instantiate(void *state) {
    auto *a = static_cast<ifn_adaptive_owner *>(state);
    if (a->capturing || a->exec || !a->root)
        return (int)cudaErrorInvalidResourceHandle;
    return (int)cudaGraphInstantiate(&a->exec, a->root, 0);
}

extern "C" int ifn_adaptive_launch(void *state, void *stream) {
    auto *a = static_cast<ifn_adaptive_owner *>(state);
    if (!a->exec)
        return (int)cudaErrorInvalidResourceHandle;
    return (int)cudaGraphLaunch(a->exec, static_cast<cudaStream_t>(stream));
}

extern "C" int ifn_adaptive_cleanup(void *state) {
    auto *a = static_cast<ifn_adaptive_owner *>(state);
    int first = 0;
    if (a->capturing) {
        first = ifn_end_adaptive_capture(a, 0);
        if (a->capturing)
            return first ? first : (int)cudaErrorStreamCaptureInvalidated;
    }
#define IFN_CHECKED_WEIGHTS_LOCAL_RELEASE_HANDLE(field, call)                                      \
    do {                                                                                           \
        if (a->field) {                                                                            \
            cudaError_t e = (call);                                                                \
            if (e == cudaSuccess)                                                                  \
                a->field = nullptr;                                                                \
            else if (!first)                                                                       \
                first = (int)e;                                                                    \
        }                                                                                          \
    } while (0)
    IFN_CHECKED_WEIGHTS_LOCAL_RELEASE_HANDLE(exec, cudaGraphExecDestroy(a->exec));
    if (!a->exec) {
        IFN_CHECKED_WEIGHTS_LOCAL_RELEASE_HANDLE(extra, cudaGraphDestroy(a->extra));
        IFN_CHECKED_WEIGHTS_LOCAL_RELEASE_HANDLE(root, cudaGraphDestroy(a->root));
    }
#undef IFN_CHECKED_WEIGHTS_LOCAL_RELEASE_HANDLE
    return first;
}

extern "C" int ifn_adaptive_live(const void *state) {
    const auto *a = static_cast<const ifn_adaptive_owner *>(state);
    return a->root || a->exec || a->extra || a->capturing;
}
