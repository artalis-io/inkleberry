/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Original kernels, additive error-returning dispatch with session-owned cuBLAS.
 * Compile this instead of a second kernels_prefill object. Original entrypoints
 * remain available to the Zig oracle but are never used by the C session. */
#include "../../cuda/kernels_prefill.cu"
#include "../checked_prefill.h"
using namespace squid;

static int blas_status(cublasStatus_t s) {
    return s == CUBLAS_STATUS_SUCCESS ? 0 : 10000 + (int)s;
}

#define CUDA_TRY(call)                                                                             \
    do {                                                                                           \
        cudaError_t e = (call);                                                                    \
        if (e != cudaSuccess)                                                                      \
            return (int)e;                                                                         \
    } while (0)
#define BLAS_TRY(call)                                                                             \
    do {                                                                                           \
        int e = blas_status(call);                                                                 \
        if (e)                                                                                     \
            return e;                                                                              \
    } while (0)
#define CALL_TRY(call)                                                                             \
    do {                                                                                           \
        int e = (call);                                                                            \
        if (e)                                                                                     \
            return e;                                                                              \
    } while (0)

extern "C" int ifn_blas_create(void **out, void *stream) {
    cublasHandle_t handle = nullptr;
    cublasStatus_t status = cublasCreate(&handle);
    *out = handle;
    if (status != CUBLAS_STATUS_SUCCESS)
        return blas_status(status);
    BLAS_TRY(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    BLAS_TRY(cublasSetStream(handle, static_cast<cudaStream_t>(stream)));
    return 0;
}

extern "C" int ifn_blas_destroy(void *handle) {
    return blas_status(cublasDestroy(static_cast<cublasHandle_t>(handle)));
}

extern "C" int ifn_prefill_gemm_h(void *handle, const CLinearWeight *w, const void *x, float *y,
                                  void *scratch, int32_t batch, void *raw_stream) {
    cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    __half *dst = static_cast<__half *>(scratch);
    if (w->kind == 2) {
        size_t blocks = (size_t)w->rows * (w->cols >> 4);
        p_dequant_nvfp4_to_f16_kernel<<<(int)((blocks + 255) / 256), 256, 0, stream>>>(
            static_cast<const uint8_t *>(w->data), w->scales, dst, blocks);
    } else if (w->kind == 1) {
        size_t pairs = (size_t)w->rows * (w->cols >> 1);
        p_dequant_fp8_to_f16_kernel<<<(int)((pairs + 255) / 256), 256, 0, stream>>>(
            static_cast<const uint8_t *>(w->data), w->row_scales, dst, w->rows, w->cols);
    } else {
        size_t n = (size_t)w->rows * w->cols;
        p_dequant_bf16_to_f16_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(w->data), dst, n);
    }
    CUDA_TRY(cudaGetLastError());
    float alpha = w->kind == 2 ? w->global_scale : 1.0f, beta = 0.0f;
    BLAS_TRY(cublasGemmEx(static_cast<cublasHandle_t>(handle), CUBLAS_OP_T, CUBLAS_OP_N, w->rows,
                          batch, w->cols, &alpha, dst, CUDA_R_16F, w->cols, x, CUDA_R_16F, w->cols,
                          &beta, y, CUDA_R_32F, w->rows, CUBLAS_COMPUTE_32F,
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return 0;
}

extern "C" int ifn_prefill_controller_one(void *handle, const float *x, const float *saved,
                                          const float *prior, const float *query,
                                          const CControllerWeights *cw, const void *norm,
                                          float *out, void *out_h, float *msg, void *cat, float *d,
                                          void *dh, float *f, float *g, void *scratch,
                                          int32_t batch, void *raw_stream) {
    cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    p_ctrl_prep_h_kernel<<<batch, 256, 0, stream>>>(x, saved, prior, query, msg,
                                                    static_cast<__half *>(cat));
    CUDA_TRY(cudaGetLastError());
    CLinearWeight down = cw->down_nvfp4, forget = cw->forget_nvfp4, gain = cw->gain_nvfp4;
    if (!down.data) {
        down = {0, 64, 4608, cw->down_w, nullptr, nullptr, 1};
        forget = {0, 2304, 64, cw->forget_w, nullptr, nullptr, 1};
        gain = {0, 2304, 64, cw->gain_w, nullptr, nullptr, 1};
    }
    CALL_TRY(ifn_prefill_gemm_h(handle, &down, cat, d, scratch, batch, raw_stream));
    launch_prefill_f32_to_f16(d, static_cast<__half *>(dh), batch * 64, stream);
    CUDA_TRY(cudaGetLastError());
    CALL_TRY(ifn_prefill_gemm_h(handle, &forget, dh, f, scratch, batch, raw_stream));
    CALL_TRY(ifn_prefill_gemm_h(handle, &gain, dh, g, scratch, batch, raw_stream));
    p_ctrl_finish_kernel<<<batch, 256, 0, stream>>>(
        x, msg, f, g, static_cast<const __nv_bfloat16 *>(norm), out, static_cast<__half *>(out_h));
    return (int)cudaGetLastError();
}

extern "C" int ifn_prefill_global_sconv_prep_store(
    void *handle, float *qrgs, float *kp, float *vp, const void *ik, const void *iv,
    const void *ksw, const void *vsw, const void *ksp, const void *vsp, float *kr, float *vr,
    const void *qcs, const void *rcs, void *kc, void *vc, void *k_scale, void *v_scale, uint8_t fmt,
    int32_t cap, float *taps, void *scratch, int32_t batch, const int32_t *pos, void *raw_stream) {
    cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    CLinearWeight kw = {0, 32, 2304, const_cast<void *>(ksp), nullptr, nullptr, 1}, vw = kw;
    vw.data = const_cast<void *>(vsp);
    CALL_TRY(ifn_prefill_gemm_h(handle, &kw, ik, taps, scratch, batch, raw_stream));
    CALL_TRY(
        ifn_prefill_gemm_h(handle, &vw, iv, taps + (size_t)batch * 32, scratch, batch, raw_stream));
    p_dynamic_sconv_kv_inplace_kernel<<<dim3(KV_DIM / 256, 2), 256, 0, stream>>>(
        kp, vp, static_cast<const __nv_bfloat16 *>(ksw), static_cast<const __nv_bfloat16 *>(vsw),
        taps, taps + (size_t)batch * 32, kr, vr, batch, pos);
    CUDA_TRY(cudaGetLastError());
#define STORE(F)                                                                                   \
    p_global_attn_prep_store_kernel<F><<<dim3(64, batch), 128, 0, stream>>>(                       \
        qrgs, kp, vp, static_cast<const __nv_bfloat16 *>(qcs),                                     \
        static_cast<const __nv_bfloat16 *>(rcs), kc, vc, static_cast<__half *>(k_scale),           \
        static_cast<__half *>(v_scale), cap, pos)
    if (fmt == 1) {
        STORE(1);
    } else if (fmt == 2) {
        STORE(2);
    } else {
        STORE(0);
    }
#undef STORE
    return (int)cudaGetLastError();
}
