/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_CHECKED_PREFILL_H
#define SQ_CHECKED_PREFILL_H
#include "cuda_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
int sq_blas_create(void **out, void *stream);
int sq_blas_destroy(void *handle);
int sq_prefill_gemm_h(void *handle, const CLinearWeight *w, const void *x_h, float *y,
                      void *w_f16_scratch, int32_t batch_size, void *stream);
int sq_prefill_controller_one(void *handle, const float *x, const float *saved,
                              const float *prior_last, const float *cp_q,
                              const CControllerWeights *cw, const void *post_norm, float *out,
                              void *out_h, float *scratch_msg, void *scratch_cat_h,
                              float *scratch_d, void *scratch_d_h, float *scratch_f,
                              float *scratch_g, void *w_f16_scratch, int32_t batch_size,
                              void *stream);
int sq_prefill_global_sconv_prep_store(void *handle, float *qrgs, float *k_proj, float *v_proj,
                                       const void *in_k_h, const void *in_v_h, const void *k_sw,
                                       const void *v_sw, const void *k_sp, const void *v_sp,
                                       float *k_ring, float *v_ring, const void *q_cs,
                                       const void *r_cs, void *k_cache, void *v_cache,
                                       void *k_scale, void *v_scale, uint8_t kv_fmt, int32_t kv_cap,
                                       float *scratch_taps, void *w_f16_scratch, int32_t batch_size,
                                       const int32_t *d_pos, void *stream);
#ifdef __cplusplus
}
#endif
#endif
