/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Pinned Engine.prefill_chunk_batched with optional DSpark tap commits. */
#include "engine_internal.h"
#include "checked_prefill.h"
#ifdef IFN_TEST_NUMERICS
void ifn_test_numeric_trace(ifn_session *, int32_t, unsigned, const char *, const float *, size_t);
#define IFN_PREFILL_NUMERIC_TAP(visit, stage, data, width)                                         \
    ifn_test_numeric_trace(s, n, visit, stage, data, width)
#else
#define IFN_PREFILL_NUMERIC_TAP(visit, stage, data, width) ((void)0)
#endif
#define IFN_ENGINE_PREFILL_LOCAL_CHECK(call)                                                       \
    do {                                                                                           \
        ifn_ar_check(s, (call), "batched prefill");                                                \
        if (s->status != IFN_OK)                                                                   \
            return;                                                                                \
    } while (0)

static void ifn_engine_prefill_local_gemm(ifn_session *s, const ifn_cuda_c_linear_weight *w,
                                          const void *x, float *y, int32_t n) {
    if (s->status == IFN_OK)
        ifn_ar_check(s, ifn_prefill_gemm_h(s->blas, w, x, y, s->p_w_f16, n, s->stream),
                     "prefill GEMM");
}

static void ifn_engine_prefill_local_controller(ifn_session *s, const float *x, const float *q,
                                                const ifn_cuda_c_controller_weights *cw,
                                                const void *norm, float *out, void *out_h,
                                                int32_t n) {
    if (s->status == IFN_OK)
        ifn_ar_check(s,
                     ifn_prefill_controller_one(s->blas, x, s->p_saved, s->p_prior_last, q, cw,
                                                norm, out, out_h, s->p_ctrl_msg, s->p_ctrl_cat_h,
                                                s->p_ctrl_d, s->p_ctrl_d_h, s->p_ctrl_f,
                                                s->p_ctrl_g, s->p_w_f16, n, s->stream),
                     "prefill controller");
}

void ifn_ar_prefill(ifn_session *s, const int32_t *tokens, int32_t n) {
    if (s->status != IFN_OK)
        return;
    if (!s->blas)
        IFN_ENGINE_PREFILL_LOCAL_CHECK(ifn_blas_create(&s->blas, s->stream));
    int32_t pos[256];
    for (int32_t i = 0; i < n / 8; i++)
        pos[i] = s->position + i * 8;
    const ifn_weight_backend *backend = &s->weights->backend;
    IFN_ENGINE_PREFILL_LOCAL_CHECK(
        backend->copy(backend->context, s->p_pos_vec, pos, (size_t)(n / 8) * 4, 0, s->stream));
    IFN_ENGINE_PREFILL_LOCAL_CHECK(
        backend->copy(backend->context, s->p_tokens, tokens, (size_t)n * 4, 0, s->stream));
    void *stream = s->stream;
    const ifn_ar_model *w = &s->model;
    ifn_launch_embed_norm(w->embed, w->embed_norm, s->p_tokens, s->p_hidden, n, stream);
    IFN_PREFILL_NUMERIC_TAP(0, "embedding", s->p_hidden, 2304);
    int32_t blocks = 0;
    bool prior = false, pending = false;
    for (unsigned vi = 0; vi < 48; vi++) {
        ifn_visit v = s->schedule[vi];
        unsigned l = v.layer;
        const ifn_layer *lw = &w->layers[l];
        bool prelude = false;
        if (v.loop && l == 6) {
            if (v.pass == 0)
                prelude = true;
            else {
                ifn_launch_extract_block(s->p_blocks, 4, 2, s->p_saved, n, stream);
                ifn_launch_rmsnorm(s->p_hidden, NULL, s->p_prior_last, n, 2304, 2304, 1e-6f,
                                   stream);
                IFN_ENGINE_PREFILL_LOCAL_CHECK(backend->copy(
                    backend->context, s->p_hidden, s->p_prelude, (size_t)n * 2304 * 4, 2, stream));
                blocks = 1;
                prior = true;
            }
        }
        bool use_prior = v.loop && prior, distinct = v.global || use_prior, boundary = l % 6 == 0;
        if (pending) {
            const ifn_layer *prev = &w->layers[s->schedule[vi - 1].layer];
            ifn_launch_prefill_sconv_post_res(s->p_moe_slot_out, 1, prev->mlp_sconv_w,
                                              s->mlp_ring[vi - 1], s->p_residual, s->p_hidden,
                                              prelude ? s->p_prelude : NULL, NULL, NULL, 0, 0, n,
                                              s->p_pos_vec, NULL, NULL, NULL, stream);
        }
        if (s->use_mtp && (vi == 5 || vi == 23 || vi == 41))
            ifn_launch_prefill_copy_taps(s->p_hidden, s->p_taps,
                                         vi == 5    ? 0
                                         : vi == 23 ? 1
                                                    : 2,
                                         n, stream);
        pending = false;
        const void *norm[3] = {NULL, NULL, NULL};
        float *out[3] = {NULL, NULL, NULL};
        void *half[3] = {NULL, NULL, NULL};
        int32_t count = 1;
        if (!distinct) {
            norm[0] = lw->attn_norm;
            out[0] = s->p_attn_in[0];
            half[0] = s->p_attn_in_h[0];
        } else if (v.global && !use_prior) {
            norm[0] = lw->q_in_norm;
            norm[1] = lw->k_in_norm;
            norm[2] = lw->v_in_norm;
            for (unsigned j = 0; j < 3; j++) {
                out[j] = s->p_attn_in[j];
                half[j] = s->p_attn_in_h[j];
            }
            count = 3;
        } else if (v.global) {
            for (unsigned j = 0; j < 3; j++)
                out[j] = s->p_attn_in[j];
            count = 3;
        } else
            out[0] = s->p_mlp_in;
        ifn_launch_fused_depth_norm(s->p_hidden, boundary ? s->p_partial : NULL, s->p_blocks,
                                    blocks, l ? w->attn_queries[l - 1] : NULL, norm[0], norm[1],
                                    norm[2], out[0], out[1], out[2], half[0], half[1], half[2],
                                    count, n, stream);
        if (use_prior) {
            unsigned loc = l - 6;
            const float *q[3] = {w->cross_pass_queries + 2 * loc * 2304,
                                 w->cross_pass_kv_queries + 2 * loc * 2304,
                                 w->cross_pass_kv_queries + (2 * loc + 1) * 2304};
            const void *norms[3] = {v.global ? lw->q_in_norm : lw->attn_norm,
                                    v.global ? lw->k_in_norm : lw->attn_norm,
                                    v.global ? lw->v_in_norm : lw->attn_norm};
            for (unsigned j = 0; j < 3; j++)
                ifn_engine_prefill_local_controller(s, v.global ? s->p_attn_in[j] : s->p_mlp_in,
                                                    q[j], &lw->ctrl[j], norms[j], s->p_attn_in[j],
                                                    s->p_attn_in_h[j], n);
        }
        IFN_PREFILL_NUMERIC_TAP(vi, "attention_input", s->p_attn_in[0], 2304);
        void *ik = s->p_attn_in_h[distinct ? 1 : 0], *iv = s->p_attn_in_h[distinct ? 2 : 0];
        ifn_engine_prefill_local_gemm(s, &lw->w_qrgs, s->p_attn_in_h[0], s->p_qrgs, n);
        IFN_PREFILL_NUMERIC_TAP(vi, "qrgs", s->p_qrgs, 6552);
        ifn_engine_prefill_local_gemm(s, &lw->w_k, ik, s->p_k_proj, n);
        ifn_engine_prefill_local_gemm(s, &lw->w_v, iv, s->p_v_proj, n);
        if (s->status != IFN_OK)
            return;
        IFN_PREFILL_NUMERIC_TAP(vi, "key_projection", s->p_k_proj, 1024);
        if (v.global) {
            IFN_ENGINE_PREFILL_LOCAL_CHECK(ifn_prefill_global_sconv_prep_store(
                s->blas, s->p_qrgs, s->p_k_proj, s->p_v_proj, ik, iv, lw->k_sconv_w, lw->v_sconv_w,
                lw->k_sconv_proj, lw->v_sconv_proj, s->k_ring[vi], s->v_ring[vi], lw->q_clip_scale,
                lw->rel_clip_scale, s->k_cache[vi], s->v_cache[vi], s->k_scale[vi], s->v_scale[vi],
                (uint8_t)s->config.kv_format, s->kv_cap[vi], s->p_ctrl_d, s->p_w_f16, n,
                s->p_pos_vec, stream));
            ifn_launch_prefill_flash_attn_tc(s->p_qrgs, lw->rel_logits_proj, s->k_cache[vi],
                                             s->v_cache[vi], s->k_scale[vi], s->v_scale[vi],
                                             (uint8_t)s->config.kv_format, s->p_attn_gated_h,
                                             s->kv_cap[vi], 0, n, s->p_pos_vec, stream);
        } else
            for (int32_t off = 0; off < n;) {
                int32_t sub = n - off > 256 ? 256 : n - off;
                float *qrgs = s->p_qrgs + (size_t)off * 6552, *k = s->p_k_proj + (size_t)off * 1024,
                      *val = s->p_v_proj + (size_t)off * 1024;
                const int32_t *p = s->p_pos_vec + off / 8;
                ifn_launch_prefill_local_sconv_prep_store(
                    qrgs, k, val, lw->k_sconv_w, lw->v_sconv_w, s->k_ring[vi], s->v_ring[vi],
                    lw->q_clip_scale, lw->rel_clip_scale, s->k_cache[vi], s->v_cache[vi],
                    s->kv_cap[vi], sub, p, stream);
                ifn_launch_prefill_flash_attn_tc(
                    qrgs, lw->rel_logits_proj, s->k_cache[vi], s->v_cache[vi], NULL, NULL, 0,
                    (unsigned char *)s->p_attn_gated_h + (size_t)off * 3072 * 2, s->kv_cap[vi],
                    1024, sub, p, stream);
                off += sub;
            }
        ifn_engine_prefill_local_gemm(s, &lw->w_o, s->p_attn_gated_h, s->p_o_proj, n);
        if (s->status != IFN_OK)
            return;
        IFN_PREFILL_NUMERIC_TAP(vi, "attention_output_projection", s->p_o_proj, 2304);
        ifn_launch_prefill_sconv_post_res(s->p_o_proj, 0, lw->attn_sconv_w, s->attn_ring[vi],
                                          s->p_hidden, s->p_residual, NULL, s->p_blocks,
                                          s->p_partial, blocks, boundary ? 1 : 0, n, s->p_pos_vec,
                                          NULL, NULL, NULL, stream);
        if (boundary)
            blocks++;
        ifn_launch_fused_depth_norm(s->p_residual, NULL, s->p_blocks, blocks, w->mlp_queries[l],
                                    use_prior ? NULL : lw->mlp_norm, NULL, NULL,
                                    use_prior ? s->p_mlp_in : s->p_mlp_normed, NULL, NULL,
                                    use_prior ? NULL : s->p_mlp_normed_h, NULL, NULL, 1, n, stream);
        if (use_prior)
            ifn_engine_prefill_local_controller(
                s, s->p_mlp_in, w->cross_pass_queries + (2 * (l - 6) + 1) * 2304, &lw->ctrl[3],
                lw->mlp_norm, s->p_mlp_normed, s->p_mlp_normed_h, n);
        ifn_cuda_c_linear_weight router = lw->router_nvfp4;
        if (s->weights->config.mode != 2) {
            router = (ifn_cuda_c_linear_weight){0, 128, 2304, lw->router_w, NULL, NULL, 1};
        }
        ifn_engine_prefill_local_gemm(s, &router, s->p_mlp_normed_h, s->p_router_logits, n);
        if (s->status != IFN_OK)
            return;
        ifn_launch_prefill_route_and_sort(s->p_router_logits, lw->router_bias, s->p_route_ids,
                                          s->p_route_w, s->p_exp_offsets, s->p_sorted_tok,
                                          s->p_sorted_slot, s->p_sorted_w, n, stream);
        IFN_PREFILL_NUMERIC_TAP(vi, "router_logits", s->p_router_logits, 128);
        IFN_PREFILL_NUMERIC_TAP(vi, "router_weights", s->p_route_w, 8);
        ifn_launch_prefill_moe_gate_up_h(&lw->w13, s->p_mlp_normed_h, s->p_exp_offsets,
                                         s->p_sorted_tok, s->p_sorted_slot, s->p_moe_act_h, n,
                                         stream);
        ifn_launch_prefill_moe_down_h(&lw->w2, s->p_moe_act_h, s->p_exp_offsets, s->p_sorted_tok,
                                      s->p_sorted_slot, s->p_sorted_w, s->p_moe_slot_out, n,
                                      stream);
        if (vi == 17 || vi == 29)
            ifn_launch_prefill_sconv_post_res(
                s->p_moe_slot_out, 1, lw->mlp_sconv_w, s->mlp_ring[vi], s->p_residual, s->p_hidden,
                NULL, NULL, NULL, 0, 0, n, s->p_pos_vec, NULL, NULL, NULL, stream);
        else
            pending = true;
        IFN_ENGINE_PREFILL_LOCAL_CHECK(ifn_cuda_launch_status());
    }
    ifn_launch_prefill_sconv_post_res(s->p_moe_slot_out, 1,
                                      w->layers[s->schedule[47].layer].mlp_sconv_w, s->mlp_ring[47],
                                      s->p_residual, s->p_hidden, NULL, NULL, NULL, 0, 0, n,
                                      s->p_pos_vec, NULL, NULL, NULL, stream);
    ifn_launch_fused_depth_norm(s->p_hidden, NULL, s->p_blocks, blocks, w->output_query,
                                w->final_norm, NULL, NULL, s->p_mlp_normed, NULL, NULL, NULL, NULL,
                                NULL, 1, n, stream);
    IFN_PREFILL_NUMERIC_TAP(48, "final_norm", s->p_mlp_normed, 2304);
    if (s->use_mtp) {
        ifn_launch_prefill_copy_taps(s->p_mlp_normed, s->p_taps, 3, n, stream);
        for (int32_t i = 0; i < n / 8; i++)
            ifn_spec_commit(s, s->p_taps + (size_t)i * 4 * 8 * 2304, 8, s->p_pos_vec + i);
    }
    IFN_ENGINE_PREFILL_LOCAL_CHECK(ifn_cuda_launch_status());
    IFN_ENGINE_PREFILL_LOCAL_CHECK(backend->sync(backend->context, stream));
    s->position += n;
}

#undef IFN_PREFILL_NUMERIC_TAP
