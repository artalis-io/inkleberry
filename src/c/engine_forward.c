/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Target-forward translation of pinned Engine.record_target_forward. Optional
 * DSpark commit/verification; ablation and LoopCD remain disabled. */
#include "engine_internal.h"

void sq_ar_forward(sq_session *s, int32_t batch, bool lm) {
    const sq_ar_model *w = &s->model;
    void *stream = s->stream;
    int32_t blocks = 0;
    bool prior = false, pending = false;
    squid_launch_embed_norm(w->embed, w->embed_norm, s->d_tokens, s->hidden, batch, stream);
    for (unsigned vi = 0; vi < 48; vi++) {
        sq_visit v = s->schedule[vi];
        unsigned l = v.layer;
        const sq_layer *lw = &w->layers[l];
        float *pre_tap = NULL;
        if (v.loop && l == 6) {
            if (v.pass == 0)
                pre_tap = s->prelude;
            else {
                squid_launch_extract_block(s->blocks, 4, 2, s->saved, batch, stream);
                squid_launch_rmsnorm(s->hidden, NULL, s->prior_last, batch, 2304, 2304, 1e-6f,
                                     stream);
                sq_ar_check(s,
                            squid_cuda_memcpy_d2d_async(s->hidden, s->prelude,
                                                        (size_t)batch * 2304 * 4, stream),
                            "restore prelude");
                blocks = 1;
                prior = true;
            }
        }
        if (s->use_mtp && (vi == 5 || vi == 23 || vi == 41)) {
            unsigned slot = vi == 5 ? 0 : vi == 23 ? 1 : 2;
            pre_tap = s->taps + (size_t)slot * (size_t)batch * 2304;
        }
        bool use_prior = v.loop && prior, distinct = v.global || use_prior;
        float *partial = l % 6 == 0 ? s->partial : NULL;
        bool half = lw->w_qrgs.kind != 0;
        const void *n[3] = {NULL, NULL, NULL};
        float *o[3] = {NULL, NULL, NULL};
        void *oh[3] = {NULL, NULL, NULL};
        int32_t count = 1;
        if (!distinct) {
            n[0] = lw->attn_norm;
            o[0] = s->attn_in[0];
            oh[0] = half ? s->attn_in_h[0] : NULL;
        } else if (v.global && !use_prior) {
            n[0] = lw->q_in_norm;
            n[1] = lw->k_in_norm;
            n[2] = lw->v_in_norm;
            for (unsigned j = 0; j < 3; j++) {
                o[j] = s->attn_in[j];
                oh[j] = half ? s->attn_in_h[j] : NULL;
            }
            count = 3;
        } else if (v.global) {
            for (unsigned j = 0; j < 3; j++)
                o[j] = s->attn_in[j];
            count = 3;
        } else
            o[0] = s->mlp_in;
        if (pending) {
            const sq_layer *prev = &w->layers[s->schedule[vi - 1].layer];
            squid_launch_mega_mlp_post_next_depth_norm(
                s->moe_down, 0, prev->mlp_sconv_w, s->mlp_ring[vi - 1], s->residual, s->hidden,
                pre_tap, NULL, partial, s->blocks, blocks, w->attn_queries[l - 1], n[0], n[1], n[2],
                o[0], o[1], o[2], oh[0], oh[1], oh[2], count, batch, s->d_pos, NULL, NULL, NULL,
                vi == 42 ? s->loopcd_h1 : NULL, vi == 42 ? s->loopcd_h2 : NULL,
                vi == 42 ? s->d_loopcd_cfg : NULL, stream);
            pending = false;
        } else
            squid_launch_fused_depth_norm(s->hidden, partial, s->blocks, blocks,
                                          l ? w->attn_queries[l - 1] : NULL, n[0], n[1], n[2], o[0],
                                          o[1], o[2], oh[0], oh[1], oh[2], count, batch, stream);
        if (use_prior) {
            unsigned loc = l - 6;
            float *xs[3] = {v.global ? s->attn_in[0] : s->mlp_in,
                            v.global ? s->attn_in[1] : s->mlp_in,
                            v.global ? s->attn_in[2] : s->mlp_in};
            float *outs[3] = {s->attn_in[0], s->attn_in[1], s->attn_in[2]};
            void *out_h[3] = {s->attn_in_h[0], s->attn_in_h[1], s->attn_in_h[2]};
            const float *queries[3] = {w->cross_pass_queries + 2 * loc * 2304,
                                       w->cross_pass_kv_queries + 2 * loc * 2304,
                                       w->cross_pass_kv_queries + (2 * loc + 1) * 2304};
            const CControllerWeights *cw[3] = {&lw->ctrl[0], &lw->ctrl[1], &lw->ctrl[2]};
            const void *norms[3] = {v.global ? lw->q_in_norm : lw->attn_norm,
                                    v.global ? lw->k_in_norm : lw->attn_norm,
                                    v.global ? lw->v_in_norm : lw->attn_norm};
            if (s->weights->config.mode == 2)
                squid_launch_controller_mega_nvfp4(xs, s->saved, s->prior_last, queries, cw, norms,
                                                   outs, half ? out_h : NULL, 3, batch, stream);
            else
                squid_launch_controller_merge_batched(
                    xs, s->saved, s->prior_last, queries, cw, norms, outs, half ? out_h : NULL, 3,
                    s->ctrl_msg, s->ctrl_cat, s->ctrl_d, batch, stream);
        }
        float *in_q = s->attn_in[0], *in_k = s->attn_in[distinct ? 1 : 0],
              *in_v = s->attn_in[distinct ? 2 : 0];
        if (half)
            squid_launch_qkvrg_fused(&lw->w_qrgs, &lw->w_k, &lw->w_v, s->attn_in_h[0],
                                     s->attn_in_h[distinct ? 1 : 0], s->attn_in_h[distinct ? 2 : 0],
                                     s->qrgs, s->k_proj, s->v_proj, batch, stream);
        else {
            squid_launch_linear(&lw->w_qrgs, in_q, s->qrgs, batch, 0, stream);
            squid_launch_linear(&lw->w_k, in_k, s->k_proj, batch, 0, stream);
            squid_launch_linear(&lw->w_v, in_v, s->v_proj, batch, 0, stream);
        }
        if (v.global)
            squid_launch_mega_global_sconv_prep_store(
                s->qrgs, s->k_proj, s->v_proj, in_k, in_v, lw->k_sconv_w, lw->v_sconv_w,
                lw->k_sconv_proj, lw->v_sconv_proj, s->k_ring[vi], s->v_ring[vi], lw->q_clip_scale,
                lw->rel_clip_scale, s->k_cache[vi], s->v_cache[vi], s->k_scale[vi], s->v_scale[vi],
                (uint8_t)s->config.kv_format, s->kv_cap[vi], batch, s->d_pos, stream);
        else
            squid_launch_mega_local_sconv_prep_store(
                s->qrgs, s->k_proj, s->v_proj, lw->k_sconv_w, lw->v_sconv_w, s->k_ring[vi],
                s->v_ring[vi], lw->q_clip_scale, lw->rel_clip_scale, s->k_cache[vi], s->v_cache[vi],
                s->kv_cap[vi], batch, s->d_pos, stream);
        squid_launch_relative_attention_gated(
            s->qrgs, lw->rel_logits_proj, s->k_cache[vi], s->v_cache[vi], s->k_scale[vi],
            s->v_scale[vi], v.global ? (uint8_t)s->config.kv_format : 0, s->split_buf,
            half ? NULL : s->attn_gated, half ? s->attn_gated_h : NULL, s->kv_cap[vi],
            v.global ? 0 : 1024, batch, s->d_pos, stream);
        if (half)
            squid_launch_linear_h(&lw->w_o, s->attn_gated_h, s->o_proj, batch, 0, stream);
        else
            squid_launch_linear(&lw->w_o, s->attn_gated, s->o_proj, batch, 0, stream);
        bool boundary = l % 6 == 0;
        squid_launch_mega_attn_post_mlp_in(
            s->o_proj, lw->attn_sconv_w, s->attn_ring[vi], s->hidden, s->partial, s->blocks, blocks,
            boundary ? 1 : 0, s->residual, w->mlp_queries[l], use_prior ? NULL : lw->mlp_norm,
            use_prior ? s->mlp_in : s->mlp_normed, use_prior ? NULL : s->mlp_normed_h, batch,
            s->d_pos, NULL, NULL, NULL, stream);
        if (boundary)
            blocks++;
        if (use_prior) {
            float *xs[1] = {s->mlp_in}, *outs[1] = {s->mlp_normed};
            void *outs_h[1] = {s->mlp_normed_h};
            const float *q[3] = {w->cross_pass_queries + (2 * (l - 6) + 1) * 2304, NULL, NULL};
            const CControllerWeights *cw[3] = {&lw->ctrl[3], NULL, NULL};
            const void *norms[3] = {lw->mlp_norm, NULL, NULL};
            if (s->weights->config.mode == 2)
                squid_launch_controller_mega_nvfp4(xs, s->saved, s->prior_last, q, cw, norms, outs,
                                                   outs_h, 1, batch, stream);
            else
                squid_launch_controller_merge_batched(xs, s->saved, s->prior_last, q, cw, norms,
                                                      outs, outs_h, 1, s->ctrl_msg, s->ctrl_cat,
                                                      s->ctrl_d, batch, stream);
        }
        if (s->weights->config.mode == 2)
            squid_launch_route_topk_nvfp4(s->mlp_normed_h, &lw->router_nvfp4, lw->router_bias,
                                          s->router_logits, s->route_ids, s->route_w, NULL, batch,
                                          stream);
        else
            squid_launch_route_topk(s->mlp_normed, lw->router_w, lw->router_bias, s->router_logits,
                                    s->route_ids, s->route_w, NULL, batch, stream);
        squid_launch_moe_gate_up_h(&lw->w13, s->mlp_normed_h, s->route_ids, s->exp_groups,
                                   s->moe_act_h, batch, stream);
        squid_launch_moe_down_h(&lw->w2, s->moe_act_h, s->route_ids, s->route_w, s->exp_groups,
                                s->moe_down, batch, stream);
        if (vi == 17 || vi == 29) {
            squid_launch_mega_mlp_post_res(
                s->moe_down, 0, lw->mlp_sconv_w, s->mlp_ring[vi], s->residual, s->hidden,
                vi == 17 ? s->loopcd_h1 : s->loopcd_h2, batch, s->d_pos, NULL, NULL, NULL, stream);
            pending = false;
        } else
            pending = true;
        if (s->status != SQ_OK)
            return;
    }
    squid_launch_mega_mlp_post_next_depth_norm(
        s->moe_down, 0, w->layers[s->schedule[47].layer].mlp_sconv_w, s->mlp_ring[47], s->residual,
        s->hidden, NULL, s->use_mtp ? s->taps + 3 * (size_t)batch * 2304 : NULL, NULL, s->blocks,
        blocks, w->output_query, w->final_norm, NULL, NULL, s->mlp_normed, NULL, NULL, NULL, NULL,
        NULL, 1, batch, s->d_pos, NULL, NULL, NULL, NULL, NULL, NULL, stream);
    sq_spec_commit(s, s->taps, batch, s->d_pos);
    if (lm)
        squid_launch_lm_head_argmax(
            &w->unembed, s->mlp_normed, s->d_next_tokens, s->lm_logits, s->lm_blk_val,
            s->lm_blk_idx, s->d_sample_cfg, s->d_pos, s->verifying ? s->d_drafts : NULL,
            s->verifying ? s->d_draft_probs : NULL, s->verifying ? s->d_draft_q_sel : NULL,
            s->verifying ? s->d_draft_is_delta : NULL,
            s->verifying ? w->dspark.inv_draft_vocab_ids : NULL, w->dspark.draft_vocab_size, batch,
            stream);
}
