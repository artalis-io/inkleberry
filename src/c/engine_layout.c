/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Two-pass target and optional fixed-width DSpark workspace layout. B<=8
 * matches target-forward geometry; adaptive/prefix buffers are not allocated. */
#include "engine_internal.h"

static void *take(sq_session *s, size_t count, size_t width) {
    size_t bytes, aligned, end;
    if (s->status != SQ_OK)
        return NULL;
    if (!sq_size_mul(count, width, &bytes) || !sq_size_align(bytes, 256, &aligned) ||
        !sq_size_add(s->offset, aligned, &end) || (s->arena && end > s->capacity)) {
        s->status = SQ_LIMIT;
        return NULL;
    }
    void *p = s->arena ? (unsigned char *)s->arena + s->offset : NULL;
    s->offset = end;
    s->allocations++;
    return p;
}

void sq_ar_layout(sq_session *s, size_t groups) {
    s->offset = s->allocations = 0;
    sq_ar_schedule(s->schedule);
    for (unsigned i = 0; i < 48; i++) {
        bool global = s->schedule[i].global;
        size_t cap = global ? s->config.context_tokens : 1280;
        unsigned format = global ? s->config.kv_format : 0;
        s->kv_cap[i] = (int32_t)cap;
        size_t width = format == 0 ? 2048 : format == 1 ? 1024 : 512;
        s->k_cache[i] = take(s, cap, width);
        s->v_cache[i] = take(s, cap, width);
        s->k_scale[i] = format ? take(s, cap, format == 1 ? 16 : 32) : NULL;
        s->v_scale[i] = format ? take(s, cap, format == 1 ? 16 : 32) : NULL;
        s->k_ring[i] = take(s, 16 * 1024, 4);
        s->v_ring[i] = take(s, 16 * 1024, 4);
        s->attn_ring[i] = take(s, 16 * 2304, 4);
        s->mlp_ring[i] = take(s, 16 * 2304, 4);
    }
#define F(name, n) s->name = take(s, (n), 4)
#define H(name, n) s->name = take(s, (n), 2)
    if (s->weights->config.with_mtp) {
        for (unsigned i = 0; i < 3; i++) {
            size_t cap = i != 1                              ? 1032
                         : s->config.context_tokens > 524288 ? 65536
                                                             : s->config.context_tokens;
            unsigned format = i != 1 ? 0 : s->config.kv_format;
            s->ds_cap[i] = (int32_t)cap;
            size_t width = format == 0 ? 2048 : format == 1 ? 1024 : 512;
            s->ds_k[i] = take(s, cap, width);
            s->ds_v[i] = take(s, cap, width);
            s->ds_ks[i] = format ? take(s, cap, format == 1 ? 16 : 32) : NULL;
            s->ds_vs[i] = format ? take(s, cap, format == 1 ? 16 : 32) : NULL;
        }
        F(ds_f_ring, 16 * 2304);
        F(taps, 4 * 8 * 2304);
        F(ds_mix, 3 * 8 * 2304);
        F(ds_kv, 8 * 2 * 1024);
        F(ds_ws, 2000000);
        F(d_anchor, 1);
        F(d_drafts, 7);
        F(d_accepted, 8);
        F(d_num_acc, 1);
        F(d_draft_is_delta, 1);
        F(d_draft_probs, 7 * 200058);
        F(d_draft_q_sel, 7);
        F(d_conf, 7);
        F(d_width, 1);
    }
    F(split_buf, 2050000);
    F(d_pos, 1);
    F(d_tokens, 8);
    F(d_next_tokens, 8);
    F(hidden, 8 * 2304);
    F(prelude, 8 * 2304);
    F(partial, 8 * 2304);
    F(blocks, 8 * 4 * 2304);
    F(saved, 8 * 2304);
    F(prior_last, 8 * 2304);
    for (unsigned i = 0; i < 3; i++) {
        s->attn_in[i] = take(s, 8 * 2304, 4);
        s->attn_in_h[i] = take(s, 8 * 2304, 2);
    }
    F(qrgs, 8 * 6552);
    F(k_proj, 8 * 1024);
    F(v_proj, 8 * 1024);
    F(attn_gated, 8 * 3072);
    H(attn_gated_h, 8 * 3072);
    F(o_proj, 8 * 2304);
    F(residual, 8 * 2304);
    F(mlp_in, 8 * 2304);
    F(mlp_normed, 8 * 2304);
    H(mlp_normed_h, 8 * 2304);
    F(ctrl_msg, 3 * 8 * 2304);
    F(ctrl_cat, 3 * 8 * 2 * 2304);
    F(ctrl_d, 3 * 8 * 64);
    F(router_logits, 8 * 128);
    F(route_ids, 8 * 8);
    F(route_w, 8 * 8);
    s->exp_groups = take(s, groups, 1);
    H(moe_act_h, 8 * 8 * 1152);
    F(moe_down, 8 * 2304);
    F(lm_blk_val, 8 * 1024);
    F(lm_blk_idx, 8 * 1024);
    F(lm_logits, 8 * 200058);
    s->d_sample_cfg = take(s, 1, sizeof(CSampleConfig));
    s->d_loopcd_cfg = take(s, 1, sizeof(CLoopCDConfig));
    F(loopcd_h1, 8 * 2304);
    F(loopcd_h2, 8 * 2304);

    enum { P = 2048 };

    if (s->weights->config.with_mtp) {
        F(p_taps, 4 * P * 2304);
    }
    F(p_tokens, P);
    F(p_pos_vec, P / 8);
    F(p_hidden, P * 2304);
    F(p_prelude, P * 2304);
    F(p_partial, P * 2304);
    F(p_blocks, P * 4 * 2304);
    F(p_saved, P * 2304);
    F(p_prior_last, P * 2304);
    for (unsigned i = 0; i < 3; i++) {
        s->p_attn_in[i] = take(s, P * 2304, 4);
        s->p_attn_in_h[i] = take(s, P * 2304, 2);
    }
    F(p_qrgs, P * 6552);
    F(p_k_proj, P * 1024);
    F(p_v_proj, P * 1024);
    H(p_attn_gated_h, P * 3072);
    F(p_o_proj, P * 2304);
    F(p_residual, P * 2304);
    F(p_mlp_in, P * 2304);
    F(p_mlp_normed, P * 2304);
    H(p_mlp_normed_h, P * 2304);
    F(p_router_logits, P * 128);
    F(p_route_ids, P * 8);
    F(p_route_w, P * 8);
    F(p_exp_offsets, 1024);
    F(p_sorted_tok, P * 8);
    F(p_sorted_slot, P * 8);
    F(p_sorted_w, P * 8);
    H(p_moe_act_h, P * 8 * 1152);
    F(p_moe_slot_out, P * 8 * 2304);
    H(p_w_f16, 6552 * 2304);
    F(p_ctrl_msg, P * 2304);
    H(p_ctrl_cat_h, P * 4608);
    F(p_ctrl_d, P * 64);
    H(p_ctrl_d_h, P * 64);
    s->p_ctrl_f = s->p_qrgs;
    s->p_ctrl_g = s->p_qrgs ? s->p_qrgs + P * 2304 : NULL;
#undef F
#undef H
}
