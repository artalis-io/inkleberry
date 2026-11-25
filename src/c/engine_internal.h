/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Fixed Inkling AR structures translated from pinned engine.zig/weights.zig. */
#ifndef SQ_ENGINE_INTERNAL_H
#define SQ_ENGINE_INTERNAL_H
#include "weights_internal.h"
#include "sq_engine.h"
#include "cuda_abi.h"

typedef struct {
    void *attn_norm;
    void *q_in_norm;
    void *k_in_norm;
    void *v_in_norm;
    void *mlp_norm;
    CLinearWeight w_qrgs;
    CLinearWeight w_kv;
    CLinearWeight w_k;
    CLinearWeight w_v;
    CLinearWeight w_o;
    void *k_sconv_w;
    void *v_sconv_w;
    void *k_sconv_proj;
    void *v_sconv_proj;
    void *attn_sconv_w;
    void *mlp_sconv_w;
    void *q_clip_scale;
    void *rel_clip_scale;
    void *rel_logits_proj;
    CControllerWeights ctrl[4];
    void *router_w;
    CLinearWeight router_nvfp4;
    void *router_bias;
    CExpertBank w13;
    CExpertBank w2;
} sq_layer;

typedef struct {
    void *embed, *embed_norm, *final_norm;
    CLinearWeight unembed;
    float *attn_queries[23], *mlp_queries[24], *output_query;
    float *cross_pass_queries, *cross_pass_kv_queries;
    sq_layer layers[24];
} sq_ar_model;

typedef struct {
    uint32_t layer, pass;
    bool loop, global;
} sq_visit;

struct sq_session {
    sq_options allocator;
    sq_session_config config;
    sq_gpu_weights *weights;
    sq_ar_model model;
    sq_visit schedule[48];
    atomic_flag busy;
    atomic_bool cancelled;
    bool poisoned, ready, retained;
    int32_t position;
    void *arena, *stream, *graph, *blas;
    size_t capacity, offset, allocations;
    int32_t kv_cap[48];
    void *k_cache[48], *v_cache[48], *k_scale[48], *v_scale[48];
    float *k_ring[48], *v_ring[48], *attn_ring[48], *mlp_ring[48];
    float *split_buf;
    int32_t *d_pos, *d_tokens, *d_next_tokens;
    float *hidden, *prelude, *partial, *blocks, *saved, *prior_last;
    float *attn_in[3];
    void *attn_in_h[3];
    float *qrgs, *k_proj, *v_proj, *attn_gated;
    void *attn_gated_h;
    float *o_proj, *residual, *mlp_in, *mlp_normed;
    void *mlp_normed_h;
    float *ctrl_msg, *ctrl_cat, *ctrl_d, *router_logits;
    int32_t *route_ids;
    float *route_w;
    void *exp_groups, *moe_act_h;
    float *moe_down;
    float *lm_blk_val;
    int32_t *lm_blk_idx;
    float *lm_logits;
    CSampleConfig *d_sample_cfg;
    CLoopCDConfig *d_loopcd_cfg;
    float *loopcd_h1, *loopcd_h2;
    int32_t *p_tokens, *p_pos_vec;
    float *p_hidden, *p_prelude, *p_partial, *p_blocks, *p_saved, *p_prior_last;
    float *p_attn_in[3];
    void *p_attn_in_h[3];
    float *p_qrgs, *p_k_proj, *p_v_proj;
    void *p_attn_gated_h;
    float *p_o_proj, *p_residual, *p_mlp_in, *p_mlp_normed;
    void *p_mlp_normed_h;
    float *p_router_logits;
    int32_t *p_route_ids;
    float *p_route_w;
    int32_t *p_exp_offsets, *p_sorted_tok, *p_sorted_slot;
    float *p_sorted_w;
    void *p_moe_act_h;
    float *p_moe_slot_out;
    void *p_w_f16;
    float *p_ctrl_msg;
    void *p_ctrl_cat_h;
    float *p_ctrl_d;
    void *p_ctrl_d_h;
    float *p_ctrl_f, *p_ctrl_g;
    sq_status status;
    char failure[192];
};

sq_status sq_ar_bind(sq_gpu_weights *, sq_ar_model *);
void sq_ar_schedule(sq_visit[48]);
void sq_ar_layout(sq_session *, size_t);
void sq_ar_prefill(sq_session *, const int32_t *, int32_t);
void sq_ar_forward(sq_session *, int32_t, bool);
void sq_ar_check(sq_session *, int, const char *);
#endif
