/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Fixed Inkling target/DSpark structures translated from pinned engine.zig/weights.zig. */
#ifndef IFN_ENGINE_INTERNAL_H
#define IFN_ENGINE_INTERNAL_H
#include "weights_internal.h"
#include "ifn_engine.h"
#include "engine_control.h"
#include "cuda_abi.h"
#include "lookup.h"
#include "prefix.h"

typedef struct {
    void *attn_norm;
    void *q_in_norm;
    void *k_in_norm;
    void *v_in_norm;
    void *mlp_norm;
    ifn_cuda_c_linear_weight w_qrgs;
    ifn_cuda_c_linear_weight w_kv;
    ifn_cuda_c_linear_weight w_k;
    ifn_cuda_c_linear_weight w_v;
    ifn_cuda_c_linear_weight w_o;
    void *k_sconv_w;
    void *v_sconv_w;
    void *k_sconv_proj;
    void *v_sconv_proj;
    void *attn_sconv_w;
    void *mlp_sconv_w;
    void *q_clip_scale;
    void *rel_clip_scale;
    void *rel_logits_proj;
    ifn_cuda_c_controller_weights ctrl[4];
    void *router_w;
    ifn_cuda_c_linear_weight router_nvfp4;
    void *router_bias;
    ifn_cuda_c_expert_bank w13;
    ifn_cuda_c_expert_bank w2;
} ifn_layer;

typedef struct {
    void *embed, *embed_norm, *final_norm;
    ifn_cuda_c_linear_weight unembed;
    float *attn_queries[23], *mlp_queries[24], *output_query;
    float *cross_pass_queries, *cross_pass_kv_queries;
    ifn_layer layers[24];
    ifn_cuda_c_dspark_weights dspark;
} ifn_ar_model;

typedef struct {
    uint32_t layer, pass;
    bool loop, global;
} ifn_visit;

struct ifn_session {
    ifn_options allocator;
    ifn_session_config config;
    ifn_gpu_weights *weights;
    ifn_ar_model model;
    ifn_visit schedule[48];
    atomic_flag busy;
    atomic_bool cancelled;
    const ifn_run_control *request_control;
    bool poisoned, ready, retained;
    int32_t position;
    void *arena, *stream, *graph, *blas;
    void *spec_graph[8];
    void *adaptive_state, *draft_graph, *lookup_graph, *done_event, *pinned;
    int32_t *h_accepted, *h_count, *h_width;
    float adaptive_threshold;
    bool draft_in_flight;
    ifn_lookup lookup;
    ifn_decode_info decode_info;
    ifn_prefix prefix;
    ifn_prefix_info prefix_info;
    bool use_mtp, verifying;
    int32_t ds_cap[3];
    void *ds_k[3], *ds_v[3], *ds_ks[3], *ds_vs[3];
    float *ds_f_ring, *taps, *ds_mix, *ds_kv, *ds_ws;
    int32_t *d_anchor, *d_drafts, *d_accepted, *d_num_acc, *d_draft_is_delta;
    float *d_draft_probs, *d_draft_q_sel, *d_conf;
    int32_t *d_width;
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
    ifn_cuda_c_sample_config *d_sample_cfg;
    ifn_cuda_c_loop_cd_config *d_loopcd_cfg;
    float *loopcd_h1, *loopcd_h2;
    int32_t *p_tokens, *p_pos_vec;
    float *p_taps;
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
    ifn_status status;
    char failure[192];
};

ifn_status ifn_decode_resolve(const ifn_decode_options *, ifn_decode_options *);
void ifn_spec_draft(ifn_session *, uint32_t, float *);
void ifn_spec_target(ifn_session *, uint32_t);
void ifn_spec_verify(ifn_session *, uint32_t);
void ifn_adaptive_step(ifn_session *, const ifn_decode_options *, bool);
ifn_status ifn_prefix_allocate(ifn_session *);
void ifn_prefix_free(ifn_session *);
void ifn_prefix_invalidate(ifn_session *);
uint32_t ifn_prefix_begin(ifn_session *, const int32_t *, uint32_t, uint32_t, bool);
void ifn_prefix_save(ifn_session *, unsigned, uint32_t);
void ifn_prefix_prompt(ifn_session *, const int32_t *, uint32_t, bool);
void ifn_prefix_feed(ifn_session *, int32_t);
void ifn_prefix_finish(ifn_session *, bool);
void ifn_adaptive_drain(ifn_session *);
void ifn_spec_step(ifn_session *, uint32_t);
void ifn_spec_commit(ifn_session *, const float *, int32_t, const int32_t *);
ifn_status ifn_ar_bind(ifn_gpu_weights *, ifn_ar_model *);
void ifn_ar_schedule(ifn_visit[48]);
void ifn_ar_layout(ifn_session *, size_t);
void ifn_ar_prefill(ifn_session *, const int32_t *, int32_t);
void ifn_ar_forward(ifn_session *, int32_t, bool);
void ifn_ar_check(ifn_session *, int, const char *);
#endif
