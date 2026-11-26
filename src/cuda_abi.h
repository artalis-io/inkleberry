/* Generated from abi/cuda.json by tools/generate_cuda_abi.py; do not edit.
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis */
#ifndef IFN_CUDA_ABI_H
#define IFN_CUDA_ABI_H
#include <stddef.h>
#include <stdint.h>
#ifndef IFN_CUDA_ABI_NO_TYPES
typedef struct ifn_cuda_c_sample_config {
    float inv_temp;
    float top_p;
    int32_t top_k;
    uint32_t pad;
    uint64_t seed;
} ifn_cuda_c_sample_config;
typedef struct ifn_cuda_c_loop_cd_config {
    float omega;
    int32_t ref_pass;
} ifn_cuda_c_loop_cd_config;
typedef struct ifn_cuda_c_linear_weight {
    uint8_t kind;
    int32_t rows;
    int32_t cols;
    void * data;
    uint8_t * scales;
    float * row_scales;
    float global_scale;
} ifn_cuda_c_linear_weight;
typedef struct ifn_cuda_c_expert_bank {
    uint8_t is_nvfp4;
    int32_t n_experts;
    int32_t rows;
    int32_t cols;
    uint8_t * data;
    uint8_t * scales;
    float * expert_scale;
} ifn_cuda_c_expert_bank;
typedef struct ifn_cuda_c_controller_weights {
    void * down_w;
    void * forget_w;
    void * gain_w;
    ifn_cuda_c_linear_weight down_nvfp4;
    ifn_cuda_c_linear_weight forget_nvfp4;
    ifn_cuda_c_linear_weight gain_nvfp4;
} ifn_cuda_c_controller_weights;
typedef struct ifn_cuda_c_dspark_layer_weights {
    uint8_t is_local;
    void * input_norm;
    void * post_norm;
    void * dflare_norm;
    void * q_norm;
    void * k_norm;
    void * k_norm_ctx;
    ifn_cuda_c_linear_weight w_qkv;
    ifn_cuda_c_linear_weight w_o;
    ifn_cuda_c_linear_weight w_gu;
    ifn_cuda_c_linear_weight w_down;
    ifn_cuda_c_linear_weight w_kv_ctx;
} ifn_cuda_c_dspark_layer_weights;
typedef struct ifn_cuda_c_dspark_weights {
    uint8_t loaded;
    float * fuse_alpha;
    void * mask_embed;
    void * norm_w;
    void * markov_dflare_norm;
    void * markov_w1;
    void * rnn_w_sp;
    void * rnn_w_h;
    void * rnn_w_f;
    void * rnn_bias;
    void * hidden_out_w;
    void * hidden_out_b;
    int32_t draft_vocab_size;
    int32_t * draft_vocab_ids;
    int32_t * inv_draft_vocab_ids;
    ifn_cuda_c_linear_weight unembed_draft;
    ifn_cuda_c_linear_weight fold_w;
    float * fold_b;
    ifn_cuda_c_dspark_layer_weights layers[3];
} ifn_cuda_c_dspark_weights;
#endif
#ifdef __cplusplus
extern "C" {
#endif
int32_t ifn_cuda_malloc(void * * ptr, size_t bytes);
int32_t ifn_cuda_free(void * ptr);
int32_t ifn_cuda_malloc_host_pinned(void * * ptr, size_t bytes, uint8_t mapped);
int32_t ifn_cuda_free_host(void * ptr);
int32_t ifn_cuda_host_get_device_pointer(void * * dev_ptr, void * host_ptr);
int32_t ifn_cuda_memcpy_h2d(void * dst, const void * src, size_t bytes);
int32_t ifn_cuda_memcpy_d2h(void * dst, const void * src, size_t bytes);
int32_t ifn_cuda_memcpy_h2d_async(void * dst, const void * src, size_t bytes, void * stream);
int32_t ifn_cuda_memcpy_d2h_async(void * dst, const void * src, size_t bytes, void * stream);
int32_t ifn_cuda_memcpy_d2d_async(void * dst, const void * src, size_t bytes, void * stream);
int32_t ifn_cuda_memset_async(void * dst, int32_t value, size_t bytes, void * stream);
int32_t ifn_cuda_device_sync(void);
int32_t ifn_cuda_stream_create(void * * out_stream);
int32_t ifn_cuda_stream_destroy(void * stream);
int32_t ifn_cuda_stream_sync(void * stream);
int32_t ifn_cuda_event_create(void * * out_event, uint8_t disable_timing);
int32_t ifn_cuda_event_destroy(void * ev);
int32_t ifn_cuda_event_record(void * ev, void * stream);
int32_t ifn_cuda_event_sync(void * ev);
int32_t ifn_cuda_event_elapsed_ms(float * out_ms, void * ev0, void * ev1);
int32_t ifn_cuda_stream_wait_event(void * stream, void * ev);
int32_t ifn_cuda_graph_begin(void * stream);
int32_t ifn_cuda_graph_end_instantiate(void * stream, void * * out_exec);
int32_t ifn_cuda_graph_launch(void * exec, void * stream);
int32_t ifn_cuda_graph_destroy(void * exec);
float ifn_pack_bf16_to_nvfp4(const void * src_bf16_dev, void * dst_data, uint8_t * dst_scales, float * d_amax_scratch, int32_t rows, int32_t cols);
void ifn_pack_bf16_to_fp8(const void * src_bf16_dev, void * dst_data, float * dst_row_scales, int32_t rows, int32_t cols);
void ifn_convert_mxfp8_to_nvfp4_experts(const uint8_t * src_w, const uint8_t * src_s, uint8_t * dst_w, uint8_t * dst_s, float * dst_expert_scale, int32_t n_experts, int32_t rows, int32_t cols);
void ifn_split_dspark_rnn(const void * src_bf16, void * w_sp, void * w_h, void * w_f);
void ifn_compute_dspark_fold(const void * unembed_bf16, const void * hidden_out_w, const void * hidden_out_b, float * fold_b, void * fold_bf16_out, int32_t vocab_count, int32_t hidden_cols, int32_t markov_rank);
void ifn_launch_embed_norm(const void * embed, const void * embed_norm, const int32_t * tokens, float * out, int32_t batch_size, void * stream);
void ifn_launch_rmsnorm(const float * x, const void * weight, float * out, int32_t batch_size, int32_t width, int32_t group_size, float eps, void * stream);
void ifn_launch_extract_block(const float * blocks, int32_t max_blocks, int32_t slot_idx, float * dst, int32_t batch_size, void * stream);
void ifn_launch_fused_depth_norm(const float * hidden, float * partial_out, const float * blocks, int32_t n_blocks, const float * queries, const void * n0, const void * n1, const void * n2, float * o0, float * o1, float * o2, void * oh0, void * oh1, void * oh2, int32_t n_out, int32_t batch_size, void * stream);
void ifn_launch_controller_mega_nvfp4(float * const * xs, const float * saved, const float * prior_last, const float * const * cp_queries, const ifn_cuda_c_controller_weights * const * cws_in, const void * const * post_norms, float * const * outs, void * const * outs_h, int32_t n_streams, int32_t batch_size, void * stream);
void ifn_launch_controller_merge_batched(float * const * xs, const float * saved, const float * prior_last, const float * const * cp_queries, const ifn_cuda_c_controller_weights * const * cws_in, const void * const * post_norms, float * const * outs, void * const * outs_h, int32_t n_streams, float * scratch_msg, float * scratch_cat, float * scratch_d, int32_t batch_size, void * stream);
void ifn_launch_qkvrg_fused(const ifn_cuda_c_linear_weight * w_qrgs, const ifn_cuda_c_linear_weight * w_k, const ifn_cuda_c_linear_weight * w_v, const void * in_qh, const void * in_kh, const void * in_vh, float * qrgs_out, float * k_out, float * v_out, int32_t batch_size, void * stream);
void ifn_launch_linear(const ifn_cuda_c_linear_weight * w, const float * x, float * y, int32_t batch_size, uint8_t add_res, void * stream);
void ifn_launch_linear_h(const ifn_cuda_c_linear_weight * w, const void * x_h, float * y, int32_t batch_size, uint8_t add_res, void * stream);
void ifn_launch_mega_global_sconv_prep_store(float * qrgs, const float * k_proj, const float * v_proj, const float * in_k, const float * in_v, const void * k_sw, const void * v_sw, const void * k_sp, const void * v_sp, float * k_ring, float * v_ring, const void * q_cs, const void * r_cs, void * k_cache, void * v_cache, void * k_scale, void * v_scale, uint8_t kv_fmt, int32_t kv_cap, int32_t batch_size, const int32_t * d_pos, void * stream);
void ifn_launch_mega_local_sconv_prep_store(float * qrgs, const float * k_proj, const float * v_proj, const void * k_sw, const void * v_sw, float * k_ring, float * v_ring, const void * q_cs, const void * r_cs, void * k_cache, void * v_cache, int32_t kv_cap, int32_t batch_size, const int32_t * d_pos, void * stream);
void ifn_launch_relative_attention_gated(const float * qrgs, const void * rel_proj, const void * k_cache, const void * v_cache, const void * k_scale, const void * v_scale, uint8_t kv_fmt, float * split_buf, float * out_gated, void * out_gated_h, int32_t kv_cap, int32_t window, int32_t batch_size, const int32_t * d_pos, void * stream);
void ifn_launch_mega_attn_post_mlp_in(const float * o_proj, const void * attn_sw, float * attn_ring, const float * hidden, const float * partial_prev, float * blocks, int32_t block_count_before, uint8_t is_boundary, float * residual_out, const float * mlp_query, const void * mlp_norm_w, float * mlp_out, void * mlp_out_h, int32_t batch_size, const int32_t * d_pos, const float * abl_r, const float * abl_d, const float * abl_p, void * stream);
void ifn_launch_route_topk_nvfp4(const void * m_h, const ifn_cuda_c_linear_weight * router_nvfp4, const float * router_bias, float * scratch_logits, int32_t * out_ids, float * out_weights, void * groups, int32_t batch_size, void * stream);
void ifn_launch_route_topk(const float * m, const void * router_w, const float * router_bias, float * scratch_logits, int32_t * out_ids, float * out_weights, void * groups, int32_t batch_size, void * stream);
void ifn_launch_moe_gate_up_h(const ifn_cuda_c_expert_bank * w13, const void * x_h, const int32_t * route_ids, const void * groups, void * out_act_h, int32_t batch_size, void * stream);
void ifn_launch_moe_down_h(const ifn_cuda_c_expert_bank * w2, const void * act_h, const int32_t * route_ids, const float * route_w, const void * groups, float * out_down, int32_t batch_size, void * stream);
void ifn_launch_mega_mlp_post_res(const float * moe_down_or_slots, uint8_t moe_is_slotted, const void * mlp_sw, float * mlp_ring, const float * residual, float * hidden_out, float * loopcd_save_out, int32_t batch_size, const int32_t * d_pos, const float * abl_r, const float * abl_d, const float * abl_p, void * stream);
void ifn_launch_loopcd_hidden_apply(float * hidden, const float * ref_hidden, float omega, int32_t batch_size, void * stream);
void ifn_launch_mega_mlp_post_next_depth_norm(const float * moe_down_or_slots, uint8_t moe_is_slotted, const void * prev_mlp_sw, float * prev_mlp_ring, const float * residual, float * hidden_out, float * pre_tap_out, float * post_tap_out, float * partial_out, const float * blocks, int32_t n_blocks, const float * queries, const void * n0, const void * n1, const void * n2, float * o0, float * o1, float * o2, void * oh0, void * oh1, void * oh2, int32_t n_out, int32_t batch_size, const int32_t * d_pos, const float * abl_r, const float * abl_d, const float * abl_p, const float * loopcd_h1, const float * loopcd_h2, const ifn_cuda_c_loop_cd_config * loopcd_cfg, void * stream);
void ifn_launch_lm_head_argmax(const ifn_cuda_c_linear_weight * unembed, const float * x, int32_t * out_tokens, float * out_logits, float * scratch_vals, int32_t * scratch_idxs, const ifn_cuda_c_sample_config * sample_cfg, const int32_t * d_pos, const int32_t * d_draft_tokens, const float * d_draft_probs, const float * d_draft_q_sel, const int32_t * d_draft_is_delta, const int32_t * inv_draft_vocab, int32_t draft_vocab_size, int32_t batch_size, void * stream);
void ifn_launch_dspark_commit(const ifn_cuda_c_dspark_weights * dw_in, const float * taps, void * const * ctx_k, void * const * ctx_v, void * const * ctx_ks, void * const * ctx_vs, const int32_t * ctx_cap, uint8_t global_kv_fmt, float * f_ring, float * scratch_mix, float * scratch_kv, int32_t batch_size, const int32_t * d_pos, void * stream);
void ifn_launch_dspark_draft(const void * embed, const void * embed_norm, const ifn_cuda_c_dspark_weights * dw_in, const int32_t * d_anchor_token, const int32_t * d_pos, int32_t num_drafts, void * const * ctx_k, void * const * ctx_v, const void * const * ctx_ks, const void * const * ctx_vs, const int32_t * ctx_cap, uint8_t global_kv_fmt, float * split_buf, const float * f_ring, float * workspace, int32_t * out_draft_tokens, float * out_draft_conf, const ifn_cuda_c_sample_config * sample_cfg, float * out_draft_probs, float * out_draft_q_sel, int32_t * out_draft_is_delta, void * stream);
int32_t ifn_adaptive_graph_create(const int32_t * d_anchor, const int32_t * d_drafts, const float * d_draft_conf, float threshold, int32_t * d_tokens, int32_t * d_chosen_width, const int32_t * d_accepted, int32_t * h_accepted, const int32_t * d_num_acc, int32_t * h_num_acc, int32_t * h_chosen_width, void * stream, void * * out_graph);
int32_t ifn_adaptive_graph_branch_begin(void * graph, int32_t branch_idx, void * stream);
int32_t ifn_adaptive_graph_branch_end(void * stream);
int32_t ifn_adaptive_graph_instantiate(void * graph);
int32_t ifn_adaptive_graph_launch(void * graph, void * stream);
int32_t ifn_adaptive_graph_destroy(void * graph);
void ifn_launch_spec_verify(const int32_t * target_preds, const int32_t * draft_tokens, int32_t batch_size, int32_t * out_accepted, int32_t * out_num_accepted, int32_t * d_pos, int32_t * d_next_anchor, void * stream);
void ifn_launch_inc_pos_and_copy_tok(int32_t * d_pos, int32_t * d_tok_in, const int32_t * d_tok_out, void * stream);
void ifn_launch_inc_pos_by(int32_t * d_pos, int32_t delta, void * stream);
void ifn_launch_assemble_verify_tokens(const int32_t * d_anchor, const int32_t * d_drafts, int32_t * d_tokens, int32_t num_drafts, void * stream);
size_t ifn_sizeof_expert_groups(void);
void ifn_launch_prefill_warmup_cublas(void * stream);
void ifn_launch_prefill_f32_to_f16(const float * src, void * dst_h, int32_t count, void * stream);
void ifn_launch_prefill_gemm_h(const ifn_cuda_c_linear_weight * w, const void * x_h, float * y, void * w_f16_scratch, int32_t batch_size, void * stream);
void ifn_launch_prefill_sconv_post_res(const float * src_or_slots, int32_t is_slotted, const void * sconv_w, float * ring, const float * residual, float * out_res, float * pre_tap_out, float * blocks, const float * partial, int32_t block_idx, int32_t is_boundary, int32_t batch_size, const int32_t * d_pos, const float * abl_r, const float * abl_d, const float * abl_p, void * stream);
void ifn_launch_prefill_copy_taps(const float * src, float * dst_taps, int32_t tap_slot, int32_t batch_size, void * stream);
void ifn_launch_prefill_local_sconv_prep_store(float * qrgs, float * k_proj, float * v_proj, const void * k_sw, const void * v_sw, float * k_ring, float * v_ring, const void * q_cs, const void * r_cs, void * k_cache, void * v_cache, int32_t kv_cap, int32_t batch_size, const int32_t * d_pos, void * stream);
void ifn_launch_prefill_global_sconv_prep_store(float * qrgs, float * k_proj, float * v_proj, const void * in_k_h, const void * in_v_h, const void * k_sw, const void * v_sw, const void * k_sp, const void * v_sp, float * k_ring, float * v_ring, const void * q_cs, const void * r_cs, void * k_cache, void * v_cache, void * k_scale, void * v_scale, uint8_t kv_fmt, int32_t kv_cap, float * scratch_taps, void * w_f16_scratch, int32_t batch_size, const int32_t * d_pos, void * stream);
void ifn_launch_prefill_flash_attn_tc(const float * qrgs, const void * rel_proj, const void * k_cache, const void * v_cache, const void * k_scale, const void * v_scale, uint8_t kv_fmt, void * out_gated_h, int32_t kv_cap, int32_t window, int32_t batch_size, const int32_t * d_pos, void * stream);
void ifn_launch_prefill_controller_one(const float * x, const float * saved, const float * prior_last, const float * cp_q, const ifn_cuda_c_controller_weights * cw, const void * post_norm, float * out, void * out_h, float * scratch_msg, void * scratch_cat_h, float * scratch_d, void * scratch_d_h, float * scratch_f, float * scratch_g, void * w_f16_scratch, int32_t batch_size, void * stream);
void ifn_launch_prefill_route_and_sort(const float * router_logits, const float * router_bias, int32_t * route_ids, float * route_w, int32_t * expert_offsets, int32_t * sorted_tok, int32_t * sorted_slot, float * sorted_w, int32_t batch_size, void * stream);
void ifn_launch_prefill_moe_gate_up_h(const ifn_cuda_c_expert_bank * w13, const void * x_h, const int32_t * expert_offsets, const int32_t * sorted_tok, const int32_t * sorted_slot, void * out_act_h, int32_t batch_size, void * stream);
void ifn_launch_prefill_moe_down_h(const ifn_cuda_c_expert_bank * w2, const void * act_h, const int32_t * expert_offsets, const int32_t * sorted_tok, const int32_t * sorted_slot, const float * sorted_w, float * out_slots, int32_t batch_size, void * stream);
#ifdef __cplusplus
}
#endif
#endif
