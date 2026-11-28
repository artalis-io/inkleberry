/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "engine_internal.h"
#include <stdio.h>
#include <string.h>

static void *ifn_engine_bind_local_address(ifn_gpu_weights *w, uint64_t offset) {
    return offset == IFN_WEIGHT_NO_OFFSET ? NULL : (unsigned char *)w->arena + (size_t)offset;
}

static ifn_status ifn_engine_bind_local_bind(ifn_gpu_weights *w, const char *label, void *out,
                                             uint32_t type) {
    for (size_t i = 0; i < w->view_count; i++) {
        const ifn_weight_view *v = &w->views[i];
        if (strcmp(label, v->label))
            continue;
        if (v->type != type)
            return ifn_error(IFN_FORMAT, "AR weight type mismatch");
        if (type == IFN_WEIGHT_RAW) {
            void *p = ifn_engine_bind_local_address(w, v->data);
            memcpy(out, &p, sizeof p);
        } else if (type == IFN_WEIGHT_LINEAR) {
            ifn_cuda_c_linear_weight x = {0};
            x.kind = (uint8_t)v->precision;
            x.rows = (int32_t)v->rows;
            x.cols = (int32_t)v->cols;
            x.data = ifn_engine_bind_local_address(w, v->data);
            x.scales = ifn_engine_bind_local_address(w, v->scales);
            x.row_scales = ifn_engine_bind_local_address(w, v->row_scales);
            x.global_scale = v->global_scale;
            memcpy(out, &x, sizeof x);
        } else {
            ifn_cuda_c_expert_bank x = {0};
            x.is_nvfp4 = v->precision == 2;
            x.n_experts = (int32_t)v->experts;
            x.rows = (int32_t)v->rows;
            x.cols = (int32_t)v->cols;
            x.data = ifn_engine_bind_local_address(w, v->data);
            x.scales = ifn_engine_bind_local_address(w, v->scales);
            x.expert_scale = ifn_engine_bind_local_address(w, v->expert_scale);
            memcpy(out, &x, sizeof x);
        }
        return IFN_OK;
    }
    return ifn_error(IFN_FORMAT, "missing AR weight view");
}

ifn_status ifn_ar_bind(ifn_gpu_weights *w, ifn_ar_model *m) {
    memset(m, 0, sizeof *m);
#define IFN_ENGINE_BIND_LOCAL_B(label, member, type)                                               \
    do {                                                                                           \
        ifn_status rc = ifn_engine_bind_local_bind(w, label, &(member), type);                     \
        if (rc != IFN_OK)                                                                          \
            return rc;                                                                             \
    } while (0)
    IFN_ENGINE_BIND_LOCAL_B("embed", m->embed, IFN_WEIGHT_RAW);
    IFN_ENGINE_BIND_LOCAL_B("embed_norm", m->embed_norm, IFN_WEIGHT_RAW);
    IFN_ENGINE_BIND_LOCAL_B("final_norm", m->final_norm, IFN_WEIGHT_RAW);
    IFN_ENGINE_BIND_LOCAL_B("output_query", m->output_query, IFN_WEIGHT_RAW);
    IFN_ENGINE_BIND_LOCAL_B("cross_pass_queries", m->cross_pass_queries, IFN_WEIGHT_RAW);
    IFN_ENGINE_BIND_LOCAL_B("cross_pass_kv_queries", m->cross_pass_kv_queries, IFN_WEIGHT_RAW);
    IFN_ENGINE_BIND_LOCAL_B("unembed", m->unembed, IFN_WEIGHT_LINEAR);
    char label[128];
    for (unsigned i = 0; i < IFN_INKLING_LAYERS - 1; i++) {
        (void)snprintf(label, sizeof label, "attn_queries.%u", i);
        IFN_ENGINE_BIND_LOCAL_B(label, m->attn_queries[i], IFN_WEIGHT_RAW);
    }
    for (unsigned i = 0; i < IFN_INKLING_LAYERS; i++) {
        (void)snprintf(label, sizeof label, "mlp_queries.%u", i);
        IFN_ENGINE_BIND_LOCAL_B(label, m->mlp_queries[i], IFN_WEIGHT_RAW);
    }
    for (unsigned i = 0; i < IFN_INKLING_LAYERS; i++) {
        if ((i + 1) % IFN_INKLING_BLOCK_LAYERS != 0) {
            (void)snprintf(label, sizeof label, "layers.%u.attn_norm", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].attn_norm, IFN_WEIGHT_RAW);
        }
        if ((i + 1) % IFN_INKLING_BLOCK_LAYERS == 0) {
            (void)snprintf(label, sizeof label, "layers.%u.q_in_norm", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].q_in_norm, IFN_WEIGHT_RAW);
        }
        if ((i + 1) % IFN_INKLING_BLOCK_LAYERS == 0) {
            (void)snprintf(label, sizeof label, "layers.%u.k_in_norm", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].k_in_norm, IFN_WEIGHT_RAW);
        }
        if ((i + 1) % IFN_INKLING_BLOCK_LAYERS == 0) {
            (void)snprintf(label, sizeof label, "layers.%u.v_in_norm", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].v_in_norm, IFN_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.mlp_norm", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].mlp_norm, IFN_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w_qrgs", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].w_qrgs, IFN_WEIGHT_LINEAR);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w_kv", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].w_kv, IFN_WEIGHT_LINEAR);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w_k", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].w_k, IFN_WEIGHT_LINEAR);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w_v", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].w_v, IFN_WEIGHT_LINEAR);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w_o", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].w_o, IFN_WEIGHT_LINEAR);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.k_sconv_w", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].k_sconv_w, IFN_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.v_sconv_w", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].v_sconv_w, IFN_WEIGHT_RAW);
        }
        if ((i + 1) % IFN_INKLING_BLOCK_LAYERS == 0) {
            (void)snprintf(label, sizeof label, "layers.%u.k_sconv_proj", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].k_sconv_proj, IFN_WEIGHT_RAW);
        }
        if ((i + 1) % IFN_INKLING_BLOCK_LAYERS == 0) {
            (void)snprintf(label, sizeof label, "layers.%u.v_sconv_proj", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].v_sconv_proj, IFN_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.attn_sconv_w", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].attn_sconv_w, IFN_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.mlp_sconv_w", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].mlp_sconv_w, IFN_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.q_clip_scale", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].q_clip_scale, IFN_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.rel_clip_scale", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].rel_clip_scale, IFN_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.rel_logits_proj", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].rel_logits_proj, IFN_WEIGHT_RAW);
        }
        if (w->config.mode != 2) {
            (void)snprintf(label, sizeof label, "layers.%u.router_w", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].router_w, IFN_WEIGHT_RAW);
        }
        if (w->config.mode == 2) {
            (void)snprintf(label, sizeof label, "layers.%u.router_nvfp4", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].router_nvfp4, IFN_WEIGHT_LINEAR);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.router_bias", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].router_bias, IFN_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w13", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].w13, IFN_WEIGHT_EXPERT);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w2", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].w2, IFN_WEIGHT_EXPERT);
        }
        if (i >= IFN_INKLING_LOOP_START && i < IFN_INKLING_LOOP_END)
            for (unsigned j = 0; j < 4; j++) {
                if (w->config.mode == 2) {
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.down_nvfp4", i, j);
                    IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].ctrl[j].down_nvfp4,
                                            IFN_WEIGHT_LINEAR);
                } else {
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.down_w", i, j);
                    IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].ctrl[j].down_w, IFN_WEIGHT_RAW);
                }
                if (w->config.mode == 2) {
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.forget_nvfp4", i, j);
                    IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].ctrl[j].forget_nvfp4,
                                            IFN_WEIGHT_LINEAR);
                } else {
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.forget_w", i, j);
                    IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].ctrl[j].forget_w, IFN_WEIGHT_RAW);
                }
                if (w->config.mode == 2) {
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.gain_nvfp4", i, j);
                    IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].ctrl[j].gain_nvfp4,
                                            IFN_WEIGHT_LINEAR);
                } else {
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.gain_w", i, j);
                    IFN_ENGINE_BIND_LOCAL_B(label, m->layers[i].ctrl[j].gain_w, IFN_WEIGHT_RAW);
                }
            }
    }
    if (w->config.with_mtp) {
        m->dspark.loaded = 1;
        m->dspark.draft_vocab_size = 200058;
        IFN_ENGINE_BIND_LOCAL_B("dspark.fuse_alpha", m->dspark.fuse_alpha, IFN_WEIGHT_RAW);
        IFN_ENGINE_BIND_LOCAL_B("dspark.mask_embed", m->dspark.mask_embed, IFN_WEIGHT_RAW);
        IFN_ENGINE_BIND_LOCAL_B("dspark.norm_w", m->dspark.norm_w, IFN_WEIGHT_RAW);
        IFN_ENGINE_BIND_LOCAL_B("dspark.markov_dflare_norm", m->dspark.markov_dflare_norm,
                                IFN_WEIGHT_RAW);
        IFN_ENGINE_BIND_LOCAL_B("dspark.markov_w1", m->dspark.markov_w1, IFN_WEIGHT_RAW);
        IFN_ENGINE_BIND_LOCAL_B("dspark.rnn_w_sp", m->dspark.rnn_w_sp, IFN_WEIGHT_RAW);
        IFN_ENGINE_BIND_LOCAL_B("dspark.rnn_w_h", m->dspark.rnn_w_h, IFN_WEIGHT_RAW);
        IFN_ENGINE_BIND_LOCAL_B("dspark.rnn_w_f", m->dspark.rnn_w_f, IFN_WEIGHT_RAW);
        IFN_ENGINE_BIND_LOCAL_B("dspark.rnn_bias", m->dspark.rnn_bias, IFN_WEIGHT_RAW);
        IFN_ENGINE_BIND_LOCAL_B("dspark.hidden_out_w", m->dspark.hidden_out_w, IFN_WEIGHT_RAW);
        IFN_ENGINE_BIND_LOCAL_B("dspark.hidden_out_b", m->dspark.hidden_out_b, IFN_WEIGHT_RAW);
        IFN_ENGINE_BIND_LOCAL_B("dspark.unembed_draft", m->dspark.unembed_draft, IFN_WEIGHT_LINEAR);
        IFN_ENGINE_BIND_LOCAL_B("dspark.fold_w", m->dspark.fold_w, IFN_WEIGHT_LINEAR);
        IFN_ENGINE_BIND_LOCAL_B("dspark.fold_b", m->dspark.fold_b, IFN_WEIGHT_RAW);
        for (unsigned i = 0; i < 3; i++) {
            m->dspark.layers[i].is_local = i != 1;
            (void)snprintf(label, sizeof label, "dspark.layers.%u.input_norm", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->dspark.layers[i].input_norm, IFN_WEIGHT_RAW);
            (void)snprintf(label, sizeof label, "dspark.layers.%u.post_norm", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->dspark.layers[i].post_norm, IFN_WEIGHT_RAW);
            (void)snprintf(label, sizeof label, "dspark.layers.%u.dflare_norm", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->dspark.layers[i].dflare_norm, IFN_WEIGHT_RAW);
            (void)snprintf(label, sizeof label, "dspark.layers.%u.q_norm", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->dspark.layers[i].q_norm, IFN_WEIGHT_RAW);
            (void)snprintf(label, sizeof label, "dspark.layers.%u.k_norm", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->dspark.layers[i].k_norm, IFN_WEIGHT_RAW);
            (void)snprintf(label, sizeof label, "dspark.layers.%u.k_norm_ctx", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->dspark.layers[i].k_norm_ctx, IFN_WEIGHT_RAW);
            (void)snprintf(label, sizeof label, "dspark.layers.%u.w_qkv", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->dspark.layers[i].w_qkv, IFN_WEIGHT_LINEAR);
            (void)snprintf(label, sizeof label, "dspark.layers.%u.w_o", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->dspark.layers[i].w_o, IFN_WEIGHT_LINEAR);
            (void)snprintf(label, sizeof label, "dspark.layers.%u.w_gu", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->dspark.layers[i].w_gu, IFN_WEIGHT_LINEAR);
            (void)snprintf(label, sizeof label, "dspark.layers.%u.w_down", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->dspark.layers[i].w_down, IFN_WEIGHT_LINEAR);
            (void)snprintf(label, sizeof label, "dspark.layers.%u.w_kv_ctx", i);
            IFN_ENGINE_BIND_LOCAL_B(label, m->dspark.layers[i].w_kv_ctx, IFN_WEIGHT_LINEAR);
        }
    }

#undef IFN_ENGINE_BIND_LOCAL_B
    return IFN_OK;
}

void ifn_ar_schedule(ifn_visit out[IFN_INKLING_VISITS]) {
    unsigned i = 0;
    for (unsigned l = 0; l < IFN_INKLING_LOOP_START; l++)
        out[i++] = (ifn_visit){l, 0, false, (l + 1) % IFN_INKLING_BLOCK_LAYERS == 0};
    for (unsigned p = 0; p < IFN_INKLING_LOOP_PASSES; p++)
        for (unsigned l = IFN_INKLING_LOOP_START; l < IFN_INKLING_LOOP_END; l++)
            out[i++] = (ifn_visit){l, p, true, (l + 1) % IFN_INKLING_BLOCK_LAYERS == 0};
    for (unsigned l = IFN_INKLING_LOOP_END; l < IFN_INKLING_LAYERS; l++)
        out[i++] = (ifn_visit){l, 0, false, (l + 1) % IFN_INKLING_BLOCK_LAYERS == 0};
}
