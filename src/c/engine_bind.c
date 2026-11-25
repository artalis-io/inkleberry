/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "engine_internal.h"
#include <stdio.h>
#include <string.h>

static void *address(sq_gpu_weights *w, uint64_t offset) {
    return offset == SQ_WEIGHT_NO_OFFSET ? NULL : (unsigned char *)w->arena + (size_t)offset;
}

static sq_status bind(sq_gpu_weights *w, const char *label, void *out, uint32_t type) {
    for (size_t i = 0; i < w->view_count; i++) {
        const sq_weight_view *v = &w->views[i];
        if (strcmp(label, v->label))
            continue;
        if (v->type != type)
            return sq_error(SQ_FORMAT, "AR weight type mismatch");
        if (type == SQ_WEIGHT_RAW) {
            void *p = address(w, v->data);
            memcpy(out, &p, sizeof p);
        } else if (type == SQ_WEIGHT_LINEAR) {
            CLinearWeight x = {0};
            x.kind = (uint8_t)v->precision;
            x.rows = (int32_t)v->rows;
            x.cols = (int32_t)v->cols;
            x.data = address(w, v->data);
            x.scales = address(w, v->scales);
            x.row_scales = address(w, v->row_scales);
            x.global_scale = v->global_scale;
            memcpy(out, &x, sizeof x);
        } else {
            CExpertBank x = {0};
            x.is_nvfp4 = v->precision == 2;
            x.n_experts = (int32_t)v->experts;
            x.rows = (int32_t)v->rows;
            x.cols = (int32_t)v->cols;
            x.data = address(w, v->data);
            x.scales = address(w, v->scales);
            x.expert_scale = address(w, v->expert_scale);
            memcpy(out, &x, sizeof x);
        }
        return SQ_OK;
    }
    return sq_error(SQ_FORMAT, "missing AR weight view");
}

sq_status sq_ar_bind(sq_gpu_weights *w, sq_ar_model *m) {
    memset(m, 0, sizeof *m);
#define B(label, member, type)                                                                     \
    do {                                                                                           \
        sq_status rc = bind(w, label, &(member), type);                                            \
        if (rc != SQ_OK)                                                                           \
            return rc;                                                                             \
    } while (0)
    B("embed", m->embed, SQ_WEIGHT_RAW);
    B("embed_norm", m->embed_norm, SQ_WEIGHT_RAW);
    B("final_norm", m->final_norm, SQ_WEIGHT_RAW);
    B("output_query", m->output_query, SQ_WEIGHT_RAW);
    B("cross_pass_queries", m->cross_pass_queries, SQ_WEIGHT_RAW);
    B("cross_pass_kv_queries", m->cross_pass_kv_queries, SQ_WEIGHT_RAW);
    B("unembed", m->unembed, SQ_WEIGHT_LINEAR);
    char label[128];
    for (unsigned i = 0; i < 23; i++) {
        (void)snprintf(label, sizeof label, "attn_queries.%u", i);
        B(label, m->attn_queries[i], SQ_WEIGHT_RAW);
    }
    for (unsigned i = 0; i < 24; i++) {
        (void)snprintf(label, sizeof label, "mlp_queries.%u", i);
        B(label, m->mlp_queries[i], SQ_WEIGHT_RAW);
    }
    for (unsigned i = 0; i < 24; i++) {
        if (i % 6 != 5) {
            (void)snprintf(label, sizeof label, "layers.%u.attn_norm", i);
            B(label, m->layers[i].attn_norm, SQ_WEIGHT_RAW);
        }
        if (i % 6 == 5) {
            (void)snprintf(label, sizeof label, "layers.%u.q_in_norm", i);
            B(label, m->layers[i].q_in_norm, SQ_WEIGHT_RAW);
        }
        if (i % 6 == 5) {
            (void)snprintf(label, sizeof label, "layers.%u.k_in_norm", i);
            B(label, m->layers[i].k_in_norm, SQ_WEIGHT_RAW);
        }
        if (i % 6 == 5) {
            (void)snprintf(label, sizeof label, "layers.%u.v_in_norm", i);
            B(label, m->layers[i].v_in_norm, SQ_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.mlp_norm", i);
            B(label, m->layers[i].mlp_norm, SQ_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w_qrgs", i);
            B(label, m->layers[i].w_qrgs, SQ_WEIGHT_LINEAR);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w_kv", i);
            B(label, m->layers[i].w_kv, SQ_WEIGHT_LINEAR);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w_k", i);
            B(label, m->layers[i].w_k, SQ_WEIGHT_LINEAR);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w_v", i);
            B(label, m->layers[i].w_v, SQ_WEIGHT_LINEAR);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w_o", i);
            B(label, m->layers[i].w_o, SQ_WEIGHT_LINEAR);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.k_sconv_w", i);
            B(label, m->layers[i].k_sconv_w, SQ_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.v_sconv_w", i);
            B(label, m->layers[i].v_sconv_w, SQ_WEIGHT_RAW);
        }
        if (i % 6 == 5) {
            (void)snprintf(label, sizeof label, "layers.%u.k_sconv_proj", i);
            B(label, m->layers[i].k_sconv_proj, SQ_WEIGHT_RAW);
        }
        if (i % 6 == 5) {
            (void)snprintf(label, sizeof label, "layers.%u.v_sconv_proj", i);
            B(label, m->layers[i].v_sconv_proj, SQ_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.attn_sconv_w", i);
            B(label, m->layers[i].attn_sconv_w, SQ_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.mlp_sconv_w", i);
            B(label, m->layers[i].mlp_sconv_w, SQ_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.q_clip_scale", i);
            B(label, m->layers[i].q_clip_scale, SQ_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.rel_clip_scale", i);
            B(label, m->layers[i].rel_clip_scale, SQ_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.rel_logits_proj", i);
            B(label, m->layers[i].rel_logits_proj, SQ_WEIGHT_RAW);
        }
        if (w->config.mode != 2) {
            (void)snprintf(label, sizeof label, "layers.%u.router_w", i);
            B(label, m->layers[i].router_w, SQ_WEIGHT_RAW);
        }
        if (w->config.mode == 2) {
            (void)snprintf(label, sizeof label, "layers.%u.router_nvfp4", i);
            B(label, m->layers[i].router_nvfp4, SQ_WEIGHT_LINEAR);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.router_bias", i);
            B(label, m->layers[i].router_bias, SQ_WEIGHT_RAW);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w13", i);
            B(label, m->layers[i].w13, SQ_WEIGHT_EXPERT);
        }
        if (true) {
            (void)snprintf(label, sizeof label, "layers.%u.w2", i);
            B(label, m->layers[i].w2, SQ_WEIGHT_EXPERT);
        }
        if (i >= 6 && i < 18)
            for (unsigned j = 0; j < 4; j++) {
                if (w->config.mode == 2) {
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.down_nvfp4", i, j);
                    B(label, m->layers[i].ctrl[j].down_nvfp4, SQ_WEIGHT_LINEAR);
                } else {
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.down_w", i, j);
                    B(label, m->layers[i].ctrl[j].down_w, SQ_WEIGHT_RAW);
                }
                if (w->config.mode == 2) {
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.forget_nvfp4", i, j);
                    B(label, m->layers[i].ctrl[j].forget_nvfp4, SQ_WEIGHT_LINEAR);
                } else {
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.forget_w", i, j);
                    B(label, m->layers[i].ctrl[j].forget_w, SQ_WEIGHT_RAW);
                }
                if (w->config.mode == 2) {
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.gain_nvfp4", i, j);
                    B(label, m->layers[i].ctrl[j].gain_nvfp4, SQ_WEIGHT_LINEAR);
                } else {
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.gain_w", i, j);
                    B(label, m->layers[i].ctrl[j].gain_w, SQ_WEIGHT_RAW);
                }
            }
    }
#undef B
    return SQ_OK;
}

void sq_ar_schedule(sq_visit out[48]) {
    unsigned i = 0;
    for (unsigned l = 0; l < 6; l++)
        out[i++] = (sq_visit){l, 0, false, l % 6 == 5};
    for (unsigned p = 0; p < 3; p++)
        for (unsigned l = 6; l < 18; l++)
            out[i++] = (sq_visit){l, p, true, l % 6 == 5};
    for (unsigned l = 18; l < 24; l++)
        out[i++] = (sq_visit){l, 0, false, l % 6 == 5};
}
