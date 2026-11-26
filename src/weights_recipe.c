/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Two-pass translation of pinned src/weights.zig: identical allocation order,
 * concatenation, slicing, quantization choices and full-vocabulary MTP folding.
 * Checked arena arithmetic and recoverable failures replace assertions. */
#include "weights_internal.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

enum { H = 2304, Q = 3072, KV = 1024, E = 128, I = 1152, V = 201024, VU = 200058, M = 9216 };

static void *pointer(ifn_gpu_weights *w, uint64_t offset) {
    return w->execute && offset != UINT64_MAX ? (unsigned char *)w->arena + (size_t)offset : NULL;
}

static ifn_weight_view *view(ifn_gpu_weights *w, const char *label, uint32_t type) {
    if (w->status != IFN_OK)
        return NULL;
    if (w->view_count == IFN_WEIGHT_MAX_VIEWS || strlen(label) >= sizeof w->views[0].label) {
        ifn_weight_fail(w, IFN_LIMIT, "weight view limit");
        return NULL;
    }
    ifn_weight_view *v = &w->views[w->view_count++];
    memset(v, 0, sizeof *v);
    memcpy(v->label, label, strlen(label) + 1);
    v->type = type;
    v->data = v->scales = v->row_scales = v->expert_scale = UINT64_MAX;
    v->global_scale = 1.0f;
    return v;
}

static uint64_t block(ifn_gpu_weights *w, const char *label, const char *suffix, size_t bytes) {
    if (w->status != IFN_OK)
        return UINT64_MAX;
    size_t aligned, end;
    if (!bytes || !ifn_size_align(bytes, 256, &aligned) ||
        !ifn_size_add(w->offset, aligned, &end) || (w->execute && end > w->capacity) ||
        w->span_count == IFN_WEIGHT_MAX_SPANS) {
        ifn_weight_fail(w, IFN_LIMIT, "weight arena overflow");
        return UINT64_MAX;
    }
    ifn_weight_span *s = &w->spans[w->span_count];
    int n = snprintf(s->label, sizeof s->label, "%s%s", label, suffix);
    if (n < 0 || (size_t)n >= sizeof s->label) {
        ifn_weight_fail(w, IFN_LIMIT, "weight span label limit");
        return UINT64_MAX;
    }
    s->offset = w->offset;
    s->bytes = bytes;
    w->offset = end;
    w->span_count++;
    return s->offset;
}

static void copy_tensor(ifn_gpu_weights *w, uint32_t source, const char *name, void *dst,
                        size_t bytes, bool prefix) {
    if (!w->execute || w->status != IFN_OK)
        return;
    const void *data = NULL;
    size_t n = 0;
    if (ifn_inkling_lookup(w->checkpoint, source, name, &data, &n) != IFN_OK || n < bytes ||
        (!prefix && n != bytes)) {
        ifn_weight_fail(w, IFN_FORMAT, "weight source size mismatch");
        return;
    }
    ifn_weight_backend_check(w, w->backend.copy(w->backend.context, dst, data, bytes, 0, w->stream),
                             "upload tensor");
}

static void raw(ifn_gpu_weights *w, const char *label, uint32_t source, const char *name,
                size_t bytes) {
    ifn_weight_view *v = view(w, label, IFN_WEIGHT_RAW);
    if (!v)
        return;
    v->data = block(w, label, "", bytes);
    if (w->status == IFN_OK)
        copy_tensor(w, source, name, pointer(w, v->data), bytes, false);
}

static void stage(ifn_gpu_weights *w, uint32_t source, const char *name, size_t offset,
                  size_t bytes, bool prefix) {
    if (w->status != IFN_OK || !w->execute)
        return;
    if (offset > w->plan.staging_bytes || bytes > w->plan.staging_bytes - offset) {
        ifn_weight_fail(w, IFN_LIMIT, "weight staging overflow");
        return;
    }
    copy_tensor(w, source, name, (unsigned char *)w->stage + offset, bytes, prefix);
}

static ifn_weight_view *linear(ifn_gpu_weights *w, const char *label, unsigned rows, unsigned cols,
                               unsigned kind, const void *input) {
    ifn_weight_view *v = view(w, label, IFN_WEIGHT_LINEAR);
    if (!v)
        return NULL;
    size_t elements;
    if (!rows || !cols || cols % 16 || kind > 2 || !ifn_size_mul(rows, cols, &elements)) {
        ifn_weight_fail(w, IFN_LIMIT, "invalid linear geometry");
        return NULL;
    }
    v->precision = kind;
    v->rows = rows;
    v->cols = cols;
    v->data = block(w, label, ".data",
                    kind == 0   ? elements * 2
                    : kind == 1 ? elements
                                : elements / 2);
    if (kind == 1)
        v->row_scales = block(w, label, ".row_scales", (size_t)rows * 4);
    if (kind == 2)
        v->scales = block(w, label, ".scales", elements / 16);
    if (w->execute && w->status == IFN_OK) {
        if (kind == 0)
            ifn_weight_backend_check(w,
                                     w->backend.copy(w->backend.context, pointer(w, v->data), input,
                                                     elements * 2, 2, w->stream),
                                     "copy BF16 matrix");
        else
            ifn_weight_backend_check(
                w,
                w->backend.pack(w->backend.context, input, pointer(w, v->data),
                                pointer(w, kind == 1 ? v->row_scales : v->scales), w->amax,
                                (int)rows, (int)cols, (int)kind, &v->global_scale, w->stream),
                "quantize matrix");
    }
    return v;
}

static void staged_linear(ifn_gpu_weights *w, const char *label, uint32_t source, const char *name,
                          unsigned rows, unsigned cols, unsigned kind, bool prefix) {
    stage(w, source, name, 0, (size_t)rows * cols * 2, prefix);
    (void)linear(w, label, rows, cols, kind, w->stage);
}

static void concat(ifn_gpu_weights *w, const char *label, uint32_t source, const char *prefix,
                   const char *const *suffix, const unsigned *rows, size_t count, unsigned cols,
                   unsigned kind) {
    size_t offset = 0, total = 0;
    char name[160];
    for (size_t i = 0; i < count && w->status == IFN_OK; i++) {
        int n = snprintf(name, sizeof name, "%s%s", prefix, suffix[i]);
        if (n < 0 || (size_t)n >= sizeof name) {
            ifn_weight_fail(w, IFN_LIMIT, "tensor label limit");
            return;
        }
        size_t bytes = (size_t)rows[i] * cols * 2;
        stage(w, source, name, offset, bytes, false);
        offset += bytes;
        total += rows[i];
    }
    (void)linear(w, label, (unsigned)total, cols, kind, w->stage);
}

static float f32le(const unsigned char *p) {
    uint32_t bits =
        (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static void expert(ifn_gpu_weights *w, const char *label, const char *prefix, const char *bank,
                   unsigned layer, unsigned cols) {
    ifn_weight_view *v = view(w, label, IFN_WEIGHT_EXPERT);
    if (!v)
        return;
    v->rows = H;
    v->cols = cols;
    v->experts = E;
    v->precision = layer < 20 || w->config.mode == 2 ? 2 : 1;
    size_t elements = (size_t)E * H * cols;
    char wn[160], sn[192], an[192], dn[192];
    (void)snprintf(wn, sizeof wn, "%smlp.experts.%s_weight", prefix, bank);
    (void)snprintf(sn, sizeof sn, "%s_scale", wn);
    if (layer < 20) {
        v->data = block(w, label, ".data", elements / 2);
        v->scales = block(w, label, ".scales", elements / 16);
        copy_tensor(w, 0, wn, pointer(w, v->data), elements / 2, false);
        copy_tensor(w, 0, sn, pointer(w, v->scales), elements / 16, false);
        v->expert_scale = block(w, label, ".expert_scale", E * 4);
        if (w->execute && w->status == IFN_OK) {
            (void)snprintf(an, sizeof an, "%s_amax", wn);
            (void)snprintf(dn, sizeof dn, "%s_outer_scale_denominator", wn);
            const void *ap = NULL, *dp = NULL;
            size_t as = 0, ds = 0;
            if (ifn_inkling_lookup(w->checkpoint, 0, an, &ap, &as) != IFN_OK ||
                ifn_inkling_lookup(w->checkpoint, 0, dn, &dp, &ds) != IFN_OK || as != E * 4 ||
                ds != 4) {
                ifn_weight_fail(w, IFN_FORMAT, "invalid expert scale metadata");
                return;
            }
            float denom = f32le(dp), scales[E];
            if (!isfinite(denom) || denom <= 0 || !isfinite(1.0f / denom)) {
                ifn_weight_fail(w, IFN_FORMAT, "invalid expert scale denominator");
                return;
            }
            float inverse = 1.0f / denom;
            for (size_t i = 0; i < E; i++) {
                float amax = f32le((const unsigned char *)ap + i * 4);
                scales[i] = amax * inverse;
                if (!isfinite(amax) || amax < 0 || !isfinite(scales[i])) {
                    ifn_weight_fail(w, IFN_FORMAT, "invalid expert amax");
                    return;
                }
            }
            ifn_weight_backend_check(w,
                                     w->backend.copy(w->backend.context,
                                                     pointer(w, v->expert_scale), scales,
                                                     sizeof scales, 0, w->stream),
                                     "upload expert scales");
        }
    } else if (w->config.mode != 2) {
        v->data = block(w, label, ".data", elements);
        v->scales = block(w, label, ".scales", elements / 32);
        copy_tensor(w, 0, wn, pointer(w, v->data), elements, false);
        copy_tensor(w, 0, sn, pointer(w, v->scales), elements / 32, false);
    } else {
        stage(w, 0, wn, 0, elements, false);
        stage(w, 0, sn, elements, elements / 32, false);
        v->expert_scale = block(w, label, ".expert_scale", E * 4);
        v->data = block(w, label, ".data", elements / 2);
        v->scales = block(w, label, ".scales", elements / 16);
        if (w->execute && w->status == IFN_OK)
            ifn_weight_backend_check(w,
                                     w->backend.expert(w->backend.context, w->stage,
                                                       (unsigned char *)w->stage + elements,
                                                       pointer(w, v->data), pointer(w, v->scales),
                                                       pointer(w, v->expert_scale), E, H, (int)cols,
                                                       w->stream),
                                     "convert MXFP8 experts");
    }
}

static void fuse(ifn_gpu_weights *w) {
    ifn_weight_view *v = view(w, "dspark.fuse_alpha", IFN_WEIGHT_RAW);
    if (!v)
        return;
    v->data = block(w, v->label, "", 64);
    if (!w->execute || w->status != IFN_OK)
        return;
    const void *data = NULL;
    size_t bytes = 0;
    if (ifn_inkling_lookup(w->checkpoint, 1, "fuse_weight", &data, &bytes) != IFN_OK ||
        bytes != 32) {
        ifn_weight_fail(w, IFN_FORMAT, "invalid fusion weights");
        return;
    }
    float alpha[16];
    const unsigned char *raw_bytes = data;
    for (size_t row = 0; row < 4; row++) {
        double values[4], maximum = -1e30, sum = 0;
        for (size_t col = 0; col < 4; col++) {
            size_t i = (row * 4 + col) * 2;
            uint32_t bits = ((uint32_t)raw_bytes[i] | (uint32_t)raw_bytes[i + 1] << 8) << 16;
            float f;
            memcpy(&f, &bits, 4);
            if (!isfinite(f)) {
                ifn_weight_fail(w, IFN_FORMAT, "non-finite fusion weight");
                return;
            }
            values[col] = f;
            if (values[col] > maximum)
                maximum = values[col];
        }
        for (size_t col = 0; col < 4; col++) {
            values[col] = exp(values[col] - maximum);
            sum += values[col];
        }
        if (!isfinite(sum) || sum <= 0) {
            ifn_weight_fail(w, IFN_FORMAT, "invalid fusion normalization");
            return;
        }
        for (size_t col = 0; col < 4; col++)
            alpha[row * 4 + col] = (float)(values[col] / sum);
    }
    ifn_weight_backend_check(
        w,
        w->backend.copy(w->backend.context, pointer(w, v->data), alpha, sizeof alpha, 0, w->stream),
        "upload fusion alpha");
}

static ifn_weight_view *named(ifn_gpu_weights *w, const char *name) {
    for (size_t i = 0; i < w->view_count; i++)
        if (!strcmp(w->views[i].label, name))
            return &w->views[i];
    ifn_weight_fail(w, IFN_FORMAT, "missing internal weight view");
    return NULL;
}

static void alias(ifn_gpu_weights *w, const char *label, const ifn_weight_view *source) {
    if (!source || w->status != IFN_OK)
        return;
    ifn_weight_view *v = view(w, label, source->type);
    if (!v)
        return;
    char name[128];
    memcpy(name, v->label, sizeof name);
    *v = *source;
    memcpy(v->label, name, sizeof name);
}

void ifn_weights_recipe(ifn_gpu_weights *w) {
    w->offset = w->span_count = w->view_count = 0;
    unsigned kind = w->config.mode, mtp_kind = kind == 0 || w->config.mtp_bf16 ? 0 : 2;
    raw(w, "embed", 0, "model.llm.embed.weight", (size_t)V * H * 2);
    raw(w, "embed_norm", 0, "model.llm.embed_norm.weight", H * 2);
    raw(w, "final_norm", 0, "model.llm.norm.weight", H * 2);
    staged_linear(w, "unembed", 0, "model.llm.unembed.weight", VU, H, kind, true);
    char name[160], label[128], prefix[96];
    for (unsigned i = 0; i < 23; i++) {
        (void)snprintf(name, sizeof name, "model.llm.attn_res.attn_queries.%u", i);
        (void)snprintf(label, sizeof label, "attn_queries.%u", i);
        raw(w, label, 0, name, ((i + 2) % 6 == 0 ? 3u : 1u) * H * 4);
    }
    for (unsigned i = 0; i < 24; i++) {
        (void)snprintf(name, sizeof name, "model.llm.attn_res.mlp_queries.%u", i);
        (void)snprintf(label, sizeof label, "mlp_queries.%u", i);
        raw(w, label, 0, name, H * 4);
    }
    raw(w, "output_query", 0, "model.llm.attn_res.output_query", H * 4);
    raw(w, "cross_pass_queries", 0, "model.llm.middle_cycle_cross_pass_queries", 24 * H * 4);
    raw(w, "cross_pass_kv_queries", 0, "model.llm.middle_cycle_cross_pass_kv_queries", 24 * H * 4);
    for (unsigned i = 0; i < 24 && w->status == IFN_OK; i++) {
        (void)snprintf(prefix, sizeof prefix, "model.llm.layers.%u.", i);
#define RAW(field, suffix, bytes)                                                                  \
    do {                                                                                           \
        (void)snprintf(label, sizeof label, "layers.%u.%s", i, field);                             \
        (void)snprintf(name, sizeof name, "%s%s", prefix, suffix);                                 \
        raw(w, label, 0, name, bytes);                                                             \
    } while (0)
#define LINEAR(field, suffix, rows, cols, k)                                                       \
    do {                                                                                           \
        (void)snprintf(label, sizeof label, "layers.%u.%s", i, field);                             \
        (void)snprintf(name, sizeof name, "%s%s", prefix, suffix);                                 \
        staged_linear(w, label, 0, name, rows, cols, k, false);                                    \
    } while (0)
        bool global = (i + 1) % 6 == 0;
        if (global) {
            RAW("q_in_norm", "q_input_norm.weight", H * 2);
            RAW("k_in_norm", "k_input_norm.weight", H * 2);
            RAW("v_in_norm", "v_input_norm.weight", H * 2);
        } else
            RAW("attn_norm", "attn_norm.weight", H * 2);
        RAW("mlp_norm", "mlp_norm.weight", H * 2);
        const char *qrgs[] = {"attn.wq_du.weight", "attn.wr_du.weight", "attn.gate_attn.weight",
                              "attn.attn_sink_proj.weight"};
        const unsigned qrgs_rows[] = {Q, 384, Q, 24};
        (void)snprintf(label, sizeof label, "layers.%u.w_qrgs", i);
        concat(w, label, 0, prefix, qrgs, qrgs_rows, 4, H, kind);
        const char *kv[] = {"attn.wk_dv.weight", "attn.wv_dv.weight"};
        const unsigned kv_rows[] = {KV, KV};
        (void)snprintf(label, sizeof label, "layers.%u.w_kv", i);
        concat(w, label, 0, prefix, kv, kv_rows, 2, H, kind);
        ifn_weight_view *v = named(w, label);
        (void)snprintf(label, sizeof label, "layers.%u.w_k", i);
        alias(w, label, v);
        ifn_weight_view *k = named(w, label);
        if (k)
            k->rows = KV;
        (void)snprintf(label, sizeof label, "layers.%u.w_v", i);
        alias(w, label, v);
        ifn_weight_view *vv = named(w, label);
        if (vv) {
            vv->rows = KV;
            vv->data += (size_t)KV * H * (kind == 0 ? 2u : 1u) / (kind == 2 ? 2u : 1u);
            if (kind == 1)
                vv->row_scales += KV * 4;
            if (kind == 2)
                vv->scales += (size_t)KV * H / 16;
        }
        LINEAR("w_o", "attn.wo_ud.weight", H, Q, kind);
        if (global) {
            RAW("k_sconv_w", "attn.k_sconv.static_w", 4 * KV * 2);
            RAW("v_sconv_w", "attn.v_sconv.static_w", 4 * KV * 2);
            RAW("k_sconv_proj", "attn.k_sconv.weight_proj", 32 * H * 2);
            RAW("v_sconv_proj", "attn.v_sconv.weight_proj", 32 * H * 2);
        } else {
            RAW("k_sconv_w", "attn.k_sconv.weight", KV * 4 * 2);
            RAW("v_sconv_w", "attn.v_sconv.weight", KV * 4 * 2);
        }
        RAW("attn_sconv_w", "attn_sconv.weight", H * 4 * 2);
        RAW("mlp_sconv_w", "mlp_sconv.weight", H * 4 * 2);
        RAW("q_clip_scale", "attn.q_l2_clip.scale", Q * 2);
        RAW("rel_clip_scale", "attn.rel_l2_clip.scale", 384 * 2);
        RAW("rel_logits_proj", "attn.rel_logits_proj.proj", 16 * 1024 * 2);
        if (i >= 6 && i < 18) {
            const char *stems[] = {"attn", "k", "v", "mlp"};
            for (unsigned j = 0; j < 4; j++) {
                const char *parts[] = {"down", "forget", "gain"};
                for (unsigned part = 0; part < 3; part++) {
                    (void)snprintf(name, sizeof name, "%scontrollers.%s.%s_weight", prefix,
                                   stems[j], parts[part]);
                    (void)snprintf(label, sizeof label, "layers.%u.ctrl.%u.%s_%s", i, j,
                                   parts[part], kind == 2 ? "nvfp4" : "w");
                    unsigned rows = part ? H : 64, cols = part ? 64 : 2 * H;
                    if (kind == 2)
                        staged_linear(w, label, 0, name, rows, cols, 2, false);
                    else
                        raw(w, label, 0, name, (size_t)rows * cols * 2);
                }
            }
        }
        if (kind == 2)
            LINEAR("router_nvfp4", "mlp.gate.weight", E, H, 2);
        else
            RAW("router_w", "mlp.gate.weight", E * H * 2);
        RAW("router_bias", "mlp.gate.bias", E * 4);
        (void)snprintf(label, sizeof label, "layers.%u.w13", i);
        expert(w, label, prefix, "w13", i, H);
        (void)snprintf(label, sizeof label, "layers.%u.w2", i);
        expert(w, label, prefix, "w2", i, I);
#undef LINEAR
#undef RAW
    }
    if (!w->config.with_mtp || w->status != IFN_OK)
        return;
    fuse(w);
    raw(w, "dspark.mask_embed", 1, "mask_embed", H * 2);
    raw(w, "dspark.norm_w", 1, "norm.weight", H * 2);
    raw(w, "dspark.markov_dflare_norm", 1, "markov_head.dflare_norm.weight", H * 2);
    raw(w, "dspark.markov_w1", 1, "markov_head.markov_w1.weight", (size_t)V * 256 * 2);
    raw(w, "dspark.rnn_bias", 1, "markov_head.rnn_proj.bias", 768 * 2);
    raw(w, "dspark.hidden_out_w", 1, "markov_head.hidden_out.weight", H * 256 * 2);
    raw(w, "dspark.hidden_out_b", 1, "markov_head.hidden_out.bias", H * 2);
    stage(w, 1, "markov_head.rnn_proj.weight", 0, 768 * 5120 * 2, false);
    ifn_weight_view *sp = view(w, "dspark.rnn_w_sp", 0);
    if (!sp)
        return;
    sp->data = block(w, sp->label, "", 768 * 512 * 2);
    ifn_weight_view *rh = view(w, "dspark.rnn_w_h", 0);
    if (!rh)
        return;
    rh->data = block(w, rh->label, "", 768 * H * 2);
    ifn_weight_view *rf = view(w, "dspark.rnn_w_f", 0);
    if (!rf)
        return;
    rf->data = block(w, rf->label, "", 768 * H * 2);
    if (w->execute && w->status == IFN_OK)
        ifn_weight_backend_check(w,
                                 w->backend.rnn(w->backend.context, w->stage, pointer(w, sp->data),
                                                pointer(w, rh->data), pointer(w, rf->data),
                                                w->stream),
                                 "split MTP RNN");
    stage(w, 0, "model.llm.unembed.weight", 0, (size_t)VU * H * 2, true);
    alias(w, "dspark.unembed_draft", named(w, "unembed"));
    ifn_weight_view *bias = view(w, "dspark.fold_b", 0);
    if (!bias)
        return;
    bias->data = block(w, bias->label, "", VU * 4);
    if (w->execute && w->status == IFN_OK) {
        ifn_weight_view *hw = named(w, "dspark.hidden_out_w"),
                        *hb = named(w, "dspark.hidden_out_b");
        if (w->status == IFN_OK)
            ifn_weight_backend_check(w,
                                     w->backend.fold(w->backend.context, w->stage,
                                                     pointer(w, hw->data), pointer(w, hb->data),
                                                     pointer(w, bias->data), w->fold_temp,
                                                     w->stream),
                                     "fold MTP Markov head");
    }
    (void)linear(w, "dspark.fold_w", VU, 256, mtp_kind, w->fold_temp);
    for (unsigned i = 0; i < 3 && w->status == IFN_OK; i++) {
        (void)snprintf(prefix, sizeof prefix, "layers.%u.", i);
#define RAW(field, suffix, bytes)                                                                  \
    do {                                                                                           \
        (void)snprintf(label, sizeof label, "dspark.layers.%u.%s", i, field);                      \
        (void)snprintf(name, sizeof name, "%s%s", prefix, suffix);                                 \
        raw(w, label, 1, name, bytes);                                                             \
    } while (0)
#define CONCAT(field, parts, rows, count)                                                          \
    do {                                                                                           \
        (void)snprintf(label, sizeof label, "dspark.layers.%u.%s", i, field);                      \
        concat(w, label, 1, prefix, parts, rows, count, H, mtp_kind);                              \
    } while (0)
#define LINEAR(field, suffix, rows, cols)                                                          \
    do {                                                                                           \
        (void)snprintf(label, sizeof label, "dspark.layers.%u.%s", i, field);                      \
        (void)snprintf(name, sizeof name, "%s%s", prefix, suffix);                                 \
        staged_linear(w, label, 1, name, rows, cols, mtp_kind, false);                             \
    } while (0)
        RAW("input_norm", "input_layernorm.weight", H * 2);
        RAW("post_norm", "post_attention_layernorm.weight", H * 2);
        RAW("dflare_norm", "dflare_norm.weight", H * 2);
        RAW("q_norm", "self_attn.q_norm.weight", 128 * 2);
        RAW("k_norm", "self_attn.k_norm.weight", 128 * 2);
        RAW("k_norm_ctx", "self_attn.k_norm_ctx.weight", 128 * 2);
        const char *qkv[] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                             "self_attn.v_proj.weight"};
        const unsigned qr[] = {Q, KV, KV};
        CONCAT("w_qkv", qkv, qr, 3);
        LINEAR("w_o", "self_attn.o_proj.weight", H, Q);
        const char *gu[] = {"mlp.gate_proj.weight", "mlp.up_proj.weight"};
        const unsigned gr[] = {M, M};
        CONCAT("w_gu", gu, gr, 2);
        LINEAR("w_down", "mlp.down_proj.weight", H, M);
        const char *kv[] = {"self_attn.k_proj_ctx.weight", "self_attn.v_proj_ctx.weight"};
        const unsigned kr[] = {KV, KV};
        CONCAT("w_kv_ctx", kv, kr, 2);
#undef RAW
#undef CONCAT
#undef LINEAR
    }
}
