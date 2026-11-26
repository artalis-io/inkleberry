/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Host binding barrier derived from pinned src/weights.zig and constants.zig.
 * No upload, quantization, device allocation or implicit optional-MTP fallback. */
#include "inkling.h"
#include <stdio.h>
#include <string.h>

enum {
    IFN_INKLING_LOCAL_H = 2304,
    IFN_INKLING_LOCAL_Q = 3072,
    IFN_INKLING_LOCAL_KV = 1024,
    IFN_INKLING_LOCAL_R = 384,
    IFN_INKLING_LOCAL_E = 128,
    IFN_INKLING_LOCAL_I = 1152,
    IFN_INKLING_LOCAL_V = 201024,
    IFN_INKLING_LOCAL_M = 9216
};

struct ifn_inkling_checkpoint {
    ifn_options allocator;
    ifn_safetensors *files[2];
    const ifn_tensor_info *bound[2048];
    uint32_t sources[2048];
    size_t count;
};

static ifn_status ifn_inkling_local_spec(ifn_spec_fn visit, void *context, uint32_t source,
                                         const char *prefix, const char *suffix, ifn_dtype dtype,
                                         uint64_t a, uint64_t b, uint64_t c) {
    ifn_inkling_spec s = {0};
    int n = snprintf(s.name, sizeof s.name, "%s%s", prefix, suffix);
    if (n < 0 || (size_t)n >= sizeof s.name)
        return ifn_error(IFN_LIMIT, "internal tensor name limit");
    s.source = source;
    s.dtype = dtype;
    s.rank = c ? 3 : b ? 2 : 1;
    s.shape[0] = a;
    s.shape[1] = b;
    s.shape[2] = c;
    return visit(context, &s);
}

#define IFN_INKLING_LOCAL_S(src, pre, suf, type, a, b, c)                                          \
    do {                                                                                           \
        ifn_status st = ifn_inkling_local_spec(visit, context, src, pre, suf, type, a, b, c);      \
        if (st != IFN_OK)                                                                          \
            return st;                                                                             \
    } while (0)

ifn_status ifn_inkling_visit_specs(uint32_t include_mtp, ifn_spec_fn visit, void *context) {
    if (!visit || include_mtp > 1)
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid schema visitor");
    const char *base = "model.llm.";
    IFN_INKLING_LOCAL_S(0, base, "embed.weight", IFN_BF16, IFN_INKLING_LOCAL_V, IFN_INKLING_LOCAL_H,
                        0);
    IFN_INKLING_LOCAL_S(0, base, "embed_norm.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 0, 0);
    IFN_INKLING_LOCAL_S(0, base, "norm.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 0, 0);
    IFN_INKLING_LOCAL_S(0, base, "unembed.weight", IFN_BF16, IFN_INKLING_LOCAL_V,
                        IFN_INKLING_LOCAL_H, 0);
    char p[96];
    for (unsigned l = 0; l < 23; l++) {
        (void)snprintf(p, sizeof p, "model.llm.attn_res.attn_queries.%u", l);
        bool global = (l + 2) % 6 == 0;
        IFN_INKLING_LOCAL_S(0, p, "", IFN_F32, global ? 3 : IFN_INKLING_LOCAL_H,
                            global ? IFN_INKLING_LOCAL_H : 0, 0);
    }
    for (unsigned l = 0; l < 24; l++) {
        (void)snprintf(p, sizeof p, "model.llm.attn_res.mlp_queries.%u", l);
        IFN_INKLING_LOCAL_S(0, p, "", IFN_F32, IFN_INKLING_LOCAL_H, 0, 0);
    }
    IFN_INKLING_LOCAL_S(0, base, "attn_res.output_query", IFN_F32, IFN_INKLING_LOCAL_H, 0, 0);
    IFN_INKLING_LOCAL_S(0, base, "middle_cycle_cross_pass_queries", IFN_F32, 24,
                        IFN_INKLING_LOCAL_H, 0);
    IFN_INKLING_LOCAL_S(0, base, "middle_cycle_cross_pass_kv_queries", IFN_F32, 24,
                        IFN_INKLING_LOCAL_H, 0);
    for (unsigned l = 0; l < 24; l++) {
        (void)snprintf(p, sizeof p, "model.llm.layers.%u.", l);
        bool global = (l + 1) % 6 == 0;
        if (global) {
            IFN_INKLING_LOCAL_S(0, p, "q_input_norm.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 0, 0);
            IFN_INKLING_LOCAL_S(0, p, "k_input_norm.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 0, 0);
            IFN_INKLING_LOCAL_S(0, p, "v_input_norm.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 0, 0);
        } else
            IFN_INKLING_LOCAL_S(0, p, "attn_norm.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 0, 0);
        IFN_INKLING_LOCAL_S(0, p, "mlp_norm.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 0, 0);
        IFN_INKLING_LOCAL_S(0, p, "attn.wq_du.weight", IFN_BF16, IFN_INKLING_LOCAL_Q,
                            IFN_INKLING_LOCAL_H, 0);
        IFN_INKLING_LOCAL_S(0, p, "attn.wr_du.weight", IFN_BF16, IFN_INKLING_LOCAL_R,
                            IFN_INKLING_LOCAL_H, 0);
        IFN_INKLING_LOCAL_S(0, p, "attn.gate_attn.weight", IFN_BF16, IFN_INKLING_LOCAL_Q,
                            IFN_INKLING_LOCAL_H, 0);
        IFN_INKLING_LOCAL_S(0, p, "attn.attn_sink_proj.weight", IFN_BF16, 24, IFN_INKLING_LOCAL_H,
                            0);
        IFN_INKLING_LOCAL_S(0, p, "attn.wk_dv.weight", IFN_BF16, IFN_INKLING_LOCAL_KV,
                            IFN_INKLING_LOCAL_H, 0);
        IFN_INKLING_LOCAL_S(0, p, "attn.wv_dv.weight", IFN_BF16, IFN_INKLING_LOCAL_KV,
                            IFN_INKLING_LOCAL_H, 0);
        IFN_INKLING_LOCAL_S(0, p, "attn.wo_ud.weight", IFN_BF16, IFN_INKLING_LOCAL_H,
                            IFN_INKLING_LOCAL_Q, 0);
        if (global) {
            IFN_INKLING_LOCAL_S(0, p, "attn.k_sconv.static_w", IFN_BF16, 4, IFN_INKLING_LOCAL_KV,
                                0);
            IFN_INKLING_LOCAL_S(0, p, "attn.v_sconv.static_w", IFN_BF16, 4, IFN_INKLING_LOCAL_KV,
                                0);
            IFN_INKLING_LOCAL_S(0, p, "attn.k_sconv.weight_proj", IFN_BF16, 32, IFN_INKLING_LOCAL_H,
                                0);
            IFN_INKLING_LOCAL_S(0, p, "attn.v_sconv.weight_proj", IFN_BF16, 32, IFN_INKLING_LOCAL_H,
                                0);
        } else {
            IFN_INKLING_LOCAL_S(0, p, "attn.k_sconv.weight", IFN_BF16, IFN_INKLING_LOCAL_KV, 1, 4);
            IFN_INKLING_LOCAL_S(0, p, "attn.v_sconv.weight", IFN_BF16, IFN_INKLING_LOCAL_KV, 1, 4);
        }
        IFN_INKLING_LOCAL_S(0, p, "attn_sconv.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 1, 4);
        IFN_INKLING_LOCAL_S(0, p, "mlp_sconv.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 1, 4);
        IFN_INKLING_LOCAL_S(0, p, "attn.q_l2_clip.scale", IFN_BF16, 24, 128, 0);
        IFN_INKLING_LOCAL_S(0, p, "attn.rel_l2_clip.scale", IFN_BF16, 24, 16, 0);
        IFN_INKLING_LOCAL_S(0, p, "attn.rel_logits_proj.proj", IFN_BF16, 16, 1024, 0);
        if (l >= 6 && l < 18) {
            const char *stems[] = {"attn", "k", "v", "mlp"};
            char cpre[112];
            for (size_t j = 0; j < 4; j++) {
                (void)snprintf(cpre, sizeof cpre, "%scontrollers.%s.", p, stems[j]);
                IFN_INKLING_LOCAL_S(0, cpre, "down_weight", IFN_BF16, 64, 2 * IFN_INKLING_LOCAL_H,
                                    0);
                IFN_INKLING_LOCAL_S(0, cpre, "forget_weight", IFN_BF16, IFN_INKLING_LOCAL_H, 64, 0);
                IFN_INKLING_LOCAL_S(0, cpre, "gain_weight", IFN_BF16, IFN_INKLING_LOCAL_H, 64, 0);
            }
        }
        IFN_INKLING_LOCAL_S(0, p, "mlp.gate.weight", IFN_BF16, IFN_INKLING_LOCAL_E,
                            IFN_INKLING_LOCAL_H, 0);
        IFN_INKLING_LOCAL_S(0, p, "mlp.gate.bias", IFN_F32, IFN_INKLING_LOCAL_E, 0, 0);
        const char *banks[] = {"w13", "w2"};
        for (size_t j = 0; j < 2; j++) {
            unsigned rows = j ? IFN_INKLING_LOCAL_H : 2 * IFN_INKLING_LOCAL_I,
                     cols = j ? IFN_INKLING_LOCAL_I : IFN_INKLING_LOCAL_H;
            char n[128];
            (void)snprintf(n, sizeof n, "%smlp.experts.%s_weight", p, banks[j]);
            IFN_INKLING_LOCAL_S(0, n, "", l < 20 ? IFN_U8 : IFN_F8_E4M3, IFN_INKLING_LOCAL_E, rows,
                                l < 20 ? cols / 2 : cols);
            IFN_INKLING_LOCAL_S(0, n, "_scale", l < 20 ? IFN_F8_E4M3 : IFN_U8, IFN_INKLING_LOCAL_E,
                                rows, l < 20 ? cols / 16 : cols / 32);
            if (l < 20) {
                IFN_INKLING_LOCAL_S(0, n, "_amax", IFN_F32, IFN_INKLING_LOCAL_E, 0, 0);
                IFN_INKLING_LOCAL_S(0, n, "_outer_scale_denominator", IFN_F32, 1, 0, 0);
            }
        }
    }
    if (!include_mtp)
        return IFN_OK;
    IFN_INKLING_LOCAL_S(1, "", "fuse_weight", IFN_BF16, 4, 4, 0);
    IFN_INKLING_LOCAL_S(1, "", "mask_embed", IFN_BF16, IFN_INKLING_LOCAL_H, 0, 0);
    IFN_INKLING_LOCAL_S(1, "", "norm.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 0, 0);
    IFN_INKLING_LOCAL_S(1, "", "markov_head.dflare_norm.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 0,
                        0);
    IFN_INKLING_LOCAL_S(1, "", "markov_head.markov_w1.weight", IFN_BF16, IFN_INKLING_LOCAL_V, 256,
                        0);
    IFN_INKLING_LOCAL_S(1, "", "markov_head.rnn_proj.bias", IFN_BF16, 768, 0, 0);
    IFN_INKLING_LOCAL_S(1, "", "markov_head.rnn_proj.weight", IFN_BF16, 768, 5120, 0);
    IFN_INKLING_LOCAL_S(1, "", "markov_head.hidden_out.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 256,
                        0);
    IFN_INKLING_LOCAL_S(1, "", "markov_head.hidden_out.bias", IFN_BF16, IFN_INKLING_LOCAL_H, 0, 0);
    for (unsigned l = 0; l < 3; l++) {
        (void)snprintf(p, sizeof p, "layers.%u.", l);
        IFN_INKLING_LOCAL_S(1, p, "input_layernorm.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 0, 0);
        IFN_INKLING_LOCAL_S(1, p, "post_attention_layernorm.weight", IFN_BF16, IFN_INKLING_LOCAL_H,
                            0, 0);
        IFN_INKLING_LOCAL_S(1, p, "dflare_norm.weight", IFN_BF16, IFN_INKLING_LOCAL_H, 0, 0);
        IFN_INKLING_LOCAL_S(1, p, "self_attn.q_norm.weight", IFN_BF16, 128, 0, 0);
        IFN_INKLING_LOCAL_S(1, p, "self_attn.k_norm.weight", IFN_BF16, 128, 0, 0);
        IFN_INKLING_LOCAL_S(1, p, "self_attn.k_norm_ctx.weight", IFN_BF16, 128, 0, 0);
        IFN_INKLING_LOCAL_S(1, p, "self_attn.q_proj.weight", IFN_BF16, IFN_INKLING_LOCAL_Q,
                            IFN_INKLING_LOCAL_H, 0);
        IFN_INKLING_LOCAL_S(1, p, "self_attn.k_proj.weight", IFN_BF16, IFN_INKLING_LOCAL_KV,
                            IFN_INKLING_LOCAL_H, 0);
        IFN_INKLING_LOCAL_S(1, p, "self_attn.v_proj.weight", IFN_BF16, IFN_INKLING_LOCAL_KV,
                            IFN_INKLING_LOCAL_H, 0);
        IFN_INKLING_LOCAL_S(1, p, "self_attn.o_proj.weight", IFN_BF16, IFN_INKLING_LOCAL_H,
                            IFN_INKLING_LOCAL_Q, 0);
        IFN_INKLING_LOCAL_S(1, p, "mlp.gate_proj.weight", IFN_BF16, IFN_INKLING_LOCAL_M,
                            IFN_INKLING_LOCAL_H, 0);
        IFN_INKLING_LOCAL_S(1, p, "mlp.up_proj.weight", IFN_BF16, IFN_INKLING_LOCAL_M,
                            IFN_INKLING_LOCAL_H, 0);
        IFN_INKLING_LOCAL_S(1, p, "mlp.down_proj.weight", IFN_BF16, IFN_INKLING_LOCAL_H,
                            IFN_INKLING_LOCAL_M, 0);
        IFN_INKLING_LOCAL_S(1, p, "self_attn.k_proj_ctx.weight", IFN_BF16, IFN_INKLING_LOCAL_KV,
                            IFN_INKLING_LOCAL_H, 0);
        IFN_INKLING_LOCAL_S(1, p, "self_attn.v_proj_ctx.weight", IFN_BF16, IFN_INKLING_LOCAL_KV,
                            IFN_INKLING_LOCAL_H, 0);
    }
    return IFN_OK;
}

#undef IFN_INKLING_LOCAL_S

ifn_status ifn_inkling_check_descriptor(const ifn_tensor_info *t, const ifn_inkling_spec *s) {
    if (!t || !s || !s->rank || s->rank > 3)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing descriptor/schema");
    if (strcmp(t->name, s->name) || t->dtype != s->dtype || t->rank != s->rank)
        return ifn_error(IFN_FORMAT, "Inkling tensor name/dtype/rank mismatch");
    for (uint32_t i = 0; i < s->rank; i++)
        if (t->shape[i] != s->shape[i])
            return ifn_error(IFN_FORMAT, "Inkling tensor shape mismatch");
    return IFN_OK;
}

static ifn_status ifn_inkling_local_bind(void *context, const ifn_inkling_spec *s) {
    ifn_inkling_checkpoint *c = context;
    const ifn_tensor_info *t = NULL;
    ifn_status status = ifn_safetensors_find(c->files[s->source], s->name, strlen(s->name), &t);
    if (status == IFN_OK)
        status = ifn_inkling_check_descriptor(t, s);
    if (status != IFN_OK) {
        char message[256];
        (void)snprintf(message, sizeof message, "invalid/missing required tensor: %s", s->name);
        return ifn_error(IFN_FORMAT, message);
    }
    if (c->count == 2048)
        return ifn_error(IFN_LIMIT, "bound tensor limit");
    c->bound[c->count] = t;
    c->sources[c->count++] = s->source;
    return IFN_OK;
}

void ifn_inkling_checkpoint_close(ifn_inkling_checkpoint *c) {
    if (!c)
        return;
    ifn_options a = c->allocator;
    ifn_safetensors_close(c->files[1]);
    ifn_safetensors_close(c->files[0]);
    a.release(a.context, c);
}

ifn_status ifn_inkling_checkpoint_open(const char *main_path, const char *mtp_path,
                                       const ifn_options *options, ifn_inkling_checkpoint **out) {
    if (out)
        *out = NULL;
    if (!main_path || !main_path[0] || !out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing checkpoint path/output");
    ifn_options a;
    ifn_status status = ifn_options_resolve(options, &a);
    if (status != IFN_OK)
        return status;
    ifn_inkling_checkpoint *c = a.allocate(a.context, sizeof *c);
    if (!c)
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate checkpoint binding");
    memset(c, 0, sizeof *c);
    c->allocator = a;
    status = ifn_safetensors_open(main_path, &a, &c->files[0]);
    if (status != IFN_OK)
        goto fail;
    if (mtp_path) {
        status = ifn_safetensors_open(mtp_path, &a, &c->files[1]);
        if (status != IFN_OK)
            goto fail;
    }
    status = ifn_inkling_visit_specs(mtp_path != NULL, ifn_inkling_local_bind, c);
    if (status != IFN_OK)
        goto fail;
    size_t counts[2] = {0};
    for (size_t i = 0; i < c->count; i++)
        counts[c->sources[i]]++;
    for (size_t source = 0; source < 2; source++) {
        for (size_t i = 0; i < ifn_safetensors_count(c->files[source]); i++) {
            const ifn_tensor_info *t = NULL;
            (void)ifn_safetensors_at(c->files[source], i, &t);
            if (source == 0 && !strncmp(t->name, "model.visual.", 13))
                continue;
            if (counts[source] == 0) {
                status = ifn_error(IFN_FORMAT, "unexpected checkpoint tensors");
                goto fail;
            }
            counts[source]--;
        }
        if (counts[source]) {
            status = ifn_error(IFN_FORMAT, "checkpoint role coverage mismatch");
            goto fail;
        }
    }
    *out = c;
    return ifn_error(IFN_OK, "");
fail:
    ifn_inkling_checkpoint_close(c);
    return status;
}

size_t ifn_inkling_checkpoint_count(const ifn_inkling_checkpoint *c) {
    return c ? c->count : 0;
}

ifn_status ifn_inkling_checkpoint_at(const ifn_inkling_checkpoint *c, size_t index,
                                     uint32_t *source, const ifn_tensor_info **out) {
    if (source)
        *source = 0;
    if (out)
        *out = NULL;
    if (!c || !source || !out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing binding outputs");
    if (index >= c->count)
        return ifn_error(IFN_NOT_FOUND, "bound tensor not found");
    *source = c->sources[index];
    *out = c->bound[index];
    return ifn_error(IFN_OK, "");
}

ifn_status ifn_inkling_checkpoint_data(const ifn_inkling_checkpoint *c, size_t index,
                                       const void **out, size_t *bytes) {
    if (out)
        *out = NULL;
    if (bytes)
        *bytes = 0;
    if (!c || !out || !bytes)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing bound data output");
    if (index >= c->count)
        return ifn_error(IFN_NOT_FOUND, "bound tensor not found");
    const ifn_tensor_info *bound = c->bound[index];
    ifn_safetensors *file = c->files[c->sources[index]];
    for (size_t i = 0; i < ifn_safetensors_count(file); i++) {
        const ifn_tensor_info *t = NULL;
        (void)ifn_safetensors_at(file, i, &t);
        if (t == bound)
            return ifn_safetensors_data(file, i, out, bytes);
    }
    return ifn_error(IFN_FORMAT, "invalid binding");
}

ifn_status ifn_inkling_lookup(const ifn_inkling_checkpoint *c, uint32_t source, const char *name,
                              const void **out, size_t *bytes) {
    if (out)
        *out = NULL;
    if (bytes)
        *bytes = 0;
    if (!c || source > 1 || !name || !out || !bytes)
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid weight source");
    return ifn_safetensors_named_data(c->files[source], name, out, bytes);
}

ifn_status ifn_inkling_sha256(const ifn_inkling_checkpoint *checkpoint, uint32_t source,
                              unsigned char out[32]) {
    if (!checkpoint || source > 1 || !checkpoint->files[source])
        return ifn_error(IFN_INVALID_ARGUMENT, "missing checkpoint source");
    return ifn_safetensors_sha256(checkpoint->files[source], out);
}
