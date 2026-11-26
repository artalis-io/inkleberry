/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Pinned four-checkpoint local/recurrent cache and bounded global-KV shield.
 * Model, arena format and execution identity are session-owned and immutable. */
#include "engine_internal.h"
#include <string.h>

static size_t ifn_engine_prefix_local_add(ifn_session *s, size_t a, size_t b) {
    size_t out = 0;
    if (!ifn_size_add(a, b, &out))
        s->status = IFN_LIMIT;
    return out;
}

/* Common traversal for sizing, saving and restoring. All GPU/host ranges are
 * owned by this session. Host validity is committed only after a complete sync. */
typedef struct {
    ifn_session *s;
    unsigned char *host;
    size_t offset, capacity;
    int direction; /* -1 plan, 0 restore, 1 save */
} ifn_engine_prefix_local_transfer;

static void ifn_engine_prefix_local_span(ifn_engine_prefix_local_transfer *t, void *device,
                                         size_t bytes) {
    size_t end = ifn_engine_prefix_local_add(t->s, t->offset, bytes);
    if (t->s->status != IFN_OK)
        return;
    if (t->direction >= 0) {
        if (end > t->capacity || !device || !t->host) {
            t->s->status = IFN_LIMIT;
            return;
        }
        void *host = t->host + t->offset;
        const ifn_weight_backend *b = &t->s->weights->backend;
        ifn_ar_check(t->s,
                     b->copy(b->context, t->direction ? host : device, t->direction ? device : host,
                             bytes, t->direction, t->s->stream),
                     "prefix state transfer");
    }
    t->offset = end;
}

static void ifn_engine_prefix_local_local(ifn_engine_prefix_local_transfer *t) {
    ifn_session *s = t->s;
    for (unsigned i = 0; i < 48; i++) {
        if (!s->schedule[i].global) {
            ifn_engine_prefix_local_span(t, s->k_cache[i], (size_t)s->kv_cap[i] * 2048);
            ifn_engine_prefix_local_span(t, s->v_cache[i], (size_t)s->kv_cap[i] * 2048);
        }
        ifn_engine_prefix_local_span(t, s->k_ring[i], 16 * 1024 * 4);
        ifn_engine_prefix_local_span(t, s->v_ring[i], 16 * 1024 * 4);
        ifn_engine_prefix_local_span(t, s->attn_ring[i], 16 * 2304 * 4);
        ifn_engine_prefix_local_span(t, s->mlp_ring[i], 16 * 2304 * 4);
    }
    if (s->ds_f_ring) {
        for (unsigned i = 0; i < 3; i += 2) {
            ifn_engine_prefix_local_span(t, s->ds_k[i], (size_t)s->ds_cap[i] * 2048);
            ifn_engine_prefix_local_span(t, s->ds_v[i], (size_t)s->ds_cap[i] * 2048);
        }
        ifn_engine_prefix_local_span(t, s->ds_f_ring, 16 * 2304 * 4);
    }
}

static void ifn_engine_prefix_local_pair(ifn_engine_prefix_local_transfer *t, void *k, void *v,
                                         void *ks, void *vs, uint32_t n) {
    unsigned format = t->s->config.kv_format;
    ifn_engine_prefix_local_span(t, k, (size_t)n * (format == 0 ? 2048 : format == 1 ? 1024 : 512));
    ifn_engine_prefix_local_span(t, v, (size_t)n * (format == 0 ? 2048 : format == 1 ? 1024 : 512));
    if (format) {
        ifn_engine_prefix_local_span(t, ks, (size_t)n * (format == 1 ? 16 : 32));
        ifn_engine_prefix_local_span(t, vs, (size_t)n * (format == 1 ? 16 : 32));
    }
}

static void ifn_engine_prefix_local_global(ifn_engine_prefix_local_transfer *t, uint32_t n) {
    ifn_session *s = t->s;
    for (unsigned i = 0; i < 48; i++)
        if (s->schedule[i].global)
            ifn_engine_prefix_local_pair(t, s->k_cache[i], s->v_cache[i], s->k_scale[i],
                                         s->v_scale[i], n);
    if (s->ds_f_ring)
        ifn_engine_prefix_local_pair(t, s->ds_k[1], s->ds_v[1], s->ds_ks[1], s->ds_vs[1],
                                     n < (uint32_t)s->ds_cap[1] ? n : (uint32_t)s->ds_cap[1]);
}

static void ifn_engine_prefix_local_sync_transfer(ifn_session *s) {
    if (s->status == IFN_OK)
        ifn_ar_check(s, s->weights->backend.sync(s->weights->backend.context, s->stream),
                     "complete prefix state transfer");
}

void ifn_prefix_invalidate(ifn_session *s) {
    s->prefix.length = 0;
    s->prefix.live = -1;
    s->prefix.shield_tokens = 0;
    for (unsigned i = 0; i < 4; i++)
        s->prefix.checkpoints[i].valid = false;
}

void ifn_prefix_free(ifn_session *s) {
    ifn_prefix *p = &s->prefix;
    if (p->tokens)
        s->allocator.release(s->allocator.context, p->tokens);
    if (p->shield)
        s->allocator.release(s->allocator.context, p->shield);
    for (unsigned i = 0; i < 4; i++)
        if (p->checkpoints[i].data)
            s->allocator.release(s->allocator.context, p->checkpoints[i].data);
    memset(p, 0, sizeof *p);
    p->live = -1;
}

ifn_status ifn_prefix_allocate(ifn_session *s) {
    if (s->prefix.enabled)
        return IFN_OK;
    ifn_prefix *p = &s->prefix;
    ifn_engine_prefix_local_transfer l = {s, NULL, 0, 0, -1}, g = l;
    ifn_engine_prefix_local_local(&l);
    ifn_engine_prefix_local_global(&g, s->config.context_tokens < 4096 ? s->config.context_tokens
                                                                       : 4096);
    p->local_bytes = l.offset;
    p->shield_bytes = g.offset;
    size_t total = 0;
    for (unsigned i = 0; i < 4; i++)
        total = ifn_engine_prefix_local_add(s, total, l.offset);
    total = ifn_engine_prefix_local_add(s, total, g.offset);
    total = ifn_engine_prefix_local_add(s, total, (size_t)s->config.context_tokens * 4);
    if (s->status != IFN_OK)
        return ifn_error(s->status, "prefix size overflow");
    p->tokens = s->allocator.allocate(s->allocator.context, (size_t)s->config.context_tokens * 4);
    for (unsigned i = 0; i < 4 && p->tokens; i++) {
        p->checkpoints[i].data = s->allocator.allocate(s->allocator.context, l.offset);
        if (!p->checkpoints[i].data)
            break;
    }
    if (p->tokens && p->checkpoints[3].data)
        p->shield = s->allocator.allocate(s->allocator.context, g.offset);
    if (!p->shield) {
        ifn_prefix_free(s);
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate prefix checkpoints/shield");
    }
    p->host_bytes = total;
    p->enabled = true;
    ifn_prefix_invalidate(s);
    return IFN_OK;
}

void ifn_prefix_save(ifn_session *s, unsigned slot, uint32_t position) {
    ifn_prefix *p = &s->prefix;
    if (!p->enabled || p->shield_tokens || s->status != IFN_OK || slot >= 4)
        return;
    ifn_checkpoint *c = &p->checkpoints[slot];
    c->valid = false;
    ifn_engine_prefix_local_transfer t = {s, c->data, 0, p->local_bytes, 1};
    ifn_engine_prefix_local_local(&t);
    ifn_engine_prefix_local_sync_transfer(s);
    if (s->status == IFN_OK && t.offset == p->local_bytes) {
        c->position = position;
        c->valid = true;
    }
}

uint32_t ifn_prefix_begin(ifn_session *s, const int32_t *tokens, uint32_t n, uint32_t max_new,
                          bool mtp) {
    ifn_prefix *p = &s->prefix;
    s->prefix_info = (ifn_prefix_info)IFN_PREFIX_INFO_INIT;
    s->prefix_info.host_bytes = p->host_bytes;
    if (!p->enabled)
        return 0;
    uint32_t common = 0, limit = p->length < n - 1 ? p->length : n - 1;
    while (common < limit && p->tokens[common] == tokens[common])
        common++;
    uint32_t need = n + max_new + 16;
    p->shield_tokens = 0;
    uint32_t start = 0;
    if (p->length >= 1024 && common < 512 && need <= 4096) {
        p->shield_tokens = p->length < need ? p->length : need;
        ifn_engine_prefix_local_transfer t = {s, p->shield, 0, p->shield_bytes, 1};
        ifn_engine_prefix_local_global(&t, p->shield_tokens);
        ifn_engine_prefix_local_sync_transfer(s);
        s->prefix_info.shielded_tokens = p->shield_tokens;
        return 0;
    }
    int best = -1;
    if (common && p->use_mtp == mtp) {
        if (p->live > 0 && (uint32_t)p->live <= common)
            start = (uint32_t)p->live;
        for (unsigned i = 0; i < 4; i++)
            if (p->checkpoints[i].valid && p->checkpoints[i].position <= common &&
                p->checkpoints[i].position > start) {
                best = (int)i;
                start = p->checkpoints[i].position;
            }
    }
    if (!start)
        ifn_prefix_invalidate(s);
    else {
        if (best >= 0) {
            ifn_engine_prefix_local_transfer t = {s, p->checkpoints[best].data, 0, p->local_bytes,
                                                  0};
            ifn_engine_prefix_local_local(&t);
            ifn_engine_prefix_local_sync_transfer(s);
            s->prefix_info.checkpoint_restores++;
        }
        s->position = (int32_t)start;
        const ifn_weight_backend *b = &s->weights->backend;
        if (s->status == IFN_OK)
            ifn_ar_check(s, b->copy(b->context, s->d_pos, &s->position, 4, 0, s->stream),
                         "restore prefix position");
        s->prefix_info.reused_tokens = start;
    }
    for (unsigned i = 0; i < 4; i++)
        if (p->checkpoints[i].position > start)
            p->checkpoints[i].valid = false;
    p->live = -1;
    return start;
}

void ifn_prefix_prompt(ifn_session *s, const int32_t *tokens, uint32_t n, bool mtp) {
    ifn_prefix *p = &s->prefix;
    if (!p->enabled || p->shield_tokens || s->status != IFN_OK)
        return;
    memcpy(p->tokens, tokens, (size_t)n * 4);
    p->length = n;
    p->use_mtp = mtp;
    if (n >= 128)
        ifn_prefix_save(s, 2, n);
}

void ifn_prefix_feed(ifn_session *s, int32_t token) {
    ifn_prefix *p = &s->prefix;
    if (p->enabled && !p->shield_tokens && s->status == IFN_OK &&
        p->length < s->config.context_tokens)
        p->tokens[p->length++] = token;
}

void ifn_prefix_finish(ifn_session *s, bool cancelled) {
    ifn_prefix *p = &s->prefix;
    if (!p->enabled)
        return;
    if (cancelled || s->status != IFN_OK) {
        ifn_prefix_invalidate(s);
        return;
    }
    if (p->shield_tokens) {
        ifn_engine_prefix_local_transfer t = {s, p->shield, 0, p->shield_bytes, 0};
        ifn_engine_prefix_local_global(&t, p->shield_tokens);
        ifn_engine_prefix_local_sync_transfer(s);
        p->live = -1;
        p->shield_tokens = 0;
    } else {
        p->live = s->position;
        if (p->length >= 128)
            ifn_prefix_save(s, 3, p->length);
    }
    if (s->status != IFN_OK)
        ifn_prefix_invalidate(s);
}
