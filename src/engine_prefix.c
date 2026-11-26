/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Pinned four-checkpoint local/recurrent cache and bounded global-KV shield.
 * Model, arena format and execution identity are session-owned and immutable. */
#include "engine_internal.h"
#include <string.h>

static size_t add(sq_session *s, size_t a, size_t b) {
    size_t out = 0;
    if (!sq_size_add(a, b, &out))
        s->status = SQ_LIMIT;
    return out;
}

/* Common traversal for sizing, saving and restoring. All GPU/host ranges are
 * owned by this session. Host validity is committed only after a complete sync. */
typedef struct {
    sq_session *s;
    unsigned char *host;
    size_t offset, capacity;
    int direction; /* -1 plan, 0 restore, 1 save */
} transfer;

static void span(transfer *t, void *device, size_t bytes) {
    size_t end = add(t->s, t->offset, bytes);
    if (t->s->status != SQ_OK)
        return;
    if (t->direction >= 0) {
        if (end > t->capacity || !device || !t->host) {
            t->s->status = SQ_LIMIT;
            return;
        }
        void *host = t->host + t->offset;
        const sq_weight_backend *b = &t->s->weights->backend;
        sq_ar_check(t->s,
                    b->copy(b->context, t->direction ? host : device, t->direction ? device : host,
                            bytes, t->direction, t->s->stream),
                    "prefix state transfer");
    }
    t->offset = end;
}

static void local(transfer *t) {
    sq_session *s = t->s;
    for (unsigned i = 0; i < 48; i++) {
        if (!s->schedule[i].global) {
            span(t, s->k_cache[i], (size_t)s->kv_cap[i] * 2048);
            span(t, s->v_cache[i], (size_t)s->kv_cap[i] * 2048);
        }
        span(t, s->k_ring[i], 16 * 1024 * 4);
        span(t, s->v_ring[i], 16 * 1024 * 4);
        span(t, s->attn_ring[i], 16 * 2304 * 4);
        span(t, s->mlp_ring[i], 16 * 2304 * 4);
    }
    if (s->ds_f_ring) {
        for (unsigned i = 0; i < 3; i += 2) {
            span(t, s->ds_k[i], (size_t)s->ds_cap[i] * 2048);
            span(t, s->ds_v[i], (size_t)s->ds_cap[i] * 2048);
        }
        span(t, s->ds_f_ring, 16 * 2304 * 4);
    }
}

static void pair(transfer *t, void *k, void *v, void *ks, void *vs, uint32_t n) {
    unsigned format = t->s->config.kv_format;
    span(t, k, (size_t)n * (format == 0 ? 2048 : format == 1 ? 1024 : 512));
    span(t, v, (size_t)n * (format == 0 ? 2048 : format == 1 ? 1024 : 512));
    if (format) {
        span(t, ks, (size_t)n * (format == 1 ? 16 : 32));
        span(t, vs, (size_t)n * (format == 1 ? 16 : 32));
    }
}

static void global(transfer *t, uint32_t n) {
    sq_session *s = t->s;
    for (unsigned i = 0; i < 48; i++)
        if (s->schedule[i].global)
            pair(t, s->k_cache[i], s->v_cache[i], s->k_scale[i], s->v_scale[i], n);
    if (s->ds_f_ring)
        pair(t, s->ds_k[1], s->ds_v[1], s->ds_ks[1], s->ds_vs[1],
             n < (uint32_t)s->ds_cap[1] ? n : (uint32_t)s->ds_cap[1]);
}

static void sync_transfer(sq_session *s) {
    if (s->status == SQ_OK)
        sq_ar_check(s, s->weights->backend.sync(s->weights->backend.context, s->stream),
                    "complete prefix state transfer");
}

void sq_prefix_invalidate(sq_session *s) {
    s->prefix.length = 0;
    s->prefix.live = -1;
    s->prefix.shield_tokens = 0;
    for (unsigned i = 0; i < 4; i++)
        s->prefix.checkpoints[i].valid = false;
}

void sq_prefix_free(sq_session *s) {
    sq_prefix *p = &s->prefix;
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

sq_status sq_prefix_allocate(sq_session *s) {
    if (s->prefix.enabled)
        return SQ_OK;
    sq_prefix *p = &s->prefix;
    transfer l = {s, NULL, 0, 0, -1}, g = l;
    local(&l);
    global(&g, s->config.context_tokens < 4096 ? s->config.context_tokens : 4096);
    p->local_bytes = l.offset;
    p->shield_bytes = g.offset;
    size_t total = 0;
    for (unsigned i = 0; i < 4; i++)
        total = add(s, total, l.offset);
    total = add(s, total, g.offset);
    total = add(s, total, (size_t)s->config.context_tokens * 4);
    if (s->status != SQ_OK)
        return sq_error(s->status, "prefix size overflow");
    p->tokens = s->allocator.allocate(s->allocator.context, (size_t)s->config.context_tokens * 4);
    for (unsigned i = 0; i < 4 && p->tokens; i++) {
        p->checkpoints[i].data = s->allocator.allocate(s->allocator.context, l.offset);
        if (!p->checkpoints[i].data)
            break;
    }
    if (p->tokens && p->checkpoints[3].data)
        p->shield = s->allocator.allocate(s->allocator.context, g.offset);
    if (!p->shield) {
        sq_prefix_free(s);
        return sq_error(SQ_OUT_OF_MEMORY, "allocate prefix checkpoints/shield");
    }
    p->host_bytes = total;
    p->enabled = true;
    sq_prefix_invalidate(s);
    return SQ_OK;
}

void sq_prefix_save(sq_session *s, unsigned slot, uint32_t position) {
    sq_prefix *p = &s->prefix;
    if (!p->enabled || p->shield_tokens || s->status != SQ_OK || slot >= 4)
        return;
    sq_checkpoint *c = &p->checkpoints[slot];
    c->valid = false;
    transfer t = {s, c->data, 0, p->local_bytes, 1};
    local(&t);
    sync_transfer(s);
    if (s->status == SQ_OK && t.offset == p->local_bytes) {
        c->position = position;
        c->valid = true;
    }
}

uint32_t sq_prefix_begin(sq_session *s, const int32_t *tokens, uint32_t n, uint32_t max_new,
                         bool mtp) {
    sq_prefix *p = &s->prefix;
    s->prefix_info = (sq_prefix_info)SQ_PREFIX_INFO_INIT;
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
        transfer t = {s, p->shield, 0, p->shield_bytes, 1};
        global(&t, p->shield_tokens);
        sync_transfer(s);
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
        sq_prefix_invalidate(s);
    else {
        if (best >= 0) {
            transfer t = {s, p->checkpoints[best].data, 0, p->local_bytes, 0};
            local(&t);
            sync_transfer(s);
            s->prefix_info.checkpoint_restores++;
        }
        s->position = (int32_t)start;
        const sq_weight_backend *b = &s->weights->backend;
        if (s->status == SQ_OK)
            sq_ar_check(s, b->copy(b->context, s->d_pos, &s->position, 4, 0, s->stream),
                        "restore prefix position");
        s->prefix_info.reused_tokens = start;
    }
    for (unsigned i = 0; i < 4; i++)
        if (p->checkpoints[i].position > start)
            p->checkpoints[i].valid = false;
    p->live = -1;
    return start;
}

void sq_prefix_prompt(sq_session *s, const int32_t *tokens, uint32_t n, bool mtp) {
    sq_prefix *p = &s->prefix;
    if (!p->enabled || p->shield_tokens || s->status != SQ_OK)
        return;
    memcpy(p->tokens, tokens, (size_t)n * 4);
    p->length = n;
    p->use_mtp = mtp;
    if (n >= 128)
        sq_prefix_save(s, 2, n);
}

void sq_prefix_feed(sq_session *s, int32_t token) {
    sq_prefix *p = &s->prefix;
    if (p->enabled && !p->shield_tokens && s->status == SQ_OK &&
        p->length < s->config.context_tokens)
        p->tokens[p->length++] = token;
}

void sq_prefix_finish(sq_session *s, bool cancelled) {
    sq_prefix *p = &s->prefix;
    if (!p->enabled)
        return;
    if (cancelled || s->status != SQ_OK) {
        sq_prefix_invalidate(s);
        return;
    }
    if (p->shield_tokens) {
        transfer t = {s, p->shield, 0, p->shield_bytes, 0};
        global(&t, p->shield_tokens);
        sync_transfer(s);
        p->live = -1;
        p->shield_tokens = 0;
    } else {
        p->live = s->position;
        if (p->length >= 128)
            sq_prefix_save(s, 3, p->length);
    }
    if (s->status != SQ_OK)
        sq_prefix_invalidate(s);
}
