/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Synchronous greedy AR session. No prefix cache, stochastic sampler or DSpark. */
#include "engine_internal.h"
#include "checked_prefill.h"
#include <stdio.h>
#include <string.h>
static atomic_flag live_session = ATOMIC_FLAG_INIT;

void sq_ar_check(sq_session *s, int rc, const char *op) {
    if (rc && s->status == SQ_OK) {
        s->status = (rc == 2 || rc == 10003) ? SQ_OUT_OF_MEMORY
                    : rc == 801              ? SQ_UNSUPPORTED
                                             : SQ_GPU_ERROR;
        (void)snprintf(s->failure, sizeof s->failure, "%s: CUDA status %d", op, rc);
    }
}

static sq_status result(sq_session *s) {
    return sq_error(s->status, s->status == SQ_OK ? "" : s->failure);
}

static bool lock(sq_session *s) {
    return s && !atomic_flag_test_and_set_explicit(&s->busy, memory_order_acquire);
}

static void unlock(sq_session *s) {
    atomic_flag_clear_explicit(&s->busy, memory_order_release);
}

static void sync(sq_session *s) {
    if (s->status == SQ_OK)
        sq_ar_check(s, s->weights->backend.sync(s->weights->backend.context, s->stream),
                    "synchronize session");
}

static void copy(sq_session *s, void *dst, const void *src, size_t bytes, int kind) {
    if (s->status == SQ_OK)
        sq_ar_check(
            s,
            s->weights->backend.copy(s->weights->backend.context, dst, src, bytes, kind, s->stream),
            "session copy");
}

static sq_device_scope enter(sq_session *s) {
    sq_device_scope scope = {-1, NULL};
    s->status = SQ_OK;
    s->failure[0] = 0;
    sq_ar_check(
        s,
        s->weights->backend.enter(s->weights->backend.context, s->weights->config.device, &scope),
        "select session GPU");
    return scope;
}

static sq_status leave(sq_session *s, const sq_device_scope *scope) {
    if (scope->device >= 0) {
        int restore = s->weights->backend.leave(s->weights->backend.context, scope);
        /* A normal cancellation must not hide context-restoration failure. */
        if (restore && s->status == SQ_CANCELLED)
            s->status = SQ_OK;
        sq_ar_check(s, restore, "restore caller context");
    }
    if (s->status != SQ_OK && s->status != SQ_CANCELLED)
        s->poisoned = true;
    return result(s);
}

static void reset_device(sq_session *s) {
    s->position = 0;
    if (s->status != SQ_OK)
        return;
    sq_ar_check(s, squid_cuda_memset_async(s->d_pos, 0, 4, s->stream), "reset position");
    for (unsigned i = 0; i < 48 && s->status == SQ_OK; i++) {
        sq_ar_check(s, squid_cuda_memset_async(s->k_ring[i], 0, 16 * 1024 * 4, s->stream),
                    "reset K ring");
        sq_ar_check(s, squid_cuda_memset_async(s->v_ring[i], 0, 16 * 1024 * 4, s->stream),
                    "reset V ring");
        sq_ar_check(s, squid_cuda_memset_async(s->attn_ring[i], 0, 16 * 2304 * 4, s->stream),
                    "reset attention ring");
        sq_ar_check(s, squid_cuda_memset_async(s->mlp_ring[i], 0, 16 * 2304 * 4, s->stream),
                    "reset MLP ring");
    }
    CSampleConfig cfg = {0};
    cfg.top_p = 1;
    CLoopCDConfig loop = {0};
    loop.ref_pass = 1;
    copy(s, s->d_sample_cfg, &cfg, sizeof cfg, 0);
    copy(s, s->d_loopcd_cfg, &loop, sizeof loop, 0);
    sync(s);
}

static bool image_lock(sq_gpu_weights *w) {
    return !atomic_flag_test_and_set_explicit(&w->busy, memory_order_acquire);
}

static void image_unlock(sq_gpu_weights *w) {
    atomic_flag_clear_explicit(&w->busy, memory_order_release);
}

sq_status sq_session_create(sq_gpu_weights *w, const sq_session_config *input,
                            const sq_options *options, sq_session **out) {
    if (out)
        *out = NULL;
    if (!w || !out)
        return sq_error(SQ_INVALID_ARGUMENT, "missing session arguments");
    sq_session_config config = SQ_SESSION_CONFIG_INIT;
    if (input) {
        if (input->struct_size < sizeof *input || input->api_version != SQ_API_VERSION)
            return sq_error(SQ_INVALID_ARGUMENT, "unsupported session configuration size/version");
        config = *input;
    }
    if (config.context_tokens < 64 || config.context_tokens > 1048576 || config.kv_format > 2 ||
        config.use_graphs > 1 || config.reserved)
        return sq_error(SQ_INVALID_ARGUMENT, "invalid session configuration");
    sq_options allocator;
    sq_status rc = sq_options_resolve(options, &allocator);
    if (rc != SQ_OK)
        return rc;
    if (!image_lock(w))
        return sq_error(SQ_BUSY, "weight operation already active");
    if (!w->ready || atomic_flag_test_and_set_explicit(&live_session, memory_order_acquire)) {
        image_unlock(w);
        return sq_error(SQ_BUSY, "image not ready or another session exists");
    }
    sq_session *s = allocator.allocate(allocator.context, sizeof *s);
    if (!s) {
        atomic_flag_clear_explicit(&live_session, memory_order_release);
        image_unlock(w);
        return sq_error(SQ_OUT_OF_MEMORY, "allocate session bookkeeping");
    }
    memset(s, 0, sizeof *s);
    s->allocator = allocator;
    s->config = config;
    s->weights = w;
    s->retained = true;
    atomic_flag_clear(&s->busy);
    atomic_init(&s->cancelled, false);
    w->session_users++;
    rc = sq_ar_bind(w, &s->model);
    image_unlock(w);
    *out = s;
    if (rc != SQ_OK) {
        s->poisoned = true;
        return rc;
    }
    size_t groups = squid_sizeof_expert_groups();
    if (!groups || groups > 1048576) {
        s->poisoned = true;
        return sq_error(SQ_UNSUPPORTED, "unsupported expert group workspace");
    }
    sq_ar_layout(s, groups);
    s->capacity = s->offset;
    if (s->status != SQ_OK) {
        s->poisoned = true;
        return sq_error(s->status, "session layout overflow");
    }
    sq_device_scope scope = enter(s);
    const sq_weight_backend *b = &w->backend;
    size_t available = 0;
    if (s->status == SQ_OK)
        sq_ar_check(s, b->available(b->context, &available), "query session memory");
    if (s->status == SQ_OK && s->capacity > available) {
        s->status = SQ_OUT_OF_MEMORY;
        (void)snprintf(s->failure, sizeof s->failure, "insufficient free GPU memory for session");
    }
    if (s->status == SQ_OK)
        sq_ar_check(s, b->stream_create(b->context, &s->stream), "create session stream");
    if (s->status == SQ_OK)
        sq_ar_check(s, b->allocate(b->context, &s->arena, s->capacity), "allocate session arena");
    if (s->status == SQ_OK) {
        sq_ar_layout(s, groups);
        if (s->offset != s->capacity)
            s->status = SQ_LIMIT;
        reset_device(s);
    }
    rc = leave(s, &scope);
    s->ready = rc == SQ_OK;
    return rc;
}

static sq_status begin(sq_session *s, sq_device_scope *scope) {
    if (!s)
        return sq_error(SQ_INVALID_ARGUMENT, "missing session");
    if (!lock(s))
        return sq_error(SQ_BUSY, "session operation already active");
    if (!s->ready || s->poisoned) {
        unlock(s);
        return sq_error(SQ_GPU_ERROR, "session is not ready; destroy it");
    }
    atomic_store_explicit(&s->cancelled, false, memory_order_release);
    *scope = enter(s);
    return SQ_OK;
}

static sq_status end(sq_session *s, const sq_device_scope *scope) {
    sq_status rc = leave(s, scope);
    unlock(s);
    return rc;
}

static bool tokens_valid(const int32_t *tokens, size_t n) {
    if (!tokens || !n)
        return false;
    for (size_t i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= 200058)
            return false;
    return true;
}

static bool cancelled(sq_session *s) {
    return atomic_load_explicit(&s->cancelled, memory_order_acquire);
}

static void run(sq_session *s, const int32_t *tokens, int32_t n, bool lm) {
    copy(s, s->d_tokens, tokens, (size_t)n * 4, 0);
    copy(s, s->d_pos, &s->position, 4, 0);
    if (s->status == SQ_OK) {
        sq_ar_forward(s, n, lm);
        sq_ar_check(s, sq_cuda_launch_status(), "target forward launch");
        sync(s);
        if (s->status == SQ_OK)
            s->position += n;
    }
}

static void cancel_result(sq_session *s) {
    /* Retire work before returning; no prefix state is published. */
    if (s->status == SQ_OK) {
        reset_device(s);
        if (s->status == SQ_OK) {
            s->status = SQ_CANCELLED;
            (void)snprintf(s->failure, sizeof s->failure, "generation cancelled");
        }
    }
}

sq_status sq_session_reset(sq_session *s) {
    sq_device_scope scope;
    sq_status rc = begin(s, &scope);
    if (rc != SQ_OK)
        return rc;
    reset_device(s);
    return end(s, &scope);
}

sq_status sq_session_cancel(sq_session *s) {
    if (!s)
        return sq_error(SQ_INVALID_ARGUMENT, "missing session");
    atomic_store_explicit(&s->cancelled, true, memory_order_release);
    return sq_error(SQ_OK, "");
}

sq_status sq_session_logits(sq_session *s, const int32_t *tokens, size_t n, uint32_t reset,
                            float *out, size_t capacity) {
    if (!s || n > 8 || reset > 1 || !out || capacity < 200058 || !tokens_valid(tokens, n))
        return sq_error(SQ_INVALID_ARGUMENT,
                        "logits needs 1..8 valid IDs and 200058 output floats");
    sq_device_scope scope;
    sq_status rc = begin(s, &scope);
    if (rc != SQ_OK)
        return rc;
    if (n > s->config.context_tokens - (reset ? 0 : (uint32_t)s->position)) {
        sq_status restore = end(s, &scope);
        return restore == SQ_OK ? sq_error(SQ_LIMIT, "logits context limit") : restore;
    }
    if (reset)
        reset_device(s);
    CSampleConfig cfg = {0};
    cfg.inv_temp = -1;
    cfg.top_p = 1;
    copy(s, s->d_sample_cfg, &cfg, sizeof cfg, 0);
    run(s, tokens, (int32_t)n, true);
    copy(s, out, s->lm_logits + (n - 1) * 200058, 200058 * 4, 1);
    sync(s);
    if (cancelled(s))
        cancel_result(s);
    return end(s, &scope);
}

sq_status sq_session_generate(sq_session *s, const int32_t *tokens, size_t n, size_t max_new,
                              int32_t *out, size_t capacity, sq_token_callback callback,
                              void *context, sq_generate_stats *stats) {
    if (stats)
        memset(stats, 0, sizeof *stats);
    if (!s || !max_new || max_new > capacity || !out || !stats || n > s->config.context_tokens ||
        max_new > s->config.context_tokens - n || !tokens_valid(tokens, n))
        return sq_error(SQ_INVALID_ARGUMENT, "invalid greedy generation bounds or token IDs");
    sq_device_scope scope;
    sq_status rc = begin(s, &scope);
    if (rc != SQ_OK)
        return rc;
    reset_device(s);
    stats->prompt_tokens = (uint32_t)n;
    stats->last_token = -1;
    int32_t last_batch = 0;
    for (size_t i = 0; i < n && s->status == SQ_OK && !cancelled(s);) {
        if (n - i > 32) {
            size_t avail = (n - i - 1) & ~(size_t)7;
            int32_t chunk = (int32_t)(avail > 2048 ? 2048 : avail);
            sq_ar_prefill(s, tokens + i, chunk);
            i += (size_t)chunk;
        } else {
            last_batch = (int32_t)(n - i > 8 ? 8 : n - i);
            run(s, tokens + i, last_batch, i + (size_t)last_batch == n);
            i += (size_t)last_batch;
        }
    }
    int32_t token = -1;
    if (!cancelled(s))
        copy(s, &token, s->d_next_tokens + (last_batch ? last_batch - 1 : 0), 4, 1);
    sync(s);
    if (s->status == SQ_OK && !cancelled(s) && s->config.use_graphs && !s->graph) {
        sq_ar_check(s, squid_cuda_graph_begin(s->stream), "begin AR graph");
        if (s->status == SQ_OK) {
            sq_ar_forward(s, 1, true);
            sq_ar_check(s, sq_cuda_launch_status(), "capture target forward");
            /* Even on a launch error, end capture to restore the stream. */
            sq_ar_check(s, squid_cuda_graph_end_instantiate(s->stream, &s->graph),
                        "instantiate AR graph");
        }
    }
    for (size_t i = 0; i < max_new && s->status == SQ_OK && !cancelled(s); i++) {
        if (token < 0 || token >= 200058) {
            sq_ar_check(s, 1, "invalid generated token");
            break;
        }
        out[i] = token;
        stats->generated_tokens = (uint32_t)(i + 1);
        stats->last_token = token;
        if (callback && !callback(context, token)) {
            stats->finish_reason = SQ_FINISH_CALLBACK;
            break;
        }
        if (token == 200006) {
            stats->finish_reason = SQ_FINISH_EOS;
            break;
        }
        if (i + 1 == max_new || cancelled(s))
            break;
        if (s->config.use_graphs) {
            copy(s, s->d_tokens, &token, 4, 0);
            copy(s, s->d_pos, &s->position, 4, 0);
            if (s->status == SQ_OK)
                sq_ar_check(s, squid_cuda_graph_launch(s->graph, s->stream), "launch AR graph");
            sync(s);
            if (s->status == SQ_OK)
                s->position++;
        } else
            run(s, &token, 1, true);
        copy(s, &token, s->d_next_tokens, 4, 1);
        sync(s);
    }
    if (cancelled(s)) {
        stats->finish_reason = SQ_FINISH_CANCELLED;
        cancel_result(s);
    }
    return end(s, &scope);
}

sq_status sq_session_destroy(sq_session **handle) {
    if (!handle)
        return sq_error(SQ_INVALID_ARGUMENT, "missing session handle pointer");
    sq_session *s = *handle;
    if (!s)
        return sq_error(SQ_OK, "");
    if (!lock(s))
        return sq_error(SQ_BUSY, "session operation already active");
    s->ready = false;
    const sq_weight_backend *b = &s->weights->backend;
    s->status = SQ_OK;
    s->failure[0] = 0;
    if (s->arena || s->stream || s->graph || s->blas) {
        sq_device_scope scope = enter(s);
        if (s->status == SQ_OK) {
            if (s->stream)
                sync(s);
            if (s->graph) {
                int e = squid_cuda_graph_destroy(s->graph);
                if (!e)
                    s->graph = NULL;
                sq_ar_check(s, e, "destroy AR graph");
            }
            if (s->blas && !s->graph) {
                int e = sq_blas_destroy(s->blas);
                if (!e)
                    s->blas = NULL;
                sq_ar_check(s, e, "destroy session cuBLAS");
            }
            if (s->arena && !s->graph && !s->blas) {
                int e = b->release(b->context, s->arena);
                if (!e)
                    s->arena = NULL;
                sq_ar_check(s, e, "free session arena");
            }
            if (s->stream && !s->arena && !s->graph && !s->blas) {
                int e = b->stream_destroy(b->context, s->stream);
                if (!e)
                    s->stream = NULL;
                sq_ar_check(s, e, "destroy session stream");
            }
        }
        (void)leave(s, &scope);
    }
    sq_status rc = result(s);
    if (s->arena || s->stream || s->graph || s->blas) {
        unlock(s);
        return rc;
    }
    if (!image_lock(s->weights)) {
        unlock(s);
        return sq_error(SQ_BUSY, "weight operation active; retry session destroy");
    }
    if (s->retained)
        s->weights->session_users--;
    image_unlock(s->weights);
    sq_options allocator = s->allocator;
    *handle = NULL;
    allocator.release(allocator.context, s);
    atomic_flag_clear_explicit(&live_session, memory_order_release);
    return rc;
}
