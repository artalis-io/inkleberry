/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Synchronous AR/fixed/adaptive DSpark session with opt-in prefix reuse. */
#include "engine_internal.h"
#include "checked_prefill.h"
#include <stdio.h>
#include <string.h>
static atomic_flag live_session = ATOMIC_FLAG_INIT;

void ifn_ar_check(ifn_session *s, int rc, const char *op) {
    if (rc && s->status == IFN_OK) {
        s->status = (rc == 2 || rc == 10003) ? IFN_OUT_OF_MEMORY
                    : rc == 801              ? IFN_UNSUPPORTED
                                             : IFN_GPU_ERROR;
        (void)snprintf(s->failure, sizeof s->failure, "%s: CUDA status %d", op, rc);
    }
}

static ifn_status result(ifn_session *s) {
    return ifn_error(s->status, s->status == IFN_OK ? "" : s->failure);
}

static bool lock(ifn_session *s) {
    return s && !atomic_flag_test_and_set_explicit(&s->busy, memory_order_acquire);
}

static void unlock(ifn_session *s) {
    atomic_flag_clear_explicit(&s->busy, memory_order_release);
}

static void sync(ifn_session *s) {
    if (s->status == IFN_OK)
        ifn_ar_check(s, s->weights->backend.sync(s->weights->backend.context, s->stream),
                     "synchronize session");
}

static void copy(ifn_session *s, void *dst, const void *src, size_t bytes, int kind) {
    if (s->status == IFN_OK)
        ifn_ar_check(
            s,
            s->weights->backend.copy(s->weights->backend.context, dst, src, bytes, kind, s->stream),
            "session copy");
}

static ifn_device_scope enter(ifn_session *s) {
    ifn_device_scope scope = {-1, NULL};
    s->status = IFN_OK;
    s->failure[0] = 0;
    ifn_ar_check(
        s,
        s->weights->backend.enter(s->weights->backend.context, s->weights->config.device, &scope),
        "select session GPU");
    return scope;
}

static ifn_status leave(ifn_session *s, const ifn_device_scope *scope) {
    if (scope->device >= 0) {
        int restore = s->weights->backend.leave(s->weights->backend.context, scope);
        /* A normal cancellation must not hide context-restoration failure. */
        if (restore && s->status == IFN_CANCELLED)
            s->status = IFN_OK;
        ifn_ar_check(s, restore, "restore caller context");
    }
    if (s->status != IFN_OK && s->status != IFN_CANCELLED) {
        ifn_prefix_invalidate(s);
        s->poisoned = true;
    }
    return result(s);
}

static void reset_device(ifn_session *s) {
    s->position = 0;
    s->use_mtp = s->verifying = false;
    if (s->status != IFN_OK)
        return;
    ifn_ar_check(s, squid_cuda_memset_async(s->d_pos, 0, 4, s->stream), "reset position");
    for (unsigned i = 0; i < 48 && s->status == IFN_OK; i++) {
        ifn_ar_check(s, squid_cuda_memset_async(s->k_ring[i], 0, 16 * 1024 * 4, s->stream),
                     "reset K ring");
        ifn_ar_check(s, squid_cuda_memset_async(s->v_ring[i], 0, 16 * 1024 * 4, s->stream),
                     "reset V ring");
        ifn_ar_check(s, squid_cuda_memset_async(s->attn_ring[i], 0, 16 * 2304 * 4, s->stream),
                     "reset attention ring");
        ifn_ar_check(s, squid_cuda_memset_async(s->mlp_ring[i], 0, 16 * 2304 * 4, s->stream),
                     "reset MLP ring");
    }
    if (s->ds_f_ring) {
        ifn_ar_check(s, squid_cuda_memset_async(s->ds_f_ring, 0, 16 * 2304 * 4, s->stream),
                     "reset DSpark ring");
        ifn_ar_check(s, squid_cuda_memset_async(s->d_draft_is_delta, 0, 4, s->stream),
                     "reset draft mode");
    }
    CSampleConfig cfg = {0};
    cfg.top_p = 1;
    CLoopCDConfig loop = {0};
    loop.ref_pass = 1;
    copy(s, s->d_sample_cfg, &cfg, sizeof cfg, 0);
    copy(s, s->d_loopcd_cfg, &loop, sizeof loop, 0);
    sync(s);
}

static bool image_lock(ifn_gpu_weights *w) {
    return !atomic_flag_test_and_set_explicit(&w->busy, memory_order_acquire);
}

static void image_unlock(ifn_gpu_weights *w) {
    atomic_flag_clear_explicit(&w->busy, memory_order_release);
}

ifn_status ifn_session_create(ifn_gpu_weights *w, const ifn_session_config *input,
                              const ifn_options *options, ifn_session **out) {
    if (out)
        *out = NULL;
    if (!w || !out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing session arguments");
    ifn_session_config config = IFN_SESSION_CONFIG_INIT;
    if (input) {
        if (input->struct_size < sizeof *input || input->api_version != IFN_API_VERSION)
            return ifn_error(IFN_INVALID_ARGUMENT,
                             "unsupported session configuration size/version");
        config = *input;
    }
    if (config.context_tokens < 64 || config.context_tokens > 1048576 || config.kv_format > 2 ||
        config.use_graphs > 1 || config.reserved)
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid session configuration");
    ifn_options allocator;
    ifn_status rc = ifn_options_resolve(options, &allocator);
    if (rc != IFN_OK)
        return rc;
    if (!image_lock(w))
        return ifn_error(IFN_BUSY, "weight operation already active");
    if (!w->ready || atomic_flag_test_and_set_explicit(&live_session, memory_order_acquire)) {
        image_unlock(w);
        return ifn_error(IFN_BUSY, "image not ready or another session exists");
    }
    ifn_session *s = allocator.allocate(allocator.context, sizeof *s);
    if (!s) {
        atomic_flag_clear_explicit(&live_session, memory_order_release);
        image_unlock(w);
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate session bookkeeping");
    }
    memset(s, 0, sizeof *s);
    s->allocator = allocator;
    s->config = config;
    s->weights = w;
    s->retained = true;
    atomic_flag_clear(&s->busy);
    atomic_init(&s->cancelled, false);
    w->session_users++;
    rc = ifn_ar_bind(w, &s->model);
    image_unlock(w);
    *out = s;
    if (rc != IFN_OK) {
        s->poisoned = true;
        return rc;
    }
    if (w->config.with_mtp) {
        size_t bytes = ifn_adaptive_bytes();
        if (!bytes || bytes > 4096) {
            s->poisoned = true;
            return ifn_error(IFN_UNSUPPORTED, "unsupported adaptive state size");
        }
        s->adaptive_state = allocator.allocate(allocator.context, bytes);
        if (!s->adaptive_state) {
            s->poisoned = true;
            return ifn_error(IFN_OUT_OF_MEMORY, "allocate adaptive graph bookkeeping");
        }
        ifn_adaptive_init(s->adaptive_state);
    }
    size_t groups = squid_sizeof_expert_groups();
    if (!groups || groups > 1048576) {
        s->poisoned = true;
        return ifn_error(IFN_UNSUPPORTED, "unsupported expert group workspace");
    }
    ifn_ar_layout(s, groups);
    s->capacity = s->offset;
    if (s->status != IFN_OK) {
        s->poisoned = true;
        return ifn_error(s->status, "session layout overflow");
    }
    ifn_device_scope scope = enter(s);
    const ifn_weight_backend *b = &w->backend;
    size_t available = 0;
    if (s->status == IFN_OK)
        ifn_ar_check(s, b->available(b->context, &available), "query session memory");
    if (s->status == IFN_OK && s->capacity > available) {
        s->status = IFN_OUT_OF_MEMORY;
        (void)snprintf(s->failure, sizeof s->failure, "insufficient free GPU memory for session");
    }
    if (s->status == IFN_OK)
        ifn_ar_check(s, b->stream_create(b->context, &s->stream), "create session stream");
    if (s->status == IFN_OK)
        ifn_ar_check(s, b->allocate(b->context, &s->arena, s->capacity), "allocate session arena");
    if (s->status == IFN_OK && w->config.with_mtp) {
        ifn_ar_check(s, squid_cuda_malloc_host_pinned(&s->pinned, 17 * 4, 0),
                     "allocate adaptive pinned outputs");
        if (s->pinned) {
            s->h_accepted = s->pinned;
            s->h_count = (int32_t *)s->pinned + 8;
            s->h_width = (int32_t *)s->pinned + 9;
        }
        if (s->status == IFN_OK)
            ifn_ar_check(s, squid_cuda_event_create(&s->done_event, 1),
                         "create adaptive completion event");
    }
    if (s->status == IFN_OK) {
        ifn_ar_layout(s, groups);
        if (s->offset != s->capacity)
            s->status = IFN_LIMIT;
        reset_device(s);
    }
    rc = leave(s, &scope);
    s->ready = rc == IFN_OK;
    return rc;
}

static ifn_status begin(ifn_session *s, ifn_device_scope *scope) {
    if (!s)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing session");
    if (!lock(s))
        return ifn_error(IFN_BUSY, "session operation already active");
    if (!s->ready || s->poisoned) {
        unlock(s);
        return ifn_error(IFN_GPU_ERROR, "session is not ready; destroy it");
    }
    atomic_store_explicit(&s->cancelled, false, memory_order_release);
    *scope = enter(s);
    return IFN_OK;
}

static ifn_status end(ifn_session *s, const ifn_device_scope *scope) {
    ifn_status rc = leave(s, scope);
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

static bool cancelled(ifn_session *s) {
    return atomic_load_explicit(&s->cancelled, memory_order_acquire) ||
           (s->request_control && s->request_control->cancelled(s->request_control->context));
}

static void run(ifn_session *s, const int32_t *tokens, int32_t n, bool lm) {
    copy(s, s->d_tokens, tokens, (size_t)n * 4, 0);
    copy(s, s->d_pos, &s->position, 4, 0);
    if (s->status == IFN_OK) {
        ifn_ar_forward(s, n, lm);
        ifn_ar_check(s, ifn_cuda_launch_status(), "target forward launch");
        sync(s);
        if (s->status == IFN_OK)
            s->position += n;
    }
}

static void cancel_result(ifn_session *s) {
    /* Cancellation can arrive during the final snapshot after prefix_finish's
     * flag read. Invalidate here too, before reset destroys live local state. */
    ifn_prefix_invalidate(s);
    if (s->status == IFN_OK) {
        reset_device(s);
        if (s->status == IFN_OK) {
            s->status = IFN_CANCELLED;
            (void)snprintf(s->failure, sizeof s->failure, "generation cancelled");
        }
    }
}

ifn_status ifn_session_reset(ifn_session *s) {
    ifn_device_scope scope;
    ifn_status rc = begin(s, &scope);
    if (rc != IFN_OK)
        return rc;
    ifn_prefix_invalidate(s);
    s->prefix_info = (ifn_prefix_info)IFN_PREFIX_INFO_INIT;
    reset_device(s);
    ifn_lookup_reset(&s->lookup);
    s->decode_info = (ifn_decode_info)IFN_DECODE_INFO_INIT;
    return end(s, &scope);
}

ifn_status ifn_session_cancel(ifn_session *s) {
    if (!s)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing session");
    atomic_store_explicit(&s->cancelled, true, memory_order_release);
    return ifn_error(IFN_OK, "");
}

ifn_status ifn_session_logits(ifn_session *s, const int32_t *tokens, size_t n, uint32_t reset,
                              float *out, size_t capacity) {
    if (!s || n > 8 || reset > 1 || !out || capacity < 200058 || !tokens_valid(tokens, n))
        return ifn_error(IFN_INVALID_ARGUMENT,
                         "logits needs 1..8 valid IDs and 200058 output floats");
    ifn_device_scope scope;
    ifn_status rc = begin(s, &scope);
    if (rc != IFN_OK)
        return rc;
    if (n > s->config.context_tokens - (reset ? 0 : (uint32_t)s->position)) {
        ifn_status restore = end(s, &scope);
        return restore == IFN_OK ? ifn_error(IFN_LIMIT, "logits context limit") : restore;
    }
    ifn_prefix_invalidate(s);
    s->use_mtp = s->verifying = false;
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

ifn_status ifn_session_generate(ifn_session *s, const int32_t *tokens, size_t n, size_t max_new,
                                int32_t *out, size_t capacity, ifn_token_callback callback,
                                void *context, ifn_generate_stats *stats) {
    if (!stats)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing generation stats");
    ifn_decode_stats detail = {0};
    ifn_status rc = ifn_session_generate_ex(s, tokens, n, max_new, NULL, out, capacity, callback,
                                            context, &detail);
    *stats = detail.generation;
    return rc;
}

static bool emit(ifn_session *s, int32_t token, int32_t *out, ifn_token_callback callback,
                 void *context, ifn_generate_stats *stats) {
    if (token < 0 || token >= 200058) {
        ifn_ar_check(s, 1, "invalid generated token");
        return false;
    }
    if (out)
        out[stats->generated_tokens] = token;
    stats->generated_tokens++;
    stats->last_token = token;
    if (callback && !callback(context, token)) {
        stats->finish_reason = IFN_FINISH_CALLBACK;
        return false;
    }
    if (token == 200006) {
        stats->finish_reason = IFN_FINISH_EOS;
        return false;
    }
    return true;
}

static ifn_status generate_impl(ifn_session *s, const int32_t *tokens, size_t n, size_t max_new,
                                const ifn_decode_options *input, int32_t *out, size_t capacity,
                                ifn_token_callback callback, void *context,
                                ifn_decode_stats *detail, const ifn_run_control *control) {
    if (detail)
        memset(detail, 0, sizeof *detail);
    ifn_decode_options options;
    ifn_status rc = ifn_decode_resolve(input, &options);
    if (rc != IFN_OK)
        return rc;
    if (!s || !max_new || (!control && (max_new > capacity || !out)) || !detail ||
        n > s->config.context_tokens || max_new > s->config.context_tokens - n ||
        !tokens_valid(tokens, n))
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid generation bounds or token IDs");
    if (options.adaptive && !s->config.use_graphs)
        return ifn_error(IFN_UNSUPPORTED, "adaptive decoding requires CUDA graphs");
    if (options.draft_tokens && !s->model.dspark.loaded)
        return ifn_error(IFN_UNSUPPORTED, "DSpark requires MTP weights; load them explicitly");
    ifn_device_scope scope;
    rc = begin(s, &scope);
    if (rc != IFN_OK)
        return rc;
    s->request_control = control;
    if (cancelled(s)) {
        ifn_prefix_invalidate(s);
        if (s->status == IFN_OK) {
            s->status = IFN_CANCELLED;
            (void)snprintf(s->failure, sizeof s->failure, "request cancelled before generation");
        }
        s->request_control = NULL;
        return end(s, &scope);
    }
    uint32_t start_pos =
        ifn_prefix_begin(s, tokens, (uint32_t)n, (uint32_t)max_new, options.draft_tokens != 0);
    if (!start_pos)
        reset_device(s);
    s->decode_info = (ifn_decode_info)IFN_DECODE_INFO_INIT;
    s->use_mtp = options.draft_tokens != 0;
    CSampleConfig sample = {0};
    sample.inv_temp = options.temperature >= 1e-4f ? 1.0f / options.temperature : 0;
    sample.top_p = options.top_p;
    sample.top_k = options.top_k;
    sample.seed = options.seed;
    copy(s, s->d_sample_cfg, &sample, sizeof sample, 0);
    ifn_generate_stats *stats = &detail->generation;
    stats->prompt_tokens = (uint32_t)n;
    stats->last_token = -1;
    uint32_t targets[2] = {0, 0};
    unsigned slots[2] = {0, 1};
    if (s->prefix.enabled && !s->prefix.shield_tokens) {
        uint32_t hint = options.system_prefix_tokens;
        if (hint >= 128 && hint > start_pos && hint < n)
            targets[0] = hint;
        uint32_t p2k = ((uint32_t)n - 1) / 2048 * 2048;
        if (p2k >= 2048 && p2k > start_pos && p2k < n && p2k != targets[0])
            targets[1] = p2k;
        if (!targets[0] || (targets[1] && targets[1] < targets[0])) {
            uint32_t t = targets[0];
            targets[0] = targets[1];
            targets[1] = t;
            slots[0] = 1;
            slots[1] = 0;
        }
    }
    size_t offset = start_pos;
    for (unsigned target = 0; target < 2 && s->status == IFN_OK && !cancelled(s); target++) {
        if (!targets[target])
            continue;
        while (offset < targets[target] && s->status == IFN_OK && !cancelled(s)) {
            size_t remain = targets[target] - offset;
            if (remain >= 32) {
                int32_t chunk = (int32_t)(remain > 2048 ? 2048 : remain & ~(size_t)7);
                ifn_ar_prefill(s, tokens + offset, chunk);
                offset += (size_t)chunk;
            } else {
                int32_t chunk = (int32_t)(remain > 8 ? 8 : remain);
                run(s, tokens + offset, chunk, false);
                offset += (size_t)chunk;
            }
        }
        if (!cancelled(s))
            ifn_prefix_save(s, slots[target], (uint32_t)offset);
    }
    int32_t last_batch = 0;
    for (size_t i = offset; i < n && s->status == IFN_OK && !cancelled(s);) {
        if (n - i > 32) {
            size_t avail = (n - i - 1) & ~(size_t)7;
            int32_t chunk = (int32_t)(avail > 2048 ? 2048 : avail);
            ifn_ar_prefill(s, tokens + i, chunk);
            i += (size_t)chunk;
        } else {
            last_batch = (int32_t)(n - i > 8 ? 8 : n - i);
            run(s, tokens + i, last_batch, i + (size_t)last_batch == n);
            i += (size_t)last_batch;
        }
    }
    if (!cancelled(s))
        ifn_prefix_prompt(s, tokens, (uint32_t)n, options.draft_tokens != 0);
    int32_t token = -1;
    if (!cancelled(s))
        copy(s, &token, s->d_next_tokens + (last_batch ? last_batch - 1 : 0), 4, 1);
    sync(s);
    bool keep =
        s->status == IFN_OK && !cancelled(s) && emit(s, token, out, callback, context, stats);
    uint32_t drafts = options.draft_tokens;
    if (keep && drafts && stats->generated_tokens < max_new) {
        copy(s, s->d_anchor, &token, 4, 0);
        copy(s, s->d_pos, &s->position, 4, 0);
        if (options.prompt_lookup) {
            ifn_lookup_reset(&s->lookup);
            size_t start = n > 8192 ? n - 8192 : 0;
            for (size_t i = start; i < n; i++)
                ifn_lookup_push(&s->lookup, tokens[i]);
            ifn_lookup_push(&s->lookup, token);
        }
        while (stats->generated_tokens < max_new && s->status == IFN_OK && !cancelled(s) &&
               (uint32_t)s->position + drafts + 1 <= s->config.context_tokens) {
            int32_t accepted[8] = {0}, count = 0, width = (int32_t)drafts + 1;
            bool lookup = false;
            if (options.adaptive) {
                lookup = options.prompt_lookup &&
                         ifn_lookup_match(&s->lookup, (int32_t *)s->pinned + 10);
                ifn_adaptive_step(s, &options, lookup);
                if (s->status == IFN_OK) {
                    count = *s->h_count;
                    width = lookup ? 8 : *s->h_width;
                    memcpy(accepted, s->h_accepted, sizeof accepted);
                }
            } else {
                ifn_spec_step(s, drafts);
                copy(s, &count, s->d_num_acc, 4, 1);
                copy(s, accepted, s->d_accepted, (drafts + 1) * 4, 1);
                sync(s);
            }
            if (s->status != IFN_OK)
                break;
            if ((options.adaptive && width != 2 && width != 4 && width != 8) || count < 1 ||
                count > width) {
                ifn_ar_check(s, 1, "invalid accepted token count");
                break;
            }
            detail->speculative_steps++;
            detail->total_drafts += (uint32_t)width - 1;
            if (options.adaptive) {
                if (width == 2)
                    s->decode_info.width2++;
                else if (width == 4)
                    s->decode_info.width4++;
                else
                    s->decode_info.width8++;
                if (lookup)
                    s->decode_info.lookup_steps++;
            }
            detail->accepted_drafts += (uint32_t)count - 1;
            for (int32_t i = 0; i < count; i++)
                ifn_prefix_feed(s, i == 0 ? token : accepted[i - 1]);
            s->position += count;
            for (int32_t i = 0; i < count && stats->generated_tokens < max_new && !cancelled(s);
                 i++) {
                token = accepted[i];
                if (options.prompt_lookup)
                    ifn_lookup_push(&s->lookup, token);
                keep = emit(s, token, out, callback, context, stats);
                if (!keep)
                    break;
            }
            if (!keep)
                break;
        }
    }
    ifn_adaptive_drain(s);
    if (keep && !drafts && stats->generated_tokens < max_new && s->status == IFN_OK &&
        !cancelled(s) && s->config.use_graphs && !s->graph) {
        ifn_ar_check(s, squid_cuda_graph_begin(s->stream), "begin AR graph");
        if (s->status == IFN_OK) {
            ifn_ar_forward(s, 1, true);
            ifn_ar_check(s, ifn_cuda_launch_status(), "capture target forward");
            ifn_ar_check(s, squid_cuda_graph_end_instantiate(s->stream, &s->graph),
                         "instantiate AR graph");
        }
    }
    /* Speculation uses eager B=1 at the context boundary, exactly as Zig. */
    while (keep && stats->generated_tokens < max_new && s->status == IFN_OK && !cancelled(s)) {
        int32_t fed = token;
        if (s->config.use_graphs && !drafts) {
            copy(s, s->d_tokens, &token, 4, 0);
            copy(s, s->d_pos, &s->position, 4, 0);
            if (s->status == IFN_OK)
                ifn_ar_check(s, squid_cuda_graph_launch(s->graph, s->stream), "launch AR graph");
            sync(s);
            if (s->status == IFN_OK)
                s->position++;
        } else
            run(s, &token, 1, true);
        ifn_prefix_feed(s, fed);
        copy(s, &token, s->d_next_tokens, 4, 1);
        sync(s);
        if (s->status == IFN_OK && !cancelled(s))
            keep = emit(s, token, out, callback, context, stats);
    }
    ifn_prefix_finish(s, cancelled(s));
    if (cancelled(s)) {
        stats->finish_reason = IFN_FINISH_CANCELLED;
        cancel_result(s);
    }
    s->request_control = NULL;
    return end(s, &scope);
}

ifn_status ifn_session_generate_ex(ifn_session *s, const int32_t *tokens, size_t n, size_t max_new,
                                   const ifn_decode_options *input, int32_t *out, size_t capacity,
                                   ifn_token_callback callback, void *context,
                                   ifn_decode_stats *detail) {
    return generate_impl(s, tokens, n, max_new, input, out, capacity, callback, context, detail,
                         NULL);
}

ifn_status ifn_session_generate_controlled(ifn_session *s, const int32_t *tokens, size_t n,
                                           size_t max_new, const ifn_decode_options *input,
                                           ifn_token_callback callback, void *context,
                                           ifn_decode_stats *detail,
                                           const ifn_run_control *control) {
    if (!control || !control->cancelled)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing request cancellation control");
    return generate_impl(s, tokens, n, max_new, input, NULL, 0, callback, context, detail, control);
}

ifn_status ifn_session_owned_plan(ifn_gpu_weights *weights, const ifn_session_config *config,
                                  const ifn_options *options, uint64_t *bytes) {
    if (!weights || !config || !options || !bytes)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing session plan arguments");
    *bytes = 0;
    ifn_session *s = options->allocate(options->context, sizeof *s);
    if (!s)
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate session plan");
    memset(s, 0, sizeof *s);
    s->weights = weights;
    s->config = *config;
    size_t groups = squid_sizeof_expert_groups();
    if (!groups || groups > 1048576) {
        options->release(options->context, s);
        return ifn_error(IFN_UNSUPPORTED, "unsupported expert group size");
    }
    ifn_ar_layout(s, groups);
    ifn_status rc = s->status;
    if (rc == IFN_OK)
        *bytes = s->offset;
    options->release(options->context, s);
    return rc;
}

static bool has_graphs(const ifn_session *s) {
    if (s->graph || s->draft_graph || s->lookup_graph ||
        (s->adaptive_state && ifn_adaptive_live(s->adaptive_state)))
        return true;
    for (unsigned i = 1; i < 8; i++)
        if (s->spec_graph[i])
            return true;
    return false;
}

ifn_status ifn_session_prefix_cache(ifn_session *s, uint32_t enabled) {
    if (!s || enabled > 1)
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid prefix cache configuration");
    if (!lock(s))
        return ifn_error(IFN_BUSY, "session operation already active");
    ifn_status rc = IFN_OK;
    if (!s->ready || s->poisoned)
        rc = ifn_error(IFN_GPU_ERROR, "session is not ready");
    else if (!enabled)
        ifn_prefix_free(s);
    else {
        s->status = IFN_OK;
        s->failure[0] = 0;
        rc = ifn_prefix_allocate(s);
    }
    if (rc == IFN_OK) {
        s->prefix_info = (ifn_prefix_info)IFN_PREFIX_INFO_INIT;
        rc = ifn_error(IFN_OK, "");
    }
    unlock(s);
    return rc;
}

ifn_status ifn_session_prefix_info(ifn_session *s, ifn_prefix_info *out) {
    if (!s || !out || out->struct_size < sizeof *out || out->api_version != IFN_API_VERSION)
        return ifn_error(IFN_INVALID_ARGUMENT, "initialize prefix info size/version");
    if (!lock(s))
        return ifn_error(IFN_BUSY, "session operation already active");
    *out = s->prefix_info;
    out->struct_size = sizeof *out;
    out->api_version = IFN_API_VERSION;
    out->host_bytes = s->prefix.host_bytes;
    unlock(s);
    return ifn_error(IFN_OK, "");
}

ifn_status ifn_session_decode_info(ifn_session *s, ifn_decode_info *out) {
    if (!s || !out || out->struct_size < sizeof *out || out->api_version != IFN_API_VERSION)
        return ifn_error(IFN_INVALID_ARGUMENT, "initialize decode info size/version");
    if (!lock(s))
        return ifn_error(IFN_BUSY, "session operation already active");
    *out = s->decode_info;
    out->struct_size = sizeof *out;
    out->api_version = IFN_API_VERSION;
    unlock(s);
    return ifn_error(IFN_OK, "");
}

ifn_status ifn_session_destroy(ifn_session **handle) {
    if (!handle)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing session handle pointer");
    ifn_session *s = *handle;
    if (!s)
        return ifn_error(IFN_OK, "");
    if (!lock(s))
        return ifn_error(IFN_BUSY, "session operation already active");
    s->ready = false;
    if (s->adaptive_state && !ifn_adaptive_live(s->adaptive_state)) {
        s->allocator.release(s->allocator.context, s->adaptive_state);
        s->adaptive_state = NULL;
    }
    const ifn_weight_backend *b = &s->weights->backend;
    s->status = IFN_OK;
    s->failure[0] = 0;
    if (s->arena || s->stream || has_graphs(s) || s->blas || s->adaptive_state || s->pinned ||
        s->done_event) {
        ifn_device_scope scope = enter(s);
        if (s->status == IFN_OK) {
            if (s->stream)
                sync(s);
            if (s->adaptive_state) {
                int e = ifn_adaptive_cleanup(s->adaptive_state);
                if (!e) {
                    s->allocator.release(s->allocator.context, s->adaptive_state);
                    s->adaptive_state = NULL;
                }
                ifn_ar_check(s, e, "destroy conditional graph state");
            }
            void **extra[2] = {&s->draft_graph, &s->lookup_graph};
            for (unsigned i = 0; i < 2; i++)
                if (*extra[i]) {
                    int e = squid_cuda_graph_destroy(*extra[i]);
                    if (!e)
                        *extra[i] = NULL;
                    ifn_ar_check(s, e, "destroy adaptive component graph");
                }
            for (unsigned i = 1; i < 8; i++)
                if (s->spec_graph[i]) {
                    int e = squid_cuda_graph_destroy(s->spec_graph[i]);
                    if (!e)
                        s->spec_graph[i] = NULL;
                    ifn_ar_check(s, e, "destroy DSpark graph");
                }
            if (s->graph) {
                int e = squid_cuda_graph_destroy(s->graph);
                if (!e)
                    s->graph = NULL;
                ifn_ar_check(s, e, "destroy AR graph");
            }
            if (s->blas && !has_graphs(s)) {
                int e = ifn_blas_destroy(s->blas);
                if (!e)
                    s->blas = NULL;
                ifn_ar_check(s, e, "destroy session cuBLAS");
            }
            if (s->arena && !has_graphs(s) && !s->blas) {
                int e = b->release(b->context, s->arena);
                if (!e)
                    s->arena = NULL;
                ifn_ar_check(s, e, "free session arena");
            }
            if (s->pinned && !has_graphs(s)) {
                int e = squid_cuda_free_host(s->pinned);
                if (!e)
                    s->pinned = NULL;
                ifn_ar_check(s, e, "free adaptive pinned outputs");
            }
            if (s->done_event && !has_graphs(s)) {
                int e = squid_cuda_event_destroy(s->done_event);
                if (!e)
                    s->done_event = NULL;
                ifn_ar_check(s, e, "destroy adaptive completion event");
            }
            if (s->stream && !s->arena && !has_graphs(s) && !s->blas && !s->pinned &&
                !s->done_event) {
                int e = b->stream_destroy(b->context, s->stream);
                if (!e)
                    s->stream = NULL;
                ifn_ar_check(s, e, "destroy session stream");
            }
        }
        (void)leave(s, &scope);
    }
    ifn_status rc = result(s);
    if (s->arena || s->stream || has_graphs(s) || s->blas || s->adaptive_state || s->pinned ||
        s->done_event) {
        unlock(s);
        return rc;
    }
    if (!image_lock(s->weights)) {
        unlock(s);
        return ifn_error(IFN_BUSY, "weight operation active; retry session destroy");
    }
    if (s->retained)
        s->weights->session_users--;
    image_unlock(s->weights);
    ifn_prefix_free(s);
    ifn_options allocator = s->allocator;
    *handle = NULL;
    allocator.release(allocator.context, s);
    atomic_flag_clear_explicit(&live_session, memory_order_release);
    return rc;
}
