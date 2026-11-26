/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "infer_internal.h"
#include <math.h>

static bool ifn_infer_request_local_cancelled(void *context) {
    ifn_infer_request *r = context;
    return (atomic_load_explicit(&r->control, memory_order_acquire) & IFN_INFER_CANCEL) != 0;
}

static ifn_status ifn_infer_request_local_array_valid(ifn_infer_array a, size_t element,
                                                      size_t maximum) {
    if (a.reserved || a.count > maximum || (a.count && (!a.data || a.stride < element)) ||
        (a.count && a.count > SIZE_MAX / a.stride))
        return IFN_INVALID_ARGUMENT;
    if (!a.count && (a.data || a.stride))
        return IFN_INVALID_ARGUMENT;
    return IFN_OK;
}

static ifn_status ifn_infer_request_local_request_options(ifn_infer_request *r,
                                                          const ifn_infer_request_config_v1 *c,
                                                          ifn_chat_options *chat) {
    ifn_status rc = ifn_infer_input(c, sizeof *c);
    if (rc)
        return rc;
    uint64_t supported = IFN_INFER_FEATURES;
    if (!r->session->model->weights_config.with_mtp)
        supported &= ~IFN_INFER_CAP_SPECULATION;
    if (c->required_features & ~supported)
        return IFN_UNSUPPORTED;
    if (!c->max_output_tokens || c->max_output_tokens > r->session->config.context_tokens ||
        !isfinite(c->temperature) || c->temperature < 0 || !isfinite(c->top_p) || c->top_p <= 0 ||
        c->top_p > 1 || c->top_k < 0 || c->top_k > 200058 || c->output_kind < IFN_OUTPUT_TEXT ||
        c->output_kind > IFN_OUTPUT_DISCARD || c->max_event_bytes < 4 ||
        c->max_event_bytes > 65536 || c->reserved ||
        (c->output_flags & ~IFN_INFER_OUTPUT_REASONING) || c->history_policy > 1 ||
        !ifn_infer_span(c->cache_scope, sizeof r->scope))
        return IFN_INVALID_ARGUMENT;
    rc = ifn_infer_request_local_array_valid(c->extensions, sizeof(ifn_infer_extension_v1), 16);
    if (rc)
        return rc;
    if (c->extensions.count > 1)
        return IFN_UNSUPPORTED;
    r->decode = (ifn_decode_options)IFN_DECODE_OPTIONS_INIT;
    r->decode.temperature = c->temperature;
    r->decode.top_p = c->top_p;
    r->decode.top_k = c->top_k;
    r->decode.seed = c->seed;
    *chat = (ifn_chat_options)IFN_CHAT_OPTIONS_INIT;
    chat->keep_history_thinking = c->history_policy;
    for (size_t i = 0; i < c->extensions.count; i++) {
        const unsigned char *p =
            (const unsigned char *)c->extensions.data + i * c->extensions.stride;
        ifn_infer_extension_v1 ext;
        memcpy(&ext, p, sizeof ext);
        /* Read headers/tails as bytes, including padded array elements. */
        if (ext.hdr.struct_size > c->extensions.stride)
            return IFN_INVALID_ARGUMENT;
        rc = ifn_infer_input(p, sizeof ext);
        if (rc)
            return rc;
        if (ext.id != IFN_INFER_EXT_INKLING_REQUEST || ext.version != 1 ||
            ext.payload_bytes < sizeof(ifn_infer_inkling_request_v1) ||
            ext.payload_bytes > IFN_INFER_RECORD_MAX_BYTES || !ext.payload)
            return IFN_UNSUPPORTED;
        ifn_infer_inkling_request_v1 v;
        memcpy(&v, ext.payload, sizeof v);
        if (v.hdr.struct_size > ext.payload_bytes)
            return IFN_INVALID_ARGUMENT;
        rc = ifn_infer_input(ext.payload, sizeof v);
        if (rc)
            return rc;
        if (v.draft_tokens > 7 || v.adaptive > 1 || v.prompt_lookup > 1 ||
            v.thinking_effort > IFN_THINK_MAX || v.enable_thinking > 1 ||
            !isfinite(v.spec_threshold) ||
            (v.adaptive && (v.spec_threshold <= 0 || v.spec_threshold > 1)))
            return IFN_INVALID_ARGUMENT;
        if ((v.draft_tokens && !r->session->model->weights_config.with_mtp) ||
            (v.adaptive && (v.draft_tokens != 7 || !r->session->config.use_graphs)) ||
            (v.prompt_lookup && !v.adaptive))
            return IFN_UNSUPPORTED;
        r->decode.draft_tokens = v.draft_tokens;
        r->decode.adaptive = v.adaptive;
        r->decode.prompt_lookup = v.prompt_lookup;
        if (v.adaptive)
            r->decode.spec_threshold = v.spec_threshold;
        chat->effort = v.thinking_effort;
        chat->enable_thinking = v.enable_thinking;
    }
    r->config = *c;
    /* Borrowed configuration pointers never survive creation. */
    r->config.extensions = (ifn_infer_array){0};
    r->config.stops = (ifn_infer_array){0};
    r->config.cache_scope = (ifn_infer_bytes){0};
    r->scope_bytes = (size_t)c->cache_scope.bytes;
    if (r->scope_bytes)
        memcpy(r->scope, c->cache_scope.data, r->scope_bytes);
    return IFN_OK;
}

static ifn_status ifn_infer_request_local_encode(ifn_infer_request *r, const ifn_infer_input_v1 *in,
                                                 const ifn_chat_options *chat) {
    ifn_status rc = ifn_infer_input(in, sizeof *in);
    if (rc)
        return rc;
    if (in->reserved || !ifn_infer_span(in->text, IFN_INFER_MAX_INPUT))
        return IFN_INVALID_ARGUMENT;
    if (in->kind != IFN_INPUT_CHAT &&
        (chat->effort || chat->enable_thinking || chat->keep_history_thinking))
        return IFN_UNSUPPORTED;
    size_t capacity = r->session->config.context_tokens - (size_t)r->config.max_output_tokens;
    if (!capacity)
        return IFN_LIMIT;
    size_t bytes;
    if (!ifn_size_mul(capacity, sizeof(int32_t), &bytes))
        return IFN_LIMIT;
    r->tokens = ifn_infer_alloc(&r->allocator, bytes);
    if (!r->tokens)
        return IFN_OUT_OF_MEMORY;
    if (in->kind == IFN_INPUT_TOKENS) {
        if (in->text.data || in->text.bytes || ifn_infer_no_extensions(in->messages) ||
            !in->tokens || !in->token_count || in->token_count > capacity ||
            r->config.output_kind == IFN_OUTPUT_TEXT)
            return IFN_INVALID_ARGUMENT;
        for (size_t i = 0; i < in->token_count; i++) {
            if (in->tokens[i] < 0 || in->tokens[i] >= 200058)
                return IFN_INVALID_ARGUMENT;
            r->tokens[i] = in->tokens[i];
        }
        r->token_count = (size_t)in->token_count;
        return IFN_OK;
    }
    if (in->tokens || in->token_count)
        return IFN_INVALID_ARGUMENT;
    ifn_tokenizer_scratch *scratch = NULL;
    rc = ifn_tokenizer_scratch_create(&r->allocator, &scratch);
    if (rc)
        return rc;
    if (in->kind == IFN_INPUT_RAW_TEXT) {
        if (ifn_infer_no_extensions(in->messages))
            rc = IFN_INVALID_ARGUMENT;
        else
            rc = ifn_tokenizer_encode(r->session->model->tokenizer, in->text.data,
                                      (size_t)in->text.bytes, 0, scratch, r->tokens, capacity,
                                      &r->token_count);
    } else if (in->kind == IFN_INPUT_CHAT) {
        rc = ifn_infer_request_local_array_valid(in->messages, sizeof(ifn_infer_message_v1), 1024);
        if (in->text.data || in->text.bytes || !in->messages.count)
            rc = IFN_INVALID_ARGUMENT;
        ifn_chat_message *messages = NULL;
        if (!rc) {
            messages =
                ifn_infer_alloc(&r->allocator, (size_t)in->messages.count * sizeof *messages);
            if (!messages)
                rc = IFN_OUT_OF_MEMORY;
        }
        uint64_t total = 0;
        for (size_t i = 0; !rc && i < in->messages.count; i++) {
            ifn_infer_message_v1 m;
            const unsigned char *p =
                (const unsigned char *)in->messages.data + i * in->messages.stride;
            memcpy(&m, p, sizeof m);
            if (m.hdr.struct_size > in->messages.stride) {
                rc = IFN_INVALID_ARGUMENT;
                break;
            }
            rc = ifn_infer_input(p, sizeof m);
            if (rc)
                break;
            if (m.content_present > 1 || m.role < IFN_ROLE_SYSTEM || m.role > IFN_ROLE_ASSISTANT ||
                !ifn_infer_span(m.content, IFN_INFER_MAX_INPUT) ||
                !ifn_infer_span(m.reasoning, IFN_INFER_MAX_INPUT) || !ifn_infer_span(m.name, 128) ||
                (!m.content_present && (m.content.data || m.content.bytes))) {
                rc = IFN_INVALID_ARGUMENT;
                break;
            }
            total += m.content.bytes + m.reasoning.bytes + m.name.bytes;
            if (total > IFN_INFER_MAX_INPUT) {
                rc = IFN_LIMIT;
                break;
            }
            messages[i] = (ifn_chat_message){
                m.role - 1,
                0,
                m.content_present ? (m.content.data ? m.content.data : "") : NULL,
                (size_t)m.content.bytes,
                m.reasoning.data,
                (size_t)m.reasoning.bytes,
                m.name.data,
                (size_t)m.name.bytes};
        }
        if (!rc)
            rc = ifn_chat_encode(r->session->model->tokenizer, messages, (size_t)in->messages.count,
                                 chat, scratch, r->tokens, capacity, &r->token_count,
                                 &r->decode.system_prefix_tokens);
        ifn_infer_free(&r->allocator, messages);
    } else
        rc = IFN_UNSUPPORTED;
    ifn_tokenizer_scratch_destroy(scratch);
    return !rc && !r->token_count ? IFN_INVALID_ARGUMENT : rc;
}

static uint32_t ifn_infer_request_local_emit_event(ifn_infer_request *r, uint32_t kind,
                                                   const void *text, size_t bytes,
                                                   const int32_t *tokens, size_t count) {
    if (ifn_infer_request_local_cancelled(r))
        return IFN_CALLBACK_CANCEL;
    ifn_infer_event_v1 v = {0};
    v.hdr.struct_size = v.hdr.written_size = sizeof v;
    v.kind = kind;
    v.sequence = r->sequence++;
    v.generated_tokens_so_far = atomic_load_explicit(&r->generated, memory_order_relaxed);
    v.text = (ifn_infer_bytes){text, bytes};
    v.tokens = tokens;
    v.token_count = count;
    uint32_t action = r->emit(r->context, &v);
    if (kind == IFN_EVENT_CONTENT)
        r->result.content_bytes += bytes;
    if (kind == IFN_EVENT_REASONING)
        r->result.reasoning_bytes += bytes;
    uint64_t now = ifn_infer_now();
    if (now && r->started) {
        if (kind == IFN_EVENT_CONTENT && !(r->result.timing_valid & IFN_INFER_TIME_FIRST_CONTENT)) {
            r->result.first_content_ns = now - r->started;
            r->result.timing_valid |= IFN_INFER_TIME_FIRST_CONTENT;
        }
        if (kind == IFN_EVENT_REASONING &&
            !(r->result.timing_valid & IFN_INFER_TIME_FIRST_REASONING)) {
            r->result.first_reasoning_ns = now - r->started;
            r->result.timing_valid |= IFN_INFER_TIME_FIRST_REASONING;
        }
    }
    if (action > IFN_CALLBACK_CANCEL) {
        r->callback_status = IFN_INVALID_ARGUMENT;
        return IFN_CALLBACK_STOP;
    }
    if (action == IFN_CALLBACK_CANCEL) {
        uint32_t disposition;
        (void)ifn_infer_request_cancel(r, &disposition, NULL);
    }
    if (action)
        r->callback_action = action;
    return action;
}

static int ifn_infer_request_local_text_event(void *context, uint32_t channel, const char *p,
                                              size_t n) {
    ifn_infer_request *r = context;
    if (ifn_infer_request_local_cancelled(r))
        return 0;
    if (channel == IFN_STREAM_REASONING && !(r->config.output_flags & IFN_INFER_OUTPUT_REASONING))
        return 1;
    while (n) {
        size_t take = n > r->config.max_event_bytes ? r->config.max_event_bytes : n;
        if (take < n)
            while (take && (((unsigned char)p[take] & 0xc0) == 0x80))
                take--;
        if (!take) {
            r->callback_status = IFN_FORMAT;
            return 0;
        }
        if (ifn_infer_request_local_emit_event(
                r, channel == IFN_STREAM_REASONING ? IFN_EVENT_REASONING : IFN_EVENT_CONTENT, p,
                take, NULL, 0))
            return 0;
        p += take;
        n -= take;
    }
    return 1;
}

static int ifn_infer_request_local_token_event(void *context, int32_t token) {
    ifn_infer_request *r = context;
    atomic_fetch_add_explicit(&r->generated, 1, memory_order_relaxed);
    if (ifn_infer_request_local_cancelled(r))
        return 0;
    if (r->config.output_kind == IFN_OUTPUT_RAW_TOKENS)
        return ifn_infer_request_local_emit_event(r, IFN_EVENT_TOKENS, NULL, 0, &token, 1) ==
               IFN_CALLBACK_CONTINUE;
    if (r->config.output_kind == IFN_OUTPUT_DISCARD)
        return 1;
    ifn_status rc = ifn_stream_push_token(r->stream, token);
    if (rc) {
        r->callback_status = rc;
        return 0;
    }
    ifn_stream_info info = IFN_STREAM_INFO_INIT;
    rc = ifn_stream_get_info(r->stream, &info);
    if (rc) {
        r->callback_status = rc;
        return 0;
    }
    return info.finish_reason == IFN_STREAM_RUNNING;
}

static void ifn_infer_request_local_release_request(ifn_infer_request *r) {
    (void)ifn_stream_destroy(&r->stream);
    ifn_infer_free(&r->allocator, r->tokens);
    ifn_options a = r->allocator;
    ifn_infer_free(&a, r);
}

ifn_status ifn_infer_request_create(ifn_infer_session *s, const ifn_infer_input_v1 *input,
                                    const ifn_infer_request_config_v1 *config,
                                    ifn_infer_request **out, ifn_infer_error_v1 *e) {
    if (out)
        *out = NULL;
    if (!s || !out)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE,
                               "invalid request arguments");
    if (!ifn_infer_lock(&s->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_VALIDATE, "session busy");
    if (s->state != IFN_HANDLE_READY || s->children == UINT32_MAX) {
        ifn_infer_unlock(&s->busy);
        return ifn_infer_error(e, IFN_INFER_NOT_READY, IFN_STAGE_VALIDATE, "session not ready");
    }
    uint64_t start = ifn_infer_now();
    ifn_infer_request *r = ifn_infer_alloc(&s->allocator, sizeof *r);
    if (!r) {
        ifn_infer_unlock(&s->busy);
        return ifn_infer_error(e, IFN_OUT_OF_MEMORY, IFN_STAGE_VALIDATE, "allocate request");
    }
    memset(r, 0, sizeof *r);
    r->session = s;
    r->allocator = s->allocator;
    atomic_init(&r->control, 0);
    atomic_init(&r->phase, IFN_REQUEST_PREPARED);
    atomic_init(&r->running, false);
    atomic_init(&r->generated, 0);
    atomic_init(&r->processed, 0);
    IFN_INFER_INIT(r->terminal_error);
    ifn_chat_options chat;
    ifn_status rc = ifn_infer_request_local_request_options(r, config, &chat);
    if (!rc)
        rc = ifn_infer_request_local_encode(r, input, &chat);
    ifn_text_span stops[8];
    if (!rc)
        rc = ifn_infer_request_local_array_valid(config->stops, sizeof(ifn_infer_bytes), 8);
    if (!rc && config->output_kind != IFN_OUTPUT_TEXT && config->stops.count)
        rc = IFN_UNSUPPORTED;
    for (size_t i = 0; !rc && i < config->stops.count; i++) {
        ifn_infer_bytes span;
        memcpy(&span, (const unsigned char *)config->stops.data + i * config->stops.stride,
               sizeof span);
        if (!ifn_infer_span(span, 64) || !span.bytes) {
            rc = IFN_INVALID_ARGUMENT;
            break;
        }
        stops[i] = (ifn_text_span){span.data, (size_t)span.bytes};
    }
    if (!rc && config->output_kind == IFN_OUTPUT_TEXT) {
        ifn_stream_options options = IFN_STREAM_OPTIONS_INIT;
        options.parse_chat = input->kind == IFN_INPUT_CHAT;
        options.stops = stops;
        options.stop_count = (size_t)config->stops.count;
        rc = ifn_stream_create(s->model->tokenizer, &options, &r->allocator,
                               ifn_infer_request_local_text_event, r, &r->stream);
    }
    if (rc) {
        ifn_infer_request_local_release_request(r);
        ifn_infer_unlock(&s->busy);
        return ifn_infer_native_error(e, rc, IFN_STAGE_VALIDATE);
    }
    s->children++;
    uint64_t now = ifn_infer_now();
    if (start && now) {
        r->result.prepare_ns = now - start;
        r->result.timing_valid |= IFN_INFER_TIME_PREPARE;
    }
    r->result.input_tokens = r->token_count;
    r->result.resolved_execution_profile = ifn_infer_text(s->profile);
    *out = r;
    ifn_infer_unlock(&s->busy);
    return ifn_infer_error(e, IFN_OK, 0, NULL);
}

static ifn_status ifn_infer_request_local_scope_begin(ifn_infer_request *r) {
    ifn_infer_session *s = r->session;
    bool same = r->scope_bytes && r->scope_bytes == s->cache_scope_bytes &&
                !memcmp(r->scope, s->cache_scope, r->scope_bytes);
    if (!same) {
        ifn_status rc = ifn_session_prefix_cache(s->native, 0);
        s->cache_scope_bytes = 0;
        if (rc)
            return rc;
    }
    if (r->scope_bytes && !same) {
        ifn_status rc = ifn_session_prefix_cache(s->native, 1);
        if (rc)
            return rc;
        memcpy(s->cache_scope, r->scope, r->scope_bytes);
        s->cache_scope_bytes = r->scope_bytes;
    }
    return IFN_OK;
}

ifn_status ifn_infer_request_run(ifn_infer_request *r, ifn_infer_emit_fn emit, void *context,
                                 ifn_infer_error_v1 *e) {
    if (!r || (!emit && r->config.output_kind != IFN_OUTPUT_DISCARD))
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE,
                               "missing request or output callback");
    ifn_infer_session *s = r->session;
    if (!ifn_infer_lock(&s->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_VALIDATE, "session already executing");
    if (atomic_load_explicit(&r->phase, memory_order_acquire) != IFN_REQUEST_PREPARED) {
        ifn_infer_unlock(&s->busy);
        return ifn_infer_error(e, IFN_INFER_INVALID_STATE, IFN_STAGE_VALIDATE,
                               "request is one-shot");
    }
    if (s->state != IFN_HANDLE_READY) {
        ifn_infer_unlock(&s->busy);
        return ifn_infer_error(e, IFN_INFER_NOT_READY, IFN_STAGE_VALIDATE, "session not ready");
    }
    atomic_store_explicit(&r->running, true, memory_order_release);
    atomic_store_explicit(&r->phase, IFN_REQUEST_RUNNING, memory_order_release);
    r->emit = emit;
    r->context = context;
    r->started = ifn_infer_now();
    ifn_status rc = IFN_OK;
    bool native_called = false;
    ifn_decode_stats stats = {0};
    if (!ifn_infer_request_local_cancelled(r))
        rc = ifn_infer_request_local_scope_begin(r);
    if (!rc && !ifn_infer_request_local_cancelled(r)) {
        native_called = true;
        ifn_run_control control = {ifn_infer_request_local_cancelled, r};
        rc = ifn_session_generate_controlled(
            s->native, r->tokens, r->token_count, (size_t)r->config.max_output_tokens, &r->decode,
            ifn_infer_request_local_token_event, r, &stats, &control);
        if (rc && rc != IFN_CANCELLED)
            s->state = IFN_HANDLE_POISONED;
        if (!rc || rc == IFN_CANCELLED) {
            ifn_prefix_info p = IFN_PREFIX_INFO_INIT;
            if (ifn_session_prefix_info(s->native, &p) == IFN_OK)
                r->result.prefix_reused_tokens = p.reused_tokens;
        }
    }
    if (native_called)
        atomic_store_explicit(&r->processed, r->token_count, memory_order_relaxed);
    if (!rc && r->callback_status)
        rc = r->callback_status;
    if (!rc && !ifn_infer_request_local_cancelled(r) && r->stream) {
        rc = ifn_stream_finish(r->stream, IFN_STREAM_LENGTH);
        if (!rc && r->callback_status)
            rc = r->callback_status;
    }
    /* One atomic seal is the cancellation/completion linearization point.
     * Session lease stays held through all cancellation/error invalidation. */
    unsigned sealed = atomic_fetch_or_explicit(&r->control, IFN_INFER_SEALED, memory_order_acq_rel);
    bool was_cancelled = (sealed & IFN_INFER_CANCEL) != 0 || rc == IFN_CANCELLED;
    ifn_status cleanup = IFN_OK;
    if ((was_cancelled || rc) && s->state == IFN_HANDLE_READY) {
        cleanup = ifn_session_reset(s->native);
        if (cleanup)
            s->state = IFN_HANDLE_POISONED;
        s->cache_scope_bytes = 0;
    }
    ifn_status primary = rc;
    if (cleanup && (!rc || rc == IFN_CANCELLED))
        rc = cleanup;
    if (!rc && was_cancelled)
        rc = IFN_CANCELLED;
    r->result.status = rc;
    r->result.cancel_requested = (sealed & IFN_INFER_CANCEL) != 0;
    r->result.generated_tokens = atomic_load_explicit(&r->generated, memory_order_relaxed);
    r->result.finish_reason = rc == IFN_CANCELLED ? IFN_FINISH_CANCELLED_V1
                              : rc                ? IFN_FINISH_ERROR_V1
                              : stats.generation.finish_reason == IFN_FINISH_EOS ? IFN_FINISH_EOS_V1
                              : r->callback_action == IFN_CALLBACK_STOP
                                  ? IFN_FINISH_CALLBACK_V1
                                  : IFN_FINISH_OUTPUT_LIMIT_V1;
    if (r->stream) {
        ifn_stream_info info = IFN_STREAM_INFO_INIT;
        if (ifn_stream_get_info(r->stream, &info) == IFN_OK) {
            r->result.utf8_replacements = info.utf8_replacements;
            if (!rc && info.finish_reason == IFN_STREAM_STOP)
                r->result.finish_reason = IFN_FINISH_STOP_STRING_V1;
            if (!rc && info.finish_reason == IFN_STREAM_EOS)
                r->result.finish_reason = IFN_FINISH_EOS_V1;
        }
    }
    uint64_t now = ifn_infer_now();
    if (now && r->started) {
        r->result.run_ns = now - r->started;
        r->result.timing_valid |= IFN_INFER_TIME_RUN;
    }
    (void)ifn_infer_native_error(&r->terminal_error, rc, IFN_STAGE_DECODE);
    if (s->state == IFN_HANDLE_POISONED)
        r->terminal_error.flags |= IFN_INFER_ERR_SESSION_POISONED;
    if (r->sequence)
        r->terminal_error.flags |= IFN_INFER_ERR_OUTPUT_STARTED;
    if (cleanup) {
        r->terminal_error.cleanup_status = cleanup;
        r->terminal_error.flags |= IFN_INFER_ERR_CLEANUP_PENDING;
        if (primary && primary != IFN_CANCELLED)
            r->terminal_error.status = primary;
    }
    r->emit = NULL;
    r->context = NULL;
    atomic_store_explicit(&r->phase,
                          rc == IFN_CANCELLED ? IFN_REQUEST_CANCELLED
                          : rc                ? IFN_REQUEST_FAILED
                                              : IFN_REQUEST_COMPLETED,
                          memory_order_release);
    if (e)
        (void)ifn_infer_output(e, &r->terminal_error, sizeof r->terminal_error);
    /* No access to r after releasing session: close holds the same lease. */
    atomic_store_explicit(&r->running, false, memory_order_release);
    ifn_infer_unlock(&s->busy);
    return rc;
}

ifn_status ifn_infer_request_cancel(ifn_infer_request *r, uint32_t *disposition,
                                    ifn_infer_error_v1 *e) {
    if (disposition)
        *disposition = 0;
    if (!r || !disposition)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE,
                               "missing cancellation arguments");
    unsigned old = atomic_load_explicit(&r->control, memory_order_acquire);
    for (;;) {
        if (old & IFN_INFER_SEALED) {
            *disposition = IFN_CANCEL_TOO_LATE;
            break;
        }
        if (old & IFN_INFER_CANCEL) {
            *disposition = IFN_CANCEL_ALREADY_REQUESTED;
            break;
        }
        if (atomic_compare_exchange_weak_explicit(&r->control, &old, old | IFN_INFER_CANCEL,
                                                  memory_order_acq_rel, memory_order_acquire)) {
            *disposition = IFN_CANCEL_ACCEPTED;
            break;
        }
    }
    return ifn_infer_error(e, IFN_OK, 0, NULL);
}

ifn_status ifn_infer_request_progress(const ifn_infer_request *r, ifn_infer_progress_v1 *out,
                                      ifn_infer_error_v1 *e) {
    if (!r)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE, "missing request");
    ifn_infer_progress_v1 v = {0};
    v.state = atomic_load_explicit(&r->phase, memory_order_acquire);
    v.cancel_requested =
        (atomic_load_explicit(&r->control, memory_order_acquire) & IFN_INFER_CANCEL) != 0;
    v.generated_tokens = atomic_load_explicit(&r->generated, memory_order_relaxed);
    v.prompt_tokens_processed = atomic_load_explicit(&r->processed, memory_order_relaxed);
    return ifn_infer_native_error(e, ifn_infer_output(out, &v, sizeof v), IFN_STAGE_VALIDATE);
}

ifn_status ifn_infer_request_result(const ifn_infer_request *r, ifn_infer_result_v1 *out,
                                    ifn_infer_error_v1 *e) {
    if (!r)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE, "missing request");
    unsigned phase = atomic_load_explicit(&r->phase, memory_order_acquire);
    if (phase == IFN_REQUEST_PREPARED || phase == IFN_REQUEST_RUNNING)
        return ifn_infer_error(e, IFN_INFER_NOT_READY, IFN_STAGE_VALIDATE,
                               "request has no terminal result");
    ifn_status rc = ifn_infer_output(out, &r->result, sizeof r->result);
    return ifn_infer_native_error(e, rc, IFN_STAGE_VALIDATE);
}

ifn_status ifn_infer_request_close(ifn_infer_request **ptr, ifn_infer_error_v1 *e) {
    if (!ptr)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_CLOSE, "missing handle pointer");
    ifn_infer_request *r = *ptr;
    if (!r)
        return ifn_infer_error(e, IFN_OK, 0, NULL);
    ifn_infer_session *s = r->session;
    if (!ifn_infer_lock(&s->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_CLOSE, "session busy");
    if (atomic_load_explicit(&r->running, memory_order_acquire)) {
        ifn_infer_unlock(&s->busy);
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_CLOSE, "request running");
    }
    s->children--;
    ifn_infer_request_local_release_request(r);
    *ptr = NULL;
    ifn_infer_unlock(&s->busy);
    return ifn_infer_error(e, IFN_OK, 0, NULL);
}
