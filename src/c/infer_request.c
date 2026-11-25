/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "infer_internal.h"
#include <math.h>

static bool cancelled(void *context) {
    sq_infer_request *r = context;
    return (atomic_load_explicit(&r->control, memory_order_acquire) & SI_CANCEL) != 0;
}

static sq_status array_valid(sq_infer_array a, size_t element, size_t maximum) {
    if (a.reserved || a.count > maximum || (a.count && (!a.data || a.stride < element)) ||
        (a.count && a.count > SIZE_MAX / a.stride))
        return SQ_INVALID_ARGUMENT;
    if (!a.count && (a.data || a.stride))
        return SQ_INVALID_ARGUMENT;
    return SQ_OK;
}

static sq_status request_options(sq_infer_request *r, const sq_infer_request_config_v1 *c,
                                 sq_chat_options *chat) {
    sq_status rc = si_input(c, sizeof *c);
    if (rc)
        return rc;
    uint64_t supported = SI_FEATURES;
    if (!r->session->model->weights_config.with_mtp)
        supported &= ~SQ_INFER_CAP_SPECULATION;
    if (c->required_features & ~supported)
        return SQ_UNSUPPORTED;
    if (!c->max_output_tokens || c->max_output_tokens > r->session->config.context_tokens ||
        !isfinite(c->temperature) || c->temperature < 0 || !isfinite(c->top_p) || c->top_p <= 0 ||
        c->top_p > 1 || c->top_k < 0 || c->top_k > 200058 || c->output_kind < SQ_OUTPUT_TEXT ||
        c->output_kind > SQ_OUTPUT_DISCARD || c->max_event_bytes < 4 ||
        c->max_event_bytes > 65536 || c->reserved ||
        (c->output_flags & ~SQ_INFER_OUTPUT_REASONING) || c->history_policy > 1 ||
        !si_span(c->cache_scope, sizeof r->scope))
        return SQ_INVALID_ARGUMENT;
    rc = array_valid(c->extensions, sizeof(sq_infer_extension_v1), 16);
    if (rc)
        return rc;
    if (c->extensions.count > 1)
        return SQ_UNSUPPORTED;
    r->decode = (sq_decode_options)SQ_DECODE_OPTIONS_INIT;
    r->decode.temperature = c->temperature;
    r->decode.top_p = c->top_p;
    r->decode.top_k = c->top_k;
    r->decode.seed = c->seed;
    *chat = (sq_chat_options)SQ_CHAT_OPTIONS_INIT;
    chat->keep_history_thinking = c->history_policy;
    for (size_t i = 0; i < c->extensions.count; i++) {
        const unsigned char *p =
            (const unsigned char *)c->extensions.data + i * c->extensions.stride;
        sq_infer_extension_v1 ext;
        memcpy(&ext, p, sizeof ext);
        /* Read headers/tails as bytes, including padded array elements. */
        if (ext.hdr.struct_size > c->extensions.stride)
            return SQ_INVALID_ARGUMENT;
        rc = si_input(p, sizeof ext);
        if (rc)
            return rc;
        if (ext.id != SQ_INFER_EXT_SQUID_REQUEST || ext.version != 1 ||
            ext.payload_bytes < sizeof(sq_infer_squid_request_v1) ||
            ext.payload_bytes > SQ_INFER_RECORD_MAX_BYTES || !ext.payload)
            return SQ_UNSUPPORTED;
        sq_infer_squid_request_v1 v;
        memcpy(&v, ext.payload, sizeof v);
        if (v.hdr.struct_size > ext.payload_bytes)
            return SQ_INVALID_ARGUMENT;
        rc = si_input(ext.payload, sizeof v);
        if (rc)
            return rc;
        if (v.draft_tokens > 7 || v.adaptive > 1 || v.prompt_lookup > 1 ||
            v.thinking_effort > SQ_THINK_MAX || v.enable_thinking > 1 ||
            !isfinite(v.spec_threshold) ||
            (v.adaptive && (v.spec_threshold <= 0 || v.spec_threshold > 1)))
            return SQ_INVALID_ARGUMENT;
        if ((v.draft_tokens && !r->session->model->weights_config.with_mtp) ||
            (v.adaptive && (v.draft_tokens != 7 || !r->session->config.use_graphs)) ||
            (v.prompt_lookup && !v.adaptive))
            return SQ_UNSUPPORTED;
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
    r->config.extensions = (sq_infer_array){0};
    r->config.stops = (sq_infer_array){0};
    r->config.cache_scope = (sq_infer_bytes){0};
    r->scope_bytes = (size_t)c->cache_scope.bytes;
    if (r->scope_bytes)
        memcpy(r->scope, c->cache_scope.data, r->scope_bytes);
    return SQ_OK;
}

static sq_status encode(sq_infer_request *r, const sq_infer_input_v1 *in,
                        const sq_chat_options *chat) {
    sq_status rc = si_input(in, sizeof *in);
    if (rc)
        return rc;
    if (in->reserved || !si_span(in->text, SI_MAX_INPUT))
        return SQ_INVALID_ARGUMENT;
    if (in->kind != SQ_INPUT_CHAT &&
        (chat->effort || chat->enable_thinking || chat->keep_history_thinking))
        return SQ_UNSUPPORTED;
    size_t capacity = r->session->config.context_tokens - (size_t)r->config.max_output_tokens;
    if (!capacity)
        return SQ_LIMIT;
    size_t bytes;
    if (!sq_size_mul(capacity, sizeof(int32_t), &bytes))
        return SQ_LIMIT;
    r->tokens = si_alloc(&r->allocator, bytes);
    if (!r->tokens)
        return SQ_OUT_OF_MEMORY;
    if (in->kind == SQ_INPUT_TOKENS) {
        if (in->text.data || in->text.bytes || si_no_extensions(in->messages) || !in->tokens ||
            !in->token_count || in->token_count > capacity ||
            r->config.output_kind == SQ_OUTPUT_TEXT)
            return SQ_INVALID_ARGUMENT;
        for (size_t i = 0; i < in->token_count; i++) {
            if (in->tokens[i] < 0 || in->tokens[i] >= 200058)
                return SQ_INVALID_ARGUMENT;
            r->tokens[i] = in->tokens[i];
        }
        r->token_count = (size_t)in->token_count;
        return SQ_OK;
    }
    if (in->tokens || in->token_count)
        return SQ_INVALID_ARGUMENT;
    sq_tokenizer_scratch *scratch = NULL;
    rc = sq_tokenizer_scratch_create(&r->allocator, &scratch);
    if (rc)
        return rc;
    if (in->kind == SQ_INPUT_RAW_TEXT) {
        if (si_no_extensions(in->messages))
            rc = SQ_INVALID_ARGUMENT;
        else
            rc = sq_tokenizer_encode(r->session->model->tokenizer, in->text.data,
                                     (size_t)in->text.bytes, 0, scratch, r->tokens, capacity,
                                     &r->token_count);
    } else if (in->kind == SQ_INPUT_CHAT) {
        rc = array_valid(in->messages, sizeof(sq_infer_message_v1), 1024);
        if (in->text.data || in->text.bytes || !in->messages.count)
            rc = SQ_INVALID_ARGUMENT;
        sq_chat_message *messages = NULL;
        if (!rc) {
            messages = si_alloc(&r->allocator, (size_t)in->messages.count * sizeof *messages);
            if (!messages)
                rc = SQ_OUT_OF_MEMORY;
        }
        uint64_t total = 0;
        for (size_t i = 0; !rc && i < in->messages.count; i++) {
            sq_infer_message_v1 m;
            const unsigned char *p =
                (const unsigned char *)in->messages.data + i * in->messages.stride;
            memcpy(&m, p, sizeof m);
            if (m.hdr.struct_size > in->messages.stride) {
                rc = SQ_INVALID_ARGUMENT;
                break;
            }
            rc = si_input(p, sizeof m);
            if (rc)
                break;
            if (m.content_present > 1 || m.role < SQ_ROLE_SYSTEM || m.role > SQ_ROLE_ASSISTANT ||
                !si_span(m.content, SI_MAX_INPUT) || !si_span(m.reasoning, SI_MAX_INPUT) ||
                !si_span(m.name, 128) ||
                (!m.content_present && (m.content.data || m.content.bytes))) {
                rc = SQ_INVALID_ARGUMENT;
                break;
            }
            total += m.content.bytes + m.reasoning.bytes + m.name.bytes;
            if (total > SI_MAX_INPUT) {
                rc = SQ_LIMIT;
                break;
            }
            messages[i] =
                (sq_chat_message){m.role - 1,
                                  0,
                                  m.content_present ? (m.content.data ? m.content.data : "") : NULL,
                                  (size_t)m.content.bytes,
                                  m.reasoning.data,
                                  (size_t)m.reasoning.bytes,
                                  m.name.data,
                                  (size_t)m.name.bytes};
        }
        if (!rc)
            rc = sq_chat_encode(r->session->model->tokenizer, messages, (size_t)in->messages.count,
                                chat, scratch, r->tokens, capacity, &r->token_count,
                                &r->decode.system_prefix_tokens);
        si_free(&r->allocator, messages);
    } else
        rc = SQ_UNSUPPORTED;
    sq_tokenizer_scratch_destroy(scratch);
    return !rc && !r->token_count ? SQ_INVALID_ARGUMENT : rc;
}

static uint32_t emit_event(sq_infer_request *r, uint32_t kind, const void *text, size_t bytes,
                           const int32_t *tokens, size_t count) {
    if (cancelled(r))
        return SQ_CALLBACK_CANCEL;
    sq_infer_event_v1 v = {0};
    v.hdr.struct_size = v.hdr.written_size = sizeof v;
    v.kind = kind;
    v.sequence = r->sequence++;
    v.generated_tokens_so_far = atomic_load_explicit(&r->generated, memory_order_relaxed);
    v.text = (sq_infer_bytes){text, bytes};
    v.tokens = tokens;
    v.token_count = count;
    uint32_t action = r->emit(r->context, &v);
    if (kind == SQ_EVENT_CONTENT)
        r->result.content_bytes += bytes;
    if (kind == SQ_EVENT_REASONING)
        r->result.reasoning_bytes += bytes;
    uint64_t now = si_now();
    if (now && r->started) {
        if (kind == SQ_EVENT_CONTENT && !(r->result.timing_valid & SQ_INFER_TIME_FIRST_CONTENT)) {
            r->result.first_content_ns = now - r->started;
            r->result.timing_valid |= SQ_INFER_TIME_FIRST_CONTENT;
        }
        if (kind == SQ_EVENT_REASONING &&
            !(r->result.timing_valid & SQ_INFER_TIME_FIRST_REASONING)) {
            r->result.first_reasoning_ns = now - r->started;
            r->result.timing_valid |= SQ_INFER_TIME_FIRST_REASONING;
        }
    }
    if (action > SQ_CALLBACK_CANCEL) {
        r->callback_status = SQ_INVALID_ARGUMENT;
        return SQ_CALLBACK_STOP;
    }
    if (action == SQ_CALLBACK_CANCEL) {
        uint32_t disposition;
        (void)si_request_cancel(r, &disposition, NULL);
    }
    if (action)
        r->callback_action = action;
    return action;
}

static int text_event(void *context, uint32_t channel, const char *p, size_t n) {
    sq_infer_request *r = context;
    if (cancelled(r))
        return 0;
    if (channel == SQ_STREAM_REASONING && !(r->config.output_flags & SQ_INFER_OUTPUT_REASONING))
        return 1;
    while (n) {
        size_t take = n > r->config.max_event_bytes ? r->config.max_event_bytes : n;
        if (take < n)
            while (take && (((unsigned char)p[take] & 0xc0) == 0x80))
                take--;
        if (!take) {
            r->callback_status = SQ_FORMAT;
            return 0;
        }
        if (emit_event(r, channel == SQ_STREAM_REASONING ? SQ_EVENT_REASONING : SQ_EVENT_CONTENT, p,
                       take, NULL, 0))
            return 0;
        p += take;
        n -= take;
    }
    return 1;
}

static int token_event(void *context, int32_t token) {
    sq_infer_request *r = context;
    atomic_fetch_add_explicit(&r->generated, 1, memory_order_relaxed);
    if (cancelled(r))
        return 0;
    if (r->config.output_kind == SQ_OUTPUT_RAW_TOKENS)
        return emit_event(r, SQ_EVENT_TOKENS, NULL, 0, &token, 1) == SQ_CALLBACK_CONTINUE;
    if (r->config.output_kind == SQ_OUTPUT_DISCARD)
        return 1;
    sq_status rc = sq_stream_push_token(r->stream, token);
    if (rc) {
        r->callback_status = rc;
        return 0;
    }
    sq_stream_info info = SQ_STREAM_INFO_INIT;
    rc = sq_stream_get_info(r->stream, &info);
    if (rc) {
        r->callback_status = rc;
        return 0;
    }
    return info.finish_reason == SQ_STREAM_RUNNING;
}

static void release_request(sq_infer_request *r) {
    (void)sq_stream_destroy(&r->stream);
    si_free(&r->allocator, r->tokens);
    sq_options a = r->allocator;
    si_free(&a, r);
}

sq_status si_request_create(sq_infer_session *s, const sq_infer_input_v1 *input,
                            const sq_infer_request_config_v1 *config, sq_infer_request **out,
                            sq_infer_error_v1 *e) {
    if (out)
        *out = NULL;
    if (!s || !out)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE, "invalid request arguments");
    if (!si_lock(&s->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_VALIDATE, "session busy");
    if (s->state != SQ_HANDLE_READY || s->children == UINT32_MAX) {
        si_unlock(&s->busy);
        return si_error(e, SQ_INFER_NOT_READY, SQ_STAGE_VALIDATE, "session not ready");
    }
    uint64_t start = si_now();
    sq_infer_request *r = si_alloc(&s->allocator, sizeof *r);
    if (!r) {
        si_unlock(&s->busy);
        return si_error(e, SQ_OUT_OF_MEMORY, SQ_STAGE_VALIDATE, "allocate request");
    }
    memset(r, 0, sizeof *r);
    r->session = s;
    r->allocator = s->allocator;
    atomic_init(&r->control, 0);
    atomic_init(&r->phase, SQ_REQUEST_PREPARED);
    atomic_init(&r->running, false);
    atomic_init(&r->generated, 0);
    atomic_init(&r->processed, 0);
    SQ_INFER_INIT(r->terminal_error);
    sq_chat_options chat;
    sq_status rc = request_options(r, config, &chat);
    if (!rc)
        rc = encode(r, input, &chat);
    sq_text_span stops[8];
    if (!rc)
        rc = array_valid(config->stops, sizeof(sq_infer_bytes), 8);
    if (!rc && config->output_kind != SQ_OUTPUT_TEXT && config->stops.count)
        rc = SQ_UNSUPPORTED;
    for (size_t i = 0; !rc && i < config->stops.count; i++) {
        sq_infer_bytes span;
        memcpy(&span, (const unsigned char *)config->stops.data + i * config->stops.stride,
               sizeof span);
        if (!si_span(span, 64) || !span.bytes) {
            rc = SQ_INVALID_ARGUMENT;
            break;
        }
        stops[i] = (sq_text_span){span.data, (size_t)span.bytes};
    }
    if (!rc && config->output_kind == SQ_OUTPUT_TEXT) {
        sq_stream_options options = SQ_STREAM_OPTIONS_INIT;
        options.parse_chat = input->kind == SQ_INPUT_CHAT;
        options.stops = stops;
        options.stop_count = (size_t)config->stops.count;
        rc = sq_stream_create(s->model->tokenizer, &options, &r->allocator, text_event, r,
                              &r->stream);
    }
    if (rc) {
        release_request(r);
        si_unlock(&s->busy);
        return si_native_error(e, rc, SQ_STAGE_VALIDATE);
    }
    s->children++;
    uint64_t now = si_now();
    if (start && now) {
        r->result.prepare_ns = now - start;
        r->result.timing_valid |= SQ_INFER_TIME_PREPARE;
    }
    r->result.input_tokens = r->token_count;
    r->result.resolved_execution_profile = si_text(s->profile);
    *out = r;
    si_unlock(&s->busy);
    return si_error(e, SQ_OK, 0, NULL);
}

static sq_status scope_begin(sq_infer_request *r) {
    sq_infer_session *s = r->session;
    bool same = r->scope_bytes && r->scope_bytes == s->cache_scope_bytes &&
                !memcmp(r->scope, s->cache_scope, r->scope_bytes);
    if (!same) {
        sq_status rc = sq_session_prefix_cache(s->native, 0);
        s->cache_scope_bytes = 0;
        if (rc)
            return rc;
    }
    if (r->scope_bytes && !same) {
        sq_status rc = sq_session_prefix_cache(s->native, 1);
        if (rc)
            return rc;
        memcpy(s->cache_scope, r->scope, r->scope_bytes);
        s->cache_scope_bytes = r->scope_bytes;
    }
    return SQ_OK;
}

sq_status si_request_run(sq_infer_request *r, sq_infer_emit_fn emit, void *context,
                         sq_infer_error_v1 *e) {
    if (!r || (!emit && r->config.output_kind != SQ_OUTPUT_DISCARD))
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE,
                        "missing request or output callback");
    sq_infer_session *s = r->session;
    if (!si_lock(&s->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_VALIDATE, "session already executing");
    if (atomic_load_explicit(&r->phase, memory_order_acquire) != SQ_REQUEST_PREPARED) {
        si_unlock(&s->busy);
        return si_error(e, SQ_INFER_INVALID_STATE, SQ_STAGE_VALIDATE, "request is one-shot");
    }
    if (s->state != SQ_HANDLE_READY) {
        si_unlock(&s->busy);
        return si_error(e, SQ_INFER_NOT_READY, SQ_STAGE_VALIDATE, "session not ready");
    }
    atomic_store_explicit(&r->running, true, memory_order_release);
    atomic_store_explicit(&r->phase, SQ_REQUEST_RUNNING, memory_order_release);
    r->emit = emit;
    r->context = context;
    r->started = si_now();
    sq_status rc = SQ_OK;
    bool native_called = false;
    sq_decode_stats stats = {0};
    if (!cancelled(r))
        rc = scope_begin(r);
    if (!rc && !cancelled(r)) {
        native_called = true;
        sq_run_control control = {cancelled, r};
        rc = sq_session_generate_controlled(s->native, r->tokens, r->token_count,
                                            (size_t)r->config.max_output_tokens, &r->decode,
                                            token_event, r, &stats, &control);
        if (rc && rc != SQ_CANCELLED)
            s->state = SQ_HANDLE_POISONED;
        if (!rc || rc == SQ_CANCELLED) {
            sq_prefix_info p = SQ_PREFIX_INFO_INIT;
            if (sq_session_prefix_info(s->native, &p) == SQ_OK)
                r->result.prefix_reused_tokens = p.reused_tokens;
        }
    }
    if (native_called)
        atomic_store_explicit(&r->processed, r->token_count, memory_order_relaxed);
    if (!rc && r->callback_status)
        rc = r->callback_status;
    if (!rc && !cancelled(r) && r->stream) {
        rc = sq_stream_finish(r->stream, SQ_STREAM_LENGTH);
        if (!rc && r->callback_status)
            rc = r->callback_status;
    }
    /* One atomic seal is the cancellation/completion linearization point.
     * Session lease stays held through all cancellation/error invalidation. */
    unsigned sealed = atomic_fetch_or_explicit(&r->control, SI_SEALED, memory_order_acq_rel);
    bool was_cancelled = (sealed & SI_CANCEL) != 0 || rc == SQ_CANCELLED;
    sq_status cleanup = SQ_OK;
    if ((was_cancelled || rc) && s->state == SQ_HANDLE_READY) {
        cleanup = sq_session_reset(s->native);
        if (cleanup)
            s->state = SQ_HANDLE_POISONED;
        s->cache_scope_bytes = 0;
    }
    sq_status primary = rc;
    if (cleanup && (!rc || rc == SQ_CANCELLED))
        rc = cleanup;
    if (!rc && was_cancelled)
        rc = SQ_CANCELLED;
    r->result.status = rc;
    r->result.cancel_requested = (sealed & SI_CANCEL) != 0;
    r->result.generated_tokens = atomic_load_explicit(&r->generated, memory_order_relaxed);
    r->result.finish_reason = rc == SQ_CANCELLED ? SQ_FINISH_CANCELLED_V1
                              : rc               ? SQ_FINISH_ERROR_V1
                              : stats.generation.finish_reason == SQ_FINISH_EOS ? SQ_FINISH_EOS_V1
                              : r->callback_action == SQ_CALLBACK_STOP ? SQ_FINISH_CALLBACK_V1
                                                                       : SQ_FINISH_OUTPUT_LIMIT_V1;
    if (r->stream) {
        sq_stream_info info = SQ_STREAM_INFO_INIT;
        if (sq_stream_get_info(r->stream, &info) == SQ_OK) {
            r->result.utf8_replacements = info.utf8_replacements;
            if (!rc && info.finish_reason == SQ_STREAM_STOP)
                r->result.finish_reason = SQ_FINISH_STOP_STRING_V1;
            if (!rc && info.finish_reason == SQ_STREAM_EOS)
                r->result.finish_reason = SQ_FINISH_EOS_V1;
        }
    }
    uint64_t now = si_now();
    if (now && r->started) {
        r->result.run_ns = now - r->started;
        r->result.timing_valid |= SQ_INFER_TIME_RUN;
    }
    (void)si_native_error(&r->terminal_error, rc, SQ_STAGE_DECODE);
    if (s->state == SQ_HANDLE_POISONED)
        r->terminal_error.flags |= SQ_INFER_ERR_SESSION_POISONED;
    if (r->sequence)
        r->terminal_error.flags |= SQ_INFER_ERR_OUTPUT_STARTED;
    if (cleanup) {
        r->terminal_error.cleanup_status = cleanup;
        r->terminal_error.flags |= SQ_INFER_ERR_CLEANUP_PENDING;
        if (primary && primary != SQ_CANCELLED)
            r->terminal_error.status = primary;
    }
    r->emit = NULL;
    r->context = NULL;
    atomic_store_explicit(&r->phase,
                          rc == SQ_CANCELLED ? SQ_REQUEST_CANCELLED
                          : rc               ? SQ_REQUEST_FAILED
                                             : SQ_REQUEST_COMPLETED,
                          memory_order_release);
    if (e)
        (void)si_output(e, &r->terminal_error, sizeof r->terminal_error);
    /* No access to r after releasing session: close holds the same lease. */
    atomic_store_explicit(&r->running, false, memory_order_release);
    si_unlock(&s->busy);
    return rc;
}

sq_status si_request_cancel(sq_infer_request *r, uint32_t *disposition, sq_infer_error_v1 *e) {
    if (disposition)
        *disposition = 0;
    if (!r || !disposition)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE,
                        "missing cancellation arguments");
    unsigned old = atomic_load_explicit(&r->control, memory_order_acquire);
    for (;;) {
        if (old & SI_SEALED) {
            *disposition = SQ_CANCEL_TOO_LATE;
            break;
        }
        if (old & SI_CANCEL) {
            *disposition = SQ_CANCEL_ALREADY_REQUESTED;
            break;
        }
        if (atomic_compare_exchange_weak_explicit(&r->control, &old, old | SI_CANCEL,
                                                  memory_order_acq_rel, memory_order_acquire)) {
            *disposition = SQ_CANCEL_ACCEPTED;
            break;
        }
    }
    return si_error(e, SQ_OK, 0, NULL);
}

sq_status si_request_progress(const sq_infer_request *r, sq_infer_progress_v1 *out,
                              sq_infer_error_v1 *e) {
    if (!r)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE, "missing request");
    sq_infer_progress_v1 v = {0};
    v.state = atomic_load_explicit(&r->phase, memory_order_acquire);
    v.cancel_requested = (atomic_load_explicit(&r->control, memory_order_acquire) & SI_CANCEL) != 0;
    v.generated_tokens = atomic_load_explicit(&r->generated, memory_order_relaxed);
    v.prompt_tokens_processed = atomic_load_explicit(&r->processed, memory_order_relaxed);
    return si_native_error(e, si_output(out, &v, sizeof v), SQ_STAGE_VALIDATE);
}

sq_status si_request_result(const sq_infer_request *r, sq_infer_result_v1 *out,
                            sq_infer_error_v1 *e) {
    if (!r)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE, "missing request");
    unsigned phase = atomic_load_explicit(&r->phase, memory_order_acquire);
    if (phase == SQ_REQUEST_PREPARED || phase == SQ_REQUEST_RUNNING)
        return si_error(e, SQ_INFER_NOT_READY, SQ_STAGE_VALIDATE, "request has no terminal result");
    sq_status rc = si_output(out, &r->result, sizeof r->result);
    return si_native_error(e, rc, SQ_STAGE_VALIDATE);
}

sq_status si_request_close(sq_infer_request **ptr, sq_infer_error_v1 *e) {
    if (!ptr)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_CLOSE, "missing handle pointer");
    sq_infer_request *r = *ptr;
    if (!r)
        return si_error(e, SQ_OK, 0, NULL);
    sq_infer_session *s = r->session;
    if (!si_lock(&s->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_CLOSE, "session busy");
    if (atomic_load_explicit(&r->running, memory_order_acquire)) {
        si_unlock(&s->busy);
        return si_error(e, SQ_BUSY, SQ_STAGE_CLOSE, "request running");
    }
    s->children--;
    release_request(r);
    *ptr = NULL;
    si_unlock(&s->busy);
    return si_error(e, SQ_OK, 0, NULL);
}
