/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#define _POSIX_C_SOURCE 200809L
#include "infer_internal.h"
#include <stdio.h>
#include <time.h>

/* Header is maximally aligned and private to our allocator, never an ABI record. */
typedef union {
    max_align_t alignment;

    struct {
        size_t bytes;
    } value;
} si_allocation;

static void *budget_allocate(void *context, size_t bytes) {
    si_budget *b = context;
    size_t total;
    if (!sq_size_add(sizeof(si_allocation), bytes, &total))
        return NULL;
    uint_fast64_t old = atomic_load_explicit(&b->live, memory_order_relaxed);
    do {
        if (old > b->limit || total > b->limit - old)
            return NULL;
    } while (!atomic_compare_exchange_weak_explicit(&b->live, &old, old + total,
                                                    memory_order_relaxed, memory_order_relaxed));
    si_allocation *p = b->upstream.allocate(b->upstream.context, total);
    if (!p) {
        atomic_fetch_sub_explicit(&b->live, total, memory_order_relaxed);
        return NULL;
    }
    p->value.bytes = total;
    return p + 1;
}

static void budget_release(void *context, void *ptr) {
    if (!ptr)
        return;
    si_budget *b = context;
    si_allocation *p = (si_allocation *)ptr - 1;
    size_t total = p->value.bytes;
    b->upstream.release(b->upstream.context, p);
    atomic_fetch_sub_explicit(&b->live, total, memory_order_relaxed);
}

void si_budget_init(si_budget *b, const sq_options *upstream, uint64_t limit, sq_options *out) {
    b->upstream = *upstream;
    b->limit = limit;
    atomic_init(&b->live, 0);
    *out = (sq_options)SQ_OPTIONS_INIT;
    out->context = b;
    out->allocate = budget_allocate;
    out->release = budget_release;
}

void *si_alloc(const sq_options *a, size_t n) {
    return a->allocate(a->context, n);
}

void si_free(const sq_options *a, void *p) {
    if (p)
        a->release(a->context, p);
}

bool si_lock(atomic_flag *f) {
    return !atomic_flag_test_and_set_explicit(f, memory_order_acquire);
}

void si_unlock(atomic_flag *f) {
    atomic_flag_clear_explicit(f, memory_order_release);
}

sq_status si_input(const void *record, size_t known) {
    if (!record)
        return SQ_INVALID_ARGUMENT;
    sq_infer_in h;
    memcpy(&h, record, sizeof h);
    if (h.struct_size < known || h.struct_size > SQ_INFER_RECORD_MAX_BYTES || h.flags)
        return SQ_INVALID_ARGUMENT;
    const unsigned char *p = record;
    for (size_t i = known; i < h.struct_size; i++)
        if (p[i])
            return SQ_UNSUPPORTED;
    return SQ_OK;
}

sq_status si_output(void *out, const void *value, size_t known) {
    if (!out)
        return SQ_INVALID_ARGUMENT;
    sq_infer_out *h = out;
    uint32_t capacity = h->struct_size;
    if (capacity < known || capacity > SQ_INFER_RECORD_MAX_BYTES)
        return SQ_INVALID_ARGUMENT;
    memcpy(out, value, known);
    h->struct_size = capacity;
    h->written_size = (uint32_t)known;
    return SQ_OK;
}

sq_status si_error(sq_infer_error_v1 *out, sq_status status, uint32_t stage, const char *text) {
    if (out) {
        sq_infer_error_v1 e = {0};
        e.status = status;
        e.stage = status ? stage : SQ_STAGE_NONE;
        if (status && text) {
            (void)snprintf(e.message, sizeof e.message, "%s", text);
            e.message_bytes = (uint32_t)strlen(e.message);
        }
        (void)si_output(out, &e, sizeof e);
    }
    return status;
}

sq_status si_native_error(sq_infer_error_v1 *e, sq_status rc, uint32_t stage) {
    /* Backend diagnostics contain operation names/codes, never input contents. */
    sq_status status = si_error(e, rc, stage, rc ? sq_status_string(rc) : "");
    if (e && e->hdr.struct_size >= sizeof *e && e->hdr.struct_size <= SQ_INFER_RECORD_MAX_BYTES &&
        rc && stage == SQ_STAGE_CLOSE)
        e->flags |= SQ_INFER_ERR_CLEANUP_PENDING;
    return status;
}

bool si_span(sq_infer_bytes s, size_t max) {
    return s.bytes <= max && (!s.bytes || s.data);
}

bool si_equal(sq_infer_bytes s, const char *text) {
    return si_span(s, 4096) && s.bytes == strlen(text) &&
           (!s.bytes || !memcmp(s.data, text, (size_t)s.bytes));
}

sq_infer_bytes si_text(const char *s) {
    return (sq_infer_bytes){s, strlen(s)};
}

sq_status si_no_extensions(sq_infer_array a) {
    return a.reserved             ? SQ_INVALID_ARGUMENT
           : a.count              ? SQ_UNSUPPORTED
           : (a.data || a.stride) ? SQ_INVALID_ARGUMENT
                                  : SQ_OK;
}

uint64_t si_now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t))
        return 0;
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

sq_status si_defaults(uint32_t kind, void *out, sq_infer_error_v1 *e) {
    size_t size = kind == SQ_DEFAULT_MODEL     ? sizeof(sq_infer_model_config_v1)
                  : kind == SQ_DEFAULT_SESSION ? sizeof(sq_infer_session_config_v1)
                  : kind == SQ_DEFAULT_REQUEST ? sizeof(sq_infer_request_config_v1)
                                               : 0;
    if (!size || !out || ((sq_infer_in *)out)->struct_size < size ||
        ((sq_infer_in *)out)->struct_size > SQ_INFER_RECORD_MAX_BYTES)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE, "invalid defaults record");
    uint32_t capacity = ((sq_infer_in *)out)->struct_size;
    memset(out, 0, size);
    ((sq_infer_in *)out)->struct_size = capacity;
    if (kind == SQ_DEFAULT_MODEL) {
        sq_infer_model_config_v1 *c = out;
        c->max_owned_host_bytes = c->max_owned_device_bytes = UINT64_MAX;
        c->device_id = si_text("0");
        c->weight_profile = si_text("nvfp4-ar");
    } else if (kind == SQ_DEFAULT_SESSION) {
        sq_infer_session_config_v1 *c = out;
        c->max_owned_host_bytes = c->max_owned_device_bytes = UINT64_MAX;
        c->context_tokens = 8192;
        c->execution_profile = si_text("fp8-graph");
    } else {
        sq_infer_request_config_v1 *c = out;
        c->max_output_tokens = 256;
        c->top_p = 0.95f;
        c->output_kind = SQ_OUTPUT_TEXT;
        c->max_event_bytes = 4096;
    }
    return si_error(e, SQ_OK, 0, NULL);
}

sq_status si_backend_info(sq_infer_backend_info_v1 *out, sq_infer_error_v1 *e) {
    sq_infer_backend_info_v1 v = {0};
    v.backend_id = si_text("artalis.squid");
    v.engine_revision =
        si_text("upstream:d10c0abb2c8990be81c08ba84c66a8333a691b33;infer:1.0-experimental");
    v.compiled_features = SI_FEATURES;
    v.pointer_bits = sizeof(void *) * 8;
    v.limit_scope = SQ_LIMIT_PROVIDER_PROCESS;
    v.max_prepared_sessions = v.max_active_runs = 1;
    return si_error(e, si_output(out, &v, sizeof v), SQ_STAGE_VALIDATE, "invalid backend output");
}

static sq_status query_extension(uint64_t id, uint32_t version, const void **out,
                                 sq_infer_error_v1 *e) {
    (void)id;
    (void)version;
    if (out)
        *out = NULL;
    return si_error(e, out ? SQ_UNSUPPORTED : SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE,
                    "no extension function table");
}

static const sq_infer_api_v1 api = {
    sizeof(sq_infer_api_v1), SQ_INFER_ABI_MAJOR, SQ_INFER_ABI_MINOR, 0,
    si_backend_info,         si_defaults,        si_model_create,    si_model_plan,
    si_model_load,           si_model_info,      si_model_close,     si_session_create,
    si_session_plan,         si_session_prepare, si_session_info,    si_session_clear,
    si_session_close,        si_request_create,  si_request_run,     si_request_cancel,
    si_request_progress,     si_request_result,  si_request_close,   query_extension};

sq_status sq_infer_get_api(uint32_t major, uint32_t minor, const void **out) {
    if (!out)
        return SQ_INVALID_ARGUMENT;
    *out = NULL;
    if (sizeof(void *) != 8 || major != SQ_INFER_ABI_MAJOR || minor > SQ_INFER_ABI_MINOR)
        return SQ_INFER_ABI_MISMATCH;
    *out = &api;
    return SQ_OK;
}
