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
} ifn_infer_allocation;

static void *ifn_infer_local_budget_allocate(void *context, size_t bytes) {
    ifn_infer_budget *b = context;
    size_t total;
    if (!ifn_size_add(sizeof(ifn_infer_allocation), bytes, &total))
        return NULL;
    uint_fast64_t old = atomic_load_explicit(&b->live, memory_order_relaxed);
    do {
        if (old > b->limit || total > b->limit - old)
            return NULL;
    } while (!atomic_compare_exchange_weak_explicit(&b->live, &old, old + total,
                                                    memory_order_relaxed, memory_order_relaxed));
    ifn_infer_allocation *p = b->upstream.allocate(b->upstream.context, total);
    if (!p) {
        atomic_fetch_sub_explicit(&b->live, total, memory_order_relaxed);
        return NULL;
    }
    p->value.bytes = total;
    return p + 1;
}

static void ifn_infer_local_budget_release(void *context, void *ptr) {
    if (!ptr)
        return;
    ifn_infer_budget *b = context;
    ifn_infer_allocation *p = (ifn_infer_allocation *)ptr - 1;
    size_t total = p->value.bytes;
    b->upstream.release(b->upstream.context, p);
    atomic_fetch_sub_explicit(&b->live, total, memory_order_relaxed);
}

void ifn_infer_budget_init(ifn_infer_budget *b, const ifn_options *upstream, uint64_t limit,
                           ifn_options *out) {
    b->upstream = *upstream;
    b->limit = limit;
    atomic_init(&b->live, 0);
    *out = (ifn_options)IFN_OPTIONS_INIT;
    out->context = b;
    out->allocate = ifn_infer_local_budget_allocate;
    out->release = ifn_infer_local_budget_release;
}

void *ifn_infer_alloc(const ifn_options *a, size_t n) {
    return a->allocate(a->context, n);
}

void ifn_infer_free(const ifn_options *a, void *p) {
    if (p)
        a->release(a->context, p);
}

bool ifn_infer_lock(atomic_flag *f) {
    return !atomic_flag_test_and_set_explicit(f, memory_order_acquire);
}

void ifn_infer_unlock(atomic_flag *f) {
    atomic_flag_clear_explicit(f, memory_order_release);
}

ifn_status ifn_infer_input(const void *record, size_t known) {
    if (!record)
        return IFN_INVALID_ARGUMENT;
    ifn_infer_in h;
    memcpy(&h, record, sizeof h);
    if (h.struct_size < known || h.struct_size > IFN_INFER_RECORD_MAX_BYTES || h.flags)
        return IFN_INVALID_ARGUMENT;
    const unsigned char *p = record;
    for (size_t i = known; i < h.struct_size; i++)
        if (p[i])
            return IFN_UNSUPPORTED;
    return IFN_OK;
}

ifn_status ifn_infer_output(void *out, const void *value, size_t known) {
    if (!out)
        return IFN_INVALID_ARGUMENT;
    ifn_infer_out *h = out;
    uint32_t capacity = h->struct_size;
    if (capacity < known || capacity > IFN_INFER_RECORD_MAX_BYTES)
        return IFN_INVALID_ARGUMENT;
    memcpy(out, value, known);
    h->struct_size = capacity;
    h->written_size = (uint32_t)known;
    return IFN_OK;
}

ifn_status ifn_infer_error(ifn_infer_error_v1 *out, ifn_status status, uint32_t stage,
                           const char *text) {
    if (out) {
        ifn_infer_error_v1 e = {0};
        e.status = status;
        e.stage = status ? stage : IFN_STAGE_NONE;
        if (status && text) {
            (void)snprintf(e.message, sizeof e.message, "%s", text);
            e.message_bytes = (uint32_t)strlen(e.message);
        }
        (void)ifn_infer_output(out, &e, sizeof e);
    }
    return status;
}

ifn_status ifn_infer_native_error(ifn_infer_error_v1 *e, ifn_status rc, uint32_t stage) {
    /* Backend diagnostics contain operation names/codes, never input contents. */
    ifn_status status = ifn_infer_error(e, rc, stage, rc ? ifn_status_string(rc) : "");
    if (e && e->hdr.struct_size >= sizeof *e && e->hdr.struct_size <= IFN_INFER_RECORD_MAX_BYTES &&
        rc && stage == IFN_STAGE_CLOSE)
        e->flags |= IFN_INFER_ERR_CLEANUP_PENDING;
    return status;
}

bool ifn_infer_span(ifn_infer_bytes s, size_t max) {
    return s.bytes <= max && (!s.bytes || s.data);
}

bool ifn_infer_equal(ifn_infer_bytes s, const char *text) {
    return ifn_infer_span(s, 4096) && s.bytes == strlen(text) &&
           (!s.bytes || !memcmp(s.data, text, (size_t)s.bytes));
}

ifn_infer_bytes ifn_infer_text(const char *s) {
    return (ifn_infer_bytes){s, strlen(s)};
}

ifn_status ifn_infer_no_extensions(ifn_infer_array a) {
    return a.reserved             ? IFN_INVALID_ARGUMENT
           : a.count              ? IFN_UNSUPPORTED
           : (a.data || a.stride) ? IFN_INVALID_ARGUMENT
                                  : IFN_OK;
}

uint64_t ifn_infer_now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t))
        return 0;
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

ifn_status ifn_infer_defaults(uint32_t kind, void *out, ifn_infer_error_v1 *e) {
    size_t size = kind == IFN_DEFAULT_MODEL     ? sizeof(ifn_infer_model_config_v1)
                  : kind == IFN_DEFAULT_SESSION ? sizeof(ifn_infer_session_config_v1)
                  : kind == IFN_DEFAULT_REQUEST ? sizeof(ifn_infer_request_config_v1)
                                                : 0;
    if (!size || !out || ((ifn_infer_in *)out)->struct_size < size ||
        ((ifn_infer_in *)out)->struct_size > IFN_INFER_RECORD_MAX_BYTES)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE,
                               "invalid defaults record");
    uint32_t capacity = ((ifn_infer_in *)out)->struct_size;
    memset(out, 0, size);
    ((ifn_infer_in *)out)->struct_size = capacity;
    if (kind == IFN_DEFAULT_MODEL) {
        ifn_infer_model_config_v1 *c = out;
        c->max_owned_host_bytes = c->max_owned_device_bytes = UINT64_MAX;
        c->device_id = ifn_infer_text("0");
        c->weight_profile = ifn_infer_text("nvfp4-ar");
    } else if (kind == IFN_DEFAULT_SESSION) {
        ifn_infer_session_config_v1 *c = out;
        c->max_owned_host_bytes = c->max_owned_device_bytes = UINT64_MAX;
        c->context_tokens = 8192;
        c->execution_profile = ifn_infer_text("fp8-graph");
    } else {
        ifn_infer_request_config_v1 *c = out;
        c->max_output_tokens = 256;
        c->top_p = 0.95f;
        c->output_kind = IFN_OUTPUT_TEXT;
        c->max_event_bytes = 4096;
    }
    return ifn_infer_error(e, IFN_OK, 0, NULL);
}

ifn_status ifn_infer_backend_info(ifn_infer_backend_info_v1 *out, ifn_infer_error_v1 *e) {
    ifn_infer_backend_info_v1 v = {0};
    v.backend_id = ifn_infer_text("artalis.inkleberry");
    v.engine_revision =
        ifn_infer_text("upstream:d10c0abb2c8990be81c08ba84c66a8333a691b33;infer:1.0-experimental");
    v.compiled_features = IFN_INFER_FEATURES;
    v.pointer_bits = sizeof(void *) * 8;
    v.limit_scope = IFN_LIMIT_PROVIDER_PROCESS;
    v.max_prepared_sessions = v.max_active_runs = 1;
    return ifn_infer_error(e, ifn_infer_output(out, &v, sizeof v), IFN_STAGE_VALIDATE,
                           "invalid backend output");
}

static ifn_status ifn_infer_local_query_extension(uint64_t id, uint32_t version, const void **out,
                                                  ifn_infer_error_v1 *e) {
    (void)id;
    (void)version;
    if (out)
        *out = NULL;
    return ifn_infer_error(e, out ? IFN_UNSUPPORTED : IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE,
                           "no extension function table");
}

static const ifn_infer_api_v1 ifn_infer_local_api = {
    sizeof(ifn_infer_api_v1),   IFN_INFER_ABI_MAJOR,
    IFN_INFER_ABI_MINOR,        0,
    ifn_infer_backend_info,     ifn_infer_defaults,
    ifn_infer_model_create,     ifn_infer_model_plan,
    ifn_infer_model_load,       ifn_infer_model_info,
    ifn_infer_model_close,      ifn_infer_session_create,
    ifn_infer_session_plan,     ifn_infer_session_prepare,
    ifn_infer_session_info,     ifn_infer_session_clear,
    ifn_infer_session_close,    ifn_infer_request_create,
    ifn_infer_request_run,      ifn_infer_request_cancel,
    ifn_infer_request_progress, ifn_infer_request_result,
    ifn_infer_request_close,    ifn_infer_local_query_extension};

ifn_status ifn_infer_get_api(uint32_t major, uint32_t minor, const void **out) {
    if (!out)
        return IFN_INVALID_ARGUMENT;
    *out = NULL;
    if (sizeof(void *) != 8 || major != IFN_INFER_ABI_MAJOR || minor > IFN_INFER_ABI_MINOR)
        return IFN_INFER_ABI_MISMATCH;
    *out = &ifn_infer_local_api;
    return IFN_OK;
}
