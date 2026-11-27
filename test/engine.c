/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Control-flow/resource simulation only. Numerics require model-backed native qualification. */
#include "mock_gpu.h"
#include "engine_internal.h"
#include "ifn_infer.h"
#include "checked_prefill.h"
#include <math.h>
static int ifn_engine_local_zero_fault, ifn_engine_local_graph_fault, ifn_engine_local_blas_fault,
    ifn_engine_local_launch_fault;
static int ifn_engine_local_marker;
static int ifn_engine_local_pinned_fault, ifn_engine_local_event_fault,
    ifn_engine_local_pinned_free_fault, ifn_engine_local_event_free_fault;
static size_t ifn_engine_local_pinned_live, ifn_engine_local_events_live;

static void *ifn_engine_local_oom(void *c, size_t n) {
    (void)c;
    (void)n;
    return NULL;
}

static void ifn_engine_local_ignored_free(void *c, void *p) {
    (void)c;
    (void)p;
}

static size_t ifn_engine_local_allocation_calls, ifn_engine_local_allocation_live;

static void *ifn_engine_local_second_oom(void *context, size_t bytes) {
    (void)context;
    if (++ifn_engine_local_allocation_calls == 2)
        return NULL;
    void *p = malloc(bytes);
    if (p)
        ifn_engine_local_allocation_live++;
    return p;
}

static void ifn_engine_local_counted_free(void *context, void *p) {
    (void)context;
    if (p) {
        ifn_engine_local_allocation_live--;
        free(p);
    }
}

typedef struct {
    size_t calls, fail_at, live;
} ifn_engine_local_cache_allocator;

static void *ifn_engine_local_cache_allocate(void *context, size_t bytes) {
    ifn_engine_local_cache_allocator *a = context;
    if (++a->calls == a->fail_at)
        return NULL;
    void *p = malloc(bytes);
    if (p)
        a->live++;
    return p;
}

static void ifn_engine_local_cache_release(void *context, void *p) {
    ifn_engine_local_cache_allocator *a = context;
    if (p) {
        a->live--;
        free(p);
    }
}

static size_t ifn_engine_local_forward_calls;
static const void *ifn_engine_local_mock_count_pointer, *ifn_engine_local_mock_accepted_pointer;
static int32_t ifn_engine_local_mock_accepted_count = 1;

size_t ifn_sizeof_expert_groups(void) {
    return 4096;
}

int32_t ifn_cuda_memset_async(void *p, int32_t v, size_t n, void *s) {
    (void)p;
    (void)v;
    (void)n;
    (void)s;
    return ifn_engine_local_zero_fault;
}

int ifn_cuda_launch_status(void) {
    return ifn_engine_local_launch_fault;
}

int32_t ifn_cuda_graph_begin(void *s) {
    (void)s;
    return ifn_engine_local_graph_fault;
}

int32_t ifn_cuda_graph_end_instantiate(void *s, void **out) {
    (void)s;
    *out = ifn_engine_local_graph_fault ? NULL : &ifn_engine_local_marker;
    return ifn_engine_local_graph_fault;
}

int32_t ifn_cuda_graph_launch(void *g, void *s) {
    (void)g;
    (void)s;
    return ifn_engine_local_graph_fault;
}

int32_t ifn_cuda_graph_destroy(void *g) {
    (void)g;
    return ifn_engine_local_graph_fault;
}

int ifn_blas_destroy(void *p) {
    (void)p;
    return ifn_engine_local_blas_fault;
}

void ifn_ar_forward(ifn_session *s, int32_t n, bool lm) {
    (void)s;
    (void)n;
    (void)lm;
    ifn_engine_local_forward_calls++;
}

void ifn_ar_prefill(ifn_session *s, const int32_t *tokens, int32_t n) {
    (void)tokens;
    s->blas = &ifn_engine_local_marker;
    s->position += n;
}

static int ifn_engine_local_session_copy(void *p, void *dst, const void *src, size_t n, int kind,
                                         void *stream) {
    int rc = ifn_mock_gpu_local_copy(p, dst, src, n, kind, stream);
    if (!rc && kind == 1) {
        memset(dst, 0, n);
        if (src == ifn_engine_local_mock_count_pointer)
            memcpy(dst, &ifn_engine_local_mock_accepted_count, 4);
        else if (src == ifn_engine_local_mock_accepted_pointer)
            for (size_t i = 0; i < n / 4; i++) {
                int32_t token = 42;
                memcpy((unsigned char *)dst + i * 4, &token, 4);
            }
        else if (n == 4) {
            int32_t token = 42;
            memcpy(dst, &token, 4);
        }
    }
    return rc;
}

static ifn_gpu_weights *ifn_engine_local_image_with_mtp(ifn_mock_gpu_local_mock *m,
                                                        uint32_t with_mtp) {
    ifn_gpu_weights *w = NULL;
    ifn_weight_config c = IFN_WEIGHT_CONFIG_INIT;
    c.with_mtp = with_mtp;
    ifn_weight_backend b = ifn_mock_gpu_local_backend(m);
    b.copy = ifn_engine_local_session_copy;
    if (ifn_weights_create(&c, NULL, &b, &w) != IFN_OK)
        return NULL;
    if (ifn_mock_gpu_local_alloc_device(m, &w->arena, w->plan.arena_bytes)) {
        (void)ifn_weights_destroy(&w);
        return NULL;
    }
    w->ready = true;
    return w;
}

static ifn_gpu_weights *ifn_engine_local_image(ifn_mock_gpu_local_mock *m) {
    return ifn_engine_local_image_with_mtp(m, 0);
}

static int ifn_engine_local_no_memory(void *p, size_t *out) {
    (void)p;
    *out = 0;
    return 0;
}

static int ifn_engine_local_stopped(void *ctx, int32_t token) {
    (void)ctx;
    (void)token;
    return 0;
}

static int ifn_engine_local_cancelled_cb(void *ctx, int32_t token) {
    (void)token;
    (void)ifn_session_cancel(ctx);
    return 1;
}

static unsigned ifn_engine_local_drain_callback_count;

static int ifn_engine_local_fail_drain(void *ctx, int32_t token) {
    (void)token;
    ifn_mock_gpu_local_mock *m = ctx;
    if (++ifn_engine_local_drain_callback_count == 2) {
        m->fail_op = SYNC;
        m->fail_nth = m->calls[SYNC] + 1;
        return 0;
    }
    return 1;
}

static int ifn_engine_local_callback_busy(void *ctx, int32_t token) {
    (void)token;
    ifn_session *s = ctx, *copy_s = s;
    ifn_decode_info info = IFN_DECODE_INFO_INIT;
    ifn_prefix_info prefix = IFN_PREFIX_INFO_INIT;
    if (ifn_session_prefix_cache(s, 1) != IFN_BUSY ||
        ifn_session_prefix_info(s, &prefix) != IFN_BUSY)
        return 0;
    return ifn_session_decode_info(s, &info) == IFN_BUSY && ifn_session_reset(s) == IFN_BUSY &&
           ifn_session_destroy(&copy_s) == IFN_BUSY && copy_s == s;
}

static ifn_session *ifn_engine_local_late_cancel_session;
static size_t ifn_engine_local_late_cancel_sync;

static int ifn_engine_local_late_prefix_sync(void *context, void *stream) {
    ifn_mock_gpu_local_mock *m = context;
    int rc = ifn_mock_gpu_local_sync_stream(context, stream);
    if (ifn_engine_local_late_cancel_session &&
        m->calls[SYNC] == ifn_engine_local_late_cancel_sync) {
        (void)ifn_session_cancel(ifn_engine_local_late_cancel_session);
        ifn_engine_local_late_cancel_session = NULL;
    }
    return rc;
}

static int ifn_engine_local_arm_final_snapshot_cancel(void *context, int32_t token) {
    (void)token;
    ifn_session *session = context;
    ifn_mock_gpu_local_mock *m = session->weights->backend.context;
    ifn_engine_local_late_cancel_session = session;
    ifn_engine_local_late_cancel_sync = m->calls[SYNC] + 1;
    return 0;
}

typedef struct {
    ifn_session *session;
    ifn_status status;
} ifn_engine_local_generate_args;

static void *ifn_engine_local_generate_thread(void *p) {
    ifn_engine_local_generate_args *a = p;
    int32_t tokens[4] = {1, 2, 3, 4}, out[8];
    ifn_generate_stats stats;
    a->status = ifn_session_generate(a->session, tokens, 4, 8, out, 8, NULL, NULL, &stats);
    return NULL;
}

static bool ifn_engine_local_request_cancelled(void *context) {
    return atomic_load_explicit((atomic_bool *)context, memory_order_acquire);
}

static int ifn_engine_local_request_cancel_token(void *context, int32_t token) {
    (void)token;
    atomic_store_explicit((atomic_bool *)context, true, memory_order_release);
    return 1;
}

int main(void) {
    ifn_visit visits[48];
    ifn_ar_schedule(visits);
    for (unsigned i = 0; i < 48; i++) {
        unsigned layer = i < 6 ? i : i < 42 ? 6 + (i - 6) % 12 : i - 24;
        unsigned pass = i >= 6 && i < 42 ? (i - 6) / 12 : 0;
        IFN_MOCK_GPU_LOCAL_CHECK(visits[i].layer == layer && visits[i].pass == pass &&
                                 visits[i].loop == (i >= 6 && i < 42) &&
                                 visits[i].global == ((layer + 1) % 6 == 0));
    }
    IFN_MOCK_GPU_LOCAL_CHECK(!strcmp(ifn_status_string(IFN_CANCELLED), "cancelled"));
    ifn_session_config c = IFN_SESSION_CONFIG_INIT;
    c.context_tokens = 128;
    ifn_session *s = NULL;
    ifn_mock_gpu_local_mock m;
    ifn_mock_gpu_local_reset(&m);
    ifn_gpu_weights *w = ifn_engine_local_image(&m);
    IFN_MOCK_GPU_LOCAL_CHECK(w);
    c.kv_format = 3;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_INVALID_ARGUMENT && !s);
    c.kv_format = 1;
    ifn_options oom_options = IFN_OPTIONS_INIT;
    oom_options.allocate = ifn_engine_local_oom;
    oom_options.release = ifn_engine_local_ignored_free;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, &oom_options, &s) == IFN_OUT_OF_MEMORY &&
                             !s);
    ifn_session_config bad = c;
    bad.struct_size = 0;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &bad, NULL, &s) == IFN_INVALID_ARGUMENT && !s);
    bad = c;
    bad.api_version = 0;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &bad, NULL, &s) == IFN_INVALID_ARGUMENT && !s);
    bad = c;
    bad.reserved = 1;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &bad, NULL, &s) == IFN_INVALID_ARGUMENT && !s);
    bad = c;
    bad.context_tokens = 1048577;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &bad, NULL, &s) == IFN_INVALID_ARGUMENT && !s);
    w->backend.available = ifn_engine_local_no_memory;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OUT_OF_MEMORY && s &&
                             !s->arena && !s->stream);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK && !s && m.live == 1);
    w->backend.available = ifn_mock_gpu_local_available;
    uint32_t view_type = w->views[0].type;
    w->views[0].type = IFN_WEIGHT_EXPERT;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_FORMAT && s);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK && !s && m.live == 1);
    w->views[0].type = view_type;
    for (unsigned format = 0; format < 3; format++) {
        c.kv_format = format;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK && s);
        IFN_MOCK_GPU_LOCAL_CHECK(w->session_users == 1 && s->offset == s->capacity && m.live == 2);
        ifn_decode_options unsupported = IFN_DECODE_OPTIONS_INIT;
        unsupported.draft_tokens = 1;
        ifn_decode_stats unsupported_stats;
        int32_t unsupported_in = 1, unsupported_out = 0;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate_ex(s, &unsupported_in, 1, 1, &unsupported,
                                                         &unsupported_out, 1, NULL, NULL,
                                                         &unsupported_stats) == IFN_UNSUPPORTED &&
                                 !s->poisoned);

        ifn_session *other = NULL;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &other) == IFN_BUSY && !other);
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_destroy(&w) == IFN_BUSY && w && w->ready);
        int32_t tokens[40] = {1, 2, 3, 4}, out[8];
        ifn_generate_stats stats;
        atomic_bool request_cancel;
        atomic_init(&request_cancel, true);
        ifn_run_control control = {ifn_engine_local_request_cancelled, &request_cancel};
        ifn_decode_stats controlled_stats;
        size_t before_controlled = ifn_engine_local_forward_calls;
        IFN_MOCK_GPU_LOCAL_CHECK(
            ifn_session_generate_controlled(s, tokens, 4, 8, NULL, NULL, NULL, &controlled_stats,
                                            &control) == IFN_CANCELLED &&
            ifn_engine_local_forward_calls == before_controlled && !s->request_control);
        atomic_store(&request_cancel, false);
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate_controlled(
                                     s, tokens, 4, 8, NULL, ifn_engine_local_request_cancel_token,
                                     (void *)&request_cancel, &controlled_stats,
                                     &control) == IFN_CANCELLED &&
                                 controlled_stats.generation.generated_tokens == 1 &&
                                 !s->request_control && s->position == 0 && !s->poisoned);
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8,
                                                      ifn_engine_local_callback_busy, s,
                                                      &stats) == IFN_OK &&
                                 stats.generated_tokens == 8);
        IFN_MOCK_GPU_LOCAL_CHECK(
            ifn_session_generate(s, tokens, 40, 8, out, 8, ifn_engine_local_stopped, NULL,
                                 &stats) == IFN_OK &&
            stats.finish_reason == IFN_FINISH_CALLBACK && stats.generated_tokens == 1);
        IFN_MOCK_GPU_LOCAL_CHECK(
            ifn_session_generate(s, tokens, 4, 8, out, 8, ifn_engine_local_cancelled_cb, s,
                                 &stats) == IFN_CANCELLED &&
            stats.finish_reason == IFN_FINISH_CANCELLED && s->position == 0 && !s->poisoned);
        int32_t one = 1;
        float unused = 0;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate(s, &one, SIZE_MAX, 1, out, 8, NULL, NULL,
                                                      &stats) == IFN_INVALID_ARGUMENT);
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_logits(s, &one, SIZE_MAX, 1, &unused, 200058) ==
                                 IFN_INVALID_ARGUMENT);
        int32_t invalid = -1;
        size_t prior = ifn_engine_local_forward_calls;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate(s, &invalid, 1, 8, out, 8, NULL, NULL,
                                                      &stats) == IFN_INVALID_ARGUMENT &&
                                 ifn_engine_local_forward_calls == prior);
        IFN_MOCK_GPU_LOCAL_CHECK(
            ifn_session_generate(s, tokens, 4, 8, out, 8, NULL, NULL, &stats) == IFN_OK &&
            stats.generated_tokens == 8);
        s->position = 128;
        float value = 0;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_logits(s, tokens, 1, 0, &value, 200058) == IFN_LIMIT &&
                                 !s->poisoned && m.device == 42);
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK && !s && !w->session_users &&
                                 m.live == 1);
    }
    /* Acquisition faults: preserve caller ownership and allow deterministic cleanup. */
    int ops[] = {IFN_MOCK_GPU_LOCAL_ENTER,
                 IFN_MOCK_GPU_LOCAL_AVAILABLE,
                 CREATE,
                 ALLOC,
                 SYNC,
                 IFN_MOCK_GPU_LOCAL_COPY,
                 LEAVE};
    for (size_t i = 0; i < sizeof ops / sizeof *ops; i++) {
        m.fail_op = ops[i];
        m.fail_nth = m.calls[ops[i]] + 1;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) != IFN_OK && s);
        m.fail_op = -1;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK && !s && m.live == 1 &&
                                 !w->session_users);
    }
    ifn_engine_local_zero_fault = 700;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_GPU_ERROR && s);
    ifn_engine_local_zero_fault = 0;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK);
    for (unsigned op = 0; op < 4; op++) {
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
        if (op < 2) {
            m.fail_op = op ? IFN_MOCK_GPU_LOCAL_DESTROY : IFN_MOCK_GPU_LOCAL_FREE;
            m.persistent = true;
        }
        if (op == 2)
            s->blas = &ifn_engine_local_marker, ifn_engine_local_blas_fault = 700;
        if (op == 3)
            s->graph = &ifn_engine_local_marker, ifn_engine_local_graph_fault = 700;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_GPU_ERROR && s && w->session_users);
        m.fail_op = -1;
        m.persistent = false;
        ifn_engine_local_blas_fault = ifn_engine_local_graph_fault = 0;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK && !s && m.live == 1);
    }
    /* A pre-cancelled request cannot hide a failed device acquisition. */
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
    atomic_bool pre_cancel;
    atomic_init(&pre_cancel, true);
    ifn_run_control pre_control = {ifn_engine_local_request_cancelled, &pre_cancel};
    ifn_decode_stats pre_stats;
    int32_t pre_tokens[2] = {1, 2};
    m.fail_op = IFN_MOCK_GPU_LOCAL_ENTER;
    m.fail_nth = m.calls[IFN_MOCK_GPU_LOCAL_ENTER] + 1;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate_controlled(s, pre_tokens, 2, 8, NULL, NULL, NULL,
                                                             &pre_stats,
                                                             &pre_control) == IFN_GPU_ERROR &&
                             s->poisoned && !s->request_control);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_native_diagnostic_get().provider == IFN_PROVIDER_CUDA &&
                             ifn_native_diagnostic_get().code == 700);
    m.fail_op = -1;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK);
    /* MTP-only acquisition and retained cleanup failures. */
    /* GPU execution errors poison; validation failures do not. */
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
    ifn_engine_local_launch_fault = 700;
    int32_t tokens[4] = {1, 2, 3, 4}, out[8];
    ifn_generate_stats stats;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8, NULL, NULL, &stats) ==
                                 IFN_GPU_ERROR &&
                             s->poisoned);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_native_diagnostic_get().provider == IFN_PROVIDER_CUDA &&
                             ifn_native_diagnostic_get().code == 700);
    ifn_engine_local_launch_fault = 0;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_reset(s) == IFN_GPU_ERROR);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
    m.fail_op = LEAVE;
    m.fail_nth = m.calls[LEAVE] + 1;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8,
                                                  ifn_engine_local_cancelled_cb, s,
                                                  &stats) == IFN_GPU_ERROR &&
                             s->poisoned && stats.finish_reason == IFN_FINISH_CANCELLED);
    m.fail_op = -1;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK);
    m.device = 42;
    m.context = &m.context_marker;
    /* An independent thread cancels a generation blocked in the private backend. */
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
    atomic_store(&m.pause, 1);
    atomic_store(&m.arrived, 0);
    atomic_store(&m.resume, 0);
    ifn_engine_local_generate_args args = {s, IFN_OK};
    ifn_test_thread thread;
    IFN_MOCK_GPU_LOCAL_CHECK(
        !ifn_test_thread_create(&thread, NULL, ifn_engine_local_generate_thread, &args));
    while (!atomic_load(&m.arrived))
        ifn_test_yield();
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_reset(s) == IFN_BUSY &&
                             ifn_session_destroy(&s) == IFN_BUSY && s);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_cancel(s) == IFN_OK);
    atomic_store(&m.resume, 1);
    IFN_MOCK_GPU_LOCAL_CHECK(!ifn_test_thread_join(thread, NULL) && args.status == IFN_CANCELLED);
    atomic_store(&m.pause, 0);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_reset(s) == IFN_OK);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_destroy(&w) == IFN_OK && !w && !m.live);
    ifn_mock_gpu_local_reset(&m);
    w = ifn_engine_local_image_with_mtp(&m, 1);
    IFN_MOCK_GPU_LOCAL_CHECK(w);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK &&
                             s->model.dspark.loaded && s->ds_f_ring && s->p_taps);
    ifn_decode_options options = IFN_DECODE_OPTIONS_INIT;
    options.temperature = 0.8f;
    options.seed = UINT64_MAX;
    options.draft_tokens = 3;
    ifn_decode_stats detail;
    ifn_engine_local_mock_accepted_count = 2;
    IFN_MOCK_GPU_LOCAL_CHECK(
        ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, NULL, NULL, &detail) == IFN_OK &&
        detail.generation.generated_tokens == 8 && detail.speculative_steps == 4 &&
        detail.accepted_drafts == 4);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8,
                                                     ifn_engine_local_cancelled_cb, s,
                                                     &detail) == IFN_CANCELLED &&
                             !s->poisoned);
    options.adaptive = 1;
    options.draft_tokens = 7;
    ifn_decode_info info = IFN_DECODE_INFO_INIT;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_decode_info(s, &info) == IFN_OK);
    info.struct_size = 0;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_decode_info(s, &info) == IFN_INVALID_ARGUMENT);
    info = (ifn_decode_info)IFN_DECODE_INFO_INIT;
    IFN_MOCK_GPU_LOCAL_CHECK(
        ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, NULL, NULL, &detail) == IFN_OK &&
        !s->draft_in_flight && !s->poisoned);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_decode_info(s, &info) == IFN_OK && info.width2 > 0);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8,
                                                     ifn_engine_local_cancelled_cb, s,
                                                     &detail) == IFN_CANCELLED &&
                             !s->draft_in_flight && !s->poisoned);
    c.use_graphs = 0;
    s->config.use_graphs = 0;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, NULL, NULL,
                                                     &detail) == IFN_UNSUPPORTED &&
                             !s->poisoned);
    c.use_graphs = s->config.use_graphs = 1;
    /* Real scheduler drain preserves the first error and retains uncertain work. */
    s->draft_in_flight = true;
    s->status = IFN_OK;
    m.fail_op = SYNC;
    m.fail_nth = m.calls[SYNC] + 1;
    ifn_adaptive_drain(s);
    IFN_MOCK_GPU_LOCAL_CHECK(s->status == IFN_GPU_ERROR && s->draft_in_flight);
    s->status = IFN_OUT_OF_MEMORY;
    m.fail_nth = m.calls[SYNC] + 1;
    ifn_adaptive_drain(s);
    IFN_MOCK_GPU_LOCAL_CHECK(s->status == IFN_OUT_OF_MEMORY && s->draft_in_flight);
    m.fail_op = -1;
    ifn_adaptive_drain(s);
    IFN_MOCK_GPU_LOCAL_CHECK(s->status == IFN_OUT_OF_MEMORY && !s->draft_in_flight);
    s->status = IFN_OK;
    options.adaptive = 0;
    options.temperature = NAN;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, NULL, NULL,
                                                     &detail) == IFN_INVALID_ARGUMENT &&
                             !s->poisoned);
    options.temperature = 0;
    options.draft_tokens = 7;
    int32_t long_tokens[128] = {1, 2, 3, 4};
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate_ex(s, long_tokens, 121, 7, &options, out, 8, NULL,
                                                     NULL, &detail) == IFN_OK &&
                             detail.speculative_steps == 0 &&
                             detail.generation.generated_tokens == 7);
    ifn_engine_local_mock_accepted_count = 0;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, NULL, NULL,
                                                     &detail) == IFN_GPU_ERROR &&
                             s->poisoned);
    ifn_engine_local_graph_fault = 700;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_GPU_ERROR && s && s->arena);
    ifn_engine_local_graph_fault = 0;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK && !s);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_destroy(&w) == IFN_OK && !w && !m.live);
    ifn_mock_gpu_local_reset(&m);
    w = ifn_engine_local_image_with_mtp(&m, 1);
    IFN_MOCK_GPU_LOCAL_CHECK(w);
    ifn_options fail_host = IFN_OPTIONS_INIT;
    fail_host.allocate = ifn_engine_local_second_oom;
    fail_host.release = ifn_engine_local_counted_free;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, &fail_host, &s) == IFN_OUT_OF_MEMORY && s &&
                             !s->adaptive_state && !s->stream);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK &&
                             !ifn_engine_local_allocation_live && !w->session_users);
    for (int failure = 0; failure < 2; failure++) {
        ifn_engine_local_pinned_fault = failure == 0;
        ifn_engine_local_event_fault = failure == 1;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) != IFN_OK && s);
        ifn_engine_local_pinned_fault = ifn_engine_local_event_fault = 0;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK &&
                                 !ifn_engine_local_pinned_live && !ifn_engine_local_events_live);
    }
    for (int failure = 0; failure < 3; failure++) {
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
        if (failure == 0) {
            *(int *)s->adaptive_state = 1;
            ifn_engine_local_graph_fault = 700;
        } else if (failure == 1)
            ifn_engine_local_pinned_free_fault = 700;
        else
            ifn_engine_local_event_free_fault = 700;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_GPU_ERROR && s && s->stream);
        if (failure == 0)
            IFN_MOCK_GPU_LOCAL_CHECK(s->arena && s->pinned && s->done_event && s->adaptive_state);
        ifn_engine_local_graph_fault = ifn_engine_local_pinned_free_fault =
            ifn_engine_local_event_free_fault = 0;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK && !s &&
                                 !ifn_engine_local_pinned_live && !ifn_engine_local_events_live);
    }
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
    options.adaptive = 1;
    ifn_engine_local_mock_accepted_count = 1;
    IFN_MOCK_GPU_LOCAL_CHECK(
        ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, ifn_engine_local_fail_drain, &m,
                                &detail) == IFN_GPU_ERROR &&
        s->poisoned && s->draft_in_flight && detail.generation.generated_tokens == 2 &&
        strstr(ifn_last_error(), "drain"));
    m.fail_op = -1;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_decode_info(s, &info) == IFN_OK && info.width2 == 1);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_reset(s) == IFN_GPU_ERROR);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK && !ifn_engine_local_pinned_live &&
                             !ifn_engine_local_events_live);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_destroy(&w) == IFN_OK && !m.live);
    ifn_engine_local_mock_count_pointer = ifn_engine_local_mock_accepted_pointer = NULL;
    ifn_engine_local_mock_accepted_count = 1;
    ifn_mock_gpu_local_reset(&m);
    w = ifn_engine_local_image_with_mtp(&m, 1);
    IFN_MOCK_GPU_LOCAL_CHECK(w);
    c.context_tokens = 2048;
    ifn_engine_local_cache_allocator a = {0};
    ifn_options cache_options = IFN_OPTIONS_INIT;
    cache_options.context = &a;
    cache_options.allocate = ifn_engine_local_cache_allocate;
    cache_options.release = ifn_engine_local_cache_release;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_create(w, &c, &cache_options, &s) == IFN_OK &&
                             a.live == 2);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_prefix_cache(s, 2) == IFN_INVALID_ARGUMENT);
    for (unsigned i = 1; i <= 6; i++) {
        a.fail_at = a.calls + i;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_prefix_cache(s, 1) == IFN_OUT_OF_MEMORY &&
                                 a.live == 2 && !s->prefix.enabled && !s->prefix.tokens);
        a.fail_at = 0;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_prefix_cache(s, 1) == IFN_OK && a.live == 8);
        size_t calls = a.calls;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_prefix_cache(s, 1) == IFN_OK && a.calls == calls);
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_prefix_cache(s, 0) == IFN_OK && a.live == 2);
    }
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_prefix_cache(s, 1) == IFN_OK);
    ifn_prefix_info pi = IFN_PREFIX_INFO_INIT;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_prefix_info(s, &pi) == IFN_OK && pi.host_bytes > 0);
    pi.struct_size = 0;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_prefix_info(s, &pi) == IFN_INVALID_ARGUMENT);
    pi = (ifn_prefix_info)IFN_PREFIX_INFO_INIT;
    size_t cache_calls = a.calls;
    memcpy(s->prefix.tokens, tokens, 4 * 4);
    s->prefix.length = 4;
    s->prefix.live = 2;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8,
                                                  ifn_engine_local_callback_busy, s,
                                                  &stats) == IFN_OK);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_prefix_info(s, &pi) == IFN_OK && pi.reused_tokens == 2 &&
                             !pi.checkpoint_restores && a.calls == cache_calls);
    s->prefix.length = 1024;
    s->prefix.live = -1;
    for (unsigned i = 0; i < 1024; i++)
        s->prefix.tokens[i] = 5;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8, NULL, NULL, &stats) ==
                             IFN_OK);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_prefix_info(s, &pi) == IFN_OK &&
                             pi.shielded_tokens == 28 && s->prefix.length == 1024 &&
                             s->prefix.live == -1);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8,
                                                  ifn_engine_local_cancelled_cb, s,
                                                  &stats) == IFN_CANCELLED &&
                             s->prefix.length == 0);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_prefix_cache(s, 0) == IFN_OK &&
                             ifn_session_prefix_cache(s, 1) == IFN_OK);
    int32_t prompt[129];
    for (unsigned i = 0; i < 129; i++)
        prompt[i] = 1;
    w->backend.sync = ifn_engine_local_late_prefix_sync;
    IFN_MOCK_GPU_LOCAL_CHECK(
        ifn_session_generate(s, prompt, 129, 8, out, 8, ifn_engine_local_arm_final_snapshot_cancel,
                             s, &stats) == IFN_CANCELLED &&
        !ifn_engine_local_late_cancel_session && !s->prefix.length && s->prefix.live == -1);
    for (unsigned i = 0; i < 4; i++)
        IFN_MOCK_GPU_LOCAL_CHECK(!s->prefix.checkpoints[i].valid);
    w->backend.sync = ifn_mock_gpu_local_sync_stream;

    for (unsigned i = 0; i < 256; i++)
        s->prefix.tokens[i] = 1;
    s->prefix.length = 256;
    s->prefix.live = -1;
    s->prefix.checkpoints[0].position = 128;
    s->prefix.checkpoints[0].valid = true;
    s->prefix.use_mtp = false;
    m.fail_op = IFN_MOCK_GPU_LOCAL_COPY;
    m.fail_nth = m.calls[IFN_MOCK_GPU_LOCAL_COPY] + 1;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_generate(s, prompt, 129, 8, out, 8, NULL, NULL, &stats) ==
                                 IFN_GPU_ERROR &&
                             s->poisoned && !s->prefix.length);
    m.fail_op = -1;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_session_destroy(&s) == IFN_OK && a.live == 0 &&
                             !ifn_engine_local_pinned_live && !ifn_engine_local_events_live);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_destroy(&w) == IFN_OK && !m.live);
    puts("AR session simulation: bounds, retention, faults, poisoning, cleanup retry, callback "
         "BUSY and cross-thread cancellation passed");
    return 0;
}

void ifn_spec_step(ifn_session *s, uint32_t drafts) {
    ifn_engine_local_mock_count_pointer = s->d_num_acc;
    ifn_engine_local_mock_accepted_pointer = s->d_accepted;
    if (s->config.use_graphs)
        s->spec_graph[drafts] = &ifn_engine_local_marker;
}

size_t ifn_adaptive_bytes(void) {
    return 32;
}

void ifn_adaptive_init(void *p) {
    memset(p, 0, 32);
}

int ifn_adaptive_live(const void *p) {
    return *(const int *)p;
}

int ifn_adaptive_cleanup(void *p) {
    if (ifn_engine_local_graph_fault)
        return ifn_engine_local_graph_fault;
    memset(p, 0, 32);
    return 0;
}

static int32_t *ifn_engine_local_adaptive_accepted, *ifn_engine_local_adaptive_count,
    *ifn_engine_local_adaptive_width;

int ifn_adaptive_prepare(void *state, const int32_t *anchor, const int32_t *drafts,
                         const float *confidence, float threshold, int32_t *tokens, int32_t *width,
                         const int32_t *accepted, int32_t *host_accepted, const int32_t *count,
                         int32_t *host_count, int32_t *host_width, void *stream) {
    (void)anchor;
    (void)drafts;
    (void)confidence;
    (void)threshold;
    (void)tokens;
    (void)width;
    (void)accepted;
    (void)count;
    (void)stream;
    *(int *)state = 1;
    ifn_engine_local_adaptive_accepted = host_accepted;
    ifn_engine_local_adaptive_count = host_count;
    ifn_engine_local_adaptive_width = host_width;
    return ifn_engine_local_graph_fault;
}

int ifn_adaptive_branch_begin(void *state, int branch, void *stream) {
    (void)state;
    (void)branch;
    (void)stream;
    return ifn_engine_local_graph_fault;
}

int ifn_adaptive_branch_end(void *state) {
    (void)state;
    return ifn_engine_local_graph_fault;
}

int ifn_adaptive_instantiate(void *state) {
    (void)state;
    return ifn_engine_local_graph_fault;
}

int ifn_adaptive_launch(void *state, void *stream) {
    (void)state;
    (void)stream;
    *ifn_engine_local_adaptive_count = 1;
    *ifn_engine_local_adaptive_width = 2;
    ifn_engine_local_adaptive_accepted[0] = 42;
    return ifn_engine_local_graph_fault;
}

void ifn_spec_draft(ifn_session *s, uint32_t n, float *confidence) {
    (void)s;
    (void)n;
    (void)confidence;
}

void ifn_spec_verify(ifn_session *s, uint32_t n) {
    (void)s;
    (void)n;
}

void ifn_spec_target(ifn_session *s, uint32_t n) {
    (void)s;
    (void)n;
}

int32_t ifn_cuda_memcpy_d2h_async(void *dst, const void *src, size_t n, void *stream) {
    (void)dst;
    (void)src;
    (void)n;
    (void)stream;
    return ifn_engine_local_graph_fault;
}

int32_t ifn_cuda_event_record(void *event, void *stream) {
    (void)event;
    (void)stream;
    return ifn_engine_local_graph_fault;
}

int32_t ifn_cuda_event_sync(void *event) {
    (void)event;
    return ifn_engine_local_graph_fault;
}

int32_t ifn_cuda_malloc_host_pinned(void **out, size_t n, uint8_t mapped) {
    (void)mapped;
    if (ifn_engine_local_pinned_fault) {
        *out = NULL;
        return 2;
    }
    *out = calloc(1, n);
    if (*out)
        ifn_engine_local_pinned_live++;
    return *out ? 0 : 2;
}

int32_t ifn_cuda_free_host(void *p) {
    if (ifn_engine_local_pinned_free_fault)
        return ifn_engine_local_pinned_free_fault;
    free(p);
    ifn_engine_local_pinned_live--;
    return 0;
}

int32_t ifn_cuda_event_create(void **out, uint8_t disable) {
    (void)disable;
    if (ifn_engine_local_event_fault) {
        *out = NULL;
        return 2;
    }
    *out = &ifn_engine_local_marker;
    ifn_engine_local_events_live++;
    return 0;
}

int32_t ifn_cuda_event_destroy(void *p) {
    (void)p;
    if (ifn_engine_local_event_free_fault)
        return ifn_engine_local_event_free_fault;
    ifn_engine_local_events_live--;
    return 0;
}
