/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Control-flow/resource simulation only. Numerics require real GPU oracle. */
#include "mock_gpu.h"
#include "engine_internal.h"
#include "checked_prefill.h"
#include <math.h>
static int zero_fault, graph_fault, blas_fault, launch_fault;
static int marker;
static int pinned_fault, event_fault, pinned_free_fault, event_free_fault;
static size_t pinned_live, events_live;

static void *oom(void *c, size_t n) {
    (void)c;
    (void)n;
    return NULL;
}

static void ignored_free(void *c, void *p) {
    (void)c;
    (void)p;
}

static size_t allocation_calls, allocation_live;

static void *second_oom(void *context, size_t bytes) {
    (void)context;
    if (++allocation_calls == 2)
        return NULL;
    void *p = malloc(bytes);
    if (p)
        allocation_live++;
    return p;
}

static void counted_free(void *context, void *p) {
    (void)context;
    if (p) {
        allocation_live--;
        free(p);
    }
}

typedef struct {
    size_t calls, fail_at, live;
} cache_allocator;

static void *cache_allocate(void *context, size_t bytes) {
    cache_allocator *a = context;
    if (++a->calls == a->fail_at)
        return NULL;
    void *p = malloc(bytes);
    if (p)
        a->live++;
    return p;
}

static void cache_release(void *context, void *p) {
    cache_allocator *a = context;
    if (p) {
        a->live--;
        free(p);
    }
}

static size_t forward_calls;
static const void *mock_count_pointer, *mock_accepted_pointer;
static int32_t mock_accepted_count = 1;

size_t squid_sizeof_expert_groups(void) {
    return 4096;
}

int32_t squid_cuda_memset_async(void *p, int32_t v, size_t n, void *s) {
    (void)p;
    (void)v;
    (void)n;
    (void)s;
    return zero_fault;
}

int ifn_cuda_launch_status(void) {
    return launch_fault;
}

int32_t squid_cuda_graph_begin(void *s) {
    (void)s;
    return graph_fault;
}

int32_t squid_cuda_graph_end_instantiate(void *s, void **out) {
    (void)s;
    *out = graph_fault ? NULL : &marker;
    return graph_fault;
}

int32_t squid_cuda_graph_launch(void *g, void *s) {
    (void)g;
    (void)s;
    return graph_fault;
}

int32_t squid_cuda_graph_destroy(void *g) {
    (void)g;
    return graph_fault;
}

int ifn_blas_destroy(void *p) {
    (void)p;
    return blas_fault;
}

void ifn_ar_forward(ifn_session *s, int32_t n, bool lm) {
    (void)s;
    (void)n;
    (void)lm;
    forward_calls++;
}

void ifn_ar_prefill(ifn_session *s, const int32_t *tokens, int32_t n) {
    (void)tokens;
    s->blas = &marker;
    s->position += n;
}

static int session_copy(void *p, void *dst, const void *src, size_t n, int kind, void *stream) {
    int rc = copy(p, dst, src, n, kind, stream);
    if (!rc && kind == 1) {
        memset(dst, 0, n);
        if (src == mock_count_pointer)
            memcpy(dst, &mock_accepted_count, 4);
        else if (src == mock_accepted_pointer)
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

static ifn_gpu_weights *image_with_mtp(mock *m, uint32_t with_mtp) {
    ifn_gpu_weights *w = NULL;
    ifn_weight_config c = IFN_WEIGHT_CONFIG_INIT;
    c.with_mtp = with_mtp;
    ifn_weight_backend b = backend(m);
    b.copy = session_copy;
    if (ifn_weights_create(&c, NULL, &b, &w) != IFN_OK)
        return NULL;
    if (alloc_device(m, &w->arena, w->plan.arena_bytes)) {
        (void)ifn_weights_destroy(&w);
        return NULL;
    }
    w->ready = true;
    return w;
}

static ifn_gpu_weights *image(mock *m) {
    return image_with_mtp(m, 0);
}

static int no_memory(void *p, size_t *out) {
    (void)p;
    *out = 0;
    return 0;
}

static int stopped(void *ctx, int32_t token) {
    (void)ctx;
    (void)token;
    return 0;
}

static int cancelled_cb(void *ctx, int32_t token) {
    (void)token;
    (void)ifn_session_cancel(ctx);
    return 1;
}

static unsigned drain_callback_count;

static int fail_drain(void *ctx, int32_t token) {
    (void)token;
    mock *m = ctx;
    if (++drain_callback_count == 2) {
        m->fail_op = SYNC;
        m->fail_nth = m->calls[SYNC] + 1;
        return 0;
    }
    return 1;
}

static int callback_busy(void *ctx, int32_t token) {
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

static ifn_session *late_cancel_session;
static size_t late_cancel_sync;

static int late_prefix_sync(void *context, void *stream) {
    mock *m = context;
    int rc = sync_stream(context, stream);
    if (late_cancel_session && m->calls[SYNC] == late_cancel_sync) {
        (void)ifn_session_cancel(late_cancel_session);
        late_cancel_session = NULL;
    }
    return rc;
}

static int arm_final_snapshot_cancel(void *context, int32_t token) {
    (void)token;
    ifn_session *session = context;
    mock *m = session->weights->backend.context;
    late_cancel_session = session;
    late_cancel_sync = m->calls[SYNC] + 1;
    return 0;
}

typedef struct {
    ifn_session *session;
    ifn_status status;
} generate_args;

static void *generate_thread(void *p) {
    generate_args *a = p;
    int32_t tokens[4] = {1, 2, 3, 4}, out[8];
    ifn_generate_stats stats;
    a->status = ifn_session_generate(a->session, tokens, 4, 8, out, 8, NULL, NULL, &stats);
    return NULL;
}

static bool request_cancelled(void *context) {
    return atomic_load_explicit((atomic_bool *)context, memory_order_acquire);
}

static int request_cancel_token(void *context, int32_t token) {
    (void)token;
    atomic_store_explicit((atomic_bool *)context, true, memory_order_release);
    return 1;
}

int main(void) {
    CHECK(!strcmp(ifn_status_string(IFN_CANCELLED), "cancelled"));
    ifn_session_config c = IFN_SESSION_CONFIG_INIT;
    c.context_tokens = 128;
    ifn_session *s = NULL;
    mock m;
    reset(&m);
    ifn_gpu_weights *w = image(&m);
    CHECK(w);
    c.kv_format = 3;
    CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_INVALID_ARGUMENT && !s);
    c.kv_format = 1;
    ifn_options oom_options = IFN_OPTIONS_INIT;
    oom_options.allocate = oom;
    oom_options.release = ignored_free;
    CHECK(ifn_session_create(w, &c, &oom_options, &s) == IFN_OUT_OF_MEMORY && !s);
    ifn_session_config bad = c;
    bad.struct_size = 0;
    CHECK(ifn_session_create(w, &bad, NULL, &s) == IFN_INVALID_ARGUMENT && !s);
    bad = c;
    bad.api_version = 0;
    CHECK(ifn_session_create(w, &bad, NULL, &s) == IFN_INVALID_ARGUMENT && !s);
    bad = c;
    bad.reserved = 1;
    CHECK(ifn_session_create(w, &bad, NULL, &s) == IFN_INVALID_ARGUMENT && !s);
    bad = c;
    bad.context_tokens = 1048577;
    CHECK(ifn_session_create(w, &bad, NULL, &s) == IFN_INVALID_ARGUMENT && !s);
    w->backend.available = no_memory;
    CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OUT_OF_MEMORY && s && !s->arena && !s->stream);
    CHECK(ifn_session_destroy(&s) == IFN_OK && !s && m.live == 1);
    w->backend.available = available;
    uint32_t view_type = w->views[0].type;
    w->views[0].type = IFN_WEIGHT_EXPERT;
    CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_FORMAT && s);
    CHECK(ifn_session_destroy(&s) == IFN_OK && !s && m.live == 1);
    w->views[0].type = view_type;
    for (unsigned format = 0; format < 3; format++) {
        c.kv_format = format;
        CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK && s);
        CHECK(w->session_users == 1 && s->offset == s->capacity && m.live == 2);
        ifn_decode_options unsupported = IFN_DECODE_OPTIONS_INIT;
        unsupported.draft_tokens = 1;
        ifn_decode_stats unsupported_stats;
        int32_t unsupported_in = 1, unsupported_out = 0;
        CHECK(ifn_session_generate_ex(s, &unsupported_in, 1, 1, &unsupported, &unsupported_out, 1,
                                      NULL, NULL, &unsupported_stats) == IFN_UNSUPPORTED &&
              !s->poisoned);

        ifn_session *other = NULL;
        CHECK(ifn_session_create(w, &c, NULL, &other) == IFN_BUSY && !other);
        CHECK(ifn_weights_destroy(&w) == IFN_BUSY && w && w->ready);
        int32_t tokens[40] = {1, 2, 3, 4}, out[8];
        ifn_generate_stats stats;
        atomic_bool request_cancel;
        atomic_init(&request_cancel, true);
        ifn_run_control control = {request_cancelled, &request_cancel};
        ifn_decode_stats controlled_stats;
        size_t before_controlled = forward_calls;
        CHECK(ifn_session_generate_controlled(s, tokens, 4, 8, NULL, NULL, NULL, &controlled_stats,
                                              &control) == IFN_CANCELLED &&
              forward_calls == before_controlled && !s->request_control);
        atomic_store(&request_cancel, false);
        CHECK(ifn_session_generate_controlled(s, tokens, 4, 8, NULL, request_cancel_token,
                                              &request_cancel, &controlled_stats,
                                              &control) == IFN_CANCELLED &&
              controlled_stats.generation.generated_tokens == 1 && !s->request_control &&
              s->position == 0 && !s->poisoned);
        CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8, callback_busy, s, &stats) == IFN_OK &&
              stats.generated_tokens == 8);
        CHECK(ifn_session_generate(s, tokens, 40, 8, out, 8, stopped, NULL, &stats) == IFN_OK &&
              stats.finish_reason == IFN_FINISH_CALLBACK && stats.generated_tokens == 1);
        CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8, cancelled_cb, s, &stats) ==
                  IFN_CANCELLED &&
              stats.finish_reason == IFN_FINISH_CANCELLED && s->position == 0 && !s->poisoned);
        int32_t one = 1;
        float unused = 0;
        CHECK(ifn_session_generate(s, &one, SIZE_MAX, 1, out, 8, NULL, NULL, &stats) ==
              IFN_INVALID_ARGUMENT);
        CHECK(ifn_session_logits(s, &one, SIZE_MAX, 1, &unused, 200058) == IFN_INVALID_ARGUMENT);
        int32_t invalid = -1;
        size_t prior = forward_calls;
        CHECK(ifn_session_generate(s, &invalid, 1, 8, out, 8, NULL, NULL, &stats) ==
                  IFN_INVALID_ARGUMENT &&
              forward_calls == prior);
        CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8, NULL, NULL, &stats) == IFN_OK &&
              stats.generated_tokens == 8);
        s->position = 128;
        float value = 0;
        CHECK(ifn_session_logits(s, tokens, 1, 0, &value, 200058) == IFN_LIMIT && !s->poisoned &&
              m.device == 42);
        CHECK(ifn_session_destroy(&s) == IFN_OK && !s && !w->session_users && m.live == 1);
    }
    /* Acquisition faults: preserve caller ownership and allow deterministic cleanup. */
    int ops[] = {ENTER, AVAILABLE, CREATE, ALLOC, SYNC, COPY, LEAVE};
    for (size_t i = 0; i < sizeof ops / sizeof *ops; i++) {
        m.fail_op = ops[i];
        m.fail_nth = m.calls[ops[i]] + 1;
        CHECK(ifn_session_create(w, &c, NULL, &s) != IFN_OK && s);
        m.fail_op = -1;
        CHECK(ifn_session_destroy(&s) == IFN_OK && !s && m.live == 1 && !w->session_users);
    }
    zero_fault = 700;
    CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_GPU_ERROR && s);
    zero_fault = 0;
    CHECK(ifn_session_destroy(&s) == IFN_OK);
    for (unsigned op = 0; op < 4; op++) {
        CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
        if (op < 2) {
            m.fail_op = op ? DESTROY : FREE;
            m.persistent = true;
        }
        if (op == 2)
            s->blas = &marker, blas_fault = 700;
        if (op == 3)
            s->graph = &marker, graph_fault = 700;
        CHECK(ifn_session_destroy(&s) == IFN_GPU_ERROR && s && w->session_users);
        m.fail_op = -1;
        m.persistent = false;
        blas_fault = graph_fault = 0;
        CHECK(ifn_session_destroy(&s) == IFN_OK && !s && m.live == 1);
    }
    /* A pre-cancelled request cannot hide a failed device acquisition. */
    CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
    atomic_bool pre_cancel;
    atomic_init(&pre_cancel, true);
    ifn_run_control pre_control = {request_cancelled, &pre_cancel};
    ifn_decode_stats pre_stats;
    int32_t pre_tokens[2] = {1, 2};
    m.fail_op = ENTER;
    m.fail_nth = m.calls[ENTER] + 1;
    CHECK(ifn_session_generate_controlled(s, pre_tokens, 2, 8, NULL, NULL, NULL, &pre_stats,
                                          &pre_control) == IFN_GPU_ERROR &&
          s->poisoned && !s->request_control);
    m.fail_op = -1;
    CHECK(ifn_session_destroy(&s) == IFN_OK);
    /* MTP-only acquisition and retained cleanup failures. */
    /* GPU execution errors poison; validation failures do not. */
    CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
    launch_fault = 700;
    int32_t tokens[4] = {1, 2, 3, 4}, out[8];
    ifn_generate_stats stats;
    CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8, NULL, NULL, &stats) == IFN_GPU_ERROR &&
          s->poisoned);
    launch_fault = 0;
    CHECK(ifn_session_reset(s) == IFN_GPU_ERROR);
    CHECK(ifn_session_destroy(&s) == IFN_OK);
    CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
    m.fail_op = LEAVE;
    m.fail_nth = m.calls[LEAVE] + 1;
    CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8, cancelled_cb, s, &stats) == IFN_GPU_ERROR &&
          s->poisoned && stats.finish_reason == IFN_FINISH_CANCELLED);
    m.fail_op = -1;
    CHECK(ifn_session_destroy(&s) == IFN_OK);
    m.device = 42;
    m.context = &m.context_marker;
    /* An independent thread cancels a generation blocked in the private backend. */
    CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
    atomic_store(&m.pause, 1);
    atomic_store(&m.arrived, 0);
    atomic_store(&m.resume, 0);
    generate_args args = {s, IFN_OK};
    pthread_t thread;
    CHECK(!pthread_create(&thread, NULL, generate_thread, &args));
    while (!atomic_load(&m.arrived))
        sched_yield();
    CHECK(ifn_session_reset(s) == IFN_BUSY && ifn_session_destroy(&s) == IFN_BUSY && s);
    CHECK(ifn_session_cancel(s) == IFN_OK);
    atomic_store(&m.resume, 1);
    CHECK(!pthread_join(thread, NULL) && args.status == IFN_CANCELLED);
    atomic_store(&m.pause, 0);
    CHECK(ifn_session_reset(s) == IFN_OK);
    CHECK(ifn_session_destroy(&s) == IFN_OK);
    CHECK(ifn_weights_destroy(&w) == IFN_OK && !w && !m.live);
    reset(&m);
    w = image_with_mtp(&m, 1);
    CHECK(w);
    CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK && s->model.dspark.loaded && s->ds_f_ring &&
          s->p_taps);
    ifn_decode_options options = IFN_DECODE_OPTIONS_INIT;
    options.temperature = 0.8f;
    options.seed = UINT64_MAX;
    options.draft_tokens = 3;
    ifn_decode_stats detail;
    mock_accepted_count = 2;
    CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, NULL, NULL, &detail) ==
              IFN_OK &&
          detail.generation.generated_tokens == 8 && detail.speculative_steps == 4 &&
          detail.accepted_drafts == 4);
    CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, cancelled_cb, s, &detail) ==
              IFN_CANCELLED &&
          !s->poisoned);
    options.adaptive = 1;
    options.draft_tokens = 7;
    ifn_decode_info info = IFN_DECODE_INFO_INIT;
    CHECK(ifn_session_decode_info(s, &info) == IFN_OK);
    info.struct_size = 0;
    CHECK(ifn_session_decode_info(s, &info) == IFN_INVALID_ARGUMENT);
    info = (ifn_decode_info)IFN_DECODE_INFO_INIT;
    CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, NULL, NULL, &detail) ==
              IFN_OK &&
          !s->draft_in_flight && !s->poisoned);
    CHECK(ifn_session_decode_info(s, &info) == IFN_OK && info.width2 > 0);
    CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, cancelled_cb, s, &detail) ==
              IFN_CANCELLED &&
          !s->draft_in_flight && !s->poisoned);
    c.use_graphs = 0;
    s->config.use_graphs = 0;
    CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, NULL, NULL, &detail) ==
              IFN_UNSUPPORTED &&
          !s->poisoned);
    c.use_graphs = s->config.use_graphs = 1;
    /* Real scheduler drain preserves the first error and retains uncertain work. */
    s->draft_in_flight = true;
    s->status = IFN_OK;
    m.fail_op = SYNC;
    m.fail_nth = m.calls[SYNC] + 1;
    ifn_adaptive_drain(s);
    CHECK(s->status == IFN_GPU_ERROR && s->draft_in_flight);
    s->status = IFN_OUT_OF_MEMORY;
    m.fail_nth = m.calls[SYNC] + 1;
    ifn_adaptive_drain(s);
    CHECK(s->status == IFN_OUT_OF_MEMORY && s->draft_in_flight);
    m.fail_op = -1;
    ifn_adaptive_drain(s);
    CHECK(s->status == IFN_OUT_OF_MEMORY && !s->draft_in_flight);
    s->status = IFN_OK;
    options.adaptive = 0;
    options.temperature = NAN;
    CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, NULL, NULL, &detail) ==
              IFN_INVALID_ARGUMENT &&
          !s->poisoned);
    options.temperature = 0;
    options.draft_tokens = 7;
    int32_t long_tokens[128] = {1, 2, 3, 4};
    CHECK(ifn_session_generate_ex(s, long_tokens, 121, 7, &options, out, 8, NULL, NULL, &detail) ==
              IFN_OK &&
          detail.speculative_steps == 0 && detail.generation.generated_tokens == 7);
    mock_accepted_count = 0;
    CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, NULL, NULL, &detail) ==
              IFN_GPU_ERROR &&
          s->poisoned);
    graph_fault = 700;
    CHECK(ifn_session_destroy(&s) == IFN_GPU_ERROR && s && s->arena);
    graph_fault = 0;
    CHECK(ifn_session_destroy(&s) == IFN_OK && !s);
    CHECK(ifn_weights_destroy(&w) == IFN_OK && !w && !m.live);
    reset(&m);
    w = image_with_mtp(&m, 1);
    CHECK(w);
    ifn_options fail_host = IFN_OPTIONS_INIT;
    fail_host.allocate = second_oom;
    fail_host.release = counted_free;
    CHECK(ifn_session_create(w, &c, &fail_host, &s) == IFN_OUT_OF_MEMORY && s &&
          !s->adaptive_state && !s->stream);
    CHECK(ifn_session_destroy(&s) == IFN_OK && !allocation_live && !w->session_users);
    for (int failure = 0; failure < 2; failure++) {
        pinned_fault = failure == 0;
        event_fault = failure == 1;
        CHECK(ifn_session_create(w, &c, NULL, &s) != IFN_OK && s);
        pinned_fault = event_fault = 0;
        CHECK(ifn_session_destroy(&s) == IFN_OK && !pinned_live && !events_live);
    }
    for (int failure = 0; failure < 3; failure++) {
        CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
        if (failure == 0) {
            *(int *)s->adaptive_state = 1;
            graph_fault = 700;
        } else if (failure == 1)
            pinned_free_fault = 700;
        else
            event_free_fault = 700;
        CHECK(ifn_session_destroy(&s) == IFN_GPU_ERROR && s && s->stream);
        if (failure == 0)
            CHECK(s->arena && s->pinned && s->done_event && s->adaptive_state);
        graph_fault = pinned_free_fault = event_free_fault = 0;
        CHECK(ifn_session_destroy(&s) == IFN_OK && !s && !pinned_live && !events_live);
    }
    CHECK(ifn_session_create(w, &c, NULL, &s) == IFN_OK);
    options.adaptive = 1;
    mock_accepted_count = 1;
    CHECK(ifn_session_generate_ex(s, tokens, 4, 8, &options, out, 8, fail_drain, &m, &detail) ==
              IFN_GPU_ERROR &&
          s->poisoned && s->draft_in_flight && detail.generation.generated_tokens == 2 &&
          strstr(ifn_last_error(), "drain"));
    m.fail_op = -1;
    CHECK(ifn_session_decode_info(s, &info) == IFN_OK && info.width2 == 1);
    CHECK(ifn_session_reset(s) == IFN_GPU_ERROR);
    CHECK(ifn_session_destroy(&s) == IFN_OK && !pinned_live && !events_live);
    CHECK(ifn_weights_destroy(&w) == IFN_OK && !m.live);
    mock_count_pointer = mock_accepted_pointer = NULL;
    mock_accepted_count = 1;
    reset(&m);
    w = image_with_mtp(&m, 1);
    CHECK(w);
    c.context_tokens = 2048;
    cache_allocator a = {0};
    ifn_options cache_options = IFN_OPTIONS_INIT;
    cache_options.context = &a;
    cache_options.allocate = cache_allocate;
    cache_options.release = cache_release;
    CHECK(ifn_session_create(w, &c, &cache_options, &s) == IFN_OK && a.live == 2);
    CHECK(ifn_session_prefix_cache(s, 2) == IFN_INVALID_ARGUMENT);
    for (unsigned i = 1; i <= 6; i++) {
        a.fail_at = a.calls + i;
        CHECK(ifn_session_prefix_cache(s, 1) == IFN_OUT_OF_MEMORY && a.live == 2 &&
              !s->prefix.enabled && !s->prefix.tokens);
        a.fail_at = 0;
        CHECK(ifn_session_prefix_cache(s, 1) == IFN_OK && a.live == 8);
        size_t calls = a.calls;
        CHECK(ifn_session_prefix_cache(s, 1) == IFN_OK && a.calls == calls);
        CHECK(ifn_session_prefix_cache(s, 0) == IFN_OK && a.live == 2);
    }
    CHECK(ifn_session_prefix_cache(s, 1) == IFN_OK);
    ifn_prefix_info pi = IFN_PREFIX_INFO_INIT;
    CHECK(ifn_session_prefix_info(s, &pi) == IFN_OK && pi.host_bytes > 0);
    pi.struct_size = 0;
    CHECK(ifn_session_prefix_info(s, &pi) == IFN_INVALID_ARGUMENT);
    pi = (ifn_prefix_info)IFN_PREFIX_INFO_INIT;
    size_t cache_calls = a.calls;
    memcpy(s->prefix.tokens, tokens, 4 * 4);
    s->prefix.length = 4;
    s->prefix.live = 2;
    CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8, callback_busy, s, &stats) == IFN_OK);
    CHECK(ifn_session_prefix_info(s, &pi) == IFN_OK && pi.reused_tokens == 2 &&
          !pi.checkpoint_restores && a.calls == cache_calls);
    s->prefix.length = 1024;
    s->prefix.live = -1;
    for (unsigned i = 0; i < 1024; i++)
        s->prefix.tokens[i] = 5;
    CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8, NULL, NULL, &stats) == IFN_OK);
    CHECK(ifn_session_prefix_info(s, &pi) == IFN_OK && pi.shielded_tokens == 28 &&
          s->prefix.length == 1024 && s->prefix.live == -1);
    CHECK(ifn_session_generate(s, tokens, 4, 8, out, 8, cancelled_cb, s, &stats) == IFN_CANCELLED &&
          s->prefix.length == 0);
    CHECK(ifn_session_prefix_cache(s, 0) == IFN_OK && ifn_session_prefix_cache(s, 1) == IFN_OK);
    int32_t prompt[129];
    for (unsigned i = 0; i < 129; i++)
        prompt[i] = 1;
    w->backend.sync = late_prefix_sync;
    CHECK(ifn_session_generate(s, prompt, 129, 8, out, 8, arm_final_snapshot_cancel, s, &stats) ==
              IFN_CANCELLED &&
          !late_cancel_session && !s->prefix.length && s->prefix.live == -1);
    for (unsigned i = 0; i < 4; i++)
        CHECK(!s->prefix.checkpoints[i].valid);
    w->backend.sync = sync_stream;

    for (unsigned i = 0; i < 256; i++)
        s->prefix.tokens[i] = 1;
    s->prefix.length = 256;
    s->prefix.live = -1;
    s->prefix.checkpoints[0].position = 128;
    s->prefix.checkpoints[0].valid = true;
    s->prefix.use_mtp = false;
    m.fail_op = COPY;
    m.fail_nth = m.calls[COPY] + 1;
    CHECK(ifn_session_generate(s, prompt, 129, 8, out, 8, NULL, NULL, &stats) == IFN_GPU_ERROR &&
          s->poisoned && !s->prefix.length);
    m.fail_op = -1;
    CHECK(ifn_session_destroy(&s) == IFN_OK && a.live == 0 && !pinned_live && !events_live);
    CHECK(ifn_weights_destroy(&w) == IFN_OK && !m.live);
    puts("AR session simulation: bounds, retention, faults, poisoning, cleanup retry, callback "
         "BUSY and cross-thread cancellation passed");
    return 0;
}

void ifn_spec_step(ifn_session *s, uint32_t drafts) {
    mock_count_pointer = s->d_num_acc;
    mock_accepted_pointer = s->d_accepted;
    if (s->config.use_graphs)
        s->spec_graph[drafts] = &marker;
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
    if (graph_fault)
        return graph_fault;
    memset(p, 0, 32);
    return 0;
}

static int32_t *adaptive_accepted, *adaptive_count, *adaptive_width;

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
    adaptive_accepted = host_accepted;
    adaptive_count = host_count;
    adaptive_width = host_width;
    return graph_fault;
}

int ifn_adaptive_branch_begin(void *state, int branch, void *stream) {
    (void)state;
    (void)branch;
    (void)stream;
    return graph_fault;
}

int ifn_adaptive_branch_end(void *state) {
    (void)state;
    return graph_fault;
}

int ifn_adaptive_instantiate(void *state) {
    (void)state;
    return graph_fault;
}

int ifn_adaptive_launch(void *state, void *stream) {
    (void)state;
    (void)stream;
    *adaptive_count = 1;
    *adaptive_width = 2;
    adaptive_accepted[0] = 42;
    return graph_fault;
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

int32_t squid_cuda_memcpy_d2h_async(void *dst, const void *src, size_t n, void *stream) {
    (void)dst;
    (void)src;
    (void)n;
    (void)stream;
    return graph_fault;
}

int32_t squid_cuda_event_record(void *event, void *stream) {
    (void)event;
    (void)stream;
    return graph_fault;
}

int32_t squid_cuda_event_sync(void *event) {
    (void)event;
    return graph_fault;
}

int32_t squid_cuda_malloc_host_pinned(void **out, size_t n, uint8_t mapped) {
    (void)mapped;
    if (pinned_fault) {
        *out = NULL;
        return 2;
    }
    *out = calloc(1, n);
    if (*out)
        pinned_live++;
    return *out ? 0 : 2;
}

int32_t squid_cuda_free_host(void *p) {
    if (pinned_free_fault)
        return pinned_free_fault;
    free(p);
    pinned_live--;
    return 0;
}

int32_t squid_cuda_event_create(void **out, uint8_t disable) {
    (void)disable;
    if (event_fault) {
        *out = NULL;
        return 2;
    }
    *out = &marker;
    events_live++;
    return 0;
}

int32_t squid_cuda_event_destroy(void *p) {
    (void)p;
    if (event_free_fault)
        return event_free_fault;
    events_live--;
    return 0;
}
