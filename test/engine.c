/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Control-flow/resource simulation only. Numerics require real GPU oracle. */
#include "mock_gpu.h"
#include "engine_internal.h"
#include "checked_prefill.h"
static int zero_fault, graph_fault, blas_fault, launch_fault;
static int marker;

static void *oom(void *c, size_t n) {
    (void)c;
    (void)n;
    return NULL;
}

static void ignored_free(void *c, void *p) {
    (void)c;
    (void)p;
}

static size_t forward_calls;

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

int sq_cuda_launch_status(void) {
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

int sq_blas_destroy(void *p) {
    (void)p;
    return blas_fault;
}

void sq_ar_forward(sq_session *s, int32_t n, bool lm) {
    (void)s;
    (void)n;
    (void)lm;
    forward_calls++;
}

void sq_ar_prefill(sq_session *s, const int32_t *tokens, int32_t n) {
    (void)tokens;
    s->blas = &marker;
    s->position += n;
}

static int session_copy(void *p, void *dst, const void *src, size_t n, int kind, void *stream) {
    int rc = copy(p, dst, src, n, kind, stream);
    if (!rc && kind == 1) {
        memset(dst, 0, n);
        if (n == 4) {
            int32_t token = 42;
            memcpy(dst, &token, 4);
        }
    }
    return rc;
}

static sq_gpu_weights *image(mock *m) {
    sq_gpu_weights *w = NULL;
    sq_weight_config c = SQ_WEIGHT_CONFIG_INIT;
    c.with_mtp = 0;
    sq_weight_backend b = backend(m);
    b.copy = session_copy;
    if (sq_weights_create(&c, NULL, &b, &w) != SQ_OK)
        return NULL;
    if (alloc_device(m, &w->arena, w->plan.arena_bytes)) {
        (void)sq_weights_destroy(&w);
        return NULL;
    }
    w->ready = true;
    return w;
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
    (void)sq_session_cancel(ctx);
    return 1;
}

static int callback_busy(void *ctx, int32_t token) {
    (void)token;
    sq_session *s = ctx, *copy_s = s;
    return sq_session_reset(s) == SQ_BUSY && sq_session_destroy(&copy_s) == SQ_BUSY && copy_s == s;
}

typedef struct {
    sq_session *session;
    sq_status status;
} generate_args;

static void *generate_thread(void *p) {
    generate_args *a = p;
    int32_t tokens[4] = {1, 2, 3, 4}, out[8];
    sq_generate_stats stats;
    a->status = sq_session_generate(a->session, tokens, 4, 8, out, 8, NULL, NULL, &stats);
    return NULL;
}

int main(void) {
    CHECK(!strcmp(sq_status_string(SQ_CANCELLED), "cancelled"));
    sq_session_config c = SQ_SESSION_CONFIG_INIT;
    c.context_tokens = 128;
    sq_session *s = NULL;
    mock m;
    reset(&m);
    sq_gpu_weights *w = image(&m);
    CHECK(w);
    c.kv_format = 3;
    CHECK(sq_session_create(w, &c, NULL, &s) == SQ_INVALID_ARGUMENT && !s);
    c.kv_format = 1;
    sq_options oom_options = SQ_OPTIONS_INIT;
    oom_options.allocate = oom;
    oom_options.release = ignored_free;
    CHECK(sq_session_create(w, &c, &oom_options, &s) == SQ_OUT_OF_MEMORY && !s);
    sq_session_config bad = c;
    bad.struct_size = 0;
    CHECK(sq_session_create(w, &bad, NULL, &s) == SQ_INVALID_ARGUMENT && !s);
    bad = c;
    bad.api_version = 0;
    CHECK(sq_session_create(w, &bad, NULL, &s) == SQ_INVALID_ARGUMENT && !s);
    bad = c;
    bad.reserved = 1;
    CHECK(sq_session_create(w, &bad, NULL, &s) == SQ_INVALID_ARGUMENT && !s);
    bad = c;
    bad.context_tokens = 1048577;
    CHECK(sq_session_create(w, &bad, NULL, &s) == SQ_INVALID_ARGUMENT && !s);
    w->backend.available = no_memory;
    CHECK(sq_session_create(w, &c, NULL, &s) == SQ_OUT_OF_MEMORY && s && !s->arena && !s->stream);
    CHECK(sq_session_destroy(&s) == SQ_OK && !s && m.live == 1);
    w->backend.available = available;
    uint32_t view_type = w->views[0].type;
    w->views[0].type = SQ_WEIGHT_EXPERT;
    CHECK(sq_session_create(w, &c, NULL, &s) == SQ_FORMAT && s);
    CHECK(sq_session_destroy(&s) == SQ_OK && !s && m.live == 1);
    w->views[0].type = view_type;
    for (unsigned format = 0; format < 3; format++) {
        c.kv_format = format;
        CHECK(sq_session_create(w, &c, NULL, &s) == SQ_OK && s);
        CHECK(w->session_users == 1 && s->offset == s->capacity && m.live == 2);
        sq_session *other = NULL;
        CHECK(sq_session_create(w, &c, NULL, &other) == SQ_BUSY && !other);
        CHECK(sq_weights_destroy(&w) == SQ_BUSY && w && w->ready);
        int32_t tokens[40] = {1, 2, 3, 4}, out[8];
        sq_generate_stats stats;
        CHECK(sq_session_generate(s, tokens, 4, 8, out, 8, callback_busy, s, &stats) == SQ_OK &&
              stats.generated_tokens == 8);
        CHECK(sq_session_generate(s, tokens, 40, 8, out, 8, stopped, NULL, &stats) == SQ_OK &&
              stats.finish_reason == SQ_FINISH_CALLBACK && stats.generated_tokens == 1);
        CHECK(sq_session_generate(s, tokens, 4, 8, out, 8, cancelled_cb, s, &stats) ==
                  SQ_CANCELLED &&
              stats.finish_reason == SQ_FINISH_CANCELLED && s->position == 0 && !s->poisoned);
        int32_t one = 1;
        float unused = 0;
        CHECK(sq_session_generate(s, &one, SIZE_MAX, 1, out, 8, NULL, NULL, &stats) ==
              SQ_INVALID_ARGUMENT);
        CHECK(sq_session_logits(s, &one, SIZE_MAX, 1, &unused, 200058) == SQ_INVALID_ARGUMENT);
        int32_t invalid = -1;
        size_t prior = forward_calls;
        CHECK(sq_session_generate(s, &invalid, 1, 8, out, 8, NULL, NULL, &stats) ==
                  SQ_INVALID_ARGUMENT &&
              forward_calls == prior);
        CHECK(sq_session_generate(s, tokens, 4, 8, out, 8, NULL, NULL, &stats) == SQ_OK &&
              stats.generated_tokens == 8);
        s->position = 128;
        float value = 0;
        CHECK(sq_session_logits(s, tokens, 1, 0, &value, 200058) == SQ_LIMIT && !s->poisoned &&
              m.device == 42);
        CHECK(sq_session_destroy(&s) == SQ_OK && !s && !w->session_users && m.live == 1);
    }
    /* Acquisition faults: preserve caller ownership and allow deterministic cleanup. */
    int ops[] = {ENTER, AVAILABLE, CREATE, ALLOC, SYNC, COPY, LEAVE};
    for (size_t i = 0; i < sizeof ops / sizeof *ops; i++) {
        m.fail_op = ops[i];
        m.fail_nth = m.calls[ops[i]] + 1;
        CHECK(sq_session_create(w, &c, NULL, &s) != SQ_OK && s);
        m.fail_op = -1;
        CHECK(sq_session_destroy(&s) == SQ_OK && !s && m.live == 1 && !w->session_users);
    }
    zero_fault = 700;
    CHECK(sq_session_create(w, &c, NULL, &s) == SQ_GPU_ERROR && s);
    zero_fault = 0;
    CHECK(sq_session_destroy(&s) == SQ_OK);
    for (unsigned op = 0; op < 4; op++) {
        CHECK(sq_session_create(w, &c, NULL, &s) == SQ_OK);
        if (op < 2) {
            m.fail_op = op ? DESTROY : FREE;
            m.persistent = true;
        }
        if (op == 2)
            s->blas = &marker, blas_fault = 700;
        if (op == 3)
            s->graph = &marker, graph_fault = 700;
        CHECK(sq_session_destroy(&s) == SQ_GPU_ERROR && s && w->session_users);
        m.fail_op = -1;
        m.persistent = false;
        blas_fault = graph_fault = 0;
        CHECK(sq_session_destroy(&s) == SQ_OK && !s && m.live == 1);
    }
    /* GPU execution errors poison; validation failures do not. */
    CHECK(sq_session_create(w, &c, NULL, &s) == SQ_OK);
    launch_fault = 700;
    int32_t tokens[4] = {1, 2, 3, 4}, out[8];
    sq_generate_stats stats;
    CHECK(sq_session_generate(s, tokens, 4, 8, out, 8, NULL, NULL, &stats) == SQ_GPU_ERROR &&
          s->poisoned);
    launch_fault = 0;
    CHECK(sq_session_reset(s) == SQ_GPU_ERROR);
    CHECK(sq_session_destroy(&s) == SQ_OK);
    CHECK(sq_session_create(w, &c, NULL, &s) == SQ_OK);
    m.fail_op = LEAVE;
    m.fail_nth = m.calls[LEAVE] + 1;
    CHECK(sq_session_generate(s, tokens, 4, 8, out, 8, cancelled_cb, s, &stats) == SQ_GPU_ERROR &&
          s->poisoned && stats.finish_reason == SQ_FINISH_CANCELLED);
    m.fail_op = -1;
    CHECK(sq_session_destroy(&s) == SQ_OK);
    m.device = 42;
    m.context = &m.context_marker;
    /* An independent thread cancels a generation blocked in the private backend. */
    CHECK(sq_session_create(w, &c, NULL, &s) == SQ_OK);
    atomic_store(&m.pause, 1);
    atomic_store(&m.arrived, 0);
    atomic_store(&m.resume, 0);
    generate_args args = {s, SQ_OK};
    pthread_t thread;
    CHECK(!pthread_create(&thread, NULL, generate_thread, &args));
    while (!atomic_load(&m.arrived))
        sched_yield();
    CHECK(sq_session_reset(s) == SQ_BUSY && sq_session_destroy(&s) == SQ_BUSY && s);
    CHECK(sq_session_cancel(s) == SQ_OK);
    atomic_store(&m.resume, 1);
    CHECK(!pthread_join(thread, NULL) && args.status == SQ_CANCELLED);
    atomic_store(&m.pause, 0);
    CHECK(sq_session_reset(s) == SQ_OK);
    CHECK(sq_session_destroy(&s) == SQ_OK);
    CHECK(sq_weights_destroy(&w) == SQ_OK && !w && !m.live);
    puts("AR session simulation: bounds, retention, faults, poisoning, cleanup retry, callback "
         "BUSY and cross-thread cancellation passed");
    return 0;
}
