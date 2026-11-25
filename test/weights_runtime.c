/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Private backend simulation checks control flow/ownership, NOT GPU numerics. */
#define _DEFAULT_SOURCE
#include "weights_internal.h"
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "weights:%d: %s (%s)\n", __LINE__, #x, sq_last_error());               \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

enum {
    ENTER,
    LEAVE,
    AVAILABLE,
    ALLOC,
    FREE,
    CREATE,
    DESTROY,
    SYNC,
    COPY,
    PACK,
    EXPERT,
    RNN,
    FOLD,
    OPS
};

typedef struct {
    size_t calls[OPS], fail_nth;
    int fail_op, device;
    bool persistent, free_also_fails;
    int context_marker, selected_marker;
    void *context;
    void *allocated[8];
    size_t sizes[8], live;
    bool stream;
    atomic_int pause, arrived, resume;
} mock;

static void reset(mock *m) {
    memset(m, 0, sizeof *m);
    m->fail_op = -1;
    m->device = 42;
    m->context = &m->context_marker;
    atomic_init(&m->pause, 0);
    atomic_init(&m->arrived, 0);
    atomic_init(&m->resume, 0);
}

static bool fail(mock *m, int op) {
    m->calls[op]++;
    return (m->fail_op == op && (m->persistent || m->calls[op] == m->fail_nth)) ||
           (op == FREE && m->free_also_fails);
}

static int enter(void *p, int device, sq_device_scope *old) {
    mock *m = p;
    old->device = m->device;
    old->context = m->context;
    if (fail(m, ENTER))
        return 700;
    m->device = device;
    m->context = &m->selected_marker;
    return 0;
}

static int leave(void *p, const sq_device_scope *old) {
    mock *m = p;
    if (fail(m, LEAVE))
        return 700;
    m->device = old->device;
    m->context = old->context;
    return 0;
}

static int available(void *p, size_t *bytes) {
    mock *m = p;
    if (fail(m, AVAILABLE))
        return 700;
    *bytes = SIZE_MAX;
    return 0;
}

static int alloc_device(void *p, void **out, size_t bytes) {
    mock *m = p;
    *out = NULL;
    if (fail(m, ALLOC))
        return 2;
    void *v = mmap(NULL, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (v == MAP_FAILED)
        return 2;
    for (size_t i = 0; i < 8; i++)
        if (!m->allocated[i]) {
            m->allocated[i] = v;
            m->sizes[i] = bytes;
            m->live++;
            *out = v;
            return 0;
        }
    (void)munmap(v, bytes);
    return 2;
}

static int release_device(void *p, void *data) {
    mock *m = p;
    if (fail(m, FREE))
        return 700;
    for (size_t i = 0; i < 8; i++)
        if (m->allocated[i] == data) {
            if (munmap(data, m->sizes[i]))
                return 700;
            m->allocated[i] = NULL;
            m->live--;
            return 0;
        }
    return 700;
}

static int create_stream(void *p, void **out) {
    mock *m = p;
    *out = NULL;
    if (fail(m, CREATE))
        return 700;
    m->stream = true;
    *out = &m->stream;
    return 0;
}

static int destroy_stream(void *p, void *s) {
    mock *m = p;
    (void)s;
    if (fail(m, DESTROY))
        return 700;
    m->stream = false;
    return 0;
}

static int sync_stream(void *p, void *s) {
    (void)s;
    return fail(p, SYNC) ? 700 : 0;
}

static int copy(void *p, void *a, const void *b, size_t bytes, int kind, void *s) {
    mock *m = p;
    (void)a;
    (void)b;
    (void)bytes;
    (void)kind;
    (void)s;
    if (atomic_load(&m->pause)) {
        atomic_store(&m->arrived, 1);
        while (!atomic_load(&m->resume))
            sched_yield();
    }
    return fail(m, COPY) ? 700 : 0;
}

static int pack(void *p, const void *a, void *b, void *c, void *d, int r, int k, int kind,
                float *scale, void *s) {
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    (void)r;
    (void)k;
    (void)kind;
    (void)s;
    *scale = 1.0f;
    return fail(p, PACK) ? 700 : 0;
}

static int expert(void *p, const void *a, const void *b, void *c, void *d, void *e, int n, int r,
                  int k, void *s) {
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    (void)e;
    (void)n;
    (void)r;
    (void)k;
    (void)s;
    return fail(p, EXPERT) ? 700 : 0;
}

static int rnn(void *p, const void *a, void *b, void *c, void *d, void *s) {
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    (void)s;
    return fail(p, RNN) ? 700 : 0;
}

static int fold(void *p, const void *a, const void *b, const void *c, void *d, void *e, void *s) {
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    (void)e;
    (void)s;
    return fail(p, FOLD) ? 700 : 0;
}

static sq_weight_backend backend(mock *m) {
    return (sq_weight_backend){m,
                               enter,
                               leave,
                               available,
                               alloc_device,
                               release_device,
                               create_stream,
                               destroy_stream,
                               sync_stream,
                               copy,
                               pack,
                               expert,
                               rnn,
                               fold};
}

typedef struct {
    sq_gpu_weights *weights;
    const sq_inkling_checkpoint *checkpoint;
    sq_status status;
} thread_args;

static void *load_thread(void *p) {
    thread_args *a = p;
    a->status = sq_weights_load(a->weights, a->checkpoint);
    return NULL;
}

static void *oom(void *c, size_t n) {
    (void)c;
    (void)n;
    return NULL;
}

static void ignored_free(void *c, void *p) {
    (void)c;
    (void)p;
}

int main(int argc, char **argv) {
    sq_weight_config config = SQ_WEIGHT_CONFIG_INIT;
    sq_weight_plan plan = {0};
    sq_options a = SQ_OPTIONS_INIT;
    a.allocate = oom;
    a.release = ignored_free;
    CHECK(sq_weights_plan(&config, &a, &plan) == SQ_OUT_OF_MEMORY && !plan.arena_bytes);
    config.reserved = 1;
    CHECK(sq_weights_plan(&config, NULL, &plan) == SQ_INVALID_ARGUMENT);
    config.reserved = 0;
    config.api_version++;
    CHECK(sq_weights_plan(&config, NULL, &plan) == SQ_INVALID_ARGUMENT);
    config.api_version = SQ_API_VERSION;
    CHECK(argc == 3 || argc == 4);
    sq_inkling_checkpoint *checkpoint = NULL;
    CHECK(sq_inkling_checkpoint_open(argv[1], argv[2], NULL, &checkpoint) == SQ_OK);
    mock m;
    reset(&m);
    sq_weight_backend b = backend(&m);
    sq_gpu_weights *w = NULL;
    CHECK(sq_weights_create(&config, NULL, &b, &w) == SQ_OK);
    if (argc == 4) {
        CHECK(sq_weights_load(w, checkpoint) == SQ_FORMAT && !w->ready && !m.live && !m.stream);
        CHECK(sq_weights_destroy(&w) == SQ_OK && !w);
        sq_inkling_checkpoint_close(checkpoint);
        return 0;
    }
    CHECK(sq_weights_load(w, checkpoint) == SQ_OK && w->ready && m.live == 1 && m.stream &&
          m.device == 42 && m.context == &m.context_marker);
    CHECK(w->offset == w->plan.arena_bytes && w->span_count == w->plan.allocations);
    size_t ends = 0;
    for (size_t i = 0; i < w->span_count; i++) {
        CHECK(w->spans[i].offset == ends && !(ends % 256));
        ends += (size_t)((w->spans[i].bytes + 255) & ~UINT64_C(255));
        for (size_t j = 0; j < i; j++)
            CHECK(strcmp(w->spans[j].label, w->spans[i].label));
    }
    CHECK(ends == w->plan.arena_bytes);
    for (size_t i = 0; i < w->view_count; i++)
        for (size_t j = 0; j < i; j++)
            CHECK(strcmp(w->views[j].label, w->views[i].label));
    CHECK(sq_weights_load(w, checkpoint) == SQ_BUSY);
    char byte;
    CHECK(sq_weights_read(w, 0, w->spans[0].bytes, &byte, 1) == SQ_INVALID_ARGUMENT);
    CHECK(sq_weights_read(w, 0, w->spans[0].bytes, NULL, 0) == SQ_OK && m.device == 42 &&
          m.context == &m.context_marker);
    size_t late_copy = m.calls[COPY], late_pack = m.calls[PACK];
    CHECK(sq_weights_destroy(&w) == SQ_OK && !w && !m.live && !m.stream && m.device == 42 &&
          m.context == &m.context_marker);
    const int ops[] = {ENTER, AVAILABLE, ALLOC, ALLOC, ALLOC, ALLOC, CREATE, COPY, COPY, PACK,
                       PACK,  EXPERT,    RNN,   FOLD,  SYNC,  FREE,  FREE,   FREE, LEAVE};
    const size_t nth[] = {1, 1, 1, 2, 3, 4, 1, 1, late_copy, 1, late_pack, 1, 1, 1, 1, 1, 2, 3, 1};
    for (size_t i = 0; i < sizeof ops / sizeof ops[0]; i++) {
        reset(&m);
        b = backend(&m);
        m.fail_op = ops[i];
        m.fail_nth = nth[i];
        CHECK(sq_weights_create(&config, NULL, &b, &w) == SQ_OK);
        sq_status status = sq_weights_load(w, checkpoint);
        CHECK(status == (ops[i] == ALLOC ? SQ_OUT_OF_MEMORY : SQ_GPU_ERROR) && !w->ready);
        m.fail_op = -1;
        CHECK(sq_weights_destroy(&w) == SQ_OK && !w && !m.live && !m.stream);
        if (ops[i] != LEAVE)
            CHECK(m.device == 42 && m.context == &m.context_marker);
    }
    /* Free failures retain ownership, and destruction is retryable. */
    for (int op = FREE; op <= DESTROY; op++) {
        if (op == CREATE)
            continue;
        reset(&m);
        b = backend(&m);
        CHECK(sq_weights_create(&config, NULL, &b, &w) == SQ_OK &&
              sq_weights_load(w, checkpoint) == SQ_OK);
        m.fail_op = op;
        m.persistent = true;
        CHECK(sq_weights_destroy(&w) == SQ_GPU_ERROR && w);
        m.fail_op = -1;
        CHECK(sq_weights_destroy(&w) == SQ_OK && !w && !m.live && !m.stream && m.device == 42 &&
              m.context == &m.context_marker);
    }
    reset(&m);
    b = backend(&m);
    m.fail_op = COPY;
    m.fail_nth = 20;
    m.free_also_fails = true;
    CHECK(sq_weights_create(&config, NULL, &b, &w) == SQ_OK);
    CHECK(sq_weights_load(w, checkpoint) == SQ_GPU_ERROR && strstr(sq_last_error(), "upload") &&
          m.live);
    m.free_also_fails = false;
    CHECK(sq_weights_destroy(&w) == SQ_OK && !w && !m.live && !m.stream);
    /* Real concurrent entry rejection, with a backend held mid-copy. */
    reset(&m);
    b = backend(&m);
    CHECK(sq_weights_create(&config, NULL, &b, &w) == SQ_OK);
    atomic_store(&m.pause, 1);
    thread_args args = {w, checkpoint, SQ_INVALID_ARGUMENT};
    pthread_t thread;
    CHECK(!pthread_create(&thread, NULL, load_thread, &args));
    while (!atomic_load(&m.arrived))
        sched_yield();
    sq_weight_span span;
    sq_weight_view v;
    CHECK(sq_weights_load(w, checkpoint) == SQ_BUSY && sq_weights_get_plan(w, &plan) == SQ_BUSY &&
          sq_weights_get_span(w, 0, &span) == SQ_BUSY && sq_weights_get_view(w, 0, &v) == SQ_BUSY &&
          sq_weights_read(w, 0, 0, &byte, 1) == SQ_BUSY && sq_weights_destroy(&w) == SQ_BUSY && w);
    atomic_store(&m.resume, 1);
    CHECK(!pthread_join(thread, NULL) && args.status == SQ_OK);
    CHECK(sq_weights_destroy(&w) == SQ_OK && !w && !m.live && !m.stream);
    sq_inkling_checkpoint_close(checkpoint);
    puts("GPU ownership simulation: allocation/operation/cleanup failures, retained retry handles, "
         "device/context restoration and concurrent BUSY passed");
    return 0;
}
