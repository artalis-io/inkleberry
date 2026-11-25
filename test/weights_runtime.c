/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "mock_gpu.h"

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
