/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "mock_gpu.h"
#include "inkling_quant.h"

typedef struct {
    ifn_gpu_weights *weights;
    const ifn_inkling_checkpoint *checkpoint;
    ifn_status status;
} ifn_weights_runtime_local_thread_args;

static void *ifn_weights_runtime_local_load_thread(void *p) {
    ifn_weights_runtime_local_thread_args *a = p;
    a->status = ifn_weights_load(a->weights, a->checkpoint);
    return NULL;
}

static void *ifn_weights_runtime_local_oom(void *c, size_t n) {
    (void)c;
    (void)n;
    return NULL;
}

static void ifn_weights_runtime_local_ignored_free(void *c, void *p) {
    (void)c;
    (void)p;
}

int main(int argc, char **argv) {
    for (unsigned mode = 0; mode < 3; mode++) {
        for (unsigned layer = 0; layer < 24; layer++)
            IFN_MOCK_GPU_LOCAL_CHECK(ifn_inkling_expert_precision(mode, layer) ==
                                     (mode == 2 || layer < 20 ? 2u : 1u));
        for (unsigned force = 0; force < 2; force++)
            IFN_MOCK_GPU_LOCAL_CHECK(ifn_inkling_mtp_precision(mode, force) ==
                                     (mode == 0 || force ? 0u : 2u));
    }

    ifn_weight_config config = IFN_WEIGHT_CONFIG_INIT;
    ifn_weight_plan plan = {0};
    ifn_options a = IFN_OPTIONS_INIT;
    a.allocate = ifn_weights_runtime_local_oom;
    a.release = ifn_weights_runtime_local_ignored_free;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_plan(&config, &a, &plan) == IFN_OUT_OF_MEMORY &&
                             !plan.arena_bytes);
    config.reserved = 1;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_plan(&config, NULL, &plan) == IFN_INVALID_ARGUMENT);
    config.reserved = 0;
    config.api_version++;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_plan(&config, NULL, &plan) == IFN_INVALID_ARGUMENT);
    config.api_version = IFN_API_VERSION;
    IFN_MOCK_GPU_LOCAL_CHECK(argc == 3 || argc == 4);
    ifn_inkling_checkpoint *checkpoint = NULL;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_inkling_checkpoint_open(argv[1], argv[2], NULL, &checkpoint) ==
                             IFN_OK);
    ifn_mock_gpu_local_mock m;
    ifn_mock_gpu_local_reset(&m);
    ifn_weight_backend b = ifn_mock_gpu_local_backend(&m);
    ifn_gpu_weights *w = NULL;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_create(&config, NULL, &b, &w) == IFN_OK);
    if (argc == 4) {
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_load(w, checkpoint) == IFN_FORMAT && !w->ready &&
                                 !m.live && !m.stream);
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_destroy(&w) == IFN_OK && !w);
        ifn_inkling_checkpoint_close(checkpoint);
        return 0;
    }
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_load(w, checkpoint) == IFN_OK && w->ready && m.live == 1 &&
                             m.stream && m.device == 42 && m.context == &m.context_marker);
    IFN_MOCK_GPU_LOCAL_CHECK(w->offset == w->plan.arena_bytes &&
                             w->span_count == w->plan.allocations);
    size_t ends = 0;
    for (size_t i = 0; i < w->span_count; i++) {
        IFN_MOCK_GPU_LOCAL_CHECK(w->spans[i].offset == ends && !(ends % 256));
        ends += (size_t)((w->spans[i].bytes + 255) & ~UINT64_C(255));
        for (size_t j = 0; j < i; j++)
            IFN_MOCK_GPU_LOCAL_CHECK(strcmp(w->spans[j].label, w->spans[i].label));
    }
    IFN_MOCK_GPU_LOCAL_CHECK(ends == w->plan.arena_bytes);
    for (size_t i = 0; i < w->view_count; i++)
        for (size_t j = 0; j < i; j++)
            IFN_MOCK_GPU_LOCAL_CHECK(strcmp(w->views[j].label, w->views[i].label));
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_load(w, checkpoint) == IFN_BUSY);
    char byte;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_read(w, 0, w->spans[0].bytes, &byte, 1) ==
                             IFN_INVALID_ARGUMENT);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_read(w, 0, w->spans[0].bytes, NULL, 0) == IFN_OK &&
                             m.device == 42 && m.context == &m.context_marker);
    size_t late_copy = m.calls[IFN_MOCK_GPU_LOCAL_COPY], late_pack = m.calls[PACK];
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_destroy(&w) == IFN_OK && !w && !m.live && !m.stream &&
                             m.device == 42 && m.context == &m.context_marker);
    const int ops[] = {IFN_MOCK_GPU_LOCAL_ENTER,
                       IFN_MOCK_GPU_LOCAL_AVAILABLE,
                       ALLOC,
                       ALLOC,
                       ALLOC,
                       ALLOC,
                       CREATE,
                       IFN_MOCK_GPU_LOCAL_COPY,
                       IFN_MOCK_GPU_LOCAL_COPY,
                       PACK,
                       PACK,
                       IFN_MOCK_GPU_LOCAL_EXPERT,
                       RNN,
                       IFN_MOCK_GPU_LOCAL_FOLD,
                       SYNC,
                       IFN_MOCK_GPU_LOCAL_FREE,
                       IFN_MOCK_GPU_LOCAL_FREE,
                       IFN_MOCK_GPU_LOCAL_FREE,
                       LEAVE};
    const size_t nth[] = {1, 1, 1, 2, 3, 4, 1, 1, late_copy, 1, late_pack, 1, 1, 1, 1, 1, 2, 3, 1};
    for (size_t i = 0; i < sizeof ops / sizeof ops[0]; i++) {
        ifn_mock_gpu_local_reset(&m);
        b = ifn_mock_gpu_local_backend(&m);
        m.fail_op = ops[i];
        m.fail_nth = nth[i];
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_create(&config, NULL, &b, &w) == IFN_OK);
        ifn_status status = ifn_weights_load(w, checkpoint);
        IFN_MOCK_GPU_LOCAL_CHECK(
            status == (ifn_status)(ops[i] == ALLOC ? IFN_OUT_OF_MEMORY : IFN_GPU_ERROR) &&
            !w->ready);
        m.fail_op = -1;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_destroy(&w) == IFN_OK && !w && !m.live && !m.stream);
        if (ops[i] != LEAVE)
            IFN_MOCK_GPU_LOCAL_CHECK(m.device == 42 && m.context == &m.context_marker);
    }
    /* Free failures retain ownership, and destruction is retryable. */
    for (int op = IFN_MOCK_GPU_LOCAL_FREE; op <= IFN_MOCK_GPU_LOCAL_DESTROY; op++) {
        if (op == CREATE)
            continue;
        ifn_mock_gpu_local_reset(&m);
        b = ifn_mock_gpu_local_backend(&m);
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_create(&config, NULL, &b, &w) == IFN_OK &&
                                 ifn_weights_load(w, checkpoint) == IFN_OK);
        m.fail_op = op;
        m.persistent = true;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_destroy(&w) == IFN_GPU_ERROR && w);
        m.fail_op = -1;
        IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_destroy(&w) == IFN_OK && !w && !m.live && !m.stream &&
                                 m.device == 42 && m.context == &m.context_marker);
    }
    ifn_mock_gpu_local_reset(&m);
    b = ifn_mock_gpu_local_backend(&m);
    m.fail_op = IFN_MOCK_GPU_LOCAL_COPY;
    m.fail_nth = 20;
    m.free_also_fails = true;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_create(&config, NULL, &b, &w) == IFN_OK);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_load(w, checkpoint) == IFN_GPU_ERROR &&
                             strstr(ifn_last_error(), "upload") && m.live);
    m.free_also_fails = false;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_destroy(&w) == IFN_OK && !w && !m.live && !m.stream);
    /* Real concurrent entry rejection, with a backend held mid-copy. */
    ifn_mock_gpu_local_reset(&m);
    b = ifn_mock_gpu_local_backend(&m);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_create(&config, NULL, &b, &w) == IFN_OK);
    atomic_store(&m.pause, 1);
    ifn_weights_runtime_local_thread_args args = {w, checkpoint, IFN_INVALID_ARGUMENT};
    ifn_test_thread thread;
    IFN_MOCK_GPU_LOCAL_CHECK(
        !ifn_test_thread_create(&thread, NULL, ifn_weights_runtime_local_load_thread, &args));
    while (!atomic_load(&m.arrived))
        ifn_test_yield();
    ifn_weight_span span;
    ifn_weight_view v;
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_load(w, checkpoint) == IFN_BUSY &&
                             ifn_weights_get_plan(w, &plan) == IFN_BUSY &&
                             ifn_weights_get_span(w, 0, &span) == IFN_BUSY &&
                             ifn_weights_get_view(w, 0, &v) == IFN_BUSY &&
                             ifn_weights_read(w, 0, 0, &byte, 1) == IFN_BUSY &&
                             ifn_weights_destroy(&w) == IFN_BUSY && w);
    atomic_store(&m.resume, 1);
    IFN_MOCK_GPU_LOCAL_CHECK(!ifn_test_thread_join(thread, NULL) && args.status == IFN_OK);
    IFN_MOCK_GPU_LOCAL_CHECK(ifn_weights_destroy(&w) == IFN_OK && !w && !m.live && !m.stream);
    ifn_inkling_checkpoint_close(checkpoint);
    puts("GPU ownership simulation: allocation/operation/cleanup failures, retained retry handles, "
         "device/context restoration and concurrent BUSY passed");
    return 0;
}
