#ifndef IFN_TEST_MOCK_GPU_H
#define IFN_TEST_MOCK_GPU_H
/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Private backend simulation checks control flow/ownership, NOT GPU numerics. */
#define _DEFAULT_SOURCE
#include "weights_internal.h"
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define IFN_MOCK_GPU_LOCAL_CHECK(x)                                                                \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "weights:%d: %s (%s)\n", __LINE__, #x, ifn_last_error());              \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

enum {
    IFN_MOCK_GPU_LOCAL_ENTER,
    LEAVE,
    IFN_MOCK_GPU_LOCAL_AVAILABLE,
    ALLOC,
    IFN_MOCK_GPU_LOCAL_FREE,
    CREATE,
    IFN_MOCK_GPU_LOCAL_DESTROY,
    SYNC,
    IFN_MOCK_GPU_LOCAL_COPY,
    PACK,
    IFN_MOCK_GPU_LOCAL_EXPERT,
    RNN,
    IFN_MOCK_GPU_LOCAL_FOLD,
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
} ifn_mock_gpu_local_mock;

static void ifn_mock_gpu_local_reset(ifn_mock_gpu_local_mock *m) {
    memset(m, 0, sizeof *m);
    m->fail_op = -1;
    m->device = 42;
    m->context = &m->context_marker;
    atomic_init(&m->pause, 0);
    atomic_init(&m->arrived, 0);
    atomic_init(&m->resume, 0);
}

static bool ifn_mock_gpu_local_fail(ifn_mock_gpu_local_mock *m, int op) {
    m->calls[op]++;
    return (m->fail_op == op && (m->persistent || m->calls[op] == m->fail_nth)) ||
           (op == IFN_MOCK_GPU_LOCAL_FREE && m->free_also_fails);
}

static int ifn_mock_gpu_local_enter(void *p, int device, ifn_device_scope *old) {
    ifn_mock_gpu_local_mock *m = p;
    old->device = m->device;
    old->context = m->context;
    if (ifn_mock_gpu_local_fail(m, IFN_MOCK_GPU_LOCAL_ENTER))
        return 700;
    m->device = device;
    m->context = &m->selected_marker;
    return 0;
}

static int ifn_mock_gpu_local_leave(void *p, const ifn_device_scope *old) {
    ifn_mock_gpu_local_mock *m = p;
    if (ifn_mock_gpu_local_fail(m, LEAVE))
        return 700;
    m->device = old->device;
    m->context = old->context;
    return 0;
}

static int ifn_mock_gpu_local_available(void *p, size_t *bytes) {
    ifn_mock_gpu_local_mock *m = p;
    if (ifn_mock_gpu_local_fail(m, IFN_MOCK_GPU_LOCAL_AVAILABLE))
        return 700;
    *bytes = SIZE_MAX;
    return 0;
}

static int ifn_mock_gpu_local_alloc_device(void *p, void **out, size_t bytes) {
    ifn_mock_gpu_local_mock *m = p;
    *out = NULL;
    if (ifn_mock_gpu_local_fail(m, ALLOC))
        return 2;
    void *v = ifn_test_reserve(bytes);
    if (!v)
        return 2;
    for (size_t i = 0; i < 8; i++)
        if (!m->allocated[i]) {
            m->allocated[i] = v;
            m->sizes[i] = bytes;
            m->live++;
            *out = v;
            return 0;
        }
    (void)ifn_test_unreserve(v, bytes);
    return 2;
}

static int ifn_mock_gpu_local_release_device(void *p, void *data) {
    ifn_mock_gpu_local_mock *m = p;
    if (ifn_mock_gpu_local_fail(m, IFN_MOCK_GPU_LOCAL_FREE))
        return 700;
    for (size_t i = 0; i < 8; i++)
        if (m->allocated[i] == data) {
            if (ifn_test_unreserve(data, m->sizes[i]))
                return 700;
            m->allocated[i] = NULL;
            m->live--;
            return 0;
        }
    return 700;
}

static int ifn_mock_gpu_local_create_stream(void *p, void **out) {
    ifn_mock_gpu_local_mock *m = p;
    *out = NULL;
    if (ifn_mock_gpu_local_fail(m, CREATE))
        return 700;
    m->stream = true;
    *out = &m->stream;
    return 0;
}

static int ifn_mock_gpu_local_destroy_stream(void *p, void *s) {
    ifn_mock_gpu_local_mock *m = p;
    (void)s;
    if (ifn_mock_gpu_local_fail(m, IFN_MOCK_GPU_LOCAL_DESTROY))
        return 700;
    m->stream = false;
    return 0;
}

static int ifn_mock_gpu_local_sync_stream(void *p, void *s) {
    (void)s;
    return ifn_mock_gpu_local_fail(p, SYNC) ? 700 : 0;
}

static int ifn_mock_gpu_local_copy(void *p, void *a, const void *b, size_t bytes, int kind,
                                   void *s) {
    ifn_mock_gpu_local_mock *m = p;
    (void)a;
    (void)b;
    (void)bytes;
    (void)kind;
    (void)s;
    if (atomic_load(&m->pause)) {
        atomic_store(&m->arrived, 1);
        while (!atomic_load(&m->resume))
            ifn_test_yield();
    }
    return ifn_mock_gpu_local_fail(m, IFN_MOCK_GPU_LOCAL_COPY) ? 700 : 0;
}

static int ifn_mock_gpu_local_pack(void *p, const void *a, void *b, void *c, void *d, int r, int k,
                                   int kind, float *scale, void *s) {
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    (void)r;
    (void)k;
    (void)kind;
    (void)s;
    *scale = 1.0f;
    return ifn_mock_gpu_local_fail(p, PACK) ? 700 : 0;
}

static int ifn_mock_gpu_local_expert(void *p, const void *a, const void *b, void *c, void *d,
                                     void *e, int n, int r, int k, void *s) {
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    (void)e;
    (void)n;
    (void)r;
    (void)k;
    (void)s;
    return ifn_mock_gpu_local_fail(p, IFN_MOCK_GPU_LOCAL_EXPERT) ? 700 : 0;
}

static int ifn_mock_gpu_local_rnn(void *p, const void *a, void *b, void *c, void *d, void *s) {
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    (void)s;
    return ifn_mock_gpu_local_fail(p, RNN) ? 700 : 0;
}

static int ifn_mock_gpu_local_fold(void *p, const void *a, const void *b, const void *c, void *d,
                                   void *e, void *s) {
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    (void)e;
    (void)s;
    return ifn_mock_gpu_local_fail(p, IFN_MOCK_GPU_LOCAL_FOLD) ? 700 : 0;
}

static ifn_weight_backend ifn_mock_gpu_local_backend(ifn_mock_gpu_local_mock *m) {
    return (ifn_weight_backend){m,
                                ifn_mock_gpu_local_enter,
                                ifn_mock_gpu_local_leave,
                                ifn_mock_gpu_local_available,
                                ifn_mock_gpu_local_alloc_device,
                                ifn_mock_gpu_local_release_device,
                                ifn_mock_gpu_local_create_stream,
                                ifn_mock_gpu_local_destroy_stream,
                                ifn_mock_gpu_local_sync_stream,
                                ifn_mock_gpu_local_copy,
                                ifn_mock_gpu_local_pack,
                                ifn_mock_gpu_local_expert,
                                ifn_mock_gpu_local_rnn,
                                ifn_mock_gpu_local_fold};
}

#endif
