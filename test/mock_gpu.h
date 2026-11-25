#ifndef SQ_TEST_MOCK_GPU_H
#define SQ_TEST_MOCK_GPU_H
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

#endif
