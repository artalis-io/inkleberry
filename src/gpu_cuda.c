/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "weights_internal.h"
#include "cuda_abi.h"
#include "checked_cuda.h"

static int enter(void *c, int d, ifn_device_scope *old) {
    (void)c;
    return ifn_cuda_enter(d, old);
}

static int leave(void *c, const ifn_device_scope *d) {
    (void)c;
    return ifn_cuda_leave(d);
}

static int available(void *c, size_t *n) {
    (void)c;
    return ifn_cuda_available(n);
}

static int allocate(void *c, void **p, size_t n) {
    (void)c;
    return squid_cuda_malloc(p, n);
}

static int release(void *c, void *p) {
    (void)c;
    return squid_cuda_free(p);
}

static int stream_create(void *c, void **p) {
    (void)c;
    return squid_cuda_stream_create(p);
}

static int stream_destroy(void *c, void *p) {
    (void)c;
    return squid_cuda_stream_destroy(p);
}

static int sync_stream(void *c, void *s) {
    (void)c;
    return squid_cuda_stream_sync(s);
}

static int copy(void *c, void *out, const void *in, size_t n, int kind, void *s) {
    (void)c;
    /* H2D can borrow stack-derived scales; completion before return is required. */
    int rc = kind == 0   ? squid_cuda_memcpy_h2d_async(out, in, n, s)
             : kind == 1 ? squid_cuda_memcpy_d2h_async(out, in, n, s)
                         : squid_cuda_memcpy_d2d_async(out, in, n, s);
    if (!rc && kind == 0)
        rc = squid_cuda_stream_sync(s);
    return rc;
}

static int pack(void *c, const void *a, void *b, void *d, void *e, int r, int k, int kind,
                float *scale, void *s) {
    (void)c;
    return ifn_cuda_pack(a, b, d, e, r, k, kind, scale, s);
}

static int expert(void *c, const void *a, const void *b, void *d, void *e, void *f, int n, int r,
                  int k, void *s) {
    (void)c;
    return ifn_cuda_expert(a, b, d, e, f, n, r, k, s);
}

static int rnn(void *c, const void *a, void *b, void *d, void *e, void *s) {
    (void)c;
    return ifn_cuda_rnn(a, b, d, e, s);
}

static int fold(void *c, const void *a, const void *b, const void *d, void *e, void *f, void *s) {
    (void)c;
    return ifn_cuda_fold(a, b, d, e, f, 200058, s);
}

static const ifn_weight_backend backend = {
    NULL,           enter,       leave, available, allocate, release, stream_create,
    stream_destroy, sync_stream, copy,  pack,      expert,   rnn,     fold};

ifn_status ifn_gpu_weights_create(const ifn_weight_config *c, const ifn_options *a,
                                  ifn_gpu_weights **out) {
    return ifn_weights_create(c, a, &backend, out);
}

ifn_status ifn_gpu_weights_load(ifn_gpu_weights *w, const ifn_inkling_checkpoint *c) {
    return ifn_weights_load(w, c);
}

ifn_status ifn_gpu_weights_destroy(ifn_gpu_weights **w) {
    return ifn_weights_destroy(w);
}

ifn_status ifn_gpu_weights_get_plan(ifn_gpu_weights *w, ifn_weight_plan *p) {
    return ifn_weights_get_plan(w, p);
}

ifn_status ifn_gpu_weights_get_span(ifn_gpu_weights *w, size_t i, ifn_weight_span *p) {
    return ifn_weights_get_span(w, i, p);
}

ifn_status ifn_gpu_weights_get_view(ifn_gpu_weights *w, size_t i, ifn_weight_view *p) {
    return ifn_weights_get_view(w, i, p);
}

ifn_status ifn_gpu_weights_read(ifn_gpu_weights *w, size_t i, uint64_t o, void *p, size_t n) {
    return ifn_weights_read(w, i, o, p, n);
}
