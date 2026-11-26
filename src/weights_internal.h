/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_WEIGHTS_INTERNAL_H
#define IFN_WEIGHTS_INTERNAL_H
#include "ifn_gpu.h"
#include "internal.h"
#include "checked_cuda.h"
#include <stdatomic.h>
#define IFN_WEIGHT_MAX_SPANS 2048
#define IFN_WEIGHT_MAX_VIEWS 1024

typedef struct {
    void *context;
    int (*enter)(void *, int, ifn_device_scope *);
    int (*leave)(void *, const ifn_device_scope *);
    int (*available)(void *, size_t *);
    int (*allocate)(void *, void **, size_t);
    int (*release)(void *, void *);
    int (*stream_create)(void *, void **);
    int (*stream_destroy)(void *, void *);
    int (*sync)(void *, void *);
    int (*copy)(void *, void *, const void *, size_t, int, void *); /* 0 H2D, 1 D2H, 2 D2D */
    int (*pack)(void *, const void *, void *, void *, void *, int, int, int, float *, void *);
    int (*expert)(void *, const void *, const void *, void *, void *, void *, int, int, int,
                  void *);
    int (*rnn)(void *, const void *, void *, void *, void *, void *);
    int (*fold)(void *, const void *, const void *, const void *, void *, void *, void *);
} ifn_weight_backend;

struct ifn_gpu_weights {
    ifn_options allocator;
    ifn_weight_config config;
    ifn_weight_plan plan;
    ifn_weight_backend backend;
    atomic_flag busy;
    void *arena, *stage, *amax, *fold_temp, *stream;
    ifn_weight_span spans[IFN_WEIGHT_MAX_SPANS];
    ifn_weight_view views[IFN_WEIGHT_MAX_VIEWS];
    size_t span_count, view_count, offset, capacity;
    size_t session_users;
    bool execute, ready;
    const ifn_inkling_checkpoint *checkpoint;
    ifn_status status;
    int backend_error;
    char failure[192];
};

ifn_status ifn_weights_create(const ifn_weight_config *, const ifn_options *,
                              const ifn_weight_backend *, ifn_gpu_weights **);
ifn_status ifn_weights_load(ifn_gpu_weights *, const ifn_inkling_checkpoint *);
ifn_status ifn_weights_destroy(ifn_gpu_weights **);
ifn_status ifn_weights_read(ifn_gpu_weights *, size_t, uint64_t, void *, size_t);
ifn_status ifn_weights_get_plan(ifn_gpu_weights *, ifn_weight_plan *);
ifn_status ifn_weights_get_span(ifn_gpu_weights *, size_t, ifn_weight_span *);
ifn_status ifn_weights_get_view(ifn_gpu_weights *, size_t, ifn_weight_view *);
void ifn_weights_recipe(ifn_gpu_weights *);
void ifn_weight_fail(ifn_gpu_weights *, ifn_status, const char *);
void ifn_weight_backend_check(ifn_gpu_weights *, int, const char *);
ifn_status ifn_inkling_lookup(const ifn_inkling_checkpoint *, uint32_t, const char *, const void **,
                              size_t *);
#endif
