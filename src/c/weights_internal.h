/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_WEIGHTS_INTERNAL_H
#define SQ_WEIGHTS_INTERNAL_H
#include "sq_gpu.h"
#include "internal.h"
#include "checked_cuda.h"
#include <stdatomic.h>
#define SQ_WEIGHT_MAX_SPANS 2048
#define SQ_WEIGHT_MAX_VIEWS 1024

typedef struct {
    void *context;
    int (*enter)(void *, int, sq_device_scope *);
    int (*leave)(void *, const sq_device_scope *);
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
} sq_weight_backend;

struct sq_gpu_weights {
    sq_options allocator;
    sq_weight_config config;
    sq_weight_plan plan;
    sq_weight_backend backend;
    atomic_flag busy;
    void *arena, *stage, *amax, *fold_temp, *stream;
    sq_weight_span spans[SQ_WEIGHT_MAX_SPANS];
    sq_weight_view views[SQ_WEIGHT_MAX_VIEWS];
    size_t span_count, view_count, offset, capacity;
    bool execute, ready;
    const sq_inkling_checkpoint *checkpoint;
    sq_status status;
    int backend_error;
    char failure[192];
};

sq_status sq_weights_create(const sq_weight_config *, const sq_options *, const sq_weight_backend *,
                            sq_gpu_weights **);
sq_status sq_weights_load(sq_gpu_weights *, const sq_inkling_checkpoint *);
sq_status sq_weights_destroy(sq_gpu_weights **);
sq_status sq_weights_read(sq_gpu_weights *, size_t, uint64_t, void *, size_t);
sq_status sq_weights_get_plan(sq_gpu_weights *, sq_weight_plan *);
sq_status sq_weights_get_span(sq_gpu_weights *, size_t, sq_weight_span *);
sq_status sq_weights_get_view(sq_gpu_weights *, size_t, sq_weight_view *);
void sq_weights_recipe(sq_gpu_weights *);
void sq_weight_fail(sq_gpu_weights *, sq_status, const char *);
void sq_weight_backend_check(sq_gpu_weights *, int, const char *);
sq_status sq_inkling_lookup(const sq_inkling_checkpoint *, uint32_t, const char *, const void **,
                            size_t *);
#endif
