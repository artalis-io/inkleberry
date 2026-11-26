/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_ENGINE_CONTROL_H
#define SQ_ENGINE_CONTROL_H
#include "sq_engine.h"
#include <stdbool.h>

/* Private borrowed request control. Installed only under the engine execution
 * lease, polled at native safe points, removed before that lease is released. */
typedef struct {
    bool (*cancelled)(void *);
    void *context;
} sq_run_control;

sq_status sq_session_generate_controlled(sq_session *, const int32_t *, size_t, size_t,
                                         const sq_decode_options *, sq_token_callback, void *,
                                         sq_decode_stats *, const sq_run_control *);
sq_status sq_session_owned_plan(sq_gpu_weights *, const sq_session_config *, const sq_options *,
                                uint64_t *);
#endif
