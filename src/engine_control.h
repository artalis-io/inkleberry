/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_ENGINE_CONTROL_H
#define IFN_ENGINE_CONTROL_H
#include "ifn_engine.h"
#include <stdbool.h>

/* Private borrowed request control. Installed only under the engine execution
 * lease, polled at native safe points, removed before that lease is released. */
typedef struct {
    bool (*cancelled)(void *);
    void *context;
} ifn_run_control;

ifn_status ifn_session_generate_controlled(ifn_session *, const int32_t *, size_t, size_t,
                                           const ifn_decode_options *, ifn_token_callback, void *,
                                           ifn_decode_stats *, const ifn_run_control *);
ifn_status ifn_session_owned_plan(ifn_gpu_weights *, const ifn_session_config *,
                                  const ifn_options *, uint64_t *);
#endif
