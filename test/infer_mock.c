/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Fake execution checks facade ownership/concurrency, never GPU numerics. */
#include "infer_internal.h"
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

struct sq_gpu_weights {
    sq_options allocator;
};

struct sq_session {
    sq_options allocator;
};

atomic_int si_mock_pause, si_mock_arrived, si_mock_resume;
atomic_uint si_mock_resets, si_mock_calls, si_mock_enables;
atomic_uint si_mock_fail, si_mock_close_fail, si_mock_upload_fail;
const char *si_mock_tokenizer;

sq_status sq_gpu_weights_create(const sq_weight_config *c, const sq_options *a,
                                sq_gpu_weights **out) {
    (void)c;
    *out = si_alloc(a, sizeof **out);
    if (!*out)
        return SQ_OUT_OF_MEMORY;
    (*out)->allocator = *a;
    return SQ_OK;
}

sq_status sq_gpu_weights_load(sq_gpu_weights *w, const sq_inkling_checkpoint *cp) {
    (void)w;
    (void)cp;
    return atomic_load(&si_mock_upload_fail) ? SQ_GPU_ERROR : SQ_OK;
}

sq_status sq_gpu_weights_destroy(sq_gpu_weights **w) {
    if (*w) {
        if (atomic_load(&si_mock_close_fail))
            return SQ_GPU_ERROR;
        si_free(&(*w)->allocator, *w);
    }
    *w = NULL;
    return SQ_OK;
}

sq_status sq_session_create(sq_gpu_weights *w, const sq_session_config *c, const sq_options *a,
                            sq_session **out) {
    (void)w;
    (void)c;
    *out = si_alloc(a, sizeof **out);
    if (!*out)
        return SQ_OUT_OF_MEMORY;
    (*out)->allocator = *a;
    return SQ_OK;
}

sq_status sq_session_destroy(sq_session **s) {
    if (*s) {
        if (atomic_load(&si_mock_close_fail))
            return SQ_GPU_ERROR;
        si_free(&(*s)->allocator, *s);
    }
    *s = NULL;
    return SQ_OK;
}

sq_status sq_session_owned_plan(sq_gpu_weights *w, const sq_session_config *c, const sq_options *a,
                                uint64_t *out) {
    (void)w;
    (void)c;
    (void)a;
    *out = 1024;
    return SQ_OK;
}

sq_status sq_session_prefix_cache(sq_session *s, uint32_t enabled) {
    (void)s;
    if (enabled)
        atomic_fetch_add(&si_mock_enables, 1);
    return SQ_OK;
}

sq_status sq_session_prefix_info(sq_session *s, sq_prefix_info *out) {
    (void)s;
    out->reused_tokens = 0;
    return SQ_OK;
}

sq_status sq_session_reset(sq_session *s) {
    (void)s;
    atomic_fetch_add(&si_mock_resets, 1);
    return atomic_load(&si_mock_fail) == 2 ? SQ_GPU_ERROR : SQ_OK;
}

sq_status sq_session_generate_controlled(sq_session *s, const int32_t *tokens, size_t n,
                                         size_t limit, const sq_decode_options *options,
                                         sq_token_callback cb, void *ctx, sq_decode_stats *stats,
                                         const sq_run_control *control) {
    (void)s;
    (void)tokens;
    (void)options;
    atomic_fetch_add(&si_mock_calls, 1);
    if (atomic_load(&si_mock_pause) == 1) {
        atomic_store(&si_mock_arrived, 1);
        while (!atomic_load(&si_mock_resume))
            sched_yield();
    }
    stats->generation.prompt_tokens = (uint32_t)n;
    if (control->cancelled(control->context))
        return SQ_CANCELLED;
    if (atomic_load(&si_mock_fail) == 1)
        return SQ_GPU_ERROR;
    for (size_t i = 0; i < limit; i++) {
        if (control->cancelled(control->context))
            return SQ_CANCELLED;
        /* Fixture token IDs 65/66 are ordinary UTF-8 text, no draft tokens. */
        int32_t token = i % 2 ? 66 : 65;
        stats->generation.generated_tokens++;
        if (!cb(ctx, token)) {
            stats->generation.finish_reason = SQ_FINISH_CALLBACK;
            break;
        }
    }
    if (atomic_load(&si_mock_pause) == 2) {
        atomic_store(&si_mock_arrived, 1);
        while (!atomic_load(&si_mock_resume))
            sched_yield();
    }
    return SQ_OK;
}
