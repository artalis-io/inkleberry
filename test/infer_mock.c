/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Fake execution checks facade ownership/concurrency, never GPU numerics. */
#include "infer_internal.h"
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

struct ifn_gpu_weights {
    ifn_options allocator;
};

struct ifn_session {
    ifn_options allocator;
};

atomic_int ifn_infer_mock_pause, ifn_infer_mock_arrived, ifn_infer_mock_resume;
atomic_uint ifn_infer_mock_uploads;
atomic_uint ifn_infer_mock_resets, ifn_infer_mock_calls, ifn_infer_mock_enables;
atomic_uint ifn_infer_mock_fail, ifn_infer_mock_close_fail, ifn_infer_mock_upload_fail;
const char *ifn_infer_mock_tokenizer;

ifn_status ifn_gpu_weights_create(const ifn_weight_config *c, const ifn_options *a,
                                  ifn_gpu_weights **out) {
    (void)c;
    *out = ifn_infer_alloc(a, sizeof **out);
    if (!*out)
        return IFN_OUT_OF_MEMORY;
    (*out)->allocator = *a;
    return IFN_OK;
}

ifn_status ifn_gpu_weights_load(ifn_gpu_weights *w, const ifn_inkling_checkpoint *cp) {
    (void)w;
    (void)cp;
    atomic_fetch_add(&ifn_infer_mock_uploads, 1);
    return atomic_load(&ifn_infer_mock_upload_fail) ? IFN_GPU_ERROR : IFN_OK;
}

ifn_status ifn_gpu_weights_destroy(ifn_gpu_weights **w) {
    if (*w) {
        if (atomic_load(&ifn_infer_mock_close_fail))
            return IFN_GPU_ERROR;
        ifn_infer_free(&(*w)->allocator, *w);
    }
    *w = NULL;
    return IFN_OK;
}

ifn_status ifn_session_create(ifn_gpu_weights *w, const ifn_session_config *c, const ifn_options *a,
                              ifn_session **out) {
    (void)w;
    (void)c;
    *out = ifn_infer_alloc(a, sizeof **out);
    if (!*out)
        return IFN_OUT_OF_MEMORY;
    (*out)->allocator = *a;
    return IFN_OK;
}

ifn_status ifn_session_destroy(ifn_session **s) {
    if (*s) {
        if (atomic_load(&ifn_infer_mock_close_fail))
            return IFN_GPU_ERROR;
        ifn_infer_free(&(*s)->allocator, *s);
    }
    *s = NULL;
    return IFN_OK;
}

ifn_status ifn_session_owned_plan(ifn_gpu_weights *w, const ifn_session_config *c,
                                  const ifn_options *a, uint64_t *out) {
    (void)w;
    (void)c;
    (void)a;
    *out = 1024;
    return IFN_OK;
}

ifn_status ifn_session_prefix_cache(ifn_session *s, uint32_t enabled) {
    (void)s;
    if (enabled)
        atomic_fetch_add(&ifn_infer_mock_enables, 1);
    return IFN_OK;
}

ifn_status ifn_session_prefix_info(ifn_session *s, ifn_prefix_info *out) {
    (void)s;
    out->reused_tokens = 0;
    return IFN_OK;
}

ifn_status ifn_session_reset(ifn_session *s) {
    (void)s;
    atomic_fetch_add(&ifn_infer_mock_resets, 1);
    return atomic_load(&ifn_infer_mock_fail) == 2
               ? ifn_native_error(IFN_GPU_ERROR, 700, "mock reset failure")
               : ifn_error(IFN_OK, "");
}

ifn_status ifn_session_generate_controlled(ifn_session *s, const int32_t *tokens, size_t n,
                                           size_t limit, const ifn_decode_options *options,
                                           ifn_token_callback cb, void *ctx,
                                           ifn_decode_stats *stats,
                                           const ifn_run_control *control) {
    (void)s;
    (void)tokens;
    (void)options;
    atomic_fetch_add(&ifn_infer_mock_calls, 1);
    if (atomic_load(&ifn_infer_mock_pause) == 1) {
        atomic_store(&ifn_infer_mock_arrived, 1);
        while (!atomic_load(&ifn_infer_mock_resume))
            sched_yield();
    }
    stats->generation.prompt_tokens = (uint32_t)n;
    if (control->cancelled(control->context))
        return IFN_CANCELLED;
    if (atomic_load(&ifn_infer_mock_fail) == 1)
        return ifn_native_error(IFN_GPU_ERROR, 719, "mock native operation");
    for (size_t i = 0; i < limit; i++) {
        if (control->cancelled(control->context))
            return IFN_CANCELLED;
        /* Fixture token IDs 65/66 are ordinary UTF-8 text, no draft tokens. */
        int32_t token = i % 2 ? 66 : 65;
        stats->generation.generated_tokens++;
        if (!cb(ctx, token)) {
            stats->generation.finish_reason = IFN_FINISH_CALLBACK;
            break;
        }
    }
    if (atomic_load(&ifn_infer_mock_pause) == 2) {
        atomic_store(&ifn_infer_mock_arrived, 1);
        while (!atomic_load(&ifn_infer_mock_resume))
            sched_yield();
    }
    return IFN_OK;
}
