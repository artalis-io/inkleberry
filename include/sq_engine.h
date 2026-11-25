/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_ENGINE_H
#define SQ_ENGINE_H
#include "sq_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct sq_session sq_session;

typedef struct {
    size_t struct_size;
    uint32_t api_version, context_tokens, kv_format, use_graphs, reserved;
} sq_session_config;

/* clang-format off */
#define SQ_SESSION_CONFIG_INIT {sizeof(sq_session_config),SQ_API_VERSION,8192,1,1,0}
/* clang-format on */

enum { SQ_KV_BF16, SQ_KV_FP8, SQ_KV_Q4 };

enum { SQ_FINISH_LENGTH, SQ_FINISH_EOS, SQ_FINISH_CALLBACK, SQ_FINISH_CANCELLED };

typedef struct {
    uint32_t prompt_tokens, generated_tokens, finish_reason;
    int32_t last_token;
} sq_generate_stats;

typedef int (*sq_token_callback)(void *context, int32_t token);
/* One live session per process initially. create retains the image; image
 * destruction returns BUSY until session destruction succeeds. create publishes
 * a handle even on partial device initialization failure: always destroy it.
 * Configuration/input errors do not publish a handle or mutate GPU state. */
SQ_API sq_status sq_session_create(sq_gpu_weights *, const sq_session_config *, const sq_options *,
                                   sq_session **);
/* Every generate starts an independent request; no prefix caching. Greedy only.
 * Callback 0 stops normally. Cancel is the only reentrant operation; cooperative
 * between completed forward chunks, never CUDA kernel preemption. Output/inputs
 * are caller-owned counted spans, borrowed until this synchronous call returns.
 * Callbacks must preserve the CUDA device/current context before returning. */
SQ_API sq_status sq_session_generate(sq_session *, const int32_t *, size_t, size_t, int32_t *,
                                     size_t, sq_token_callback, void *, sq_generate_stats *);
/* Teacher-forced tokens in batches <=8. Output is the last token's full
 * post-softcap FP32 vocabulary (200058 values). reset=1 begins an independent
 * sequence; reset=0 continues it. No sampling or generation occurs. */
SQ_API sq_status sq_session_logits(sq_session *, const int32_t *, size_t, uint32_t, float *,
                                   size_t);
SQ_API sq_status sq_session_cancel(sq_session *);
SQ_API sq_status sq_session_reset(sq_session *);
SQ_API sq_status sq_session_destroy(sq_session **);
#ifdef __cplusplus
}
#endif
#endif
