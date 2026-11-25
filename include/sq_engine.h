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

typedef struct {
    size_t struct_size;
    uint32_t api_version;
    float temperature, top_p;
    int32_t top_k;
    uint64_t seed;
    uint32_t draft_tokens, reserved;
    uint32_t adaptive, prompt_lookup;
    float spec_threshold;
    uint32_t adaptive_reserved;
} sq_decode_options;

/* clang-format off */
#define SQ_DECODE_OPTIONS_INIT {sizeof(sq_decode_options),SQ_API_VERSION,0.0f,0.95f,0,0,0,0,0,0,0.8f,0}
/* clang-format on */

typedef struct {
    sq_generate_stats generation;
    uint32_t speculative_steps, total_drafts, accepted_drafts;
} sq_decode_stats;

typedef struct {
    size_t struct_size;
    uint32_t api_version, lookup_steps, width2, width4, width8, draft_launches;
} sq_decode_info;

/* clang-format off */
#define SQ_DECODE_INFO_INIT {sizeof(sq_decode_info),SQ_API_VERSION,0,0,0,0,0}
/* clang-format on */
/* Initialize output with SQ_DECODE_INFO_INIT. Read-only completed-request
 * diagnostics, also available after cancellation/error; BUSY while active. */
SQ_API sq_status sq_session_decode_info(sq_session *, sq_decode_info *);

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
/* Extended generation: finite temperature >=0, 0<top_p<=1, top_k=0..200058,
 * deterministic uint64 seed (including zero), fixed draft width 0..7.
 * Temperature <1e-4 or top_k=1 selects greedy; zero drafts selects AR.
 * Positive draft width requires an image loaded with MTP; never falls back
 * silently. Adaptive mode requires seven drafts, graphs and a finite threshold
 * in (0,1]; prompt lookup is optional in adaptive mode. Initialize with
 * SQ_DECODE_OPTIONS_INIT. The legacy options prefix ending at reserved remains
 * accepted with adaptive/lookup disabled. No prefix caching.
 * Keep output/stats storage separate from input/options storage.
 * Callback/cancellation/ownership contracts match sq_session_generate. */
SQ_API sq_status sq_session_generate_ex(sq_session *, const int32_t *, size_t, size_t,
                                        const sq_decode_options *, int32_t *, size_t,
                                        sq_token_callback, void *, sq_decode_stats *);

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
