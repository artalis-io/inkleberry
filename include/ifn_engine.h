/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_ENGINE_H
#define IFN_ENGINE_H
#include "ifn_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ifn_session ifn_session;

typedef struct {
    size_t struct_size;
    uint32_t api_version, context_tokens, kv_format, use_graphs, reserved;
} ifn_session_config;

/* clang-format off */
#define IFN_SESSION_CONFIG_INIT {sizeof(ifn_session_config),IFN_API_VERSION,8192,1,1,0}
/* clang-format on */

enum { IFN_KV_BF16, IFN_KV_FP8, IFN_KV_Q4 };

enum { IFN_FINISH_LENGTH, IFN_FINISH_EOS, IFN_FINISH_CALLBACK, IFN_FINISH_CANCELLED };

typedef struct {
    uint32_t prompt_tokens, generated_tokens, finish_reason;
    int32_t last_token;
} ifn_generate_stats;

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
    uint32_t system_prefix_tokens, prefix_reserved;
} ifn_decode_options;

/* clang-format off */
#define IFN_DECODE_OPTIONS_INIT {sizeof(ifn_decode_options),IFN_API_VERSION,0.0f,0.95f,0,0,0,0,0,0,0.8f,0,0,0}
/* clang-format on */

typedef struct {
    ifn_generate_stats generation;
    uint32_t speculative_steps, total_drafts, accepted_drafts;
} ifn_decode_stats;

typedef struct {
    size_t struct_size;
    uint32_t api_version, lookup_steps, width2, width4, width8, draft_launches;
} ifn_decode_info;

/* clang-format off */
#define IFN_DECODE_INFO_INIT {sizeof(ifn_decode_info),IFN_API_VERSION,0,0,0,0,0}
/* clang-format on */
/* Initialize output with IFN_DECODE_INFO_INIT. Read-only completed-request
 * diagnostics, also available after cancellation/error; BUSY while active. */
IFN_API ifn_status ifn_session_decode_info(ifn_session *, ifn_decode_info *);

typedef struct {
    size_t struct_size;
    uint32_t api_version, reused_tokens, shielded_tokens, checkpoint_restores;
    uint64_t host_bytes;
} ifn_prefix_info;

/* clang-format off */
#define IFN_PREFIX_INFO_INIT {sizeof(ifn_prefix_info),IFN_API_VERSION,0,0,0,0}
/* clang-format on */
/* Explicit opt-in, session-private host RAM cache; no additional device arena.
 * Allocation failure leaves the previous configuration intact. Disable, reset,
 * logits and cancelled/failed execution invalidate reuse. BUSY while active. */
IFN_API ifn_status ifn_session_prefix_cache(ifn_session *, uint32_t enabled);
IFN_API ifn_status ifn_session_prefix_info(ifn_session *, ifn_prefix_info *);

typedef int (*ifn_token_callback)(void *context, int32_t token);
/* One live session per process initially. create retains the image; image
 * destruction returns BUSY until session destruction succeeds. create publishes
 * a handle even on partial device initialization failure: always destroy it.
 * Configuration/input errors do not publish a handle or mutate GPU state. */
IFN_API ifn_status ifn_session_create(ifn_gpu_weights *, const ifn_session_config *,
                                      const ifn_options *, ifn_session **);
/* Each generate has an independent RNG/output sequence. Greedy only.
 * Prefix reuse is disabled by default; explicitly enabled caches are session-private.
 * Callback 0 stops normally. Cancel is the only reentrant operation; cooperative
 * between completed forward chunks, never CUDA kernel preemption. Output/inputs
 * are caller-owned counted spans, borrowed until this synchronous call returns.
 * Callbacks must preserve the CUDA device/current context before returning. */
IFN_API ifn_status ifn_session_generate(ifn_session *, const int32_t *, size_t, size_t, int32_t *,
                                        size_t, ifn_token_callback, void *, ifn_generate_stats *);
/* Extended generation: finite temperature >=0, 0<top_p<=1, top_k=0..200058,
 * deterministic uint64 seed (including zero), fixed draft width 0..7.
 * Temperature <1e-4 or top_k=1 selects greedy; zero drafts selects AR.
 * Positive draft width requires an image loaded with MTP; never falls back
 * silently. Adaptive mode requires seven drafts, graphs and a finite threshold
 * in (0,1]; prompt lookup is optional in adaptive mode. Initialize with
 * IFN_DECODE_OPTIONS_INIT. The legacy options prefix ending at reserved remains
 * accepted with adaptive/lookup disabled; the preceding adaptive-options size
 * is also accepted. system_prefix_tokens is an advisory checkpoint boundary,
 * used only with enabled prefix caching (128 <= hint < prompt length).
 * Keep output/stats storage separate from input/options storage.
 * Callback/cancellation/ownership contracts match ifn_session_generate. */
IFN_API ifn_status ifn_session_generate_ex(ifn_session *, const int32_t *, size_t, size_t,
                                           const ifn_decode_options *, int32_t *, size_t,
                                           ifn_token_callback, void *, ifn_decode_stats *);

/* Teacher-forced tokens in batches <=8. Output is the last token's full
 * post-softcap FP32 vocabulary (200058 values). reset=1 begins an independent
 * sequence; reset=0 continues it. No sampling or generation occurs. */
IFN_API ifn_status ifn_session_logits(ifn_session *, const int32_t *, size_t, uint32_t, float *,
                                      size_t);
IFN_API ifn_status ifn_session_cancel(ifn_session *);
IFN_API ifn_status ifn_session_reset(ifn_session *);
IFN_API ifn_status ifn_session_destroy(ifn_session **);
#ifdef __cplusplus
}
#endif
#endif
