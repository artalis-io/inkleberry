/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Real public C prefix API versus the unchanged Zig Engine. */
#include "decode_compare.c"
static float request_temperature = 0.8f, request_p = 0.95f;
static int32_t request_k = 40;

void sq_prefix_test_sampler(float temperature, float p, int32_t k) {
    request_temperature = temperature;
    request_p = p;
    request_k = k;
}

static uint32_t maximum_reused;
static size_t prefix_requests, prefix_hits, prefix_restores, prefix_shields, live_hits;

int sq_prefix_test_start(const char *dir, uint32_t mode, uint32_t kv, uint32_t graphs,
                         uint32_t capacity, uint32_t mtp) {
    if (sq_decode_start(dir, mode, kv, graphs, capacity, mtp ? 7 : 0, 64))
        return 1;
    return sq_session_prefix_cache(session, 1) != SQ_OK;
}

int sq_prefix_test_check(const int32_t *prompt, uint32_t n, uint32_t hint, uint32_t drafts,
                         uint32_t adaptive, uint64_t seed, const int32_t *expected, uint32_t count,
                         uint32_t cached, uint32_t steps, uint32_t total, uint32_t accepted) {
    sq_decode_options o = SQ_DECODE_OPTIONS_INIT;
    o.temperature = request_temperature;
    o.top_p = request_p;
    o.top_k = request_k;
    o.seed = seed;
    o.system_prefix_tokens = hint;
    o.draft_tokens = drafts;
    o.adaptive = adaptive;
    o.prompt_lookup = adaptive;
    sq_decode_stats stats;
    sq_prefix_info info = SQ_PREFIX_INFO_INIT;
    sq_status rc =
        sq_session_generate_ex(session, prompt, n, 16, &o, c_output, 64, NULL, NULL, &stats);
    if (rc != SQ_OK || sq_session_prefix_info(session, &info) != SQ_OK ||
        stats.generation.generated_tokens != count ||
        memcmp(c_output, expected, (size_t)count * 4) || info.reused_tokens != cached ||
        stats.speculative_steps != steps || stats.total_drafts != total ||
        stats.accepted_drafts != accepted) {
        fprintf(stderr,
                "prefix mismatch request=%zu n=%u drafts=%u adaptive=%u cached C/Zig=%u/%u "
                "count=%u/%u status=%u %s\n",
                prefix_requests, n, drafts, adaptive, info.reused_tokens, cached,
                stats.generation.generated_tokens, count, rc, sq_last_error());
        for (uint32_t i = 0; i < count && i < stats.generation.generated_tokens; i++)
            if (c_output[i] != expected[i]) {
                fprintf(stderr, "first mismatch %u: %d/%d\n", i, c_output[i], expected[i]);
                break;
            }
        return 1;
    }
    if (info.reused_tokens > maximum_reused)
        maximum_reused = info.reused_tokens;
    if (info.reused_tokens && !info.checkpoint_restores)
        live_hits++;
    prefix_requests++;
    prefix_hits += info.reused_tokens > 0;
    prefix_restores += info.checkpoint_restores;
    prefix_shields += info.shielded_tokens > 0;
    return 0;
}

int sq_prefix_test_reset(void) {
    return sq_session_reset(session) != SQ_OK;
}

int sq_prefix_test_finish(uint32_t success) {
    int rc = sq_decode_finish(0);
    if (!rc && success)
        printf("{\"prefix_parity\":true,\"requests\":%zu,\"hits\":%zu,\"restores\":%zu,\"shields\":"
               "%zu,\"live_hits\":%zu,\"maximum_reused\":%u}\n",
               prefix_requests, prefix_hits, prefix_restores, prefix_shields, live_hits,
               maximum_reused);
    return rc;
}
