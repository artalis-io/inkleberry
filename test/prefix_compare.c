/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Real public C prefix API versus the unchanged Zig Engine. */
#include "decode_compare.c"
static float ifn_prefix_compare_local_request_temperature = 0.8f,
             ifn_prefix_compare_local_request_p = 0.95f;
static int32_t ifn_prefix_compare_local_request_k = 40;

void ifn_prefix_test_sampler(float temperature, float p, int32_t k) {
    ifn_prefix_compare_local_request_temperature = temperature;
    ifn_prefix_compare_local_request_p = p;
    ifn_prefix_compare_local_request_k = k;
}

static uint32_t ifn_prefix_compare_local_maximum_reused;
static size_t ifn_prefix_compare_local_prefix_requests, ifn_prefix_compare_local_prefix_hits,
    ifn_prefix_compare_local_prefix_restores, ifn_prefix_compare_local_prefix_shields,
    ifn_prefix_compare_local_live_hits;

int ifn_prefix_test_start(const char *dir, uint32_t mode, uint32_t kv, uint32_t graphs,
                          uint32_t capacity, uint32_t mtp) {
    if (ifn_decode_start(dir, mode, kv, graphs, capacity, mtp ? 7 : 0, 64))
        return 1;
    return ifn_session_prefix_cache(ifn_ar_compare_local_session, 1) != IFN_OK;
}

int ifn_prefix_test_check(const int32_t *prompt, uint32_t n, uint32_t hint, uint32_t drafts,
                          uint32_t adaptive, uint64_t seed, const int32_t *expected, uint32_t count,
                          uint32_t cached, uint32_t steps, uint32_t total, uint32_t accepted) {
    ifn_decode_options o = IFN_DECODE_OPTIONS_INIT;
    o.temperature = ifn_prefix_compare_local_request_temperature;
    o.top_p = ifn_prefix_compare_local_request_p;
    o.top_k = ifn_prefix_compare_local_request_k;
    o.seed = seed;
    o.system_prefix_tokens = hint;
    o.draft_tokens = drafts;
    o.adaptive = adaptive;
    o.prompt_lookup = adaptive;
    ifn_decode_stats stats;
    ifn_prefix_info info = IFN_PREFIX_INFO_INIT;
    ifn_status rc = ifn_session_generate_ex(ifn_ar_compare_local_session, prompt, n, 16, &o,
                                            ifn_ar_compare_local_c_output, 64, NULL, NULL, &stats);
    if (rc != IFN_OK || ifn_session_prefix_info(ifn_ar_compare_local_session, &info) != IFN_OK ||
        stats.generation.generated_tokens != count ||
        memcmp(ifn_ar_compare_local_c_output, expected, (size_t)count * 4) ||
        info.reused_tokens != cached || stats.speculative_steps != steps ||
        stats.total_drafts != total || stats.accepted_drafts != accepted) {
        fprintf(stderr,
                "prefix mismatch request=%zu n=%u drafts=%u adaptive=%u cached C/Zig=%u/%u "
                "count=%u/%u status=%u %s\n",
                ifn_prefix_compare_local_prefix_requests, n, drafts, adaptive, info.reused_tokens,
                cached, stats.generation.generated_tokens, count, rc, ifn_last_error());
        for (uint32_t i = 0; i < count && i < stats.generation.generated_tokens; i++)
            if (ifn_ar_compare_local_c_output[i] != expected[i]) {
                fprintf(stderr, "first mismatch %u: %d/%d\n", i, ifn_ar_compare_local_c_output[i],
                        expected[i]);
                break;
            }
        return 1;
    }
    if (info.reused_tokens > ifn_prefix_compare_local_maximum_reused)
        ifn_prefix_compare_local_maximum_reused = info.reused_tokens;
    if (info.reused_tokens && !info.checkpoint_restores)
        ifn_prefix_compare_local_live_hits++;
    ifn_prefix_compare_local_prefix_requests++;
    ifn_prefix_compare_local_prefix_hits += info.reused_tokens > 0;
    ifn_prefix_compare_local_prefix_restores += info.checkpoint_restores;
    ifn_prefix_compare_local_prefix_shields += info.shielded_tokens > 0;
    return 0;
}

int ifn_prefix_test_reset(void) {
    return ifn_session_reset(ifn_ar_compare_local_session) != IFN_OK;
}

int ifn_prefix_test_finish(uint32_t success) {
    int rc = ifn_decode_finish(0);
    if (!rc && success)
        printf("{\"prefix_parity\":true,\"requests\":%zu,\"hits\":%zu,\"restores\":%zu,\"shields\":"
               "%zu,\"live_hits\":%zu,\"maximum_reused\":%u}\n",
               ifn_prefix_compare_local_prefix_requests, ifn_prefix_compare_local_prefix_hits,
               ifn_prefix_compare_local_prefix_restores, ifn_prefix_compare_local_prefix_shields,
               ifn_prefix_compare_local_live_hits, ifn_prefix_compare_local_maximum_reused);
    return rc;
}
