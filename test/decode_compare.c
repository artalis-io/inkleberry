/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Reuse the real public-API fixture loader, tokenizer and context checks. */
#include "ar_compare.c"
static size_t ifn_decode_compare_local_requests, ifn_decode_compare_local_verified_steps,
    ifn_decode_compare_local_verified_accepted;

int ifn_decode_start(const char *dir, uint32_t mode, uint32_t kv, uint32_t graphs, uint32_t count,
                     uint32_t drafts, uint32_t extra) {
    ifn_ar_compare_local_test_mtp = drafts != 0;
    ifn_ar_compare_local_test_context_extra = extra;
    return ifn_ar_test_start(dir, mode, kv, graphs, count, "Explain briefly why the sky is blue.");
}

int ifn_decode_check(float temperature, float top_p, int32_t k, uint64_t seed, uint32_t drafts,
                     const int32_t *expected, uint32_t count, uint32_t steps, uint32_t total,
                     uint32_t accepted) {
    ifn_decode_options options = IFN_DECODE_OPTIONS_INIT;
    options.temperature = temperature;
    options.top_p = top_p;
    options.top_k = k;
    options.seed = seed;
    options.draft_tokens = drafts;
    ifn_decode_stats stats;
    ifn_status rc = ifn_session_generate_ex(
        ifn_ar_compare_local_session, ifn_ar_compare_local_tokens, ifn_ar_compare_local_token_count,
        32, &options, ifn_ar_compare_local_c_output, 64, NULL, NULL, &stats);
    if (rc != IFN_OK || stats.generation.generated_tokens != count ||
        memcmp(ifn_ar_compare_local_c_output, expected, (size_t)count * 4) ||
        stats.speculative_steps != steps || stats.total_drafts != total ||
        stats.accepted_drafts != accepted) {
        fprintf(stderr,
                "decode mismatch seed=%llu T=%g p=%g k=%d drafts=%u C=%u/%u/%u Zig=%u/%u/%u "
                "status=%u %s\n",
                (unsigned long long)seed, (double)temperature, (double)top_p, k, drafts,
                stats.generation.generated_tokens, stats.speculative_steps, stats.accepted_drafts,
                count, steps, accepted, rc, ifn_last_error());
        for (uint32_t i = 0; i < count && i < stats.generation.generated_tokens; i++)
            if (ifn_ar_compare_local_c_output[i] != expected[i]) {
                fprintf(stderr, "first divergent token index=%u C=%d Zig=%d\n", i,
                        ifn_ar_compare_local_c_output[i], expected[i]);
                break;
            }
        return 1;
    }
    ifn_decode_compare_local_requests++;
    ifn_decode_compare_local_verified_steps += steps;
    ifn_decode_compare_local_verified_accepted += accepted;
    return 0;
}

int ifn_decode_finish(uint32_t success) {
    int rc = ifn_ar_test_finish(0);
    if (!rc && success)
        printf("{\"decode_parity\":true,\"requests\":%zu,\"speculative_steps\":%zu,\"accepted_"
               "drafts\":%zu}\n",
               ifn_decode_compare_local_requests, ifn_decode_compare_local_verified_steps,
               ifn_decode_compare_local_verified_accepted);
    return rc;
}
