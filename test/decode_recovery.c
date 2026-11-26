/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * C-only sampler/speculation state isolation, cancellation and ownership. */
#include "decode_compare.c"
static unsigned ifn_decode_recovery_local_emitted, ifn_decode_recovery_local_cutoff;

static int ifn_decode_recovery_local_stop_callback(void *context, int32_t token) {
    (void)context;
    (void)token;
    ifn_decode_recovery_local_emitted++;
    return ifn_decode_recovery_local_emitted < ifn_decode_recovery_local_cutoff;
}

static int ifn_decode_recovery_local_cancel_callback(void *context, int32_t token) {
    (void)context;
    (void)token;
    ifn_decode_recovery_local_emitted++;
    if (ifn_decode_recovery_local_emitted == ifn_decode_recovery_local_cutoff)
        (void)ifn_session_cancel(ifn_ar_compare_local_session);
    return 1;
}

#define IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(x)                                                \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "decode recovery:%d %s\n", __LINE__, ifn_last_error());                \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    int failed = 0;
    int32_t expected[64], other[64];
    ifn_decode_stats baseline, stats;
    ifn_decode_options options = IFN_DECODE_OPTIONS_INIT;
    options.temperature = 0.8f;
    options.top_p = 0.95f;
    options.top_k = 40;
    options.seed = 42;
    options.draft_tokens = 7;
#ifdef IFN_TEST_ADAPTIVE
    options.adaptive = 1;
    options.prompt_lookup = 1;
#endif
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(ifn_decode_start(argv[1], 2, 1, 1, 8192, 7, 64) == 0);
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(
        ifn_session_generate_ex(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                                ifn_ar_compare_local_token_count, 32, &options, expected, 64, NULL,
                                NULL, &baseline) == IFN_OK);
    ifn_decode_recovery_local_cutoff =
        baseline.generation.generated_tokens < 4 ? baseline.generation.generated_tokens : 4;
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(ifn_decode_recovery_local_cutoff > 0);
    ifn_decode_recovery_local_emitted = 0;
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(
        ifn_session_generate_ex(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                                ifn_ar_compare_local_token_count, 32, &options, other, 64,
                                ifn_decode_recovery_local_stop_callback, NULL, &stats) == IFN_OK &&
        stats.generation.finish_reason == IFN_FINISH_CALLBACK &&
        stats.generation.generated_tokens == ifn_decode_recovery_local_cutoff &&
        !memcmp(other, expected, ifn_decode_recovery_local_cutoff * 4));
    ifn_decode_recovery_local_emitted = 0;
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(
        ifn_session_generate_ex(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                                ifn_ar_compare_local_token_count, 32, &options, other, 64,
                                ifn_decode_recovery_local_cancel_callback, NULL,
                                &stats) == IFN_CANCELLED &&
        stats.generation.finish_reason == IFN_FINISH_CANCELLED &&
        stats.generation.generated_tokens == ifn_decode_recovery_local_cutoff &&
        !memcmp(other, expected, ifn_decode_recovery_local_cutoff * 4));
    /* Exercise an independent AR request between two identical DSpark requests. */
    ifn_decode_options ar = options;
    ar.draft_tokens = 0;
    ar.adaptive = ar.prompt_lookup = 0;
    ar.seed = UINT64_MAX;
    const int32_t alternate[3] = {100, 200, 300};
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(
        ifn_session_generate_ex(ifn_ar_compare_local_session, alternate, 3, 16, &ar, other, 64,
                                NULL, NULL, &stats) == IFN_OK);
    ifn_decode_options narrower = options;
    narrower.draft_tokens = 3;
    narrower.adaptive = narrower.prompt_lookup = 0;
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(
        ifn_session_generate_ex(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                                ifn_ar_compare_local_token_count, 32, &narrower, other, 64, NULL,
                                NULL, &stats) == IFN_OK);
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(
        ifn_session_generate_ex(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                                ifn_ar_compare_local_token_count, 32, &options, other, 64, NULL,
                                NULL, &stats) == IFN_OK &&
        stats.generation.generated_tokens == baseline.generation.generated_tokens &&
        !memcmp(other, expected, stats.generation.generated_tokens * 4) &&
        stats.speculative_steps == baseline.speculative_steps &&
        stats.accepted_drafts == baseline.accepted_drafts);
#ifdef IFN_TEST_ADAPTIVE
    /* Rebuild the conditional graph, then restore its previous threshold. */
    ifn_decode_options changed = options;
    changed.spec_threshold = 0.2f;
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(
        ifn_session_generate_ex(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                                ifn_ar_compare_local_token_count, 32, &changed, other, 64, NULL,
                                NULL, &stats) == IFN_OK);
#endif
    ifn_test_thread thread;
    double requested = 0;
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(
        !ifn_test_thread_create(&thread, NULL, ifn_ar_compare_local_cancel_thread, &requested));
    ifn_status cancelled_status = ifn_session_generate_ex(
        ifn_ar_compare_local_session, ifn_ar_compare_local_tokens, ifn_ar_compare_local_token_count,
        32, &options, other, 64, NULL, NULL, &stats);
    double finished = ifn_ar_compare_local_now();
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(
        !ifn_test_thread_join(thread, NULL) && cancelled_status == IFN_CANCELLED &&
        stats.generation.finish_reason == IFN_FINISH_CANCELLED);
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(
        ifn_session_generate_ex(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                                ifn_ar_compare_local_token_count, 32, &options, other, 64, NULL,
                                NULL, &stats) == IFN_OK &&
        stats.generation.generated_tokens == baseline.generation.generated_tokens &&
        !memcmp(other, expected, stats.generation.generated_tokens * 4) &&
        stats.speculative_steps == baseline.speculative_steps &&
        stats.accepted_drafts == baseline.accepted_drafts);
#ifdef IFN_TEST_ADAPTIVE
    ifn_decode_info info = IFN_DECODE_INFO_INIT;
    IFN_DECODE_RECOVERY_LOCAL_CHECK_RECOVERY(
        ifn_session_decode_info(ifn_ar_compare_local_session, &info) == IFN_OK &&
        info.draft_launches > 0);
    puts("{\"native_adaptive_recovery\":true}");
#endif
    printf("{\"native_decode_recovery\":true,\"prompt_tokens\":8192,\"callback_cutoff\":%u,"
           "\"prefill_cancel_return_seconds\":%.6f}\n",
           ifn_decode_recovery_local_cutoff, finished - requested);
done:
    if (ifn_decode_finish(0))
        failed = 1;
    return failed;
}
