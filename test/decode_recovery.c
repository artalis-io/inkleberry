/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * C-only sampler/speculation state isolation, cancellation and ownership. */
#include "decode_compare.c"
static unsigned emitted, cutoff;

static int stop_callback(void *context, int32_t token) {
    (void)context;
    (void)token;
    emitted++;
    return emitted < cutoff;
}

static int cancel_callback(void *context, int32_t token) {
    (void)context;
    (void)token;
    emitted++;
    if (emitted == cutoff)
        (void)sq_session_cancel(session);
    return 1;
}

#define CHECK_RECOVERY(x)                                                                          \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "decode recovery:%d %s\n", __LINE__, sq_last_error());                 \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    int failed = 0;
    int32_t expected[64], other[64];
    sq_decode_stats baseline, stats;
    sq_decode_options options = SQ_DECODE_OPTIONS_INIT;
    options.temperature = 0.8f;
    options.top_p = 0.95f;
    options.top_k = 40;
    options.seed = 42;
    options.draft_tokens = 7;
#ifdef SQ_TEST_ADAPTIVE
    options.adaptive = 1;
    options.prompt_lookup = 1;
#endif
    CHECK_RECOVERY(sq_decode_start(argv[1], 2, 1, 1, 8192, 7, 64) == 0);
    CHECK_RECOVERY(sq_session_generate_ex(session, tokens, token_count, 32, &options, expected, 64,
                                          NULL, NULL, &baseline) == SQ_OK);
    cutoff = baseline.generation.generated_tokens < 4 ? baseline.generation.generated_tokens : 4;
    CHECK_RECOVERY(cutoff > 0);
    emitted = 0;
    CHECK_RECOVERY(sq_session_generate_ex(session, tokens, token_count, 32, &options, other, 64,
                                          stop_callback, NULL, &stats) == SQ_OK &&
                   stats.generation.finish_reason == SQ_FINISH_CALLBACK &&
                   stats.generation.generated_tokens == cutoff &&
                   !memcmp(other, expected, cutoff * 4));
    emitted = 0;
    CHECK_RECOVERY(sq_session_generate_ex(session, tokens, token_count, 32, &options, other, 64,
                                          cancel_callback, NULL, &stats) == SQ_CANCELLED &&
                   stats.generation.finish_reason == SQ_FINISH_CANCELLED &&
                   stats.generation.generated_tokens == cutoff &&
                   !memcmp(other, expected, cutoff * 4));
    /* Exercise an independent AR request between two identical DSpark requests. */
    sq_decode_options ar = options;
    ar.draft_tokens = 0;
    ar.adaptive = ar.prompt_lookup = 0;
    ar.seed = UINT64_MAX;
    const int32_t alternate[3] = {100, 200, 300};
    CHECK_RECOVERY(sq_session_generate_ex(session, alternate, 3, 16, &ar, other, 64, NULL, NULL,
                                          &stats) == SQ_OK);
    sq_decode_options narrower = options;
    narrower.draft_tokens = 3;
    narrower.adaptive = narrower.prompt_lookup = 0;
    CHECK_RECOVERY(sq_session_generate_ex(session, tokens, token_count, 32, &narrower, other, 64,
                                          NULL, NULL, &stats) == SQ_OK);
    CHECK_RECOVERY(sq_session_generate_ex(session, tokens, token_count, 32, &options, other, 64,
                                          NULL, NULL, &stats) == SQ_OK &&
                   stats.generation.generated_tokens == baseline.generation.generated_tokens &&
                   !memcmp(other, expected, stats.generation.generated_tokens * 4) &&
                   stats.speculative_steps == baseline.speculative_steps &&
                   stats.accepted_drafts == baseline.accepted_drafts);
#ifdef SQ_TEST_ADAPTIVE
    /* Rebuild the conditional graph, then restore its previous threshold. */
    sq_decode_options changed = options;
    changed.spec_threshold = 0.2f;
    CHECK_RECOVERY(sq_session_generate_ex(session, tokens, token_count, 32, &changed, other, 64,
                                          NULL, NULL, &stats) == SQ_OK);
#endif
    pthread_t thread;
    double requested = 0;
    CHECK_RECOVERY(!pthread_create(&thread, NULL, cancel_thread, &requested));
    sq_status cancelled_status = sq_session_generate_ex(session, tokens, token_count, 32, &options,
                                                        other, 64, NULL, NULL, &stats);
    double finished = now();
    CHECK_RECOVERY(!pthread_join(thread, NULL) && cancelled_status == SQ_CANCELLED &&
                   stats.generation.finish_reason == SQ_FINISH_CANCELLED);
    CHECK_RECOVERY(sq_session_generate_ex(session, tokens, token_count, 32, &options, other, 64,
                                          NULL, NULL, &stats) == SQ_OK &&
                   stats.generation.generated_tokens == baseline.generation.generated_tokens &&
                   !memcmp(other, expected, stats.generation.generated_tokens * 4) &&
                   stats.speculative_steps == baseline.speculative_steps &&
                   stats.accepted_drafts == baseline.accepted_drafts);
#ifdef SQ_TEST_ADAPTIVE
    sq_decode_info info = SQ_DECODE_INFO_INIT;
    CHECK_RECOVERY(sq_session_decode_info(session, &info) == SQ_OK && info.draft_launches > 0);
    puts("{\"native_adaptive_recovery\":true}");
#endif
    printf("{\"native_decode_recovery\":true,\"prompt_tokens\":8192,\"callback_cutoff\":%u,"
           "\"prefill_cancel_return_seconds\":%.6f}\n",
           cutoff, finished - requested);
done:
    if (sq_decode_finish(0))
        failed = 1;
    return failed;
}
