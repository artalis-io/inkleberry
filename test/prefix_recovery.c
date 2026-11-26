/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * C-only prefix invalidation, cancellation, snapshot reuse and teardown. */
#include "prefix_compare.c"
static unsigned seen;

static int stop_at_four(void *ctx, int32_t token) {
    (void)ctx;
    (void)token;
    return ++seen < 4;
}

static int stop_at_one(void *ctx, int32_t token) {
    (void)ctx;
    (void)token;
    return 0;
}

static int cancel_at_one(void *ctx, int32_t token) {
    (void)ctx;
    (void)token;
    (void)ifn_session_cancel(session);
    return 1;
}

static int cancel_at_four(void *ctx, int32_t token) {
    (void)ctx;
    (void)token;
    if (++seen == 4)
        (void)ifn_session_cancel(session);
    return 1;
}

#define REQUIRE(x)                                                                                 \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "prefix recovery:%d %s\n", __LINE__, ifn_last_error());                \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    int failed = 0;
    int32_t baseline[64], out[64];
    ifn_decode_options o = IFN_DECODE_OPTIONS_INIT;
    o.system_prefix_tokens = 128;
    o.draft_tokens = 7;
    o.adaptive = 1;
    o.prompt_lookup = 1;
    o.temperature = 0.8f;
    o.top_p = 0.95f;
    o.top_k = 40;
    o.seed = 42;
    ifn_decode_stats initial, stats;
    ifn_prefix_info info = IFN_PREFIX_INFO_INIT;
    REQUIRE(!ifn_prefix_test_start(argv[1], 2, 1, 1, 8192, 1));
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, baseline, 64, NULL, NULL,
                                    &initial) == IFN_OK &&
            initial.generation.generated_tokens >= 4);
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            IFN_OK);
    REQUIRE(ifn_session_prefix_info(session, &info) == IFN_OK && info.reused_tokens == 2048 &&
            !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    REQUIRE(ifn_session_generate_ex(session, tokens + 7, 33, 16, &o, out, 64, NULL, NULL, &stats) ==
            IFN_OK);
    REQUIRE(ifn_session_prefix_info(session, &info) == IFN_OK && info.shielded_tokens > 0);
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
                IFN_OK &&
            !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    REQUIRE(ifn_session_generate_ex(session, tokens + 7, 33, 16, &o, out, 64, stop_at_one, NULL,
                                    &stats) == IFN_OK);
    REQUIRE(ifn_session_prefix_info(session, &info) == IFN_OK && info.shielded_tokens > 0);
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
                IFN_OK &&
            !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    REQUIRE(ifn_session_generate_ex(session, tokens + 7, 33, 16, &o, out, 64, cancel_at_one, NULL,
                                    &stats) == IFN_CANCELLED);
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
                IFN_OK &&
            !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    REQUIRE(ifn_session_prefix_info(session, &info) == IFN_OK && !info.reused_tokens);
    seen = 0;
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, stop_at_four, NULL,
                                    &stats) == IFN_OK &&
            stats.generation.generated_tokens == 4 &&
            stats.generation.finish_reason == IFN_FINISH_CALLBACK);
    seen = 0;
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, cancel_at_four, NULL,
                                    &stats) == IFN_CANCELLED &&
            stats.generation.generated_tokens == 4);
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            IFN_OK);
    REQUIRE(ifn_session_prefix_info(session, &info) == IFN_OK && !info.reused_tokens &&
            stats.generation.generated_tokens == initial.generation.generated_tokens &&
            !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    pthread_t thread;
    double requested = 0;
    REQUIRE(!pthread_create(&thread, NULL, cancel_thread, &requested));
    ifn_status rc =
        ifn_session_generate_ex(session, tokens, 8192, 16, &o, out, 64, NULL, NULL, &stats);
    double finished = now();
    REQUIRE(!pthread_join(thread, NULL) && rc == IFN_CANCELLED);
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            IFN_OK);
    REQUIRE(ifn_session_prefix_info(session, &info) == IFN_OK && !info.reused_tokens &&
            !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    REQUIRE(ifn_session_reset(session) == IFN_OK);
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            IFN_OK);
    REQUIRE(ifn_session_prefix_info(session, &info) == IFN_OK && !info.reused_tokens);
    float *values = malloc(200058 * 4);
    REQUIRE(values);
    rc = ifn_session_logits(session, tokens, 3, 1, values, 200058);
    free(values);
    REQUIRE(rc == IFN_OK);
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            IFN_OK);
    REQUIRE(ifn_session_prefix_info(session, &info) == IFN_OK && !info.reused_tokens);
    REQUIRE(ifn_session_prefix_cache(session, 0) == IFN_OK);
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            IFN_OK);
    REQUIRE(ifn_session_prefix_info(session, &info) == IFN_OK && !info.host_bytes &&
            !info.reused_tokens && !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    REQUIRE(ifn_session_prefix_cache(session, 1) == IFN_OK);
    REQUIRE(ifn_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            IFN_OK);
    REQUIRE(ifn_session_prefix_info(session, &info) == IFN_OK && !info.reused_tokens);
    printf("{\"native_prefix_recovery\":true,\"prefill_cancel_return_seconds\":%.6f,\"host_bytes\":"
           "%llu}\n",
           finished - requested, (unsigned long long)info.host_bytes);
done:
    if (ifn_prefix_test_finish(0))
        failed = 1;
    return failed;
}
