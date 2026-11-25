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
    (void)sq_session_cancel(session);
    return 1;
}

static int cancel_at_four(void *ctx, int32_t token) {
    (void)ctx;
    (void)token;
    if (++seen == 4)
        (void)sq_session_cancel(session);
    return 1;
}

#define REQUIRE(x)                                                                                 \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "prefix recovery:%d %s\n", __LINE__, sq_last_error());                 \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    int failed = 0;
    int32_t baseline[64], out[64];
    sq_decode_options o = SQ_DECODE_OPTIONS_INIT;
    o.system_prefix_tokens = 128;
    o.draft_tokens = 7;
    o.adaptive = 1;
    o.prompt_lookup = 1;
    o.temperature = 0.8f;
    o.top_p = 0.95f;
    o.top_k = 40;
    o.seed = 42;
    sq_decode_stats initial, stats;
    sq_prefix_info info = SQ_PREFIX_INFO_INIT;
    REQUIRE(!sq_prefix_test_start(argv[1], 2, 1, 1, 8192, 1));
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, baseline, 64, NULL, NULL,
                                   &initial) == SQ_OK &&
            initial.generation.generated_tokens >= 4);
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            SQ_OK);
    REQUIRE(sq_session_prefix_info(session, &info) == SQ_OK && info.reused_tokens == 2048 &&
            !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    REQUIRE(sq_session_generate_ex(session, tokens + 7, 33, 16, &o, out, 64, NULL, NULL, &stats) ==
            SQ_OK);
    REQUIRE(sq_session_prefix_info(session, &info) == SQ_OK && info.shielded_tokens > 0);
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
                SQ_OK &&
            !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    REQUIRE(sq_session_generate_ex(session, tokens + 7, 33, 16, &o, out, 64, stop_at_one, NULL,
                                   &stats) == SQ_OK);
    REQUIRE(sq_session_prefix_info(session, &info) == SQ_OK && info.shielded_tokens > 0);
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
                SQ_OK &&
            !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    REQUIRE(sq_session_generate_ex(session, tokens + 7, 33, 16, &o, out, 64, cancel_at_one, NULL,
                                   &stats) == SQ_CANCELLED);
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
                SQ_OK &&
            !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    REQUIRE(sq_session_prefix_info(session, &info) == SQ_OK && !info.reused_tokens);
    seen = 0;
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, stop_at_four, NULL,
                                   &stats) == SQ_OK &&
            stats.generation.generated_tokens == 4 &&
            stats.generation.finish_reason == SQ_FINISH_CALLBACK);
    seen = 0;
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, cancel_at_four, NULL,
                                   &stats) == SQ_CANCELLED &&
            stats.generation.generated_tokens == 4);
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            SQ_OK);
    REQUIRE(sq_session_prefix_info(session, &info) == SQ_OK && !info.reused_tokens &&
            stats.generation.generated_tokens == initial.generation.generated_tokens &&
            !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    pthread_t thread;
    double requested = 0;
    REQUIRE(!pthread_create(&thread, NULL, cancel_thread, &requested));
    sq_status rc =
        sq_session_generate_ex(session, tokens, 8192, 16, &o, out, 64, NULL, NULL, &stats);
    double finished = now();
    REQUIRE(!pthread_join(thread, NULL) && rc == SQ_CANCELLED);
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            SQ_OK);
    REQUIRE(sq_session_prefix_info(session, &info) == SQ_OK && !info.reused_tokens &&
            !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    REQUIRE(sq_session_reset(session) == SQ_OK);
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            SQ_OK);
    REQUIRE(sq_session_prefix_info(session, &info) == SQ_OK && !info.reused_tokens);
    float *values = malloc(200058 * 4);
    REQUIRE(values);
    rc = sq_session_logits(session, tokens, 3, 1, values, 200058);
    free(values);
    REQUIRE(rc == SQ_OK);
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            SQ_OK);
    REQUIRE(sq_session_prefix_info(session, &info) == SQ_OK && !info.reused_tokens);
    REQUIRE(sq_session_prefix_cache(session, 0) == SQ_OK);
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            SQ_OK);
    REQUIRE(sq_session_prefix_info(session, &info) == SQ_OK && !info.host_bytes &&
            !info.reused_tokens && !memcmp(out, baseline, stats.generation.generated_tokens * 4));
    REQUIRE(sq_session_prefix_cache(session, 1) == SQ_OK);
    REQUIRE(sq_session_generate_ex(session, tokens, 2049, 16, &o, out, 64, NULL, NULL, &stats) ==
            SQ_OK);
    REQUIRE(sq_session_prefix_info(session, &info) == SQ_OK && !info.reused_tokens);
    printf("{\"native_prefix_recovery\":true,\"prefill_cancel_return_seconds\":%.6f,\"host_bytes\":"
           "%llu}\n",
           finished - requested, (unsigned long long)info.host_bytes);
done:
    if (sq_prefix_test_finish(0))
        failed = 1;
    return failed;
}
