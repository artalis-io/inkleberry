/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Native C public-API qualification, invoked by unchanged Zig engine oracle. */
#define _POSIX_C_SOURCE 200809L
#include "sq_engine.h"
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern int sq_ar_test_context(sq_session *);
static sq_gpu_weights *weights;
static double cancel_seconds;
static sq_session *session;
static int32_t *tokens;
static size_t token_count, checked_logits;
static float logits[200058];
static int32_t c_output[64];
static uint32_t generated;

int sq_ar_test_start(const char *directory, uint32_t mode, uint32_t kv, uint32_t graphs,
                     uint32_t count, const char *text) {
    char path[1200], rendered[8192];
    size_t bytes = 0, n = 0;
    int32_t encoded[4096];
    sq_tokenizer *t = NULL;
    sq_tokenizer_scratch *scratch = NULL;
    if (strlen(directory) > 1000 || strlen(text) > 4096 || !count || count > 131072)
        return 1;
    (void)snprintf(path, sizeof path, "%s/tokenizer.json", directory);
    sq_status rc = sq_tokenizer_open(path, NULL, &t);
    if (rc == SQ_OK)
        rc = sq_tokenizer_scratch_create(NULL, &scratch);
    if (rc == SQ_OK)
        rc = sq_render_user_prompt(text, strlen(text), SQ_THINK_NONE, rendered, sizeof rendered,
                                   &bytes);
    if (rc == SQ_OK)
        rc = sq_tokenizer_encode(t, rendered, bytes, 1, scratch, encoded, 4096, &n);
    sq_tokenizer_scratch_destroy(scratch);
    sq_tokenizer_close(t);
    if (rc != SQ_OK || !n)
        return 1;
    token_count = count;
    tokens = malloc(token_count * sizeof *tokens);
    if (!tokens)
        return 1;
    for (size_t i = 0; i < token_count; i++)
        tokens[i] = encoded[i % n];
    sq_inkling_checkpoint *cp = NULL;
    (void)snprintf(path, sizeof path, "%s/model-00001-of-00001.safetensors", directory);
    rc = sq_inkling_checkpoint_open(path, NULL, NULL, &cp);
    sq_weight_config wc = SQ_WEIGHT_CONFIG_INIT;
    wc.mode = mode;
    wc.with_mtp = 0;
    if (rc == SQ_OK)
        rc = sq_gpu_weights_create(&wc, NULL, &weights);
    if (rc == SQ_OK)
        rc = sq_gpu_weights_load(weights, cp);
    sq_inkling_checkpoint_close(cp);
    sq_session_config config = SQ_SESSION_CONFIG_INIT;
    config.context_tokens = count + 64 < 64 ? 64 : count + 64;
    config.kv_format = kv;
    config.use_graphs = graphs;
    if (rc == SQ_OK)
        rc = sq_session_create(weights, &config, NULL, &session);
    if (rc != SQ_OK) {
        fprintf(stderr, "C AR initialize: %s\n", sq_last_error());
        return 1;
    }
    if (sq_ar_test_context(session))
        return 1;
    sq_gpu_weights *copy = weights;
    if (sq_gpu_weights_destroy(&copy) != SQ_BUSY || copy != weights)
        return 1;
    sq_session *second = NULL;
    if (sq_session_create(weights, &config, NULL, &second) != SQ_BUSY || second)
        return 1;
    return 0;
}

const int32_t *sq_ar_test_tokens(void) {
    return tokens;
}

int sq_ar_test_logits(size_t offset, const float *reference, size_t n) {
    if (!n || n > 8 || offset + n > token_count)
        return 1;
    if (sq_session_logits(session, tokens + offset, n, offset == 0, logits, 200058) != SQ_OK) {
        fprintf(stderr, "C logits: %s\n", sq_last_error());
        return 1;
    }
    reference += (n - 1) * 200058;
    for (size_t i = 0; i < 200058; i++) {
        if (!isfinite(logits[i]) || memcmp(logits + i, reference + i, 4)) {
            fprintf(stderr, "logits mismatch pos=%zu vocab=%zu C=%.9g Zig=%.9g\n", offset + n - 1,
                    i, logits[i], reference[i]);
            return 1;
        }
    }
    checked_logits += 200058;
    return 0;
}

int sq_ar_test_generate(const int32_t *expected, size_t n) {
    sq_generate_stats stats;
    sq_status rc =
        sq_session_generate(session, tokens, token_count, 32, c_output, 64, NULL, NULL, &stats);
    if (rc != SQ_OK || stats.generated_tokens != n ||
        memcmp(c_output, expected, n * sizeof *expected)) {
        fprintf(stderr, "C/Zig generation mismatch: status=%u C=%u Zig=%zu %s\n", rc,
                stats.generated_tokens, n, sq_last_error());
        return 1;
    }
    generated = stats.generated_tokens;
    return 0;
}

static int stop_after_one(void *ctx, int32_t token) {
    (void)ctx;
    (void)token;
    return 0;
}

static int cancel_after_one(void *ctx, int32_t token) {
    (void)token;
    return sq_session_cancel(ctx) == SQ_OK;
}

int sq_ar_test_reuse(void) {
    sq_generate_stats stats;
    int32_t out[64];
    int32_t bad = -1;
    if (sq_session_generate(session, &bad, 1, 2, out, 64, NULL, NULL, &stats) !=
        SQ_INVALID_ARGUMENT)
        return 1;
    if (sq_session_generate(session, tokens, token_count, 2, out, 64, stop_after_one, NULL,
                            &stats) != SQ_OK ||
        stats.finish_reason != SQ_FINISH_CALLBACK || stats.generated_tokens != 1)
        return 1;
    if (sq_session_generate(session, tokens, token_count, 32, out, 64, cancel_after_one, session,
                            &stats) != SQ_CANCELLED ||
        stats.finish_reason != SQ_FINISH_CANCELLED || stats.generated_tokens != 1)
        return 1;
    const int32_t alternate[3] = {100, 200, 300};
    if (sq_session_generate(session, alternate, 3, 16, out, 64, NULL, NULL, &stats) != SQ_OK)
        return 1;
    if (sq_session_reset(session) != SQ_OK)
        return 1;
    if (sq_session_generate(session, tokens, token_count, 32, out, 64, NULL, NULL, &stats) !=
            SQ_OK ||
        stats.generated_tokens != generated || memcmp(out, c_output, generated * sizeof *out))
        return 1;
    return 0;
}

static double now(void) {
    struct timespec t;
    (void)clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void *cancel_thread(void *context) {
    double *requested = context;
    struct timespec delay = {0, 50000000};
    (void)nanosleep(&delay, NULL);
    *requested = now();
    (void)sq_session_cancel(session);
    return NULL;
}

int sq_ar_test_cancel_prefill(void) {
    pthread_t thread;
    double requested = 0;
    if (pthread_create(&thread, NULL, cancel_thread, &requested))
        return 1;
    int32_t output[64];
    sq_generate_stats stats;
    sq_status rc =
        sq_session_generate(session, tokens, token_count, 32, output, 64, NULL, NULL, &stats);
    double finished = now();
    if (pthread_join(thread, NULL) || rc != SQ_CANCELLED ||
        stats.finish_reason != SQ_FINISH_CANCELLED)
        return 1;
    cancel_seconds = finished - requested;
    rc = sq_session_generate(session, tokens, token_count, 32, output, 64, NULL, NULL, &stats);
    return rc != SQ_OK || stats.generated_tokens != generated ||
           memcmp(output, c_output, generated * sizeof *output);
}

int sq_ar_test_finish(uint32_t success) {
    int failed = sq_session_destroy(&session) != SQ_OK;
    if (sq_gpu_weights_destroy(&weights) != SQ_OK)
        failed = 1;
    free(tokens);
    tokens = NULL;
    if (!failed && success)
        printf("{\"c_ar_parity\":true,\"prompt_tokens\":%zu,\"logits_compared\":%zu,\"generated_"
               "tokens\":%u,\"cancel_return_seconds\":%.6f,\"caller_context_preserved\":true}\n",
               token_count, checked_logits, generated, cancel_seconds);
    return failed;
}
