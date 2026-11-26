/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Native C public-API qualification shared by native recovery tests. */
#define _POSIX_C_SOURCE 200809L
#include "ifn_engine.h"
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern int ifn_ar_test_context(ifn_session *);
static ifn_gpu_weights *ifn_ar_compare_local_weights;
static int ifn_ar_compare_local_test_mtp;
static uint32_t ifn_ar_compare_local_test_context_extra = 64;
static double ifn_ar_compare_local_cancel_seconds;
static ifn_session *ifn_ar_compare_local_session;
static int32_t *ifn_ar_compare_local_tokens;
static size_t ifn_ar_compare_local_token_count, ifn_ar_compare_local_checked_logits;
static float ifn_ar_compare_local_logits[200058];
static int32_t ifn_ar_compare_local_c_output[64];
static uint32_t ifn_ar_compare_local_generated;

int ifn_ar_test_start(const char *directory, uint32_t mode, uint32_t kv, uint32_t graphs,
                      uint32_t count, const char *text) {
    char path[1200], rendered[8192];
    size_t bytes = 0, n = 0;
    int32_t encoded[4096];
    ifn_tokenizer *t = NULL;
    ifn_tokenizer_scratch *scratch = NULL;
    if (strlen(directory) > 1000 || strlen(text) > 4096 || !count || count > 131072)
        return 1;
    (void)snprintf(path, sizeof path, "%s/tokenizer.json", directory);
    ifn_status rc = ifn_tokenizer_open(path, NULL, &t);
    if (rc == IFN_OK)
        rc = ifn_tokenizer_scratch_create(NULL, &scratch);
    if (rc == IFN_OK)
        rc = ifn_render_user_prompt(text, strlen(text), IFN_THINK_NONE, rendered, sizeof rendered,
                                    &bytes);
    if (rc == IFN_OK)
        rc = ifn_tokenizer_encode(t, rendered, bytes, 1, scratch, encoded, 4096, &n);
    ifn_tokenizer_scratch_destroy(scratch);
    ifn_tokenizer_close(t);
    if (rc != IFN_OK || !n)
        return 1;
    ifn_ar_compare_local_token_count = count;
    ifn_ar_compare_local_tokens =
        malloc(ifn_ar_compare_local_token_count * sizeof *ifn_ar_compare_local_tokens);
    if (!ifn_ar_compare_local_tokens)
        return 1;
    for (size_t i = 0; i < ifn_ar_compare_local_token_count; i++)
        ifn_ar_compare_local_tokens[i] = encoded[i % n];
    ifn_inkling_checkpoint *cp = NULL;
    (void)snprintf(path, sizeof path, "%s/model-00001-of-00001.safetensors", directory);
    char mtp_path[1200];
    (void)snprintf(mtp_path, sizeof mtp_path, "%s/mtp.safetensors", directory);
    rc = ifn_inkling_checkpoint_open(path, ifn_ar_compare_local_test_mtp ? mtp_path : NULL, NULL,
                                     &cp);
    ifn_weight_config wc = IFN_WEIGHT_CONFIG_INIT;
    wc.mode = mode;
    wc.with_mtp = (uint32_t)ifn_ar_compare_local_test_mtp;
    if (rc == IFN_OK)
        rc = ifn_gpu_weights_create(&wc, NULL, &ifn_ar_compare_local_weights);
    if (rc == IFN_OK)
        rc = ifn_gpu_weights_load(ifn_ar_compare_local_weights, cp);
    ifn_inkling_checkpoint_close(cp);
    ifn_session_config config = IFN_SESSION_CONFIG_INIT;
    config.context_tokens = count + ifn_ar_compare_local_test_context_extra < 64
                                ? 64
                                : count + ifn_ar_compare_local_test_context_extra;
    config.kv_format = kv;
    config.use_graphs = graphs;
    if (rc == IFN_OK)
        rc = ifn_session_create(ifn_ar_compare_local_weights, &config, NULL,
                                &ifn_ar_compare_local_session);
    if (rc != IFN_OK) {
        fprintf(stderr, "C AR initialize: %s\n", ifn_last_error());
        return 1;
    }
    if (ifn_ar_test_context(ifn_ar_compare_local_session))
        return 1;
    ifn_gpu_weights *copy = ifn_ar_compare_local_weights;
    if (ifn_gpu_weights_destroy(&copy) != IFN_BUSY || copy != ifn_ar_compare_local_weights)
        return 1;
    ifn_session *second = NULL;
    if (ifn_session_create(ifn_ar_compare_local_weights, &config, NULL, &second) != IFN_BUSY ||
        second)
        return 1;
    return 0;
}

const int32_t *ifn_ar_test_tokens(void) {
    return ifn_ar_compare_local_tokens;
}

int ifn_ar_test_logits(size_t offset, const float *reference, size_t n) {
    if (!n || n > 8 || offset + n > ifn_ar_compare_local_token_count)
        return 1;
    if (ifn_session_logits(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens + offset, n,
                           offset == 0, ifn_ar_compare_local_logits, 200058) != IFN_OK) {
        fprintf(stderr, "C logits: %s\n", ifn_last_error());
        return 1;
    }
    reference += (n - 1) * 200058;
    for (size_t i = 0; i < 200058; i++) {
        if (!isfinite(ifn_ar_compare_local_logits[i]) ||
            memcmp(ifn_ar_compare_local_logits + i, reference + i, 4)) {
            fprintf(stderr, "logits mismatch pos=%zu vocab=%zu C=%.9g Zig=%.9g\n", offset + n - 1,
                    i, ifn_ar_compare_local_logits[i], reference[i]);
            return 1;
        }
    }
    ifn_ar_compare_local_checked_logits += 200058;
    return 0;
}

int ifn_ar_test_generate(const int32_t *expected, size_t n) {
    ifn_generate_stats stats;
    ifn_status rc = ifn_session_generate(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                                         ifn_ar_compare_local_token_count, 32,
                                         ifn_ar_compare_local_c_output, 64, NULL, NULL, &stats);
    if (rc != IFN_OK || stats.generated_tokens != n ||
        memcmp(ifn_ar_compare_local_c_output, expected, n * sizeof *expected)) {
        fprintf(stderr, "C/Zig generation mismatch: status=%u C=%u Zig=%zu %s\n", rc,
                stats.generated_tokens, n, ifn_last_error());
        return 1;
    }
    ifn_ar_compare_local_generated = stats.generated_tokens;
    return 0;
}

static int ifn_ar_compare_local_stop_after_one(void *ctx, int32_t token) {
    (void)ctx;
    (void)token;
    return 0;
}

static int ifn_ar_compare_local_cancel_after_one(void *ctx, int32_t token) {
    (void)token;
    return ifn_session_cancel(ctx) == IFN_OK;
}

int ifn_ar_test_reuse(void) {
    ifn_generate_stats stats;
    int32_t out[64];
    int32_t bad = -1;
    if (ifn_session_generate(ifn_ar_compare_local_session, &bad, 1, 2, out, 64, NULL, NULL,
                             &stats) != IFN_INVALID_ARGUMENT)
        return 1;
    if (ifn_session_generate(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                             ifn_ar_compare_local_token_count, 2, out, 64,
                             ifn_ar_compare_local_stop_after_one, NULL, &stats) != IFN_OK ||
        stats.finish_reason != IFN_FINISH_CALLBACK || stats.generated_tokens != 1)
        return 1;
    if (ifn_session_generate(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                             ifn_ar_compare_local_token_count, 32, out, 64,
                             ifn_ar_compare_local_cancel_after_one, ifn_ar_compare_local_session,
                             &stats) != IFN_CANCELLED ||
        stats.finish_reason != IFN_FINISH_CANCELLED || stats.generated_tokens != 1)
        return 1;
    const int32_t alternate[3] = {100, 200, 300};
    if (ifn_session_generate(ifn_ar_compare_local_session, alternate, 3, 16, out, 64, NULL, NULL,
                             &stats) != IFN_OK)
        return 1;
    if (ifn_session_reset(ifn_ar_compare_local_session) != IFN_OK)
        return 1;
    if (ifn_session_generate(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                             ifn_ar_compare_local_token_count, 32, out, 64, NULL, NULL,
                             &stats) != IFN_OK ||
        stats.generated_tokens != ifn_ar_compare_local_generated ||
        memcmp(out, ifn_ar_compare_local_c_output, ifn_ar_compare_local_generated * sizeof *out))
        return 1;
    return 0;
}

static double ifn_ar_compare_local_now(void) {
    struct timespec t;
    (void)clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void *ifn_ar_compare_local_cancel_thread(void *context) {
    double *requested = context;
    struct timespec delay = {0, 50000000};
    (void)nanosleep(&delay, NULL);
    *requested = ifn_ar_compare_local_now();
    (void)ifn_session_cancel(ifn_ar_compare_local_session);
    return NULL;
}

int ifn_ar_test_cancel_prefill(void) {
    pthread_t thread;
    double requested = 0;
    if (pthread_create(&thread, NULL, ifn_ar_compare_local_cancel_thread, &requested))
        return 1;
    int32_t output[64];
    ifn_generate_stats stats;
    ifn_status rc =
        ifn_session_generate(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                             ifn_ar_compare_local_token_count, 32, output, 64, NULL, NULL, &stats);
    double finished = ifn_ar_compare_local_now();
    if (pthread_join(thread, NULL) || rc != IFN_CANCELLED ||
        stats.finish_reason != IFN_FINISH_CANCELLED)
        return 1;
    ifn_ar_compare_local_cancel_seconds = finished - requested;
    rc = ifn_session_generate(ifn_ar_compare_local_session, ifn_ar_compare_local_tokens,
                              ifn_ar_compare_local_token_count, 32, output, 64, NULL, NULL, &stats);
    return rc != IFN_OK || stats.generated_tokens != ifn_ar_compare_local_generated ||
           memcmp(output, ifn_ar_compare_local_c_output,
                  ifn_ar_compare_local_generated * sizeof *output);
}

int ifn_ar_test_finish(uint32_t success) {
    int failed = ifn_session_destroy(&ifn_ar_compare_local_session) != IFN_OK;
    if (ifn_gpu_weights_destroy(&ifn_ar_compare_local_weights) != IFN_OK)
        failed = 1;
    free(ifn_ar_compare_local_tokens);
    ifn_ar_compare_local_tokens = NULL;
    if (!failed && success)
        printf("{\"c_ar_parity\":true,\"prompt_tokens\":%zu,\"logits_compared\":%zu,\"generated_"
               "tokens\":%u,\"cancel_return_seconds\":%.6f,\"caller_context_preserved\":true}\n",
               ifn_ar_compare_local_token_count, ifn_ar_compare_local_checked_logits,
               ifn_ar_compare_local_generated, ifn_ar_compare_local_cancel_seconds);
    return failed;
}
