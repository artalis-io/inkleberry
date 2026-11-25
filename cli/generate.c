/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sq_engine.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int print_token(void *context, int32_t token) {
    const void *bytes = NULL;
    size_t n = 0;
    if (sq_tokenizer_decode(context, token, 1, &bytes, &n) != SQ_OK)
        return 0;
    return !n || fwrite(bytes, 1, n, stdout) == n;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 7) {
        fputs("usage: squid-c-generate MODEL_DIR PROMPT [MAX_NEW=32] [exact|fp8|nvfp4] "
              "[bf16|fp8|q4] [eager|graph]\n",
              stderr);
        return 2;
    }
    size_t max_new = 32;
    if (argc > 3) {
        char *end = NULL;
        errno = 0;
        unsigned long x = strtoul(argv[3], &end, 10);
        if (errno || !end || *end || !x || x > 4096)
            return 2;
        max_new = x;
    }
    sq_weight_config wc = SQ_WEIGHT_CONFIG_INIT;
    wc.with_mtp = 0;
    if (argc > 4) {
        wc.mode = !strcmp(argv[4], "exact")   ? 0
                  : !strcmp(argv[4], "fp8")   ? 1
                  : !strcmp(argv[4], "nvfp4") ? 2
                                              : 99;
        if (wc.mode > 2)
            return 2;
    }
    sq_session_config sc = SQ_SESSION_CONFIG_INIT;
    if (argc > 5) {
        sc.kv_format = !strcmp(argv[5], "bf16")  ? 0
                       : !strcmp(argv[5], "fp8") ? 1
                       : !strcmp(argv[5], "q4")  ? 2
                                                 : 99;
        if (sc.kv_format > 2)
            return 2;
    }
    if (argc > 6) {
        if (strcmp(argv[6], "eager") && strcmp(argv[6], "graph"))
            return 2;
        sc.use_graphs = !strcmp(argv[6], "graph");
    }
    if (strlen(argv[1]) > 1000 || strlen(argv[2]) > 1048576)
        return 2;
    char path[1200];
    sq_inkling_checkpoint *cp = NULL;
    sq_gpu_weights *w = NULL;
    sq_session *s = NULL;
    sq_tokenizer *t = NULL;
    sq_tokenizer_scratch *scratch = NULL;
    sq_status rc = SQ_OK;
    char *prompt = NULL;
    int32_t *ids = NULL, *out = NULL;
    size_t prompt_n = 0, n = 0;
    (void)snprintf(path, sizeof path, "%s/tokenizer.json", argv[1]);
    rc = sq_tokenizer_open(path, NULL, &t);
    if (rc != SQ_OK)
        goto cleanup;
    rc = sq_tokenizer_scratch_create(NULL, &scratch);
    if (rc != SQ_OK)
        goto cleanup;
    rc = sq_render_user_prompt(argv[2], strlen(argv[2]), SQ_THINK_NONE, NULL, 0, &prompt_n);
    if (rc != SQ_BUFFER_TOO_SMALL)
        goto cleanup;
    prompt = malloc(prompt_n);
    ids = malloc(sc.context_tokens * sizeof *ids);
    out = malloc(max_new * sizeof *out);
    if (!prompt || !ids || !out) {
        rc = SQ_OUT_OF_MEMORY;
        goto cleanup;
    }
    rc =
        sq_render_user_prompt(argv[2], strlen(argv[2]), SQ_THINK_NONE, prompt, prompt_n, &prompt_n);
    if (rc != SQ_OK)
        goto cleanup;
    rc = sq_tokenizer_encode(t, prompt, prompt_n, 1, scratch, ids, sc.context_tokens, &n);
    if (rc != SQ_OK)
        goto cleanup;
    (void)snprintf(path, sizeof path, "%s/model-00001-of-00001.safetensors", argv[1]);
    rc = sq_inkling_checkpoint_open(path, NULL, NULL, &cp);
    if (rc != SQ_OK)
        goto cleanup;
    rc = sq_gpu_weights_create(&wc, NULL, &w);
    if (rc != SQ_OK)
        goto cleanup;
    rc = sq_gpu_weights_load(w, cp);
    if (rc != SQ_OK)
        goto cleanup;
    sq_inkling_checkpoint_close(cp);
    cp = NULL;
    rc = sq_session_create(w, &sc, NULL, &s);
    if (rc != SQ_OK)
        goto cleanup;
    sq_generate_stats stats;
    rc = sq_session_generate(s, ids, n, max_new, out, max_new, print_token, t, &stats);
    if (rc == SQ_OK)
        fprintf(stderr, "\nprompt=%u generated=%u finish=%u\n", stats.prompt_tokens,
                stats.generated_tokens, stats.finish_reason);
cleanup:
    if (rc != SQ_OK)
        fprintf(stderr, "C AR: %s (%s)\n", sq_status_string(rc), sq_last_error());
    if (sq_session_destroy(&s) != SQ_OK)
        rc = SQ_GPU_ERROR;
    if (sq_gpu_weights_destroy(&w) != SQ_OK)
        rc = SQ_GPU_ERROR;
    sq_inkling_checkpoint_close(cp);
    sq_tokenizer_scratch_destroy(scratch);
    sq_tokenizer_close(t);
    free(out);
    free(ids);
    free(prompt);
    return rc == SQ_OK ? 0 : 1;
}
