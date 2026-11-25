/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sq_engine.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int unsigned_value(const char *s, uint64_t *out) {
    if (!*s)
        return 0;
    for (const char *p = s; *p; p++)
        if (*p < '0' || *p > '9')
            return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(s, &end, 10);
    if (errno || !end || *end)
        return 0;
    *out = (uint64_t)value;
    return 1;
}

static int print_token(void *context, int32_t token) {
    const void *bytes = NULL;
    size_t n = 0;
    if (sq_tokenizer_decode(context, token, 1, &bytes, &n) != SQ_OK)
        return 0;
    return !n || fwrite(bytes, 1, n, stdout) == n;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fputs("usage: squid-c-generate MODEL_DIR PROMPT [MAX_NEW] [exact|fp8|nvfp4] [bf16|fp8|q4] "
              "[eager|graph] "
              "[--temperature T] [--top-p P] [--top-k K] [--seed U64] [--drafts 0..7] "
              "[--adaptive-threshold T] [--prompt-lookup 0|1] [--prefix-cache 0|1] "
              "[--system-prefix N]\n",
              stderr);
        return 2;
    }
    size_t max_new = 32;
    sq_weight_config wc = SQ_WEIGHT_CONFIG_INIT;
    wc.with_mtp = 0;
    sq_session_config sc = SQ_SESSION_CONFIG_INIT;
    sq_decode_options options = SQ_DECODE_OPTIONS_INIT;
    uint32_t prefix_cache = 0;
    int index = 3, position = 0;
    while (index < argc && strncmp(argv[index], "--", 2)) {
        const char *value = argv[index++];
        if (position == 0) {
            uint64_t x;
            if (!unsigned_value(value, &x) || !x || x > 4096)
                return 2;
            max_new = (size_t)x;
        } else if (position == 1) {
            wc.mode = !strcmp(value, "exact")   ? 0
                      : !strcmp(value, "fp8")   ? 1
                      : !strcmp(value, "nvfp4") ? 2
                                                : 99;
            if (wc.mode > 2)
                return 2;
        } else if (position == 2) {
            sc.kv_format = !strcmp(value, "bf16")  ? 0
                           : !strcmp(value, "fp8") ? 1
                           : !strcmp(value, "q4")  ? 2
                                                   : 99;
            if (sc.kv_format > 2)
                return 2;
        } else if (position == 3) {
            if (strcmp(value, "eager") && strcmp(value, "graph"))
                return 2;
            sc.use_graphs = !strcmp(value, "graph");
        } else
            return 2;
        position++;
    }
    while (index < argc) {
        if (index + 1 >= argc)
            return 2;
        const char *name = argv[index++], *value = argv[index++];
        uint64_t x;
        if (!strcmp(name, "--temperature") || !strcmp(name, "--top-p") ||
            !strcmp(name, "--adaptive-threshold")) {
            char *end = NULL;
            errno = 0;
            float f = strtof(value, &end);
            if (errno || end == value || *end || !isfinite(f))
                return 2;
            if (!strcmp(name, "--temperature")) {
                if (f < 0)
                    return 2;
                options.temperature = f;
            } else {
                if (f <= 0 || f > 1)
                    return 2;
                if (!strcmp(name, "--adaptive-threshold")) {
                    options.adaptive = 1;
                    options.spec_threshold = f;
                } else
                    options.top_p = f;
            }
        } else {
            if (!unsigned_value(value, &x))
                return 2;
            if (!strcmp(name, "--prefix-cache")) {
                if (x > 1)
                    return 2;
                prefix_cache = (uint32_t)x;
            } else if (!strcmp(name, "--system-prefix")) {
                if (x > 1048576)
                    return 2;
                options.system_prefix_tokens = (uint32_t)x;
            } else if (!strcmp(name, "--top-k")) {
                if (x > 200058)
                    return 2;
                options.top_k = (int32_t)x;
            } else if (!strcmp(name, "--seed"))
                options.seed = x;
            else if (!strcmp(name, "--prompt-lookup")) {
                if (x > 1)
                    return 2;
                options.prompt_lookup = (uint32_t)x;
            } else if (!strcmp(name, "--drafts")) {
                if (x > 7)
                    return 2;
                options.draft_tokens = (uint32_t)x;
            } else
                return 2;
        }
    }
    if ((options.adaptive && (options.draft_tokens != 7 || !sc.use_graphs)) ||
        (options.prompt_lookup && !options.adaptive))
        return 2;
    wc.with_mtp = options.draft_tokens != 0;
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
    char mtp_path[1200];
    (void)snprintf(mtp_path, sizeof mtp_path, "%s/mtp.safetensors", argv[1]);
    rc = sq_inkling_checkpoint_open(path, wc.with_mtp ? mtp_path : NULL, NULL, &cp);
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
    if (prefix_cache) {
        rc = sq_session_prefix_cache(s, 1);
        if (rc != SQ_OK)
            goto cleanup;
    }
    sq_decode_stats stats;
    rc = sq_session_generate_ex(s, ids, n, max_new, &options, out, max_new, print_token, t, &stats);
    if (rc == SQ_OK)
        fprintf(stderr, "\nprompt=%u generated=%u finish=%u\n", stats.generation.prompt_tokens,
                stats.generation.generated_tokens, stats.generation.finish_reason);
    if (rc == SQ_OK && prefix_cache) {
        sq_prefix_info info = SQ_PREFIX_INFO_INIT;
        rc = sq_session_prefix_info(s, &info);
        if (rc == SQ_OK)
            fprintf(stderr, "cached=%u shielded=%u host_bytes=%llu\n", info.reused_tokens,
                    info.shielded_tokens, (unsigned long long)info.host_bytes);
    }
    if (rc == SQ_OK && options.draft_tokens)
        fprintf(stderr, "spec_steps=%u drafts=%u accepted=%u\n", stats.speculative_steps,
                stats.total_drafts, stats.accepted_drafts);
    if (rc == SQ_OK && options.adaptive) {
        sq_decode_info info = SQ_DECODE_INFO_INIT;
        rc = sq_session_decode_info(s, &info);
        if (rc == SQ_OK)
            fprintf(stderr, "lookup=%u widths2/4/8=%u/%u/%u drafts_launched=%u\n",
                    info.lookup_steps, info.width2, info.width4, info.width8, info.draft_launches);
    }
cleanup:
    if (rc != SQ_OK)
        fprintf(stderr, "C decode: %s (%s)\n", sq_status_string(rc), sq_last_error());
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
