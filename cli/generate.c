/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn_engine.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ifn_generate_local_unsigned_value(const char *s, uint64_t *out) {
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

static int ifn_generate_local_print_token(void *context, int32_t token) {
    const void *bytes = NULL;
    size_t n = 0;
    if (ifn_tokenizer_decode(context, token, 1, &bytes, &n) != IFN_OK)
        return 0;
    return !n || fwrite(bytes, 1, n, stdout) == n;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fputs(
            "usage: inkleberry-generate MODEL_DIR PROMPT [MAX_NEW] [exact|fp8|nvfp4] [bf16|fp8|q4] "
            "[eager|graph] "
            "[--temperature T] [--top-p P] [--top-k K] [--seed U64] [--drafts 0..7] "
            "[--adaptive-threshold T] [--prompt-lookup 0|1] [--prefix-cache 0|1] "
            "[--system-prefix N]\n",
            stderr);
        return 2;
    }
    size_t max_new = 32;
    ifn_weight_config wc = IFN_WEIGHT_CONFIG_INIT;
    wc.with_mtp = 0;
    ifn_session_config sc = IFN_SESSION_CONFIG_INIT;
    ifn_decode_options options = IFN_DECODE_OPTIONS_INIT;
    uint32_t prefix_cache = 0;
    int index = 3, position = 0;
    while (index < argc && strncmp(argv[index], "--", 2)) {
        const char *value = argv[index++];
        if (position == 0) {
            uint64_t x;
            if (!ifn_generate_local_unsigned_value(value, &x) || !x || x > 4096)
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
            if (!ifn_generate_local_unsigned_value(value, &x))
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
    ifn_inkling_checkpoint *cp = NULL;
    ifn_gpu_weights *w = NULL;
    ifn_session *s = NULL;
    ifn_tokenizer *t = NULL;
    ifn_tokenizer_scratch *scratch = NULL;
    ifn_status rc = IFN_OK;
    char *prompt = NULL;
    int32_t *ids = NULL, *out = NULL;
    size_t prompt_n = 0, n = 0;
    (void)snprintf(path, sizeof path, "%s/tokenizer.json", argv[1]);
    rc = ifn_tokenizer_open(path, NULL, &t);
    if (rc != IFN_OK)
        goto cleanup;
    rc = ifn_tokenizer_scratch_create(NULL, &scratch);
    if (rc != IFN_OK)
        goto cleanup;
    rc = ifn_render_user_prompt(argv[2], strlen(argv[2]), IFN_THINK_NONE, NULL, 0, &prompt_n);
    if (rc != IFN_BUFFER_TOO_SMALL)
        goto cleanup;
    prompt = malloc(prompt_n);
    ids = malloc(sc.context_tokens * sizeof *ids);
    out = malloc(max_new * sizeof *out);
    if (!prompt || !ids || !out) {
        rc = IFN_OUT_OF_MEMORY;
        goto cleanup;
    }
    rc = ifn_render_user_prompt(argv[2], strlen(argv[2]), IFN_THINK_NONE, prompt, prompt_n,
                                &prompt_n);
    if (rc != IFN_OK)
        goto cleanup;
    rc = ifn_tokenizer_encode(t, prompt, prompt_n, 1, scratch, ids, sc.context_tokens, &n);
    if (rc != IFN_OK)
        goto cleanup;
    (void)snprintf(path, sizeof path, "%s/model-00001-of-00001.safetensors", argv[1]);
    char mtp_path[1200];
    (void)snprintf(mtp_path, sizeof mtp_path, "%s/mtp.safetensors", argv[1]);
    rc = ifn_inkling_checkpoint_open(path, wc.with_mtp ? mtp_path : NULL, NULL, &cp);
    if (rc != IFN_OK)
        goto cleanup;
    rc = ifn_gpu_weights_create(&wc, NULL, &w);
    if (rc != IFN_OK)
        goto cleanup;
    rc = ifn_gpu_weights_load(w, cp);
    if (rc != IFN_OK)
        goto cleanup;
    ifn_inkling_checkpoint_close(cp);
    cp = NULL;
    rc = ifn_session_create(w, &sc, NULL, &s);
    if (rc != IFN_OK)
        goto cleanup;
    if (prefix_cache) {
        rc = ifn_session_prefix_cache(s, 1);
        if (rc != IFN_OK)
            goto cleanup;
    }
    ifn_decode_stats stats;
    rc = ifn_session_generate_ex(s, ids, n, max_new, &options, out, max_new,
                                 ifn_generate_local_print_token, t, &stats);
    if (rc == IFN_OK)
        fprintf(stderr, "\nprompt=%u generated=%u finish=%u\n", stats.generation.prompt_tokens,
                stats.generation.generated_tokens, stats.generation.finish_reason);
    if (rc == IFN_OK && prefix_cache) {
        ifn_prefix_info info = IFN_PREFIX_INFO_INIT;
        rc = ifn_session_prefix_info(s, &info);
        if (rc == IFN_OK)
            fprintf(stderr, "cached=%u shielded=%u host_bytes=%llu\n", info.reused_tokens,
                    info.shielded_tokens, (unsigned long long)info.host_bytes);
    }
    if (rc == IFN_OK && options.draft_tokens)
        fprintf(stderr, "spec_steps=%u drafts=%u accepted=%u\n", stats.speculative_steps,
                stats.total_drafts, stats.accepted_drafts);
    if (rc == IFN_OK && options.adaptive) {
        ifn_decode_info info = IFN_DECODE_INFO_INIT;
        rc = ifn_session_decode_info(s, &info);
        if (rc == IFN_OK)
            fprintf(stderr, "lookup=%u widths2/4/8=%u/%u/%u drafts_launched=%u\n",
                    info.lookup_steps, info.width2, info.width4, info.width8, info.draft_launches);
    }
cleanup:
    if (rc != IFN_OK)
        fprintf(stderr, "C decode: %s (%s)\n", ifn_status_string(rc), ifn_last_error());
    if (ifn_session_destroy(&s) != IFN_OK)
        rc = IFN_GPU_ERROR;
    if (ifn_gpu_weights_destroy(&w) != IFN_OK)
        rc = IFN_GPU_ERROR;
    ifn_inkling_checkpoint_close(cp);
    ifn_tokenizer_scratch_destroy(scratch);
    ifn_tokenizer_close(t);
    free(out);
    free(ids);
    free(prompt);
    return rc == IFN_OK ? 0 : 1;
}
