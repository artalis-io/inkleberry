/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ifn_semantic_output(void *context, int32_t token) {
    const void *bytes = NULL;
    size_t count = 0;
    return ifn_tokenizer_decode(context, token, 0, &bytes, &count) == IFN_OK &&
           (!count || fwrite(bytes, 1, count, stdout) == count);
}

int main(int argc, char **argv) {
    if (argc != 4 || strlen(argv[1]) > 1500 || strlen(argv[3]) != 1 || argv[3][0] < '0' ||
        argv[3][0] > '3')
        return 2;
    unsigned profile = (unsigned)(argv[3][0] - '0');
    FILE *input = fopen(argv[2], "rb");
    if (!input)
        return 2;
    char text[100001], rendered[110000], path[1800], mtp[1800];
    size_t length = fread(text, 1, sizeof text, input);
    int bad = ferror(input) || length == sizeof text || !length;
    if (fclose(input) || bad || memchr(text, 0, length))
        return 2;
    text[length] = 0;
    int32_t ids[32768], output[2048];
    ifn_tokenizer *tokenizer = NULL;
    ifn_tokenizer_scratch *scratch = NULL;
    ifn_inkling_checkpoint *checkpoint = NULL;
    ifn_gpu_weights *weights = NULL;
    ifn_session *session = NULL;
    ifn_status status = IFN_OK;
#define IFN_SEMANTIC_TRY(call)                                                                     \
    do {                                                                                           \
        status = (call);                                                                           \
        if (status)                                                                                \
            goto done;                                                                             \
    } while (0)
    snprintf(path, sizeof path, "%s/tokenizer.json", argv[1]);
    IFN_SEMANTIC_TRY(ifn_tokenizer_open(path, NULL, &tokenizer));
    IFN_SEMANTIC_TRY(ifn_tokenizer_scratch_create(NULL, &scratch));
    size_t bytes = 0, tokens = 0;
    IFN_SEMANTIC_TRY(
        ifn_render_user_prompt(text, length, IFN_THINK_NONE, rendered, sizeof rendered, &bytes));
    IFN_SEMANTIC_TRY(
        ifn_tokenizer_encode(tokenizer, rendered, bytes, 1, scratch, ids, 32768, &tokens));
    if (tokens > 32768 - 2048) {
        status = IFN_LIMIT;
        goto done;
    }
    snprintf(path, sizeof path, "%s/model-00001-of-00001.safetensors", argv[1]);
    snprintf(mtp, sizeof mtp, "%s/mtp.safetensors", argv[1]);
    IFN_SEMANTIC_TRY(ifn_inkling_checkpoint_open(path, mtp, NULL, &checkpoint));
    ifn_weight_config wc = IFN_WEIGHT_CONFIG_INIT;
    wc.mode = 2;
    wc.with_mtp = 1;
    IFN_SEMANTIC_TRY(ifn_gpu_weights_create(&wc, NULL, &weights));
    IFN_SEMANTIC_TRY(ifn_gpu_weights_load(weights, checkpoint));
    ifn_inkling_checkpoint_close(checkpoint);
    checkpoint = NULL;
    ifn_session_config sc = IFN_SESSION_CONFIG_INIT;
    sc.context_tokens = 32768;
    sc.kv_format = 1;
    sc.use_graphs = 1;
    IFN_SEMANTIC_TRY(ifn_session_create(weights, &sc, NULL, &session));
    ifn_decode_options options = IFN_DECODE_OPTIONS_INIT;
    options.seed = 42;
    options.temperature = profile == 3 ? 0.7f : 0;
    options.draft_tokens = profile == 1 || profile == 2 ? 7 : 0;
    options.adaptive = options.prompt_lookup = profile == 2;
    ifn_decode_stats stats;
    IFN_SEMANTIC_TRY(ifn_session_generate_ex(session, ids, tokens, 2048, &options, output, 2048,
                                             ifn_semantic_output, tokenizer, &stats));
    fprintf(stderr,
            "{\"semantic_version\":1,\"profile\":%u,\"prompt_tokens\":%u,"
            "\"generated_tokens\":%u,\"finish_reason\":%u,\"max_output_tokens\":2048}\n",
            profile, stats.generation.prompt_tokens, stats.generation.generated_tokens,
            stats.generation.finish_reason);
    if (fflush(stdout))
        status = IFN_IO;
done:
    if (status)
        fprintf(stderr, "semantic: %s: %s\n", ifn_status_string(status), ifn_last_error());
    if (ifn_session_destroy(&session))
        status = IFN_GPU_ERROR;
    if (ifn_gpu_weights_destroy(&weights))
        status = IFN_GPU_ERROR;
    ifn_inkling_checkpoint_close(checkpoint);
    ifn_tokenizer_scratch_destroy(scratch);
    ifn_tokenizer_close(tokenizer);
    return status ? 1 : 0;
}
