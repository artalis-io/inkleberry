/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Diagnostic only: preserve generation; replay LM head into separate buffers. */
#include "engine_internal.h"
#include "benchmark_prompts.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define IFN_BOUNDARY_TRY(x)                                                                        \
    do {                                                                                           \
        if ((x)) {                                                                                 \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

typedef struct {
    ifn_session *session;
    unsigned target, emitted, row, width, last_lookup, prompt_tokens;
    int32_t last_position;
    int32_t prefix[96];
    const char *directory;
    unsigned which, profile;
    int enabled, captured, failed;
    void *buffers[4];
    float logits[200058], norm[8 * 2304];
} ifn_boundary_context;

static int ifn_boundary_file(const char *directory, unsigned which, unsigned profile,
                             const char *suffix, const void *data, size_t size) {
    char path[2048];
    int n =
        snprintf(path, sizeof path, "%s/case-%u-profile-%u.%s", directory, which, profile, suffix);
    if (n < 0 || (size_t)n >= sizeof path)
        return 1;
    FILE *f = fopen(path, "wbx");
    if (!f)
        return 1;
    int failed = fwrite(data, 1, size, f) != size;
    return fclose(f) || failed;
}

static int ifn_boundary_token(void *pointer, int32_t token) {
    ifn_boundary_context *c = pointer;
    ifn_session *s = c->session;
    if (!c->emitted) {
        c->width = c->prompt_tokens % 8;
        if (!c->width)
            c->width = 8;
        c->row = c->width - 1;
    } else if (s->position != c->last_position) {
        c->row = 0;
        if (!c->profile || c->profile == 3)
            c->width = 1;
        else if (c->profile == 1 || s->decode_info.lookup_steps != c->last_lookup)
            c->width = 8;
        else
            c->width = (unsigned)*s->h_width;
    } else
        c->row++;
    c->last_position = s->position;
    c->last_lookup = (unsigned)s->decode_info.lookup_steps;
    if (c->emitted < c->target && token != c->prefix[c->emitted]) {
        c->failed = 1;
        return 0;
    }
    if (c->enabled && c->emitted == c->target) {
        if (c->width < 1 || c->width > 8 || c->row >= c->width || ifn_cuda_stream_sync(s->stream) ||
            ifn_cuda_memcpy_d2h(c->norm, s->mlp_normed, (size_t)c->width * 2304 * sizeof(float))) {
            c->failed = 1;
            return 0;
        }
        /* NULL sample config requests full post-softcap logits. All destinations
         * are separate, and original generation/verification buffers stay intact. */
        ifn_launch_lm_head_argmax(&s->model.unembed, s->mlp_normed, c->buffers[0], c->buffers[1],
                                  c->buffers[2], c->buffers[3], NULL, s->d_pos, NULL, NULL, NULL,
                                  NULL, NULL, 0, (int32_t)c->width, s->stream);
        if (ifn_cuda_launch_status() || ifn_cuda_stream_sync(s->stream) ||
            ifn_cuda_memcpy_d2h(c->logits, (const float *)c->buffers[1] + (size_t)c->row * 200058,
                                sizeof c->logits)) {
            c->failed = 1;
            return 0;
        }
        for (unsigned i = 0; i < 200058; i++)
            if (!isfinite(c->logits[i])) {
                c->failed = 1;
                return 0;
            }
        if (ifn_boundary_file(c->directory, c->which, c->profile, "logits.f32", c->logits,
                              sizeof c->logits) ||
            ifn_boundary_file(c->directory, c->which, c->profile, "norm.f32", c->norm,
                              (size_t)c->width * 2304 * sizeof(float))) {
            c->failed = 1;
            return 0;
        }
        printf("{\"kind\":\"capture\",\"case\":%u,\"profile\":%u,\"output_index\":%u,"
               "\"width\":%u,\"row\":%u,\"position\":%d,\"selected_token\":%d}\n",
               c->which, c->profile, c->target, c->width, c->row, s->position, token);
        fflush(stdout);
        c->captured = 1;
    }
    c->emitted++;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4 || strlen(argv[1]) > 1500 || strlen(argv[3]) > 1500)
        return 2;
    int failed = 0;
    ifn_tokenizer *tokenizer = NULL;
    ifn_tokenizer_scratch *scratch = NULL;
    ifn_inkling_checkpoint *checkpoint = NULL;
    ifn_gpu_weights *weights = NULL;
    ifn_session *session = NULL;
    ifn_boundary_context *c = calloc(1, sizeof *c);
    FILE *requests = NULL;
    char path[1800], mtp[1800], text[100001], rendered[110000];
    int32_t ids[32768], outputs[2][96];
    unsigned count = 0;
    IFN_BOUNDARY_TRY(!c);
    snprintf(path, sizeof path, "%s/tokenizer.json", argv[1]);
    IFN_BOUNDARY_TRY(ifn_tokenizer_open(path, NULL, &tokenizer));
    IFN_BOUNDARY_TRY(ifn_tokenizer_scratch_create(NULL, &scratch));
    snprintf(path, sizeof path, "%s/model-00001-of-00001.safetensors", argv[1]);
    snprintf(mtp, sizeof mtp, "%s/mtp.safetensors", argv[1]);
    IFN_BOUNDARY_TRY(ifn_inkling_checkpoint_open(path, mtp, NULL, &checkpoint));
    ifn_weight_config wc = IFN_WEIGHT_CONFIG_INIT;
    wc.mode = 2;
    wc.with_mtp = 1;
    IFN_BOUNDARY_TRY(ifn_gpu_weights_create(&wc, NULL, &weights));
    IFN_BOUNDARY_TRY(ifn_gpu_weights_load(weights, checkpoint));
    ifn_inkling_checkpoint_close(checkpoint);
    checkpoint = NULL;
    ifn_session_config sc = IFN_SESSION_CONFIG_INIT;
    sc.context_tokens = 32768;
    sc.use_graphs = 1;
    sc.kv_format = 1;
    IFN_BOUNDARY_TRY(ifn_session_create(weights, &sc, NULL, &session));
    const size_t sizes[] = {8 * 4, 8 * 200058 * 4, 8 * 1024 * 4, 8 * 1024 * 4};
    for (unsigned i = 0; i < 4; i++)
        IFN_BOUNDARY_TRY(ifn_cuda_malloc(&c->buffers[i], sizes[i]));
    requests = fopen(argv[2], "r");
    IFN_BOUNDARY_TRY(!requests);
    unsigned expected_prompt;
    while (fscanf(requests, "%u %u %u %u", &c->which, &c->profile, &c->target, &expected_prompt) ==
           4) {
        IFN_BOUNDARY_TRY(c->which < 4 || c->which > 6 || c->profile > 3 || c->target >= 96);
        for (unsigned i = 0; i < c->target; i++) {
            IFN_BOUNDARY_TRY(fscanf(requests, "%d", &c->prefix[i]) != 1);
            IFN_BOUNDARY_TRY(c->prefix[i] < 0 || c->prefix[i] >= 200058);
        }
        IFN_BOUNDARY_TRY(ifn_benchmark_prompt(c->which, text, sizeof text));
        size_t bytes = 0, tokens = 0;
        IFN_BOUNDARY_TRY(ifn_render_user_prompt(text, strlen(text), IFN_THINK_NONE, rendered,
                                                sizeof rendered, &bytes));
        IFN_BOUNDARY_TRY(
            ifn_tokenizer_encode(tokenizer, rendered, bytes, 1, scratch, ids, 32768, &tokens));
        IFN_BOUNDARY_TRY(tokens != expected_prompt);
        IFN_BOUNDARY_TRY(
            ifn_boundary_file(argv[3], c->which, c->profile, "prompt.i32", ids, tokens * 4));
        c->prompt_tokens = (unsigned)tokens;
        c->session = session;
        c->directory = argv[3];
        ifn_decode_options options = IFN_DECODE_OPTIONS_INIT;
        options.seed = 42;
        options.temperature = c->profile == 3 ? 0.7f : 0;
        options.draft_tokens = c->profile == 1 || c->profile == 2 ? 7 : 0;
        options.adaptive = options.prompt_lookup = c->profile == 2;
        unsigned generated[2];
        for (unsigned run = 0; run < 2; run++) {
            IFN_BOUNDARY_TRY(ifn_session_reset(session));
            c->enabled = (int)run;
            c->emitted = c->captured = c->failed = 0;
            c->row = c->width = c->last_lookup = 0;
            c->last_position = -1;
            ifn_decode_stats stats;
            IFN_BOUNDARY_TRY(ifn_session_generate_ex(session, ids, tokens, 96, &options,
                                                     outputs[run], 96, ifn_boundary_token, c,
                                                     &stats));
            IFN_BOUNDARY_TRY(c->failed || (run && !c->captured));
            generated[run] = stats.generation.generated_tokens;
        }
        IFN_BOUNDARY_TRY(generated[0] != generated[1] ||
                         memcmp(outputs[0], outputs[1], generated[0] * sizeof(int32_t)));
        printf("{\"kind\":\"result\",\"case\":%u,\"profile\":%u,\"prompt_tokens\":%zu,"
               "\"capture_preserved_output\":true,\"tokens\":[",
               c->which, c->profile, tokens);
        for (unsigned i = 0; i < generated[0]; i++)
            printf("%s%d", i ? "," : "", outputs[0][i]);
        puts("]}");
        fflush(stdout);
        count++;
    }
    IFN_BOUNDARY_TRY(!feof(requests) || count != 11);
done:
    if (failed)
        fprintf(stderr, "boundary capture failed: %s\n", ifn_last_error());
    if (requests && fclose(requests))
        failed = 1;
    if (c)
        for (unsigned i = 0; i < 4; i++)
            if (c->buffers[i] && ifn_cuda_free(c->buffers[i]))
                failed = 1;
    free(c);
    if (ifn_session_destroy(&session))
        failed = 1;
    if (ifn_gpu_weights_destroy(&weights))
        failed = 1;
    ifn_inkling_checkpoint_close(checkpoint);
    ifn_tokenizer_scratch_destroy(scratch);
    ifn_tokenizer_close(tokenizer);
    return failed;
}
