/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#define _POSIX_C_SOURCE 200809L
#include "ifn_engine.h"
#include "ifn_infer.h"
#include "ifn_infer_inkling.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "host_clock.h"

static uint64_t ifn_benchmark_now(void) {
    return ifn_host_now_ns();
}

typedef struct {
    uint64_t first;
    int32_t *tokens;
    size_t count;
} ifn_benchmark_clock;

#ifndef IFN_BENCH_FACADE
static int ifn_benchmark_token(void *p, int32_t token) {
    (void)token;
    ifn_benchmark_clock *c = p;
    if (!c->first)
        c->first = ifn_benchmark_now();
    return 1;
}

#else
static uint32_t ifn_benchmark_event(void *p, const ifn_infer_event_v1 *event) {
    ifn_benchmark_clock *clock = p;
    if (event->kind != IFN_EVENT_TOKENS || event->token_count > 128 - clock->count)
        return IFN_CALLBACK_CANCEL;
    if (!clock->first)
        clock->first = ifn_benchmark_now();
    memcpy(clock->tokens + clock->count, event->tokens,
           (size_t)event->token_count * sizeof(int32_t));
    clock->count += (size_t)event->token_count;
    return IFN_CALLBACK_CONTINUE;
}
#endif

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    ifn_tokenizer *t = NULL;
    ifn_tokenizer_scratch *scratch = NULL;
    ifn_inkling_checkpoint *cp = NULL;
    ifn_gpu_weights *weights = NULL;
    ifn_session *session = NULL;
    char path[2048], mtp[2048], text[160000], rendered[170000];
    int32_t ids[32768], out[128];
    ifn_status rc = IFN_OK;
#ifdef IFN_BENCH_FACADE
    (void)cp;
    (void)weights;
    (void)session;
    (void)mtp;
    const void *raw = NULL;
    if (ifn_infer_get_api(1, 0, &raw))
        return 1;
    const ifn_infer_api_v1 *api = raw;
    ifn_infer_model *model = NULL;
    ifn_infer_session *facade = NULL;
    ifn_infer_request *request = NULL;
    ifn_infer_error_v1 error;
    IFN_INFER_INIT(error);
#endif
#define IFN_BENCH_TRY(call)                                                                        \
    do {                                                                                           \
        rc = (call);                                                                               \
        if (rc)                                                                                    \
            goto done;                                                                             \
    } while (0)
    if (strlen(argv[1]) > 1700)
        return 2;
    snprintf(path, sizeof path, "%s/tokenizer.json", argv[1]);
    IFN_BENCH_TRY(ifn_tokenizer_open(path, NULL, &t));
    IFN_BENCH_TRY(ifn_tokenizer_scratch_create(NULL, &scratch));
#ifndef IFN_BENCH_FACADE
    snprintf(path, sizeof path, "%s/model-00001-of-00001.safetensors", argv[1]);
    snprintf(mtp, sizeof mtp, "%s/mtp.safetensors", argv[1]);
    IFN_BENCH_TRY(ifn_inkling_checkpoint_open(path, mtp, NULL, &cp));
    ifn_weight_config wc = IFN_WEIGHT_CONFIG_INIT;
    IFN_BENCH_TRY(ifn_gpu_weights_create(&wc, NULL, &weights));
    IFN_BENCH_TRY(ifn_gpu_weights_load(weights, cp));
    ifn_session_config sc = IFN_SESSION_CONFIG_INIT;
    sc.context_tokens = 32768;
    IFN_BENCH_TRY(ifn_session_create(weights, &sc, NULL, &session));
#else
    ifn_infer_model_config_v1 mc;
    IFN_INFER_INIT(mc);
    ifn_infer_session_config_v1 sc;
    IFN_INFER_INIT(sc);
    ifn_infer_source_v1 source;
    IFN_INFER_INIT(source);
    IFN_BENCH_TRY(api->defaults(IFN_DEFAULT_MODEL, &mc, &error));
    IFN_BENCH_TRY(api->defaults(IFN_DEFAULT_SESSION, &sc, &error));
    mc.weight_profile = (ifn_infer_bytes){"nvfp4-mtp", 9};
    sc.context_tokens = 32768;
    source.directory = (ifn_infer_bytes){argv[1], strlen(argv[1])};
    source.format_id = (ifn_infer_bytes){"inkling-safetensors", sizeof("inkling-safetensors") - 1};
    IFN_BENCH_TRY(api->model_create(&mc, &model, &error));
    IFN_BENCH_TRY(api->model_load(model, &source, &error));
    IFN_BENCH_TRY(api->session_create(model, &sc, &facade, &error));
    IFN_BENCH_TRY(api->session_prepare(facade, &error));
#endif
    static const char *const ifn_benchmark_prompts[] = {
        "What is 9 + 8? Answer with the number only.",
        "Explain briefly why a delivery truck must obey a gross vehicle weight limit.",
        "Write one short paragraph describing a quiet autumn morning.",
        "Write a C function that computes the maximum of an integer array. Return only code.",
        "Fix this Python function to avoid a mutable default argument: def add(x, items=[]): "
        "items.append(x); return items. Return code.",
        "Extract JSON with name and quantity: The customer is Acme and the quantity is 17. Return "
        "only JSON.",
        "What is the access code for record NEEDLE? Copy just the code."};
    static const char *const ifn_benchmark_profiles[] = {"ar", "fixed", "adaptive", "sampled"};
    for (unsigned profile = 0; profile < 4; profile++) {
        for (unsigned c = 0; c < 7; c++) {
            text[0] = 0;
            if (c == 6) {
                const char *pad = "Record cedar contains ordinary inventory and no access code. ";
                for (unsigned i = 0; i < 400; i++)
                    strcat(text, pad);
                strcat(text, "Record NEEDLE has access code COBALT-7319.\n");
                for (unsigned i = 0; i < 400; i++)
                    strcat(text, pad);
            }
            strcat(text, ifn_benchmark_prompts[c]);
            size_t bytes = 0, count = 0;
            IFN_BENCH_TRY(ifn_render_user_prompt(text, strlen(text), IFN_THINK_NONE, rendered,
                                                 sizeof rendered, &bytes));
            IFN_BENCH_TRY(ifn_tokenizer_encode(t, rendered, bytes, 1, scratch, ids, 32768, &count));
            ifn_decode_options options = IFN_DECODE_OPTIONS_INIT;
            options.draft_tokens = profile == 1 || profile == 2 ? 7 : 0;
            options.adaptive = options.prompt_lookup = profile == 2;
            options.temperature = profile == 3 ? 0.7f : 0;
            options.seed = 42;
            for (unsigned run = 0; run < 6; run++) {
#ifndef IFN_BENCH_FACADE
                IFN_BENCH_TRY(ifn_session_reset(session));
#else
                IFN_BENCH_TRY(api->session_clear_cache(facade, &error));
#endif
                ifn_benchmark_clock clock = {0};
                ifn_decode_stats stats = {0};
#ifndef IFN_BENCH_FACADE
                uint64_t start = ifn_benchmark_now();
                IFN_BENCH_TRY(ifn_session_generate_ex(session, ids, count, 96, &options, out, 128,
                                                      ifn_benchmark_token, &clock, &stats));
#else
                clock.tokens = out;
                ifn_infer_request_config_v1 config;
                IFN_INFER_INIT(config);
                ifn_infer_input_v1 input;
                IFN_INFER_INIT(input);
                ifn_infer_inkling_request_v1 parameters;
                IFN_INFER_INIT(parameters);
                ifn_infer_extension_v1 ext;
                IFN_INFER_INIT(ext);
                IFN_BENCH_TRY(api->defaults(IFN_DEFAULT_REQUEST, &config, &error));
                config.output_kind = IFN_OUTPUT_RAW_TOKENS;
                config.max_output_tokens = 96;
                config.temperature = options.temperature;
                config.seed = 42;
                parameters.draft_tokens = options.draft_tokens;
                parameters.adaptive = options.adaptive;
                parameters.prompt_lookup = options.prompt_lookup;
                parameters.spec_threshold = options.spec_threshold;
                ext.id = IFN_INFER_EXT_INKLING_REQUEST;
                ext.version = 1;
                ext.payload = &parameters;
                ext.payload_bytes = sizeof parameters;
                config.extensions = (ifn_infer_array){&ext, 1, sizeof ext, 0};
                input.kind = IFN_INPUT_TOKENS;
                input.tokens = ids;
                input.token_count = count;
                uint64_t start = ifn_benchmark_now();
                IFN_BENCH_TRY(api->request_create(facade, &input, &config, &request, &error));
                IFN_BENCH_TRY(api->request_run(request, ifn_benchmark_event, &clock, &error));
                ifn_infer_result_v1 result;
                IFN_INFER_INIT(result);
                IFN_BENCH_TRY(api->request_result(request, &result, &error));
                stats.generation.generated_tokens = (uint32_t)result.generated_tokens;
                IFN_BENCH_TRY(api->request_close(&request, &error));
#endif
                uint64_t end = ifn_benchmark_now();
                printf("{\"case\":%u,\"profile\":\"%s\",\"run\":%u,\"warmup\":%s,\"prompt_tokens\":"
                       "%zu,\"elapsed_ns\":%llu,\"first_token_ns\":%llu,\"generated\":%u,"
                       "\"drafts\":%u,\"accepted\":%u,\"tokens\":[",
                       c, ifn_benchmark_profiles[profile], run, run ? "false" : "true", count,
                       (unsigned long long)(end - start),
                       (unsigned long long)(clock.first ? clock.first - start : 0),
                       stats.generation.generated_tokens, stats.total_drafts,
                       stats.accepted_drafts);
                for (uint32_t i = 0; i < stats.generation.generated_tokens; i++)
                    printf("%s%d", i ? "," : "", out[i]);
                puts("]}");
                fflush(stdout);
            }
        }
    }
done:
    if (rc)
        fprintf(stderr, "benchmark: %s: %s\n", ifn_status_string(rc), ifn_last_error());
#ifndef IFN_BENCH_FACADE
    if (ifn_session_destroy(&session))
        rc = IFN_GPU_ERROR;
    if (ifn_gpu_weights_destroy(&weights))
        rc = IFN_GPU_ERROR;
    ifn_inkling_checkpoint_close(cp);
#else
    if (rc)
        fprintf(stderr, "facade: %s %.*s\n", ifn_status_string(rc), (int)error.message_bytes,
                error.message);
    if (api->request_close(&request, NULL))
        rc = IFN_GPU_ERROR;
    if (api->session_close(&facade, NULL))
        rc = IFN_GPU_ERROR;
    if (api->model_close(&model, NULL))
        rc = IFN_GPU_ERROR;
#endif
    ifn_tokenizer_scratch_destroy(scratch);
    ifn_tokenizer_close(t);
    return rc ? 1 : 0;
}
