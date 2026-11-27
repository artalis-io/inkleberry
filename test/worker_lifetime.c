/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Linux qualification client. Only installed public ABI headers; no engine/CUDA APIs. */
#define _POSIX_C_SOURCE 200809L
#include <ifn_infer.h>
#include <ifn_infer_inkling.h>
#include "platform.h"
#include <inttypes.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IFN_WORKER_TOKEN_LIMIT 128
#define IFN_WORKER_REQUIRE(x)                                                                      \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "worker-lifetime:%d: %s\n", __LINE__, #x);                             \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

typedef struct {
    int32_t tokens[IFN_WORKER_TOKEN_LIMIT];
    size_t count;
    uint64_t first_ns, last_ns;
    int cancel;
} ifn_worker_output;

static uint64_t ifn_worker_now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t))
        abort();
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}

static uint32_t ifn_worker_collect(void *context, const ifn_infer_event_v1 *event) {
    ifn_worker_output *o = context;
    if (event->kind != IFN_EVENT_TOKENS)
        return IFN_CALLBACK_CONTINUE;
    if (event->token_count > IFN_WORKER_TOKEN_LIMIT - o->count)
        return IFN_CALLBACK_CANCEL;
    uint64_t now = ifn_worker_now();
    if (!o->count)
        o->first_ns = now;
    o->last_ns = now;
    memcpy(o->tokens + o->count, event->tokens, (size_t)event->token_count * sizeof(int32_t));
    o->count += (size_t)event->token_count;
    return o->cancel ? IFN_CALLBACK_CANCEL : IFN_CALLBACK_CONTINUE;
}

static uint64_t ifn_worker_rss(void) {
    char line[256];
    uint64_t kib = 0;
    FILE *f = fopen("/proc/self/status", "r");
    if (!f)
        return 0; /* unavailable explicitly represented as zero */
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "VmRSS: %" SCNu64 " kB", &kib) == 1)
            break;
    fclose(f);
    return kib * 1024;
}

typedef struct {
    const ifn_infer_api_v1 *api;
    ifn_infer_request *request;
    ifn_worker_output *output;
    atomic_bool finished;
    ifn_status status;
} ifn_worker_task;

static void *ifn_worker_run(void *context) {
    ifn_worker_task *t = context;
    t->status = t->api->request_run(t->request, ifn_worker_collect, t->output, NULL);
    atomic_store(&t->finished, true);
    return NULL;
}

static uint32_t ifn_worker_random(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static int ifn_worker_number(const char *text, uint64_t maximum, uint64_t *value) {
    char *end;
    if (!*text || *text == '-')
        return 0;
    *value = strtoull(text, &end, 10);
    return !*end && *value > 0 && *value <= maximum;
}

int main(int argc, char **argv) {
    /* MODEL SEED CYCLES: seed and cycle count replay the complete ordered history. */
    uint64_t seed, cycles, seconds = 0, completed = 0;
    if ((argc != 4 && argc != 5) || !ifn_worker_number(argv[2], UINT32_MAX, &seed) ||
        !ifn_worker_number(argv[3], 10000000, &cycles) ||
        (argc == 5 && !ifn_worker_number(argv[4], 86400, &seconds))) {
        fputs("usage: worker-lifetime MODEL NONZERO_SEED CYCLES [SECONDS]\n", stderr);
        return 2;
    }
    int failed = 0;
    uint32_t random = (uint32_t)seed;
    const void *table = NULL;
    const ifn_infer_api_v1 *api = NULL;
    ifn_infer_model *model = NULL;
    ifn_infer_session *session = NULL;
    ifn_infer_request *request = NULL;
    ifn_worker_output baseline[2] = {{0}};
    ifn_infer_model_config_v1 mc;
    ifn_infer_session_config_v1 sc;
    ifn_infer_request_config_v1 rc;
    ifn_infer_source_v1 source;
    ifn_infer_input_v1 input;
    IFN_INFER_INIT(mc);
    IFN_INFER_INIT(sc);
    IFN_INFER_INIT(rc);
    IFN_INFER_INIT(source);
    IFN_INFER_INIT(input);
    IFN_WORKER_REQUIRE(ifn_infer_get_api(1, 0, &table) == IFN_OK);
    api = table;
    IFN_WORKER_REQUIRE(api->defaults(IFN_DEFAULT_MODEL, &mc, NULL) == IFN_OK);
    IFN_WORKER_REQUIRE(api->defaults(IFN_DEFAULT_SESSION, &sc, NULL) == IFN_OK);
    IFN_WORKER_REQUIRE(api->defaults(IFN_DEFAULT_REQUEST, &rc, NULL) == IFN_OK);
    mc.weight_profile = (ifn_infer_bytes){"nvfp4-mtp", 9};
    source.directory = (ifn_infer_bytes){argv[1], strlen(argv[1])};
    source.format_id = (ifn_infer_bytes){"inkling-safetensors", 19};
    rc.temperature = 0;
    rc.seed = 42;
    rc.output_kind = IFN_OUTPUT_RAW_TOKENS;
    rc.max_output_tokens = IFN_WORKER_TOKEN_LIMIT;
    ifn_infer_inkling_request_v1 tuning;
    IFN_INFER_INIT(tuning);
    tuning.draft_tokens = 7;
    tuning.adaptive = 1;
    tuning.prompt_lookup = 1;
    tuning.spec_threshold = 0.8f;
    ifn_infer_extension_v1 extension;
    IFN_INFER_INIT(extension);
    extension.id = IFN_INFER_EXT_INKLING_REQUEST;
    extension.version = 1;
    extension.payload = &tuning;
    extension.payload_bytes = sizeof tuning;
    rc.extensions = (ifn_infer_array){&extension, 1, sizeof extension, 0};
    IFN_WORKER_REQUIRE(api->model_create(&mc, &model, NULL) == IFN_OK);
    IFN_WORKER_REQUIRE(api->model_load(model, &source, NULL) == IFN_OK);
    char system[8192] = "";
    for (unsigned i = 0; i < 64; i++)
        strcat(system, "You are a concise mathematical assistant. Return the numeric answer. ");
    ifn_infer_message_v1 messages[2];
    IFN_INFER_INIT(messages[0]);
    IFN_INFER_INIT(messages[1]);
    messages[0].role = IFN_ROLE_SYSTEM;
    messages[0].content_present = 1;
    messages[0].content = (ifn_infer_bytes){system, strlen(system)};
    messages[1].role = IFN_ROLE_USER;
    messages[1].content_present = 1;
    const char *suffixes[] = {"What is 17 plus 25?", "What is 19 plus 28?"};
    input.kind = IFN_INPUT_CHAT;
    input.messages = (ifn_infer_array){messages, 2, sizeof messages[0], 0};
    printf("{\"kind\":\"start\",\"seed\":%" PRIu64 ",\"cycles\":%" PRIu64 "}\n", seed, cycles);
    fflush(stdout);
    /* Two fresh sessions establish baselines, then a third remains loaded throughout stress. */
    for (unsigned b = 0; b < 2; b++) {
        IFN_WORKER_REQUIRE(api->session_create(model, &sc, &session, NULL) == IFN_OK);
        IFN_WORKER_REQUIRE(api->session_prepare(session, NULL) == IFN_OK);
        messages[1].content = (ifn_infer_bytes){suffixes[b], strlen(suffixes[b])};
        IFN_WORKER_REQUIRE(api->request_create(session, &input, &rc, &request, NULL) == IFN_OK);
        IFN_WORKER_REQUIRE(api->request_run(request, ifn_worker_collect, &baseline[b], NULL) ==
                           IFN_OK);
        IFN_WORKER_REQUIRE(baseline[b].count > 0);
        printf("{\"kind\":\"baseline\",\"suffix\":%u,\"output\":[", b);
        for (size_t i = 0; i < baseline[b].count; i++)
            printf("%s%d", i ? "," : "", baseline[b].tokens[i]);
        puts("]}");
        fflush(stdout);
        IFN_WORKER_REQUIRE(api->request_close(&request, NULL) == IFN_OK);
        IFN_WORKER_REQUIRE(api->session_close(&session, NULL) == IFN_OK);
    }
    IFN_WORKER_REQUIRE(api->session_create(model, &sc, &session, NULL) == IFN_OK);
    IFN_WORKER_REQUIRE(api->session_prepare(session, NULL) == IFN_OK);
    uint64_t stress_started = ifn_worker_now();
    for (uint64_t cycle = 0; cycle < cycles; cycle++) {
        if (cycle && seconds && ifn_worker_now() - stress_started >= seconds * UINT64_C(1000000000))
            break;
        unsigned suffix = ifn_worker_random(&random) & 1;
        unsigned mode = ifn_worker_random(&random) % 3;
        /* Alternate scopes on every cycle; first request must miss even for a shared prefix.
         * Same-scope suffix changes exercise retained system-prefix reuse. */
        rc.cache_scope = (ifn_infer_bytes){(cycle & 1) ? "scope-b" : "scope-a", 7};
        for (unsigned step = 0; step < 5; step++) {
            unsigned variant = suffix ^ (step == 1 ? 1u : 0u);
            messages[1].content = (ifn_infer_bytes){suffixes[variant], strlen(suffixes[variant])};
            ifn_worker_output out = {0};
            out.cancel = step == 2 && mode == 0;
            IFN_WORKER_REQUIRE(api->request_create(session, &input, &rc, &request, NULL) == IFN_OK);
            printf("{\"kind\":\"begin\",\"cycle\":%" PRIu64
                   ",\"step\":%u,\"mode\":%u,\"suffix\":%u}\n",
                   cycle, step, mode, variant);
            fflush(stdout);
            uint64_t started = ifn_worker_now();
            ifn_status status;
            uint32_t cancel = 0;
            if (step == 2 && mode == 1) {
                /* Public progress cannot identify entry into prefill: label pre-output honestly. */
                ifn_worker_task task = {api, request, &out, ATOMIC_VAR_INIT(false), IFN_OK};
                ifn_test_thread thread;
                IFN_WORKER_REQUIRE(!ifn_test_thread_create(&thread, NULL, ifn_worker_run, &task));
                ifn_infer_progress_v1 progress;
                IFN_INFER_INIT(progress);
                while (!atomic_load(&task.finished)) {
                    if (api->request_progress(request, &progress, NULL) != IFN_OK)
                        break;
                    if (progress.state == IFN_REQUEST_RUNNING) {
                        (void)api->request_cancel(request, &cancel, NULL);
                        break;
                    }
                    ifn_test_yield();
                }
                IFN_WORKER_REQUIRE(!ifn_test_thread_join(thread, NULL));
                status = task.status;
                IFN_WORKER_REQUIRE(cancel == IFN_CANCEL_ACCEPTED && out.count == 0);
            } else if (step == 2 && mode == 2) {
                IFN_WORKER_REQUIRE(api->request_cancel(request, &cancel, NULL) == IFN_OK);
                IFN_WORKER_REQUIRE(cancel == IFN_CANCEL_ACCEPTED);
                status = api->request_run(request, ifn_worker_collect, &out, NULL);
            } else {
                status = api->request_run(request, ifn_worker_collect, &out, NULL);
            }
            ifn_infer_result_v1 result;
            IFN_INFER_INIT(result);
            IFN_WORKER_REQUIRE(api->request_result(request, &result, NULL) == IFN_OK);
            int disrupted = step == 2;
            int expect_hit = step == 1 || step == 4;
            IFN_WORKER_REQUIRE(result.status == status);
            IFN_WORKER_REQUIRE(!disrupted || (mode == 0 ? out.count > 0 : out.count == 0));
            int valid = disrupted ? status == IFN_CANCELLED &&
                                        result.finish_reason == IFN_FINISH_CANCELLED_V1
                                  : status == IFN_OK && out.count == baseline[variant].count &&
                                        !memcmp(out.tokens, baseline[variant].tokens,
                                                out.count * sizeof(int32_t)) &&
                                        (expect_hit ? result.prefix_reused_tokens > 0
                                                    : result.prefix_reused_tokens == 0);
            printf("{\"kind\":\"result\",\"cycle\":%" PRIu64
                   ",\"step\":%u,\"mode\":%u,\"status\":%d,"
                   "\"valid\":%s,\"reuse\":%" PRIu64 ",\"tokens\":%zu,\"rss_bytes\":%" PRIu64
                   ",\"run_ns\":%" PRIu64 ",\"ttft_ns\":%" PRIu64 ",\"decode_ns\":%" PRIu64
                   ",\"disposition\":\"next-request\",\"output\":[",
                   cycle, step, mode, (int)status, valid ? "true" : "false",
                   result.prefix_reused_tokens, out.count, ifn_worker_rss(), result.run_ns,
                   out.first_ns ? out.first_ns - started : 0,
                   out.first_ns ? out.last_ns - out.first_ns : 0);
            for (size_t i = 0; i < out.count; i++)
                printf("%s%d", i ? "," : "", out.tokens[i]);
            puts("]}");
            fflush(stdout);
            IFN_WORKER_REQUIRE(valid);
            IFN_WORKER_REQUIRE(api->request_cancel(request, &cancel, NULL) == IFN_OK &&
                               cancel == IFN_CANCEL_TOO_LATE);
            IFN_WORKER_REQUIRE(api->request_close(&request, NULL) == IFN_OK);
        }
        completed++;
    }
done:
    if (api) {
        if (api->request_close(&request, NULL) != IFN_OK)
            failed = 1;
        if (api->session_close(&session, NULL) != IFN_OK)
            failed = 1;
        if (api->model_close(&model, NULL) != IFN_OK)
            failed = 1;
    }
    printf("{\"kind\":\"end\",\"passed\":%s,\"completed_cycles\":%" PRIu64 "}\n",
           failed ? "false" : "true", completed);
    return failed;
}
