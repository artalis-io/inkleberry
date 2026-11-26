/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Public-ABI consumer. Run only through tools/run_cuda_smoke.py idle UUID guard. */
#include "ifn_infer.h"
#include "ifn_infer_inkling.h"
#include "platform.h"
#include "platform.h"
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#define IFN_INFER_NATIVE_LOCAL_REQUIRE(x)                                                          \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "infer-native:%d: %s\n", __LINE__, #x);                                \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

typedef struct {
    char text[4096];
    size_t bytes, events;
    atomic_bool arrived, resume;
    int block;
} ifn_infer_native_local_output;

static uint32_t ifn_infer_native_local_collect(void *ctx, const ifn_infer_event_v1 *event) {
    ifn_infer_native_local_output *o = ctx;
    if (event->kind == IFN_EVENT_CONTENT) {
        if (event->text.bytes > sizeof o->text - 1 - o->bytes)
            return IFN_CALLBACK_CANCEL;
        memcpy(o->text + o->bytes, event->text.data, (size_t)event->text.bytes);
        o->bytes += (size_t)event->text.bytes;
        o->text[o->bytes] = 0;
    }
    o->events++;
    if (o->block) {
        atomic_store(&o->arrived, true);
        while (!atomic_load(&o->resume))
            ifn_test_yield();
    }
    return IFN_CALLBACK_CONTINUE;
}

typedef struct {
    const ifn_infer_api_v1 *api;
    ifn_infer_request *request;
    ifn_infer_native_local_output *out;
    ifn_status status;
} ifn_infer_native_local_task;

static void *ifn_infer_native_local_run(void *ctx) {
    ifn_infer_native_local_task *t = ctx;
    t->status = t->api->request_run(t->request, ifn_infer_native_local_collect, t->out, NULL);
    return NULL;
}

static void ifn_infer_native_local_output_init(ifn_infer_native_local_output *o) {
    memset(o, 0, sizeof *o);
    atomic_init(&o->arrived, false);
    atomic_init(&o->resume, false);
}

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    int failed = 0;
    const void *table = NULL;
    const ifn_infer_api_v1 *api = NULL;
    ifn_infer_model *model = NULL;
    ifn_infer_session *session = NULL;
    ifn_infer_request *requests[6] = {0};
    ifn_test_thread thread;
    int thread_live = 0;
    ifn_infer_native_local_output outputs[6];
    for (size_t i = 0; i < 6; i++)
        ifn_infer_native_local_output_init(&outputs[i]);
    ifn_infer_model_config_v1 mc;
    ifn_infer_session_config_v1 sc;
    ifn_infer_request_config_v1 rc;
    ifn_infer_source_v1 source;
    ifn_infer_input_v1 input;
    ifn_infer_result_v1 result;
    ifn_infer_memory_plan_v1 plan;
    IFN_INFER_INIT(mc);
    IFN_INFER_INIT(sc);
    IFN_INFER_INIT(rc);
    IFN_INFER_INIT(source);
    IFN_INFER_INIT(input);
    IFN_INFER_INIT(result);
    IFN_INFER_INIT(plan);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(ifn_infer_get_api(1, 0, &table) == IFN_OK);
    api = table;
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->defaults(IFN_DEFAULT_MODEL, &mc, NULL) == IFN_OK);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->defaults(IFN_DEFAULT_SESSION, &sc, NULL) == IFN_OK);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->defaults(IFN_DEFAULT_REQUEST, &rc, NULL) == IFN_OK);
    mc.weight_profile = (ifn_infer_bytes){"nvfp4-mtp", 9};
    source.directory = (ifn_infer_bytes){argv[1], strlen(argv[1])};
    source.format_id = (ifn_infer_bytes){"inkling-safetensors", sizeof("inkling-safetensors") - 1};
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->model_create(&mc, &model, NULL) == IFN_OK);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->model_plan(model, &source, &plan, NULL) == IFN_OK &&
                                   plan.peak_owned_device_bytes > 0);
    printf("weight_peak_bytes=%llu\n", (unsigned long long)plan.peak_owned_device_bytes);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->model_load(model, &source, NULL) == IFN_OK);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->session_create(model, &sc, &session, NULL) == IFN_OK);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->session_plan(session, &plan, NULL) == IFN_OK &&
                                   plan.owned_device_bytes > 0);
    printf("session_arena_bytes=%llu\n", (unsigned long long)plan.owned_device_bytes);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->session_prepare(session, NULL) == IFN_OK);
    char system[8192] = "";
    for (unsigned i = 0; i < 64; i++)
        strcat(system, "You are a concise mathematical assistant. Return the numeric answer. ");
    ifn_infer_message_v1 messages[4];
    for (size_t i = 0; i < 4; i++)
        IFN_INFER_INIT(messages[i]);
    const char *texts[] = {system, "What is 2 + 2?", "4.", "What is that number plus 3?"};
    const uint32_t roles[] = {IFN_ROLE_SYSTEM, IFN_ROLE_USER, IFN_ROLE_ASSISTANT, IFN_ROLE_USER};
    for (size_t i = 0; i < 4; i++) {
        messages[i].role = roles[i];
        messages[i].content_present = 1;
        messages[i].content = (ifn_infer_bytes){texts[i], strlen(texts[i])};
    }
    input.kind = IFN_INPUT_CHAT;
    input.messages = (ifn_infer_array){messages, 4, sizeof messages[0], 0};
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
    rc.max_output_tokens = 128;
    rc.max_event_bytes = 4;
    rc.cache_scope = (ifn_infer_bytes){"native-a", 8};
    for (size_t i = 0; i < 4; i++)
        IFN_INFER_NATIVE_LOCAL_REQUIRE(
            api->request_create(session, &input, &rc, &requests[i], NULL) == IFN_OK);
    rc.cache_scope = (ifn_infer_bytes){"native-b", 8};
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->request_create(session, &input, &rc, &requests[4], NULL) ==
                                   IFN_OK);
    ifn_infer_bytes stop = {"7", 1};
    rc.stops = (ifn_infer_array){&stop, 1, sizeof stop, 0};
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->request_create(session, &input, &rc, &requests[5], NULL) ==
                                   IFN_OK);
    /* All request-owned inputs are prepared; overwrite the borrowed text/options. */
    memset(system, 0, sizeof system);
    tuning.draft_tokens = 99;
    IFN_INFER_NATIVE_LOCAL_REQUIRE(
        api->request_run(requests[0], ifn_infer_native_local_collect, &outputs[0], NULL) == IFN_OK);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(strstr(outputs[0].text, "7") != NULL);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->request_result(requests[0], &result, NULL) == IFN_OK &&
                                   result.input_tokens > 128);
    printf("answer=%s prompt=%llu output=%llu run_ns=%llu\n", outputs[0].text,
           (unsigned long long)result.input_tokens, (unsigned long long)result.generated_tokens,
           (unsigned long long)result.run_ns);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(
        api->request_run(requests[1], ifn_infer_native_local_collect, &outputs[1], NULL) == IFN_OK);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(!strcmp(outputs[0].text, outputs[1].text));
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->request_result(requests[1], &result, NULL) == IFN_OK &&
                                   result.prefix_reused_tokens > 0);
    printf("prefix_reused=%llu\n", (unsigned long long)result.prefix_reused_tokens);
    outputs[2].block = 1;
    ifn_infer_native_local_task work = {api, requests[2], &outputs[2], 999};
    IFN_INFER_NATIVE_LOCAL_REQUIRE(
        !ifn_test_thread_create(&thread, NULL, ifn_infer_native_local_run, &work));
    thread_live = 1;
    while (!atomic_load(&outputs[2].arrived))
        ifn_test_yield();
    uint32_t disposition;
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->request_cancel(requests[0], &disposition, NULL) == IFN_OK &&
                                   disposition == IFN_CANCEL_TOO_LATE);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->request_run(requests[3], ifn_infer_native_local_collect,
                                                    &outputs[3], NULL) == IFN_BUSY);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->request_cancel(requests[2], &disposition, NULL) == IFN_OK &&
                                   disposition == IFN_CANCEL_ACCEPTED);
    atomic_store(&outputs[2].resume, true);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(!ifn_test_thread_join(thread, NULL));
    thread_live = 0;
    IFN_INFER_NATIVE_LOCAL_REQUIRE(work.status == IFN_CANCELLED);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->request_result(requests[2], &result, NULL) == IFN_OK &&
                                   result.finish_reason == IFN_FINISH_CANCELLED_V1);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(
        api->request_run(requests[3], ifn_infer_native_local_collect, &outputs[3], NULL) == IFN_OK);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(!strcmp(outputs[0].text, outputs[3].text));
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->request_result(requests[3], &result, NULL) == IFN_OK &&
                                   result.prefix_reused_tokens == 0);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(
        api->request_run(requests[4], ifn_infer_native_local_collect, &outputs[4], NULL) == IFN_OK);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(!strcmp(outputs[0].text, outputs[4].text));
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->request_result(requests[4], &result, NULL) == IFN_OK &&
                                   result.prefix_reused_tokens == 0);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(
        api->request_run(requests[5], ifn_infer_native_local_collect, &outputs[5], NULL) == IFN_OK);
    IFN_INFER_NATIVE_LOCAL_REQUIRE(api->request_result(requests[5], &result, NULL) == IFN_OK &&
                                   result.finish_reason == IFN_FINISH_STOP_STRING_V1 &&
                                   !outputs[5].bytes);
    puts("native ABI: copied chat, adaptive/lookup, answer parity, cache reuse, independent "
         "streams, BUSY, stale cancel, active cancel, cache invalidation, scope isolation, stop "
         "filtering and replay passed");
done:
    if (thread_live) {
        atomic_store(&outputs[2].resume, true);
        (void)ifn_test_thread_join(thread, NULL);
    }
    if (api) {
        for (size_t i = 0; i < 6; i++)
            if (api->request_close(&requests[i], NULL) != IFN_OK)
                failed = 1;
        if (api->session_close(&session, NULL) != IFN_OK)
            failed = 1;
        if (api->model_close(&model, NULL) != IFN_OK)
            failed = 1;
    }
    return failed;
}
