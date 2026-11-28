/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "infer_internal.h"
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
extern atomic_uint ifn_infer_mock_uploads;
extern atomic_int ifn_infer_mock_pause, ifn_infer_mock_arrived, ifn_infer_mock_resume;
extern atomic_uint ifn_infer_mock_resets, ifn_infer_mock_calls, ifn_infer_mock_enables,
    ifn_infer_mock_fail, ifn_infer_mock_close_fail, ifn_infer_mock_upload_fail;
#define IFN_INFER_LOCAL_CHECK(x)                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "infer:%d: %s\n", __LINE__, #x);                                       \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
static const ifn_infer_api_v1 *ifn_infer_local_api;
static size_t ifn_infer_local_allocations, ifn_infer_local_releases, ifn_infer_local_fail_at;

static void *ifn_infer_local_allocate(void *ctx, size_t n) {
    (void)ctx;
    if (++ifn_infer_local_allocations == ifn_infer_local_fail_at)
        return NULL;
    return malloc(n);
}

static void ifn_infer_local_release(void *ctx, void *p) {
    (void)ctx;
    if (p)
        ifn_infer_local_releases++;
    free(p);
}

static ifn_infer_model_config_v1 ifn_infer_local_mc;
static ifn_infer_session_config_v1 ifn_infer_local_sc;
static ifn_infer_request_config_v1 ifn_infer_local_rc;
static ifn_infer_input_v1 ifn_infer_local_input;
static int32_t ifn_infer_local_prompt[] = {65, 66};

static uint32_t ifn_infer_local_count_event(void *ctx, const ifn_infer_event_v1 *event) {
    size_t *n = ctx;
    if (event->sequence != *n || !event->token_count || event->tokens[0] != ((*n % 2) ? 66 : 65))
        return 99;
    (*n)++;
    return IFN_CALLBACK_CONTINUE;
}

typedef struct {
    ifn_infer_request *r;
    ifn_status status;
    size_t events;
} ifn_infer_local_worker;

static void *ifn_infer_local_run(void *ctx) {
    ifn_infer_local_worker *w = ctx;
    w->status =
        ifn_infer_local_api->request_run(w->r, ifn_infer_local_count_event, &w->events, NULL);
    return NULL;
}

static uint32_t ifn_infer_local_cancel_event(void *ctx, const ifn_infer_event_v1 *e) {
    (void)ctx;
    (void)e;
    return IFN_CALLBACK_CANCEL;
}

static uint32_t ifn_infer_local_stop_event(void *ctx, const ifn_infer_event_v1 *e) {
    (void)ctx;
    (void)e;
    return IFN_CALLBACK_STOP;
}

static uint32_t ifn_infer_local_invalid_event(void *ctx, const ifn_infer_event_v1 *e) {
    (void)ctx;
    (void)e;
    return 999;
}

static ifn_status ifn_infer_local_ready_model(ifn_infer_model **m, const char *fixture) {
    ifn_status status = ifn_infer_local_api->model_create(&ifn_infer_local_mc, m, NULL);
    if (status)
        return status;
    status = ifn_gpu_weights_create(&(*m)->weights_config, &(*m)->allocator, &(*m)->weights);
    if (!status && fixture)
        status = ifn_tokenizer_open(fixture, &(*m)->allocator, &(*m)->tokenizer);
    if (!status)
        (*m)->state = IFN_HANDLE_READY;
    return status;
}

static void ifn_infer_local_log(void *context, uint32_t level, const ifn_infer_error_v1 *error) {
    ifn_infer_error_v1 *copy = context;
    if (level == IFN_LOG_ERROR)
        *copy = *error;
}

static int ifn_infer_local_diagnostics(void) {
    ifn_infer_error_v1 error, logged;
    IFN_INFER_INIT(error);
    IFN_INFER_INIT(logged);
    ifn_native_error(IFN_GPU_ERROR, 10007, "cuBLAS test failure");
    IFN_INFER_LOCAL_CHECK(ifn_infer_native_error(&error, IFN_GPU_ERROR, IFN_STAGE_DECODE) ==
                          IFN_GPU_ERROR);
    IFN_INFER_LOCAL_CHECK(error.provider == IFN_PROVIDER_CUBLAS && error.provider_code == 7 &&
                          !strcmp(error.message, "cuBLAS test failure"));
    ifn_error(IFN_OK, "");
    IFN_INFER_LOCAL_CHECK(error.provider_code == 7);
    ifn_native_error(IFN_GPU_ERROR, 20201, "driver test failure");
    ifn_infer_native_error(&error, IFN_GPU_ERROR, IFN_STAGE_LOAD);
    IFN_INFER_LOCAL_CHECK(error.provider == IFN_PROVIDER_CUDA_DRIVER && error.provider_code == 201);
    ifn_infer_logger_v1 logger;
    IFN_INFER_INIT(logger);
    logger.callback = ifn_infer_local_log;
    logger.context = &logged;
    ifn_infer_extension_v1 ext;
    IFN_INFER_INIT(ext);
    ext.id = IFN_INFER_EXT_LOGGER;
    ext.version = 1;
    ext.payload = &logger;
    ext.payload_bytes = sizeof logger;
    ifn_infer_local_mc.extensions = (ifn_infer_array){&ext, 1, sizeof ext, 0};
    ifn_infer_model *m = NULL;
    logger.hdr.struct_size = IFN_INFER_RECORD_MAX_BYTES;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_create(&ifn_infer_local_mc, &m, NULL) ==
                              IFN_INVALID_ARGUMENT &&
                          !m);
    logger.hdr.struct_size = sizeof logger;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_ready_model(&m, NULL) == IFN_OK);
    ifn_infer_local_mc.extensions = (ifn_infer_array){0};
    ifn_infer_session *s = NULL;
    ifn_infer_request *r = NULL;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_create(m, &ifn_infer_local_sc, &s, NULL) ==
                          IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_prepare(s, NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_create(
                              s, &ifn_infer_local_input, &ifn_infer_local_rc, &r, NULL) == IFN_OK);
    atomic_store(&ifn_infer_mock_fail, 1);
    size_t count = 0;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_run(r, ifn_infer_local_count_event, &count,
                                                           NULL) == IFN_GPU_ERROR);
    IFN_INFER_LOCAL_CHECK(logged.provider == IFN_PROVIDER_CUDA && logged.provider_code == 719 &&
                          logged.stage == IFN_STAGE_DECODE &&
                          !strcmp(logged.message, "mock native operation"));
    atomic_store(&ifn_infer_mock_fail, 0);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_close(&r, NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_close(&s, NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&m, NULL) == IFN_OK);
    return 0;
}

static int ifn_infer_local_reject_manifest(const char *directory, const char *tokenizer_hash) {
    static const char ifn_zero_hash[] =
        "0000000000000000000000000000000000000000000000000000000000000000";
    ifn_infer_model_config_v1 c = ifn_infer_local_mc;
    c.required_features |= IFN_INFER_CAP_ARTIFACT_VERIFY;
    ifn_infer_model *model = NULL;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_create(&c, &model, NULL) == IFN_UNSUPPORTED &&
                          !model);
    ifn_infer_artifact_manifest_v1 manifest;
    IFN_INFER_INIT(manifest);
    manifest.tokenizer_sha256 = (ifn_infer_bytes){ifn_zero_hash, 64};
    manifest.weights_sha256 = (ifn_infer_bytes){ifn_zero_hash, 64};
    ifn_infer_extension_v1 ext;
    IFN_INFER_INIT(ext);
    ext.id = IFN_INFER_EXT_ARTIFACT_MANIFEST;
    ext.version = 1;
    ext.payload = &manifest;
    ext.payload_bytes = sizeof manifest;
    c.extensions = (ifn_infer_array){&ext, 1, sizeof ext, 0};
    ifn_infer_source_v1 source;
    IFN_INFER_INIT(source);
    source.directory = (ifn_infer_bytes){directory, strlen(directory)};
    source.format_id = (ifn_infer_bytes){"inkling-safetensors", sizeof("inkling-safetensors") - 1};
    ifn_infer_error_v1 error;
    IFN_INFER_INIT(error);
    uint32_t os_provider = IFN_PROVIDER_OS;
#ifdef _WIN32
    os_provider = IFN_PROVIDER_WIN32;
#endif
    for (unsigned pass = 0; pass < 2; pass++) {
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_create(&c, &model, NULL) == IFN_OK);
        ifn_status rc = ifn_infer_local_api->model_load(model, &source, &error);
        IFN_INFER_LOCAL_CHECK(rc == (ifn_status)(pass ? IFN_IO : IFN_FORMAT));
        IFN_INFER_LOCAL_CHECK(!atomic_load(&ifn_infer_mock_uploads));
        if (pass)
            IFN_INFER_LOCAL_CHECK(error.provider == os_provider && error.provider_code > 0);
        else
            IFN_INFER_LOCAL_CHECK(strstr(error.message, "checksum mismatch"));
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&model, NULL) == IFN_OK);
        manifest.tokenizer_sha256 = (ifn_infer_bytes){tokenizer_hash, 64};
    }
    return 0;
}

static int ifn_infer_local_lifecycle(const char *fixture) {
    ifn_infer_model *m = NULL;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_ready_model(&m, fixture) == IFN_OK);
    ifn_infer_session *s = NULL;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_create(m, &ifn_infer_local_sc, &s, NULL) ==
                          IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&m, NULL) == IFN_BUSY && m);
    ifn_infer_memory_plan_v1 plan;
    IFN_INFER_INIT(plan);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_plan(s, &plan, NULL) == IFN_OK &&
                          plan.owned_device_bytes == 1024);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_prepare(s, NULL) == IFN_OK);
    ifn_infer_request *r[3] = {0};
    for (size_t i = 0; i < 3; i++)
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_create(s, &ifn_infer_local_input,
                                                                  &ifn_infer_local_rc, &r[i],
                                                                  NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_close(&s, NULL) == IFN_BUSY);
    ifn_infer_result_v1 result;
    IFN_INFER_INIT(result);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_result(r[0], &result, NULL) ==
                          IFN_INFER_NOT_READY);
    atomic_store(&ifn_infer_mock_pause, 1);
    atomic_store(&ifn_infer_mock_arrived, 0);
    atomic_store(&ifn_infer_mock_resume, 0);
    ifn_infer_local_worker first = {r[0], 999, 0};
    ifn_test_thread t;
    IFN_INFER_LOCAL_CHECK(ifn_test_thread_create(&t, NULL, ifn_infer_local_run, &first) == 0);
    while (!atomic_load(&ifn_infer_mock_arrived))
        ifn_test_yield();
    ifn_infer_local_worker second = {r[1], 999, 0};
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_run(r[1], ifn_infer_local_count_event,
                                                           &second.events, NULL) == IFN_BUSY);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_close(&r[0], NULL) == IFN_BUSY && r[0]);
    uint32_t disposition;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_cancel(r[0], &disposition, NULL) == IFN_OK &&
                          disposition == IFN_CANCEL_ACCEPTED);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_cancel(r[0], &disposition, NULL) == IFN_OK &&
                          disposition == IFN_CANCEL_ALREADY_REQUESTED);
    atomic_store(&ifn_infer_mock_resume, 1);
    IFN_INFER_LOCAL_CHECK(ifn_test_thread_join(t, NULL) == 0 && first.status == IFN_CANCELLED &&
                          !first.events);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_result(r[0], &result, NULL) == IFN_OK &&
                          result.finish_reason == IFN_FINISH_CANCELLED_V1);
    /* An old request stays alive while a new stream runs: stale cancellation
     * must have no path to the new request's native control token. */
    atomic_store(&ifn_infer_mock_arrived, 0);
    atomic_store(&ifn_infer_mock_resume, 0);
    IFN_INFER_LOCAL_CHECK(ifn_test_thread_create(&t, NULL, ifn_infer_local_run, &second) == 0);
    while (!atomic_load(&ifn_infer_mock_arrived))
        ifn_test_yield();
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_cancel(r[0], &disposition, NULL) == IFN_OK &&
                          disposition == IFN_CANCEL_TOO_LATE);
    atomic_store(&ifn_infer_mock_resume, 1);
    IFN_INFER_LOCAL_CHECK(ifn_test_thread_join(t, NULL) == 0 && second.status == IFN_OK &&
                          second.events == ifn_infer_local_rc.max_output_tokens);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_result(r[1], &result, NULL) == IFN_OK &&
                          !result.cancel_requested &&
                          result.generated_tokens == ifn_infer_local_rc.max_output_tokens);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_run(r[1], ifn_infer_local_count_event,
                                                           &second.events,
                                                           NULL) == IFN_INFER_INVALID_STATE);
    atomic_store(&ifn_infer_mock_pause, 0);
    unsigned calls = atomic_load(&ifn_infer_mock_calls);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_cancel(r[2], &disposition, NULL) == IFN_OK &&
                          disposition == IFN_CANCEL_ACCEPTED);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_run(r[2], ifn_infer_local_count_event,
                                                           &second.events, NULL) == IFN_CANCELLED);
    IFN_INFER_LOCAL_CHECK(atomic_load(&ifn_infer_mock_calls) == calls);
    for (size_t i = 0; i < 3; i++)
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_close(&r[i], NULL) == IFN_OK && !r[i]);
    for (unsigned mode = 0; mode < 3; mode++) {
        ifn_infer_request *one = NULL;
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_create(s, &ifn_infer_local_input,
                                                                  &ifn_infer_local_rc, &one,
                                                                  NULL) == IFN_OK);
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_run(
                                  one,
                                  mode == 0   ? ifn_infer_local_cancel_event
                                  : mode == 1 ? ifn_infer_local_stop_event
                                              : ifn_infer_local_invalid_event,
                                  NULL, NULL) == (ifn_status)(mode == 0   ? IFN_CANCELLED
                                                              : mode == 1 ? IFN_OK
                                                                          : IFN_INVALID_ARGUMENT));
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_result(one, &result, NULL) == IFN_OK &&
                              result.generated_tokens == 1);
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_close(&one, NULL) == IFN_OK);
    }
    atomic_store(&ifn_infer_mock_close_fail, 1);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_close(&s, NULL) == IFN_GPU_ERROR && s);
    ifn_infer_request *after_close = NULL;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_create(s, &ifn_infer_local_input,
                                                              &ifn_infer_local_rc, &after_close,
                                                              NULL) == IFN_INFER_NOT_READY &&
                          !after_close);
    atomic_store(&ifn_infer_mock_close_fail, 0);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_close(&s, NULL) == IFN_OK && !s);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&m, NULL) == IFN_OK && !m);
    return 0;
}

static int ifn_infer_local_invalid_inputs(void) {
    ifn_infer_error_v1 error;
    IFN_INFER_INIT(error);
    ifn_infer_model *m = (void *)1;
    ifn_infer_model_config_v1 c = ifn_infer_local_mc;
    c.hdr.flags = 1;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_create(&c, &m, &error) ==
                              IFN_INVALID_ARGUMENT &&
                          !m && error.status == IFN_INVALID_ARGUMENT);
    c = ifn_infer_local_mc;
    c.required_features = UINT64_C(1) << 63;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_create(&c, &m, NULL) == IFN_UNSUPPORTED && !m);
    c = ifn_infer_local_mc;
    c.allocator.release = NULL;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_create(&c, &m, NULL) == IFN_INVALID_ARGUMENT &&
                          !m);
    c = ifn_infer_local_mc;
    c.max_owned_host_bytes = 0;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_create(&c, &m, NULL) == IFN_LIMIT && !m);

    struct {
        ifn_infer_model_config_v1 c;
        uint64_t tail;
    } bigger = {ifn_infer_local_mc, 1};

    bigger.c.hdr.struct_size = sizeof bigger;
    IFN_INFER_LOCAL_CHECK(
        ifn_infer_local_api->model_create(&bigger.c, &m, NULL) == IFN_UNSUPPORTED && !m);
    bigger.tail = 0;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_create(&bigger.c, &m, NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&m, NULL) == IFN_OK);

    struct {
        ifn_infer_backend_info_v1 info;
        uint64_t canary;
    } output = {0};

    output.info.hdr.struct_size = sizeof output;
    output.canary = UINT64_C(0xdeadbeef);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->backend_info(&output.info, &error) == IFN_OK &&
                          output.canary == UINT64_C(0xdeadbeef));
    IFN_INFER_LOCAL_CHECK(output.info.hdr.struct_size == sizeof output &&
                          output.info.hdr.written_size == sizeof output.info);
    IFN_INFER_LOCAL_CHECK(output.info.max_active_runs == 1 &&
                          output.info.limit_scope == IFN_LIMIT_PROVIDER_PROCESS);
    output.info.hdr.struct_size = 8;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->backend_info(&output.info, NULL) ==
                          IFN_INVALID_ARGUMENT);
    c = ifn_infer_local_mc;
    c.max_owned_device_bytes = 0;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_create(&c, &m, NULL) == IFN_OK);
    ifn_infer_source_v1 source;
    IFN_INFER_INIT(source);
    source.directory = ifn_infer_text("/not-opened-budget-preflight");
    source.format_id = ifn_infer_text("inkling-safetensors");
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_load(m, &source, NULL) == IFN_LIMIT &&
                          m->state == IFN_HANDLE_CREATED);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&m, NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_ready_model(&m, NULL) == IFN_OK);
    ifn_infer_session *s = NULL;
    ifn_infer_session_config_v1 limited = ifn_infer_local_sc;
    limited.max_owned_device_bytes = 0;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_create(m, &limited, &s, NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_prepare(s, NULL) == IFN_LIMIT && !s->native);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_close(&s, NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_create(m, &ifn_infer_local_sc, &s, NULL) ==
                              IFN_OK &&
                          ifn_infer_local_api->session_prepare(s, NULL) == IFN_OK);
    ifn_infer_request *r = (void *)1;
    ifn_infer_request_config_v1 bad = ifn_infer_local_rc;
    bad.max_output_tokens = UINT64_MAX;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_create(s, &ifn_infer_local_input, &bad, &r,
                                                              NULL) == IFN_INVALID_ARGUMENT &&
                          !r);
    bad = ifn_infer_local_rc;
    bad.extensions.count = UINT64_MAX;
    bad.extensions.stride = 1;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_create(s, &ifn_infer_local_input, &bad, &r,
                                                              NULL) == IFN_INVALID_ARGUMENT &&
                          !r);
    ifn_infer_input_v1 malformed = ifn_infer_local_input;
    malformed.kind = IFN_INPUT_CHAT;
    malformed.tokens = NULL;
    malformed.token_count = 0;
    malformed.messages = (ifn_infer_array){ifn_infer_local_prompt, UINT64_MAX, 1, 0};
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_create(s, &malformed, &ifn_infer_local_rc,
                                                              &r, NULL) == IFN_INVALID_ARGUMENT &&
                          !r);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_close(&s, NULL) == IFN_OK &&
                          ifn_infer_local_api->model_close(&m, NULL) == IFN_OK);
    return 0;
}

static int ifn_infer_local_fault_injection(void) {
    size_t baseline_allocations = ifn_infer_local_allocations,
           baseline_releases = ifn_infer_local_releases;
    for (size_t nth = 1; nth <= 8; nth++) {
        size_t before = ifn_infer_local_allocations, freed_before = ifn_infer_local_releases;
        ifn_infer_local_fail_at = ifn_infer_local_allocations + nth;
        ifn_infer_model *m = NULL;
        ifn_infer_session *s = NULL;
        ifn_infer_request *r = NULL;
        ifn_status status = ifn_infer_local_ready_model(&m, NULL);
        if (!status)
            status = ifn_infer_local_api->session_create(m, &ifn_infer_local_sc, &s, NULL);
        if (!status)
            status = ifn_infer_local_api->session_prepare(s, NULL);
        if (!status)
            status = ifn_infer_local_api->request_create(s, &ifn_infer_local_input,
                                                         &ifn_infer_local_rc, &r, NULL);
        IFN_INFER_LOCAL_CHECK(status == IFN_OK || status == IFN_OUT_OF_MEMORY);
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_close(&r, NULL) == IFN_OK);
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_close(&s, NULL) == IFN_OK);
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&m, NULL) == IFN_OK);
        size_t attempted = ifn_infer_local_allocations - before,
               freed = ifn_infer_local_releases - freed_before;
        IFN_INFER_LOCAL_CHECK(freed == attempted - (attempted >= nth ? 1 : 0));
        ifn_infer_local_fail_at = 0;
    }
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_allocations - baseline_allocations >=
                          ifn_infer_local_releases - baseline_releases);
    return 0;
}

static uint32_t ifn_infer_local_text_event(void *ctx, const ifn_infer_event_v1 *event) {
    char *out = ctx;
    size_t n = strlen(out);
    if (event->kind != IFN_EVENT_CONTENT || n + event->text.bytes >= 32)
        return 99;
    memcpy(out + n, event->text.data, (size_t)event->text.bytes);
    out[n + event->text.bytes] = 0;
    return IFN_CALLBACK_CONTINUE;
}

static int ifn_infer_local_text_and_recovery(const char *fixture) {
    ifn_infer_model *m = NULL;
    ifn_infer_session *s = NULL;
    ifn_infer_request *r = NULL;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_ready_model(&m, fixture) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_create(m, &ifn_infer_local_sc, &s, NULL) ==
                              IFN_OK &&
                          ifn_infer_local_api->session_prepare(s, NULL) == IFN_OK);
    ifn_infer_input_v1 text;
    IFN_INFER_INIT(text);
    text.kind = IFN_INPUT_RAW_TEXT;
    text.text = ifn_infer_text("prompt");
    ifn_infer_request_config_v1 c = ifn_infer_local_rc;
    c.output_kind = IFN_OUTPUT_TEXT;
    c.max_event_bytes = 4;
    char stop[] = "BA";
    ifn_infer_bytes span = {stop, 2};
    c.stops = (ifn_infer_array){&span, 1, sizeof span, 0};
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_create(s, &text, &c, &r, NULL) == IFN_OK);
    stop[0] = 'X'; /* caller buffers no longer authoritative */
    char result_text[32] = "";
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_run(r, ifn_infer_local_text_event,
                                                           result_text, NULL) == IFN_OK &&
                          !strcmp(result_text, "A"));
    ifn_infer_result_v1 result;
    IFN_INFER_INIT(result);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_result(r, &result, NULL) == IFN_OK &&
                          result.finish_reason == IFN_FINISH_STOP_STRING_V1 &&
                          result.generated_tokens == 3);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_close(&r, NULL) == IFN_OK);
    unsigned enables = atomic_load(&ifn_infer_mock_enables);
    c = ifn_infer_local_rc;
    c.cache_scope = ifn_infer_text("tenant-a");
    for (unsigned i = 0; i < 3; i++) {
        if (i == 2)
            c.cache_scope = ifn_infer_text("tenant-b");
        size_t count = 0;
        IFN_INFER_LOCAL_CHECK(
            ifn_infer_local_api->request_create(s, &ifn_infer_local_input, &c, &r, NULL) == IFN_OK);
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_run(r, ifn_infer_local_count_event,
                                                               &count, NULL) == IFN_OK);
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_close(&r, NULL) == IFN_OK);
    }
    IFN_INFER_LOCAL_CHECK(atomic_load(&ifn_infer_mock_enables) == enables + 2);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_create(
                              s, &ifn_infer_local_input, &ifn_infer_local_rc, &r, NULL) == IFN_OK);
    atomic_store(&ifn_infer_mock_pause, 2);
    atomic_store(&ifn_infer_mock_arrived, 0);
    atomic_store(&ifn_infer_mock_resume, 0);
    ifn_infer_local_worker work = {r, 999, 0};
    ifn_test_thread t;
    IFN_INFER_LOCAL_CHECK(!ifn_test_thread_create(&t, NULL, ifn_infer_local_run, &work));
    while (!atomic_load(&ifn_infer_mock_arrived))
        ifn_test_yield();
    uint32_t disposition;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_cancel(r, &disposition, NULL) == IFN_OK &&
                          disposition == IFN_CANCEL_ACCEPTED);
    atomic_store(&ifn_infer_mock_resume, 1);
    IFN_INFER_LOCAL_CHECK(!ifn_test_thread_join(t, NULL) && work.status == IFN_CANCELLED);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_close(&r, NULL) == IFN_OK);
    atomic_store(&ifn_infer_mock_pause, 0);
    /* Accepted cancellation followed by a failed native reset must not report
     * successful cancellation or allow the poisoned session to run again. */
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_create(
                              s, &ifn_infer_local_input, &ifn_infer_local_rc, &r, NULL) == IFN_OK);
    atomic_store(&ifn_infer_mock_fail, 2);
    ifn_infer_error_v1 error;
    IFN_INFER_INIT(error);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_run(r, ifn_infer_local_cancel_event, NULL,
                                                           &error) == IFN_GPU_ERROR);
    IFN_INFER_LOCAL_CHECK(error.flags & IFN_INFER_ERR_SESSION_POISONED);
    IFN_INFER_LOCAL_CHECK(error.cleanup_status == IFN_GPU_ERROR);
    IFN_INFER_LOCAL_CHECK(error.provider == IFN_PROVIDER_CUDA && error.provider_code == 700 &&
                          error.cleanup_provider_code == 700);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_close(&r, NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_create(s, &ifn_infer_local_input,
                                                              &ifn_infer_local_rc, &r,
                                                              NULL) == IFN_INFER_NOT_READY &&
                          !r);
    atomic_store(&ifn_infer_mock_fail, 0);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_close(&s, NULL) == IFN_OK &&
                          ifn_infer_local_api->model_close(&m, NULL) == IFN_OK);
    return 0;
}

static int ifn_infer_local_cancellation_races(void) {
    ifn_infer_model *m = NULL;
    ifn_infer_session *s = NULL;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_ready_model(&m, NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_create(m, &ifn_infer_local_sc, &s, NULL) ==
                              IFN_OK &&
                          ifn_infer_local_api->session_prepare(s, NULL) == IFN_OK);
    for (unsigned i = 0; i < 100; i++) {
        ifn_infer_request *r = NULL;
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_create(s, &ifn_infer_local_input,
                                                                  &ifn_infer_local_rc, &r,
                                                                  NULL) == IFN_OK);
        ifn_infer_local_worker w = {r, 999, 0};
        ifn_test_thread t;
        IFN_INFER_LOCAL_CHECK(!ifn_test_thread_create(&t, NULL, ifn_infer_local_run, &w));
        if (i & 1)
            ifn_test_yield();
        uint32_t disposition;
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_cancel(r, &disposition, NULL) == IFN_OK);
        IFN_INFER_LOCAL_CHECK(!ifn_test_thread_join(t, NULL));
        IFN_INFER_LOCAL_CHECK((disposition == IFN_CANCEL_ACCEPTED && w.status == IFN_CANCELLED) ||
                              (disposition == IFN_CANCEL_TOO_LATE && w.status == IFN_OK));
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->request_close(&r, NULL) == IFN_OK);
    }
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_close(&s, NULL) == IFN_OK &&
                          ifn_infer_local_api->model_close(&m, NULL) == IFN_OK);
    return 0;
}

static int ifn_infer_local_load_fixture(const char *directory) {
    ifn_infer_source_v1 source;
    IFN_INFER_INIT(source);
    source.directory = ifn_infer_text(directory);
    source.format_id = ifn_infer_text("inkling-safetensors");
    ifn_infer_model *m = NULL;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_create(&ifn_infer_local_mc, &m, NULL) ==
                          IFN_OK);
    atomic_store(&ifn_infer_mock_upload_fail, 1);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_load(m, &source, NULL) == IFN_GPU_ERROR &&
                          m->checkpoint && m->state == IFN_HANDLE_FAILED);
    atomic_store(&ifn_infer_mock_close_fail, 1);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&m, NULL) == IFN_GPU_ERROR && m &&
                          m->checkpoint);
    IFN_INFER_LOCAL_CHECK(ifn_inkling_checkpoint_count(m->checkpoint) == 822);
    atomic_store(&ifn_infer_mock_close_fail, 0);
    atomic_store(&ifn_infer_mock_upload_fail, 0);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&m, NULL) == IFN_OK && !m);
    size_t calls = ifn_infer_local_allocations;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_create(&ifn_infer_local_mc, &m, NULL) ==
                              IFN_OK &&
                          ifn_infer_local_api->model_load(m, &source, NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(!m->checkpoint && m->state == IFN_HANDLE_READY);
    size_t count = ifn_infer_local_allocations - calls;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&m, NULL) == IFN_OK);
    for (size_t i = 1; i <= count; i++) {
        size_t start = ifn_infer_local_allocations, freed = ifn_infer_local_releases;
        ifn_infer_local_fail_at = ifn_infer_local_allocations + i;
        ifn_status status = ifn_infer_local_api->model_create(&ifn_infer_local_mc, &m, NULL);
        if (!status)
            status = ifn_infer_local_api->model_load(m, &source, NULL);
        IFN_INFER_LOCAL_CHECK(status == IFN_OUT_OF_MEMORY);
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&m, NULL) == IFN_OK && !m);
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_releases - freed ==
                              ifn_infer_local_allocations - start - 1);
        ifn_infer_local_fail_at = 0;
    }
    puts("facade model-load allocation faults and retained checkpoint cleanup passed (sparse "
         "metadata, mock upload)");
    return 0;
}

static int ifn_infer_local_profiles(void) {
    const char *weights[] = {"exact-ar", "fp8-ar", "nvfp4-ar", "exact-mtp", "fp8-mtp", "nvfp4-mtp"};
    const char *execution[] = {"bf16-eager", "fp8-eager", "q4-eager",
                               "bf16-graph", "fp8-graph", "q4-graph"};
    for (unsigned i = 0; i < 6; i++) {
        ifn_infer_model_config_v1 c = ifn_infer_local_mc;
        c.weight_profile = ifn_infer_text(weights[i]);
        ifn_infer_model *m = NULL;
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_create(&c, &m, NULL) == IFN_OK);
        IFN_INFER_LOCAL_CHECK(m->weights_config.mode == i % 3 &&
                              m->weights_config.with_mtp == i / 3 &&
                              !strcmp(m->profile, weights[i]));
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&m, NULL) == IFN_OK);
    }
    ifn_infer_model *m = NULL;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_ready_model(&m, NULL) == IFN_OK);
    for (unsigned i = 0; i < 6; i++) {
        ifn_infer_session_config_v1 c = ifn_infer_local_sc;
        c.execution_profile = ifn_infer_text(execution[i]);
        ifn_infer_session *s = NULL;
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_create(m, &c, &s, NULL) == IFN_OK);
        IFN_INFER_LOCAL_CHECK(s->config.kv_format == i % 3 && s->config.use_graphs == i / 3 &&
                              !strcmp(s->profile, execution[i]));
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->session_close(&s, NULL) == IFN_OK);
    }
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->model_close(&m, NULL) == IFN_OK);
    return 0;
}

int main(int argc, char **argv) {
    const void *table = (void *)1;
    IFN_INFER_LOCAL_CHECK(ifn_infer_get_api(999, 0, &table) == IFN_INFER_ABI_MISMATCH && !table);
    IFN_INFER_LOCAL_CHECK(
        ifn_infer_get_api(1, IFN_INFER_ABI_MINOR + 1, &table) == IFN_INFER_ABI_MISMATCH && !table);
    IFN_INFER_LOCAL_CHECK(ifn_infer_get_api(1, 0, &table) == IFN_OK && table);
    ifn_infer_local_api = table;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_api->struct_size == sizeof *ifn_infer_local_api &&
                          ifn_infer_local_api->abi_major == 1);
    IFN_INFER_INIT(ifn_infer_local_mc);
    IFN_INFER_INIT(ifn_infer_local_sc);
    IFN_INFER_INIT(ifn_infer_local_rc);
    IFN_INFER_INIT(ifn_infer_local_input);
    IFN_INFER_LOCAL_CHECK(
        ifn_infer_local_api->defaults(IFN_DEFAULT_MODEL, &ifn_infer_local_mc, NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(
        ifn_infer_local_api->defaults(IFN_DEFAULT_SESSION, &ifn_infer_local_sc, NULL) == IFN_OK);
    IFN_INFER_LOCAL_CHECK(
        ifn_infer_local_api->defaults(IFN_DEFAULT_REQUEST, &ifn_infer_local_rc, NULL) == IFN_OK);
    ifn_infer_local_mc.allocator =
        (ifn_infer_allocator){NULL, ifn_infer_local_allocate, ifn_infer_local_release};
    if (argc == 4 && !strcmp(argv[1], "--reject-manifest"))
        return ifn_infer_local_reject_manifest(argv[2], argv[3]);
    if (argc == 3 && !strcmp(argv[1], "--load-fixture"))
        return ifn_infer_local_load_fixture(argv[2]);
    ifn_infer_local_rc.max_output_tokens = 8;
    ifn_infer_local_rc.output_kind = IFN_OUTPUT_RAW_TOKENS;
    ifn_infer_local_input.kind = IFN_INPUT_TOKENS;
    ifn_infer_local_input.tokens = ifn_infer_local_prompt;
    ifn_infer_local_input.token_count = 2;
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_profiles() == 0);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_diagnostics() == 0);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_invalid_inputs() == 0);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_lifecycle(argc > 1 ? argv[1] : NULL) == 0);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_fault_injection() == 0);
    if (argc > 1)
        IFN_INFER_LOCAL_CHECK(ifn_infer_local_text_and_recovery(argv[1]) == 0);
    IFN_INFER_LOCAL_CHECK(ifn_infer_local_cancellation_races() == 0);
    puts("inference ABI negotiation, ownership, independent streams, stale cancel and allocation "
         "faults passed (mock execution)");
    return 0;
}
