/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "infer_internal.h"
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
extern atomic_int si_mock_pause, si_mock_arrived, si_mock_resume;
extern atomic_uint si_mock_resets, si_mock_calls, si_mock_enables, si_mock_fail, si_mock_close_fail,
    si_mock_upload_fail;
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "infer:%d: %s\n", __LINE__, #x);                                       \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
static const sq_infer_api_v1 *api;
static size_t allocations, releases, fail_at;

static void *allocate(void *ctx, size_t n) {
    (void)ctx;
    if (++allocations == fail_at)
        return NULL;
    return malloc(n);
}

static void release(void *ctx, void *p) {
    (void)ctx;
    if (p)
        releases++;
    free(p);
}

static sq_infer_model_config_v1 mc;
static sq_infer_session_config_v1 sc;
static sq_infer_request_config_v1 rc;
static sq_infer_input_v1 input;
static int32_t prompt[] = {65, 66};

static uint32_t count_event(void *ctx, const sq_infer_event_v1 *event) {
    size_t *n = ctx;
    if (event->sequence != *n || !event->token_count || event->tokens[0] != ((*n % 2) ? 66 : 65))
        return 99;
    (*n)++;
    return SQ_CALLBACK_CONTINUE;
}

typedef struct {
    sq_infer_request *r;
    sq_status status;
    size_t events;
} worker;

static void *run(void *ctx) {
    worker *w = ctx;
    w->status = api->request_run(w->r, count_event, &w->events, NULL);
    return NULL;
}

static uint32_t cancel_event(void *ctx, const sq_infer_event_v1 *e) {
    (void)ctx;
    (void)e;
    return SQ_CALLBACK_CANCEL;
}

static uint32_t stop_event(void *ctx, const sq_infer_event_v1 *e) {
    (void)ctx;
    (void)e;
    return SQ_CALLBACK_STOP;
}

static uint32_t invalid_event(void *ctx, const sq_infer_event_v1 *e) {
    (void)ctx;
    (void)e;
    return 999;
}

static sq_status ready_model(sq_infer_model **m, const char *fixture) {
    sq_status status = api->model_create(&mc, m, NULL);
    if (status)
        return status;
    status = sq_gpu_weights_create(&(*m)->weights_config, &(*m)->allocator, &(*m)->weights);
    if (!status && fixture)
        status = sq_tokenizer_open(fixture, &(*m)->allocator, &(*m)->tokenizer);
    if (!status)
        (*m)->state = SQ_HANDLE_READY;
    return status;
}

static int lifecycle(const char *fixture) {
    sq_infer_model *m = NULL;
    CHECK(ready_model(&m, fixture) == SQ_OK);
    sq_infer_session *s = NULL;
    CHECK(api->session_create(m, &sc, &s, NULL) == SQ_OK);
    CHECK(api->model_close(&m, NULL) == SQ_BUSY && m);
    sq_infer_memory_plan_v1 plan;
    SQ_INFER_INIT(plan);
    CHECK(api->session_plan(s, &plan, NULL) == SQ_OK && plan.owned_device_bytes == 1024);
    CHECK(api->session_prepare(s, NULL) == SQ_OK);
    sq_infer_request *r[3] = {0};
    for (size_t i = 0; i < 3; i++)
        CHECK(api->request_create(s, &input, &rc, &r[i], NULL) == SQ_OK);
    CHECK(api->session_close(&s, NULL) == SQ_BUSY);
    sq_infer_result_v1 result;
    SQ_INFER_INIT(result);
    CHECK(api->request_result(r[0], &result, NULL) == SQ_INFER_NOT_READY);
    atomic_store(&si_mock_pause, 1);
    atomic_store(&si_mock_arrived, 0);
    atomic_store(&si_mock_resume, 0);
    worker first = {r[0], 999, 0};
    pthread_t t;
    CHECK(pthread_create(&t, NULL, run, &first) == 0);
    while (!atomic_load(&si_mock_arrived))
        sched_yield();
    worker second = {r[1], 999, 0};
    CHECK(api->request_run(r[1], count_event, &second.events, NULL) == SQ_BUSY);
    CHECK(api->request_close(&r[0], NULL) == SQ_BUSY && r[0]);
    uint32_t disposition;
    CHECK(api->request_cancel(r[0], &disposition, NULL) == SQ_OK &&
          disposition == SQ_CANCEL_ACCEPTED);
    CHECK(api->request_cancel(r[0], &disposition, NULL) == SQ_OK &&
          disposition == SQ_CANCEL_ALREADY_REQUESTED);
    atomic_store(&si_mock_resume, 1);
    CHECK(pthread_join(t, NULL) == 0 && first.status == SQ_CANCELLED && !first.events);
    CHECK(api->request_result(r[0], &result, NULL) == SQ_OK &&
          result.finish_reason == SQ_FINISH_CANCELLED_V1);
    /* An old request stays alive while a new stream runs: stale cancellation
     * must have no path to the new request's native control token. */
    atomic_store(&si_mock_arrived, 0);
    atomic_store(&si_mock_resume, 0);
    CHECK(pthread_create(&t, NULL, run, &second) == 0);
    while (!atomic_load(&si_mock_arrived))
        sched_yield();
    CHECK(api->request_cancel(r[0], &disposition, NULL) == SQ_OK &&
          disposition == SQ_CANCEL_TOO_LATE);
    atomic_store(&si_mock_resume, 1);
    CHECK(pthread_join(t, NULL) == 0 && second.status == SQ_OK &&
          second.events == rc.max_output_tokens);
    CHECK(api->request_result(r[1], &result, NULL) == SQ_OK && !result.cancel_requested &&
          result.generated_tokens == rc.max_output_tokens);
    CHECK(api->request_run(r[1], count_event, &second.events, NULL) == SQ_INFER_INVALID_STATE);
    atomic_store(&si_mock_pause, 0);
    unsigned calls = atomic_load(&si_mock_calls);
    CHECK(api->request_cancel(r[2], &disposition, NULL) == SQ_OK &&
          disposition == SQ_CANCEL_ACCEPTED);
    CHECK(api->request_run(r[2], count_event, &second.events, NULL) == SQ_CANCELLED);
    CHECK(atomic_load(&si_mock_calls) == calls);
    for (size_t i = 0; i < 3; i++)
        CHECK(api->request_close(&r[i], NULL) == SQ_OK && !r[i]);
    for (unsigned mode = 0; mode < 3; mode++) {
        sq_infer_request *one = NULL;
        CHECK(api->request_create(s, &input, &rc, &one, NULL) == SQ_OK);
        CHECK(api->request_run(one,
                               mode == 0   ? cancel_event
                               : mode == 1 ? stop_event
                                           : invalid_event,
                               NULL, NULL) == (mode == 0   ? SQ_CANCELLED
                                               : mode == 1 ? SQ_OK
                                                           : SQ_INVALID_ARGUMENT));
        CHECK(api->request_result(one, &result, NULL) == SQ_OK && result.generated_tokens == 1);
        CHECK(api->request_close(&one, NULL) == SQ_OK);
    }
    atomic_store(&si_mock_close_fail, 1);
    CHECK(api->session_close(&s, NULL) == SQ_GPU_ERROR && s);
    sq_infer_request *after_close = NULL;
    CHECK(api->request_create(s, &input, &rc, &after_close, NULL) == SQ_INFER_NOT_READY &&
          !after_close);
    atomic_store(&si_mock_close_fail, 0);
    CHECK(api->session_close(&s, NULL) == SQ_OK && !s);
    CHECK(api->model_close(&m, NULL) == SQ_OK && !m);
    return 0;
}

static int invalid_inputs(void) {
    sq_infer_error_v1 error;
    SQ_INFER_INIT(error);
    sq_infer_model *m = (void *)1;
    sq_infer_model_config_v1 c = mc;
    c.hdr.flags = 1;
    CHECK(api->model_create(&c, &m, &error) == SQ_INVALID_ARGUMENT && !m &&
          error.status == SQ_INVALID_ARGUMENT);
    c = mc;
    c.required_features = UINT64_C(1) << 63;
    CHECK(api->model_create(&c, &m, NULL) == SQ_UNSUPPORTED && !m);
    c = mc;
    c.allocator.release = NULL;
    CHECK(api->model_create(&c, &m, NULL) == SQ_INVALID_ARGUMENT && !m);
    c = mc;
    c.max_owned_host_bytes = 0;
    CHECK(api->model_create(&c, &m, NULL) == SQ_LIMIT && !m);

    struct {
        sq_infer_model_config_v1 c;
        uint64_t tail;
    } bigger = {mc, 1};

    bigger.c.hdr.struct_size = sizeof bigger;
    CHECK(api->model_create(&bigger.c, &m, NULL) == SQ_UNSUPPORTED && !m);
    bigger.tail = 0;
    CHECK(api->model_create(&bigger.c, &m, NULL) == SQ_OK);
    CHECK(api->model_close(&m, NULL) == SQ_OK);

    struct {
        sq_infer_backend_info_v1 info;
        uint64_t canary;
    } output = {0};

    output.info.hdr.struct_size = sizeof output;
    output.canary = UINT64_C(0xdeadbeef);
    CHECK(api->backend_info(&output.info, &error) == SQ_OK &&
          output.canary == UINT64_C(0xdeadbeef));
    CHECK(output.info.hdr.struct_size == sizeof output &&
          output.info.hdr.written_size == sizeof output.info);
    CHECK(output.info.max_active_runs == 1 && output.info.limit_scope == SQ_LIMIT_PROVIDER_PROCESS);
    output.info.hdr.struct_size = 8;
    CHECK(api->backend_info(&output.info, NULL) == SQ_INVALID_ARGUMENT);
    c = mc;
    c.max_owned_device_bytes = 0;
    CHECK(api->model_create(&c, &m, NULL) == SQ_OK);
    sq_infer_source_v1 source;
    SQ_INFER_INIT(source);
    source.directory = si_text("/not-opened-budget-preflight");
    source.format_id = si_text("inkling-safetensors");
    CHECK(api->model_load(m, &source, NULL) == SQ_LIMIT && m->state == SQ_HANDLE_CREATED);
    CHECK(api->model_close(&m, NULL) == SQ_OK);
    CHECK(ready_model(&m, NULL) == SQ_OK);
    sq_infer_session *s = NULL;
    sq_infer_session_config_v1 limited = sc;
    limited.max_owned_device_bytes = 0;
    CHECK(api->session_create(m, &limited, &s, NULL) == SQ_OK);
    CHECK(api->session_prepare(s, NULL) == SQ_LIMIT && !s->native);
    CHECK(api->session_close(&s, NULL) == SQ_OK);
    CHECK(api->session_create(m, &sc, &s, NULL) == SQ_OK && api->session_prepare(s, NULL) == SQ_OK);
    sq_infer_request *r = (void *)1;
    sq_infer_request_config_v1 bad = rc;
    bad.max_output_tokens = UINT64_MAX;
    CHECK(api->request_create(s, &input, &bad, &r, NULL) == SQ_INVALID_ARGUMENT && !r);
    bad = rc;
    bad.extensions.count = UINT64_MAX;
    bad.extensions.stride = 1;
    CHECK(api->request_create(s, &input, &bad, &r, NULL) == SQ_INVALID_ARGUMENT && !r);
    sq_infer_input_v1 malformed = input;
    malformed.kind = SQ_INPUT_CHAT;
    malformed.tokens = NULL;
    malformed.token_count = 0;
    malformed.messages = (sq_infer_array){prompt, UINT64_MAX, 1, 0};
    CHECK(api->request_create(s, &malformed, &rc, &r, NULL) == SQ_INVALID_ARGUMENT && !r);
    CHECK(api->session_close(&s, NULL) == SQ_OK && api->model_close(&m, NULL) == SQ_OK);
    return 0;
}

static int fault_injection(void) {
    size_t baseline_allocations = allocations, baseline_releases = releases;
    for (size_t nth = 1; nth <= 8; nth++) {
        size_t before = allocations, freed_before = releases;
        fail_at = allocations + nth;
        sq_infer_model *m = NULL;
        sq_infer_session *s = NULL;
        sq_infer_request *r = NULL;
        sq_status status = ready_model(&m, NULL);
        if (!status)
            status = api->session_create(m, &sc, &s, NULL);
        if (!status)
            status = api->session_prepare(s, NULL);
        if (!status)
            status = api->request_create(s, &input, &rc, &r, NULL);
        CHECK(status == SQ_OK || status == SQ_OUT_OF_MEMORY);
        CHECK(api->request_close(&r, NULL) == SQ_OK);
        CHECK(api->session_close(&s, NULL) == SQ_OK);
        CHECK(api->model_close(&m, NULL) == SQ_OK);
        size_t attempted = allocations - before, freed = releases - freed_before;
        CHECK(freed == attempted - (attempted >= nth ? 1 : 0));
        fail_at = 0;
    }
    CHECK(allocations - baseline_allocations >= releases - baseline_releases);
    return 0;
}

static uint32_t text_event(void *ctx, const sq_infer_event_v1 *event) {
    char *out = ctx;
    size_t n = strlen(out);
    if (event->kind != SQ_EVENT_CONTENT || n + event->text.bytes >= 32)
        return 99;
    memcpy(out + n, event->text.data, (size_t)event->text.bytes);
    out[n + event->text.bytes] = 0;
    return SQ_CALLBACK_CONTINUE;
}

static int text_and_recovery(const char *fixture) {
    sq_infer_model *m = NULL;
    sq_infer_session *s = NULL;
    sq_infer_request *r = NULL;
    CHECK(ready_model(&m, fixture) == SQ_OK);
    CHECK(api->session_create(m, &sc, &s, NULL) == SQ_OK && api->session_prepare(s, NULL) == SQ_OK);
    sq_infer_input_v1 text;
    SQ_INFER_INIT(text);
    text.kind = SQ_INPUT_RAW_TEXT;
    text.text = si_text("prompt");
    sq_infer_request_config_v1 c = rc;
    c.output_kind = SQ_OUTPUT_TEXT;
    c.max_event_bytes = 4;
    char stop[] = "BA";
    sq_infer_bytes span = {stop, 2};
    c.stops = (sq_infer_array){&span, 1, sizeof span, 0};
    CHECK(api->request_create(s, &text, &c, &r, NULL) == SQ_OK);
    stop[0] = 'X'; /* caller buffers no longer authoritative */
    char result_text[32] = "";
    CHECK(api->request_run(r, text_event, result_text, NULL) == SQ_OK && !strcmp(result_text, "A"));
    sq_infer_result_v1 result;
    SQ_INFER_INIT(result);
    CHECK(api->request_result(r, &result, NULL) == SQ_OK &&
          result.finish_reason == SQ_FINISH_STOP_STRING_V1 && result.generated_tokens == 3);
    CHECK(api->request_close(&r, NULL) == SQ_OK);
    unsigned enables = atomic_load(&si_mock_enables);
    c = rc;
    c.cache_scope = si_text("tenant-a");
    for (unsigned i = 0; i < 3; i++) {
        if (i == 2)
            c.cache_scope = si_text("tenant-b");
        size_t count = 0;
        CHECK(api->request_create(s, &input, &c, &r, NULL) == SQ_OK);
        CHECK(api->request_run(r, count_event, &count, NULL) == SQ_OK);
        CHECK(api->request_close(&r, NULL) == SQ_OK);
    }
    CHECK(atomic_load(&si_mock_enables) == enables + 2);
    CHECK(api->request_create(s, &input, &rc, &r, NULL) == SQ_OK);
    atomic_store(&si_mock_pause, 2);
    atomic_store(&si_mock_arrived, 0);
    atomic_store(&si_mock_resume, 0);
    worker work = {r, 999, 0};
    pthread_t t;
    CHECK(!pthread_create(&t, NULL, run, &work));
    while (!atomic_load(&si_mock_arrived))
        sched_yield();
    uint32_t disposition;
    CHECK(api->request_cancel(r, &disposition, NULL) == SQ_OK && disposition == SQ_CANCEL_ACCEPTED);
    atomic_store(&si_mock_resume, 1);
    CHECK(!pthread_join(t, NULL) && work.status == SQ_CANCELLED);
    CHECK(api->request_close(&r, NULL) == SQ_OK);
    atomic_store(&si_mock_pause, 0);
    /* Accepted cancellation followed by a failed native reset must not report
     * successful cancellation or allow the poisoned session to run again. */
    CHECK(api->request_create(s, &input, &rc, &r, NULL) == SQ_OK);
    atomic_store(&si_mock_fail, 2);
    sq_infer_error_v1 error;
    SQ_INFER_INIT(error);
    CHECK(api->request_run(r, cancel_event, NULL, &error) == SQ_GPU_ERROR);
    CHECK(error.flags & SQ_INFER_ERR_SESSION_POISONED);
    CHECK(error.cleanup_status == SQ_GPU_ERROR);
    CHECK(api->request_close(&r, NULL) == SQ_OK);
    CHECK(api->request_create(s, &input, &rc, &r, NULL) == SQ_INFER_NOT_READY && !r);
    atomic_store(&si_mock_fail, 0);
    CHECK(api->session_close(&s, NULL) == SQ_OK && api->model_close(&m, NULL) == SQ_OK);
    return 0;
}

static int cancellation_races(void) {
    sq_infer_model *m = NULL;
    sq_infer_session *s = NULL;
    CHECK(ready_model(&m, NULL) == SQ_OK);
    CHECK(api->session_create(m, &sc, &s, NULL) == SQ_OK && api->session_prepare(s, NULL) == SQ_OK);
    for (unsigned i = 0; i < 100; i++) {
        sq_infer_request *r = NULL;
        CHECK(api->request_create(s, &input, &rc, &r, NULL) == SQ_OK);
        worker w = {r, 999, 0};
        pthread_t t;
        CHECK(!pthread_create(&t, NULL, run, &w));
        if (i & 1)
            sched_yield();
        uint32_t disposition;
        CHECK(api->request_cancel(r, &disposition, NULL) == SQ_OK);
        CHECK(!pthread_join(t, NULL));
        CHECK((disposition == SQ_CANCEL_ACCEPTED && w.status == SQ_CANCELLED) ||
              (disposition == SQ_CANCEL_TOO_LATE && w.status == SQ_OK));
        CHECK(api->request_close(&r, NULL) == SQ_OK);
    }
    CHECK(api->session_close(&s, NULL) == SQ_OK && api->model_close(&m, NULL) == SQ_OK);
    return 0;
}

static int load_fixture(const char *directory) {
    sq_infer_source_v1 source;
    SQ_INFER_INIT(source);
    source.directory = si_text(directory);
    source.format_id = si_text("inkling-safetensors");
    sq_infer_model *m = NULL;
    CHECK(api->model_create(&mc, &m, NULL) == SQ_OK);
    atomic_store(&si_mock_upload_fail, 1);
    CHECK(api->model_load(m, &source, NULL) == SQ_GPU_ERROR && m->checkpoint &&
          m->state == SQ_HANDLE_FAILED);
    atomic_store(&si_mock_close_fail, 1);
    CHECK(api->model_close(&m, NULL) == SQ_GPU_ERROR && m && m->checkpoint);
    CHECK(sq_inkling_checkpoint_count(m->checkpoint) == 822);
    atomic_store(&si_mock_close_fail, 0);
    atomic_store(&si_mock_upload_fail, 0);
    CHECK(api->model_close(&m, NULL) == SQ_OK && !m);
    size_t calls = allocations;
    CHECK(api->model_create(&mc, &m, NULL) == SQ_OK && api->model_load(m, &source, NULL) == SQ_OK);
    CHECK(!m->checkpoint && m->state == SQ_HANDLE_READY);
    size_t count = allocations - calls;
    CHECK(api->model_close(&m, NULL) == SQ_OK);
    for (size_t i = 1; i <= count; i++) {
        size_t start = allocations, freed = releases;
        fail_at = allocations + i;
        sq_status status = api->model_create(&mc, &m, NULL);
        if (!status)
            status = api->model_load(m, &source, NULL);
        CHECK(status == SQ_OUT_OF_MEMORY);
        CHECK(api->model_close(&m, NULL) == SQ_OK && !m);
        CHECK(releases - freed == allocations - start - 1);
        fail_at = 0;
    }
    puts("facade model-load allocation faults and retained checkpoint cleanup passed (sparse "
         "metadata, mock upload)");
    return 0;
}

int main(int argc, char **argv) {
    const void *table = (void *)1;
    CHECK(sq_infer_get_api(999, 0, &table) == SQ_INFER_ABI_MISMATCH && !table);
    CHECK(sq_infer_get_api(1, 1, &table) == SQ_INFER_ABI_MISMATCH && !table);
    CHECK(sq_infer_get_api(1, 0, &table) == SQ_OK && table);
    api = table;
    CHECK(api->struct_size == sizeof *api && api->abi_major == 1);
    SQ_INFER_INIT(mc);
    SQ_INFER_INIT(sc);
    SQ_INFER_INIT(rc);
    SQ_INFER_INIT(input);
    CHECK(api->defaults(SQ_DEFAULT_MODEL, &mc, NULL) == SQ_OK);
    CHECK(api->defaults(SQ_DEFAULT_SESSION, &sc, NULL) == SQ_OK);
    CHECK(api->defaults(SQ_DEFAULT_REQUEST, &rc, NULL) == SQ_OK);
    mc.allocator = (sq_infer_allocator){NULL, allocate, release};
    if (argc == 3 && !strcmp(argv[1], "--load-fixture"))
        return load_fixture(argv[2]);
    rc.max_output_tokens = 8;
    rc.output_kind = SQ_OUTPUT_RAW_TOKENS;
    input.kind = SQ_INPUT_TOKENS;
    input.tokens = prompt;
    input.token_count = 2;
    CHECK(invalid_inputs() == 0);
    CHECK(lifecycle(argc > 1 ? argv[1] : NULL) == 0);
    CHECK(fault_injection() == 0);
    if (argc > 1)
        CHECK(text_and_recovery(argv[1]) == 0);
    CHECK(cancellation_races() == 0);
    puts("inference ABI negotiation, ownership, independent streams, stale cancel and allocation "
         "faults passed (mock execution)");
    return 0;
}
