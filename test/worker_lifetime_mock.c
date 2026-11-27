/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Public-table test double: proves the qualification client rejects broken behavior,
 * not that CUDA is correct. No private headers or inference implementation. */
#define _POSIX_C_SOURCE 200809L
#include <ifn_infer.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <time.h>

struct ifn_infer_model {
    int unused;
};

struct ifn_infer_session {
    char scope[8];
    int recovered;
};

struct ifn_infer_request {
    ifn_infer_session *session;
    pthread_t owner;
    char scope[8];
    int32_t token;
    atomic_uint cancelled, state;
    ifn_infer_result_v1 result;
};

static int ifn_worker_mock_mode(const char *name) {
    const char *mode = getenv("INKLEBERRY_WORKER_MOCK_FAULT");
    return mode && !strcmp(mode, name);
}

static ifn_status ifn_worker_mock_defaults(uint32_t kind, void *record, ifn_infer_error_v1 *e) {
    (void)kind;
    (void)record;
    (void)e;
    return IFN_OK;
}

static ifn_status ifn_worker_mock_model(const ifn_infer_model_config_v1 *c, ifn_infer_model **m,
                                        ifn_infer_error_v1 *e) {
    (void)c;
    (void)e;
    *m = calloc(1, sizeof **m);
    return *m ? IFN_OK : IFN_OUT_OF_MEMORY;
}

static ifn_status ifn_worker_mock_load(ifn_infer_model *m, const ifn_infer_source_v1 *s,
                                       ifn_infer_error_v1 *e) {
    (void)m;
    (void)s;
    (void)e;
    return IFN_OK;
}

static ifn_status ifn_worker_mock_model_close(ifn_infer_model **m, ifn_infer_error_v1 *e) {
    (void)e;
    free(*m);
    *m = NULL;
    return IFN_OK;
}

static ifn_status ifn_worker_mock_session(ifn_infer_model *m, const ifn_infer_session_config_v1 *c,
                                          ifn_infer_session **s, ifn_infer_error_v1 *e) {
    (void)m;
    (void)c;
    (void)e;
    *s = calloc(1, sizeof **s);
    return *s ? IFN_OK : IFN_OUT_OF_MEMORY;
}

static ifn_status ifn_worker_mock_prepare(ifn_infer_session *s, ifn_infer_error_v1 *e) {
    (void)s;
    (void)e;
    return IFN_OK;
}

static ifn_status ifn_worker_mock_session_close(ifn_infer_session **s, ifn_infer_error_v1 *e) {
    (void)e;
    free(*s);
    *s = NULL;
    return IFN_OK;
}

static ifn_status ifn_worker_mock_request(ifn_infer_session *s, const ifn_infer_input_v1 *in,
                                          const ifn_infer_request_config_v1 *c,
                                          ifn_infer_request **r, ifn_infer_error_v1 *e) {
    (void)e;
    *r = calloc(1, sizeof **r);
    if (!*r)
        return IFN_OUT_OF_MEMORY;
    (*r)->session = s;
    (*r)->owner = pthread_self();
    if (c->cache_scope.bytes)
        memcpy((*r)->scope, c->cache_scope.data, (size_t)c->cache_scope.bytes);
    const ifn_infer_message_v1 *messages = in->messages.data;
    (*r)->token = ((const char *)messages[1].content.data)[9];
    atomic_init(&(*r)->state, IFN_REQUEST_PREPARED);
    atomic_init(&(*r)->cancelled, 0);
    return IFN_OK;
}

static ifn_status ifn_worker_mock_run(ifn_infer_request *r, ifn_infer_emit_fn emit, void *ctx,
                                      ifn_infer_error_v1 *e) {
    (void)e;
    atomic_store(&r->state, IFN_REQUEST_RUNNING);
    /* The threaded mock waits for the controller instead of relying on timing. */
    if (!pthread_equal(r->owner, pthread_self()))
        while (!atomic_load(&r->cancelled))
            sched_yield();
    r->result.prefix_reused_tokens = *r->scope && !strcmp(r->scope, r->session->scope) ? 128 : 0;
    if (*r->scope && ifn_worker_mock_mode("cross-scope"))
        r->result.prefix_reused_tokens = 128;
    strcpy(r->session->scope, r->scope);
    ifn_infer_event_v1 event;
    IFN_INFER_INIT(event);
    event.kind = IFN_EVENT_TOKENS;
    int32_t token = r->token;
    if (r->session->recovered && ifn_worker_mock_mode("corruption"))
        token++;
    event.tokens = &token;
    event.token_count = 1;
    if (!atomic_load(&r->cancelled) && emit(ctx, &event) == IFN_CALLBACK_CANCEL)
        atomic_store(&r->cancelled, 1);
    ifn_status status = atomic_load(&r->cancelled) ? IFN_CANCELLED : IFN_OK;
    if (status == IFN_CANCELLED) {
        r->session->scope[0] = 0;
        r->session->recovered = 1;
    }
    if (ifn_worker_mock_mode("reject"))
        status = IFN_GPU_ERROR;
    if (ifn_worker_mock_mode("ignore-cancel"))
        status = IFN_OK;
    r->result.status = status;
    r->result.finish_reason = status == IFN_CANCELLED ? IFN_FINISH_CANCELLED_V1 : IFN_FINISH_EOS_V1;
    r->result.run_ns = 1000000;
    atomic_store(&r->state, IFN_REQUEST_COMPLETED);
    return status;
}

static ifn_status ifn_worker_mock_cancel(ifn_infer_request *r, uint32_t *d, ifn_infer_error_v1 *e) {
    (void)e;
    *d =
        atomic_load(&r->state) == IFN_REQUEST_COMPLETED ? IFN_CANCEL_TOO_LATE : IFN_CANCEL_ACCEPTED;
    if (*d == IFN_CANCEL_ACCEPTED)
        atomic_store(&r->cancelled, 1);
    return IFN_OK;
}

static ifn_status ifn_worker_mock_progress(const ifn_infer_request *r, ifn_infer_progress_v1 *p,
                                           ifn_infer_error_v1 *e) {
    (void)e;
    p->state = atomic_load(&r->state);
    return IFN_OK;
}

static ifn_status ifn_worker_mock_result(const ifn_infer_request *r, ifn_infer_result_v1 *out,
                                         ifn_infer_error_v1 *e) {
    (void)e;
    *out = r->result;
    return IFN_OK;
}

static ifn_status ifn_worker_mock_close(ifn_infer_request **r, ifn_infer_error_v1 *e) {
    (void)e;
    free(*r);
    *r = NULL;
    return IFN_OK;
}

ifn_status ifn_infer_get_api(uint32_t major, uint32_t minor, const void **out) {
    (void)major;
    (void)minor;
    static const ifn_infer_api_v1 table = {.defaults = ifn_worker_mock_defaults,
                                           .model_create = ifn_worker_mock_model,
                                           .model_load = ifn_worker_mock_load,
                                           .model_close = ifn_worker_mock_model_close,
                                           .session_create = ifn_worker_mock_session,
                                           .session_prepare = ifn_worker_mock_prepare,
                                           .session_close = ifn_worker_mock_session_close,
                                           .request_create = ifn_worker_mock_request,
                                           .request_run = ifn_worker_mock_run,
                                           .request_cancel = ifn_worker_mock_cancel,
                                           .request_progress = ifn_worker_mock_progress,
                                           .request_result = ifn_worker_mock_result,
                                           .request_close = ifn_worker_mock_close};
    *out = &table;
    return IFN_OK;
}
