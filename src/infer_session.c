/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "infer_internal.h"

static ifn_status ifn_infer_session_local_plan_locked(ifn_infer_session *s,
                                                      ifn_infer_memory_plan_v1 *v) {
    uint64_t bytes = 0;
    ifn_status rc = ifn_session_owned_plan(s->model->weights, &s->config, &s->allocator, &bytes);
    if (rc)
        return rc;
    memset(v, 0, sizeof *v);
    v->owned_host_bytes = sizeof *s + atomic_load(&s->budget.live);
    v->owned_device_bytes = v->peak_owned_device_bytes = bytes;
    v->unknown_fields = UINT64_C(1) | IFN_INFER_UNKNOWN_VENDOR;
    return IFN_OK;
}

ifn_status ifn_infer_session_create(ifn_infer_model *m, const ifn_infer_session_config_v1 *c,
                                    ifn_infer_session **out, ifn_infer_error_v1 *e) {
    if (out)
        *out = NULL;
    ifn_status rc = ifn_infer_input(c, sizeof *c);
    if (rc || !out || !m)
        return ifn_infer_error(e, rc ? rc : IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE,
                               "invalid session arguments");
    rc = ifn_infer_no_extensions(c->extensions);
    if (rc)
        return ifn_infer_native_error(e, rc, IFN_STAGE_VALIDATE);
    ifn_session_config sc = IFN_SESSION_CONFIG_INIT;
    if (c->context_tokens < 64 || c->context_tokens > 1048576 ||
        c->max_owned_host_bytes < sizeof(ifn_infer_session))
        return ifn_infer_error(e, IFN_LIMIT, IFN_STAGE_VALIDATE,
                               "session context or host budget invalid");
    sc.context_tokens = (uint32_t)c->context_tokens;
    const char *profiles[] = {"bf16-eager", "fp8-eager", "q4-eager",
                              "bf16-graph", "fp8-graph", "q4-graph"};
    const char *profile = NULL;
    for (uint32_t i = 0; i < 6; i++)
        if (ifn_infer_equal(c->execution_profile, profiles[i])) {
            sc.kv_format = i % 3;
            sc.use_graphs = i / 3;
            profile = profiles[i];
        }
    if (!profile)
        return ifn_infer_error(e, IFN_UNSUPPORTED, IFN_STAGE_VALIDATE, "unknown execution profile");
    if (!ifn_infer_lock(&m->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_VALIDATE, "model busy");
    if (m->state != IFN_HANDLE_READY || m->children == UINT32_MAX) {
        ifn_infer_unlock(&m->busy);
        return ifn_infer_error(e, IFN_INFER_NOT_READY, IFN_STAGE_VALIDATE, "model not ready");
    }
    ifn_infer_session *s = ifn_infer_alloc(&m->allocator, sizeof *s);
    if (!s) {
        ifn_infer_unlock(&m->busy);
        return ifn_infer_error(e, IFN_OUT_OF_MEMORY, IFN_STAGE_VALIDATE, "allocate session");
    }
    memset(s, 0, sizeof *s);
    s->model = m;
    s->config = sc;
    s->profile = profile;
    s->device_budget = c->max_owned_device_bytes;
    ifn_infer_budget_init(&s->budget, &m->allocator, c->max_owned_host_bytes - sizeof *s,
                          &s->allocator);
    atomic_flag_clear(&s->busy);
    m->children++;
    ifn_infer_unlock(&m->busy);
    *out = s;
    return ifn_infer_error(e, IFN_OK, 0, NULL);
}

ifn_status ifn_infer_session_plan(ifn_infer_session *s, ifn_infer_memory_plan_v1 *out,
                                  ifn_infer_error_v1 *e) {
    if (!s)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE, "missing session");
    if (!ifn_infer_lock(&s->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_VALIDATE, "session busy");
    ifn_infer_memory_plan_v1 v = {0};
    ifn_status rc = ifn_infer_session_local_plan_locked(s, &v);
    if (!rc)
        rc = ifn_infer_output(out, &v, sizeof v);
    ifn_infer_unlock(&s->busy);
    return ifn_infer_model_native_error(s->model, e, rc, IFN_STAGE_VALIDATE);
}

ifn_status ifn_infer_session_prepare(ifn_infer_session *s, ifn_infer_error_v1 *e) {
    if (!s)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_PREPARE, "missing session");
    if (!ifn_infer_lock(&s->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_PREPARE, "session busy");
    ifn_status rc = s->state == IFN_HANDLE_CREATED ? IFN_OK : IFN_INFER_INVALID_STATE;
    ifn_infer_memory_plan_v1 v = {0};
    if (!rc)
        rc = ifn_infer_session_local_plan_locked(s, &v);
    if (!rc && v.peak_owned_device_bytes > s->device_budget)
        rc = IFN_LIMIT;
    if (!rc) {
        s->state = IFN_HANDLE_PREPARING;
        rc = ifn_session_create(s->model->weights, &s->config, &s->allocator, &s->native);
        s->state = rc ? ((rc == IFN_BUSY && !s->native) ? IFN_HANDLE_CREATED : IFN_HANDLE_FAILED)
                      : IFN_HANDLE_READY;
    }
    ifn_infer_unlock(&s->busy);
    return ifn_infer_model_native_error(s->model, e, rc, IFN_STAGE_PREPARE);
}

ifn_status ifn_infer_session_info(ifn_infer_session *s, ifn_infer_session_info_v1 *out,
                                  ifn_infer_error_v1 *e) {
    if (!s)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE, "missing session");
    if (!ifn_infer_lock(&s->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_VALIDATE, "session busy");
    ifn_infer_session_info_v1 v = {0};
    v.state = s->state;
    v.max_active_runs = 1;
    v.context_tokens = s->config.context_tokens;
    v.enabled_features =
        s->state == IFN_HANDLE_READY ? IFN_INFER_FEATURES & ~IFN_INFER_CAP_PREFIX_CACHE : 0;
    if (!s->model->weights_config.with_mtp)
        v.enabled_features &= ~IFN_INFER_CAP_SPECULATION;
    if (s->cache_scope_bytes)
        v.enabled_features |= IFN_INFER_CAP_PREFIX_CACHE;
    if (!s->model->verify_artifacts)
        v.enabled_features &= ~IFN_INFER_CAP_ARTIFACT_VERIFY;
    v.resolved_execution_profile = ifn_infer_text(s->profile);
    ifn_status rc = ifn_infer_output(out, &v, sizeof v);
    ifn_infer_unlock(&s->busy);
    return ifn_infer_model_native_error(s->model, e, rc, IFN_STAGE_VALIDATE);
}

ifn_status ifn_infer_session_clear(ifn_infer_session *s, ifn_infer_error_v1 *e) {
    if (!s)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE, "missing session");
    if (!ifn_infer_lock(&s->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_VALIDATE, "session busy");
    ifn_status rc =
        s->state == IFN_HANDLE_READY ? ifn_session_prefix_cache(s->native, 0) : IFN_INFER_NOT_READY;
    if (!rc)
        s->cache_scope_bytes = 0;
    ifn_infer_unlock(&s->busy);
    return ifn_infer_model_native_error(s->model, e, rc, IFN_STAGE_VALIDATE);
}

ifn_status ifn_infer_session_close(ifn_infer_session **ptr, ifn_infer_error_v1 *e) {
    if (!ptr)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_CLOSE, "missing handle pointer");
    ifn_infer_session *s = *ptr;
    if (!s)
        return ifn_infer_error(e, IFN_OK, 0, NULL);
    if (!ifn_infer_lock(&s->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_CLOSE, "session busy");
    if (s->children) {
        ifn_infer_unlock(&s->busy);
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_CLOSE, "session has requests");
    }
    ifn_infer_model *m = s->model;
    if (!ifn_infer_lock(&m->busy)) {
        ifn_infer_unlock(&s->busy);
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_CLOSE, "model busy");
    }
    s->state = IFN_HANDLE_CLOSING;
    ifn_status rc = ifn_session_destroy(&s->native);
    if (rc) {
        ifn_infer_unlock(&m->busy);
        ifn_infer_unlock(&s->busy);
        return ifn_infer_model_native_error(s->model, e, rc, IFN_STAGE_CLOSE);
    }
    m->children--;
    ifn_infer_free(&m->allocator, s);
    *ptr = NULL;
    ifn_infer_unlock(&m->busy);
    return ifn_infer_error(e, IFN_OK, 0, NULL);
}
