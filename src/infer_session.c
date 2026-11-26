/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "infer_internal.h"

static sq_status plan_locked(sq_infer_session *s, sq_infer_memory_plan_v1 *v) {
    uint64_t bytes = 0;
    sq_status rc = sq_session_owned_plan(s->model->weights, &s->config, &s->allocator, &bytes);
    if (rc)
        return rc;
    memset(v, 0, sizeof *v);
    v->owned_host_bytes = sizeof *s + atomic_load(&s->budget.live);
    v->owned_device_bytes = v->peak_owned_device_bytes = bytes;
    v->unknown_fields = UINT64_C(1) | SI_UNKNOWN_VENDOR;
    return SQ_OK;
}

sq_status si_session_create(sq_infer_model *m, const sq_infer_session_config_v1 *c,
                            sq_infer_session **out, sq_infer_error_v1 *e) {
    if (out)
        *out = NULL;
    sq_status rc = si_input(c, sizeof *c);
    if (rc || !out || !m)
        return si_error(e, rc ? rc : SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE,
                        "invalid session arguments");
    rc = si_no_extensions(c->extensions);
    if (rc)
        return si_native_error(e, rc, SQ_STAGE_VALIDATE);
    sq_session_config sc = SQ_SESSION_CONFIG_INIT;
    if (c->context_tokens < 64 || c->context_tokens > 1048576 ||
        c->max_owned_host_bytes < sizeof(sq_infer_session))
        return si_error(e, SQ_LIMIT, SQ_STAGE_VALIDATE, "session context or host budget invalid");
    sc.context_tokens = (uint32_t)c->context_tokens;
    const char *profiles[] = {"bf16-eager", "fp8-eager", "q4-eager",
                              "bf16-graph", "fp8-graph", "q4-graph"};
    const char *profile = NULL;
    for (uint32_t i = 0; i < 6; i++)
        if (si_equal(c->execution_profile, profiles[i])) {
            sc.kv_format = i % 3;
            sc.use_graphs = i / 3;
            profile = profiles[i];
        }
    if (!profile)
        return si_error(e, SQ_UNSUPPORTED, SQ_STAGE_VALIDATE, "unknown execution profile");
    if (!si_lock(&m->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_VALIDATE, "model busy");
    if (m->state != SQ_HANDLE_READY || m->children == UINT32_MAX) {
        si_unlock(&m->busy);
        return si_error(e, SQ_INFER_NOT_READY, SQ_STAGE_VALIDATE, "model not ready");
    }
    sq_infer_session *s = si_alloc(&m->allocator, sizeof *s);
    if (!s) {
        si_unlock(&m->busy);
        return si_error(e, SQ_OUT_OF_MEMORY, SQ_STAGE_VALIDATE, "allocate session");
    }
    memset(s, 0, sizeof *s);
    s->model = m;
    s->config = sc;
    s->profile = profile;
    s->device_budget = c->max_owned_device_bytes;
    si_budget_init(&s->budget, &m->allocator, c->max_owned_host_bytes - sizeof *s, &s->allocator);
    atomic_flag_clear(&s->busy);
    m->children++;
    si_unlock(&m->busy);
    *out = s;
    return si_error(e, SQ_OK, 0, NULL);
}

sq_status si_session_plan(sq_infer_session *s, sq_infer_memory_plan_v1 *out, sq_infer_error_v1 *e) {
    if (!s)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE, "missing session");
    if (!si_lock(&s->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_VALIDATE, "session busy");
    sq_infer_memory_plan_v1 v;
    sq_status rc = plan_locked(s, &v);
    if (!rc)
        rc = si_output(out, &v, sizeof v);
    si_unlock(&s->busy);
    return si_native_error(e, rc, SQ_STAGE_VALIDATE);
}

sq_status si_session_prepare(sq_infer_session *s, sq_infer_error_v1 *e) {
    if (!s)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_PREPARE, "missing session");
    if (!si_lock(&s->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_PREPARE, "session busy");
    sq_status rc = s->state == SQ_HANDLE_CREATED ? SQ_OK : SQ_INFER_INVALID_STATE;
    sq_infer_memory_plan_v1 v;
    if (!rc)
        rc = plan_locked(s, &v);
    if (!rc && v.peak_owned_device_bytes > s->device_budget)
        rc = SQ_LIMIT;
    if (!rc) {
        s->state = SQ_HANDLE_PREPARING;
        rc = sq_session_create(s->model->weights, &s->config, &s->allocator, &s->native);
        s->state = rc ? ((rc == SQ_BUSY && !s->native) ? SQ_HANDLE_CREATED : SQ_HANDLE_FAILED)
                      : SQ_HANDLE_READY;
    }
    si_unlock(&s->busy);
    return si_native_error(e, rc, SQ_STAGE_PREPARE);
}

sq_status si_session_info(sq_infer_session *s, sq_infer_session_info_v1 *out,
                          sq_infer_error_v1 *e) {
    if (!s)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE, "missing session");
    if (!si_lock(&s->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_VALIDATE, "session busy");
    sq_infer_session_info_v1 v = {0};
    v.state = s->state;
    v.max_active_runs = 1;
    v.context_tokens = s->config.context_tokens;
    v.enabled_features = s->state == SQ_HANDLE_READY ? SI_FEATURES & ~SQ_INFER_CAP_PREFIX_CACHE : 0;
    if (!s->model->weights_config.with_mtp)
        v.enabled_features &= ~SQ_INFER_CAP_SPECULATION;
    if (s->cache_scope_bytes)
        v.enabled_features |= SQ_INFER_CAP_PREFIX_CACHE;
    v.resolved_execution_profile = si_text(s->profile);
    sq_status rc = si_output(out, &v, sizeof v);
    si_unlock(&s->busy);
    return si_native_error(e, rc, SQ_STAGE_VALIDATE);
}

sq_status si_session_clear(sq_infer_session *s, sq_infer_error_v1 *e) {
    if (!s)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE, "missing session");
    if (!si_lock(&s->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_VALIDATE, "session busy");
    sq_status rc =
        s->state == SQ_HANDLE_READY ? sq_session_prefix_cache(s->native, 0) : SQ_INFER_NOT_READY;
    if (!rc)
        s->cache_scope_bytes = 0;
    si_unlock(&s->busy);
    return si_native_error(e, rc, SQ_STAGE_VALIDATE);
}

sq_status si_session_close(sq_infer_session **ptr, sq_infer_error_v1 *e) {
    if (!ptr)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_CLOSE, "missing handle pointer");
    sq_infer_session *s = *ptr;
    if (!s)
        return si_error(e, SQ_OK, 0, NULL);
    if (!si_lock(&s->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_CLOSE, "session busy");
    if (s->children) {
        si_unlock(&s->busy);
        return si_error(e, SQ_BUSY, SQ_STAGE_CLOSE, "session has requests");
    }
    sq_infer_model *m = s->model;
    if (!si_lock(&m->busy)) {
        si_unlock(&s->busy);
        return si_error(e, SQ_BUSY, SQ_STAGE_CLOSE, "model busy");
    }
    s->state = SQ_HANDLE_CLOSING;
    sq_status rc = sq_session_destroy(&s->native);
    if (rc) {
        si_unlock(&m->busy);
        si_unlock(&s->busy);
        return si_native_error(e, rc, SQ_STAGE_CLOSE);
    }
    m->children--;
    si_free(&m->allocator, s);
    *ptr = NULL;
    si_unlock(&m->busy);
    return si_error(e, SQ_OK, 0, NULL);
}
