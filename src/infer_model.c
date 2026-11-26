/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "infer_internal.h"
#include <stdio.h>

static uint64_t features(const sq_infer_model *m) {
    return m->weights_config.with_mtp ? SI_FEATURES : SI_FEATURES & ~SQ_INFER_CAP_SPECULATION;
}

static sq_status source_paths(const sq_infer_source_v1 *s, char paths[3][4096]) {
    sq_status rc = si_input(s, sizeof *s);
    if (rc)
        return rc;
    if (!si_equal(s->format_id, "inkling-safetensors") || !si_span(s->directory, 4000) ||
        !s->directory.bytes || memchr(s->directory.data, 0, (size_t)s->directory.bytes))
        return SQ_INVALID_ARGUMENT;
    const char *names[] = {"tokenizer.json", "model-00001-of-00001.safetensors", "mtp.safetensors"};
    for (size_t i = 0; i < 3; i++) {
        int n = snprintf(paths[i], 4096, "%.*s/%s", (int)s->directory.bytes,
                         (const char *)s->directory.data, names[i]);
        if (n < 0 || n >= 4096)
            return SQ_LIMIT;
    }
    return SQ_OK;
}

sq_status si_model_create(const sq_infer_model_config_v1 *c, sq_infer_model **out,
                          sq_infer_error_v1 *e) {
    if (out)
        *out = NULL;
    sq_status rc = si_input(c, sizeof *c);
    if (rc || !out)
        return si_error(e, rc ? rc : SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE,
                        "invalid model configuration");
    rc = si_no_extensions(c->extensions);
    if (rc || (c->required_features & ~SI_FEATURES))
        return si_error(e, rc ? rc : SQ_UNSUPPORTED, SQ_STAGE_VALIDATE,
                        "unsupported model features");
    sq_options a = SQ_OPTIONS_INIT;
    a.context = c->allocator.context;
    a.allocate = c->allocator.allocate;
    a.release = c->allocator.release;
    sq_options resolved;
    rc = sq_options_resolve(&a, &resolved);
    a = resolved;
    if (rc)
        return si_native_error(e, rc, SQ_STAGE_VALIDATE);
    if (c->max_owned_host_bytes < sizeof(sq_infer_model))
        return si_error(e, SQ_LIMIT, SQ_STAGE_VALIDATE, "model host budget too small");
    sq_weight_config wc = SQ_WEIGHT_CONFIG_INIT;
    wc.with_mtp = 0;
    const char *profiles[] = {"exact-ar",  "fp8-ar",  "nvfp4-ar",
                              "exact-mtp", "fp8-mtp", "nvfp4-mtp"};
    const char *profile = NULL;
    for (uint32_t i = 0; i < 6; i++)
        if (si_equal(c->weight_profile, profiles[i])) {
            wc.mode = i % 3;
            wc.with_mtp = i / 3;
            profile = profiles[i];
        }
    if (!profile || !si_span(c->device_id, 10) || !c->device_id.bytes)
        return si_error(e, SQ_UNSUPPORTED, SQ_STAGE_VALIDATE,
                        "unknown weight profile or device identifier");
    uint32_t device = 0;
    for (size_t i = 0; i < c->device_id.bytes; i++) {
        unsigned char ch = ((const unsigned char *)c->device_id.data)[i];
        if (ch < '0' || ch > '9' || device > 214748364u || (device == 214748364u && ch > '7'))
            return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE,
                            "device must be a nonnegative CUDA ordinal");
        device = device * 10 + (uint32_t)(ch - '0');
    }
    wc.device = (int32_t)device;
    if (!wc.with_mtp && (c->required_features & SQ_INFER_CAP_SPECULATION))
        return si_error(e, SQ_UNSUPPORTED, SQ_STAGE_VALIDATE,
                        "speculation requires an MTP profile");
    sq_infer_model *m = si_alloc(&a, sizeof *m);
    if (!m)
        return si_error(e, SQ_OUT_OF_MEMORY, SQ_STAGE_VALIDATE, "allocate model");
    memset(m, 0, sizeof *m);
    si_budget_init(&m->budget, &a, c->max_owned_host_bytes - sizeof *m, &m->allocator);
    atomic_flag_clear(&m->busy);
    (void)snprintf(m->device_id, sizeof m->device_id, "%u", device);
    m->weights_config = wc;
    m->profile = profile;
    m->required = c->required_features;
    m->device_budget = c->max_owned_device_bytes;
    *out = m;
    return si_error(e, SQ_OK, 0, NULL);
}

sq_status si_model_plan(sq_infer_model *m, const sq_infer_source_v1 *s,
                        sq_infer_memory_plan_v1 *out, sq_infer_error_v1 *e) {
    if (!m)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE, "missing model");
    if (!si_lock(&m->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_VALIDATE, "model busy");
    char paths[3][4096];
    sq_status rc = source_paths(s, paths);
    sq_weight_plan plan = {0};
    if (!rc)
        rc = sq_weights_plan(&m->weights_config, &m->allocator, &plan);
    if (!rc) {
        sq_infer_memory_plan_v1 v = {0};
        v.owned_host_bytes = sizeof *m + atomic_load(&m->budget.live);
        v.owned_device_bytes = plan.arena_bytes;
        v.peak_owned_device_bytes = plan.peak_device_bytes;
        /* Tokenizer/maps and vendor allocations are not known before loading. */
        v.unknown_fields = UINT64_C(3) | SI_UNKNOWN_VENDOR;
        rc = si_output(out, &v, sizeof v);
    }
    si_unlock(&m->busy);
    return si_native_error(e, rc, SQ_STAGE_VALIDATE);
}

sq_status si_model_load(sq_infer_model *m, const sq_infer_source_v1 *s, sq_infer_error_v1 *e) {
    if (!m)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_LOAD, "missing model");
    if (!si_lock(&m->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_LOAD, "model busy");
    sq_status rc = m->state == SQ_HANDLE_CREATED ? SQ_OK : SQ_INFER_INVALID_STATE;
    char paths[3][4096];
    if (!rc)
        rc = source_paths(s, paths);
    sq_weight_plan plan;
    if (!rc)
        rc = sq_weights_plan(&m->weights_config, &m->allocator, &plan);
    if (!rc && plan.peak_device_bytes > m->device_budget)
        rc = SQ_LIMIT;
    if (rc) {
        si_unlock(&m->busy);
        return si_native_error(e, rc, SQ_STAGE_LOAD);
    }
    m->state = SQ_HANDLE_PREPARING;
    rc = sq_tokenizer_open(paths[0], &m->allocator, &m->tokenizer);
    if (!rc)
        rc = sq_inkling_checkpoint_open(paths[1], m->weights_config.with_mtp ? paths[2] : NULL,
                                        &m->allocator, &m->checkpoint);
    if (!rc)
        rc = sq_gpu_weights_create(&m->weights_config, &m->allocator, &m->weights);
    if (!rc)
        rc = sq_gpu_weights_load(m->weights, m->checkpoint);
    if (!rc) {
        sq_inkling_checkpoint_close(m->checkpoint);
        m->checkpoint = NULL;
    }
    /* On failure retain mapped input until native cleanup has succeeded. Even
     * a failed stream synchronization must not shorten borrowed DMA lifetime. */
    m->state = rc ? SQ_HANDLE_FAILED : SQ_HANDLE_READY;
    si_unlock(&m->busy);
    return si_native_error(e, rc, SQ_STAGE_LOAD);
}

sq_status si_model_info(sq_infer_model *m, sq_infer_model_info_v1 *out, sq_infer_error_v1 *e) {
    if (!m)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_VALIDATE, "missing model");
    if (!si_lock(&m->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_VALIDATE, "model busy");
    sq_infer_model_info_v1 v = {0};
    v.state = m->state;
    v.max_prepared_sessions = 1;
    v.supported_features = features(m);
    v.enabled_features = m->state == SQ_HANDLE_READY ? features(m) & ~SQ_INFER_CAP_PREFIX_CACHE : 0;
    v.max_context_tokens = 1048576;
    v.max_messages = 1024;
    v.max_input_bytes = SI_MAX_INPUT;
    v.max_stops = 8;
    v.max_stop_bytes = 64;
    v.max_event_bytes = 65536;
    v.model_id = si_text("inkling-2-mini");
    v.resolved_weight_profile = si_text(m->profile);
    v.resolved_device_id = si_text(m->device_id);
    v.tokenizer_id = si_text("inkling-bpe");
    v.template_id = si_text("inkling-text-v1");
    /* No invented artifact hash: fingerprint_verified remains false. */
    sq_status rc = si_output(out, &v, sizeof v);
    si_unlock(&m->busy);
    return si_native_error(e, rc, SQ_STAGE_VALIDATE);
}

sq_status si_model_close(sq_infer_model **ptr, sq_infer_error_v1 *e) {
    if (!ptr)
        return si_error(e, SQ_INVALID_ARGUMENT, SQ_STAGE_CLOSE, "missing handle pointer");
    sq_infer_model *m = *ptr;
    if (!m)
        return si_error(e, SQ_OK, 0, NULL);
    if (!si_lock(&m->busy))
        return si_error(e, SQ_BUSY, SQ_STAGE_CLOSE, "model busy");
    if (m->children) {
        si_unlock(&m->busy);
        return si_error(e, SQ_BUSY, SQ_STAGE_CLOSE, "model has sessions");
    }
    m->state = SQ_HANDLE_CLOSING;
    sq_status rc = sq_gpu_weights_destroy(&m->weights);
    if (rc) {
        si_unlock(&m->busy);
        return si_native_error(e, rc, SQ_STAGE_CLOSE);
    }
    sq_inkling_checkpoint_close(m->checkpoint);
    sq_tokenizer_close(m->tokenizer);
    sq_options a = m->budget.upstream;
    si_free(&a, m);
    *ptr = NULL;
    return si_error(e, SQ_OK, 0, NULL);
}
