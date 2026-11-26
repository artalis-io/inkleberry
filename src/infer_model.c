/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "infer_internal.h"
#include <stdio.h>

static uint64_t features(const ifn_infer_model *m) {
    return m->weights_config.with_mtp ? SI_FEATURES : SI_FEATURES & ~IFN_INFER_CAP_SPECULATION;
}

static ifn_status source_paths(const ifn_infer_source_v1 *s, char paths[3][4096]) {
    ifn_status rc = si_input(s, sizeof *s);
    if (rc)
        return rc;
    if (!si_equal(s->format_id, "inkling-safetensors") || !si_span(s->directory, 4000) ||
        !s->directory.bytes || memchr(s->directory.data, 0, (size_t)s->directory.bytes))
        return IFN_INVALID_ARGUMENT;
    const char *names[] = {"tokenizer.json", "model-00001-of-00001.safetensors", "mtp.safetensors"};
    for (size_t i = 0; i < 3; i++) {
        int n = snprintf(paths[i], 4096, "%.*s/%s", (int)s->directory.bytes,
                         (const char *)s->directory.data, names[i]);
        if (n < 0 || n >= 4096)
            return IFN_LIMIT;
    }
    return IFN_OK;
}

ifn_status si_model_create(const ifn_infer_model_config_v1 *c, ifn_infer_model **out,
                           ifn_infer_error_v1 *e) {
    if (out)
        *out = NULL;
    ifn_status rc = si_input(c, sizeof *c);
    if (rc || !out)
        return si_error(e, rc ? rc : IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE,
                        "invalid model configuration");
    rc = si_no_extensions(c->extensions);
    if (rc || (c->required_features & ~SI_FEATURES))
        return si_error(e, rc ? rc : IFN_UNSUPPORTED, IFN_STAGE_VALIDATE,
                        "unsupported model features");
    ifn_options a = IFN_OPTIONS_INIT;
    a.context = c->allocator.context;
    a.allocate = c->allocator.allocate;
    a.release = c->allocator.release;
    ifn_options resolved;
    rc = ifn_options_resolve(&a, &resolved);
    a = resolved;
    if (rc)
        return si_native_error(e, rc, IFN_STAGE_VALIDATE);
    if (c->max_owned_host_bytes < sizeof(ifn_infer_model))
        return si_error(e, IFN_LIMIT, IFN_STAGE_VALIDATE, "model host budget too small");
    ifn_weight_config wc = IFN_WEIGHT_CONFIG_INIT;
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
        return si_error(e, IFN_UNSUPPORTED, IFN_STAGE_VALIDATE,
                        "unknown weight profile or device identifier");
    uint32_t device = 0;
    for (size_t i = 0; i < c->device_id.bytes; i++) {
        unsigned char ch = ((const unsigned char *)c->device_id.data)[i];
        if (ch < '0' || ch > '9' || device > 214748364u || (device == 214748364u && ch > '7'))
            return si_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE,
                            "device must be a nonnegative CUDA ordinal");
        device = device * 10 + (uint32_t)(ch - '0');
    }
    wc.device = (int32_t)device;
    if (!wc.with_mtp && (c->required_features & IFN_INFER_CAP_SPECULATION))
        return si_error(e, IFN_UNSUPPORTED, IFN_STAGE_VALIDATE,
                        "speculation requires an MTP profile");
    ifn_infer_model *m = si_alloc(&a, sizeof *m);
    if (!m)
        return si_error(e, IFN_OUT_OF_MEMORY, IFN_STAGE_VALIDATE, "allocate model");
    memset(m, 0, sizeof *m);
    si_budget_init(&m->budget, &a, c->max_owned_host_bytes - sizeof *m, &m->allocator);
    atomic_flag_clear(&m->busy);
    (void)snprintf(m->device_id, sizeof m->device_id, "%u", device);
    m->weights_config = wc;
    m->profile = profile;
    m->required = c->required_features;
    m->device_budget = c->max_owned_device_bytes;
    *out = m;
    return si_error(e, IFN_OK, 0, NULL);
}

ifn_status si_model_plan(ifn_infer_model *m, const ifn_infer_source_v1 *s,
                         ifn_infer_memory_plan_v1 *out, ifn_infer_error_v1 *e) {
    if (!m)
        return si_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE, "missing model");
    if (!si_lock(&m->busy))
        return si_error(e, IFN_BUSY, IFN_STAGE_VALIDATE, "model busy");
    char paths[3][4096];
    ifn_status rc = source_paths(s, paths);
    ifn_weight_plan plan = {0};
    if (!rc)
        rc = ifn_weights_plan(&m->weights_config, &m->allocator, &plan);
    if (!rc) {
        ifn_infer_memory_plan_v1 v = {0};
        v.owned_host_bytes = sizeof *m + atomic_load(&m->budget.live);
        v.owned_device_bytes = plan.arena_bytes;
        v.peak_owned_device_bytes = plan.peak_device_bytes;
        /* Tokenizer/maps and vendor allocations are not known before loading. */
        v.unknown_fields = UINT64_C(3) | SI_UNKNOWN_VENDOR;
        rc = si_output(out, &v, sizeof v);
    }
    si_unlock(&m->busy);
    return si_native_error(e, rc, IFN_STAGE_VALIDATE);
}

ifn_status si_model_load(ifn_infer_model *m, const ifn_infer_source_v1 *s, ifn_infer_error_v1 *e) {
    if (!m)
        return si_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_LOAD, "missing model");
    if (!si_lock(&m->busy))
        return si_error(e, IFN_BUSY, IFN_STAGE_LOAD, "model busy");
    ifn_status rc = m->state == IFN_HANDLE_CREATED ? IFN_OK : IFN_INFER_INVALID_STATE;
    char paths[3][4096];
    if (!rc)
        rc = source_paths(s, paths);
    ifn_weight_plan plan;
    if (!rc)
        rc = ifn_weights_plan(&m->weights_config, &m->allocator, &plan);
    if (!rc && plan.peak_device_bytes > m->device_budget)
        rc = IFN_LIMIT;
    if (rc) {
        si_unlock(&m->busy);
        return si_native_error(e, rc, IFN_STAGE_LOAD);
    }
    m->state = IFN_HANDLE_PREPARING;
    rc = ifn_tokenizer_open(paths[0], &m->allocator, &m->tokenizer);
    if (!rc)
        rc = ifn_inkling_checkpoint_open(paths[1], m->weights_config.with_mtp ? paths[2] : NULL,
                                         &m->allocator, &m->checkpoint);
    if (!rc)
        rc = ifn_gpu_weights_create(&m->weights_config, &m->allocator, &m->weights);
    if (!rc)
        rc = ifn_gpu_weights_load(m->weights, m->checkpoint);
    if (!rc) {
        ifn_inkling_checkpoint_close(m->checkpoint);
        m->checkpoint = NULL;
    }
    /* On failure retain mapped input until native cleanup has succeeded. Even
     * a failed stream synchronization must not shorten borrowed DMA lifetime. */
    m->state = rc ? IFN_HANDLE_FAILED : IFN_HANDLE_READY;
    si_unlock(&m->busy);
    return si_native_error(e, rc, IFN_STAGE_LOAD);
}

ifn_status si_model_info(ifn_infer_model *m, ifn_infer_model_info_v1 *out, ifn_infer_error_v1 *e) {
    if (!m)
        return si_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE, "missing model");
    if (!si_lock(&m->busy))
        return si_error(e, IFN_BUSY, IFN_STAGE_VALIDATE, "model busy");
    ifn_infer_model_info_v1 v = {0};
    v.state = m->state;
    v.max_prepared_sessions = 1;
    v.supported_features = features(m);
    v.enabled_features =
        m->state == IFN_HANDLE_READY ? features(m) & ~IFN_INFER_CAP_PREFIX_CACHE : 0;
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
    ifn_status rc = si_output(out, &v, sizeof v);
    si_unlock(&m->busy);
    return si_native_error(e, rc, IFN_STAGE_VALIDATE);
}

ifn_status si_model_close(ifn_infer_model **ptr, ifn_infer_error_v1 *e) {
    if (!ptr)
        return si_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_CLOSE, "missing handle pointer");
    ifn_infer_model *m = *ptr;
    if (!m)
        return si_error(e, IFN_OK, 0, NULL);
    if (!si_lock(&m->busy))
        return si_error(e, IFN_BUSY, IFN_STAGE_CLOSE, "model busy");
    if (m->children) {
        si_unlock(&m->busy);
        return si_error(e, IFN_BUSY, IFN_STAGE_CLOSE, "model has sessions");
    }
    m->state = IFN_HANDLE_CLOSING;
    ifn_status rc = ifn_gpu_weights_destroy(&m->weights);
    if (rc) {
        si_unlock(&m->busy);
        return si_native_error(e, rc, IFN_STAGE_CLOSE);
    }
    ifn_inkling_checkpoint_close(m->checkpoint);
    ifn_tokenizer_close(m->tokenizer);
    ifn_options a = m->budget.upstream;
    si_free(&a, m);
    *ptr = NULL;
    return si_error(e, IFN_OK, 0, NULL);
}
