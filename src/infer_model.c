/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "infer_internal.h"
#include <stdio.h>

static ifn_status ifn_infer_model_local_extensions(ifn_infer_array array,
                                                   ifn_infer_logger_v1 *logger,
                                                   ifn_infer_artifact_manifest_v1 *manifest,
                                                   bool *verify) {
    if (!array.count)
        return ifn_infer_no_extensions(array);
    if (array.count > 2 || !array.data || array.reserved ||
        array.stride < sizeof(ifn_infer_extension_v1) || array.stride > IFN_INFER_RECORD_MAX_BYTES)
        return IFN_INVALID_ARGUMENT;
    bool logged = false;
    for (size_t i = 0; i < array.count; i++) {
        const unsigned char *row = (const unsigned char *)array.data + i * array.stride;
        ifn_infer_extension_v1 ext;
        memcpy(&ext, row, sizeof ext);
        if (ext.hdr.struct_size > array.stride)
            return IFN_INVALID_ARGUMENT;
        ifn_status rc = ifn_infer_input(row, sizeof ext);
        if (rc)
            return rc;
        size_t known = ext.id == IFN_INFER_EXT_LOGGER              ? sizeof *logger
                       : ext.id == IFN_INFER_EXT_ARTIFACT_MANIFEST ? sizeof *manifest
                                                                   : 0;
        if (!known)
            return IFN_UNSUPPORTED;
        if (ext.version != 1 || !ext.payload || ext.payload_bytes < known ||
            ext.payload_bytes > IFN_INFER_RECORD_MAX_BYTES)
            return IFN_INVALID_ARGUMENT;
        ifn_infer_in hdr;
        memcpy(&hdr, ext.payload, sizeof hdr);
        if (hdr.struct_size > ext.payload_bytes)
            return IFN_INVALID_ARGUMENT;
        rc = ifn_infer_input(ext.payload, known);
        if (rc)
            return rc;
        if (ext.id == IFN_INFER_EXT_LOGGER) {
            if (logged)
                return IFN_INVALID_ARGUMENT;
            memcpy(logger, ext.payload, sizeof *logger);
            if (!logger->callback)
                return IFN_INVALID_ARGUMENT;
            logged = true;
        } else {
            if (*verify)
                return IFN_INVALID_ARGUMENT;
            memcpy(manifest, ext.payload, sizeof *manifest);
            *verify = true;
        }
    }
    return IFN_OK;
}

static uint64_t ifn_infer_model_local_features(const ifn_infer_model *m) {
    uint64_t features = m->weights_config.with_mtp
                            ? IFN_INFER_FEATURES
                            : IFN_INFER_FEATURES & ~IFN_INFER_CAP_SPECULATION;
    return m->verify_artifacts ? features : features & ~IFN_INFER_CAP_ARTIFACT_VERIFY;
}

static ifn_status ifn_infer_model_local_source_paths(const ifn_infer_source_v1 *s,
                                                     char paths[3][4096]) {
    ifn_status rc = ifn_infer_input(s, sizeof *s);
    if (rc)
        return rc;
    if (!ifn_infer_equal(s->format_id, "inkling-safetensors") ||
        !ifn_infer_span(s->directory, 4000) || !s->directory.bytes ||
        memchr(s->directory.data, 0, (size_t)s->directory.bytes))
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

ifn_status ifn_infer_model_create(const ifn_infer_model_config_v1 *c, ifn_infer_model **out,
                                  ifn_infer_error_v1 *e) {
    if (out)
        *out = NULL;
    ifn_status rc = ifn_infer_input(c, sizeof *c);
    if (rc || !out)
        return ifn_infer_error(e, rc ? rc : IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE,
                               "invalid model configuration");
    ifn_infer_logger_v1 logger = {0};
    ifn_infer_artifact_manifest_v1 manifest = {0};
    bool verify = false;
    rc = ifn_infer_model_local_extensions(c->extensions, &logger, &manifest, &verify);
    if (rc || (c->required_features & ~IFN_INFER_FEATURES))
        return ifn_infer_error(e, rc ? rc : IFN_UNSUPPORTED, IFN_STAGE_VALIDATE,
                               "unsupported model features");
    ifn_options a = IFN_OPTIONS_INIT;
    a.context = c->allocator.context;
    a.allocate = c->allocator.allocate;
    a.release = c->allocator.release;
    ifn_options resolved;
    rc = ifn_options_resolve(&a, &resolved);
    a = resolved;
    if (rc)
        return ifn_infer_native_error(e, rc, IFN_STAGE_VALIDATE);
    if (c->max_owned_host_bytes < sizeof(ifn_infer_model))
        return ifn_infer_error(e, IFN_LIMIT, IFN_STAGE_VALIDATE, "model host budget too small");
    ifn_weight_config wc = IFN_WEIGHT_CONFIG_INIT;
    wc.with_mtp = 0;
    const char *profiles[] = {"exact-ar",  "fp8-ar",  "nvfp4-ar",
                              "exact-mtp", "fp8-mtp", "nvfp4-mtp"};
    const char *profile = NULL;
    for (uint32_t i = 0; i < 6; i++)
        if (ifn_infer_equal(c->weight_profile, profiles[i])) {
            wc.mode = i % 3;
            wc.with_mtp = i / 3;
            profile = profiles[i];
        }
    if (!profile || !ifn_infer_span(c->device_id, 10) || !c->device_id.bytes)
        return ifn_infer_error(e, IFN_UNSUPPORTED, IFN_STAGE_VALIDATE,
                               "unknown weight profile or device identifier");
    uint32_t device = 0;
    for (size_t i = 0; i < c->device_id.bytes; i++) {
        unsigned char ch = ((const unsigned char *)c->device_id.data)[i];
        if (ch < '0' || ch > '9' || device > 214748364u || (device == 214748364u && ch > '7'))
            return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE,
                                   "device must be a nonnegative CUDA ordinal");
        device = device * 10 + (uint32_t)(ch - '0');
    }
    wc.device = (int32_t)device;
    if (!wc.with_mtp && (c->required_features & IFN_INFER_CAP_SPECULATION))
        return ifn_infer_error(e, IFN_UNSUPPORTED, IFN_STAGE_VALIDATE,
                               "speculation requires an MTP profile");
    if ((c->required_features & IFN_INFER_CAP_ARTIFACT_VERIFY) && !verify)
        return ifn_infer_error(e, IFN_UNSUPPORTED, IFN_STAGE_VALIDATE,
                               "artifact verification requires expected hashes");
    unsigned char hashes[3][32] = {{0}};
    if (verify) {
        ifn_infer_bytes values[] = {manifest.tokenizer_sha256, manifest.weights_sha256,
                                    manifest.mtp_sha256};
        for (unsigned i = 0; i < (wc.with_mtp ? 3u : 2u); i++)
            if (values[i].bytes != 64 || !ifn_sha256_unhex(values[i].data, 64, hashes[i]))
                return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE,
                                       "expected SHA-256 must be 64 hexadecimal bytes");
        if (!wc.with_mtp && (values[2].bytes || values[2].data))
            return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE,
                                   "unexpected MTP hash for AR model");
    }
    ifn_infer_model *m = ifn_infer_alloc(&a, sizeof *m);
    if (!m)
        return ifn_infer_error(e, IFN_OUT_OF_MEMORY, IFN_STAGE_VALIDATE, "allocate model");
    memset(m, 0, sizeof *m);
    ifn_infer_budget_init(&m->budget, &a, c->max_owned_host_bytes - sizeof *m, &m->allocator);
    atomic_flag_clear(&m->busy);
    (void)snprintf(m->device_id, sizeof m->device_id, "%u", device);
    m->verify_artifacts = verify;
    memcpy(m->expected_sha256, hashes, sizeof hashes);
    m->log = logger.callback;
    m->log_context = logger.context;
    m->weights_config = wc;
    m->profile = profile;
    m->required = c->required_features;
    m->device_budget = c->max_owned_device_bytes;
    *out = m;
    return ifn_infer_error(e, IFN_OK, 0, NULL);
}

ifn_status ifn_infer_model_plan(ifn_infer_model *m, const ifn_infer_source_v1 *s,
                                ifn_infer_memory_plan_v1 *out, ifn_infer_error_v1 *e) {
    if (!m)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE, "missing model");
    if (!ifn_infer_lock(&m->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_VALIDATE, "model busy");
    char paths[3][4096];
    ifn_status rc = ifn_infer_model_local_source_paths(s, paths);
    ifn_weight_plan plan = {0};
    if (!rc)
        rc = ifn_weights_plan(&m->weights_config, &m->allocator, &plan);
    if (!rc) {
        ifn_infer_memory_plan_v1 v = {0};
        v.owned_host_bytes = sizeof *m + atomic_load(&m->budget.live);
        v.owned_device_bytes = plan.arena_bytes;
        v.peak_owned_device_bytes = plan.peak_device_bytes;
        /* Tokenizer/maps and vendor allocations are not known before loading. */
        v.unknown_fields = UINT64_C(3) | IFN_INFER_UNKNOWN_VENDOR;
        rc = ifn_infer_output(out, &v, sizeof v);
    }
    ifn_infer_unlock(&m->busy);
    return ifn_infer_model_native_error(m, e, rc, IFN_STAGE_VALIDATE);
}

ifn_status ifn_infer_model_load(ifn_infer_model *m, const ifn_infer_source_v1 *s,
                                ifn_infer_error_v1 *e) {
    if (!m)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_LOAD, "missing model");
    if (!ifn_infer_lock(&m->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_LOAD, "model busy");
    ifn_status rc = m->state == IFN_HANDLE_CREATED ? IFN_OK : IFN_INFER_INVALID_STATE;
    char paths[3][4096];
    if (!rc)
        rc = ifn_infer_model_local_source_paths(s, paths);
    ifn_weight_plan plan = {0};
    if (!rc)
        rc = ifn_weights_plan(&m->weights_config, &m->allocator, &plan);
    if (!rc && plan.peak_device_bytes > m->device_budget)
        rc = IFN_LIMIT;
    if (rc) {
        ifn_infer_unlock(&m->busy);
        return ifn_infer_model_native_error(m, e, rc, IFN_STAGE_LOAD);
    }
    m->state = IFN_HANDLE_PREPARING;
    rc = ifn_tokenizer_open_verified(
        paths[0], &m->allocator, m->verify_artifacts ? m->expected_sha256[0] : NULL, &m->tokenizer);
    if (!rc)
        rc = ifn_inkling_checkpoint_open(paths[1], m->weights_config.with_mtp ? paths[2] : NULL,
                                         &m->allocator, &m->checkpoint);
    if (!rc && m->verify_artifacts) {
        unsigned char digest[32];
        for (unsigned source = 0; !rc && source < (m->weights_config.with_mtp ? 2u : 1u);
             source++) {
            rc = ifn_inkling_sha256(m->checkpoint, source, digest);
            if (!rc && memcmp(digest, m->expected_sha256[source + 1], 32))
                rc = ifn_error(IFN_FORMAT, "checkpoint checksum mismatch");
        }
        if (!rc) {
            ifn_sha256 identity;
            ifn_sha256_init(&identity);
            (void)ifn_sha256_update(&identity, "ifn-artifacts-v1", 16);
            for (unsigned i = 0; i < (m->weights_config.with_mtp ? 3u : 2u); i++)
                (void)ifn_sha256_update(&identity, m->expected_sha256[i], 32);
            ifn_sha256_final(&identity, digest);
            ifn_sha256_hex(digest, m->artifact_fingerprint);
        }
    }
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
    ifn_infer_unlock(&m->busy);
    return ifn_infer_model_native_error(m, e, rc, IFN_STAGE_LOAD);
}

ifn_status ifn_infer_model_info(ifn_infer_model *m, ifn_infer_model_info_v1 *out,
                                ifn_infer_error_v1 *e) {
    if (!m)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_VALIDATE, "missing model");
    if (!ifn_infer_lock(&m->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_VALIDATE, "model busy");
    ifn_infer_model_info_v1 v = {0};
    v.state = m->state;
    v.max_prepared_sessions = 1;
    v.supported_features = ifn_infer_model_local_features(m);
    v.enabled_features = m->state == IFN_HANDLE_READY
                             ? ifn_infer_model_local_features(m) & ~IFN_INFER_CAP_PREFIX_CACHE
                             : 0;
    v.max_context_tokens = 1048576;
    v.max_messages = 1024;
    v.max_input_bytes = IFN_INFER_MAX_INPUT;
    v.max_stops = 8;
    v.max_stop_bytes = 64;
    v.max_event_bytes = 65536;
    v.model_id = ifn_infer_text("inkling-2-mini");
    v.resolved_weight_profile = ifn_infer_text(m->profile);
    v.resolved_device_id = ifn_infer_text(m->device_id);
    v.tokenizer_id = ifn_infer_text("inkling-bpe");
    v.template_id = ifn_infer_text("inkling-text-v1");
    v.fingerprint_verified = m->state == IFN_HANDLE_READY && m->artifact_fingerprint[0];
    if (v.fingerprint_verified)
        v.artifact_fingerprint = (ifn_infer_bytes){m->artifact_fingerprint, 64};
    ifn_status rc = ifn_infer_output(out, &v, sizeof v);
    ifn_infer_unlock(&m->busy);
    return ifn_infer_model_native_error(m, e, rc, IFN_STAGE_VALIDATE);
}

ifn_status ifn_infer_model_close(ifn_infer_model **ptr, ifn_infer_error_v1 *e) {
    if (!ptr)
        return ifn_infer_error(e, IFN_INVALID_ARGUMENT, IFN_STAGE_CLOSE, "missing handle pointer");
    ifn_infer_model *m = *ptr;
    if (!m)
        return ifn_infer_error(e, IFN_OK, 0, NULL);
    if (!ifn_infer_lock(&m->busy))
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_CLOSE, "model busy");
    if (m->children) {
        ifn_infer_unlock(&m->busy);
        return ifn_infer_error(e, IFN_BUSY, IFN_STAGE_CLOSE, "model has sessions");
    }
    m->state = IFN_HANDLE_CLOSING;
    ifn_status rc = ifn_gpu_weights_destroy(&m->weights);
    if (rc) {
        ifn_infer_unlock(&m->busy);
        return ifn_infer_model_native_error(m, e, rc, IFN_STAGE_CLOSE);
    }
    ifn_inkling_checkpoint_close(m->checkpoint);
    ifn_tokenizer_close(m->tokenizer);
    ifn_options a = m->budget.upstream;
    ifn_infer_free(&a, m);
    *ptr = NULL;
    return ifn_infer_error(e, IFN_OK, 0, NULL);
}
