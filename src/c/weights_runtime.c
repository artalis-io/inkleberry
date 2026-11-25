/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Owned GPU-image lifecycle. Backend injection is private and used by fault
 * tests; public GPU entrypoints bind only the actual CUDA backend. */
#include "weights_internal.h"
#include <stdio.h>
#include <string.h>

void sq_weight_fail(sq_gpu_weights *w, sq_status status, const char *message) {
    if (w->status != SQ_OK)
        return;
    w->status = status;
    (void)snprintf(w->failure, sizeof w->failure, "%s", message);
}

void sq_weight_backend_check(sq_gpu_weights *w, int error, const char *operation) {
    if (!error || w->status != SQ_OK)
        return;
    w->backend_error = error;
    w->status = error == 2 ? SQ_OUT_OF_MEMORY : error == 801 ? SQ_UNSUPPORTED : SQ_GPU_ERROR;
    (void)snprintf(w->failure, sizeof w->failure, "%s: CUDA status %d", operation, error);
}

static sq_status result(sq_gpu_weights *w) {
    return sq_error(w->status, w->status == SQ_OK ? "" : w->failure);
}

static bool acquire(sq_gpu_weights *w) {
    return w && !atomic_flag_test_and_set_explicit(&w->busy, memory_order_acquire);
}

static void unlock(sq_gpu_weights *w) {
    atomic_flag_clear_explicit(&w->busy, memory_order_release);
}

static bool resources(const sq_gpu_weights *w) {
    return w->arena || w->stage || w->amax || w->fold_temp || w->stream;
}

static void release_ptr(sq_gpu_weights *w, void **p) {
    if (!*p)
        return;
    int rc = w->backend.release(w->backend.context, *p);
    if (!rc)
        *p = NULL;
    sq_weight_backend_check(w, rc, "free owned GPU allocation");
}

static void temporary_cleanup(sq_gpu_weights *w) {
    release_ptr(w, &w->fold_temp);
    release_ptr(w, &w->amax);
    release_ptr(w, &w->stage);
}

static void cleanup(sq_gpu_weights *w) {
    if (w->stream)
        sq_weight_backend_check(w, w->backend.sync(w->backend.context, w->stream),
                                "synchronize owned stream");
    temporary_cleanup(w);
    release_ptr(w, &w->arena);
    if (w->stream && !w->arena && !w->stage && !w->amax && !w->fold_temp) {
        int rc = w->backend.stream_destroy(w->backend.context, w->stream);
        if (!rc)
            w->stream = NULL;
        sq_weight_backend_check(w, rc, "destroy owned stream");
    }
    w->ready = false;
}

sq_status sq_weights_create(const sq_weight_config *input, const sq_options *options,
                            const sq_weight_backend *backend, sq_gpu_weights **out) {
    if (out)
        *out = NULL;
    if (!out)
        return sq_error(SQ_INVALID_ARGUMENT, "missing weights output");
    sq_weight_config config = SQ_WEIGHT_CONFIG_INIT;
    if (input) {
        if (input->struct_size < sizeof *input || input->api_version != SQ_API_VERSION)
            return sq_error(SQ_INVALID_ARGUMENT, "unsupported weight configuration size/version");
        config = *input;
    }
    if (config.mode > 2 || config.with_mtp > 1 || config.mtp_bf16 > 1 || config.device < 0 ||
        config.reserved)
        return sq_error(SQ_INVALID_ARGUMENT, "invalid weight configuration");
    sq_options a;
    sq_status status = sq_options_resolve(options, &a);
    if (status != SQ_OK)
        return status;
    sq_gpu_weights *w = a.allocate(a.context, sizeof *w);
    if (!w)
        return sq_error(SQ_OUT_OF_MEMORY, "allocate GPU weight bookkeeping");
    memset(w, 0, sizeof *w);
    w->allocator = a;
    w->config = config;
    if (backend)
        w->backend = *backend;
    atomic_flag_clear(&w->busy);
    sq_weights_recipe(w);
    if (w->status != SQ_OK) {
        status = result(w);
        a.release(a.context, w);
        return status;
    }
    w->capacity = w->offset;
    w->plan.arena_bytes = w->offset;
    w->plan.allocations = w->span_count;
    w->plan.views = w->view_count;
    w->plan.staging_bytes = (size_t)200058 * 2304 * 2;
    w->plan.amax_bytes = 4;
    w->plan.fold_temporary_bytes = config.with_mtp ? (size_t)200058 * 256 * 2 : 0;
    size_t peak = w->offset;
    if (!sq_size_add(peak, (size_t)w->plan.staging_bytes, &peak) || !sq_size_add(peak, 4, &peak) ||
        !sq_size_add(peak, (size_t)w->plan.fold_temporary_bytes, &peak)) {
        a.release(a.context, w);
        return sq_error(SQ_LIMIT, "GPU peak allocation overflow");
    }
    w->plan.peak_device_bytes = peak;
    *out = w;
    return sq_error(SQ_OK, "");
}

sq_status sq_weights_plan(const sq_weight_config *config, const sq_options *allocator,
                          sq_weight_plan *out) {
    if (out)
        memset(out, 0, sizeof *out);
    if (!out)
        return sq_error(SQ_INVALID_ARGUMENT, "missing weight plan output");
    sq_gpu_weights *w = NULL;
    sq_status status = sq_weights_create(config, allocator, NULL, &w);
    if (status != SQ_OK)
        return status;
    *out = w->plan;
    sq_options a = w->allocator;
    a.release(a.context, w);
    return sq_error(SQ_OK, "");
}

sq_status sq_weights_load(sq_gpu_weights *w, const sq_inkling_checkpoint *checkpoint) {
    if (!w || !checkpoint)
        return sq_error(SQ_INVALID_ARGUMENT, "missing GPU weights/checkpoint");
    if (!acquire(w))
        return sq_error(SQ_BUSY, "weight operation already active");
    if (resources(w) || w->ready) {
        unlock(w);
        return sq_error(SQ_BUSY, "destroy existing or failed GPU image before reload");
    }
    if (w->config.with_mtp && sq_inkling_checkpoint_count(checkpoint) != 876) {
        unlock(w);
        return sq_error(SQ_FORMAT, "MTP requested but not admitted");
    }
    w->status = SQ_OK;
    w->failure[0] = 0;
    w->backend_error = 0;
    sq_device_scope previous = {-1, NULL};
    sq_weight_backend_check(w, w->backend.enter(w->backend.context, w->config.device, &previous),
                            "select GPU");
    if (w->status == SQ_OK) {
        size_t free_bytes = 0;
        sq_weight_backend_check(w, w->backend.available(w->backend.context, &free_bytes),
                                "query free GPU memory");
        if (w->status == SQ_OK && w->plan.peak_device_bytes > free_bytes)
            sq_weight_fail(w, SQ_OUT_OF_MEMORY,
                           "insufficient free GPU memory for planned weights and temporaries");
        if (w->status == SQ_OK)
            sq_weight_backend_check(w, w->backend.stream_create(w->backend.context, &w->stream),
                                    "create weight stream");
#define ALLOC(p, n)                                                                                \
    do {                                                                                           \
        if (w->status == SQ_OK)                                                                    \
            sq_weight_backend_check(                                                               \
                w, w->backend.allocate(w->backend.context, &w->p, (size_t)(n)), "allocate " #p);   \
    } while (0)
        ALLOC(arena, w->plan.arena_bytes);
        ALLOC(stage, w->plan.staging_bytes);
        ALLOC(amax, 4);
        if (w->plan.fold_temporary_bytes)
            ALLOC(fold_temp, w->plan.fold_temporary_bytes);
#undef ALLOC
        if (w->status == SQ_OK) {
            w->execute = true;
            w->checkpoint = checkpoint;
            sq_weights_recipe(w);
            w->checkpoint = NULL;
            w->execute = false;
            if (w->status == SQ_OK &&
                (w->offset != w->capacity || w->span_count != w->plan.allocations ||
                 w->view_count != w->plan.views))
                sq_weight_fail(w, SQ_FORMAT, "weight arena exact-fill mismatch");
            if (w->status == SQ_OK)
                sq_weight_backend_check(w, w->backend.sync(w->backend.context, w->stream),
                                        "finish GPU weight upload");
            temporary_cleanup(w);
        }
        if (w->status != SQ_OK)
            cleanup(w);
    }
    if (previous.device >= 0)
        sq_weight_backend_check(w, w->backend.leave(w->backend.context, &previous),
                                "restore caller device");
    w->ready = w->status == SQ_OK;
    sq_status status = result(w);
    unlock(w);
    return status;
}

sq_status sq_weights_destroy(sq_gpu_weights **handle) {
    if (!handle)
        return sq_error(SQ_INVALID_ARGUMENT, "missing weight handle pointer");
    sq_gpu_weights *w = *handle;
    if (!w)
        return sq_error(SQ_OK, "");
    if (!acquire(w))
        return sq_error(SQ_BUSY, "weight operation already active");
    w->status = SQ_OK;
    w->failure[0] = 0;
    w->backend_error = 0;
    w->ready = false;
    if (resources(w)) {
        sq_device_scope previous = {-1, NULL};
        sq_weight_backend_check(w,
                                w->backend.enter(w->backend.context, w->config.device, &previous),
                                "select GPU for cleanup");
        if (w->status == SQ_OK)
            cleanup(w);
        if (previous.device >= 0)
            sq_weight_backend_check(w, w->backend.leave(w->backend.context, &previous),
                                    "restore caller device");
    }
    sq_status status = result(w);
    if (resources(w)) {
        unlock(w);
        return status;
    }
    sq_options a = w->allocator;
    *handle = NULL;
    a.release(a.context, w);
    return status;
}

sq_status sq_weights_get_plan(sq_gpu_weights *w, sq_weight_plan *out) {
    if (out)
        memset(out, 0, sizeof *out);
    if (!w || !out)
        return sq_error(SQ_INVALID_ARGUMENT, "missing plan arguments");
    if (!acquire(w))
        return sq_error(SQ_BUSY, "weight operation already active");
    *out = w->plan;
    unlock(w);
    return sq_error(SQ_OK, "");
}

sq_status sq_weights_get_span(sq_gpu_weights *w, size_t index, sq_weight_span *out) {
    if (out)
        memset(out, 0, sizeof *out);
    if (!w || !out)
        return sq_error(SQ_INVALID_ARGUMENT, "missing span arguments");
    if (!acquire(w))
        return sq_error(SQ_BUSY, "weight operation already active");
    sq_status status = !w->ready ? SQ_BUSY : index >= w->span_count ? SQ_NOT_FOUND : SQ_OK;
    if (status == SQ_OK)
        *out = w->spans[index];
    unlock(w);
    return sq_error(status, status == SQ_OK ? "" : "weight span unavailable");
}

sq_status sq_weights_get_view(sq_gpu_weights *w, size_t index, sq_weight_view *out) {
    if (out)
        memset(out, 0, sizeof *out);
    if (!w || !out)
        return sq_error(SQ_INVALID_ARGUMENT, "missing view arguments");
    if (!acquire(w))
        return sq_error(SQ_BUSY, "weight operation already active");
    sq_status status = !w->ready ? SQ_BUSY : index >= w->view_count ? SQ_NOT_FOUND : SQ_OK;
    if (status == SQ_OK)
        *out = w->views[index];
    unlock(w);
    return sq_error(status, status == SQ_OK ? "" : "weight view unavailable");
}

sq_status sq_weights_read(sq_gpu_weights *w, size_t index, uint64_t offset, void *out,
                          size_t bytes) {
    if (!w || (!out && bytes))
        return sq_error(SQ_INVALID_ARGUMENT, "invalid readback arguments");
    if (!acquire(w))
        return sq_error(SQ_BUSY, "weight operation already active");
    if (!w->ready) {
        unlock(w);
        return sq_error(SQ_BUSY, "weights not ready");
    }
    if (index >= w->span_count || offset > w->spans[index].bytes ||
        bytes > w->spans[index].bytes - offset) {
        unlock(w);
        return sq_error(SQ_INVALID_ARGUMENT, "weight readback outside span");
    }
    w->status = SQ_OK;
    w->backend_error = 0;
    w->failure[0] = 0;
    sq_device_scope previous = {-1, NULL};
    sq_weight_backend_check(w, w->backend.enter(w->backend.context, w->config.device, &previous),
                            "select GPU for readback");
    if (w->status == SQ_OK && bytes) {
        sq_weight_backend_check(
            w,
            w->backend.copy(w->backend.context, out,
                            (unsigned char *)w->arena + w->spans[index].offset + offset, bytes, 1,
                            w->stream),
            "read weight payload");
        if (w->status == SQ_OK)
            sq_weight_backend_check(w, w->backend.sync(w->backend.context, w->stream),
                                    "finish weight readback");
    }
    if (previous.device >= 0)
        sq_weight_backend_check(w, w->backend.leave(w->backend.context, &previous),
                                "restore caller device");
    if (w->status != SQ_OK)
        w->ready = false;
    sq_status status = result(w);
    unlock(w);
    return status;
}
