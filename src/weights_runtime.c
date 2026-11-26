/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Owned GPU-image lifecycle. Backend injection is private and used by fault
 * tests; public GPU entrypoints bind only the actual CUDA backend. */
#include "weights_internal.h"
#include <stdio.h>
#include <string.h>

void ifn_weight_fail(ifn_gpu_weights *w, ifn_status status, const char *message) {
    if (w->status != IFN_OK)
        return;
    w->status = status;
    (void)snprintf(w->failure, sizeof w->failure, "%s", message);
}

void ifn_weight_backend_check(ifn_gpu_weights *w, int error, const char *operation) {
    if (!error || w->status != IFN_OK)
        return;
    w->backend_error = error;
    w->status = (error == 2 || error == 10003 || error == 20002) ? IFN_OUT_OF_MEMORY
                : (error == 801 || error == 20801)               ? IFN_UNSUPPORTED
                                                                 : IFN_GPU_ERROR;
    (void)snprintf(w->failure, sizeof w->failure, "%s: CUDA status %d", operation, error);
}

static ifn_status ifn_weights_runtime_local_result(ifn_gpu_weights *w) {
    return ifn_native_error(w->status, w->backend_error, w->status == IFN_OK ? "" : w->failure);
}

static bool ifn_weights_runtime_local_acquire(ifn_gpu_weights *w) {
    return w && !atomic_flag_test_and_set_explicit(&w->busy, memory_order_acquire);
}

static void ifn_weights_runtime_local_unlock(ifn_gpu_weights *w) {
    atomic_flag_clear_explicit(&w->busy, memory_order_release);
}

static bool ifn_weights_runtime_local_resources(const ifn_gpu_weights *w) {
    return w->arena || w->stage || w->amax || w->fold_temp || w->stream;
}

static void ifn_weights_runtime_local_release_ptr(ifn_gpu_weights *w, void **p) {
    if (!*p)
        return;
    int rc = w->backend.release(w->backend.context, *p);
    if (!rc)
        *p = NULL;
    ifn_weight_backend_check(w, rc, "free owned GPU allocation");
}

static void ifn_weights_runtime_local_temporary_cleanup(ifn_gpu_weights *w) {
    ifn_weights_runtime_local_release_ptr(w, &w->fold_temp);
    ifn_weights_runtime_local_release_ptr(w, &w->amax);
    ifn_weights_runtime_local_release_ptr(w, &w->stage);
}

static void ifn_weights_runtime_local_cleanup(ifn_gpu_weights *w) {
    if (w->stream)
        ifn_weight_backend_check(w, w->backend.sync(w->backend.context, w->stream),
                                 "synchronize owned stream");
    ifn_weights_runtime_local_temporary_cleanup(w);
    ifn_weights_runtime_local_release_ptr(w, &w->arena);
    if (w->stream && !w->arena && !w->stage && !w->amax && !w->fold_temp) {
        int rc = w->backend.stream_destroy(w->backend.context, w->stream);
        if (!rc)
            w->stream = NULL;
        ifn_weight_backend_check(w, rc, "destroy owned stream");
    }
    w->ready = false;
}

ifn_status ifn_weights_create(const ifn_weight_config *input, const ifn_options *options,
                              const ifn_weight_backend *backend, ifn_gpu_weights **out) {
    if (out)
        *out = NULL;
    if (!out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing weights output");
    ifn_weight_config config = IFN_WEIGHT_CONFIG_INIT;
    if (input) {
        if (input->struct_size < sizeof *input || input->api_version != IFN_API_VERSION)
            return ifn_error(IFN_INVALID_ARGUMENT, "unsupported weight configuration size/version");
        config = *input;
    }
    if (config.mode > 2 || config.with_mtp > 1 || config.mtp_bf16 > 1 || config.device < 0 ||
        config.reserved)
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid weight configuration");
    ifn_options a;
    ifn_status status = ifn_options_resolve(options, &a);
    if (status != IFN_OK)
        return status;
    ifn_gpu_weights *w = a.allocate(a.context, sizeof *w);
    if (!w)
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate GPU weight bookkeeping");
    memset(w, 0, sizeof *w);
    w->allocator = a;
    w->config = config;
    if (backend)
        w->backend = *backend;
    atomic_flag_clear(&w->busy);
    ifn_weights_recipe(w);
    if (w->status != IFN_OK) {
        status = ifn_weights_runtime_local_result(w);
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
    if (!ifn_size_add(peak, (size_t)w->plan.staging_bytes, &peak) ||
        !ifn_size_add(peak, 4, &peak) ||
        !ifn_size_add(peak, (size_t)w->plan.fold_temporary_bytes, &peak)) {
        a.release(a.context, w);
        return ifn_error(IFN_LIMIT, "GPU peak allocation overflow");
    }
    w->plan.peak_device_bytes = peak;
    *out = w;
    return ifn_error(IFN_OK, "");
}

ifn_status ifn_weights_plan(const ifn_weight_config *config, const ifn_options *allocator,
                            ifn_weight_plan *out) {
    if (out)
        memset(out, 0, sizeof *out);
    if (!out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing weight plan output");
    ifn_gpu_weights *w = NULL;
    ifn_status status = ifn_weights_create(config, allocator, NULL, &w);
    if (status != IFN_OK)
        return status;
    *out = w->plan;
    ifn_options a = w->allocator;
    a.release(a.context, w);
    return ifn_error(IFN_OK, "");
}

ifn_status ifn_weights_load(ifn_gpu_weights *w, const ifn_inkling_checkpoint *checkpoint) {
    if (!w || !checkpoint)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing GPU weights/checkpoint");
    if (!ifn_weights_runtime_local_acquire(w))
        return ifn_error(IFN_BUSY, "weight operation already active");
    if (ifn_weights_runtime_local_resources(w) || w->ready) {
        ifn_weights_runtime_local_unlock(w);
        return ifn_error(IFN_BUSY, "destroy existing or failed GPU image before reload");
    }
    if (w->config.with_mtp && ifn_inkling_checkpoint_count(checkpoint) != 876) {
        ifn_weights_runtime_local_unlock(w);
        return ifn_error(IFN_FORMAT, "MTP requested but not admitted");
    }
    w->status = IFN_OK;
    w->failure[0] = 0;
    w->backend_error = 0;
    ifn_device_scope previous = {-1, NULL};
    ifn_weight_backend_check(w, w->backend.enter(w->backend.context, w->config.device, &previous),
                             "select GPU");
    if (w->status == IFN_OK) {
        size_t free_bytes = 0;
        ifn_weight_backend_check(w, w->backend.available(w->backend.context, &free_bytes),
                                 "query free GPU memory");
        if (w->status == IFN_OK && w->plan.peak_device_bytes > free_bytes)
            ifn_weight_fail(w, IFN_OUT_OF_MEMORY,
                            "insufficient free GPU memory for planned weights and temporaries");
        if (w->status == IFN_OK)
            ifn_weight_backend_check(w, w->backend.stream_create(w->backend.context, &w->stream),
                                     "create weight stream");
#define IFN_WEIGHTS_RUNTIME_LOCAL_ALLOC(p, n)                                                      \
    do {                                                                                           \
        if (w->status == IFN_OK)                                                                   \
            ifn_weight_backend_check(                                                              \
                w, w->backend.allocate(w->backend.context, &w->p, (size_t)(n)), "allocate " #p);   \
    } while (0)
        IFN_WEIGHTS_RUNTIME_LOCAL_ALLOC(arena, w->plan.arena_bytes);
        IFN_WEIGHTS_RUNTIME_LOCAL_ALLOC(stage, w->plan.staging_bytes);
        IFN_WEIGHTS_RUNTIME_LOCAL_ALLOC(amax, 4);
        if (w->plan.fold_temporary_bytes)
            IFN_WEIGHTS_RUNTIME_LOCAL_ALLOC(fold_temp, w->plan.fold_temporary_bytes);
#undef IFN_WEIGHTS_RUNTIME_LOCAL_ALLOC
        if (w->status == IFN_OK) {
            w->execute = true;
            w->checkpoint = checkpoint;
            ifn_weights_recipe(w);
            w->checkpoint = NULL;
            w->execute = false;
            if (w->status == IFN_OK &&
                (w->offset != w->capacity || w->span_count != w->plan.allocations ||
                 w->view_count != w->plan.views))
                ifn_weight_fail(w, IFN_FORMAT, "weight arena exact-fill mismatch");
            if (w->status == IFN_OK)
                ifn_weight_backend_check(w, w->backend.sync(w->backend.context, w->stream),
                                         "finish GPU weight upload");
            ifn_weights_runtime_local_temporary_cleanup(w);
        }
        if (w->status != IFN_OK)
            ifn_weights_runtime_local_cleanup(w);
    }
    if (previous.device >= 0)
        ifn_weight_backend_check(w, w->backend.leave(w->backend.context, &previous),
                                 "restore caller device");
    w->ready = w->status == IFN_OK;
    ifn_status status = ifn_weights_runtime_local_result(w);
    ifn_weights_runtime_local_unlock(w);
    return status;
}

ifn_status ifn_weights_destroy(ifn_gpu_weights **handle) {
    if (!handle)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing weight handle pointer");
    ifn_gpu_weights *w = *handle;
    if (!w)
        return ifn_error(IFN_OK, "");
    if (!ifn_weights_runtime_local_acquire(w))
        return ifn_error(IFN_BUSY, "weight operation already active");
    if (w->session_users) {
        ifn_weights_runtime_local_unlock(w);
        return ifn_error(IFN_BUSY, "GPU image is retained by a session");
    }
    w->status = IFN_OK;
    w->failure[0] = 0;
    w->backend_error = 0;
    w->ready = false;
    if (ifn_weights_runtime_local_resources(w)) {
        ifn_device_scope previous = {-1, NULL};
        ifn_weight_backend_check(w,
                                 w->backend.enter(w->backend.context, w->config.device, &previous),
                                 "select GPU for cleanup");
        if (w->status == IFN_OK)
            ifn_weights_runtime_local_cleanup(w);
        if (previous.device >= 0)
            ifn_weight_backend_check(w, w->backend.leave(w->backend.context, &previous),
                                     "restore caller device");
    }
    ifn_status status = ifn_weights_runtime_local_result(w);
    if (ifn_weights_runtime_local_resources(w)) {
        ifn_weights_runtime_local_unlock(w);
        return status;
    }
    ifn_options a = w->allocator;
    *handle = NULL;
    a.release(a.context, w);
    return status;
}

ifn_status ifn_weights_get_plan(ifn_gpu_weights *w, ifn_weight_plan *out) {
    if (out)
        memset(out, 0, sizeof *out);
    if (!w || !out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing plan arguments");
    if (!ifn_weights_runtime_local_acquire(w))
        return ifn_error(IFN_BUSY, "weight operation already active");
    *out = w->plan;
    ifn_weights_runtime_local_unlock(w);
    return ifn_error(IFN_OK, "");
}

ifn_status ifn_weights_get_span(ifn_gpu_weights *w, size_t index, ifn_weight_span *out) {
    if (out)
        memset(out, 0, sizeof *out);
    if (!w || !out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing span arguments");
    if (!ifn_weights_runtime_local_acquire(w))
        return ifn_error(IFN_BUSY, "weight operation already active");
    ifn_status status = !w->ready ? IFN_BUSY : index >= w->span_count ? IFN_NOT_FOUND : IFN_OK;
    if (status == IFN_OK)
        *out = w->spans[index];
    ifn_weights_runtime_local_unlock(w);
    return ifn_error(status, status == IFN_OK ? "" : "weight span unavailable");
}

ifn_status ifn_weights_get_view(ifn_gpu_weights *w, size_t index, ifn_weight_view *out) {
    if (out)
        memset(out, 0, sizeof *out);
    if (!w || !out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing view arguments");
    if (!ifn_weights_runtime_local_acquire(w))
        return ifn_error(IFN_BUSY, "weight operation already active");
    ifn_status status = !w->ready ? IFN_BUSY : index >= w->view_count ? IFN_NOT_FOUND : IFN_OK;
    if (status == IFN_OK)
        *out = w->views[index];
    ifn_weights_runtime_local_unlock(w);
    return ifn_error(status, status == IFN_OK ? "" : "weight view unavailable");
}

ifn_status ifn_weights_read(ifn_gpu_weights *w, size_t index, uint64_t offset, void *out,
                            size_t bytes) {
    if (!w || (!out && bytes))
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid readback arguments");
    if (!ifn_weights_runtime_local_acquire(w))
        return ifn_error(IFN_BUSY, "weight operation already active");
    if (!w->ready) {
        ifn_weights_runtime_local_unlock(w);
        return ifn_error(IFN_BUSY, "weights not ready");
    }
    if (index >= w->span_count || offset > w->spans[index].bytes ||
        bytes > w->spans[index].bytes - offset) {
        ifn_weights_runtime_local_unlock(w);
        return ifn_error(IFN_INVALID_ARGUMENT, "weight readback outside span");
    }
    w->status = IFN_OK;
    w->backend_error = 0;
    w->failure[0] = 0;
    ifn_device_scope previous = {-1, NULL};
    ifn_weight_backend_check(w, w->backend.enter(w->backend.context, w->config.device, &previous),
                             "select GPU for readback");
    if (w->status == IFN_OK && bytes) {
        ifn_weight_backend_check(
            w,
            w->backend.copy(w->backend.context, out,
                            (unsigned char *)w->arena + w->spans[index].offset + offset, bytes, 1,
                            w->stream),
            "read weight payload");
        if (w->status == IFN_OK)
            ifn_weight_backend_check(w, w->backend.sync(w->backend.context, w->stream),
                                     "finish weight readback");
    }
    if (previous.device >= 0)
        ifn_weight_backend_check(w, w->backend.leave(w->backend.context, &previous),
                                 "restore caller device");
    if (w->status != IFN_OK)
        w->ready = false;
    ifn_status status = ifn_weights_runtime_local_result(w);
    ifn_weights_runtime_local_unlock(w);
    return status;
}
