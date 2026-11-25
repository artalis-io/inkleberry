/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Explicit ownership/error contracts follow Jev Bush's C library practices. */
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
static _Thread_local char sq_error_message[256];

const char *sq_version(void) {
    return "0.1.0-dev";
}

const char *sq_status_string(sq_status status) {
    static const char *const names[] = {
        "ok",          "invalid argument", "I/O error",      "invalid format", "out of memory",
        "not found",   "buffer too small", "limit exceeded", "busy",           "GPU error",
        "unsupported", "cancelled"};
    return status < sizeof names / sizeof names[0] ? names[status] : "unknown status";
}

const char *sq_last_error(void) {
    return sq_error_message;
}

sq_status sq_error(sq_status status, const char *message) {
    (void)snprintf(sq_error_message, sizeof sq_error_message, "%s", message);
    return status;
}

bool sq_size_add(size_t a, size_t b, size_t *out) {
    if (!out || b > SIZE_MAX - a)
        return false;
    *out = a + b;
    return true;
}

bool sq_size_mul(size_t a, size_t b, size_t *out) {
    if (!out || (b && a > SIZE_MAX / b))
        return false;
    *out = a * b;
    return true;
}

bool sq_size_align(size_t value, size_t alignment, size_t *out) {
    size_t total;
    if (!alignment || (alignment & (alignment - 1)) || !sq_size_add(value, alignment - 1, &total) ||
        !out)
        return false;
    *out = total & ~(alignment - 1);
    return true;
}

static void *default_allocate(void *context, size_t bytes) {
    (void)context;
    return malloc(bytes);
}

static void default_release(void *context, void *allocation) {
    (void)context;
    free(allocation);
}

sq_status sq_options_resolve(const sq_options *input, sq_options *out) {
    *out = (sq_options)SQ_OPTIONS_INIT;
    if (input) {
        if (input->struct_size < sizeof *input || input->api_version != SQ_API_VERSION)
            return sq_error(SQ_INVALID_ARGUMENT, "unsupported options size/version");
        *out = *input;
        if (!!out->allocate != !!out->release)
            return sq_error(SQ_INVALID_ARGUMENT, "allocator callbacks must be paired");
    }
    if (!out->allocate) {
        out->allocate = default_allocate;
        out->release = default_release;
    }
    return SQ_OK;
}
