/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Explicit ownership/error contracts follow Jev Bush's C library practices. */
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
static _Thread_local char ifn_error_message[256];
static _Thread_local ifn_native_diagnostic ifn_foundation_diagnostic;

const char *ifn_version(void) {
    return IFN_VERSION_STRING;
}

const char *ifn_status_string(ifn_status status) {
    static const char *const names[] = {
        "ok",          "invalid argument", "I/O error",      "invalid format", "out of memory",
        "not found",   "buffer too small", "limit exceeded", "busy",           "GPU error",
        "unsupported", "cancelled",        "ABI mismatch",   "invalid state",  "not ready"};
    return status < sizeof names / sizeof names[0] ? names[status] : "unknown status";
}

const char *ifn_last_error(void) {
    return ifn_error_message;
}

ifn_status ifn_error(ifn_status status, const char *message) {
    ifn_foundation_diagnostic = (ifn_native_diagnostic){status, 0, 0};
    (void)snprintf(ifn_error_message, sizeof ifn_error_message, "%s", message);
    return status;
}

ifn_native_diagnostic ifn_native_diagnostic_get(void) {
    return ifn_foundation_diagnostic;
}

ifn_status ifn_os_error(ifn_status status, int code, const char *message) {
    ifn_error(status, message);
    if (status && code) {
        ifn_foundation_diagnostic.provider = 1;
        ifn_foundation_diagnostic.code = code;
    }
    return status;
}

ifn_status ifn_win32_error(ifn_status status, uint32_t code, const char *message) {
    ifn_error(status, message);
    if (status && code) {
        ifn_foundation_diagnostic.provider = 5;
        ifn_foundation_diagnostic.code = code;
    }
    return status;
}

ifn_status ifn_native_error(ifn_status status, int code, const char *message) {
    ifn_error(status, message);
    if (status && code > 0) {
        ifn_foundation_diagnostic.provider = code >= 20000 ? 4u : code >= 10000 ? 3u : 2u;
        ifn_foundation_diagnostic.code = code >= 20000   ? code - 20000
                                         : code >= 10000 ? code - 10000
                                                         : code;
    }
    return status;
}

bool ifn_size_add(size_t a, size_t b, size_t *out) {
    if (!out || b > SIZE_MAX - a)
        return false;
    *out = a + b;
    return true;
}

bool ifn_size_mul(size_t a, size_t b, size_t *out) {
    if (!out || (b && a > SIZE_MAX / b))
        return false;
    *out = a * b;
    return true;
}

bool ifn_size_align(size_t value, size_t alignment, size_t *out) {
    size_t total;
    if (!alignment || (alignment & (alignment - 1)) ||
        !ifn_size_add(value, alignment - 1, &total) || !out)
        return false;
    *out = total & ~(alignment - 1);
    return true;
}

static void *ifn_foundation_local_default_allocate(void *context, size_t bytes) {
    (void)context;
    return malloc(bytes);
}

static void ifn_foundation_local_default_release(void *context, void *allocation) {
    (void)context;
    free(allocation);
}

ifn_status ifn_options_resolve(const ifn_options *input, ifn_options *out) {
    *out = (ifn_options)IFN_OPTIONS_INIT;
    if (input) {
        if (input->struct_size < sizeof *input || input->api_version != IFN_API_VERSION)
            return ifn_error(IFN_INVALID_ARGUMENT, "unsupported options size/version");
        *out = *input;
        if (!!out->allocate != !!out->release)
            return ifn_error(IFN_INVALID_ARGUMENT, "allocator callbacks must be paired");
    }
    if (!out->allocate) {
        out->allocate = ifn_foundation_local_default_allocate;
        out->release = ifn_foundation_local_default_release;
    }
    return IFN_OK;
}
