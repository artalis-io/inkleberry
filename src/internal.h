/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_INTERNAL_H
#define IFN_INTERNAL_H
#include "ifn_core.h"
#include <stdbool.h>
#include "sha256.h"
ifn_status ifn_safetensors_sha256(const ifn_safetensors *, unsigned char[32]);
ifn_status ifn_inkling_sha256(const ifn_inkling_checkpoint *, uint32_t, unsigned char[32]);
ifn_status ifn_tokenizer_open_verified(const char *, const ifn_options *, const unsigned char *,
                                       ifn_tokenizer **);
ifn_status ifn_error(ifn_status status, const char *message);

typedef struct {
    ifn_status status;
    uint32_t provider;
    int64_t code;
} ifn_native_diagnostic;

ifn_status ifn_os_error(ifn_status, int, const char *);
ifn_status ifn_native_error(ifn_status, int, const char *);
ifn_native_diagnostic ifn_native_diagnostic_get(void);
bool ifn_size_add(size_t a, size_t b, size_t *out);
bool ifn_size_mul(size_t a, size_t b, size_t *out);
bool ifn_size_align(size_t value, size_t alignment, size_t *out);
ifn_status ifn_options_resolve(const ifn_options *input, ifn_options *out);
ifn_status ifn_safetensors_named_data(const ifn_safetensors *, const char *, const void **,
                                      size_t *);
#endif
