/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_INTERNAL_H
#define IFN_INTERNAL_H
#include "ifn.h"
#include <stdbool.h>
ifn_status ifn_error(ifn_status status, const char *message);
bool ifn_size_add(size_t a, size_t b, size_t *out);
bool ifn_size_mul(size_t a, size_t b, size_t *out);
bool ifn_size_align(size_t value, size_t alignment, size_t *out);
ifn_status ifn_options_resolve(const ifn_options *input, ifn_options *out);
ifn_status ifn_safetensors_named_data(const ifn_safetensors *, const char *, const void **,
                                      size_t *);
#endif
