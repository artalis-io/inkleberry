/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_INTERNAL_H
#define SQ_INTERNAL_H
#include "sq.h"
#include <stdbool.h>
sq_status sq_error(sq_status status, const char *message);
bool sq_size_add(size_t a, size_t b, size_t *out);
bool sq_size_mul(size_t a, size_t b, size_t *out);
bool sq_size_align(size_t value, size_t alignment, size_t *out);
sq_status sq_options_resolve(const sq_options *input, sq_options *out);
sq_status sq_safetensors_named_data(const sq_safetensors *, const char *, const void **, size_t *);
#endif
