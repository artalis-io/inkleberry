/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_JSON_H
#define SQ_JSON_H
#include "internal.h"

typedef struct {
    const unsigned char *data;
    size_t size, pos;
} sq_json;

typedef struct {
    const char *name;
    sq_json value;
    bool present;
} sq_json_field;

void sq_json_ws(sq_json *j);
bool sq_json_take(sq_json *j, unsigned char c);
bool sq_json_string(sq_json *j, char *out, size_t capacity);
bool sq_json_u64(sq_json *j, uint64_t *out);
bool sq_json_skip(sq_json *j, unsigned depth);
bool sq_json_object(sq_json *j, sq_json_field *fields, size_t count);
bool sq_json_literal(sq_json value, const char *text);
bool sq_json_text(sq_json value, const char *text);
bool sq_json_done(sq_json *j);
bool sq_json_equal(sq_json a, sq_json b, unsigned depth);
#endif
