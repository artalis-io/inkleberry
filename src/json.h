/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_JSON_H
#define IFN_JSON_H
#include "internal.h"

typedef struct {
    const unsigned char *data;
    size_t size, pos;
} ifn_json;

typedef struct {
    const char *name;
    ifn_json value;
    bool present;
} ifn_json_field;

void ifn_json_ws(ifn_json *j);
bool ifn_json_take(ifn_json *j, unsigned char c);
bool ifn_json_string(ifn_json *j, char *out, size_t capacity);
bool ifn_json_u64(ifn_json *j, uint64_t *out);
bool ifn_json_skip(ifn_json *j, unsigned depth);
bool ifn_json_object(ifn_json *j, ifn_json_field *fields, size_t count);
bool ifn_json_literal(ifn_json value, const char *text);
bool ifn_json_text(ifn_json value, const char *text);
bool ifn_json_done(ifn_json *j);
bool ifn_json_equal(ifn_json a, ifn_json b, unsigned depth);
#endif
