/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis */
#ifndef SQ_LOOKUP_H
#define SQ_LOOKUP_H
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    int32_t history[8192];
    uint32_t count;
    uint16_t generation;
    uint64_t hash[16384];
    uint32_t newest[16384], older[16384];
    uint16_t stamp[16384];
} sq_lookup;

void sq_lookup_reset(sq_lookup *);
void sq_lookup_push(sq_lookup *, int32_t);
bool sq_lookup_match(const sq_lookup *, int32_t[7]);
#endif
