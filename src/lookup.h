/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis */
#ifndef IFN_LOOKUP_H
#define IFN_LOOKUP_H
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    int32_t history[8192];
    uint32_t count;
    uint16_t generation;
    uint64_t hash[16384];
    uint32_t newest[16384], older[16384];
    uint16_t stamp[16384];
} ifn_lookup;

void ifn_lookup_reset(ifn_lookup *);
void ifn_lookup_push(ifn_lookup *, int32_t);
bool ifn_lookup_match(const ifn_lookup *, int32_t[7]);
#endif
