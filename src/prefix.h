/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis */
#ifndef SQ_PREFIX_H
#define SQ_PREFIX_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    unsigned char *data;
    uint32_t position;
    bool valid;
} sq_checkpoint;

typedef struct {
    bool enabled, use_mtp;
    size_t local_bytes, shield_bytes, host_bytes;
    int32_t *tokens;
    unsigned char *shield;
    uint32_t length, shield_tokens;
    int32_t live;
    sq_checkpoint checkpoints[4];
} sq_prefix;
#endif
