/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_SHA256_H
#define IFN_SHA256_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    size_t used;
    unsigned char block[64];
} ifn_sha256;

void ifn_sha256_init(ifn_sha256 *);
bool ifn_sha256_update(ifn_sha256 *, const void *, size_t);
void ifn_sha256_final(ifn_sha256 *, unsigned char[32]);
bool ifn_sha256_hash(const void *, size_t, unsigned char[32]);
void ifn_sha256_hex(const unsigned char[32], char[65]);
bool ifn_sha256_unhex(const char *, size_t, unsigned char[32]);
#endif
