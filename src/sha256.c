/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sha256.h"
#include <string.h>
static const uint32_t ifn_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static uint32_t ifn_sha256_rotate(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

static void ifn_sha256_block(ifn_sha256 *s) {
    uint32_t w[64];
    for (unsigned i = 0; i < 16; i++) {
        const unsigned char *p = s->block + 4 * i;
        w[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
    }
    for (unsigned i = 16; i < 64; i++) {
        uint32_t x = w[i - 15], y = w[i - 2];
        w[i] = w[i - 16] + (ifn_sha256_rotate(x, 7) ^ ifn_sha256_rotate(x, 18) ^ (x >> 3)) +
               w[i - 7] + (ifn_sha256_rotate(y, 17) ^ ifn_sha256_rotate(y, 19) ^ (y >> 10));
    }
    uint32_t a = s->state[0], b = s->state[1], c = s->state[2], d = s->state[3], e = s->state[4],
             f = s->state[5], g = s->state[6], h = s->state[7];
    for (unsigned i = 0; i < 64; i++) {
        uint32_t t =
            h + (ifn_sha256_rotate(e, 6) ^ ifn_sha256_rotate(e, 11) ^ ifn_sha256_rotate(e, 25)) +
            ((e & f) ^ (~e & g)) + ifn_sha256_k[i] + w[i];
        uint32_t u =
            (ifn_sha256_rotate(a, 2) ^ ifn_sha256_rotate(a, 13) ^ ifn_sha256_rotate(a, 22)) +
            ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + t;
        d = c;
        c = b;
        b = a;
        a = t + u;
    }
    s->state[0] += a;
    s->state[1] += b;
    s->state[2] += c;
    s->state[3] += d;
    s->state[4] += e;
    s->state[5] += f;
    s->state[6] += g;
    s->state[7] += h;
}

void ifn_sha256_init(ifn_sha256 *s) {
    *s = (ifn_sha256){{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                       0x1f83d9ab, 0x5be0cd19},
                      0,
                      0,
                      {0}};
}

bool ifn_sha256_update(ifn_sha256 *s, const void *data, size_t n) {
    if ((!data && n) || s->bytes > UINT64_MAX / 8 || n > UINT64_MAX / 8 - s->bytes)
        return false;
    const unsigned char *p = data;
    s->bytes += n;
    while (n) {
        size_t take = 64 - s->used;
        if (take > n)
            take = n;
        memcpy(s->block + s->used, p, take);
        s->used += take;
        p += take;
        n -= take;
        if (s->used == 64) {
            ifn_sha256_block(s);
            s->used = 0;
        }
    }
    return true;
}

void ifn_sha256_final(ifn_sha256 *s, unsigned char out[32]) {
    uint64_t bits = s->bytes * 8;
    s->block[s->used++] = 0x80;
    if (s->used > 56) {
        memset(s->block + s->used, 0, 64 - s->used);
        ifn_sha256_block(s);
        s->used = 0;
    }
    memset(s->block + s->used, 0, 56 - s->used);
    for (unsigned i = 0; i < 8; i++)
        s->block[63 - i] = (unsigned char)(bits >> (8 * i));
    ifn_sha256_block(s);
    for (unsigned i = 0; i < 32; i++)
        out[i] = (unsigned char)(s->state[i / 4] >> (24 - 8 * (i % 4)));
}

bool ifn_sha256_hash(const void *data, size_t n, unsigned char out[32]) {
    ifn_sha256 s;
    ifn_sha256_init(&s);
    if (!ifn_sha256_update(&s, data, n))
        return false;
    ifn_sha256_final(&s, out);
    return true;
}

void ifn_sha256_hex(const unsigned char in[32], char out[65]) {
    static const char ifn_sha256_digits[] = "0123456789abcdef";
    for (unsigned i = 0; i < 32; i++) {
        out[2 * i] = ifn_sha256_digits[in[i] >> 4];
        out[2 * i + 1] = ifn_sha256_digits[in[i] & 15];
    }
    out[64] = 0;
}

bool ifn_sha256_unhex(const char *in, size_t n, unsigned char out[32]) {
    if (n != 64 || !in)
        return false;
    for (unsigned i = 0; i < 32; i++) {
        unsigned x = 0;
        for (unsigned j = 0; j < 2; j++) {
            unsigned char c = (unsigned char)in[2 * i + j];
            unsigned v = c >= '0' && c <= '9'   ? c - '0'
                         : c >= 'a' && c <= 'f' ? c - 'a' + 10
                         : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                                : 99;
            if (v > 15)
                return false;
            x = (x << 4) | v;
        }
        out[i] = (unsigned char)x;
    }
    return true;
}
