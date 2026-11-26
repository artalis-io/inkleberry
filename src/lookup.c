/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Pinned PromptLookupIndex translation: wrapping unsigned hash, 16-token
 * confirmation, two candidates and generation-stamped 8192-token history. */
#include "lookup.h"
#include <string.h>

static uint64_t hash4(int32_t a, int32_t b, int32_t c, int32_t d) {
    uint64_t h = (uint64_t)(int64_t)a * UINT64_C(0x9e3779b97f4a7c15);
    h = (h ^ (uint64_t)(int64_t)b) * UINT64_C(0xbf58476d1ce4e5b9);
    h = (h ^ (uint64_t)(int64_t)c) * UINT64_C(0x94d049bb133111eb);
    h = (h ^ (uint64_t)(int64_t)d) * UINT64_C(0x9e3779b97f4a7c15);
    return h ^ (h >> 31);
}

static int32_t at(const sq_lookup *s, uint32_t i) {
    return s->history[i & 8191];
}

static uint64_t suffix(const sq_lookup *s) {
    uint32_t n = s->count;
    return hash4(at(s, n - 4), at(s, n - 3), at(s, n - 2), at(s, n - 1));
}

void sq_lookup_reset(sq_lookup *s) {
    s->count = 0;
    s->generation = (uint16_t)(s->generation + 1u);
    if (!s->generation) {
        memset(s->stamp, 0, sizeof s->stamp);
        s->generation = 1;
    }
}

void sq_lookup_push(sq_lookup *s, int32_t token) {
    uint32_t i = s->count;
    s->history[i & 8191] = token;
    s->count = i + 1;
    if (s->count < 4)
        return;
    uint64_t h = suffix(s);
    size_t slot = (size_t)(h & 16383);
    if (s->stamp[slot] == s->generation && s->hash[slot] == h) {
        s->older[slot] = s->newest[slot];
        s->newest[slot] = s->count;
    } else {
        s->hash[slot] = h;
        s->newest[slot] = s->count;
        s->older[slot] = 0;
        s->stamp[slot] = s->generation;
    }
}

bool sq_lookup_match(const sq_lookup *s, int32_t out[7]) {
    uint32_t n = s->count;
    if (n < 24)
        return false;
    uint64_t h = suffix(s);
    size_t slot = (size_t)(h & 16383);
    if (s->stamp[slot] != s->generation || s->hash[slot] != h)
        return false;
    uint32_t candidate = s->newest[slot] == n ? s->older[slot] : s->newest[slot];
    if (candidate < 16 || candidate >= n || n - candidate <= 7)
        return false;
    uint32_t low = n > 8192 ? n - 8192 : 0;
    if (candidate - 16 < low)
        return false;
    for (uint32_t k = 1; k <= 16; k++)
        if (at(s, candidate - k) != at(s, n - k))
            return false;
    for (uint32_t j = 0; j < 7; j++)
        out[j] = at(s, candidate + j);
    return true;
}
