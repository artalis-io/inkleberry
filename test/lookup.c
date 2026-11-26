/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "lookup.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(x)                                                                                 \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "lookup:%d\n", __LINE__);                                              \
            free(s);                                                                               \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

int main(void) {
    ifn_lookup *s = calloc(1, sizeof *s);
    if (!s)
        return 1;
    int32_t out[7];
    for (unsigned i = 0; i < 7; i++)
        out[i] = -999;
    ifn_lookup_reset(s);
    REQUIRE(!ifn_lookup_match(s, out) && out[0] == -999);
    for (int i = 0; i < 96; i++)
        ifn_lookup_push(s, i % 32);
    REQUIRE(ifn_lookup_match(s, out));
    for (int i = 0; i < 7; i++)
        REQUIRE(out[i] == i);
    /* Same four-token suffix is insufficient: all sixteen tokens must match. */
    ifn_lookup_reset(s);
    for (int i = 0; i < 40; i++)
        ifn_lookup_push(s, i);
    for (int i = 0; i < 12; i++)
        ifn_lookup_push(s, 100 + i);
    for (int i = 36; i < 40; i++)
        ifn_lookup_push(s, i);
    REQUIRE(!ifn_lookup_match(s, out));
    /* Generation stamps prevent cross-request proposals, including uint16 wrap. */
    ifn_lookup_reset(s);
    REQUIRE(!ifn_lookup_match(s, out));
    s->generation = UINT16_MAX;
    memset(s->stamp, 0xff, sizeof s->stamp);
    ifn_lookup_reset(s);
    REQUIRE(s->generation == 1);
    for (size_t i = 0; i < 16384; i++)
        REQUIRE(s->stamp[i] == 0);
    /* A once-valid candidate outside the live ring is rejected. */
    for (int i = 0; i < 32; i++)
        ifn_lookup_push(s, i);
    for (int i = 0; i < 8200; i++)
        ifn_lookup_push(s, 1000 + i);
    for (int i = 0; i < 16; i++)
        ifn_lookup_push(s, i);
    REQUIRE(!ifn_lookup_match(s, out));
    /* Hits across ring wrap still refer to exact live sequences. */
    ifn_lookup_reset(s);
    size_t hits = 0;
    for (int i = 0; i < 20000; i++) {
        ifn_lookup_push(s, (i % 64) - 32);
        if (ifn_lookup_match(s, out)) {
            hits++;
            for (int j = 0; j < 7; j++)
                REQUIRE(out[j] == ((i + 1 + j) % 64) - 32);
        }
    }
    REQUIRE(hits > 19000);
    free(s);
    puts("lookup suffix confirmation, request isolation, generation/ring wrap and stale history "
         "passed");
    return 0;
}
