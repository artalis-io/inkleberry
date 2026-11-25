/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "engine_internal.h"
#include <math.h>
#include <stdio.h>
#define REQUIRE(x)                                                                                 \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "decode options:%d\n", __LINE__);                                      \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

int main(void) {
    sq_decode_options options = SQ_DECODE_OPTIONS_INIT, out;
    REQUIRE(sq_decode_resolve(NULL, &out) == SQ_OK && out.temperature == 0 &&
            out.draft_tokens == 0);
    options.temperature = 0.8f;
    options.top_p = 0.95f;
    options.top_k = 40;
    options.seed = UINT64_MAX;
    options.draft_tokens = 7;
    REQUIRE(sq_decode_resolve(&options, &out) == SQ_OK && out.seed == UINT64_MAX &&
            out.draft_tokens == 7 && out.top_k == 40);
    const float invalid_t[] = {NAN, INFINITY, -INFINITY, -1};
    for (size_t i = 0; i < sizeof invalid_t / sizeof *invalid_t; i++) {
        sq_decode_options bad = options;
        bad.temperature = invalid_t[i];
        REQUIRE(sq_decode_resolve(&bad, &out) == SQ_INVALID_ARGUMENT);
    }
    const float invalid_p[] = {NAN, INFINITY, -1, 0, 1.01f};
    for (size_t i = 0; i < sizeof invalid_p / sizeof *invalid_p; i++) {
        sq_decode_options bad = options;
        bad.top_p = invalid_p[i];
        REQUIRE(sq_decode_resolve(&bad, &out) == SQ_INVALID_ARGUMENT);
    }
    for (unsigned i = 0; i < 6; i++) {
        sq_decode_options bad = options;
        if (i == 0)
            bad.top_k = -1;
        if (i == 1)
            bad.top_k = 200059;
        if (i == 2)
            bad.draft_tokens = 8;
        if (i == 3)
            bad.reserved = 1;
        if (i == 4)
            bad.struct_size = 0;
        if (i == 5)
            bad.api_version = 0;
        REQUIRE(sq_decode_resolve(&bad, &out) == SQ_INVALID_ARGUMENT);
    }
    puts("sampling option boundaries, NaN/Inf rejection and full-width seed preservation passed");
    return 0;
}
