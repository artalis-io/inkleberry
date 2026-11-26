/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "engine_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define IFN_SAMPLING_LOCAL_REQUIRE(x)                                                              \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "decode options:%d\n", __LINE__);                                      \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

int main(void) {
    ifn_decode_options options = IFN_DECODE_OPTIONS_INIT, out;
    IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(NULL, &out) == IFN_OK && out.temperature == 0 &&
                               out.draft_tokens == 0);
    options.temperature = 0.8f;
    options.top_p = 0.95f;
    options.top_k = 40;
    options.seed = UINT64_MAX;
    options.draft_tokens = 7;
    IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(&options, &out) == IFN_OK &&
                               out.seed == UINT64_MAX && out.draft_tokens == 7 && out.top_k == 40);
    const float invalid_t[] = {NAN, INFINITY, -INFINITY, -1};
    for (size_t i = 0; i < sizeof invalid_t / sizeof *invalid_t; i++) {
        ifn_decode_options bad = options;
        bad.temperature = invalid_t[i];
        IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(&bad, &out) == IFN_INVALID_ARGUMENT);
    }
    const float invalid_p[] = {NAN, INFINITY, -1, 0, 1.01f};
    for (size_t i = 0; i < sizeof invalid_p / sizeof *invalid_p; i++) {
        ifn_decode_options bad = options;
        bad.top_p = invalid_p[i];
        IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(&bad, &out) == IFN_INVALID_ARGUMENT);
    }
    for (unsigned i = 0; i < 6; i++) {
        ifn_decode_options bad = options;
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
        IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(&bad, &out) == IFN_INVALID_ARGUMENT);
    }
    options = (ifn_decode_options)IFN_DECODE_OPTIONS_INIT;
    options.adaptive = 1;
    IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(&options, &out) == IFN_INVALID_ARGUMENT);
    options.draft_tokens = 7;
    options.prompt_lookup = 1;
    IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(&options, &out) == IFN_OK && out.adaptive &&
                               out.spec_threshold == 0.8f);
    options.spec_threshold = NAN;
    IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(&options, &out) == IFN_INVALID_ARGUMENT);
    options.spec_threshold = 1.01f;
    IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(&options, &out) == IFN_INVALID_ARGUMENT);
    for (unsigned i = 0; i < 7; i++) {
        ifn_decode_options bad = IFN_DECODE_OPTIONS_INIT;
        bad.draft_tokens = 7;
        if (i == 0)
            bad.adaptive = 2;
        if (i == 1)
            bad.prompt_lookup = 2;
        if (i == 2)
            bad.prompt_lookup = 1;
        if (i == 3)
            bad.adaptive_reserved = 1;
        if (i == 4)
            bad.spec_threshold = 0;
        if (i == 5)
            bad.spec_threshold = -1;
        if (i == 6)
            bad.spec_threshold = INFINITY;
        IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(&bad, &out) == IFN_INVALID_ARGUMENT);
    }
    options = (ifn_decode_options)IFN_DECODE_OPTIONS_INIT;
    size_t legacy = offsetof(ifn_decode_options, adaptive);
    options.struct_size = legacy;
    options.adaptive = 99;
    options.spec_threshold = NAN;
    unsigned char *old = malloc(legacy);
    IFN_SAMPLING_LOCAL_REQUIRE(old);
    memcpy(old, &options, legacy);
    IFN_SAMPLING_LOCAL_REQUIRE(
        ifn_decode_resolve((const ifn_decode_options *)(const void *)old, &out) == IFN_OK &&
        !out.adaptive && out.spec_threshold == 0.8f);
    free(old);
    options.struct_size = legacy + 4;
    IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(&options, &out) == IFN_INVALID_ARGUMENT);
    options = (ifn_decode_options)IFN_DECODE_OPTIONS_INIT;
    size_t previous = offsetof(ifn_decode_options, system_prefix_tokens);
    options.struct_size = previous;
    options.system_prefix_tokens = UINT32_MAX;
    options.prefix_reserved = 1;
    old = malloc(previous);
    IFN_SAMPLING_LOCAL_REQUIRE(old);
    memcpy(old, &options, previous);
    IFN_SAMPLING_LOCAL_REQUIRE(
        ifn_decode_resolve((const ifn_decode_options *)(const void *)old, &out) == IFN_OK &&
        !out.system_prefix_tokens && !out.prefix_reserved);
    free(old);
    options.struct_size = previous + 4;
    IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(&options, &out) == IFN_INVALID_ARGUMENT);
    options.struct_size = sizeof options;
    IFN_SAMPLING_LOCAL_REQUIRE(ifn_decode_resolve(&options, &out) == IFN_INVALID_ARGUMENT);
    puts("sampling option boundaries, NaN/Inf rejection and full-width seed preservation passed");
    return 0;
}
