/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "engine_internal.h"
#include <math.h>
#include <string.h>

sq_status sq_decode_resolve(const sq_decode_options *input, sq_decode_options *out) {
    const sq_decode_options defaults = SQ_DECODE_OPTIONS_INIT;
    *out = defaults;
    if (input) {
        size_t size;
        uint32_t version;
        memcpy(&size, input, sizeof size);
        const size_t legacy = offsetof(sq_decode_options, adaptive);
        const size_t adaptive = offsetof(sq_decode_options, system_prefix_tokens);
        if (size != legacy && size != adaptive && size < sizeof *input)
            return sq_error(SQ_INVALID_ARGUMENT, "unsupported decode options size");
        memcpy(&version, (const unsigned char *)input + offsetof(sq_decode_options, api_version),
               sizeof version);
        if (version != SQ_API_VERSION)
            return sq_error(SQ_INVALID_ARGUMENT, "unsupported decode options version");
        memcpy(out, input, size < sizeof *input ? size : sizeof *input);
    }
    if (!isfinite(out->temperature) || out->temperature < 0 || !isfinite(out->top_p) ||
        out->top_p <= 0 || out->top_p > 1 || out->top_k < 0 || out->top_k > 200058 ||
        out->draft_tokens > 7 || out->reserved || out->adaptive > 1 || out->prompt_lookup > 1 ||
        out->adaptive_reserved || out->prefix_reserved || !isfinite(out->spec_threshold) ||
        out->spec_threshold <= 0 || out->spec_threshold > 1 ||
        (out->adaptive && out->draft_tokens != 7) || (out->prompt_lookup && !out->adaptive))
        return sq_error(SQ_INVALID_ARGUMENT, "invalid temperature/top-p/top-k/draft options");
    return SQ_OK;
}
