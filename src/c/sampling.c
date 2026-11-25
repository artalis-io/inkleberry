/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "engine_internal.h"
#include <math.h>

sq_status sq_decode_resolve(const sq_decode_options *input, sq_decode_options *out) {
    const sq_decode_options defaults = SQ_DECODE_OPTIONS_INIT;
    *out = defaults;
    if (input) {
        if (input->struct_size < sizeof *input || input->api_version != SQ_API_VERSION)
            return sq_error(SQ_INVALID_ARGUMENT, "unsupported decode options size/version");
        *out = *input;
    }
    if (!isfinite(out->temperature) || out->temperature < 0 || !isfinite(out->top_p) ||
        out->top_p <= 0 || out->top_p > 1 || out->top_k < 0 || out->top_k > 200058 ||
        out->draft_tokens > 7 || out->reserved)
        return sq_error(SQ_INVALID_ARGUMENT, "invalid temperature/top-p/top-k/draft options");
    return SQ_OK;
}
