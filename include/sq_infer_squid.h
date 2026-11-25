/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_INFER_SQUID_H
#define SQ_INFER_SQUID_H
#include "sq_infer.h"
/* Backend-qualified extension for backend_id="artalis.squid". No JSON passthrough. */
#define SQ_INFER_EXT_SQUID_REQUEST UINT64_C(1)

typedef struct {
    sq_infer_in hdr;
    uint32_t draft_tokens, adaptive, prompt_lookup;
    float spec_threshold;
    uint32_t thinking_effort, enable_thinking;
} sq_infer_squid_request_v1;
#endif
