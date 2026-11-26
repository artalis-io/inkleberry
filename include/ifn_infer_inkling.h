/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_INFER_INKLING_H
#define IFN_INFER_INKLING_H
#include "ifn_infer.h"
/* Backend-qualified extension for backend_id="artalis.inkleberry". No JSON passthrough. */
#define IFN_INFER_EXT_INKLING_REQUEST UINT64_C(1)

typedef struct {
    ifn_infer_in hdr;
    uint32_t draft_tokens, adaptive, prompt_lookup;
    float spec_threshold;
    uint32_t thinking_effort, enable_thinking;
} ifn_infer_inkling_request_v1;
#endif
