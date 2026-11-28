/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_INKLING_QUANT_H
#define IFN_INKLING_QUANT_H
#include "ifn_weights.h"
#include "inkling_geometry.h"

/* Existing representation IDs are also the CUDA weight-kind IDs.
 * Exact retains the checkpoint's mixed expert formats, not full precision. */
static inline unsigned ifn_inkling_expert_precision(ifn_weight_mode mode, unsigned layer) {
    return layer < IFN_INKLING_NVFP4_EXPERT_LAYERS || mode == IFN_WEIGHT_NVFP4 ? IFN_WEIGHT_NVFP4
                                                                               : IFN_WEIGHT_FP8;
}

static inline unsigned ifn_inkling_mtp_precision(ifn_weight_mode mode, uint32_t force_bf16) {
    return mode == IFN_WEIGHT_EXACT || force_bf16 ? IFN_WEIGHT_EXACT : IFN_WEIGHT_NVFP4;
}
#endif
