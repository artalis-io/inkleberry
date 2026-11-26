/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_WEIGHTS_H
#define IFN_WEIGHTS_H
#include "ifn.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef uint32_t ifn_weight_mode;

enum { IFN_WEIGHT_EXACT, IFN_WEIGHT_FP8, IFN_WEIGHT_NVFP4 };

typedef struct {
    size_t struct_size;
    uint32_t api_version;
    ifn_weight_mode mode;
    uint32_t with_mtp, mtp_bf16;
    int32_t device;
    uint32_t reserved;
} ifn_weight_config;

/* clang-format off */
#define IFN_WEIGHT_CONFIG_INIT {sizeof(ifn_weight_config),IFN_API_VERSION,IFN_WEIGHT_NVFP4,1,0,0,0}
/* clang-format on */

typedef struct {
    uint64_t arena_bytes, staging_bytes, amax_bytes, fold_temporary_bytes, peak_device_bytes;
    size_t allocations, views;
} ifn_weight_plan;

/* CPU-only exact 256-byte allocation planning. No checkpoint or CUDA required.
 * This slice supports device embeddings, full draft vocab, no ablation. */
IFN_API ifn_status ifn_weights_plan(const ifn_weight_config *config, const ifn_options *allocator,
                                    ifn_weight_plan *out);
#ifdef __cplusplus
}
#endif
#endif
