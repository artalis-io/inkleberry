/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_WEIGHTS_H
#define SQ_WEIGHTS_H
#include "sq.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef uint32_t sq_weight_mode;

enum { SQ_WEIGHT_EXACT, SQ_WEIGHT_FP8, SQ_WEIGHT_NVFP4 };

typedef struct {
    size_t struct_size;
    uint32_t api_version;
    sq_weight_mode mode;
    uint32_t with_mtp, mtp_bf16;
    int32_t device;
    uint32_t reserved;
} sq_weight_config;

/* clang-format off */
#define SQ_WEIGHT_CONFIG_INIT {sizeof(sq_weight_config),SQ_API_VERSION,SQ_WEIGHT_NVFP4,1,0,0,0}
/* clang-format on */

typedef struct {
    uint64_t arena_bytes, staging_bytes, amax_bytes, fold_temporary_bytes, peak_device_bytes;
    size_t allocations, views;
} sq_weight_plan;

/* CPU-only exact 256-byte allocation planning. No checkpoint or CUDA required.
 * This slice supports device embeddings, full draft vocab, no ablation. */
SQ_API sq_status sq_weights_plan(const sq_weight_config *config, const sq_options *allocator,
                                 sq_weight_plan *out);
#ifdef __cplusplus
}
#endif
#endif
