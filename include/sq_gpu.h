/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_GPU_H
#define SQ_GPU_H
#include "sq_weights.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct sq_gpu_weights sq_gpu_weights;

typedef struct {
    char label[128];
    uint64_t offset, bytes;
} sq_weight_span;

typedef struct {
    char label[128];
    uint32_t type, precision, rows, cols, experts;
    uint64_t data, scales, row_scales, expert_scale;
    float global_scale;
} sq_weight_view;

enum { SQ_WEIGHT_RAW, SQ_WEIGHT_LINEAR, SQ_WEIGHT_EXPERT };

#define SQ_WEIGHT_NO_OFFSET UINT64_MAX
/* Optional libsq-gpu backend. create plans/allocates host bookkeeping only.
 * load borrows the validated checkpoint until return, then owns independent
 * device copies. Calls on one handle require exclusive access; overlap is BUSY.
 * Each call selects the configured device and attempts to restore the caller's
 * device. No cudaDeviceReset or caller-owned context/stream teardown occurs. */
SQ_API sq_status sq_gpu_weights_create(const sq_weight_config *config, const sq_options *allocator,
                                       sq_gpu_weights **out);
SQ_API sq_status sq_gpu_weights_load(sq_gpu_weights *weights,
                                     const sq_inkling_checkpoint *checkpoint);
SQ_API sq_status sq_gpu_weights_get_plan(sq_gpu_weights *weights, sq_weight_plan *out);
SQ_API sq_status sq_gpu_weights_get_span(sq_gpu_weights *weights, size_t index,
                                         sq_weight_span *out);
SQ_API sq_status sq_gpu_weights_get_view(sq_gpu_weights *weights, size_t index,
                                         sq_weight_view *out);
/* Bounded explicit D2H diagnostics; never returns a writable device pointer.
 * offset/bytes are relative to one payload span, excluding arena padding. */
SQ_API sq_status sq_gpu_weights_read(sq_gpu_weights *weights, size_t span, uint64_t offset,
                                     void *out, size_t bytes);
/* Frees owned resources, never the checkpoint. On incomplete CUDA cleanup the
 * handle is retained for retry. load errors also retain the created handle;
 * call destroy even after failure. NULL handles are accepted. The first backend
 * error is returned; a failed device restoration is reported, not hidden. */
SQ_API sq_status sq_gpu_weights_destroy(sq_gpu_weights **weights);
#ifdef __cplusplus
}
#endif
#endif
