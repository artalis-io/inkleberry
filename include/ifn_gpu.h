/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_GPU_H
#define IFN_GPU_H
#include "ifn_weights.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ifn_gpu_weights ifn_gpu_weights;

typedef struct {
    char label[128];
    uint64_t offset, bytes;
} ifn_weight_span;

typedef struct {
    char label[128];
    uint32_t type, precision, rows, cols, experts;
    uint64_t data, scales, row_scales, expert_scale;
    float global_scale;
} ifn_weight_view;

enum { IFN_WEIGHT_RAW, IFN_WEIGHT_LINEAR, IFN_WEIGHT_EXPERT };

#define IFN_WEIGHT_NO_OFFSET UINT64_MAX
/* Optional libinkleberry-gpu backend. create plans/allocates host bookkeeping only.
 * load borrows the validated checkpoint until return, then owns independent
 * device copies. Calls on one handle require exclusive access; overlap is BUSY.
 * Each call selects the configured device and attempts to restore the caller's
 * device. No cudaDeviceReset or caller-owned context/stream teardown occurs. */
IFN_API ifn_status ifn_gpu_weights_create(const ifn_weight_config *config,
                                          const ifn_options *allocator, ifn_gpu_weights **out);
IFN_API ifn_status ifn_gpu_weights_load(ifn_gpu_weights *weights,
                                        const ifn_inkling_checkpoint *checkpoint);
IFN_API ifn_status ifn_gpu_weights_get_plan(ifn_gpu_weights *weights, ifn_weight_plan *out);
IFN_API ifn_status ifn_gpu_weights_get_span(ifn_gpu_weights *weights, size_t index,
                                            ifn_weight_span *out);
IFN_API ifn_status ifn_gpu_weights_get_view(ifn_gpu_weights *weights, size_t index,
                                            ifn_weight_view *out);
/* Bounded explicit D2H diagnostics; never returns a writable device pointer.
 * offset/bytes are relative to one payload span, excluding arena padding. */
IFN_API ifn_status ifn_gpu_weights_read(ifn_gpu_weights *weights, size_t span, uint64_t offset,
                                        void *out, size_t bytes);
/* Frees owned resources, never the checkpoint. On incomplete CUDA cleanup the
 * handle is retained for retry. load errors also retain the created handle;
 * call destroy even after failure. NULL handles are accepted. The first backend
 * error is returned; a failed device restoration is reported, not hidden. */
IFN_API ifn_status ifn_gpu_weights_destroy(ifn_gpu_weights **weights);
#ifdef __cplusplus
}
#endif
#endif
