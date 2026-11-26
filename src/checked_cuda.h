/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_CHECKED_CUDA_H
#define IFN_CHECKED_CUDA_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int device;
    void *context;
} ifn_device_scope;

int ifn_cuda_enter(int, ifn_device_scope *);
int ifn_cuda_leave(const ifn_device_scope *);
int ifn_cuda_available(size_t *);
int ifn_cuda_launch_status(void);
int ifn_cuda_pack(const void *, void *, void *, void *, int, int, int, float *, void *);
int ifn_cuda_expert(const void *, const void *, void *, void *, void *, int, int, int, void *);
int ifn_cuda_rnn(const void *, void *, void *, void *, void *);
int ifn_cuda_fold(const void *, const void *, const void *, void *, void *, int, void *);
size_t ifn_adaptive_bytes(void);
void ifn_adaptive_init(void *);
int ifn_adaptive_prepare(void *, const int32_t *, const int32_t *, const float *, float, int32_t *,
                         int32_t *, const int32_t *, int32_t *, const int32_t *, int32_t *,
                         int32_t *, void *);
int ifn_adaptive_branch_begin(void *, int, void *);
int ifn_adaptive_branch_end(void *);
int ifn_adaptive_instantiate(void *);
int ifn_adaptive_launch(void *, void *);
int ifn_adaptive_cleanup(void *);
int ifn_adaptive_live(const void *);
#ifdef __cplusplus
}
#endif
#endif
