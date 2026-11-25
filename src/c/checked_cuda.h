/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_CHECKED_CUDA_H
#define SQ_CHECKED_CUDA_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int device;
    void *context;
} sq_device_scope;

int sq_cuda_enter(int, sq_device_scope *);
int sq_cuda_leave(const sq_device_scope *);
int sq_cuda_available(size_t *);
int sq_cuda_launch_status(void);
int sq_cuda_pack(const void *, void *, void *, void *, int, int, int, float *, void *);
int sq_cuda_expert(const void *, const void *, void *, void *, void *, int, int, int, void *);
int sq_cuda_rnn(const void *, void *, void *, void *, void *);
int sq_cuda_fold(const void *, const void *, const void *, void *, void *, int, void *);
#ifdef __cplusplus
}
#endif
#endif
