/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_INKLING_H
#define SQ_INKLING_H
#include "internal.h"

typedef struct {
    char name[128];
    sq_dtype dtype;
    uint32_t rank;
    uint64_t shape[3];
    uint32_t source;
} sq_inkling_spec;

typedef sq_status (*sq_spec_fn)(void *context, const sq_inkling_spec *spec);
sq_status sq_inkling_visit_specs(uint32_t include_mtp, sq_spec_fn visit, void *context);
sq_status sq_inkling_check_descriptor(const sq_tensor_info *tensor, const sq_inkling_spec *spec);
#endif
