/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_INKLING_H
#define IFN_INKLING_H
#include "internal.h"

typedef struct {
    char name[128];
    ifn_dtype dtype;
    uint32_t rank;
    uint64_t shape[3];
    uint32_t source;
} ifn_inkling_spec;

typedef ifn_status (*ifn_spec_fn)(void *context, const ifn_inkling_spec *spec);
ifn_status ifn_inkling_visit_specs(uint32_t include_mtp, ifn_spec_fn visit, void *context);
ifn_status ifn_inkling_check_descriptor(const ifn_tensor_info *tensor,
                                        const ifn_inkling_spec *spec);
#endif
