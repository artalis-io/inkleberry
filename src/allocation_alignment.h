/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_ALLOCATION_ALIGNMENT_H
#define IFN_ALLOCATION_ALIGNMENT_H
#include <stddef.h>
#if defined(_MSC_VER) && !defined(__cplusplus)
/* MSVC's C headers omit max_align_t. Its x64 fundamental types need at most
 * eight-byte alignment; a sixteen-byte extent also preserves malloc alignment
 * when this record is prepended to a caller allocation. */
typedef union {
    long double real;
    void *pointer;
    unsigned char extent[16];
} ifn_allocation_alignment;
#else
typedef max_align_t ifn_allocation_alignment;
#endif
#endif
