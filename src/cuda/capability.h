/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_CUDA_CAPABILITY_H
#define IFN_CUDA_CAPABILITY_H
#include "target_profile.h"

/* Architecture-specific images cannot be borrowed across GPU families. The
 * experimental 8.6 image requires a build opt-in; this is not qualification. */
static inline int ifn_cuda_capability_supported(int major, int minor) {
    return major == IFN_CUDA_TARGET_SM / 10 && minor == IFN_CUDA_TARGET_SM % 10;
}
#endif
