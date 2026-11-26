/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_CUDA_CAPABILITY_H
#define IFN_CUDA_CAPABILITY_H
/* Supported Blackwell families; build CUDA_ARCH for the actual device.
 * Architecture-specific 120a binaries are not a substitute for 121a builds. */
static inline int ifn_cuda_capability_supported(int major, int minor) {
    return major == 12 && (minor == 0 || minor == 1);
}

#endif
