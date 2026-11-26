/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_CUDA_TARGET_PROFILE_H
#define IFN_CUDA_TARGET_PROFILE_H
/* Host launch policy must not depend on __CUDA_ARCH__: NVCC leaves it undefined
 * in its host pass. The build selects exactly one image/profile per artifact. */
#ifndef IFN_CUDA_TARGET_SM
#define IFN_CUDA_TARGET_SM 120
#endif
#ifndef IFN_CUDA_ENABLE_EXPERIMENTAL_AMPERE
#define IFN_CUDA_ENABLE_EXPERIMENTAL_AMPERE 0
#endif
#if IFN_CUDA_ENABLE_EXPERIMENTAL_AMPERE != 0 && IFN_CUDA_ENABLE_EXPERIMENTAL_AMPERE != 1
#error "IFN_CUDA_ENABLE_EXPERIMENTAL_AMPERE must be 0 or 1"
#endif
#if IFN_CUDA_TARGET_SM == 86
#if !IFN_CUDA_ENABLE_EXPERIMENTAL_AMPERE
#error "sm_86 requires explicit experimental Ampere opt-in; not runtime-qualified"
#endif
#define IFN_CUDA_USE_PDL 0
#elif IFN_CUDA_TARGET_SM == 120 || IFN_CUDA_TARGET_SM == 121
#define IFN_CUDA_USE_PDL 1
#else
#error "unsupported CUDA target profile"
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != IFN_CUDA_TARGET_SM * 10
#error "CUDA device image disagrees with the host target profile"
#endif
#endif
