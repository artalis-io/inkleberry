/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_CUDA_GEOMETRY_CONTRACT_HPP
#define IFN_CUDA_GEOMETRY_CONTRACT_HPP
#include "../inkling_geometry.h"
static_assert(IFN_INKLING_HIDDEN == ifn_cuda::IFN_CUDA_HIDDEN, "Inkling hidden mismatch");
static_assert(IFN_INKLING_QUERY == ifn_cuda::IFN_CUDA_Q_DIM, "Inkling query mismatch");
static_assert(IFN_INKLING_KV == ifn_cuda::IFN_CUDA_KV_DIM, "Inkling kv mismatch");
static_assert(IFN_INKLING_RELATIVE == ifn_cuda::IFN_CUDA_R_DIM, "Inkling relative mismatch");
static_assert(IFN_INKLING_EXPERTS == ifn_cuda::IFN_CUDA_NUM_EXPERTS, "Inkling experts mismatch");
static_assert(IFN_INKLING_INTERMEDIATE == ifn_cuda::IFN_CUDA_INTERMEDIATE,
              "Inkling intermediate mismatch");
static_assert(IFN_INKLING_VOCAB_PADDED == ifn_cuda::IFN_CUDA_VOCAB_PADDED,
              "Inkling vocab_padded mismatch");
static_assert(IFN_INKLING_VOCAB == ifn_cuda::IFN_CUDA_VOCAB_UNPADDED, "Inkling vocab mismatch");
static_assert(IFN_INKLING_MTP_INTERMEDIATE == ifn_cuda::IFN_CUDA_MTP_INTERMEDIATE,
              "Inkling mtp_intermediate mismatch");
static_assert(IFN_INKLING_LAYERS == ifn_cuda::IFN_CUDA_NUM_LAYERS, "Inkling layers mismatch");
static_assert(IFN_INKLING_VISITS == ifn_cuda::IFN_CUDA_NUM_VISITS, "Inkling visits mismatch");
static_assert(IFN_INKLING_BLOCK_LAYERS == ifn_cuda::IFN_CUDA_BLOCK_LAYERS,
              "Inkling block_layers mismatch");
static_assert(IFN_INKLING_LOOP_START == ifn_cuda::IFN_CUDA_LOOP_START,
              "Inkling loop_start mismatch");
static_assert(IFN_INKLING_LOOP_END == ifn_cuda::IFN_CUDA_LOOP_END, "Inkling loop_end mismatch");
static_assert(IFN_INKLING_LOOP_PASSES == ifn_cuda::IFN_CUDA_LOOP_PASSES,
              "Inkling loop_passes mismatch");
static_assert(IFN_INKLING_MTP_LAYERS == ifn_cuda::IFN_CUDA_MTP_LAYERS,
              "Inkling mtp_layers mismatch");
static_assert(ifn_cuda::ifn_cuda_is_nvfp4_expert_layer(IFN_INKLING_NVFP4_EXPERT_LAYERS - 1) &&
                  !ifn_cuda::ifn_cuda_is_nvfp4_expert_layer(IFN_INKLING_NVFP4_EXPERT_LAYERS),
              "Inkling expert source split mismatch");
#endif
