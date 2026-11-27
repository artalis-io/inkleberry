/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_INKLING_GEOMETRY_H
#define IFN_INKLING_GEOMETRY_H
/* Fixed Inkling-2-Mini anatomy, not a configurable model registry.
 * CUDA keeps its pinned constants; checked adapters assert agreement. */
#define IFN_INKLING_HIDDEN 2304
#define IFN_INKLING_QUERY 3072
#define IFN_INKLING_KV 1024
#define IFN_INKLING_RELATIVE 384
#define IFN_INKLING_EXPERTS 128
#define IFN_INKLING_INTERMEDIATE 1152
#define IFN_INKLING_VOCAB_PADDED 201024
#define IFN_INKLING_VOCAB 200058
#define IFN_INKLING_MTP_INTERMEDIATE 9216
#define IFN_INKLING_LAYERS 24
#define IFN_INKLING_VISITS 48
#define IFN_INKLING_BLOCK_LAYERS 6
#define IFN_INKLING_LOOP_START 6
#define IFN_INKLING_LOOP_END 18
#define IFN_INKLING_LOOP_PASSES 3
#define IFN_INKLING_NVFP4_EXPERT_LAYERS 20
#define IFN_INKLING_MTP_LAYERS 3
#endif
