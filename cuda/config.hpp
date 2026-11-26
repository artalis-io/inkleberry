// Inkleberry Finn: identifier-only namespace adaptation by Artalis; see config/native-namespace.json.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace ifn_cuda {

// Pinned Inkling-2-Mini text + DSpark MTP geometry constants
constexpr int IFN_CUDA_HIDDEN = 2304;
constexpr int IFN_CUDA_NUM_LAYERS = 24;
constexpr int IFN_CUDA_NUM_VISITS = 48;
constexpr int IFN_CUDA_LOOP_START = 6;
constexpr int IFN_CUDA_LOOP_END = 18;
constexpr int IFN_CUDA_LOOP_PASSES = 3;
constexpr int IFN_CUDA_BLOCK_LAYERS = 6;
constexpr int IFN_CUDA_GATE_RANK = 64;

constexpr int IFN_CUDA_NUM_HEADS = 24;
constexpr int IFN_CUDA_NUM_KV_HEADS = 8;
constexpr int IFN_CUDA_HEAD_DIM = 128;
constexpr int IFN_CUDA_Q_DIM = IFN_CUDA_NUM_HEADS * IFN_CUDA_HEAD_DIM;     // 3072
constexpr int IFN_CUDA_KV_DIM = IFN_CUDA_NUM_KV_HEADS * IFN_CUDA_HEAD_DIM; // 1024
constexpr int IFN_CUDA_GQA_RATIO = IFN_CUDA_NUM_HEADS / IFN_CUDA_NUM_KV_HEADS; // 3

constexpr int IFN_CUDA_D_REL = 16;
constexpr int IFN_CUDA_R_DIM = IFN_CUDA_NUM_HEADS * IFN_CUDA_D_REL;        // 384
constexpr int IFN_CUDA_REL_EXTENT = 1024;
constexpr int IFN_CUDA_SWA_WINDOW = 1024;
constexpr int IFN_CUDA_CONV_TAPS = 4;

constexpr int IFN_CUDA_NUM_EXPERTS = 128;
constexpr int IFN_CUDA_TOP_K = 8;
constexpr int IFN_CUDA_INTERMEDIATE = 1152;
constexpr int IFN_CUDA_VOCAB_PADDED = 201024;
constexpr int IFN_CUDA_VOCAB_UNPADDED = 200058;
constexpr int IFN_CUDA_EOS_TOKEN = 200006;

constexpr float IFN_CUDA_RMS_EPS = 1e-6f;
constexpr float IFN_CUDA_QK_CLIP = 22.627416997969522f;
constexpr float IFN_CUDA_REL_CLIP = 8.0f;
constexpr int IFN_CUDA_LOG_SCALE_FLOOR = 8192;
constexpr float IFN_CUDA_LOG_SCALE_ALPHA = 0.1f;
constexpr float IFN_CUDA_ROUTE_SCALE = 8.0f;
constexpr float IFN_CUDA_GATE_SOFTCAP = 4.0f;
constexpr float IFN_CUDA_UP_SOFTCAP = 25.0f;
constexpr float IFN_CUDA_FINAL_SOFTCAP = 30.0f;
constexpr float IFN_CUDA_INV_SQRT_HEAD_DIM = 0.08838834764831845f; // 1/sqrt(128)
constexpr float IFN_CUDA_INV_SQRT_HIDDEN = 0.020833333333333332f;  // 1/sqrt(2304) = 1/48
constexpr float IFN_CUDA_INV_SQRT_5 = 0.4472135954999579f;         // 1/sqrt(5)
constexpr float IFN_CUDA_INV_SQRT_9 = 0.3333333333333333f;         // 1/sqrt(9)

// DSpark MTP constants
constexpr int IFN_CUDA_MTP_LAYERS = 3;
constexpr int IFN_CUDA_MTP_INTERMEDIATE = 9216;
constexpr int IFN_CUDA_MTP_BLOCK_SIZE = 8; // anchor + 7 drafts
constexpr int IFN_CUDA_MTP_MAX_DRAFTS = 7;
constexpr int IFN_CUDA_MTP_MARKOV_RANK = 256;
constexpr float IFN_CUDA_MTP_ROPE_THETA = 1000000.0f;
constexpr int IFN_CUDA_MTP_TARGET_VISITS[4] = {5, 23, 41, 48};

constexpr bool ifn_cuda_is_global_layer(int physical_layer) {
    return (physical_layer + 1) % IFN_CUDA_BLOCK_LAYERS == 0; // 5, 11, 17, 23
}

constexpr bool ifn_cuda_is_nvfp4_expert_layer(int physical_layer) {
    return physical_layer < 20; // 0..19 are NVFP4, 20..23 are MXFP8
}

constexpr bool ifn_cuda_is_mtp_local_layer(int l) {
    return l == 0 || l == 2;
}

enum class ifn_cuda_quant_mode {
    IFN_CUDA_EXACT, // Official checkpoint: NVFP4 L0-19, MXFP8 L20-23, BF16 dense & LM head
    IFN_CUDA_FP8,   // Official experts + FP8 E4M3 dense attention & LM head
    IFN_CUDA_NVFP4  // Aggressive all-NVFP4: L0-23 experts, dense attention, LM head, DSpark
};

enum class ifn_cuda_kv_format {
    IFN_CUDA_BF16 = 0, // 2.000 B/elem (exact parity)
    IFN_CUDA_FP8  = 1, // 1.016 B/elem (E4M3 + per-head FP16 scale -> up to 448K context)
    IFN_CUDA_Q4   = 2  // 0.531 B/elem (signed INT4 [-7..7] + group-64 FP16 scale -> up to 1M context!)
};

// Device-resident sampling config (uploaded via H2D before generate; zero graph rebuild!)
// inv_temp == 0.0f => pure greedy argmax with zero extra memory traffic.
struct ifn_cuda_sample_config {
    float inv_temp;
    float top_p;
    int32_t top_k;
    uint32_t pad;
    uint64_t seed;
};

// LoopCD-Hidden (arXiv:2610.02185 Eq. 2): h' = h_R + omega * (h_R - h_ref) before Coda layers.
// Lives in device memory so CUDA Graphs never rebuild when omega or ref_pass changes per request.
struct ifn_cuda_loop_cd_config {
    float omega;      // 0.0 = disabled (bit-exact identity); > 0 = contrastive extrapolation
    int32_t ref_pass; // 1 = h_1 (after pass 0, vi=17), 2 = h_2 (after pass 1, vi=29)
};

constexpr int IFN_CUDA_KV_SPLIT_K = 64;    // 8 KV heads x 64 bulk splits = 512 CTAs (3.0 blocks/SM!)
constexpr int IFN_CUDA_KV_LOCAL_SPLITS = 17; // 17 x 64 = 1088 >= 1024+8 (1 single tile per CTA!)
constexpr int IFN_CUDA_KV_TOTAL_SPLITS = IFN_CUDA_KV_SPLIT_K + IFN_CUDA_KV_LOCAL_SPLITS; // 81 slots in split_buf
constexpr int IFN_CUDA_KV_Q4_GROUP = 64;  // 2 groups per 128-dim head
constexpr int IFN_CUDA_KV_Q4_GROUPS_PER_HEAD = IFN_CUDA_HEAD_DIM / IFN_CUDA_KV_Q4_GROUP; // 2
constexpr int IFN_CUDA_KV_Q4_SCALES_PER_TOK = IFN_CUDA_NUM_KV_HEADS * IFN_CUDA_KV_Q4_GROUPS_PER_HEAD; // 16

struct ifn_cuda_visit_info {
    int visit_idx;
    int physical_layer;
    int pass;   // 0 for prelude/coda, 0..2 for loop
    bool is_loop;
    bool is_global;
};

inline void ifn_cuda_build_visit_schedule(ifn_cuda_visit_info out[IFN_CUDA_NUM_VISITS]) {
    int idx = 0;
    for (int l = 0; l < IFN_CUDA_LOOP_START; ++l) {
        out[idx] = {idx, l, 0, false, ifn_cuda_is_global_layer(l)};
        idx++;
    }
    for (int p = 0; p < IFN_CUDA_LOOP_PASSES; ++p) {
        for (int l = IFN_CUDA_LOOP_START; l < IFN_CUDA_LOOP_END; ++l) {
            out[idx] = {idx, l, p, true, ifn_cuda_is_global_layer(l)};
            idx++;
        }
    }
    for (int l = IFN_CUDA_LOOP_END; l < IFN_CUDA_NUM_LAYERS; ++l) {
        out[idx] = {idx, l, 0, false, ifn_cuda_is_global_layer(l)};
        idx++;
    }
}

#define IFN_CUDA_CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err__));                                           \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

} // namespace ifn_cuda
