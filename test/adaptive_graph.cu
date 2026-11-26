/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Force each real conditional branch using the original device selector. */
#include "checked_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void ifn_adaptive_graph_answer(int32_t *out, int32_t *count, int width) {
    int i = threadIdx.x;
    if (i < 8)
        out[i] = width * 100 + i;
    if (i == 0)
        *count = width;
}

int main() {
    void *state = std::malloc(ifn_adaptive_bytes());
    if (!state)
        return 1;
    ifn_adaptive_init(state);
    int32_t *d = nullptr, *host = nullptr;
    float *confidence = nullptr;
    cudaStream_t stream = nullptr;
    int failed = 0;
#define IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(x)                                                        \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            std::fprintf(stderr, "conditional graph:%d\n", __LINE__);                              \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)
    IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d), 32 * 4) ==
                                     cudaSuccess);
    IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(cudaMalloc(reinterpret_cast<void **>(&confidence), 7 * 4) ==
                                     cudaSuccess);
    IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(cudaMallocHost(reinterpret_cast<void **>(&host), 10 * 4) ==
                                     cudaSuccess);
    IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    {
        int32_t input[32] = {123, 401, 402, 403, 404, 405, 406, 407};
        IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(
            cudaMemcpy(d, input, sizeof input, cudaMemcpyHostToDevice) == cudaSuccess);
        const float thresholds[3] = {0.8f, 1.0f, 0.2f};
        const int widths[3] = {2, 4, 8};
        for (float threshold : thresholds) {
            IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(
                !ifn_adaptive_prepare(state, d, d + 1, confidence, threshold, d + 8, d + 16, d + 17,
                                      host, d + 25, host + 8, host + 9, stream));
            for (int branch = 0; branch < 3; branch++) {
                IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(!ifn_adaptive_branch_begin(state, branch, stream));
                ifn_adaptive_graph_answer<<<1, 8, 0, stream>>>(d + 17, d + 25, widths[branch]);
                IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(cudaGetLastError() == cudaSuccess);
                IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(!ifn_adaptive_branch_end(state));
            }
            IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(!ifn_adaptive_instantiate(state));
            for (int branch = 0; branch < 3; branch++) {
                float conf[7] = {};
                int n = branch == 0 ? 0 : branch == 1 ? 2 : 7;
                for (int i = 0; i < n; i++)
                    conf[i] = 1;
                IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(cudaMemcpy(confidence, conf, sizeof conf,
                                                            cudaMemcpyHostToDevice) == cudaSuccess);
                IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(!ifn_adaptive_launch(state, stream));
                IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
                IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(host[8] == widths[branch] &&
                                                 host[9] == widths[branch]);
                for (int i = 0; i < 8; i++)
                    IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(host[i] == widths[branch] * 100 + i);
                int32_t assembled[8];
                IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(cudaMemcpy(assembled, d + 8, sizeof assembled,
                                                            cudaMemcpyDeviceToHost) == cudaSuccess);
                IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(assembled[0] == 123);
                for (int i = 1; i < widths[branch]; i++)
                    IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(assembled[i] == 400 + i);
            }
            IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(ifn_adaptive_live(state));
            IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(!ifn_adaptive_cleanup(state));
            IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(!ifn_adaptive_live(state));
        }
        /* Cleanup before instantiation, and while a body is still capturing. */
        for (int partial = 0; partial < 2; partial++) {
            IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(
                !ifn_adaptive_prepare(state, d, d + 1, confidence, 0.8f, d + 8, d + 16, d + 17,
                                      host, d + 25, host + 8, host + 9, stream));
            IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(
                ifn_adaptive_prepare(state, d, d + 1, confidence, 0.8f, d + 8, d + 16, d + 17, host,
                                     d + 25, host + 8, host + 9, stream) != 0);
            IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(ifn_adaptive_branch_begin(state, 3, stream) != 0);
            if (partial)
                IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(!ifn_adaptive_branch_begin(state, 0, stream));
            IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(!ifn_adaptive_cleanup(state));
            IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(!ifn_adaptive_live(state));
            IFN_ADAPTIVE_GRAPH_LOCAL_REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        }
    }
done:
    if (ifn_adaptive_cleanup(state))
        failed = 1;
    if (stream && cudaStreamDestroy(stream) != cudaSuccess)
        failed = 1;
    if (host && cudaFreeHost(host) != cudaSuccess)
        failed = 1;
    if (confidence && cudaFree(confidence) != cudaSuccess)
        failed = 1;
    if (d && cudaFree(d) != cudaSuccess)
        failed = 1;
    std::free(state);
    if (!failed)
        std::puts("{\"conditional_graph_cases\":9,\"partial_cleanup_cases\":2,\"cleanup\":true}");
    return failed;
}
