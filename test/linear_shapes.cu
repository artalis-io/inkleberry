/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Diagnostic of unchanged production dispatch, not a new numerical policy.
 * Synthetic exactly representable FP4/FP16 operands; no model data required. */
#include "kernels.hpp"
#include "../src/checked_prefill.h"
#include <cuda.h>
#include <cmath>
#include <vector>

static bool ifn_shapes_check(cudaError_t error) {
    if (error == cudaSuccess)
        return true;
    std::fprintf(stderr, "linear shapes: %s\n", cudaGetErrorString(error));
    return false;
}

#define IFN_SHAPES_TRY(call)                                                                       \
    do {                                                                                           \
        if (!ifn_shapes_check(call))                                                               \
            return 1;                                                                              \
    } while (0)

static int ifn_shapes_case(int cols) {
    CUdevice device;
    CUcontext context = nullptr;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, nullptr, 0, device) != CUDA_SUCCESS)
        return 1;
    const int rows = 32;
    std::vector<unsigned char> packed(rows * cols / 2), scales(rows * cols / 16, 0x38);
    std::vector<__half> input(8 * cols);
    std::vector<double> reference(rows, 0);
    const float fp4[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    for (int k = 0; k < cols; k++) {
        float x = float((k * 73 + 19) % 2048 - 1024) / 1024;
        for (int b = 0; b < 8; b++)
            input[b * cols + k] = __float2half_rn(x);
        for (int r = 0; r < rows; r++) {
            unsigned code = unsigned((r * 37 + k * 11 + k / 7) % 16);
            packed[(r * cols + k) / 2] |= (unsigned char)(code << ((k % 2) * 4));
            reference[r] += double(x) * fp4[code & 7] * (code & 8 ? -1 : 1);
        }
    }
    void *weights = nullptr;
    unsigned char *scale_device = nullptr;
    __half *input_device = nullptr;
    float *output_device = nullptr;
    cudaStream_t stream = nullptr;
    IFN_SHAPES_TRY(cudaMalloc(&weights, packed.size()));
    IFN_SHAPES_TRY(cudaMalloc(&scale_device, scales.size()));
    IFN_SHAPES_TRY(cudaMalloc(&input_device, input.size() * sizeof(__half)));
    IFN_SHAPES_TRY(cudaMalloc(&output_device, 8 * rows * sizeof(float)));
    IFN_SHAPES_TRY(cudaStreamCreate(&stream));
    void *blas = nullptr;
    int prepare_status = ifn_blas_create(&blas, stream);
    if (prepare_status) {
        std::fprintf(stderr, "checked prefill preparation failed: %d\n", prepare_status);
        return 1;
    }
    if (ifn_blas_destroy(blas))
        return 1;
    IFN_SHAPES_TRY(cudaMemcpy(weights, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    IFN_SHAPES_TRY(cudaMemcpy(scale_device, scales.data(), scales.size(), cudaMemcpyHostToDevice));
    IFN_SHAPES_TRY(cudaMemcpy(input_device, input.data(), input.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
    ifn_cuda::ifn_cuda_linear_weight weight;
    weight.kind = ifn_cuda::ifn_cuda_weight_kind::IFN_CUDA_NVFP4;
    weight.rows = rows;
    weight.cols = cols;
    weight.data = weights;
    weight.scales = scale_device;
    std::vector<float> singleton;
    for (int batch : {1, 2, 4, 8}) {
        std::vector<float> first(batch * rows), output(batch * rows);
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        for (int mode = 0; mode < 2; mode++) {
            if (mode) {
                IFN_SHAPES_TRY(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                ifn_cuda::ifn_cuda_launch_linear_h(weight, input_device, output_device, batch,
                                                   false, stream);
                IFN_SHAPES_TRY(cudaStreamEndCapture(stream, &graph));
                IFN_SHAPES_TRY(cudaGraphInstantiate(&executable, graph, 0));
            }
            for (int repeat = 0; repeat < 5; repeat++) {
                IFN_SHAPES_TRY(
                    cudaMemsetAsync(output_device, 0xa5, output.size() * sizeof(float), stream));
                if (mode) {
                    IFN_SHAPES_TRY(cudaGraphLaunch(executable, stream));
                } else
                    ifn_cuda::ifn_cuda_launch_linear_h(weight, input_device, output_device, batch,
                                                       false, stream);
                if (ifn_cuda::ifn_cuda_take_error())
                    return 1;
                IFN_SHAPES_TRY(cudaMemcpyAsync(output.data(), output_device,
                                               output.size() * sizeof(float),
                                               cudaMemcpyDeviceToHost, stream));
                IFN_SHAPES_TRY(cudaStreamSynchronize(stream));
                for (float v : output)
                    if (!std::isfinite(v))
                        return 1;
                if (!mode && !repeat)
                    first = output;
                else if (output != first) {
                    std::fputs("non-repeatable eager/graph output\n", stderr);
                    return 1;
                }
            }
        }
        if (batch == 1)
            singleton = first;
        double maximum = 0, shape_difference = 0;
        unsigned differing = 0;
        for (int b = 0; b < batch; b++)
            for (int r = 0; r < rows; r++) {
                maximum = std::fmax(maximum, std::fabs(double(first[b * rows + r]) - reference[r]));
                shape_difference = std::fmax(shape_difference,
                                             std::fabs(double(first[b * rows + r]) - singleton[r]));
                differing += first[b * rows + r] != singleton[r];
            }
        std::printf("{\"cols\":%d,\"rows\":%d,\"batch\":%d,\"max_reference_error\":%.17g,\"max_"
                    "singleton_difference\":%.17g,\"different_elements\":%u,\"repetitions_per_"
                    "mode\":5,\"eager_graph_equal\":true}\n",
                    cols, rows, batch, maximum, shape_difference, differing);
        IFN_SHAPES_TRY(cudaGraphExecDestroy(executable));
        IFN_SHAPES_TRY(cudaGraphDestroy(graph));
    }
    IFN_SHAPES_TRY(cudaStreamDestroy(stream));
    IFN_SHAPES_TRY(cudaFree(output_device));
    IFN_SHAPES_TRY(cudaFree(input_device));
    IFN_SHAPES_TRY(cudaFree(scale_device));
    IFN_SHAPES_TRY(cudaFree(weights));
    if (cuCtxDestroy(context) != CUDA_SUCCESS)
        return 1;
    return 0;
}

int main() {
    return ifn_shapes_case(2304) || ifn_shapes_case(3072);
}
