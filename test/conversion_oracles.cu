/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Independent format equations, not parity between two shared converters. */
#include "kernels_common.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

constexpr unsigned IFN_CONVERSION_CASES = 65536;
constexpr unsigned IFN_CONVERSION_WIDTH = 13;

__global__ void ifn_conversion_oracle_kernel(float *output, unsigned seed) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= IFN_CONVERSION_CASES)
        return;
    unsigned code = i ^ seed;
    float *out = output + i * IFN_CONVERSION_WIDTH;
    float2 fp4 = __half22float2(ifn_cuda::ifn_cuda_cvt_e2m1x2_to_half2(code & 255));
    float2 fp8 = __half22float2(ifn_cuda::ifn_cuda_cvt_e4m3x2_to_half2((uint16_t)code));
    out[0] = fp4.x;
    out[1] = fp4.y;
    out[2] = fp8.x;
    out[3] = fp8.y;
    out[4] = ifn_cuda::ifn_cuda_cvt_e4m3_to_float((uint8_t)code);
    __half2 halves[4];
    ifn_cuda::ifn_cuda_cvt_e2m1x8_to_half2x4(code * 2654435761u, halves[0], halves[1], halves[2],
                                             halves[3]);
    for (unsigned j = 0; j < 4; j++) {
        float2 pair = __half22float2(halves[j]);
        out[5 + 2 * j] = pair.x;
        out[6 + 2 * j] = pair.y;
    }
}

static float ifn_conversion_expected(unsigned code, unsigned mantissa_bits, unsigned exponent_bits,
                                     int bias) {
    unsigned mantissa_mask = (1u << mantissa_bits) - 1;
    unsigned exponent_mask = (1u << exponent_bits) - 1;
    unsigned mantissa = code & mantissa_mask;
    unsigned exponent = (code >> mantissa_bits) & exponent_mask;
    if (mantissa_bits == 3 && exponent == 15 && mantissa == 7)
        return NAN;
    float magnitude = exponent ? std::ldexp(1.0f + float(mantissa) / float(1u << mantissa_bits),
                                            int(exponent) - bias)
                               : std::ldexp(float(mantissa), 1 - bias - int(mantissa_bits));
    return code & (1u << (mantissa_bits + exponent_bits)) ? -magnitude : magnitude;
}

static void ifn_conversion_equal(float got, float expected) {
    if (std::isnan(expected) ? !std::isnan(got)
                             : (got != expected || std::signbit(got) != std::signbit(expected)))
        throw "independent conversion oracle mismatch";
}

int main() {
    float *device = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    int failed = 0;
#define IFN_CONVERSION_CHECK(call)                                                                 \
    do {                                                                                           \
        if ((call) != cudaSuccess)                                                                 \
            throw #call;                                                                           \
    } while (0)
    try {
        std::vector<float> host(IFN_CONVERSION_CASES * IFN_CONVERSION_WIDTH);
        IFN_CONVERSION_CHECK(cudaMalloc(&device, host.size() * sizeof(float)));
        IFN_CONVERSION_CHECK(cudaStreamCreate(&stream));
        for (unsigned run = 0; run < 3; run++) {
            unsigned seed = run ? 0x5a5au : 0u;
            IFN_CONVERSION_CHECK(
                cudaMemsetAsync(device, 0xa5, host.size() * sizeof(float), stream));
            IFN_CONVERSION_CHECK(cudaStreamSynchronize(stream));
            if (run == 1)
                IFN_CONVERSION_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            if (run < 2)
                ifn_cuda::ifn_cuda_launch_pdl(ifn_conversion_oracle_kernel,
                                              dim3(IFN_CONVERSION_CASES / 128), 128, stream, device,
                                              seed);
            if (run == 1) {
                IFN_CONVERSION_CHECK(cudaStreamEndCapture(stream, &graph));
                IFN_CONVERSION_CHECK(cudaGraphInstantiate(&executable, graph, 0));
            }
            if (run)
                IFN_CONVERSION_CHECK(cudaGraphLaunch(executable, stream));
            IFN_CONVERSION_CHECK(cudaStreamSynchronize(stream));
            IFN_CONVERSION_CHECK(cudaMemcpy(host.data(), device, host.size() * sizeof(float),
                                            cudaMemcpyDeviceToHost));
            for (unsigned i = 0; i < IFN_CONVERSION_CASES; i++) {
                unsigned code = i ^ seed;
                const float *out = host.data() + i * IFN_CONVERSION_WIDTH;
                ifn_conversion_equal(out[0], ifn_conversion_expected(code & 15, 1, 2, 1));
                ifn_conversion_equal(out[1], ifn_conversion_expected((code >> 4) & 15, 1, 2, 1));
                ifn_conversion_equal(out[2], ifn_conversion_expected(code & 255, 3, 4, 7));
                ifn_conversion_equal(out[3], ifn_conversion_expected(code >> 8, 3, 4, 7));
                ifn_conversion_equal(out[4], ifn_conversion_expected(code & 255, 3, 4, 7));
                unsigned word = code * 2654435761u;
                for (unsigned j = 0; j < 8; j++)
                    ifn_conversion_equal(out[5 + j],
                                         ifn_conversion_expected((word >> (4 * j)) & 15, 1, 2, 1));
            }
        }
        puts("Independent FP4/FP8 equations: 65536 packed pairs, vector lanes, signed zero/NaN, "
             "poisoned eager + graph + replay passed");
    } catch (const char *error) {
        std::fprintf(stderr, "%s\n", error);
        failed = 1;
    }
    if (stream && cudaStreamSynchronize(stream) != cudaSuccess)
        failed = 1;
    if (executable && cudaGraphExecDestroy(executable) != cudaSuccess)
        failed = 1;
    if (graph && cudaGraphDestroy(graph) != cudaSuccess)
        failed = 1;
    if (stream && cudaStreamDestroy(stream) != cudaSuccess)
        failed = 1;
    if (device && cudaFree(device) != cudaSuccess)
        failed = 1;
    return failed;
}
