/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Small real-kernel fixtures for sanitizer and checked/original ABI parity. */
#include "cuda_abi.h"
#include "checked_cuda.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

struct ifn_packing_buffer {
    void *p = nullptr;
    size_t n;

    explicit ifn_packing_buffer(size_t bytes) : n(bytes) {
        if (cudaMalloc(&p, n) != cudaSuccess)
            throw "test allocation failed";
    }

    ~ifn_packing_buffer() {
        cudaFree(p);
    }

    void upload(const void *src) {
        if (cudaMemcpy(p, src, n, cudaMemcpyHostToDevice) != cudaSuccess)
            throw "test upload failed";
    }
};

static void ifn_packing_local_equal(const ifn_packing_buffer &a, const ifn_packing_buffer &b) {
    std::vector<unsigned char> x(a.n), y(b.n);
    if (a.n != b.n || cudaMemcpy(x.data(), a.p, a.n, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(y.data(), b.p, b.n, cudaMemcpyDeviceToHost) != cudaSuccess || x != y)
        throw "packing payload mismatch";
}

static void ifn_packing_local_bf16(ifn_packing_buffer &b) {
    std::vector<__nv_bfloat16> h(b.n / 2);
    for (size_t i = 0; i < h.size(); i++)
        h[i] = __float2bfloat16((static_cast<int>(i % 97) - 48) * 0.0125f);
    b.upload(h.data());
}

int main() {
    cudaStream_t stream = nullptr;
    try {
        if (cudaStreamCreate(&stream) != cudaSuccess)
            throw "stream create failed";
        {
            const int rows = 37, cols = 64;
            ifn_packing_buffer input(rows * cols * 2), amax(4);
            ifn_packing_local_bf16(input);
            for (int kind : {1, 2}) {
                ifn_packing_buffer a(rows * cols / (kind == 2 ? 2 : 1)), b(a.n);
                ifn_packing_buffer sa(kind == 2 ? rows * cols / 16 : rows * 4), sb(sa.n);
                float original = 1, checked = 1;
                if (kind == 1)
                    ifn_pack_bf16_to_fp8(input.p, a.p, static_cast<float *>(sa.p), rows, cols);
                else
                    original = ifn_pack_bf16_to_nvfp4(input.p, a.p, static_cast<uint8_t *>(sa.p),
                                                      static_cast<float *>(amax.p), rows, cols);
                if (cudaDeviceSynchronize() != cudaSuccess ||
                    ifn_cuda_pack(input.p, b.p, sb.p, amax.p, rows, cols, kind, &checked, stream) ||
                    cudaStreamSynchronize(stream) != cudaSuccess ||
                    std::memcmp(&original, &checked, sizeof original))
                    throw "BF16 packing failed";
                ifn_packing_local_equal(a, b);
                ifn_packing_local_equal(sa, sb);
            }
        }
        {
            const int e = 3, r = 37, c = 64;
            ifn_packing_buffer input(e * r * c), scales(e * r * c / 32);
            std::vector<unsigned char> values(input.n), factors(scales.n, 127);
            for (size_t i = 0; i < values.size(); i++)
                values[i] = static_cast<unsigned char>(0x20 + i % 32);
            input.upload(values.data());
            scales.upload(factors.data());
            ifn_packing_buffer a(input.n / 2), b(a.n), sa(input.n / 16), sb(sa.n), ea(e * 4),
                eb(ea.n);
            ifn_convert_mxfp8_to_nvfp4_experts(
                static_cast<uint8_t *>(input.p), static_cast<uint8_t *>(scales.p),
                static_cast<uint8_t *>(a.p), static_cast<uint8_t *>(sa.p),
                static_cast<float *>(ea.p), e, r, c);
            if (cudaDeviceSynchronize() != cudaSuccess ||
                ifn_cuda_expert(input.p, scales.p, b.p, sb.p, eb.p, e, r, c, stream) ||
                cudaStreamSynchronize(stream) != cudaSuccess)
                throw "expert packing failed";
            ifn_packing_local_equal(a, b);
            ifn_packing_local_equal(sa, sb);
            ifn_packing_local_equal(ea, eb);
        }
        {
            ifn_packing_buffer input(768 * 5120 * 2), a(768 * 512 * 2), b(a.n), ha(768 * 2304 * 2),
                hb(ha.n), fa(ha.n), fb(ha.n);
            ifn_packing_local_bf16(input);
            ifn_split_dspark_rnn(input.p, a.p, ha.p, fa.p);
            if (cudaDeviceSynchronize() != cudaSuccess ||
                ifn_cuda_rnn(input.p, b.p, hb.p, fb.p, stream) ||
                cudaStreamSynchronize(stream) != cudaSuccess)
                throw "RNN split failed";
            ifn_packing_local_equal(a, b);
            ifn_packing_local_equal(ha, hb);
            ifn_packing_local_equal(fa, fb);
        }
        {
            ifn_packing_buffer input(8 * 2304 * 2), hidden(2304 * 256 * 2), bias(2304 * 2),
                a(8 * 4), b(a.n), wa(8 * 256 * 2), wb(wa.n);
            ifn_packing_local_bf16(input);
            ifn_packing_local_bf16(hidden);
            ifn_packing_local_bf16(bias);
            ifn_compute_dspark_fold(input.p, hidden.p, bias.p, static_cast<float *>(a.p), wa.p, 8,
                                    2304, 256);
            if (cudaDeviceSynchronize() != cudaSuccess ||
                ifn_cuda_fold(input.p, hidden.p, bias.p, b.p, wb.p, 8, stream) ||
                cudaStreamSynchronize(stream) != cudaSuccess)
                throw "MTP fold failed";
            ifn_packing_local_equal(a, b);
            ifn_packing_local_equal(wa, wb);
        }
        if (cudaStreamDestroy(stream) != cudaSuccess)
            throw "stream destroy failed";
        stream = nullptr;
        std::puts("{\"checked_packing_parity\":true,\"cases\":5}");
        return 0;
    } catch (const char *message) {
        std::fprintf(stderr, "%s\n", message);
        if (stream)
            cudaStreamDestroy(stream);
        return 1;
    }
}
