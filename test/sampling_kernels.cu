/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Statistical diagnostics of unchanged upstream sampling kernels, not new math.
 * Two-token asymmetric distributions exercise direct sampling and rejection. */
#include "../cuda/kernels_gemv.cu"
#include <cstdio>
#include <vector>
using namespace ifn_cuda;

struct ifn_sampling_kernels_device {
    void *p = nullptr;

    explicit ifn_sampling_kernels_device(size_t n) {
        if (cudaMalloc(&p, n) != cudaSuccess)
            throw "allocation failed";
    }

    ~ifn_sampling_kernels_device() {
        cudaFree(p);
    }

    void put(const void *src, size_t n) {
        if (cudaMemcpy(p, src, n, cudaMemcpyHostToDevice) != cudaSuccess)
            throw "upload failed";
    }
};

int main() {
    try {
        constexpr int B = 8192, V = 2;
        std::vector<float> logits(B * V), prob(B * V), qsel(B),
            values((size_t)B * IFN_CUDA_LM_HEAD_BLOCKS, -1e30f);
        std::vector<int32_t> indices(values.size(), 0), drafts(B), output(B);
        for (int b = 0; b < B; b++) {
            logits[b * V] = logf(0.7f);
            logits[b * V + 1] = logf(0.3f);
            values[(size_t)b * IFN_CUDA_LM_HEAD_BLOCKS] = logits[b * V];
            prob[b * V] = 0.2f;
            prob[b * V + 1] = 0.8f;
            drafts[b] = b % 5 == 0 ? 0 : 1;
            qsel[b] = drafts[b] == 0 ? 0.2f : 0.8f;
        }
        ifn_sampling_kernels_device l(logits.size() * 4), q(prob.size() * 4), qs(qsel.size() * 4),
            v(values.size() * 4), i(indices.size() * 4), d(drafts.size() * 4),
            out(output.size() * 4), cfg(sizeof(ifn_cuda_sample_config)), delta(4), pos(4);
        l.put(logits.data(), logits.size() * 4);
        q.put(prob.data(), prob.size() * 4);
        qs.put(qsel.data(), qsel.size() * 4);
        v.put(values.data(), values.size() * 4);
        i.put(indices.data(), indices.size() * 4);
        d.put(drafts.data(), drafts.size() * 4);
        ifn_cuda_sample_config config = {};
        config.inv_temp = 1;
        config.top_p = 1;
        config.seed = 0x123456789abcdef0ULL;
        cfg.put(&config, sizeof config);
        int zero = 0;
        delta.put(&zero, 4);
        pos.put(&zero, 4);
        for (int mode = 0; mode < 4; mode++) {
            int is_delta = mode == 2;
            delta.put(&is_delta, 4);
            config.top_k = mode == 3 ? 1 : 0;
            cfg.put(&config, sizeof config);
            ifn_cuda_lm_head_stage2_argmax_kernel<<<B, 256>>>(
                static_cast<float *>(v.p), static_cast<int32_t *>(i.p), static_cast<float *>(l.p),
                static_cast<int32_t *>(out.p), static_cast<ifn_cuda_sample_config *>(cfg.p),
                static_cast<int32_t *>(pos.p),
                mode == 1 || mode == 2 ? static_cast<int32_t *>(d.p) : nullptr,
                static_cast<float *>(q.p), static_cast<float *>(qs.p),
                static_cast<int32_t *>(delta.p), nullptr, V, V, B);
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
                cudaMemcpy(output.data(), out.p, output.size() * 4, cudaMemcpyDeviceToHost) !=
                    cudaSuccess)
                throw "sampling launch failed";
            size_t count = 0, accepted = 0;
            for (int b = 0; b < B - 1; b++) {
                if (output[b] < 0 || output[b] >= V)
                    throw "invalid sample";
                count += output[b] == 0;
                accepted += output[b] == drafts[b];
            }
            double observed = (double)count / (B - 1);
            if (mode == 0 || mode == 1) {
                if (fabs(observed - 0.7) > 0.035)
                    throw "target distribution diagnostic failed";
            } else if (mode == 2) {
                // Dirac proposals alternate at fixed frequencies: the remaining
                // residual mass must still yield target P regardless of proposal.
                if (fabs(observed - 0.7) > 0.035)
                    throw "Dirac residual diagnostic failed";
            } else if (count != B - 1)
                throw "top-k=1 must be greedy";
            if (mode == 1 && fabs((double)accepted / (B - 1) - 0.5) > 0.035)
                throw "acceptance diagnostic failed";
            std::printf("{\"sampling_diagnostic\":%d,\"draws\":%d,\"target_zero\":%.6f,\"observed_"
                        "zero\":%.6f,\"matching_proposals\":%zu}\n",
                        mode, B - 1, mode == 3 ? 1.0 : 0.7, observed, accepted);
        }
        return 0;
    } catch (const char *e) {
        std::fprintf(stderr, "%s\n", e);
        return 1;
    }
}
