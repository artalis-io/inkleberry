/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Test-only C entrypoints consumed by a Zig oracle, plus standalone C main.
 * Full payload comparison excludes uninitialized alignment padding. */
#define _POSIX_C_SOURCE 200809L
#include "ifn_gpu.h"
#include "cuda_abi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern int ifn_test_context(ifn_gpu_weights **ifn_gpu_weights_compare_local_image, int destroy);
static ifn_gpu_weights *ifn_gpu_weights_compare_local_image;
static size_t ifn_gpu_weights_compare_local_checked_views,
    ifn_gpu_weights_compare_local_checked_spans;
static uint64_t ifn_gpu_weights_compare_local_checked_bytes;
static double ifn_gpu_weights_compare_local_elapsed;
static const void *ifn_gpu_weights_compare_local_oracle_allocation;
static unsigned char ifn_gpu_weights_compare_local_oracle_guard[64];

static double ifn_gpu_weights_compare_local_now(void) {
    struct timespec t;
    (void)clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int ifn_test_start(const char *directory, uint32_t mode, uint32_t mtp, uint64_t oracle_capacity,
                   uint64_t oracle_allocations, const void *oracle_base) {
    ifn_gpu_weights_compare_local_oracle_allocation = oracle_base;
    if (oracle_base && ifn_cuda_memcpy_d2h(ifn_gpu_weights_compare_local_oracle_guard, oracle_base,
                                           sizeof ifn_gpu_weights_compare_local_oracle_guard))
        return 1;
    char main_path[1200], mtp_path[1200];
    if (snprintf(main_path, sizeof main_path, "%s/model-00001-of-00001.safetensors", directory) <
            0 ||
        strlen(directory) > 1100)
        return 1;
    (void)snprintf(mtp_path, sizeof mtp_path, "%s/mtp.safetensors", directory);
    ifn_inkling_checkpoint *checkpoint = NULL;
    ifn_weight_config config = IFN_WEIGHT_CONFIG_INIT;
    config.mode = mode;
    config.with_mtp = mtp;
    const char *force = getenv("INKLEBERRY_MTP_BF16");
    config.mtp_bf16 = force && !strcmp(force, "1");
    double start = ifn_gpu_weights_compare_local_now();
    ifn_status status =
        ifn_inkling_checkpoint_open(main_path, mtp ? mtp_path : NULL, NULL, &checkpoint);
    if (status == IFN_OK)
        status = ifn_gpu_weights_create(&config, NULL, &ifn_gpu_weights_compare_local_image);
    if (status == IFN_OK)
        status = ifn_gpu_weights_load(ifn_gpu_weights_compare_local_image, checkpoint);
    ifn_gpu_weights_compare_local_elapsed = ifn_gpu_weights_compare_local_now() - start;
    ifn_inkling_checkpoint_close(checkpoint);
    if (status != IFN_OK) {
        fprintf(stderr, "C weight load: %s\n", ifn_last_error());
        return 1;
    }
    if (ifn_test_context(&ifn_gpu_weights_compare_local_image, 0))
        return 1;
    ifn_weight_plan plan;
    if (ifn_gpu_weights_get_plan(ifn_gpu_weights_compare_local_image, &plan) != IFN_OK)
        return 1;
    if (oracle_base &&
        (plan.arena_bytes != oracle_capacity || plan.allocations != oracle_allocations)) {
        fprintf(stderr, "arena mismatch C=%llu/%zu Zig=%llu/%llu\n",
                (unsigned long long)plan.arena_bytes, plan.allocations,
                (unsigned long long)oracle_capacity, (unsigned long long)oracle_allocations);
        return 1;
    }

    enum { CHUNK = 4 * 1024 * 1024 };

    unsigned char *a = malloc(CHUNK), *b = malloc(CHUNK);
    if (!a || !b) {
        free(a);
        free(b);
        return 1;
    }
    for (size_t i = 0; i < plan.allocations; i++) {
        ifn_weight_span span;
        if (ifn_gpu_weights_get_span(ifn_gpu_weights_compare_local_image, i, &span) != IFN_OK) {
            free(a);
            free(b);
            return 1;
        }
        for (uint64_t offset = 0; offset < span.bytes;) {
            size_t bytes = span.bytes - offset > CHUNK ? CHUNK : (size_t)(span.bytes - offset);
            if (ifn_gpu_weights_read(ifn_gpu_weights_compare_local_image, i, offset, a, bytes) !=
                IFN_OK) {
                fprintf(stderr, "C read: %s\n", ifn_last_error());
                free(a);
                free(b);
                return 1;
            }
            if (oracle_base) {
                if (ifn_cuda_memcpy_d2h(
                        b, (const unsigned char *)oracle_base + span.offset + offset, bytes) ||
                    memcmp(a, b, bytes)) {
                    fprintf(stderr, "payload mismatch: %s at chunk offset %llu\n", span.label,
                            (unsigned long long)offset);
                    free(a);
                    free(b);
                    return 1;
                }
            }
            ifn_gpu_weights_compare_local_checked_bytes += bytes;
            offset += bytes;
        }
        ifn_gpu_weights_compare_local_checked_spans++;
    }
    free(a);
    free(b);
    if (!oracle_base)
        printf("{\"standalone_c_load\":true,\"mode\":%u,\"mtp\":%u,\"arena_bytes\":%llu,\"payload_"
               "bytes_read\":%llu,\"allocations\":%zu,\"load_seconds\":%.6f}\n",
               mode, mtp, (unsigned long long)plan.arena_bytes,
               (unsigned long long)ifn_gpu_weights_compare_local_checked_bytes, plan.allocations,
               ifn_gpu_weights_compare_local_elapsed);
    return 0;
}

int ifn_test_view(const char *label, uint32_t type, uint32_t precision, uint32_t rows,
                  uint32_t cols, uint32_t experts, uint64_t data, uint64_t scales,
                  uint64_t row_scales, uint64_t expert_scale, float global_scale) {
    ifn_weight_plan plan;
    if (ifn_gpu_weights_get_plan(ifn_gpu_weights_compare_local_image, &plan) != IFN_OK)
        return 1;
    for (size_t i = 0; i < plan.views; i++) {
        ifn_weight_view v;
        if (ifn_gpu_weights_get_view(ifn_gpu_weights_compare_local_image, i, &v) != IFN_OK)
            return 1;
        if (strcmp(v.label, label))
            continue;
        if (v.type != type || v.precision != precision || v.rows != rows || v.cols != cols ||
            v.experts != experts || v.data != data || v.scales != scales ||
            v.row_scales != row_scales || v.expert_scale != expert_scale ||
            memcmp(&v.global_scale, &global_scale, 4)) {
            fprintf(stderr, "descriptor mismatch: %s\n", label);
            return 1;
        }
        ifn_gpu_weights_compare_local_checked_views++;
        return 0;
    }
    fprintf(stderr, "C view missing: %s\n", label);
    return 1;
}

int ifn_test_finish(uint32_t mode, uint32_t mtp, uint32_t oracle) {
    int failed = 0;
    ifn_weight_plan plan;
    if (ifn_gpu_weights_compare_local_image &&
        ifn_gpu_weights_get_plan(ifn_gpu_weights_compare_local_image, &plan) == IFN_OK && oracle &&
        ifn_gpu_weights_compare_local_checked_views != plan.views) {
        fprintf(stderr, "view coverage: %zu/%zu\n", ifn_gpu_weights_compare_local_checked_views,
                plan.views);
        failed = 1;
    }
    if (ifn_gpu_weights_compare_local_image && oracle && !failed &&
        ifn_test_context(&ifn_gpu_weights_compare_local_image, 1))
        failed = 1;
    ifn_status status = ifn_gpu_weights_destroy(&ifn_gpu_weights_compare_local_image);
    if (status != IFN_OK || ifn_gpu_weights_compare_local_image) {
        fprintf(stderr, "C GPU cleanup: %s\n", ifn_last_error());
        failed = 1;
    }
    if (oracle && ifn_gpu_weights_compare_local_oracle_allocation) {
        unsigned char after[64];
        if (ifn_cuda_memcpy_d2h(after, ifn_gpu_weights_compare_local_oracle_allocation,
                                sizeof after) ||
            memcmp(after, ifn_gpu_weights_compare_local_oracle_guard, sizeof after)) {
            fputs("original Zig allocation did not survive C teardown\n", stderr);
            failed = 1;
        }
    }
    if (!failed && oracle)
        printf("{\"oracle_parity\":true,\"mode\":%u,\"mtp\":%u,\"payload_spans\":%zu,\"payload_"
               "bytes\":%llu,\"descriptors\":%zu,\"c_load_seconds\":%.6f,\"caller_context_"
               "preserved\":true,\"oracle_allocation_survived\":true}\n",
               mode, mtp, ifn_gpu_weights_compare_local_checked_spans,
               (unsigned long long)ifn_gpu_weights_compare_local_checked_bytes,
               ifn_gpu_weights_compare_local_checked_views, ifn_gpu_weights_compare_local_elapsed);
    return failed;
}
#ifdef IFN_STANDALONE_WEIGHTS
int main(int argc, char **argv) {
    if (argc != 4) {
        fputs("usage: inkleberry-weights MODEL_DIRECTORY exact|fp8|nvfp4 ar|mtp\n", stderr);
        return 2;
    }
    uint32_t mode = !strcmp(argv[2], "exact")   ? 0
                    : !strcmp(argv[2], "fp8")   ? 1
                    : !strcmp(argv[2], "nvfp4") ? 2
                                                : 99;
    uint32_t mtp = !strcmp(argv[3], "ar") ? 0 : !strcmp(argv[3], "mtp") ? 1 : 99;
    if (mode > 2 || mtp > 1)
        return 2;
    int result = ifn_test_start(argv[1], mode, mtp, 0, 0, NULL);
    if (ifn_test_finish(mode, mtp, 0))
        result = 1;
    return result;
}
#endif
