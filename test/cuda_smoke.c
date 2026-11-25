/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Real C host -> unmodified CUDA kernel, with exact data and graph replay. */
#include "cuda_abi.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    enum { H = 2304, B = 2, BLOCKS = 3 };

    float input[B * BLOCKS * H], output[B * H];
    void *device_input = NULL, *device_output = NULL, *stream = NULL, *graph = NULL;
    int failed = 1;
    for (size_t i = 0; i < sizeof input / sizeof input[0]; i++)
        input[i] = (float)i - 111.0f;
#define CHECK(call)                                                                                \
    do {                                                                                           \
        int rc = (call);                                                                           \
        if (rc) {                                                                                  \
            fprintf(stderr, "%s: CUDA %d\n", #call, rc);                                           \
            goto cleanup;                                                                          \
        }                                                                                          \
    } while (0)
    CHECK(squid_cuda_malloc(&device_input, sizeof input));
    CHECK(squid_cuda_malloc(&device_output, sizeof output));
    CHECK(squid_cuda_stream_create(&stream));
    CHECK(squid_cuda_memcpy_h2d_async(device_input, input, sizeof input, stream));
    CHECK(squid_cuda_stream_sync(stream));
    /* Eager execution and two captured replays, changing the input each time. */
    for (int run = 0; run < 3; run++) {
        if (run == 0)
            squid_launch_extract_block(device_input, BLOCKS, 1, device_output, B, stream);
        if (run == 1) {
            CHECK(squid_cuda_graph_begin(stream));
            squid_launch_extract_block(device_input, BLOCKS, 1, device_output, B, stream);
            CHECK(squid_cuda_graph_end_instantiate(stream, &graph));
        }
        if (run)
            CHECK(squid_cuda_graph_launch(graph, stream));
        CHECK(squid_cuda_stream_sync(stream));
        CHECK(squid_cuda_memcpy_d2h(output, device_output, sizeof output));
        for (size_t b = 0; b < B; b++) {
            if (memcmp(output + b * H, input + (b * BLOCKS + 1) * H, H * sizeof(float))) {
                fputs("kernel output mismatch\n", stderr);
                goto cleanup;
            }
        }
        for (size_t i = 0; i < sizeof input / sizeof input[0]; i++)
            input[i] += 1.0f;
        CHECK(squid_cuda_memcpy_h2d_async(device_input, input, sizeof input, stream));
        CHECK(squid_cuda_stream_sync(stream));
    }
    failed = 0;
cleanup:
    if (stream && squid_cuda_stream_sync(stream))
        failed = 1;
    if (graph && squid_cuda_graph_destroy(graph))
        failed = 1;
    if (stream && squid_cuda_stream_destroy(stream))
        failed = 1;
    if (device_output && squid_cuda_free(device_output))
        failed = 1;
    if (device_input && squid_cuda_free(device_input))
        failed = 1;
    if (!failed)
        puts("C CUDA kernel: eager + graph replay exact data checks passed");
    return failed;
}
