/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Test-only eager teacher forcing and packed payload identity. No kernel edits. */
#include "engine_internal.h"
#include "sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *ifn_numeric_directory;
static unsigned ifn_numeric_batch;
static FILE *ifn_numeric_index;

void ifn_test_numeric_trace(ifn_session *s, int32_t batch, unsigned visit, const char *stage,
                            const float *data, size_t width) {
    if (s->status != IFN_OK)
        return;
    size_t bytes = width * (size_t)batch * sizeof(float);
    float *host = malloc(bytes);
    if (!host || ifn_cuda_stream_sync(s->stream) || ifn_cuda_memcpy_d2h(host, data, bytes)) {
        free(host);
        s->status = IFN_GPU_ERROR;
        return;
    }
    for (int32_t b = 0; b < batch; b++) {
        char path[2048];
        int n = snprintf(path, sizeof path, "%s/b%u-p%d-v%u-%s.bin", ifn_numeric_directory,
                         ifn_numeric_batch, s->position + b, visit, stage);
        FILE *file = n > 0 && (size_t)n < sizeof path ? fopen(path, "wb") : NULL;
        if (!file) {
            s->status = IFN_IO;
            break;
        }
        size_t written = fwrite(host + (size_t)b * width, sizeof(float), width, file);
        int closed = fclose(file);
        if (written != width || closed) {
            s->status = IFN_IO;
            break;
        }
        if (fprintf(ifn_numeric_index,
                    "{\"batch\":%u,\"position\":%d,\"visit\":%u,\"stage\":\"%s\",\"width\":%zu}\n",
                    ifn_numeric_batch, s->position + b, visit, stage, width) < 0)
            s->status = IFN_IO;
    }
    free(host);
}

static int ifn_numeric_spans(ifn_gpu_weights *weights) {
    char path[2048];
    if (snprintf(path, sizeof path, "%s/spans.jsonl", ifn_numeric_directory) >= (int)sizeof path)
        return 1;
    FILE *file = fopen(path, "wb");
    unsigned char *buffer = malloc(4 * 1024 * 1024);
    if (!file || !buffer) {
        if (file)
            fclose(file);
        free(buffer);
        return 1;
    }
    ifn_weight_plan plan;
    if (ifn_gpu_weights_get_plan(weights, &plan))
        return 1;
    for (size_t i = 0; i < plan.allocations; i++) {
        ifn_weight_span span;
        if (ifn_gpu_weights_get_span(weights, i, &span))
            return 1;
        ifn_sha256 hash;
        ifn_sha256_init(&hash);
        for (uint64_t offset = 0; offset < span.bytes;) {
            size_t bytes = span.bytes - offset > 4 * 1024 * 1024 ? 4 * 1024 * 1024
                                                                 : (size_t)(span.bytes - offset);
            if (ifn_gpu_weights_read(weights, i, offset, buffer, bytes) ||
                !ifn_sha256_update(&hash, buffer, bytes))
                return 1;
            offset += bytes;
        }
        unsigned char sum[32];
        char hex[65];
        ifn_sha256_final(&hash, sum);
        ifn_sha256_hex(sum, hex);
        if (fprintf(file, "{\"label\":\"%s\",\"offset\":%llu,\"bytes\":%llu,\"sha256\":\"%s\"}\n",
                    span.label, (unsigned long long)span.offset, (unsigned long long)span.bytes,
                    hex) < 0)
            return 1;
    }
    for (size_t i = 0; i < plan.views; i++) {
        ifn_weight_view view;
        uint32_t scale_bits;
        if (ifn_gpu_weights_get_view(weights, i, &view))
            return 1;
        memcpy(&scale_bits, &view.global_scale, sizeof scale_bits);
        if (fprintf(file,
                    "{\"view\":\"%s\",\"type\":%u,\"precision\":%u,\"rows\":%u,\"cols\":%u,"
                    "\"experts\":%u,\"data\":%llu,\"scales\":%llu,\"row_scales\":%llu,\"expert_"
                    "scale\":%llu,\"global_scale_bits\":%u}\n",
                    view.label, view.type, view.precision, view.rows, view.cols, view.experts,
                    (unsigned long long)view.data, (unsigned long long)view.scales,
                    (unsigned long long)view.row_scales, (unsigned long long)view.expert_scale,
                    scale_bits) < 0)
            return 1;
    }
    free(buffer);
    return fclose(file) != 0;
}

int main(int argc, char **argv) {
    if (argc != 4 || strlen(argv[1]) > 1500 || strlen(argv[2]) > 1500 || strlen(argv[3]) != 1 ||
        argv[3][0] < '0' || argv[3][0] > '2')
        return 2;
    unsigned mode = (unsigned)(argv[3][0] - '0');
    ifn_numeric_directory = argv[2];
    char path[2048], mtp[2048];
    snprintf(path, sizeof path, "%s/model-00001-of-00001.safetensors", argv[1]);
    snprintf(mtp, sizeof mtp, "%s/mtp.safetensors", argv[1]);
    ifn_inkling_checkpoint *checkpoint = NULL;
    ifn_gpu_weights *weights = NULL;
    ifn_session *session = NULL;
    ifn_weight_config config = IFN_WEIGHT_CONFIG_INIT;
    config.mode = mode;
    ifn_status status = ifn_inkling_checkpoint_open(path, mtp, NULL, &checkpoint);
    if (!status)
        status = ifn_gpu_weights_create(&config, NULL, &weights);
    if (!status)
        status = ifn_gpu_weights_load(weights, checkpoint);
    ifn_inkling_checkpoint_close(checkpoint);
    if (!status && ifn_numeric_spans(weights))
        status = IFN_IO;
    ifn_session_config sc = IFN_SESSION_CONFIG_INIT;
    sc.context_tokens = 128;
    sc.kv_format = 1;
    sc.use_graphs = 0;
    if (!status)
        status = ifn_session_create(weights, &sc, NULL, &session);
    snprintf(path, sizeof path, "%s/trace.jsonl", argv[2]);
    ifn_numeric_index = fopen(path, "wb");
    if (!ifn_numeric_index)
        status = IFN_IO;
    const int32_t tokens[8] = {40, 12, 309, 17, 8, 232, 111, 25};
    float *logits = malloc(200058 * sizeof(float));
    if (!logits)
        status = IFN_OUT_OF_MEMORY;
    for (unsigned batch = 1; !status && batch <= 8; batch *= 2) {
        ifn_numeric_batch = batch;
        for (unsigned offset = 0; !status && offset < 8; offset += batch)
            status =
                ifn_session_logits(session, tokens + offset, batch, offset == 0, logits, 200058);
    }
    free(logits);
    if (ifn_numeric_index && fclose(ifn_numeric_index))
        status = IFN_IO;
    if (status)
        fprintf(stderr, "numerical probe: %s (%s)\n", ifn_status_string(status), ifn_last_error());
    if (ifn_session_destroy(&session))
        status = IFN_GPU_ERROR;
    if (ifn_gpu_weights_destroy(&weights))
        status = IFN_GPU_ERROR;
    return status != IFN_OK;
}
