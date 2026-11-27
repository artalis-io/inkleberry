/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Test-only eager teacher forcing and packed payload identity. No kernel edits. */
#include "engine_internal.h"
#include "sha256.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *ifn_numeric_directory;
static unsigned ifn_numeric_batch, ifn_numeric_trace_from, ifn_numeric_generation;
static FILE *ifn_numeric_index;

static int ifn_numeric_device_hash(const void *device, size_t bytes, char hex[65]) {
    unsigned char *host = malloc(bytes), sum[32];
    if (!host)
        return 1;
    int failed = ifn_cuda_memcpy_d2h(host, device, bytes) || !ifn_sha256_hash(host, bytes, sum);
    if (!failed)
        ifn_sha256_hex(sum, hex);
    free(host);
    return failed;
}

static int ifn_numeric_first_gemm(ifn_session *s, int32_t batch) {
    const ifn_cuda_c_linear_weight *weight = &s->model.layers[0].w_qrgs;
    char weights[65], input[65], path[2048];
    if (ifn_numeric_device_hash(s->p_w_f16, (size_t)weight->rows * (size_t)weight->cols * 2,
                                weights) ||
        ifn_numeric_device_hash(s->p_attn_in_h[0], (size_t)batch * (size_t)weight->cols * 2, input))
        return 1;
    uint32_t alpha;
    memcpy(&alpha, &weight->global_scale, sizeof alpha);
    snprintf(path, sizeof path, "%s/first-gemm.json", ifn_numeric_directory);
    FILE *file = fopen(path, "wb");
    if (!file)
        return 1;
    int result =
        fprintf(file,
                "{\"rows\":%d,\"cols\":%d,\"batch\":%d,\"kind\":%u,\"alpha_bits\":%u,\"fp16_"
                "weights_sha256\":\"%s\",\"fp16_input_sha256\":\"%s\"}\n",
                weight->rows, weight->cols, batch, (unsigned)weight->kind, alpha, weights, input);
    return fclose(file) || result < 0;
}

void ifn_test_numeric_trace(ifn_session *s, int32_t batch, unsigned visit, const char *stage,
                            const float *data, size_t width) {
    if (ifn_numeric_generation && !strcmp(stage, "logits"))
        return;
    if (s->status != IFN_OK || (unsigned)(s->position + batch) <= ifn_numeric_trace_from)
        return;
    size_t bytes = width * (size_t)batch * sizeof(float);
    float *host = malloc(bytes);
    if (!host || ifn_cuda_stream_sync(s->stream) || ifn_cuda_memcpy_d2h(host, data, bytes)) {
        free(host);
        s->status = IFN_GPU_ERROR;
        return;
    }
    if (ifn_numeric_generation && batch > 8 && visit == 0 && !strcmp(stage, "qrgs") &&
        ifn_numeric_first_gemm(s, batch)) {
        free(host);
        s->status = IFN_IO;
        return;
    }
    for (int32_t b = 0; b < batch; b++) {
        if ((unsigned)(s->position + b) < ifn_numeric_trace_from)
            continue;
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
        goto failed;
    for (size_t i = 0; i < plan.allocations; i++) {
        ifn_weight_span span;
        if (ifn_gpu_weights_get_span(weights, i, &span))
            goto failed;
        ifn_sha256 hash;
        ifn_sha256_init(&hash);
        for (uint64_t offset = 0; offset < span.bytes;) {
            size_t bytes = span.bytes - offset > 4 * 1024 * 1024 ? 4 * 1024 * 1024
                                                                 : (size_t)(span.bytes - offset);
            if (ifn_gpu_weights_read(weights, i, offset, buffer, bytes) ||
                !ifn_sha256_update(&hash, buffer, bytes))
                goto failed;
            offset += bytes;
        }
        unsigned char sum[32];
        char hex[65];
        ifn_sha256_final(&hash, sum);
        ifn_sha256_hex(sum, hex);
        if (fprintf(file, "{\"label\":\"%s\",\"offset\":%llu,\"bytes\":%llu,\"sha256\":\"%s\"}\n",
                    span.label, (unsigned long long)span.offset, (unsigned long long)span.bytes,
                    hex) < 0)
            goto failed;
    }
    for (size_t i = 0; i < plan.views; i++) {
        ifn_weight_view view;
        uint32_t scale_bits;
        if (ifn_gpu_weights_get_view(weights, i, &view))
            goto failed;
        memcpy(&scale_bits, &view.global_scale, sizeof scale_bits);
        if (fprintf(file,
                    "{\"view\":\"%s\",\"type\":%u,\"precision\":%u,\"rows\":%u,\"cols\":%u,"
                    "\"experts\":%u,\"data\":%llu,\"scales\":%llu,\"row_scales\":%llu,\"expert_"
                    "scale\":%llu,\"global_scale_bits\":%u}\n",
                    view.label, view.type, view.precision, view.rows, view.cols, view.experts,
                    (unsigned long long)view.data, (unsigned long long)view.scales,
                    (unsigned long long)view.row_scales, (unsigned long long)view.expert_scale,
                    scale_bits) < 0)
            goto failed;
    }
    free(buffer);
    return fclose(file) != 0;
failed:
    free(buffer);
    fclose(file);
    return 1;
}

static int ifn_numeric_read_token(FILE *input, unsigned *value) {
    char word[32], *end;
    if (fscanf(input, "%31s", word) != 1)
        return feof(input) && !ferror(input) ? 0 : -1;
    errno = 0;
    unsigned long number = strtoul(word, &end, 10);
    if (strlen(word) == 31 || errno || *end || word[0] == '-' || number >= 200058)
        return -1;
    *value = (unsigned)number;
    return 1;
}

int main(int argc, char **argv) {
    if ((argc != 4 && argc != 5 && argc != 6) || strlen(argv[1]) > 1500 || strlen(argv[2]) > 1500 ||
        strlen(argv[3]) != 1 || argv[3][0] < '0' || argv[3][0] > '2')
        return 2;
    int32_t tokens[128] = {40, 12, 309, 17, 8, 232, 111, 25};
    unsigned token_count = 8, common = 0;
    if (argc >= 5) {
        FILE *input = fopen(argv[4], "rb");
        if (!input)
            return 2;
        unsigned value;
        int read;
        token_count = 0;
        if (ifn_numeric_read_token(input, &common) != 1 || common >= 128) {
            fclose(input);
            return 2;
        }
        while ((read = ifn_numeric_read_token(input, &value)) == 1) {
            if (token_count == 128) {
                fclose(input);
                return 2;
            }
            tokens[token_count++] = (int32_t)value;
        }
        fclose(input);
        if (read < 0 || common >= token_count)
            return 2;
    }
    ifn_numeric_generation = argc == 6;
    if (ifn_numeric_generation && (strcmp(argv[5], "generation") || common || token_count <= 32))
        return 2;
    ifn_numeric_trace_from = common;
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
    sc.context_tokens = ifn_numeric_generation ? 32768 : 128;
    sc.kv_format = 1;
    sc.use_graphs = 0;
    if (!status)
        status = ifn_session_create(weights, &sc, NULL, &session);
    snprintf(path, sizeof path, "%s/trace.jsonl", argv[2]);
    ifn_numeric_index = fopen(path, "wb");
    if (!ifn_numeric_index)
        status = IFN_IO;
    float *logits = malloc(200058 * sizeof(float));
    if (!logits)
        status = IFN_OUT_OF_MEMORY;
    int32_t generated = -1;
    if (ifn_numeric_generation && !status) {
        ifn_decode_options options = IFN_DECODE_OPTIONS_INIT;
        options.seed = 42;
        ifn_decode_stats stats;
        ifn_numeric_batch = 0;
        status = ifn_session_generate_ex(session, tokens, token_count, 1, &options, &generated, 1,
                                         NULL, NULL, &stats);
    }
    for (unsigned batch = 1; !ifn_numeric_generation && !status && batch <= 8; batch *= 2) {
        ifn_numeric_batch = batch;
        for (unsigned offset = 0; !status && offset < common; offset++)
            status = ifn_session_logits(session, tokens + offset, 1, offset == 0, logits, 200058);
        for (unsigned offset = common; !status && offset < token_count; offset += batch) {
            unsigned count = token_count - offset < batch ? token_count - offset : batch;
            status =
                ifn_session_logits(session, tokens + offset, count, offset == 0, logits, 200058);
        }
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
    if (!status) {
        snprintf(path, sizeof path, "%s/run.json", argv[2]);
        FILE *run = fopen(path, "wb");
        if (!run)
            return 1;
        int written = fprintf(run,
                              "{\"generation_probe\":%u,\"generated_token\":%d,\"token_count\":%u,"
                              "\"trace_from\":%u,\"mode\":%u,\"kv_format\":1,\"tokens\":[",
                              ifn_numeric_generation, generated, token_count, common, mode);
        for (unsigned i = 0; i < token_count; i++)
            if (fprintf(run, "%s%d", i ? "," : "", tokens[i]) < 0)
                written = -1;
        if (fputs("]}\n", run) < 0)
            written = -1;
        if (fclose(run) || written < 0)
            return 1;
    }
    return status != IFN_OK;
}
