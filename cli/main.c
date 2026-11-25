/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sq.h"
#include "sq_weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--version")) {
        printf("squid-c %s (host tools; optional CUDA AR CLI)\n", sq_version());
        return 0;
    }
    if ((argc == 3 || argc == 4) && !strcmp(argv[1], "weight-plan")) {
        sq_weight_config config = SQ_WEIGHT_CONFIG_INIT;
        if (!strcmp(argv[2], "exact"))
            config.mode = SQ_WEIGHT_EXACT;
        else if (!strcmp(argv[2], "fp8"))
            config.mode = SQ_WEIGHT_FP8;
        else if (strcmp(argv[2], "nvfp4"))
            return 2;
        if (argc == 4) {
            if (!strcmp(argv[3], "ar"))
                config.with_mtp = 0;
            else if (!strcmp(argv[3], "mtp-bf16"))
                config.mtp_bf16 = 1;
            else if (strcmp(argv[3], "mtp"))
                return 2;
        }
        sq_weight_plan plan;
        sq_status status = sq_weights_plan(&config, NULL, &plan);
        if (status != SQ_OK) {
            fprintf(stderr, "squid-c: %s\n", sq_last_error());
            return 1;
        }
        printf("{\"arena_bytes\":%llu,\"staging_bytes\":%llu,\"amax_bytes\":%llu,\"fold_temporary_"
               "bytes\":%llu,\"peak_device_bytes\":%llu,\"allocations\":%zu,\"views\":%zu}\n",
               (unsigned long long)plan.arena_bytes, (unsigned long long)plan.staging_bytes,
               (unsigned long long)plan.amax_bytes, (unsigned long long)plan.fold_temporary_bytes,
               (unsigned long long)plan.peak_device_bytes, plan.allocations, plan.views);
        return 0;
    }
    if ((argc == 3 || argc == 4) && !strcmp(argv[1], "validate-model")) {
        sq_inkling_checkpoint *checkpoint = NULL;
        sq_status status =
            sq_inkling_checkpoint_open(argv[2], argc == 4 ? argv[3] : NULL, NULL, &checkpoint);
        if (status != SQ_OK) {
            fprintf(stderr, "squid-c: %s: %s\n", sq_status_string(status), sq_last_error());
            return 1;
        }
        printf("{\"architecture\":\"inkling-2-mini\",\"bound_tensors\":%zu,\"mtp\":%s,\"device_"
               "upload\":false}\n",
               sq_inkling_checkpoint_count(checkpoint), argc == 4 ? "true" : "false");
        sq_inkling_checkpoint_close(checkpoint);
        return 0;
    }
    if (argc == 4 && !strcmp(argv[1], "tokenize")) {
        sq_tokenizer *t = NULL;
        sq_tokenizer_scratch *scratch = NULL;
        int32_t *ids = NULL;
        const char *failure = NULL;
        size_t n = 0;
        sq_status status = sq_tokenizer_open(argv[2], NULL, &t);
        if (status == SQ_OK)
            status = sq_tokenizer_scratch_create(NULL, &scratch);
        if (status == SQ_OK) {
            status = sq_tokenizer_encode(t, argv[3], strlen(argv[3]), 1, scratch, NULL, 0, &n);
            if (status == SQ_BUFFER_TOO_SMALL)
                status = SQ_OK;
        }
        if (status == SQ_OK && n) {
            if (n > SIZE_MAX / sizeof *ids || !(ids = malloc(n * sizeof *ids))) {
                status = SQ_OUT_OF_MEMORY;
                failure = "cannot allocate token output";
            }
        }
        if (status == SQ_OK)
            status = sq_tokenizer_encode(t, argv[3], strlen(argv[3]), 1, scratch, ids, n, &n);
        if (status == SQ_OK) {
            fputs("[", stdout);
            for (size_t i = 0; i < n; i++)
                printf("%s%d", i ? "," : "", (int)ids[i]);
            puts("]");
        } else
            fprintf(stderr, "squid-c: %s: %s\n", sq_status_string(status),
                    failure ? failure : sq_last_error());
        free(ids);
        sq_tokenizer_scratch_destroy(scratch);
        sq_tokenizer_close(t);
        return status == SQ_OK ? 0 : 1;
    }
    if (argc != 3 || strcmp(argv[1], "inspect")) {
        fputs("usage: squid-c inspect FILE.safetensors | tokenize TOKENIZER.json TEXT | "
              "validate-model MAIN.safetensors [MTP.safetensors] | "
              "weight-plan exact|fp8|nvfp4 [ar|mtp|mtp-bf16] | --version\nGeneration requires the "
              "optional "
              "squid-c-generate CUDA target.\n",
              stderr);
        return 2;
    }
    sq_safetensors *file = NULL;
    sq_status status = sq_safetensors_open(argv[2], NULL, &file);
    if (status != SQ_OK) {
        fprintf(stderr, "squid-c: %s: %s\n", sq_status_string(status), sq_last_error());
        return 1;
    }
    size_t counts[4] = {0};
    for (size_t i = 0; i < sq_safetensors_count(file); i++) {
        const sq_tensor_info *info = NULL;
        if (sq_safetensors_at(file, i, &info) != SQ_OK || info->dtype >= 4) {
            sq_safetensors_close(file);
            return 1;
        }
        counts[info->dtype]++;
    }
    printf("{\"format\":\"safetensors\",\"tensors\":%zu,\"data_bytes\":%llu,"
           "\"dtypes\":{\"U8\":%zu,\"BF16\":%zu,\"F32\":%zu,\"F8_E4M3\":%zu}}\n",
           sq_safetensors_count(file), (unsigned long long)sq_safetensors_data_bytes(file),
           counts[0], counts[1], counts[2], counts[3]);
    sq_safetensors_close(file);
    return 0;
}
