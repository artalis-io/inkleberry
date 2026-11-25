/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sq.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--version")) {
        printf("squid-c %s (checkpoint inspection; generation not ported)\n", sq_version());
        return 0;
    }
    if (argc != 3 || strcmp(argv[1], "inspect")) {
        fputs("usage: squid-c inspect FILE.safetensors | --version\nC generation is not "
              "implemented yet.\n",
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
