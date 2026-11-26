/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sq.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fputs("usage: inspect CHECKPOINT\n", stderr);
        return 2;
    }
    sq_safetensors *file = NULL;
    sq_status status = sq_safetensors_open(argv[1], NULL, &file);
    if (status != SQ_OK) {
        fprintf(stderr, "%s\n", sq_status_string(status));
        return 1;
    }
    printf("%zu tensors, %llu payload bytes\n", sq_safetensors_count(file),
           (unsigned long long)sq_safetensors_data_bytes(file));
    sq_safetensors_close(file);
    return 0;
}
