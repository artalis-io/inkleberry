/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fputs("usage: inspect CHECKPOINT\n", stderr);
        return 2;
    }
    ifn_safetensors *file = NULL;
    ifn_status status = ifn_safetensors_open(argv[1], NULL, &file);
    if (status != IFN_OK) {
        fprintf(stderr, "%s\n", ifn_status_string(status));
        return 1;
    }
    printf("%zu tensors, %llu payload bytes\n", ifn_safetensors_count(file),
           (unsigned long long)ifn_safetensors_data_bytes(file));
    ifn_safetensors_close(file);
    return 0;
}
