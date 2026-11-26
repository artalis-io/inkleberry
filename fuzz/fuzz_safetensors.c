/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn.h"
#include <stdint.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    ifn_safetensors *file = NULL;
    if (ifn_safetensors_open_memory(data, size, NULL, &file) == IFN_OK) {
        for (size_t i = 0; i < ifn_safetensors_count(file); i++) {
            const ifn_tensor_info *info = NULL;
            const void *payload = NULL;
            size_t length = 0;
            (void)ifn_safetensors_at(file, i, &info);
            (void)ifn_safetensors_data(file, i, &payload, &length);
        }
    }
    ifn_safetensors_close(file);
    return 0;
}
