/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sq.h"
#include <stdint.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    sq_safetensors *file = NULL;
    if (sq_safetensors_open_memory(data, size, NULL, &file) == SQ_OK) {
        for (size_t i = 0; i < sq_safetensors_count(file); i++) {
            const sq_tensor_info *info = NULL;
            const void *payload = NULL;
            size_t length = 0;
            (void)sq_safetensors_at(file, i, &info);
            (void)sq_safetensors_data(file, i, &payload, &length);
        }
    }
    sq_safetensors_close(file);
    return 0;
}
