/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include <sq.h>
#include <string.h>

int main(void) {
    sq_options options = SQ_OPTIONS_INIT;
    sq_safetensors *file = NULL;
    if (options.api_version != SQ_API_VERSION || !strlen(sq_version()))
        return 1;
    return sq_safetensors_open_memory(NULL, 0, &options, &file) == SQ_INVALID_ARGUMENT && !file ? 0
                                                                                                : 1;
}
