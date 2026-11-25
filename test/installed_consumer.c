/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include <sq_chat.h>
#include <string.h>

int main(void) {
    sq_stream_options stream_options = SQ_STREAM_OPTIONS_INIT;
    stream_options.parse_chat = 0;
    sq_stream *stream = NULL;
    if (sq_stream_create(NULL, &stream_options, NULL, NULL, NULL, &stream) != SQ_OK ||
        sq_stream_push_bytes(stream, "x", 1) != SQ_OK ||
        sq_stream_finish(stream, SQ_STREAM_LENGTH) != SQ_OK || sq_stream_destroy(&stream) != SQ_OK)
        return 1;
    sq_options options = SQ_OPTIONS_INIT;
    sq_safetensors *file = NULL;
    if (options.api_version != SQ_API_VERSION || !strlen(sq_version()))
        return 1;
    return sq_safetensors_open_memory(NULL, 0, &options, &file) == SQ_INVALID_ARGUMENT && !file ? 0
                                                                                                : 1;
}
