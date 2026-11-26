/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include <ifn_chat.h>
#include <string.h>

int main(void) {
    ifn_stream_options stream_options = IFN_STREAM_OPTIONS_INIT;
    stream_options.parse_chat = 0;
    ifn_stream *stream = NULL;
    if (ifn_stream_create(NULL, &stream_options, NULL, NULL, NULL, &stream) != IFN_OK ||
        ifn_stream_push_bytes(stream, "x", 1) != IFN_OK ||
        ifn_stream_finish(stream, IFN_STREAM_LENGTH) != IFN_OK ||
        ifn_stream_destroy(&stream) != IFN_OK)
        return 1;
    ifn_options options = IFN_OPTIONS_INIT;
    ifn_safetensors *file = NULL;
    if (options.api_version != IFN_API_VERSION || strcmp(ifn_version(), IFN_VERSION_STRING))
        return 1;
    return ifn_safetensors_open_memory(NULL, 0, &options, &file) == IFN_INVALID_ARGUMENT && !file
               ? 0
               : 1;
}
