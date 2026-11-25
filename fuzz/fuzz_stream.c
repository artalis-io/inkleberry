/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sq_chat.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char bytes[100000];
    size_t n;
} output;

static int collect(void *p, uint32_t channel, const char *data, size_t n) {
    (void)channel;
    output *o = p;
    if (n > sizeof o->bytes - o->n)
        abort();
    memcpy(o->bytes + o->n, data, n);
    o->n += n;
    return 1;
}

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t n) {
    if (n > 32768)
        return 0;
    sq_text_span stops[] = {{"b", 1}, {"abc", 3}, {"STOP", 4}, {"🙂", 4}};
    output a = {0}, b = {0};
    sq_stream_options o = SQ_STREAM_OPTIONS_INIT;
    o.parse_chat = 0;
    o.stops = stops;
    o.stop_count = 4;
    sq_stream *s = NULL;
    sq_stream_info first = SQ_STREAM_INFO_INIT, second = SQ_STREAM_INFO_INIT;
    if (sq_stream_create(NULL, &o, NULL, collect, &a, &s) != SQ_OK)
        abort();
    if (sq_stream_push_bytes(s, data, n) != SQ_OK ||
        sq_stream_finish(s, SQ_STREAM_LENGTH) != SQ_OK || sq_stream_get_info(s, &first) != SQ_OK)
        abort();
    if (sq_stream_reset(s, collect, &b) != SQ_OK)
        abort();
    for (size_t i = 0; i < n; i++)
        if (sq_stream_push_bytes(s, data + i, 1) != SQ_OK)
            abort();
    if (sq_stream_finish(s, SQ_STREAM_LENGTH) != SQ_OK || sq_stream_get_info(s, &second) != SQ_OK ||
        a.n != b.n || memcmp(a.bytes, b.bytes, a.n) ||
        first.finish_reason != second.finish_reason ||
        first.utf8_replacements != second.utf8_replacements)
        abort();
    if (sq_stream_destroy(&s) != SQ_OK)
        abort();
    return 0;
}
