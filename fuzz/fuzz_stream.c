/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn_chat.h"
#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char bytes[100000];
    size_t n;
} ifn_fuzz_stream_local_output;

static int ifn_fuzz_stream_local_collect(void *p, uint32_t channel, const char *data, size_t n) {
    size_t at = 0;
    uint32_t cp;
    while (at < n)
        if (!ifn_utf8_next((const unsigned char *)data, n, &at, &cp))
            abort();

    (void)channel;
    ifn_fuzz_stream_local_output *o = p;
    if (n > sizeof o->bytes - o->n)
        abort();
    memcpy(o->bytes + o->n, data, n);
    o->n += n;
    return 1;
}

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t n) {
    if (n > 32768)
        return 0;
    ifn_text_span stops[] = {{"b", 1}, {"abc", 3}, {"STOP", 4}, {"🙂", 4}};
    ifn_fuzz_stream_local_output a = {0}, b = {0};
    ifn_stream_options o = IFN_STREAM_OPTIONS_INIT;
    o.parse_chat = 0;
    o.stops = stops;
    o.stop_count = 4;
    ifn_stream *s = NULL;
    ifn_stream_info first = IFN_STREAM_INFO_INIT, second = IFN_STREAM_INFO_INIT;
    if (ifn_stream_create(NULL, &o, NULL, ifn_fuzz_stream_local_collect, &a, &s) != IFN_OK)
        abort();
    if (ifn_stream_push_bytes(s, data, n) != IFN_OK ||
        ifn_stream_finish(s, IFN_STREAM_LENGTH) != IFN_OK ||
        ifn_stream_get_info(s, &first) != IFN_OK)
        abort();
    if (ifn_stream_reset(s, ifn_fuzz_stream_local_collect, &b) != IFN_OK)
        abort();
    for (size_t i = 0; i < n; i++)
        if (ifn_stream_push_bytes(s, data + i, 1) != IFN_OK)
            abort();
    if (ifn_stream_finish(s, IFN_STREAM_LENGTH) != IFN_OK ||
        ifn_stream_get_info(s, &second) != IFN_OK || a.n != b.n || memcmp(a.bytes, b.bytes, a.n) ||
        first.finish_reason != second.finish_reason ||
        first.utf8_replacements != second.utf8_replacements)
        abort();
    if (ifn_stream_destroy(&s) != IFN_OK)
        abort();
    return 0;
}
