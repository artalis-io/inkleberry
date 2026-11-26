/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn_chat.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned char ifn_stream_driver_local_result[100000];
static size_t ifn_stream_driver_local_used;

static int ifn_stream_driver_local_collect(void *p, uint32_t channel, const char *data, size_t n) {
    size_t at = 0;
    uint32_t cp;
    while (at < n)
        if (!ifn_utf8_next((const unsigned char *)data, n, &at, &cp))
            abort();

    (void)p;
    (void)channel;
    if (n > sizeof ifn_stream_driver_local_result - ifn_stream_driver_local_used)
        return 0;
    memcpy(ifn_stream_driver_local_result + ifn_stream_driver_local_used, data, n);
    ifn_stream_driver_local_used += n;
    return 1;
}

static size_t ifn_stream_driver_local_unhex(const char *s, unsigned char *out) {
    size_t n = strlen(s) / 2;
    for (size_t i = 0; i < n; i++) {
        unsigned v;
        if (sscanf(s + i * 2, "%2x", &v) != 1)
            return SIZE_MAX;
        out[i] = (unsigned char)v;
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 11)
        return 2;
    unsigned char input[32768], stopdata[8][64];
    ifn_text_span stops[8];
    if (strlen(argv[1]) > sizeof input * 2)
        return 2;
    size_t n = ifn_stream_driver_local_unhex(argv[1], input),
           chunk = (size_t)strtoul(argv[2], NULL, 10);
    if (n == SIZE_MAX || !chunk)
        return 2;
    for (int i = 3; i < argc; i++) {
        if (strlen(argv[i]) > 128)
            return 2;
        size_t len = ifn_stream_driver_local_unhex(argv[i], stopdata[i - 3]);
        if (len == SIZE_MAX)
            return 2;
        stops[i - 3] = (ifn_text_span){(const char *)stopdata[i - 3], len};
    }
    ifn_stream_options o = IFN_STREAM_OPTIONS_INIT;
    o.parse_chat = 0;
    o.stops = stops;
    o.stop_count = (size_t)(argc - 3);
    ifn_stream *s = NULL;
    if (ifn_stream_create(NULL, &o, NULL, ifn_stream_driver_local_collect, NULL, &s) != IFN_OK)
        return 1;
    for (size_t at = 0; at < n; at += chunk) {
        size_t take = n - at;
        if (take > chunk)
            take = chunk;
        if (ifn_stream_push_bytes(s, input + at, take) != IFN_OK)
            return 1;
    }
    if (ifn_stream_finish(s, IFN_STREAM_LENGTH) != IFN_OK)
        return 1;
    ifn_stream_info info = IFN_STREAM_INFO_INIT;
    if (ifn_stream_get_info(s, &info) != IFN_OK)
        return 1;
    for (size_t i = 0; i < ifn_stream_driver_local_used; i++)
        printf("%02x", ifn_stream_driver_local_result[i]);
    printf("\n%u %llu\n", info.finish_reason, (unsigned long long)info.utf8_replacements);
    return ifn_stream_destroy(&s) != IFN_OK;
}
