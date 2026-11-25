/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sq_chat.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned char result[100000];
static size_t used;

static int collect(void *p, uint32_t channel, const char *data, size_t n) {
    size_t at = 0;
    uint32_t cp;
    while (at < n)
        if (!sq_utf8_next((const unsigned char *)data, n, &at, &cp))
            abort();

    (void)p;
    (void)channel;
    if (n > sizeof result - used)
        return 0;
    memcpy(result + used, data, n);
    used += n;
    return 1;
}

static size_t unhex(const char *s, unsigned char *out) {
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
    sq_text_span stops[8];
    if (strlen(argv[1]) > sizeof input * 2)
        return 2;
    size_t n = unhex(argv[1], input), chunk = (size_t)strtoul(argv[2], NULL, 10);
    if (n == SIZE_MAX || !chunk)
        return 2;
    for (int i = 3; i < argc; i++) {
        if (strlen(argv[i]) > 128)
            return 2;
        size_t len = unhex(argv[i], stopdata[i - 3]);
        if (len == SIZE_MAX)
            return 2;
        stops[i - 3] = (sq_text_span){(const char *)stopdata[i - 3], len};
    }
    sq_stream_options o = SQ_STREAM_OPTIONS_INIT;
    o.parse_chat = 0;
    o.stops = stops;
    o.stop_count = (size_t)(argc - 3);
    sq_stream *s = NULL;
    if (sq_stream_create(NULL, &o, NULL, collect, NULL, &s) != SQ_OK)
        return 1;
    for (size_t at = 0; at < n; at += chunk) {
        size_t take = n - at;
        if (take > chunk)
            take = chunk;
        if (sq_stream_push_bytes(s, input + at, take) != SQ_OK)
            return 1;
    }
    if (sq_stream_finish(s, SQ_STREAM_LENGTH) != SQ_OK)
        return 1;
    sq_stream_info info = SQ_STREAM_INFO_INIT;
    if (sq_stream_get_info(s, &info) != SQ_OK)
        return 1;
    for (size_t i = 0; i < used; i++)
        printf("%02x", result[i]);
    printf("\n%u %llu\n", info.finish_reason, (unsigned long long)info.utf8_replacements);
    return sq_stream_destroy(&s) != SQ_OK;
}
