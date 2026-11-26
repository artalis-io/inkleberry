/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn_chat.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define IFN_CHAT_LOCAL_CHECK(x)                                                                    \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "chat:%d %s %s\n", __LINE__, #x, ifn_last_error());                    \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

typedef struct {
    char text[100000], reason[10000];
    size_t n, r;
    ifn_stream *stream;
    int stop;
} ifn_chat_local_collector;

static int ifn_chat_local_collect(void *p, uint32_t channel, const char *data, size_t n) {
    size_t at = 0;
    uint32_t cp;
    while (at < n)
        if (!ifn_utf8_next((const unsigned char *)data, n, &at, &cp))
            abort();

    ifn_chat_local_collector *c = p;
    if (c->stream) {
        ifn_stream_info info = IFN_STREAM_INFO_INIT;
        ifn_stream *copy = c->stream;
        if (ifn_stream_get_info(copy, &info) != IFN_BUSY || ifn_stream_destroy(&copy) != IFN_BUSY ||
            ifn_stream_reset(copy, NULL, NULL) != IFN_BUSY || copy != c->stream)
            return 0;
    }
    if (channel == IFN_STREAM_CONTENT) {
        if (n > sizeof c->text - c->n)
            return 0;
        memcpy(c->text + c->n, data, n);
        c->n += n;
    } else {
        if (n > sizeof c->reason - c->r)
            return 0;
        memcpy(c->reason + c->r, data, n);
        c->r += n;
    }
    return !c->stop;
}

static void *ifn_chat_local_oom(void *p, size_t n) {
    (void)p;
    (void)n;
    return NULL;
}

static void ifn_chat_local_release(void *p, void *q) {
    (void)p;
    (void)q;
}

static int ifn_chat_local_bytes_tests(void) {
    const char *inputs[] = {"abc tail", "abX tail", "hello STOP world", "Árvíz🙂x tail",
                            "ab",       "a",        "nomatch"};
    const char *stops[][2] = {{"b", "abc"}, {"b", "abc"}, {"STOP", "ST"}, {"🙂x", "x"},
                              {"b", "abc"}, {"abc", "b"}, {"STOP", "X"}};
    const char *wanted[] = {"", "a", "hello ", "Árvíz", "a", "a", "nomatch"};
    for (unsigned i = 0; i < 7; i++)
        for (size_t chunk = 1; chunk <= strlen(inputs[i]) + 1; chunk++) {
            ifn_text_span stop[2] = {{stops[i][0], strlen(stops[i][0])},
                                     {stops[i][1], strlen(stops[i][1])}};
            ifn_stream_options o = IFN_STREAM_OPTIONS_INIT;
            o.parse_chat = 0;
            o.stops = stop;
            o.stop_count = 2;
            ifn_stream *s = NULL;
            ifn_chat_local_collector c = {0};
            IFN_CHAT_LOCAL_CHECK(
                ifn_stream_create(NULL, &o, NULL, ifn_chat_local_collect, &c, &s) == IFN_OK);
            c.stream = s;
            for (size_t at = 0; at < strlen(inputs[i]); at += chunk) {
                size_t n = strlen(inputs[i]) - at;
                if (n > chunk)
                    n = chunk;
                IFN_CHAT_LOCAL_CHECK(ifn_stream_push_bytes(s, inputs[i] + at, n) == IFN_OK);
            }
            IFN_CHAT_LOCAL_CHECK(ifn_stream_finish(s, IFN_STREAM_LENGTH) == IFN_OK);
            IFN_CHAT_LOCAL_CHECK(c.n == strlen(wanted[i]) && !memcmp(c.text, wanted[i], c.n));
            IFN_CHAT_LOCAL_CHECK(ifn_stream_finish(s, IFN_STREAM_LENGTH) == IFN_OK);
            IFN_CHAT_LOCAL_CHECK(ifn_stream_destroy(&s) == IFN_OK && !s);
        }
    ifn_stream_options o = IFN_STREAM_OPTIONS_INIT;
    o.parse_chat = 0;
    ifn_stream *s = NULL;
    ifn_chat_local_collector c = {0};
    ifn_stream_info info = IFN_STREAM_INFO_INIT;
    IFN_CHAT_LOCAL_CHECK(ifn_stream_create(NULL, &o, NULL, ifn_chat_local_collect, &c, &s) ==
                         IFN_OK);
    const char bad[] = {(char)0xf0, (char)0x9f, 'x', (char)0xff, (char)0xe2};
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_bytes(s, bad, sizeof bad) == IFN_OK &&
                         ifn_stream_finish(s, IFN_STREAM_LENGTH) == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_get_info(s, &info) == IFN_OK && info.utf8_replacements == 3);
    IFN_CHAT_LOCAL_CHECK(c.n == 10 && !memcmp(c.text, "\xef\xbf\xbdx\xef\xbf\xbd\xef\xbf\xbd", 10));
    IFN_CHAT_LOCAL_CHECK(ifn_stream_reset(s, ifn_chat_local_collect, &c) == IFN_OK);
    c.n = 0;
    c.stop = 1;
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_bytes(s, "hello", 5) == IFN_OK &&
                         ifn_stream_get_info(s, &info) == IFN_OK &&
                         info.finish_reason == IFN_STREAM_CALLBACK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_destroy(&s) == IFN_OK);
    o.replace_invalid_utf8 = 0;
    c.stop = 0;
    IFN_CHAT_LOCAL_CHECK(ifn_stream_create(NULL, &o, NULL, ifn_chat_local_collect, &c, &s) ==
                         IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_bytes(s, bad, 1) == IFN_OK &&
                         ifn_stream_finish(s, IFN_STREAM_LENGTH) == IFN_FORMAT);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_bytes(s, "ok", 2) == IFN_FORMAT &&
                         ifn_stream_reset(s, ifn_chat_local_collect, &c) == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_bytes(s, "ok", 2) == IFN_OK &&
                         ifn_stream_destroy(&s) == IFN_OK);
    ifn_options a = IFN_OPTIONS_INIT;
    a.allocate = ifn_chat_local_oom;
    a.release = ifn_chat_local_release;
    IFN_CHAT_LOCAL_CHECK(ifn_stream_create(NULL, &o, &a, NULL, NULL, &s) == IFN_OUT_OF_MEMORY &&
                         !s);
    ifn_text_span invalid = {"", 0};
    o.stops = &invalid;
    o.stop_count = 1;
    IFN_CHAT_LOCAL_CHECK(
        ifn_stream_create(NULL, &o, NULL, NULL, NULL, &s) == IFN_INVALID_ARGUMENT && !s);
    invalid = (ifn_text_span){"\xff", 1};
    IFN_CHAT_LOCAL_CHECK(ifn_stream_create(NULL, &o, NULL, NULL, NULL, &s) == IFN_FORMAT && !s);
    char large_stop[65];
    memset(large_stop, 'a', 65);
    invalid = (ifn_text_span){large_stop, 65};
    IFN_CHAT_LOCAL_CHECK(ifn_stream_create(NULL, &o, NULL, NULL, NULL, &s) == IFN_LIMIT && !s);
    o.stop_count = 9;
    IFN_CHAT_LOCAL_CHECK(ifn_stream_create(NULL, &o, NULL, NULL, NULL, &s) == IFN_LIMIT);
    return 0;
}

int main(int argc, char **argv) {
    IFN_CHAT_LOCAL_CHECK(argc == 2);
    IFN_CHAT_LOCAL_CHECK(!ifn_chat_local_bytes_tests());
    ifn_tokenizer *t = NULL;
    ifn_tokenizer_scratch *scratch = NULL;
    IFN_CHAT_LOCAL_CHECK(ifn_tokenizer_open(argv[1], NULL, &t) == IFN_OK &&
                         ifn_tokenizer_scratch_create(NULL, &scratch) == IFN_OK);
    ifn_chat_message m[] = {{IFN_CHAT_SYSTEM, 0, "Be kind", 7, NULL, 0, NULL, 0},
                            {IFN_CHAT_USER, 0, "Hi", 2, NULL, 0, NULL, 0},
                            {IFN_CHAT_ASSISTANT, 0, "Hello", 5, "secret", 6, NULL, 0},
                            {IFN_CHAT_USER, 0, "<|message_system|>", 18, NULL, 0, NULL, 0}};
    ifn_chat_options o = IFN_CHAT_OPTIONS_INIT;
    size_t needed = 0;
    uint32_t prefix = 0;
    int32_t ids[1024];
    IFN_CHAT_LOCAL_CHECK(ifn_chat_encode(t, m, 4, &o, scratch, NULL, 0, &needed, &prefix) ==
                             IFN_BUFFER_TOO_SMALL &&
                         needed > 0 && prefix > 0);
    for (unsigned i = 0; i < 1024; i++)
        ids[i] = -77;
    IFN_CHAT_LOCAL_CHECK(ifn_chat_encode(t, m, 4, &o, scratch, ids, needed - 1, &needed, &prefix) ==
                             IFN_BUFFER_TOO_SMALL &&
                         ids[0] == -77);
    IFN_CHAT_LOCAL_CHECK(ifn_chat_encode(t, m, 4, &o, scratch, ids, 1024, &needed, &prefix) ==
                         IFN_OK);
    size_t systems = 0, thinking = 0;
    for (size_t i = 0; i < needed; i++) {
        systems += ids[i] == 310;
        thinking += ids[i] == 314;
    }
    IFN_CHAT_LOCAL_CHECK(systems == 2 && thinking == 0); /* user marker stays ordinary text */
    o.keep_history_thinking = 1;
    IFN_CHAT_LOCAL_CHECK(ifn_chat_encode(t, m, 4, &o, scratch, ids, 1024, &needed, &prefix) ==
                         IFN_OK);
    thinking = 0;
    for (size_t i = 0; i < needed; i++)
        thinking += ids[i] == 314;
    IFN_CHAT_LOCAL_CHECK(thinking == 1);
    IFN_CHAT_LOCAL_CHECK(ifn_chat_encode(t, m, 1025, &o, scratch, ids, 1024, &needed, &prefix) ==
                         IFN_LIMIT);
    m[0].content = "\xff";
    m[0].content_bytes = 1;
    IFN_CHAT_LOCAL_CHECK(ifn_chat_encode(t, m, 4, &o, scratch, ids, 1024, &needed, &prefix) ==
                             IFN_FORMAT &&
                         !needed && !prefix);
    m[0].content = "Be kind";
    m[0].content_bytes = 7;
    ifn_chat_options bad = o;
    bad.effort = IFN_THINK_MAX + 1;
    IFN_CHAT_LOCAL_CHECK(ifn_chat_encode(t, m, 4, &bad, scratch, ids, 1024, &needed, &prefix) ==
                         IFN_INVALID_ARGUMENT);
    bad = o;
    bad.api_version = 0;
    IFN_CHAT_LOCAL_CHECK(ifn_chat_encode(t, m, 4, &bad, scratch, ids, 1024, &needed, &prefix) ==
                         IFN_INVALID_ARGUMENT);
    m[0].role = 99;
    IFN_CHAT_LOCAL_CHECK(ifn_chat_encode(t, m, 4, &o, scratch, ids, 1024, &needed, &prefix) ==
                             IFN_INVALID_ARGUMENT &&
                         !needed && !prefix);
    m[0].role = 0;
    m[0].content_bytes = SIZE_MAX;
    IFN_CHAT_LOCAL_CHECK(ifn_chat_encode(t, m, 4, &o, scratch, ids, 1024, &needed, &prefix) ==
                         IFN_LIMIT);
    m[0].content_bytes = 7;
    ifn_stream *s = NULL;
    ifn_chat_local_collector c = {0};
    ifn_stream_options so = IFN_STREAM_OPTIONS_INIT;
    IFN_CHAT_LOCAL_CHECK(ifn_stream_create(t, &so, NULL, ifn_chat_local_collect, &c, &s) == IFN_OK);
    c.stream = s;
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 312) == IFN_OK); /* model header */
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 'x') == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 314) == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 'r') == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 315) == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 313) == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 't') == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 316) == IFN_OK);
    ifn_stream_info info = IFN_STREAM_INFO_INIT;
    IFN_CHAT_LOCAL_CHECK(ifn_stream_get_info(s, &info) == IFN_OK &&
                         info.finish_reason == IFN_STREAM_EOS && info.tokens == 8);
    IFN_CHAT_LOCAL_CHECK(c.n == 1 && c.text[0] == 't' && c.r == 1 && c.reason[0] == 'r');
    IFN_CHAT_LOCAL_CHECK(ifn_stream_reset(s, ifn_chat_local_collect, &c) == IFN_OK &&
                         ifn_stream_push_token(s, 317) == IFN_UNSUPPORTED);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_reset(s, ifn_chat_local_collect, &c) == IFN_OK &&
                         ifn_stream_push_bytes(s, "a", 1) == IFN_OK &&
                         ifn_stream_push_token(s, 'b') == IFN_INVALID_ARGUMENT);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_reset(s, ifn_chat_local_collect, &c) == IFN_OK);
    c.n = c.r = 0;
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 260) == IFN_OK && c.n == 300);
    for (size_t i = 0; i < c.n; i++)
        IFN_CHAT_LOCAL_CHECK(c.text[i] == 'x');
    IFN_CHAT_LOCAL_CHECK(ifn_stream_reset(s, ifn_chat_local_collect, &c) == IFN_OK);
    c.n = c.r = 0;
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 314) == IFN_OK &&
                         ifn_stream_push_token(s, 0xf0) == IFN_OK &&
                         ifn_stream_push_token(s, 313) == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(c.r == 3 && !memcmp(c.reason, "\xef\xbf\xbd", 3));
    IFN_CHAT_LOCAL_CHECK(ifn_stream_destroy(&s) == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_create(t, &so, NULL, ifn_chat_local_collect, &c, &s) == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 312) == IFN_OK);
    for (unsigned i = 0; i < 3; i++)
        IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 260) == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, 260) == IFN_LIMIT);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_destroy(&s) == IFN_OK);
    char stop_bytes[] = "STOP";
    ifn_text_span held = {stop_bytes, 4};
    so.stops = &held;
    so.stop_count = 1;
    c.n = c.r = 0;
    c.stream = NULL;
    IFN_CHAT_LOCAL_CHECK(ifn_stream_create(t, &so, NULL, ifn_chat_local_collect, &c, &s) == IFN_OK);
    c.stream = s;
    memset(stop_bytes, 'x', 4);
    int32_t cross[] = {'S', 'T', 314, 'r', 313, 'O', 'P'};
    for (unsigned i = 0; i < 7; i++)
        IFN_CHAT_LOCAL_CHECK(ifn_stream_push_token(s, cross[i]) == IFN_OK);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_get_info(s, &info) == IFN_OK &&
                         info.finish_reason == IFN_STREAM_STOP && !c.n && c.r == 1);
    IFN_CHAT_LOCAL_CHECK(ifn_stream_destroy(&s) == IFN_OK);
    ifn_tokenizer_scratch_destroy(scratch);
    ifn_tokenizer_close(t);
    puts("text conversations, spoof rejection, sizing, Unicode/stops, channels, OOM, BUSY and "
         "reset passed");
    return 0;
}
