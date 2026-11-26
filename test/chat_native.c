/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "decode_compare.c"
#include "ifn_chat.h"

typedef struct {
    ifn_stream *stream;
    ifn_status error;
    char content[8192], reasoning[8192];
    size_t n, r;
} ifn_chat_native_local_output;

static int ifn_chat_native_local_text(void *p, uint32_t channel, const char *data, size_t n) {
    ifn_chat_native_local_output *o = p;
    char *out = channel == IFN_STREAM_CONTENT ? o->content : o->reasoning;
    size_t *used = channel == IFN_STREAM_CONTENT ? &o->n : &o->r;
    if (n > 8192 - *used)
        return 0;
    memcpy(out + *used, data, n);
    *used += n;
    return 1;
}

static int ifn_chat_native_local_token(void *p, int32_t id) {
    ifn_chat_native_local_output *o = p;
    o->error = ifn_stream_push_token(o->stream, id);
    ifn_stream_info info = IFN_STREAM_INFO_INIT;
    if (o->error != IFN_OK || ifn_stream_get_info(o->stream, &info) != IFN_OK)
        return 0;
    return info.finish_reason == IFN_STREAM_RUNNING;
}

#define IFN_CHAT_NATIVE_LOCAL_REQUIRE(x)                                                           \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "native chat:%d %s\n", __LINE__, ifn_last_error());                    \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    int failed = 0;
    ifn_tokenizer *t = NULL;
    ifn_tokenizer_scratch *scratch = NULL;
    ifn_stream *stream = NULL;
    int32_t prompt[8192], baseline[128], out[128];
    size_t n = 0;
    uint32_t hint = 0;
    char path[1200], system[8192];
    system[0] = 0;
    for (unsigned i = 0; i < 64; i++)
        strcat(system, "You are a concise mathematical assistant. Return the numeric answer. ");
    ifn_chat_message messages[] = {
        {IFN_CHAT_SYSTEM, 0, system, strlen(system), NULL, 0, NULL, 0},
        {IFN_CHAT_USER, 0, "What is 2 + 2?", 14, NULL, 0, NULL, 0},
        {IFN_CHAT_ASSISTANT, 0, "4.", 2, NULL, 0, NULL, 0},
        {IFN_CHAT_USER, 0, "What is that number plus 3?", 27, NULL, 0, NULL, 0}};
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(!ifn_decode_start(argv[1], 2, 1, 1, 8192, 7, 64));
    snprintf(path, sizeof path, "%s/tokenizer.json", argv[1]);
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_tokenizer_open(path, NULL, &t) == IFN_OK &&
                                  ifn_tokenizer_scratch_create(NULL, &scratch) == IFN_OK);
    ifn_chat_options chat = IFN_CHAT_OPTIONS_INIT;
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(
        ifn_chat_encode(t, messages, 4, &chat, scratch, prompt, 8192, &n, &hint) == IFN_OK &&
        hint >= 128);
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_session_prefix_cache(ifn_ar_compare_local_session, 1) ==
                                  IFN_OK);
    ifn_decode_options decode = IFN_DECODE_OPTIONS_INIT;
    decode.draft_tokens = 7;
    decode.adaptive = 1;
    decode.prompt_lookup = 1;
    decode.system_prefix_tokens = hint;
    ifn_stream_options opts = IFN_STREAM_OPTIONS_INIT;
    ifn_chat_native_local_output first = {0};
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(
        ifn_stream_create(t, &opts, NULL, ifn_chat_native_local_text, &first, &stream) == IFN_OK);
    first.stream = stream;
    ifn_decode_stats stats;
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(
        ifn_session_generate_ex(ifn_ar_compare_local_session, prompt, n, 128, &decode, baseline,
                                128, ifn_chat_native_local_token, &first, &stats) == IFN_OK &&
        first.error == IFN_OK);
    size_t output_count = stats.generation.generated_tokens;
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_stream_finish(stream, IFN_STREAM_LENGTH) == IFN_OK &&
                                  first.n > 0);
    ifn_stream_info info = IFN_STREAM_INFO_INIT;
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_stream_get_info(stream, &info) == IFN_OK &&
                                  !info.utf8_replacements);
    ifn_chat_native_local_output collected = {0};
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(
        ifn_stream_reset(stream, ifn_chat_native_local_text, &collected) == IFN_OK);
    for (size_t i = 0; i < output_count; i++)
        IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_stream_push_token(stream, baseline[i]) == IFN_OK);
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_stream_finish(stream, IFN_STREAM_LENGTH) == IFN_OK &&
                                  collected.n == first.n && collected.r == first.r &&
                                  !memcmp(collected.content, first.content, first.n) &&
                                  !memcmp(collected.reasoning, first.reasoning, first.r));
    ifn_chat_native_local_output second = {0};
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_stream_reset(stream, ifn_chat_native_local_text, &second) ==
                                  IFN_OK);
    second.stream = stream;
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(
        ifn_session_generate_ex(ifn_ar_compare_local_session, prompt, n, 128, &decode, out, 128,
                                ifn_chat_native_local_token, &second, &stats) == IFN_OK &&
        second.error == IFN_OK);
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(stats.generation.generated_tokens == output_count &&
                                  !memcmp(out, baseline, output_count * 4));
    ifn_prefix_info prefix = IFN_PREFIX_INFO_INIT;
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_session_prefix_info(ifn_ar_compare_local_session, &prefix) ==
                                      IFN_OK &&
                                  prefix.reused_tokens >= hint);
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_stream_finish(stream, IFN_STREAM_LENGTH) == IFN_OK &&
                                  second.n == first.n &&
                                  !memcmp(second.content, first.content, first.n));
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_stream_destroy(&stream) == IFN_OK);
    size_t stop_length = 1;
    while (stop_length < first.n && ((unsigned char)first.content[stop_length] & 0xc0) == 0x80)
        stop_length++;
    ifn_text_span stop = {first.content, stop_length};
    opts.stops = &stop;
    opts.stop_count = 1;
    ifn_chat_native_local_output stopped = {0};
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(
        ifn_stream_create(t, &opts, NULL, ifn_chat_native_local_text, &stopped, &stream) == IFN_OK);
    stopped.stream = stream;
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(
        ifn_session_generate_ex(ifn_ar_compare_local_session, prompt, n, 128, &decode, out, 128,
                                ifn_chat_native_local_token, &stopped, &stats) == IFN_OK &&
        stopped.error == IFN_OK);
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_stream_get_info(stream, &info) == IFN_OK &&
                                  info.finish_reason == IFN_STREAM_STOP && !stopped.n);
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_stream_destroy(&stream) == IFN_OK);
    opts.stops = NULL;
    opts.stop_count = 0;
    ifn_chat_native_local_output replay = {0};
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(
        ifn_stream_create(t, &opts, NULL, ifn_chat_native_local_text, &replay, &stream) == IFN_OK);
    replay.stream = stream;
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(
        ifn_session_generate_ex(ifn_ar_compare_local_session, prompt, n, 128, &decode, out, 128,
                                ifn_chat_native_local_token, &replay, &stats) == IFN_OK &&
        replay.error == IFN_OK);
    IFN_CHAT_NATIVE_LOCAL_REQUIRE(ifn_stream_finish(stream, IFN_STREAM_LENGTH) == IFN_OK &&
                                  stats.generation.generated_tokens == output_count &&
                                  !memcmp(out, baseline, output_count * 4) && replay.n == first.n &&
                                  !memcmp(replay.content, first.content, first.n));
    printf("{\"native_chat\":true,\"prompt_tokens\":%zu,\"generated_tokens\":%zu,\"cache_reused\":%"
           "u,\"content_bytes\":%zu,\"reasoning_bytes\":%zu,\"stream_collection_parity\":true,"
           "\"stop_verified\":true}\n",
           n, output_count, prefix.reused_tokens, first.n, first.r);
    fprintf(stderr, "Native answer: %.*s\n", (int)first.n, first.content);
done:
    if (ifn_stream_destroy(&stream) != IFN_OK)
        failed = 1;
    ifn_tokenizer_scratch_destroy(scratch);
    ifn_tokenizer_close(t);
    if (ifn_decode_finish(0))
        failed = 1;
    return failed;
}
