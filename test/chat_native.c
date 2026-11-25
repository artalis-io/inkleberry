/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "decode_compare.c"
#include "sq_chat.h"

typedef struct {
    sq_stream *stream;
    sq_status error;
    char content[8192], reasoning[8192];
    size_t n, r;
} output;

static int text(void *p, uint32_t channel, const char *data, size_t n) {
    output *o = p;
    char *out = channel == SQ_STREAM_CONTENT ? o->content : o->reasoning;
    size_t *used = channel == SQ_STREAM_CONTENT ? &o->n : &o->r;
    if (n > 8192 - *used)
        return 0;
    memcpy(out + *used, data, n);
    *used += n;
    return 1;
}

static int token(void *p, int32_t id) {
    output *o = p;
    o->error = sq_stream_push_token(o->stream, id);
    sq_stream_info info = SQ_STREAM_INFO_INIT;
    if (o->error != SQ_OK || sq_stream_get_info(o->stream, &info) != SQ_OK)
        return 0;
    return info.finish_reason == SQ_STREAM_RUNNING;
}

#define REQUIRE(x)                                                                                 \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "native chat:%d %s\n", __LINE__, sq_last_error());                     \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    int failed = 0;
    sq_tokenizer *t = NULL;
    sq_tokenizer_scratch *scratch = NULL;
    sq_stream *stream = NULL;
    int32_t prompt[8192], baseline[128], out[128];
    size_t n = 0;
    uint32_t hint = 0;
    char path[1200], system[8192];
    system[0] = 0;
    for (unsigned i = 0; i < 64; i++)
        strcat(system, "You are a concise mathematical assistant. Return the numeric answer. ");
    sq_chat_message messages[] = {
        {SQ_CHAT_SYSTEM, 0, system, strlen(system), NULL, 0, NULL, 0},
        {SQ_CHAT_USER, 0, "What is 2 + 2?", 14, NULL, 0, NULL, 0},
        {SQ_CHAT_ASSISTANT, 0, "4.", 2, NULL, 0, NULL, 0},
        {SQ_CHAT_USER, 0, "What is that number plus 3?", 27, NULL, 0, NULL, 0}};
    REQUIRE(!sq_decode_start(argv[1], 2, 1, 1, 8192, 7, 64));
    snprintf(path, sizeof path, "%s/tokenizer.json", argv[1]);
    REQUIRE(sq_tokenizer_open(path, NULL, &t) == SQ_OK &&
            sq_tokenizer_scratch_create(NULL, &scratch) == SQ_OK);
    sq_chat_options chat = SQ_CHAT_OPTIONS_INIT;
    REQUIRE(sq_chat_encode(t, messages, 4, &chat, scratch, prompt, 8192, &n, &hint) == SQ_OK &&
            hint >= 128);
    REQUIRE(sq_session_prefix_cache(session, 1) == SQ_OK);
    sq_decode_options decode = SQ_DECODE_OPTIONS_INIT;
    decode.draft_tokens = 7;
    decode.adaptive = 1;
    decode.prompt_lookup = 1;
    decode.system_prefix_tokens = hint;
    sq_stream_options opts = SQ_STREAM_OPTIONS_INIT;
    output first = {0};
    REQUIRE(sq_stream_create(t, &opts, NULL, text, &first, &stream) == SQ_OK);
    first.stream = stream;
    sq_decode_stats stats;
    REQUIRE(sq_session_generate_ex(session, prompt, n, 128, &decode, baseline, 128, token, &first,
                                   &stats) == SQ_OK &&
            first.error == SQ_OK);
    size_t output_count = stats.generation.generated_tokens;
    REQUIRE(sq_stream_finish(stream, SQ_STREAM_LENGTH) == SQ_OK && first.n > 0);
    sq_stream_info info = SQ_STREAM_INFO_INIT;
    REQUIRE(sq_stream_get_info(stream, &info) == SQ_OK && !info.utf8_replacements);
    output collected = {0};
    REQUIRE(sq_stream_reset(stream, text, &collected) == SQ_OK);
    for (size_t i = 0; i < output_count; i++)
        REQUIRE(sq_stream_push_token(stream, baseline[i]) == SQ_OK);
    REQUIRE(sq_stream_finish(stream, SQ_STREAM_LENGTH) == SQ_OK && collected.n == first.n &&
            collected.r == first.r && !memcmp(collected.content, first.content, first.n) &&
            !memcmp(collected.reasoning, first.reasoning, first.r));
    output second = {0};
    REQUIRE(sq_stream_reset(stream, text, &second) == SQ_OK);
    second.stream = stream;
    REQUIRE(sq_session_generate_ex(session, prompt, n, 128, &decode, out, 128, token, &second,
                                   &stats) == SQ_OK &&
            second.error == SQ_OK);
    REQUIRE(stats.generation.generated_tokens == output_count &&
            !memcmp(out, baseline, output_count * 4));
    sq_prefix_info prefix = SQ_PREFIX_INFO_INIT;
    REQUIRE(sq_session_prefix_info(session, &prefix) == SQ_OK && prefix.reused_tokens >= hint);
    REQUIRE(sq_stream_finish(stream, SQ_STREAM_LENGTH) == SQ_OK && second.n == first.n &&
            !memcmp(second.content, first.content, first.n));
    REQUIRE(sq_stream_destroy(&stream) == SQ_OK);
    size_t stop_length = 1;
    while (stop_length < first.n && ((unsigned char)first.content[stop_length] & 0xc0) == 0x80)
        stop_length++;
    sq_text_span stop = {first.content, stop_length};
    opts.stops = &stop;
    opts.stop_count = 1;
    output stopped = {0};
    REQUIRE(sq_stream_create(t, &opts, NULL, text, &stopped, &stream) == SQ_OK);
    stopped.stream = stream;
    REQUIRE(sq_session_generate_ex(session, prompt, n, 128, &decode, out, 128, token, &stopped,
                                   &stats) == SQ_OK &&
            stopped.error == SQ_OK);
    REQUIRE(sq_stream_get_info(stream, &info) == SQ_OK && info.finish_reason == SQ_STREAM_STOP &&
            !stopped.n);
    REQUIRE(sq_stream_destroy(&stream) == SQ_OK);
    opts.stops = NULL;
    opts.stop_count = 0;
    output replay = {0};
    REQUIRE(sq_stream_create(t, &opts, NULL, text, &replay, &stream) == SQ_OK);
    replay.stream = stream;
    REQUIRE(sq_session_generate_ex(session, prompt, n, 128, &decode, out, 128, token, &replay,
                                   &stats) == SQ_OK &&
            replay.error == SQ_OK);
    REQUIRE(sq_stream_finish(stream, SQ_STREAM_LENGTH) == SQ_OK &&
            stats.generation.generated_tokens == output_count &&
            !memcmp(out, baseline, output_count * 4) && replay.n == first.n &&
            !memcmp(replay.content, first.content, first.n));
    printf("{\"native_chat\":true,\"prompt_tokens\":%zu,\"generated_tokens\":%zu,\"cache_reused\":%"
           "u,\"content_bytes\":%zu,\"reasoning_bytes\":%zu,\"stream_collection_parity\":true,"
           "\"stop_verified\":true}\n",
           n, output_count, prefix.reused_tokens, first.n, first.r);
    fprintf(stderr, "Native answer: %.*s\n", (int)first.n, first.content);
done:
    if (sq_stream_destroy(&stream) != SQ_OK)
        failed = 1;
    sq_tokenizer_scratch_destroy(scratch);
    sq_tokenizer_close(t);
    if (sq_decode_finish(0))
        failed = 1;
    return failed;
}
