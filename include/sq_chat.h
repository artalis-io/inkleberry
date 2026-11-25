/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_CHAT_H
#define SQ_CHAT_H
#include "sq.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Text conversations only; tools and multimodal payloads need separate adapters.
 * Spans are borrowed/immutable during encode. NULL assistant content is absent;
 * non-NULL zero-length content is an empty text block. */
enum { SQ_CHAT_SYSTEM, SQ_CHAT_DEVELOPER, SQ_CHAT_USER, SQ_CHAT_ASSISTANT };

typedef struct {
    uint32_t role, reserved;
    const char *content;
    size_t content_bytes;
    const char *reasoning;
    size_t reasoning_bytes;
    const char *name;
    size_t name_bytes;
} sq_chat_message;

typedef struct {
    size_t struct_size;
    uint32_t api_version, effort, keep_history_thinking, enable_thinking, reserved;
} sq_chat_options;

/* clang-format off */
#define SQ_CHAT_OPTIONS_INIT {sizeof(sq_chat_options),SQ_API_VERSION,SQ_THINK_NONE,0,0,0}
/* clang-format on */
/* <=1024 messages, <=32 MiB combined spans, <=128-byte names, strict UTF-8.
 * Trusted protocol markers and untrusted text are encoded separately. No
 * allocation. Caller owns exclusive scratch/output; keep output separate from
 * messages/options. Sizing/short-output calls return the complete needed count
 * and leave output untouched. The legacy single-user renderer is unchanged. */
SQ_API sq_status sq_chat_encode(const sq_tokenizer *, const sq_chat_message *, size_t,
                                const sq_chat_options *, sq_tokenizer_scratch *, int32_t *, size_t,
                                size_t *needed, uint32_t *system_prefix_tokens);

typedef struct sq_stream sq_stream;

typedef struct {
    const char *data;
    size_t bytes;
} sq_text_span;

enum { SQ_STREAM_CONTENT, SQ_STREAM_REASONING };

enum {
    SQ_STREAM_RUNNING,
    SQ_STREAM_STOP,
    SQ_STREAM_EOS,
    SQ_STREAM_LENGTH,
    SQ_STREAM_CALLBACK,
    SQ_STREAM_CANCELLED,
    SQ_STREAM_ERROR
};

typedef struct {
    size_t struct_size;
    uint32_t api_version, parse_chat, replace_invalid_utf8, reserved;
    const sq_text_span *stops;
    size_t stop_count;
} sq_stream_options;

/* clang-format off */
#define SQ_STREAM_OPTIONS_INIT {sizeof(sq_stream_options),SQ_API_VERSION,1,1,0,NULL,0}
/* clang-format on */
typedef int (*sq_text_callback)(void *, uint32_t channel, const char *, size_t);

typedef struct {
    size_t struct_size;
    uint32_t api_version, finish_reason, error_status;
    uint64_t tokens, content_bytes, reasoning_bytes, utf8_replacements;
} sq_stream_info;

/* clang-format off */
#define SQ_STREAM_INFO_INIT {sizeof(sq_stream_info),SQ_API_VERSION,0,0,0,0,0,0}
/* clang-format on */
/* Bounded, session-private decoder/filter. Tokenizer is borrowed until destroy;
 * NULL is allowed for byte-only use. Stops (<=8, 1..64 UTF-8 bytes each) are
 * copied at creation. No allocation per push. Stop matching applies to content,
 * including across calls/sections, and emits no stop bytes. Callback spans are
 * borrowed until callback return; consume a chunk before returning 0 to stop.
 * Produced-byte counts are not transport-delivery receipts. One thread owns a
 * stream; callbacks may not reenter it (BUSY). Do not mix byte/token input.
 * Replacement follows Unicode maximal-subpart decoding; strict mode returns
 * FORMAT instead. Decoding errors are sticky until reset; invalid arguments do not mutate state.
 * Token counts include controls. Full tool/multimodal output is explicitly unsupported, never
 * silently dropped. */
SQ_API sq_status sq_stream_create(const sq_tokenizer *, const sq_stream_options *,
                                  const sq_options *, sq_text_callback, void *, sq_stream **);
SQ_API sq_status sq_stream_reset(sq_stream *, sq_text_callback, void *);
SQ_API sq_status sq_stream_push_bytes(sq_stream *, const void *, size_t);
SQ_API sq_status sq_stream_push_token(sq_stream *, int32_t);
/* reason = LENGTH or CANCELLED; EOS is also recognized by push_token. Flushes
 * unmatched stop prefixes and final partial UTF-8 by the configured policy.
 * Idempotent after any normal finish. */
SQ_API sq_status sq_stream_finish(sq_stream *, uint32_t reason);
SQ_API sq_status sq_stream_get_info(const sq_stream *, sq_stream_info *);
SQ_API sq_status sq_stream_destroy(sq_stream **);
#ifdef __cplusplus
}
#endif
#endif
