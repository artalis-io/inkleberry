/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_CHAT_H
#define IFN_CHAT_H
#include "ifn.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Text conversations only; tools and multimodal payloads need separate adapters.
 * Spans are borrowed/immutable during encode. NULL assistant content is absent;
 * non-NULL zero-length content is an empty text block. */
enum { IFN_CHAT_SYSTEM, IFN_CHAT_DEVELOPER, IFN_CHAT_USER, IFN_CHAT_ASSISTANT };

typedef struct {
    uint32_t role, reserved;
    const char *content;
    size_t content_bytes;
    const char *reasoning;
    size_t reasoning_bytes;
    const char *name;
    size_t name_bytes;
} ifn_chat_message;

typedef struct {
    size_t struct_size;
    uint32_t api_version, effort, keep_history_thinking, enable_thinking, reserved;
} ifn_chat_options;

/* clang-format off */
#define IFN_CHAT_OPTIONS_INIT {sizeof(ifn_chat_options),IFN_API_VERSION,IFN_THINK_NONE,0,0,0}
/* clang-format on */
/* <=1024 messages, <=32 MiB combined spans, <=128-byte names, strict UTF-8.
 * Trusted protocol markers and untrusted text are encoded separately. No
 * allocation. Caller owns exclusive scratch/output; keep output separate from
 * messages/options. Sizing/short-output calls return the complete needed count
 * and leave output untouched. The legacy single-user renderer is unchanged. */
IFN_API ifn_status ifn_chat_encode(const ifn_tokenizer *, const ifn_chat_message *, size_t,
                                   const ifn_chat_options *, ifn_tokenizer_scratch *, int32_t *,
                                   size_t, size_t *needed, uint32_t *system_prefix_tokens);

typedef struct ifn_stream ifn_stream;

typedef struct {
    const char *data;
    size_t bytes;
} ifn_text_span;

enum { IFN_STREAM_CONTENT, IFN_STREAM_REASONING };

enum {
    IFN_STREAM_RUNNING,
    IFN_STREAM_STOP,
    IFN_STREAM_EOS,
    IFN_STREAM_LENGTH,
    IFN_STREAM_CALLBACK,
    IFN_STREAM_CANCELLED,
    IFN_STREAM_ERROR
};

typedef struct {
    size_t struct_size;
    uint32_t api_version, parse_chat, replace_invalid_utf8, reserved;
    const ifn_text_span *stops;
    size_t stop_count;
} ifn_stream_options;

/* clang-format off */
#define IFN_STREAM_OPTIONS_INIT {sizeof(ifn_stream_options),IFN_API_VERSION,1,1,0,NULL,0}
/* clang-format on */
typedef int (*ifn_text_callback)(void *, uint32_t channel, const char *, size_t);

typedef struct {
    size_t struct_size;
    uint32_t api_version, finish_reason, error_status;
    uint64_t tokens, content_bytes, reasoning_bytes, utf8_replacements;
} ifn_stream_info;

/* clang-format off */
#define IFN_STREAM_INFO_INIT {sizeof(ifn_stream_info),IFN_API_VERSION,0,0,0,0,0,0}
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
IFN_API ifn_status ifn_stream_create(const ifn_tokenizer *, const ifn_stream_options *,
                                     const ifn_options *, ifn_text_callback, void *, ifn_stream **);
IFN_API ifn_status ifn_stream_reset(ifn_stream *, ifn_text_callback, void *);
IFN_API ifn_status ifn_stream_push_bytes(ifn_stream *, const void *, size_t);
IFN_API ifn_status ifn_stream_push_token(ifn_stream *, int32_t);
/* reason = LENGTH or CANCELLED; EOS is also recognized by push_token. Flushes
 * unmatched stop prefixes and final partial UTF-8 by the configured policy.
 * Idempotent after any normal finish. */
IFN_API ifn_status ifn_stream_finish(ifn_stream *, uint32_t reason);
IFN_API ifn_status ifn_stream_get_info(const ifn_stream *, ifn_stream_info *);
IFN_API ifn_status ifn_stream_destroy(ifn_stream **);
#ifdef __cplusplus
}
#endif
#endif
