/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Text subset of pinned server.zig's segmented chat encoder. */
#include "sq_chat.h"
#include "tokenizer.h"
#include <string.h>

static bool utf8(const char *p, size_t n) {
    size_t at = 0;
    uint32_t cp;
    while (at < n)
        if (!sq_utf8_next((const unsigned char *)p, n, &at, &cp))
            return false;
    return true;
}

static bool marker(const sq_tokenizer *t, const char *p) {
    int32_t id = sq_special_lookup(t, p, strlen(p));
    return id >= 0 && t->entries[id].special;
}

typedef struct {
    const sq_tokenizer *t;
    sq_tokenizer_scratch *scratch;
    int32_t *out;
    size_t capacity, count;
    sq_status status;
} encoder;

static void append(encoder *e, const char *p, size_t n, uint32_t special) {
    if (e->status != SQ_OK || !n)
        return;
    size_t count = 0;
    sq_status rc =
        sq_tokenizer_encode(e->t, p, n, special, e->scratch, e->out ? e->out + e->count : NULL,
                            e->out ? e->capacity - e->count : 0, &count);
    if (rc != SQ_OK && !(rc == SQ_BUFFER_TOO_SMALL && !e->out)) {
        e->status = rc;
        return;
    }
    if (!sq_size_add(e->count, count, &e->count))
        e->status = SQ_LIMIT;
}

static void tag(encoder *e, const char *p) {
    append(e, p, strlen(p), 1);
}

static void effort(encoder *e, uint32_t value) {
    static const char *levels[] = {"0", "0.1", "0.2", "0.7", "0.9", "0.99"};
    tag(e, "<|message_system|><|content_text|>Thinking effort level: ");
    tag(e, levels[value]);
    tag(e, "<|end_message|>");
}

static void block(encoder *e, const sq_chat_message *m, const char *kind, const char *p, size_t n) {
    static const char *roles[] = {"<|message_system|>", "<|message_system|>", "<|message_user|>",
                                  "<|message_model|>"};
    tag(e, roles[m->role]);
    append(e, m->name, m->name_bytes, 0);
    tag(e, kind);
    append(e, p, n, 0);
    tag(e, "<|end_message|>");
}

static uint32_t pass(encoder *e, const sq_chat_message *m, size_t count, const sq_chat_options *o) {
    bool emitted = false;
    uint32_t prefix = 0;
    for (size_t i = 0; i < count && e->status == SQ_OK; i++) {
        bool sys = m[i].role <= SQ_CHAT_DEVELOPER;
        if (!emitted && !sys) {
            effort(e, o->effort);
            emitted = true;
            prefix = (uint32_t)e->count;
        }
        bool reasoning =
            m[i].role == SQ_CHAT_ASSISTANT && o->keep_history_thinking && m[i].reasoning_bytes;
        if (reasoning)
            block(e, m + i, "<|content_thinking|>", m[i].reasoning, m[i].reasoning_bytes);
        bool text = m[i].content && (m[i].content_bytes || !reasoning);
        if (text)
            block(e, m + i, "<|content_text|>", m[i].content, m[i].content_bytes);
        if (m[i].role == SQ_CHAT_ASSISTANT && (reasoning || text))
            tag(e, "<|content_model_end_sampling|>");
        if (sys && !emitted)
            prefix = (uint32_t)e->count;
    }
    if (!emitted) {
        effort(e, o->effort);
        prefix = (uint32_t)e->count;
    }
    if (!o->enable_thinking)
        tag(e, "<|message_model|><|content_text|>");
    return prefix;
}

sq_status sq_chat_encode(const sq_tokenizer *t, const sq_chat_message *m, size_t count,
                         const sq_chat_options *input, sq_tokenizer_scratch *scratch, int32_t *out,
                         size_t capacity, size_t *needed, uint32_t *prefix) {
    if (needed)
        *needed = 0;
    if (prefix)
        *prefix = 0;
    if (!t || !m || !count || !scratch || !needed || !prefix || (!out && capacity))
        return sq_error(SQ_INVALID_ARGUMENT, "invalid chat arguments");
    if (count > 1024)
        return sq_error(SQ_LIMIT, "chat exceeds 1024 messages");
    sq_chat_options o = SQ_CHAT_OPTIONS_INIT;
    if (input) {
        if (input->struct_size < sizeof o || input->api_version != SQ_API_VERSION)
            return sq_error(SQ_INVALID_ARGUMENT, "chat options size/version");
        o = *input;
    }
    if (o.effort > SQ_THINK_MAX || o.keep_history_thinking > 1 || o.enable_thinking > 1 ||
        o.reserved)
        return sq_error(SQ_INVALID_ARGUMENT, "invalid chat options");
    size_t bytes = 0;
    for (size_t i = 0; i < count; i++) {
        if (m[i].role > SQ_CHAT_ASSISTANT || m[i].reserved || (!m[i].content && !m[i].reasoning) ||
            (!m[i].content && m[i].content_bytes) || (!m[i].reasoning && m[i].reasoning_bytes) ||
            (!m[i].name && m[i].name_bytes) ||
            (m[i].role != SQ_CHAT_ASSISTANT && (!m[i].content || m[i].reasoning_bytes)))
            return sq_error(SQ_INVALID_ARGUMENT, "invalid text chat message");
        if (m[i].name_bytes > 128 || !sq_size_add(bytes, m[i].content_bytes, &bytes) ||
            !sq_size_add(bytes, m[i].reasoning_bytes, &bytes) ||
            !sq_size_add(bytes, m[i].name_bytes, &bytes) || bytes > 32u * 1024u * 1024u)
            return sq_error(SQ_LIMIT, "chat text/name limit exceeded");
        if (!utf8(m[i].content, m[i].content_bytes) ||
            !utf8(m[i].reasoning, m[i].reasoning_bytes) || !utf8(m[i].name, m[i].name_bytes))
            return sq_error(SQ_FORMAT, "chat text is not UTF-8");
    }
    const char *tags[] = {"<|message_system|>",
                          "<|message_user|>",
                          "<|message_model|>",
                          "<|content_text|>",
                          "<|content_thinking|>",
                          "<|end_message|>",
                          "<|content_model_end_sampling|>"};
    for (size_t i = 0; i < sizeof tags / sizeof *tags; i++)
        if (!marker(t, tags[i]))
            return sq_error(SQ_UNSUPPORTED, "chat protocol marker missing");
    encoder e = {t, scratch, NULL, 0, 0, SQ_OK};
    uint32_t system = pass(&e, m, count, &o);
    if (e.status != SQ_OK)
        return sq_error(e.status, "chat sizing failed");
    *needed = e.count;
    *prefix = system;
    if (capacity < e.count)
        return sq_error(SQ_BUFFER_TOO_SMALL, "chat token buffer too small");
    e.out = out;
    e.capacity = capacity;
    e.count = 0;
    (void)pass(&e, m, count, &o);
    return sq_error(e.status, e.status == SQ_OK ? "" : "chat encoding failed");
}
