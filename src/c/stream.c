/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Transport-independent Unicode decoder and leftmost stop filter. Protocol
 * section classification follows the pinned Inkling server; buffers never grow. */
#include "sq_chat.h"
#include "tokenizer.h"
#include <string.h>

typedef struct {
    unsigned char bytes[4];
    size_t used, need;
} unicode;

struct sq_stream {
    sq_options allocator;
    const sq_tokenizer *tokenizer;
    sq_stream_options options;
    sq_text_callback callback;
    void *context;
    char stops[8][64];
    size_t lengths[8];
    unsigned char pending[68];
    size_t pending_bytes;
    char batch[1024];
    size_t batch_bytes;
    uint32_t batch_channel;
    unicode utf8[2];
    int32_t model, text, thinking, end, eos;
    uint32_t section, input_kind;
    size_t header_bytes;
    bool busy;
    sq_stream_info info;
};

static sq_status fail(sq_stream *s, sq_status rc, const char *message) {
    s->info.error_status = rc;
    s->info.finish_reason = SQ_STREAM_ERROR;
    return sq_error(rc, message);
}

static bool running(const sq_stream *s) {
    return s->info.finish_reason == SQ_STREAM_RUNNING;
}

static void flush(sq_stream *s) {
    if (!s->batch_bytes || !running(s))
        return;
    size_t n = s->batch_bytes;
    s->batch_bytes = 0;
    uint64_t *counter =
        s->batch_channel == SQ_STREAM_CONTENT ? &s->info.content_bytes : &s->info.reasoning_bytes;
    if (n > UINT64_MAX - *counter) {
        (void)fail(s, SQ_LIMIT, "stream byte count overflow");
        return;
    }
    *counter += n;
    if (s->callback && !s->callback(s->context, s->batch_channel, s->batch, n))
        s->info.finish_reason = SQ_STREAM_CALLBACK;
}

static void emit(sq_stream *s, uint32_t channel, const unsigned char *p, size_t n) {
    if (!running(s) || !n)
        return;
    if (s->batch_bytes && (s->batch_channel != channel || n > sizeof s->batch - s->batch_bytes))
        flush(s);
    if (!running(s))
        return;
    s->batch_channel = channel;
    memcpy(s->batch + s->batch_bytes, p, n);
    s->batch_bytes += n;
}

/* Returns byte indexes; all stop strings and pending input are valid UTF-8,
 * so a match/prefix start is necessarily a character boundary. */
static void scan(const sq_stream *s, size_t *complete, size_t *potential) {
    *complete = *potential = s->pending_bytes;
    for (size_t i = 0; i < s->pending_bytes; i++)
        for (size_t j = 0; j < s->options.stop_count; j++) {
            size_t available = s->pending_bytes - i, n = s->lengths[j];
            if (available >= n) {
                if (!memcmp(s->pending + i, s->stops[j], n) && i < *complete)
                    *complete = i;
            } else if (!memcmp(s->pending + i, s->stops[j], available) && i < *potential)
                *potential = i;
        }
}

static void filter(sq_stream *s, const unsigned char *p, size_t n, bool final) {
    if (!running(s))
        return;
    if (n) {
        memcpy(s->pending + s->pending_bytes, p, n);
        s->pending_bytes += n;
    }
    size_t complete, potential;
    scan(s, &complete, &potential);
    if (complete < s->pending_bytes && (final || potential >= complete)) {
        emit(s, SQ_STREAM_CONTENT, s->pending, complete);
        flush(s);
        if (running(s))
            s->info.finish_reason = SQ_STREAM_STOP;
        s->pending_bytes = 0;
        return;
    }
    size_t ready = final ? s->pending_bytes : (complete < potential ? complete : potential);
    emit(s, SQ_STREAM_CONTENT, s->pending, ready);
    memmove(s->pending, s->pending + ready, s->pending_bytes - ready);
    s->pending_bytes -= ready;
}

static void codepoint(sq_stream *s, uint32_t channel, const unsigned char *p, size_t n) {
    if (channel == SQ_STREAM_CONTENT)
        filter(s, p, n, false);
    else
        emit(s, channel, p, n);
}

static void replacement(sq_stream *s, uint32_t channel) {
    if (!s->options.replace_invalid_utf8) {
        (void)fail(s, SQ_FORMAT, "invalid or incomplete stream UTF-8");
        return;
    }
    s->info.utf8_replacements++;
    static const unsigned char bad[] = {0xef, 0xbf, 0xbd};
    codepoint(s, channel, bad, 3);
}

static void byte(sq_stream *s, uint32_t channel, unsigned char b) {
    unicode *u = &s->utf8[channel];
    for (;;) {
        if (!running(s))
            return;
        if (!u->used) {
            if (b < 0x80) {
                codepoint(s, channel, &b, 1);
                return;
            }
            u->need = b >= 0xc2 && b <= 0xdf   ? 2
                      : b >= 0xe0 && b <= 0xef ? 3
                      : b >= 0xf0 && b <= 0xf4 ? 4
                                               : 0;
            if (!u->need) {
                replacement(s, channel);
                return;
            }
            u->bytes[0] = b;
            u->used = 1;
            return;
        }
        bool valid = b >= 0x80 && b <= 0xbf;
        if (u->used == 1) {
            if (u->bytes[0] == 0xe0 && b < 0xa0)
                valid = false;
            if (u->bytes[0] == 0xed && b > 0x9f)
                valid = false;
            if (u->bytes[0] == 0xf0 && b < 0x90)
                valid = false;
            if (u->bytes[0] == 0xf4 && b > 0x8f)
                valid = false;
        }
        if (!valid) {
            u->used = 0;
            replacement(s, channel);
            continue;
        }
        u->bytes[u->used++] = b;
        if (u->used == u->need) {
            u->used = 0;
            codepoint(s, channel, u->bytes, u->need);
        }
        return;
    }
}

static void finish_unicode(sq_stream *s, uint32_t channel) {
    if (s->utf8[channel].used) {
        s->utf8[channel].used = 0;
        replacement(s, channel);
    }
}

static void finish(sq_stream *s, uint32_t reason) {
    if (!running(s))
        return;
    finish_unicode(s, SQ_STREAM_REASONING);
    finish_unicode(s, SQ_STREAM_CONTENT);
    filter(s, NULL, 0, true);
    flush(s);
    if (running(s))
        s->info.finish_reason = reason;
}

static bool valid_utf8(const char *p, size_t n) {
    size_t at = 0;
    uint32_t cp;
    while (at < n)
        if (!sq_utf8_next((const unsigned char *)p, n, &at, &cp))
            return false;
    return true;
}

static int32_t special(const sq_tokenizer *t, const char *p) {
    int32_t id = sq_special_lookup(t, p, strlen(p));
    return id >= 0 && t->entries[id].special ? id : -1;
}

sq_status sq_stream_create(const sq_tokenizer *t, const sq_stream_options *input,
                           const sq_options *allocator, sq_text_callback callback, void *context,
                           sq_stream **out) {
    if (out)
        *out = NULL;
    if (!out)
        return sq_error(SQ_INVALID_ARGUMENT, "missing stream output");
    sq_stream_options o = SQ_STREAM_OPTIONS_INIT;
    if (input) {
        if (input->struct_size < sizeof o || input->api_version != SQ_API_VERSION)
            return sq_error(SQ_INVALID_ARGUMENT, "stream options size/version");
        o = *input;
    }
    if (o.parse_chat > 1 || o.replace_invalid_utf8 > 1 || o.reserved || (o.stop_count && !o.stops))
        return sq_error(SQ_INVALID_ARGUMENT, "invalid stream options");
    if (o.stop_count > 8)
        return sq_error(SQ_LIMIT, "at most eight stop strings");
    for (size_t i = 0; i < o.stop_count; i++) {
        if (!o.stops[i].data || !o.stops[i].bytes)
            return sq_error(SQ_INVALID_ARGUMENT, "empty stop string");
        if (o.stops[i].bytes > 64)
            return sq_error(SQ_LIMIT, "stop string exceeds 64 bytes");
        if (!valid_utf8(o.stops[i].data, o.stops[i].bytes))
            return sq_error(SQ_FORMAT, "stop string is not UTF-8");
    }
    int32_t ids[5] = {-1, -1, -1, -1, -1};
    if (o.parse_chat) {
        if (!t)
            return sq_error(SQ_INVALID_ARGUMENT, "chat stream needs a tokenizer");
        const char *names[] = {"<|message_model|>", "<|content_text|>", "<|content_thinking|>",
                               "<|end_message|>", "<|content_model_end_sampling|>"};
        for (unsigned i = 0; i < 5; i++)
            if ((ids[i] = special(t, names[i])) < 0)
                return sq_error(SQ_UNSUPPORTED, "stream protocol marker missing");
    } else if (t)
        ids[4] = special(t, "<|content_model_end_sampling|>");
    sq_options a;
    sq_status rc = sq_options_resolve(allocator, &a);
    if (rc != SQ_OK)
        return rc;
    sq_stream *s = a.allocate(a.context, sizeof *s);
    if (!s)
        return sq_error(SQ_OUT_OF_MEMORY, "allocate stream");
    memset(s, 0, sizeof *s);
    s->allocator = a;
    s->tokenizer = t;
    s->options = o;
    s->options.stops = NULL;
    s->model = ids[0];
    s->text = ids[1];
    s->thinking = ids[2];
    s->end = ids[3];
    s->eos = ids[4];
    for (size_t i = 0; i < o.stop_count; i++) {
        memcpy(s->stops[i], o.stops[i].data, o.stops[i].bytes);
        s->lengths[i] = o.stops[i].bytes;
    }
    (void)sq_stream_reset(s, callback, context);
    *out = s;
    return sq_error(SQ_OK, "");
}

sq_status sq_stream_reset(sq_stream *s, sq_text_callback callback, void *context) {
    if (!s)
        return sq_error(SQ_INVALID_ARGUMENT, "missing stream");
    if (s->busy)
        return sq_error(SQ_BUSY, "stream callback active");
    s->callback = callback;
    s->context = context;
    s->pending_bytes = s->batch_bytes = s->header_bytes = 0;
    s->section = s->input_kind = 0;
    memset(s->utf8, 0, sizeof s->utf8);
    s->info = (sq_stream_info)SQ_STREAM_INFO_INIT;
    return sq_error(SQ_OK, "");
}

static sq_status result(sq_stream *s) {
    return sq_error(s->info.error_status, s->info.error_status ? "stream decoding failed" : "");
}

static sq_status begin(sq_stream *s, uint32_t kind) {
    if (!s)
        return sq_error(SQ_INVALID_ARGUMENT, "missing stream");
    if (s->busy)
        return sq_error(SQ_BUSY, "stream callback active");
    if (s->input_kind && s->input_kind != kind)
        return sq_error(SQ_INVALID_ARGUMENT, "cannot mix byte and token input");
    return result(s);
}

sq_status sq_stream_push_bytes(sq_stream *s, const void *data, size_t n) {
    if ((!data && n) || n > 32u * 1024u * 1024u)
        return sq_error(SQ_INVALID_ARGUMENT, "invalid stream byte span");
    sq_status rc = begin(s, 1);
    if (rc != SQ_OK)
        return rc;
    if (!running(s))
        return result(s);
    s->busy = true;
    s->input_kind = 1;
    const unsigned char *p = data;
    for (size_t i = 0; i < n && running(s); i++)
        byte(s, SQ_STREAM_CONTENT, p[i]);
    flush(s);
    s->busy = false;
    return result(s);
}

sq_status sq_stream_push_token(sq_stream *s, int32_t id) {
    sq_status rc = begin(s, 2);
    if (rc != SQ_OK)
        return rc;
    if (!s->tokenizer)
        return sq_error(SQ_INVALID_ARGUMENT, "token stream needs a tokenizer");
    const void *data = NULL;
    size_t n = 0;
    rc = sq_tokenizer_decode(s->tokenizer, id, 0, &data, &n);
    if (rc != SQ_OK)
        return rc;
    if (!running(s))
        return result(s);
    s->busy = true;
    s->input_kind = 2;
    if (s->info.tokens == UINT64_MAX) {
        (void)fail(s, SQ_LIMIT, "stream token count overflow");
        goto done;
    }
    s->info.tokens++;
    if (id == s->eos) {
        finish(s, SQ_STREAM_EOS);
        goto done;
    }
    if (s->options.parse_chat && s->tokenizer->entries[id].special) {
        if (id == s->model || id == s->text || id == s->thinking || id == s->end) {
            finish_unicode(s, s->section == 2 ? SQ_STREAM_REASONING : SQ_STREAM_CONTENT);
            flush(s);
            s->section = id == s->model ? 1 : id == s->thinking ? 2 : 0;
            s->header_bytes = 0;
        } else
            (void)fail(s, SQ_UNSUPPORTED, "unsupported generated tool/multimodal/control section");
        goto done;
    }
    if (!s->options.parse_chat && s->tokenizer->entries[id].special)
        goto done;
    if (s->section == 1) {
        if (n > 1024 - s->header_bytes)
            (void)fail(s, SQ_LIMIT, "model header exceeds 1024 bytes");
        else
            s->header_bytes += n;
        goto done;
    }
    for (size_t i = 0; i < n && running(s); i++)
        byte(s, s->section == 2 ? SQ_STREAM_REASONING : SQ_STREAM_CONTENT,
             ((const unsigned char *)data)[i]);
done:
    flush(s);
    s->busy = false;
    return result(s);
}

sq_status sq_stream_finish(sq_stream *s, uint32_t reason) {
    if (!s || (reason != SQ_STREAM_LENGTH && reason != SQ_STREAM_CANCELLED))
        return sq_error(SQ_INVALID_ARGUMENT, "invalid stream finish");
    if (s->busy)
        return sq_error(SQ_BUSY, "stream callback active");
    s->busy = true;
    finish(s, reason);
    s->busy = false;
    return result(s);
}

sq_status sq_stream_get_info(const sq_stream *s, sq_stream_info *out) {
    if (!s || !out || out->struct_size < sizeof *out || out->api_version != SQ_API_VERSION)
        return sq_error(SQ_INVALID_ARGUMENT, "initialize stream info");
    if (s->busy)
        return sq_error(SQ_BUSY, "stream callback active");
    *out = s->info;
    return sq_error(SQ_OK, "");
}

sq_status sq_stream_destroy(sq_stream **out) {
    if (!out)
        return sq_error(SQ_INVALID_ARGUMENT, "missing stream handle");
    sq_stream *s = *out;
    if (!s)
        return sq_error(SQ_OK, "");
    if (s->busy)
        return sq_error(SQ_BUSY, "stream callback active");
    sq_options a = s->allocator;
    *out = NULL;
    a.release(a.context, s);
    return sq_error(SQ_OK, "");
}
