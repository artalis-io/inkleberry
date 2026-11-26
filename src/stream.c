/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Transport-independent Unicode decoder and leftmost stop filter. Protocol
 * section classification follows the pinned Inkling server; buffers never grow. */
#include "ifn_chat.h"
#include "tokenizer.h"
#include <string.h>

typedef struct {
    unsigned char bytes[4];
    size_t used, need;
} ifn_stream_local_unicode;

struct ifn_stream {
    ifn_options allocator;
    const ifn_tokenizer *tokenizer;
    ifn_stream_options options;
    ifn_text_callback callback;
    void *context;
    char stops[8][64];
    size_t lengths[8];
    unsigned char pending[68];
    size_t pending_bytes;
    char batch[1024];
    size_t batch_bytes;
    uint32_t batch_channel;
    ifn_stream_local_unicode utf8[2];
    int32_t model, text, thinking, end, eos;
    uint32_t section, input_kind;
    size_t header_bytes;
    bool busy;
    ifn_stream_info info;
};

static ifn_status ifn_stream_local_fail(ifn_stream *s, ifn_status rc, const char *message) {
    s->info.error_status = rc;
    s->info.finish_reason = IFN_STREAM_ERROR;
    return ifn_error(rc, message);
}

static bool ifn_stream_local_running(const ifn_stream *s) {
    return s->info.finish_reason == IFN_STREAM_RUNNING;
}

static void ifn_stream_local_flush(ifn_stream *s) {
    if (!s->batch_bytes || !ifn_stream_local_running(s))
        return;
    size_t n = s->batch_bytes;
    s->batch_bytes = 0;
    uint64_t *counter =
        s->batch_channel == IFN_STREAM_CONTENT ? &s->info.content_bytes : &s->info.reasoning_bytes;
    if (n > UINT64_MAX - *counter) {
        (void)ifn_stream_local_fail(s, IFN_LIMIT, "stream byte count overflow");
        return;
    }
    *counter += n;
    if (s->callback && !s->callback(s->context, s->batch_channel, s->batch, n))
        s->info.finish_reason = IFN_STREAM_CALLBACK;
}

static void ifn_stream_local_emit(ifn_stream *s, uint32_t channel, const unsigned char *p,
                                  size_t n) {
    if (!ifn_stream_local_running(s) || !n)
        return;
    if (s->batch_bytes && (s->batch_channel != channel || n > sizeof s->batch - s->batch_bytes))
        ifn_stream_local_flush(s);
    if (!ifn_stream_local_running(s))
        return;
    s->batch_channel = channel;
    memcpy(s->batch + s->batch_bytes, p, n);
    s->batch_bytes += n;
}

/* Returns byte indexes; all stop strings and pending input are valid UTF-8,
 * so a match/prefix start is necessarily a character boundary. */
static void ifn_stream_local_scan(const ifn_stream *s, size_t *complete, size_t *potential) {
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

static void ifn_stream_local_filter(ifn_stream *s, const unsigned char *p, size_t n, bool final) {
    if (!ifn_stream_local_running(s))
        return;
    if (n) {
        memcpy(s->pending + s->pending_bytes, p, n);
        s->pending_bytes += n;
    }
    size_t complete, potential;
    ifn_stream_local_scan(s, &complete, &potential);
    if (complete < s->pending_bytes && (final || potential >= complete)) {
        ifn_stream_local_emit(s, IFN_STREAM_CONTENT, s->pending, complete);
        ifn_stream_local_flush(s);
        if (ifn_stream_local_running(s))
            s->info.finish_reason = IFN_STREAM_STOP;
        s->pending_bytes = 0;
        return;
    }
    size_t ready = final ? s->pending_bytes : (complete < potential ? complete : potential);
    ifn_stream_local_emit(s, IFN_STREAM_CONTENT, s->pending, ready);
    memmove(s->pending, s->pending + ready, s->pending_bytes - ready);
    s->pending_bytes -= ready;
}

static void ifn_stream_local_codepoint(ifn_stream *s, uint32_t channel, const unsigned char *p,
                                       size_t n) {
    if (channel == IFN_STREAM_CONTENT)
        ifn_stream_local_filter(s, p, n, false);
    else
        ifn_stream_local_emit(s, channel, p, n);
}

static void ifn_stream_local_replacement(ifn_stream *s, uint32_t channel) {
    if (!s->options.replace_invalid_utf8) {
        (void)ifn_stream_local_fail(s, IFN_FORMAT, "invalid or incomplete stream UTF-8");
        return;
    }
    s->info.utf8_replacements++;
    static const unsigned char bad[] = {0xef, 0xbf, 0xbd};
    ifn_stream_local_codepoint(s, channel, bad, 3);
}

static void ifn_stream_local_byte(ifn_stream *s, uint32_t channel, unsigned char b) {
    ifn_stream_local_unicode *u = &s->utf8[channel];
    for (;;) {
        if (!ifn_stream_local_running(s))
            return;
        if (!u->used) {
            if (b < 0x80) {
                ifn_stream_local_codepoint(s, channel, &b, 1);
                return;
            }
            u->need = b >= 0xc2 && b <= 0xdf   ? 2
                      : b >= 0xe0 && b <= 0xef ? 3
                      : b >= 0xf0 && b <= 0xf4 ? 4
                                               : 0;
            if (!u->need) {
                ifn_stream_local_replacement(s, channel);
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
            ifn_stream_local_replacement(s, channel);
            continue;
        }
        u->bytes[u->used++] = b;
        if (u->used == u->need) {
            u->used = 0;
            ifn_stream_local_codepoint(s, channel, u->bytes, u->need);
        }
        return;
    }
}

static void ifn_stream_local_finish_unicode(ifn_stream *s, uint32_t channel) {
    if (s->utf8[channel].used) {
        s->utf8[channel].used = 0;
        ifn_stream_local_replacement(s, channel);
    }
}

static void ifn_stream_local_finish(ifn_stream *s, uint32_t reason) {
    if (!ifn_stream_local_running(s))
        return;
    ifn_stream_local_finish_unicode(s, IFN_STREAM_REASONING);
    ifn_stream_local_finish_unicode(s, IFN_STREAM_CONTENT);
    ifn_stream_local_filter(s, NULL, 0, true);
    ifn_stream_local_flush(s);
    if (ifn_stream_local_running(s))
        s->info.finish_reason = reason;
}

static bool ifn_stream_local_valid_utf8(const char *p, size_t n) {
    size_t at = 0;
    uint32_t cp;
    while (at < n)
        if (!ifn_utf8_next((const unsigned char *)p, n, &at, &cp))
            return false;
    return true;
}

static int32_t ifn_stream_local_special(const ifn_tokenizer *t, const char *p) {
    int32_t id = ifn_special_lookup(t, p, strlen(p));
    return id >= 0 && t->entries[id].special ? id : -1;
}

ifn_status ifn_stream_create(const ifn_tokenizer *t, const ifn_stream_options *input,
                             const ifn_options *allocator, ifn_text_callback callback,
                             void *context, ifn_stream **out) {
    if (out)
        *out = NULL;
    if (!out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing stream output");
    ifn_stream_options o = IFN_STREAM_OPTIONS_INIT;
    if (input) {
        if (input->struct_size < sizeof o || input->api_version != IFN_API_VERSION)
            return ifn_error(IFN_INVALID_ARGUMENT, "stream options size/version");
        o = *input;
    }
    if (o.parse_chat > 1 || o.replace_invalid_utf8 > 1 || o.reserved || (o.stop_count && !o.stops))
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid stream options");
    if (o.stop_count > 8)
        return ifn_error(IFN_LIMIT, "at most eight stop strings");
    for (size_t i = 0; i < o.stop_count; i++) {
        if (!o.stops[i].data || !o.stops[i].bytes)
            return ifn_error(IFN_INVALID_ARGUMENT, "empty stop string");
        if (o.stops[i].bytes > 64)
            return ifn_error(IFN_LIMIT, "stop string exceeds 64 bytes");
        if (!ifn_stream_local_valid_utf8(o.stops[i].data, o.stops[i].bytes))
            return ifn_error(IFN_FORMAT, "stop string is not UTF-8");
    }
    int32_t ids[5] = {-1, -1, -1, -1, -1};
    if (o.parse_chat) {
        if (!t)
            return ifn_error(IFN_INVALID_ARGUMENT, "chat stream needs a tokenizer");
        const char *names[] = {"<|message_model|>", "<|content_text|>", "<|content_thinking|>",
                               "<|end_message|>", "<|content_model_end_sampling|>"};
        for (unsigned i = 0; i < 5; i++)
            if ((ids[i] = ifn_stream_local_special(t, names[i])) < 0)
                return ifn_error(IFN_UNSUPPORTED, "stream protocol marker missing");
    } else if (t)
        ids[4] = ifn_stream_local_special(t, "<|content_model_end_sampling|>");
    ifn_options a;
    ifn_status rc = ifn_options_resolve(allocator, &a);
    if (rc != IFN_OK)
        return rc;
    ifn_stream *s = a.allocate(a.context, sizeof *s);
    if (!s)
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate stream");
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
    (void)ifn_stream_reset(s, callback, context);
    *out = s;
    return ifn_error(IFN_OK, "");
}

ifn_status ifn_stream_reset(ifn_stream *s, ifn_text_callback callback, void *context) {
    if (!s)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing stream");
    if (s->busy)
        return ifn_error(IFN_BUSY, "stream callback active");
    s->callback = callback;
    s->context = context;
    s->pending_bytes = s->batch_bytes = s->header_bytes = 0;
    s->section = s->input_kind = 0;
    memset(s->utf8, 0, sizeof s->utf8);
    s->info = (ifn_stream_info)IFN_STREAM_INFO_INIT;
    return ifn_error(IFN_OK, "");
}

static ifn_status ifn_stream_local_result(ifn_stream *s) {
    return ifn_error(s->info.error_status, s->info.error_status ? "stream decoding failed" : "");
}

static ifn_status ifn_stream_local_begin(ifn_stream *s, uint32_t kind) {
    if (!s)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing stream");
    if (s->busy)
        return ifn_error(IFN_BUSY, "stream callback active");
    if (s->input_kind && s->input_kind != kind)
        return ifn_error(IFN_INVALID_ARGUMENT, "cannot mix byte and token input");
    return ifn_stream_local_result(s);
}

ifn_status ifn_stream_push_bytes(ifn_stream *s, const void *data, size_t n) {
    if ((!data && n) || n > 32u * 1024u * 1024u)
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid stream byte span");
    ifn_status rc = ifn_stream_local_begin(s, 1);
    if (rc != IFN_OK)
        return rc;
    if (!ifn_stream_local_running(s))
        return ifn_stream_local_result(s);
    s->busy = true;
    s->input_kind = 1;
    const unsigned char *p = data;
    for (size_t i = 0; i < n && ifn_stream_local_running(s); i++)
        ifn_stream_local_byte(s, IFN_STREAM_CONTENT, p[i]);
    ifn_stream_local_flush(s);
    s->busy = false;
    return ifn_stream_local_result(s);
}

ifn_status ifn_stream_push_token(ifn_stream *s, int32_t id) {
    ifn_status rc = ifn_stream_local_begin(s, 2);
    if (rc != IFN_OK)
        return rc;
    if (!s->tokenizer)
        return ifn_error(IFN_INVALID_ARGUMENT, "token stream needs a tokenizer");
    const void *data = NULL;
    size_t n = 0;
    rc = ifn_tokenizer_decode(s->tokenizer, id, 0, &data, &n);
    if (rc != IFN_OK)
        return rc;
    if (!ifn_stream_local_running(s))
        return ifn_stream_local_result(s);
    s->busy = true;
    s->input_kind = 2;
    if (s->info.tokens == UINT64_MAX) {
        (void)ifn_stream_local_fail(s, IFN_LIMIT, "stream token count overflow");
        goto done;
    }
    s->info.tokens++;
    if (id == s->eos) {
        ifn_stream_local_finish(s, IFN_STREAM_EOS);
        goto done;
    }
    if (s->options.parse_chat && s->tokenizer->entries[id].special) {
        if (id == s->model || id == s->text || id == s->thinking || id == s->end) {
            ifn_stream_local_finish_unicode(s, s->section == 2 ? IFN_STREAM_REASONING
                                                               : IFN_STREAM_CONTENT);
            ifn_stream_local_flush(s);
            s->section = id == s->model ? 1 : id == s->thinking ? 2 : 0;
            s->header_bytes = 0;
        } else
            (void)ifn_stream_local_fail(s, IFN_UNSUPPORTED,
                                        "unsupported generated tool/multimodal/control section");
        goto done;
    }
    if (!s->options.parse_chat && s->tokenizer->entries[id].special)
        goto done;
    if (s->section == 1) {
        if (n > 1024 - s->header_bytes)
            (void)ifn_stream_local_fail(s, IFN_LIMIT, "model header exceeds 1024 bytes");
        else
            s->header_bytes += n;
        goto done;
    }
    for (size_t i = 0; i < n && ifn_stream_local_running(s); i++)
        ifn_stream_local_byte(s, s->section == 2 ? IFN_STREAM_REASONING : IFN_STREAM_CONTENT,
                              ((const unsigned char *)data)[i]);
done:
    ifn_stream_local_flush(s);
    s->busy = false;
    return ifn_stream_local_result(s);
}

ifn_status ifn_stream_finish(ifn_stream *s, uint32_t reason) {
    if (!s || (reason != IFN_STREAM_LENGTH && reason != IFN_STREAM_CANCELLED))
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid stream finish");
    if (s->busy)
        return ifn_error(IFN_BUSY, "stream callback active");
    s->busy = true;
    ifn_stream_local_finish(s, reason);
    s->busy = false;
    return ifn_stream_local_result(s);
}

ifn_status ifn_stream_get_info(const ifn_stream *s, ifn_stream_info *out) {
    if (!s || !out || out->struct_size < sizeof *out || out->api_version != IFN_API_VERSION)
        return ifn_error(IFN_INVALID_ARGUMENT, "initialize stream info");
    if (s->busy)
        return ifn_error(IFN_BUSY, "stream callback active");
    *out = s->info;
    return ifn_error(IFN_OK, "");
}

ifn_status ifn_stream_destroy(ifn_stream **out) {
    if (!out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing stream handle");
    ifn_stream *s = *out;
    if (!s)
        return ifn_error(IFN_OK, "");
    if (s->busy)
        return ifn_error(IFN_BUSY, "stream callback active");
    ifn_options a = s->allocator;
    *out = NULL;
    a.release(a.context, s);
    return ifn_error(IFN_OK, "");
}
