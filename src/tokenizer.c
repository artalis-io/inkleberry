/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Translation/rework of pinned src/tokenizer.zig BPE and Unicode lexer.
 * Recoverable validation, caller-sized outputs and explicit scratch ownership;
 * upstream classification/chunk/merge behavior intentionally preserved. */
#include "tokenizer.h"
#include <string.h>

static bool ifn_tokenizer_local_cp_is_newline(uint32_t cp) {
    return cp == '\r' || cp == '\n';
}

static bool ifn_tokenizer_local_cp_is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C ||
           cp == 0x0085 || cp == 0x00A0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

static bool ifn_tokenizer_local_cp_is_digit(uint32_t cp) {
    if (cp >= '0' && cp <= '9')
        return true;
    if (cp < 0x80)
        return false;
    return (cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06F0 && cp <= 0x06F9) ||
           (cp >= 0x0966 && cp <= 0x096F) || (cp >= 0xFF10 && cp <= 0xFF19);
}

static bool ifn_tokenizer_local_cp_is_upper(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z')
        return true;
    if (cp < 0x80)
        return false;
    return (cp >= 0x00C0 && cp <= 0x00D6) || (cp >= 0x00D8 && cp <= 0x00DE) ||
           (cp >= 0x0391 && cp <= 0x03A9) || (cp >= 0x0410 && cp <= 0x042F) ||
           (cp >= 0xFF21 && cp <= 0xFF3A);
}

static bool ifn_tokenizer_local_cp_is_lower(uint32_t cp) {
    if (cp >= 'a' && cp <= 'z')
        return true;
    if (cp < 0x80)
        return false;
    return (cp >= 0x00DF && cp <= 0x00F6) || (cp >= 0x00F8 && cp <= 0x00FF) ||
           (cp >= 0x03B1 && cp <= 0x03C9) || (cp >= 0x0430 && cp <= 0x044F) ||
           (cp >= 0xFF41 && cp <= 0xFF5A);
}

static bool ifn_tokenizer_local_cp_is_other_letter_or_mark(uint32_t cp) {
    if (cp < 0x80)
        return false;
    return (cp >= 0x0100 && cp <= 0x024F) || (cp >= 0x0300 && cp <= 0x036F) ||
           (cp >= 0x0590 && cp <= 0x06FF) || (cp >= 0x0900 && cp <= 0x097F) ||
           (cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x3400 && cp <= 0x9FFF) ||
           (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0xF900 && cp <= 0xFAFF);
}

static bool ifn_tokenizer_local_cp_is_letter(uint32_t cp) {
    return ifn_tokenizer_local_cp_is_upper(cp) || ifn_tokenizer_local_cp_is_lower(cp) ||
           ifn_tokenizer_local_cp_is_other_letter_or_mark(cp);
}

static bool ifn_tokenizer_local_cp_is_cat_U(uint32_t cp) {
    return ifn_tokenizer_local_cp_is_upper(cp) ||
           ifn_tokenizer_local_cp_is_other_letter_or_mark(cp);
}

static bool ifn_tokenizer_local_cp_is_cat_L(uint32_t cp) {
    return ifn_tokenizer_local_cp_is_lower(cp) ||
           ifn_tokenizer_local_cp_is_other_letter_or_mark(cp);
}

bool ifn_utf8_next(const unsigned char *s, size_t n, size_t *pos, uint32_t *out) {
    if (*pos >= n)
        return false;
    uint32_t cp = s[(*pos)++], minimum = 0;
    unsigned rest = 0;
    if (cp < 128) {
        *out = cp;
        return true;
    }
    if (cp >= 0xc2 && cp <= 0xdf) {
        rest = 1;
        minimum = 0x80;
        cp &= 31;
    } else if (cp >= 0xe0 && cp <= 0xef) {
        rest = 2;
        minimum = 0x800;
        cp &= 15;
    } else if (cp >= 0xf0 && cp <= 0xf4) {
        rest = 3;
        minimum = 0x10000;
        cp &= 7;
    } else
        return false;
    for (unsigned i = 0; i < rest; i++) {
        if (*pos == n || (s[*pos] & 0xc0) != 0x80)
            return false;
        cp = cp * 64 + (s[(*pos)++] & 63u);
    }
    if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
        return false;
    *out = cp;
    return true;
}

static bool ifn_tokenizer_local_valid_utf8(const char *text, size_t bytes) {
    size_t p = 0;
    uint32_t cp;
    while (p < bytes)
        if (!ifn_utf8_next((const unsigned char *)text, bytes, &p, &cp))
            return false;
    return true;
}

static size_t ifn_tokenizer_local_contraction(const uint32_t *c, size_t n, size_t i) {
    if (i + 1 >= n || c[i] != '\'')
        return 0;
    uint32_t a = c[i + 1];
    if (a >= 'A' && a <= 'Z')
        a += 32;
    if (a == 's' || a == 't' || a == 'm' || a == 'd')
        return 2;
    if (i + 2 < n) {
        uint32_t b = c[i + 2];
        if (b >= 'A' && b <= 'Z')
            b += 32;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))
            return 3;
    }
    return 0;
}

static size_t ifn_tokenizer_local_branch1(const uint32_t *c, size_t n, size_t start) {
    size_t j = start;
    if (j < n && !ifn_tokenizer_local_cp_is_newline(c[j]) &&
        !ifn_tokenizer_local_cp_is_letter(c[j]) && !ifn_tokenizer_local_cp_is_digit(c[j]))
        j++;
    size_t run = j;
    while (j < n && ifn_tokenizer_local_cp_is_letter(c[j]))
        j++;
    size_t u = run;
    while (u < j && ifn_tokenizer_local_cp_is_cat_U(c[u]))
        u++;
    size_t l = u;
    if (l == j && u > run && ifn_tokenizer_local_cp_is_cat_L(c[u - 1]))
        l = u - 1;
    size_t end = l;
    while (end < j && ifn_tokenizer_local_cp_is_cat_L(c[end]))
        end++;
    return end > l ? end + ifn_tokenizer_local_contraction(c, n, end) : 0;
}

static size_t ifn_tokenizer_local_branch2(const uint32_t *c, size_t n, size_t start) {
    size_t j = start;
    if (j < n && !ifn_tokenizer_local_cp_is_newline(c[j]) &&
        !ifn_tokenizer_local_cp_is_letter(c[j]) && !ifn_tokenizer_local_cp_is_digit(c[j]))
        j++;
    size_t u = j;
    while (j < n && ifn_tokenizer_local_cp_is_cat_U(c[j]))
        j++;
    if (j == u)
        return 0;
    while (j < n && ifn_tokenizer_local_cp_is_cat_L(c[j]))
        j++;
    return j + ifn_tokenizer_local_contraction(c, n, j);
}

void ifn_tokenizer_scratch_destroy(ifn_tokenizer_scratch *s) {
    if (!s)
        return;
    ifn_options a = s->allocator;
    if (s->bpe)
        a.release(a.context, s->bpe);
    if (s->offsets)
        a.release(a.context, s->offsets);
    if (s->codepoints)
        a.release(a.context, s->codepoints);
    a.release(a.context, s);
}

ifn_status ifn_tokenizer_scratch_create(const ifn_options *options, ifn_tokenizer_scratch **out) {
    if (out)
        *out = NULL;
    if (!out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing scratch output");
    ifn_options a;
    ifn_status status = ifn_options_resolve(options, &a);
    if (status != IFN_OK)
        return status;
    ifn_tokenizer_scratch *s = a.allocate(a.context, sizeof *s);
    if (!s)
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate tokenizer scratch");
    memset(s, 0, sizeof *s);
    s->allocator = a;
    s->codepoints = a.allocate(a.context, IFN_TOK_CHUNK * sizeof *s->codepoints);
    s->offsets = a.allocate(a.context, (IFN_TOK_CHUNK + 1) * sizeof *s->offsets);
    s->bpe = a.allocate(a.context, IFN_TOK_PIECE * sizeof *s->bpe);
    if (!s->codepoints || !s->offsets || !s->bpe) {
        ifn_tokenizer_scratch_destroy(s);
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate tokenizer scratch arrays");
    }
    *out = s;
    return ifn_error(IFN_OK, "");
}

typedef struct {
    int32_t *ids;
    size_t capacity, count;
} ifn_tokenizer_local_sink;

static void ifn_tokenizer_local_emit(ifn_tokenizer_local_sink *out, int32_t id) {
    if (out->count < out->capacity)
        out->ids[out->count] = id;
    out->count++;
}

static const ifn_merge *ifn_tokenizer_local_merge(const ifn_tokenizer *t, int32_t a, int32_t b) {
    uint64_t pair = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    size_t slot = (size_t)ifn_pair_hash(pair) & (t->merge_slots - 1);
    for (size_t i = 0; i < t->merge_slots; i++, slot = (slot + 1) & (t->merge_slots - 1)) {
        if (t->merges[slot].pair == UINT64_MAX)
            return NULL;
        if (t->merges[slot].pair == pair)
            return &t->merges[slot];
    }
    return NULL;
}

static void ifn_tokenizer_local_piece(const ifn_tokenizer *t, const unsigned char *text, size_t n,
                                      ifn_tokenizer_scratch *s, ifn_tokenizer_local_sink *out) {
    size_t offset = 0;
    while (offset < n) {
        size_t end = n - offset > IFN_TOK_PIECE ? offset + IFN_TOK_PIECE : n;
        if (end < n) {
            unsigned back = 0;
            while (back < 4 && end > offset + 1 && (text[end] & 0xc0) == 0x80) {
                end--;
                back++;
            }
        }
        size_t count = end - offset;
        for (size_t i = 0; i < count; i++)
            s->bpe[i] = t->byte_ids[text[offset + i]];
        while (count >= 2) {
            const ifn_merge *best = NULL;
            size_t index = 0;
            for (size_t i = 0; i + 1 < count; i++) {
                const ifn_merge *candidate = ifn_tokenizer_local_merge(t, s->bpe[i], s->bpe[i + 1]);
                if (candidate && (!best || candidate->rank < best->rank)) {
                    best = candidate;
                    index = i;
                }
            }
            if (!best)
                break;
            int32_t a = s->bpe[index], b = s->bpe[index + 1];
            size_t write = 0;
            for (size_t read = 0; read < count; read++) {
                if (read + 1 < count && s->bpe[read] == a && s->bpe[read + 1] == b) {
                    s->bpe[write++] = best->merged;
                    read++;
                } else
                    s->bpe[write++] = s->bpe[read];
            }
            count = write;
        }
        for (size_t i = 0; i < count; i++)
            ifn_tokenizer_local_emit(out, s->bpe[i]);
        offset = end;
    }
}

static void ifn_tokenizer_local_chunk(const ifn_tokenizer *t, const unsigned char *text,
                                      size_t bytes, ifn_tokenizer_scratch *s,
                                      ifn_tokenizer_local_sink *out) {
    size_t n = 0, p = 0;
    while (p < bytes) {
        s->offsets[n] = (uint32_t)p;
        (void)ifn_utf8_next(text, bytes, &p, &s->codepoints[n]);
        n++;
    }
    s->offsets[n] = (uint32_t)bytes;
    const uint32_t *c = s->codepoints;
    for (size_t i = 0; i < n;) {
        size_t end = ifn_tokenizer_local_branch1(c, n, i);
        if (end <= i)
            end = ifn_tokenizer_local_branch2(c, n, i);
        if (end <= i) {
            if (ifn_tokenizer_local_cp_is_digit(c[i])) {
                end = i + 1;
                if (end < n && ifn_tokenizer_local_cp_is_digit(c[end]))
                    end++;
                if (end < n && ifn_tokenizer_local_cp_is_digit(c[end]))
                    end++;
            } else {
                size_t j = i;
                if (c[j] == ' ' && j + 1 < n && !ifn_tokenizer_local_cp_is_space(c[j + 1]) &&
                    !ifn_tokenizer_local_cp_is_letter(c[j + 1]) &&
                    !ifn_tokenizer_local_cp_is_digit(c[j + 1]))
                    j++;
                if (!ifn_tokenizer_local_cp_is_space(c[j]) &&
                    !ifn_tokenizer_local_cp_is_letter(c[j]) &&
                    !ifn_tokenizer_local_cp_is_digit(c[j])) {
                    while (j < n && !ifn_tokenizer_local_cp_is_space(c[j]) &&
                           !ifn_tokenizer_local_cp_is_letter(c[j]) &&
                           !ifn_tokenizer_local_cp_is_digit(c[j]))
                        j++;
                    while (j < n && (c[j] == '\r' || c[j] == '\n' || c[j] == '/'))
                        j++;
                    end = j;
                } else if (ifn_tokenizer_local_cp_is_space(c[i])) {
                    size_t ws = i, last = SIZE_MAX;
                    while (ws < n && ifn_tokenizer_local_cp_is_space(c[ws])) {
                        if (ifn_tokenizer_local_cp_is_newline(c[ws]))
                            last = ws;
                        ws++;
                    }
                    if (last != SIZE_MAX)
                        end = last + 1;
                    else if (ws == n)
                        end = ws;
                    else if (ws - i > 1)
                        end = ws - 1;
                    else
                        end = i + 1;
                } else
                    end = i + 1;
            }
        }
        ifn_tokenizer_local_piece(t, text + s->offsets[i], s->offsets[end] - s->offsets[i], s, out);
        i = end;
    }
}

static void ifn_tokenizer_local_ordinary(const ifn_tokenizer *t, const unsigned char *text,
                                         size_t bytes, ifn_tokenizer_scratch *s,
                                         ifn_tokenizer_local_sink *out) {
    size_t off = 0;
    while (off < bytes) {
        size_t end = bytes - off > IFN_TOK_CHUNK ? off + IFN_TOK_CHUNK : bytes;
        if (end < bytes) {
            size_t floor = end > off + 65536 ? end - 65536 : off + 1, split = 0, scan = end;
            while (scan > floor) {
                scan--;
                if (text[scan] == '\n') {
                    split = scan + 1;
                    break;
                }
            }
            if (!split) {
                scan = end;
                while (scan > floor) {
                    scan--;
                    if (text[scan] == ' ' || text[scan] == '\t' || text[scan] == '\r') {
                        split = scan;
                        break;
                    }
                }
            }
            if (split > off)
                end = split;
            else {
                unsigned back = 0;
                while (back < 4 && end > off + 1 && (text[end] & 0xc0) == 0x80) {
                    end--;
                    back++;
                }
            }
        }
        ifn_tokenizer_local_chunk(t, text + off, end - off, s, out);
        off = end;
    }
}

ifn_status ifn_tokenizer_encode(const ifn_tokenizer *t, const char *text, size_t bytes,
                                uint32_t parse_special, ifn_tokenizer_scratch *s, int32_t *ids,
                                size_t capacity, size_t *needed) {
    if (needed)
        *needed = 0;
    if (!t || !s || !needed || (!text && bytes) || (!ids && capacity) || parse_special > 1)
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid encode arguments");
    if (!ifn_tokenizer_local_valid_utf8(text, bytes))
        return ifn_error(IFN_FORMAT, "tokenizer input is not valid UTF-8");
    ifn_tokenizer_local_sink out = {ids, capacity, 0};
    const unsigned char *input = (const unsigned char *)text;
    size_t segment = 0, p = 0;
    if (parse_special)
        while (p < bytes && bytes - p >= 2) {
            if (input[p] != '<' || input[p + 1] != '|') {
                p++;
                continue;
            }
            size_t i = 0;
            while (i < t->special_count) {
                const ifn_token_entry *e = &t->entries[t->specials[i]];
                if (e->length <= bytes - p && !memcmp(input + p, t->arena + e->offset, e->length))
                    break;
                i++;
            }
            if (i < t->special_count) {
                ifn_tokenizer_local_ordinary(t, input + segment, p - segment, s, &out);
                ifn_tokenizer_local_emit(&out, t->specials[i]);
                p += t->entries[t->specials[i]].length;
                segment = p;
            } else
                p += 2;
        }
    if (segment < bytes)
        ifn_tokenizer_local_ordinary(t, input + segment, bytes - segment, s, &out);
    *needed = out.count;
    return ifn_error(out.count > capacity ? IFN_BUFFER_TOO_SMALL : IFN_OK,
                     out.count > capacity ? "token buffer too small" : "");
}

ifn_status ifn_tokenizer_decode(const ifn_tokenizer *t, int32_t id, uint32_t skip_special,
                                const void **out, size_t *bytes) {
    if (out)
        *out = NULL;
    if (bytes)
        *bytes = 0;
    if (!t || !out || !bytes || skip_special > 1)
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid decode arguments");
    if (id < 0 || (size_t)id >= t->id_count || !t->entries[id].present)
        return ifn_error(IFN_NOT_FOUND, "token ID not found");
    const ifn_token_entry *e = &t->entries[id];
    *out = t->arena + e->offset;
    *bytes = skip_special && e->special ? 0 : e->length;
    return ifn_error(IFN_OK, "");
}

ifn_status ifn_render_user_prompt(const char *text, size_t bytes, ifn_thinking_effort effort,
                                  char *out, size_t capacity, size_t *needed) {
    if (needed)
        *needed = 0;
    if (!needed || (!text && bytes) || (!out && capacity) || effort > IFN_THINK_MAX)
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid prompt arguments");
    if (!ifn_tokenizer_local_valid_utf8(text, bytes))
        return ifn_error(IFN_FORMAT, "prompt is not valid UTF-8");
    const char *levels[] = {"0", "0.1", "0.2", "0.7", "0.9", "0.99"};
    const char *prefix = "<|message_system|><|content_text|>Thinking effort level: ";
    const char *middle = "<|end_message|><|message_user|><|content_text|>";
    const char *suffix = "<|end_message|>";
    size_t n = bytes;
    if (!ifn_size_add(n, strlen(prefix), &n) || !ifn_size_add(n, strlen(levels[effort]), &n) ||
        !ifn_size_add(n, strlen(middle), &n) || !ifn_size_add(n, strlen(suffix), &n))
        return ifn_error(IFN_LIMIT, "prompt size overflow");
    *needed = n;
    if (capacity < n)
        return ifn_error(IFN_BUFFER_TOO_SMALL, "prompt buffer too small");
    size_t at = 0;
#define IFN_TOKENIZER_LOCAL_APPEND(p, k)                                                           \
    do {                                                                                           \
        size_t z = (k);                                                                            \
        if (z)                                                                                     \
            memcpy(out + at, (p), z);                                                              \
        at += z;                                                                                   \
    } while (0)
    IFN_TOKENIZER_LOCAL_APPEND(prefix, strlen(prefix));
    IFN_TOKENIZER_LOCAL_APPEND(levels[effort], strlen(levels[effort]));
    IFN_TOKENIZER_LOCAL_APPEND(middle, strlen(middle));
    IFN_TOKENIZER_LOCAL_APPEND(text, bytes);
    IFN_TOKENIZER_LOCAL_APPEND(suffix, strlen(suffix));
#undef IFN_TOKENIZER_LOCAL_APPEND
    return ifn_error(IFN_OK, "");
}
