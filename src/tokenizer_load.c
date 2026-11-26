/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Rework of pinned src/tokenizer.zig: bounded, recoverable loader and ownership.
 * Fixed-table semantics retained; allocation sizes adapt to actual input counts. */
#define _POSIX_C_SOURCE 200809L
#include "tokenizer.h"
#include "json.h"
#include "tokenizer_profile.h"
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

uint64_t ifn_token_hash(const void *data, size_t size) {
    const unsigned char *p = data;
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; i++)
        h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h ? h : 1;
}

uint64_t ifn_pair_hash(uint64_t key) {
    uint64_t h = key + UINT64_C(0x9e3779b97f4a7c15);
    h = (h ^ (h >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)) * UINT64_C(0x94d049bb133111eb);
    h ^= h >> 31;
    return h ? h : 1;
}

int32_t ifn_special_lookup(const ifn_tokenizer *t, const void *bytes, size_t size) {
    for (size_t i = 0; i < t->special_count; i++) {
        int32_t id = t->specials[i];
        const ifn_token_entry *e = &t->entries[id];
        if (e->length == size && !memcmp(t->arena + e->offset, bytes, size))
            return id;
    }
    return -1;
}

int32_t ifn_vocab_lookup(const ifn_tokenizer *t, const void *bytes, size_t size) {
    size_t slot = (size_t)ifn_token_hash(bytes, size) & (t->vocab_slots - 1);
    for (size_t i = 0; i < t->vocab_slots; i++, slot = (slot + 1) & (t->vocab_slots - 1)) {
        uint32_t value = t->vocab[slot];
        if (!value)
            return -1;
        const ifn_token_entry *e = &t->entries[value - 1];
        if (e->length == size && !memcmp(t->arena + e->offset, bytes, size))
            return (int32_t)(value - 1);
    }
    return -1;
}

static bool raw_token(const char *text, unsigned char *out, size_t *length) {
    size_t n = strlen(text), p = 0, k = 0;
    while (p < n) {
        uint32_t cp;
        if (!ifn_utf8_next((const unsigned char *)text, n, &p, &cp))
            return false;
        int value = -1;
        if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255))
            value = (int)cp;
        else if (cp >= 256) {
            uint32_t next = 256;
            for (unsigned b = 0; b < 256; b++) {
                if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174)
                    continue;
                if (next++ == cp) {
                    value = (int)b;
                    break;
                }
            }
        }
        if (value < 0)
            return false; /* fixed GPT byte-level alphabet */
        out[k++] = (unsigned char)value;
    }
    *length = k;
    return k > 0;
}

static bool equal(ifn_json value, const char *expected) {
    return ifn_json_equal(value, (ifn_json){(const unsigned char *)expected, strlen(expected), 0},
                          0);
}

static bool fields(ifn_json value, ifn_json_field *f, size_t n) {
    if (!ifn_json_object(&value, f, n) || !ifn_json_done(&value))
        return false;
    for (size_t i = 0; i < n; i++)
        if (!f[i].present)
            return false;
    return true;
}

static ifn_json_field field(const char *name) {
    ifn_json_field f = {0};
    f.name = name;
    return f;
}

/* A supported-profile check prevents silently treating normalization, added-
 * token whitespace flags or arbitrary regexes as the pinned Inkling tokenizer. */
static bool profile(ifn_json json, ifn_json *vocab, ifn_json *merges, ifn_json *added) {
    ifn_json_field top[] = {field("version"),        field("truncation"), field("padding"),
                            field("added_tokens"),   field("normalizer"), field("pre_tokenizer"),
                            field("post_processor"), field("decoder"),    field("model")};
    if (!fields(json, top, 9) || !ifn_json_text(top[0].value, "1.0") ||
        !equal(top[1].value, "null") || !equal(top[2].value, "null") ||
        !equal(top[4].value, "null"))
        return false;
    const char *pre = PRETOKENIZER_PROFILE;
    if (!equal(top[5].value, pre) ||
        !equal(top[6].value, "{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":"
                             "false,\"use_regex\":true}") ||
        !equal(top[7].value, "{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":"
                             "true,\"use_regex\":true}"))
        return false;
    ifn_json_field model[] = {field("type"),
                              field("dropout"),
                              field("unk_token"),
                              field("continuing_subword_prefix"),
                              field("end_of_word_suffix"),
                              field("fuse_unk"),
                              field("byte_fallback"),
                              field("ignore_merges"),
                              field("vocab"),
                              field("merges")};
    if (!fields(top[8].value, model, 10) || !ifn_json_text(model[0].value, "BPE"))
        return false;
    for (size_t i = 1; i < 5; i++)
        if (!equal(model[i].value, "null"))
            return false;
    if (!equal(model[5].value, "false") || !equal(model[6].value, "false") ||
        (!equal(model[7].value, "false") && !equal(model[7].value, "true")))
        return false;
    *vocab = model[8].value;
    *merges = model[9].value;
    *added = top[3].value;
    return true;
}

static bool added_entry(ifn_json *j, char *name, size_t *id) {
    ifn_json_field f[] = {field("id"),     field("content"), field("single_word"),
                          field("lstrip"), field("rstrip"),  field("normalized"),
                          field("special")};
    if (!ifn_json_object(j, f, 7))
        return false;
    for (size_t i = 0; i < 7; i++)
        if (!f[i].present)
            return false;
    for (size_t i = 2; i < 6; i++)
        if (!equal(f[i].value, "false"))
            return false;
    uint64_t n;
    if (!equal(f[6].value, "true") || !ifn_json_u64(&f[0].value, &n) ||
        !ifn_json_done(&f[0].value) || n >= IFN_TOK_IDS ||
        !ifn_json_string(&f[1].value, name, 4096) || !name[0])
        return false;
    *id = (size_t)n;
    return true;
}

static size_t slots(size_t count) {
    size_t n = 16;
    while (n < count * 2 && n < 1048576)
        n *= 2;
    return n;
}

static void *owned(ifn_tokenizer *t, size_t n, size_t width) {
    size_t bytes;
    if (!ifn_size_mul(n, width, &bytes))
        return NULL;
    void *p = t->allocator.allocate(t->allocator.context, bytes);
    if (p)
        memset(p, 0, bytes);
    return p;
}

static bool store(ifn_tokenizer *t, const void *bytes, size_t length, size_t id, bool special) {
    if (!length || length > 4095 || id >= t->id_count || t->entries[id].present ||
        length > t->arena_size - t->arena_used)
        return false;
    ifn_token_entry *e = &t->entries[id];
    e->offset = (uint32_t)t->arena_used;
    e->length = (uint32_t)length;
    e->present = true;
    e->special = special;
    memcpy(t->arena + t->arena_used, bytes, length);
    t->arena_used += length;
    if (special) {
        if (t->special_count == 256 || length < 2 || memcmp(bytes, "<|", 2))
            return false;
        for (size_t i = 0; i < t->special_count; i++) {
            const ifn_token_entry *p = &t->entries[t->specials[i]];
            if (p->length == length && !memcmp(t->arena + p->offset, bytes, length))
                return false;
        }
        t->specials[t->special_count++] = (int32_t)id;
    } else {
        if (ifn_vocab_lookup(t, bytes, length) >= 0)
            return false;
        size_t slot = (size_t)ifn_token_hash(bytes, length) & (t->vocab_slots - 1);
        while (t->vocab[slot])
            slot = (slot + 1) & (t->vocab_slots - 1);
        t->vocab[slot] = (uint32_t)id + 1;
    }
    return true;
}

void ifn_tokenizer_close(ifn_tokenizer *t) {
    if (!t)
        return;
    ifn_options a = t->allocator;
    if (t->merges)
        a.release(a.context, t->merges);
    if (t->vocab)
        a.release(a.context, t->vocab);
    if (t->entries)
        a.release(a.context, t->entries);
    if (t->arena)
        a.release(a.context, t->arena);
    a.release(a.context, t);
}

ifn_status ifn_tokenizer_open_memory(const void *data, size_t bytes, const ifn_options *options,
                                     ifn_tokenizer **out) {
    if (out)
        *out = NULL;
    if (!data || !out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing tokenizer input/output");
    if (bytes > IFN_TOK_JSON)
        return ifn_error(IFN_LIMIT, "tokenizer JSON exceeds 64 MiB");
    ifn_options allocator;
    ifn_status status = ifn_options_resolve(options, &allocator);
    if (status != IFN_OK)
        return status;
    ifn_json vocab, merges, added;
    if (!profile((ifn_json){data, bytes, 0}, &vocab, &merges, &added))
        return ifn_error(IFN_FORMAT, "unsupported or malformed Inkling tokenizer profile");
    /* Count before allocation; each item is independently bounded. */
    ifn_json v = vocab, m = merges, a = added;
    size_t nv = 0, nm = 0, maxid = 0, arena_size = 0;
    char name[4096];
    if (!ifn_json_take(&v, '{') || ifn_json_take(&v, '}'))
        return ifn_error(IFN_FORMAT, "empty/invalid vocabulary");
    do {
        uint64_t id;
        if (!ifn_json_string(&v, name, sizeof name) || !name[0] || !ifn_json_take(&v, ':') ||
            !ifn_json_u64(&v, &id) || id >= IFN_TOK_IDS || ++nv > IFN_TOK_IDS)
            return ifn_error(IFN_FORMAT, "invalid vocabulary ID/name/count");
        if (id > maxid)
            maxid = (size_t)id;
        arena_size += strlen(name);
    } while (ifn_json_take(&v, ','));
    if (!ifn_json_take(&v, '}') || !ifn_json_done(&v) || !ifn_json_take(&m, '['))
        return ifn_error(IFN_FORMAT, "invalid vocabulary/merges");
    if (!ifn_json_take(&m, ']')) {
        do {
            if (++nm > IFN_TOK_MERGES || !ifn_json_skip(&m, 0))
                return ifn_error(IFN_LIMIT, "invalid or excessive merge count");
        } while (ifn_json_take(&m, ','));
        if (!ifn_json_take(&m, ']'))
            return ifn_error(IFN_FORMAT, "invalid merges");
    }
    if (!ifn_json_done(&m) || !ifn_json_take(&a, '['))
        return ifn_error(IFN_FORMAT, "invalid added tokens");
    size_t ns = 0;
    if (!ifn_json_take(&a, ']')) {
        do {
            size_t id;
            if (++ns > 256 || !added_entry(&a, name, &id))
                return ifn_error(IFN_FORMAT, "invalid added token");
            if (id > maxid)
                maxid = id;
            arena_size += strlen(name);
        } while (ifn_json_take(&a, ','));
        if (!ifn_json_take(&a, ']'))
            return ifn_error(IFN_FORMAT, "invalid added tokens");
    }
    if (!ifn_json_done(&a) || arena_size > IFN_TOK_ARENA)
        return ifn_error(IFN_LIMIT, "token byte arena exceeds 8 MiB");
    ifn_tokenizer *t = allocator.allocate(allocator.context, sizeof *t);
    if (!t)
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate tokenizer");
    memset(t, 0, sizeof *t);
    t->allocator = allocator;
    t->arena_size = arena_size;
    t->id_count = maxid + 1;
    t->vocab_slots = slots(nv);
    t->merge_slots = slots(nm);
    t->arena = owned(t, arena_size, 1);
    t->entries = owned(t, t->id_count, sizeof *t->entries);
    t->vocab = owned(t, t->vocab_slots, sizeof *t->vocab);
    t->merges = owned(t, t->merge_slots, sizeof *t->merges);
    if (!t->arena || !t->entries || !t->vocab || !t->merges) {
        ifn_tokenizer_close(t);
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate tokenizer tables");
    }
    for (size_t i = 0; i < t->merge_slots; i++)
        t->merges[i].pair = UINT64_MAX;
    v = vocab;
    (void)ifn_json_take(&v, '{');
    do {
        uint64_t id;
        unsigned char raw[4096];
        size_t n;
        if (!ifn_json_string(&v, name, sizeof name) || !ifn_json_take(&v, ':') ||
            !ifn_json_u64(&v, &id) || !raw_token(name, raw, &n) ||
            !store(t, raw, n, (size_t)id, false))
            goto invalid;
    } while (ifn_json_take(&v, ','));
    a = added;
    (void)ifn_json_take(&a, '[');
    if (!ifn_json_take(&a, ']'))
        do {
            size_t id;
            if (!added_entry(&a, name, &id) || !store(t, name, strlen(name), id, true))
                goto invalid;
        } while (ifn_json_take(&a, ','));
    for (size_t i = 0; i < 256; i++) {
        unsigned char b = (unsigned char)i;
        t->byte_ids[i] = ifn_vocab_lookup(t, &b, 1);
        if (t->byte_ids[i] < 0)
            goto invalid;
    }
    for (size_t i = 1; i < t->special_count; i++) {
        int32_t x = t->specials[i];
        size_t j = i;
        while (j && t->entries[t->specials[j - 1]].length < t->entries[x].length) {
            t->specials[j] = t->specials[j - 1];
            j--;
        }
        t->specials[j] = x;
    }
    m = merges;
    (void)ifn_json_take(&m, '[');
    for (size_t rank = 0; rank < nm; rank++) {
        char left[4096], right[4096];
        unsigned char raw[8192];
        size_t na, nb;
        if (ifn_json_take(&m, '[')) {
            if (!ifn_json_string(&m, left, sizeof left) || !ifn_json_take(&m, ',') ||
                !ifn_json_string(&m, right, sizeof right) || !ifn_json_take(&m, ']'))
                goto invalid;
        } else {
            if (!ifn_json_string(&m, name, sizeof name))
                goto invalid;
            char *space = strchr(name, ' ');
            if (!space || space == name || !space[1])
                goto invalid;
            *space = 0;
            memcpy(left, name, strlen(name) + 1);
            memcpy(right, space + 1, strlen(space + 1) + 1);
        }
        if (!raw_token(left, raw, &na) || !raw_token(right, raw + na, &nb))
            goto invalid;
        int32_t ia = ifn_vocab_lookup(t, raw, na), ib = ifn_vocab_lookup(t, raw + na, nb),
                ic = ifn_vocab_lookup(t, raw, na + nb);
        if (ia < 0 || ib < 0 || ic < 0)
            goto invalid;
        uint64_t pair = ((uint64_t)(uint32_t)ia << 32) | (uint32_t)ib;
        size_t slot = (size_t)ifn_pair_hash(pair) & (t->merge_slots - 1);
        while (t->merges[slot].pair != UINT64_MAX && t->merges[slot].pair != pair)
            slot = (slot + 1) & (t->merge_slots - 1);
        if (t->merges[slot].pair == UINT64_MAX)
            t->merges[slot] = (ifn_merge){pair, (uint32_t)rank, ic};
        if (rank + 1 < nm && !ifn_json_take(&m, ','))
            goto invalid;
    }
    *out = t;
    return ifn_error(IFN_OK, "");
invalid:
    ifn_tokenizer_close(t);
    return ifn_error(IFN_FORMAT, "invalid/duplicate tokenizer vocabulary or merge");
}

ifn_status ifn_tokenizer_open(const char *path, const ifn_options *options, ifn_tokenizer **out) {
    if (out)
        *out = NULL;
    if (!path || !path[0] || !out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing tokenizer path/output");
    ifn_options checked;
    ifn_status status = ifn_options_resolve(options, &checked);
    if (status != IFN_OK)
        return status;
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0)
        return ifn_error(IFN_IO, "cannot open tokenizer");
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size <= 0 ||
        (uintmax_t)st.st_size > IFN_TOK_JSON) {
        (void)close(fd);
        return ifn_error(IFN_FORMAT, "invalid tokenizer file size/type");
    }
    size_t n = (size_t)st.st_size;
    void *map = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
    (void)close(fd);
    if (map == MAP_FAILED)
        return ifn_error(IFN_IO, "cannot map tokenizer");
    status = ifn_tokenizer_open_memory(map, n, &checked, out);
    (void)munmap(map, n);
    return status;
}
