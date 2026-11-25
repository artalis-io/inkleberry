/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Shared JSON primitives extracted from the C translation of pinned
 * src/safetensors.zig; adds bounded configuration validation/comparison. */
#include "json.h"
#include <string.h>

void sq_json_ws(sq_json *j) {
    while (j->pos < j->size && (j->data[j->pos] == ' ' || j->data[j->pos] == '\t' ||
                                j->data[j->pos] == '\r' || j->data[j->pos] == '\n'))
        j->pos++;
}

bool sq_json_take(sq_json *j, unsigned char c) {
    sq_json_ws(j);
    if (j->pos == j->size || j->data[j->pos] != c)
        return false;
    j->pos++;
    return true;
}

static bool hex4(sq_json *j, uint32_t *out) {
    uint32_t value = 0;
    for (unsigned i = 0; i < 4; i++) {
        if (j->pos == j->size)
            return false;
        unsigned char c = j->data[j->pos++];
        unsigned v;
        if (c >= '0' && c <= '9')
            v = c - '0';
        else if (c >= 'a' && c <= 'f')
            v = c - 'a' + 10u;
        else if (c >= 'A' && c <= 'F')
            v = c - 'A' + 10u;
        else
            return false;
        value = value * 16 + v;
    }
    *out = value;
    return true;
}

/* Fully validates UTF-8 and JSON escapes, even when discarding a string.
 * Names/keys use bounded decoded output; no raw escaped-name aliasing. */
bool sq_json_string(sq_json *j, char *out, size_t capacity) {
    if ((out && !capacity) || !sq_json_take(j, '"'))
        return false;
    size_t used = 0;
    while (j->pos < j->size) {
        uint32_t cp = j->data[j->pos++];
        if (cp == '"') {
            if (out)
                out[used] = 0;
            return true;
        }
        if (cp < 0x20)
            return false;
        if (cp == '\\') {
            if (j->pos == j->size)
                return false;
            unsigned char c = j->data[j->pos++];
            if (c == 'u') {
                if (!hex4(j, &cp))
                    return false;
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    uint32_t low;
                    if (j->size - j->pos < 2 || j->data[j->pos] != '\\' ||
                        j->data[j->pos + 1] != 'u')
                        return false;
                    j->pos += 2;
                    if (!hex4(j, &low) || low < 0xdc00 || low > 0xdfff)
                        return false;
                    cp = 0x10000 + (cp - 0xd800) * 1024 + low - 0xdc00;
                } else if (cp >= 0xdc00 && cp <= 0xdfff)
                    return false;
            } else if (c == '"' || c == '\\' || c == '/')
                cp = c;
            else if (c == 'b')
                cp = '\b';
            else if (c == 'f')
                cp = '\f';
            else if (c == 'n')
                cp = '\n';
            else if (c == 'r')
                cp = '\r';
            else if (c == 't')
                cp = '\t';
            else
                return false;
        } else if (cp >= 0x80) {
            unsigned remaining;
            uint32_t minimum;
            if (cp >= 0xc2 && cp <= 0xdf) {
                remaining = 1;
                minimum = 0x80;
                cp &= 0x1f;
            } else if (cp >= 0xe0 && cp <= 0xef) {
                remaining = 2;
                minimum = 0x800;
                cp &= 0x0f;
            } else if (cp >= 0xf0 && cp <= 0xf4) {
                remaining = 3;
                minimum = 0x10000;
                cp &= 7;
            } else
                return false;
            for (unsigned i = 0; i < remaining; i++) {
                if (j->pos == j->size || (j->data[j->pos] & 0xc0) != 0x80)
                    return false;
                cp = cp * 64 + (j->data[j->pos++] & 63u);
            }
            if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
                return false;
        }
        if (cp == 0)
            return false;
        unsigned char bytes[4];
        size_t n;
        if (cp < 0x80) {
            bytes[0] = (unsigned char)cp;
            n = 1;
        } else if (cp < 0x800) {
            bytes[0] = (unsigned char)(0xc0 | (cp >> 6));
            bytes[1] = (unsigned char)(0x80 | (cp & 63));
            n = 2;
        } else if (cp < 0x10000) {
            bytes[0] = (unsigned char)(0xe0 | (cp >> 12));
            bytes[1] = (unsigned char)(0x80 | ((cp >> 6) & 63));
            bytes[2] = (unsigned char)(0x80 | (cp & 63));
            n = 3;
        } else {
            bytes[0] = (unsigned char)(0xf0 | (cp >> 18));
            bytes[1] = (unsigned char)(0x80 | ((cp >> 12) & 63));
            bytes[2] = (unsigned char)(0x80 | ((cp >> 6) & 63));
            bytes[3] = (unsigned char)(0x80 | (cp & 63));
            n = 4;
        }
        if (out) {
            if (!capacity || used >= capacity || n >= capacity - used)
                return false;
            memcpy(out + used, bytes, n);
            used += n;
        }
    }
    return false;
}

bool sq_json_u64(sq_json *j, uint64_t *out) {
    sq_json_ws(j);
    if (j->pos == j->size || j->data[j->pos] < '0' || j->data[j->pos] > '9')
        return false;
    bool zero = j->data[j->pos] == '0';
    uint64_t value = 0;
    size_t start = j->pos;
    while (j->pos < j->size && j->data[j->pos] >= '0' && j->data[j->pos] <= '9') {
        unsigned digit = j->data[j->pos++] - '0';
        if (value > (UINT64_MAX - digit) / 10)
            return false;
        value = value * 10 + digit;
    }
    if (zero && j->pos - start != 1)
        return false;
    *out = value;
    return true;
}

bool sq_json_done(sq_json *j) {
    sq_json_ws(j);
    return j->pos == j->size;
}

/* Strict, depth-bounded generic validation; numbers never undergo floating
 * conversion. Object-schema callers separately reject duplicate/unknown keys. */
bool sq_json_skip(sq_json *j, unsigned depth) {
    if (depth > 32)
        return false;
    sq_json_ws(j);
    if (j->pos == j->size)
        return false;
    unsigned char c = j->data[j->pos];
    if (c == '"')
        return sq_json_string(j, NULL, 0);
    if (c == '{' || c == '[') {
        j->pos++;
        unsigned char end = c == '{' ? '}' : ']';
        if (sq_json_take(j, end))
            return true;
        do {
            if (c == '{' && (!sq_json_string(j, NULL, 0) || !sq_json_take(j, ':')))
                return false;
            if (!sq_json_skip(j, depth + 1))
                return false;
        } while (sq_json_take(j, ','));
        return sq_json_take(j, end);
    }
    const char *word = c == 't' ? "true" : c == 'f' ? "false" : c == 'n' ? "null" : NULL;
    if (word) {
        size_t n = strlen(word);
        if (n > j->size - j->pos || memcmp(j->data + j->pos, word, n))
            return false;
        j->pos += n;
        return true;
    }
    if (c == '-')
        j->pos++;
    if (j->pos == j->size)
        return false;
    c = j->data[j->pos];
    if (c == '0')
        j->pos++;
    else {
        if (c < '1' || c > '9')
            return false;
        do {
            j->pos++;
        } while (j->pos < j->size && j->data[j->pos] >= '0' && j->data[j->pos] <= '9');
    }
    if (j->pos < j->size && j->data[j->pos] == '.') {
        size_t start = ++j->pos;
        while (j->pos < j->size && j->data[j->pos] >= '0' && j->data[j->pos] <= '9')
            j->pos++;
        if (j->pos == start)
            return false;
    }
    if (j->pos < j->size && (j->data[j->pos] == 'e' || j->data[j->pos] == 'E')) {
        j->pos++;
        if (j->pos < j->size && (j->data[j->pos] == '+' || j->data[j->pos] == '-'))
            j->pos++;
        size_t start = j->pos;
        while (j->pos < j->size && j->data[j->pos] >= '0' && j->data[j->pos] <= '9')
            j->pos++;
        if (j->pos == start)
            return false;
    }
    return true;
}

bool sq_json_object(sq_json *j, sq_json_field *fields, size_t count) {
    if (!sq_json_take(j, '{'))
        return false;
    if (sq_json_take(j, '}'))
        return true;
    do {
        char name[128];
        if (!sq_json_string(j, name, sizeof name) || !sq_json_take(j, ':'))
            return false;
        size_t i = 0;
        while (i < count && strcmp(fields[i].name, name))
            i++;
        if (i == count || fields[i].present)
            return false;
        sq_json_ws(j);
        size_t start = j->pos;
        if (!sq_json_skip(j, 0))
            return false;
        fields[i].value = (sq_json){j->data + start, j->pos - start, 0};
        fields[i].present = true;
    } while (sq_json_take(j, ','));
    return sq_json_take(j, '}');
}

bool sq_json_literal(sq_json j, const char *text) {
    size_t n = strlen(text);
    sq_json_ws(&j);
    if (n > j.size - j.pos || memcmp(j.data + j.pos, text, n))
        return false;
    j.pos += n;
    return sq_json_done(&j);
}

bool sq_json_text(sq_json j, const char *text) {
    char decoded[1024];
    return sq_json_string(&j, decoded, sizeof decoded) && !strcmp(decoded, text) &&
           sq_json_done(&j);
}

bool sq_json_equal(sq_json a, sq_json b, unsigned depth) {
    if (depth > 32)
        return false;
    sq_json_ws(&a);
    sq_json_ws(&b);
    if (a.pos == a.size || b.pos == b.size || a.data[a.pos] != b.data[b.pos])
        return false;
    if (b.data[b.pos] == '{') {
        sq_json_field fields[16] = {0};
        char names[16][128];
        size_t n = 0;
        b.pos++;
        if (!sq_json_take(&b, '}')) {
            do {
                if (n == 16 || !sq_json_string(&b, names[n], sizeof names[n]) ||
                    !sq_json_take(&b, ':'))
                    return false;
                size_t start = b.pos;
                if (!sq_json_skip(&b, 0))
                    return false;
                fields[n].name = names[n];
                fields[n].value = (sq_json){b.data + start, b.pos - start, 0};
                n++;
            } while (sq_json_take(&b, ','));
            if (!sq_json_take(&b, '}'))
                return false;
        }
        sq_json_field actual[16] = {0};
        for (size_t i = 0; i < n; i++)
            actual[i].name = names[i];
        if (!sq_json_object(&a, actual, n))
            return false;
        for (size_t i = 0; i < n; i++)
            if (!actual[i].present || !sq_json_equal(actual[i].value, fields[i].value, depth + 1))
                return false;
    } else if (b.data[b.pos] == '[') {
        a.pos++;
        b.pos++;
        if (!sq_json_take(&b, ']')) {
            do {
                size_t x = a.pos, y = b.pos;
                if (!sq_json_skip(&a, 0) || !sq_json_skip(&b, 0) ||
                    !sq_json_equal((sq_json){a.data + x, a.pos - x, 0},
                                   (sq_json){b.data + y, b.pos - y, 0}, depth + 1))
                    return false;
                bool more = sq_json_take(&b, ',');
                if (sq_json_take(&a, ',') != more)
                    return false;
                if (!more)
                    break;
            } while (true);
            if (!sq_json_take(&b, ']'))
                return false;
        }
        if (!sq_json_take(&a, ']'))
            return false;
    } else if (b.data[b.pos] == '"') {
        char x[1024], y[1024];
        if (!sq_json_string(&a, x, sizeof x) || !sq_json_string(&b, y, sizeof y) || strcmp(x, y))
            return false;
    } else {
        size_t x = a.pos, y = b.pos;
        if (!sq_json_skip(&a, 0) || !sq_json_skip(&b, 0) || a.pos - x != b.pos - y ||
            memcmp(a.data + x, b.data + y, a.pos - x))
            return false;
    }
    return sq_json_done(&a) && sq_json_done(&b);
}
