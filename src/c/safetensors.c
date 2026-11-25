/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * C translation/rework of src/safetensors.zig at d10c0abb2c8990be81c08ba84c66a8333a691b33.
 * Changes: recoverable validation, dtypes/shapes, strict JSON and range checks,
 * injected allocation and memory-input support; no user-input assertions. */
#define _POSIX_C_SOURCE 200809L
#include "internal.h"
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct sq_safetensors {
    sq_options allocator;
    const unsigned char *data;
    size_t size, base, count;
    bool mapped;
    sq_tensor_info *tensors;
    uint16_t table[4096];
};

typedef struct {
    const unsigned char *data;
    size_t size, pos;
} json;

static void ws(json *j) {
    while (j->pos < j->size && (j->data[j->pos] == ' ' || j->data[j->pos] == '\t' ||
                                j->data[j->pos] == '\r' || j->data[j->pos] == '\n'))
        j->pos++;
}

static bool take(json *j, unsigned char c) {
    ws(j);
    if (j->pos == j->size || j->data[j->pos] != c)
        return false;
    j->pos++;
    return true;
}

static bool hex4(json *j, uint32_t *out) {
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
static bool string(json *j, char *out, size_t capacity) {
    if (!take(j, '"'))
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

static bool integer(json *j, uint64_t *out) {
    ws(j);
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

static uint64_t hash_name(const char *name, size_t length) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < length; i++)
        hash = (hash ^ (unsigned char)name[i]) * UINT64_C(1099511628211);
    return hash;
}

static size_t lookup(const sq_safetensors *file, const char *name, size_t length) {
    size_t slot = (size_t)(hash_name(name, length) & 4095u);
    for (size_t probe = 0; probe < 4096; probe++, slot = (slot + 1) & 4095u) {
        uint16_t index = file->table[slot];
        if (index == UINT16_MAX)
            break;
        if (strlen(file->tensors[index].name) == length &&
            !memcmp(file->tensors[index].name, name, length))
            return index;
    }
    return SIZE_MAX;
}

static bool tensor(json *j, sq_tensor_info *info) {
    if (!take(j, '{'))
        return false;
    unsigned fields = 0;
    do {
        char key[32];
        if (!string(j, key, sizeof key) || !take(j, ':'))
            return false;
        unsigned bit;
        if (!strcmp(key, "dtype")) {
            char type[32];
            bit = 1;
            if (!string(j, type, sizeof type))
                return false;
            if (!strcmp(type, "U8"))
                info->dtype = SQ_U8;
            else if (!strcmp(type, "BF16"))
                info->dtype = SQ_BF16;
            else if (!strcmp(type, "F32"))
                info->dtype = SQ_F32;
            else if (!strcmp(type, "F8_E4M3"))
                info->dtype = SQ_F8_E4M3;
            else
                return false;
        } else if (!strcmp(key, "shape")) {
            bit = 2;
            if (!take(j, '['))
                return false;
            if (!take(j, ']')) {
                do {
                    if (info->rank == SQ_MAX_RANK || !integer(j, &info->shape[info->rank]))
                        return false;
                    info->rank++;
                } while (take(j, ','));
                if (!take(j, ']'))
                    return false;
            }
        } else if (!strcmp(key, "data_offsets")) {
            uint64_t end;
            bit = 4;
            if (!take(j, '[') || !integer(j, &info->offset) || !take(j, ',') || !integer(j, &end) ||
                !take(j, ']') || end < info->offset)
                return false;
            info->bytes = end - info->offset;
        } else
            return false;
        if (fields & bit)
            return false;
        fields |= bit;
    } while (take(j, ','));
    return take(j, '}') && fields == 7;
}

static bool metadata(json *j) {
    if (!take(j, '{'))
        return false;
    if (take(j, '}'))
        return true;
    do {
        if (!string(j, NULL, 0) || !take(j, ':') || !string(j, NULL, 0))
            return false;
    } while (take(j, ','));
    return take(j, '}');
}

static sq_status parse(sq_safetensors *file) {
    if (file->size < 10)
        return sq_error(SQ_FORMAT, "truncated safetensors file");
    uint64_t header = 0;
    for (unsigned i = 0; i < 8; i++)
        header |= (uint64_t)file->data[i] << (8 * i);
    if (header < 2 || header > SQ_MAX_HEADER || header > file->size - 8)
        return sq_error(SQ_FORMAT, "invalid safetensors header length");
    file->base = 8 + (size_t)header;
    json j = {file->data + 8, (size_t)header, 0};
    if (!take(&j, '{'))
        return sq_error(SQ_FORMAT, "expected tensor directory");
    bool seen_metadata = false;
    if (!take(&j, '}')) {
        do {
            char name[SQ_MAX_NAME + 1];
            if (!string(&j, name, sizeof name) || !name[0] || !take(&j, ':'))
                return sq_error(SQ_FORMAT, "invalid tensor name/key");
            if (!strcmp(name, "__metadata__")) {
                if (seen_metadata || !metadata(&j))
                    return sq_error(SQ_FORMAT, "invalid metadata");
                seen_metadata = true;
                continue;
            }
            if (file->count == SQ_MAX_TENSORS || lookup(file, name, strlen(name)) != SIZE_MAX)
                return sq_error(SQ_FORMAT, "duplicate tensor or tensor limit exceeded");
            sq_tensor_info *info = &file->tensors[file->count];
            memset(info, 0, sizeof *info);
            memcpy(info->name, name, strlen(name) + 1);
            if (!tensor(&j, info))
                return sq_error(SQ_FORMAT, "invalid tensor schema/dtype/shape");
            size_t elements = 1;
            for (uint32_t i = 0; i < info->rank; i++) {
                if (info->shape[i] > SIZE_MAX ||
                    !sq_size_mul(elements, (size_t)info->shape[i], &elements))
                    return sq_error(SQ_FORMAT, "tensor shape overflow");
            }
            size_t expected;
            size_t item = info->dtype == SQ_F32 ? 4 : info->dtype == SQ_BF16 ? 2 : 1;
            if (!sq_size_mul(elements, item, &expected) || expected != info->bytes ||
                info->offset > file->size - file->base ||
                info->bytes > file->size - file->base - info->offset)
                return sq_error(SQ_FORMAT, "tensor size/offset mismatch");
            size_t slot = (size_t)(hash_name(name, strlen(name)) & 4095u);
            while (file->table[slot] != UINT16_MAX)
                slot = (slot + 1) & 4095u;
            file->table[slot] = (uint16_t)file->count++;
        } while (take(&j, ','));
        if (!take(&j, '}'))
            return sq_error(SQ_FORMAT, "unterminated tensor directory");
    }
    ws(&j);
    if (j.pos != j.size)
        return sq_error(SQ_FORMAT, "trailing header data");
    uint64_t total = 0, payload = file->size - file->base;
    for (size_t i = 0; i < file->count; i++) {
        const sq_tensor_info *a = &file->tensors[i];
        if (a->bytes > payload - total)
            return sq_error(SQ_FORMAT, "overlapping tensor payload");
        total += a->bytes;
        for (size_t k = 0; k < i; k++) {
            const sq_tensor_info *b = &file->tensors[k];
            if (a->bytes && b->bytes && a->offset < b->offset + b->bytes &&
                b->offset < a->offset + a->bytes)
                return sq_error(SQ_FORMAT, "overlapping tensor payload");
        }
    }
    if (total != payload)
        return sq_error(SQ_FORMAT, "unindexed tensor payload");
    return sq_error(SQ_OK, "");
}

static sq_status create(const sq_options *options, sq_safetensors **out) {
    sq_options allocator;
    sq_status status = sq_options_resolve(options, &allocator);
    if (status != SQ_OK)
        return status;
    sq_safetensors *file = allocator.allocate(allocator.context, sizeof *file);
    if (!file)
        return sq_error(SQ_OUT_OF_MEMORY, "allocate tensor file");
    memset(file, 0, sizeof *file);
    file->allocator = allocator;
    memset(file->table, 0xff, sizeof file->table);
    file->tensors = allocator.allocate(allocator.context, SQ_MAX_TENSORS * sizeof *file->tensors);
    if (!file->tensors) {
        allocator.release(allocator.context, file);
        return sq_error(SQ_OUT_OF_MEMORY, "allocate tensor directory");
    }
    *out = file;
    return SQ_OK;
}

void sq_safetensors_close(sq_safetensors *file) {
    if (!file)
        return;
    sq_options allocator = file->allocator;
    if (file->mapped)
        (void)munmap((void *)file->data, file->size);
    allocator.release(allocator.context, file->tensors);
    allocator.release(allocator.context, file);
}

sq_status sq_safetensors_open_memory(const void *data, size_t bytes, const sq_options *options,
                                     sq_safetensors **out) {
    if (out)
        *out = NULL;
    if (!out || !data)
        return sq_error(SQ_INVALID_ARGUMENT, "missing input/output");
    sq_safetensors *file = NULL;
    sq_status status = create(options, &file);
    if (status != SQ_OK)
        return status;
    file->data = data;
    file->size = bytes;
    status = parse(file);
    if (status != SQ_OK) {
        sq_safetensors_close(file);
        return status;
    }
    *out = file;
    return SQ_OK;
}

sq_status sq_safetensors_open(const char *path, const sq_options *options, sq_safetensors **out) {
    if (out)
        *out = NULL;
    if (!out || !path || !path[0])
        return sq_error(SQ_INVALID_ARGUMENT, "missing path/output");
    sq_options checked;
    sq_status status = sq_options_resolve(options, &checked);
    if (status != SQ_OK)
        return status;
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0)
        return sq_error(SQ_IO, "cannot open tensor file");
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 10 ||
        (uintmax_t)st.st_size > SIZE_MAX) {
        (void)close(fd);
        return sq_error(SQ_FORMAT, "invalid tensor file size/type");
    }
    size_t size = (size_t)st.st_size;
    void *mapping = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    (void)close(fd);
    if (mapping == MAP_FAILED)
        return sq_error(SQ_IO, "cannot map tensor file");
    sq_safetensors *file = NULL;
    status = create(&checked, &file);
    if (status != SQ_OK) {
        (void)munmap(mapping, size);
        return status;
    }
    file->data = mapping;
    file->size = size;
    file->mapped = true;
    status = parse(file);
    if (status != SQ_OK) {
        sq_safetensors_close(file);
        return status;
    }
    *out = file;
    return SQ_OK;
}

size_t sq_safetensors_count(const sq_safetensors *file) {
    return file ? file->count : 0;
}

uint64_t sq_safetensors_data_bytes(const sq_safetensors *file) {
    return file ? file->size - file->base : 0;
}

sq_status sq_safetensors_at(const sq_safetensors *file, size_t index, const sq_tensor_info **out) {
    if (out)
        *out = NULL;
    if (!file || !out)
        return sq_error(SQ_INVALID_ARGUMENT, "missing file/output");
    if (index >= file->count)
        return sq_error(SQ_NOT_FOUND, "tensor index not found");
    *out = &file->tensors[index];
    return sq_error(SQ_OK, "");
}

sq_status sq_safetensors_find(const sq_safetensors *file, const char *name, size_t length,
                              const sq_tensor_info **out) {
    if (out)
        *out = NULL;
    if (!file || !out || !name || !length || length > SQ_MAX_NAME || memchr(name, 0, length))
        return sq_error(SQ_INVALID_ARGUMENT, "invalid lookup name/output");
    return sq_safetensors_at(file, lookup(file, name, length), out);
}

sq_status sq_safetensors_data(const sq_safetensors *file, size_t index, const void **out,
                              size_t *bytes) {
    if (out)
        *out = NULL;
    if (bytes)
        *bytes = 0;
    if (!out || !bytes)
        return sq_error(SQ_INVALID_ARGUMENT, "missing data outputs");
    const sq_tensor_info *info;
    sq_status status = sq_safetensors_at(file, index, &info);
    if (status != SQ_OK)
        return status;
    *out = file->data + file->base + (size_t)info->offset;
    *bytes = (size_t)info->bytes;
    return SQ_OK;
}
