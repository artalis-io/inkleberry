/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * C translation/rework of src/safetensors.zig at d10c0abb2c8990be81c08ba84c66a8333a691b33.
 * Changes: recoverable validation, dtypes/shapes, strict JSON and range checks,
 * injected allocation and memory-input support; no user-input assertions. */
#define _POSIX_C_SOURCE 200809L
#include "internal.h"
#include "json.h"
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

static bool tensor(sq_json *j, sq_tensor_info *info) {
    if (!sq_json_take(j, '{'))
        return false;
    unsigned fields = 0;
    do {
        char key[32];
        if (!sq_json_string(j, key, sizeof key) || !sq_json_take(j, ':'))
            return false;
        unsigned bit;
        if (!strcmp(key, "dtype")) {
            char type[32];
            bit = 1;
            if (!sq_json_string(j, type, sizeof type))
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
            if (!sq_json_take(j, '['))
                return false;
            if (!sq_json_take(j, ']')) {
                do {
                    if (info->rank == SQ_MAX_RANK || !sq_json_u64(j, &info->shape[info->rank]))
                        return false;
                    info->rank++;
                } while (sq_json_take(j, ','));
                if (!sq_json_take(j, ']'))
                    return false;
            }
        } else if (!strcmp(key, "data_offsets")) {
            uint64_t end;
            bit = 4;
            if (!sq_json_take(j, '[') || !sq_json_u64(j, &info->offset) || !sq_json_take(j, ',') ||
                !sq_json_u64(j, &end) || !sq_json_take(j, ']') || end < info->offset)
                return false;
            info->bytes = end - info->offset;
        } else
            return false;
        if (fields & bit)
            return false;
        fields |= bit;
    } while (sq_json_take(j, ','));
    return sq_json_take(j, '}') && fields == 7;
}

static bool metadata(sq_json *j) {
    if (!sq_json_take(j, '{'))
        return false;
    if (sq_json_take(j, '}'))
        return true;
    do {
        if (!sq_json_string(j, NULL, 0) || !sq_json_take(j, ':') || !sq_json_string(j, NULL, 0))
            return false;
    } while (sq_json_take(j, ','));
    return sq_json_take(j, '}');
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
    sq_json j = {file->data + 8, (size_t)header, 0};
    if (!sq_json_take(&j, '{'))
        return sq_error(SQ_FORMAT, "expected tensor directory");
    bool seen_metadata = false;
    if (!sq_json_take(&j, '}')) {
        do {
            char name[SQ_MAX_NAME + 1];
            if (!sq_json_string(&j, name, sizeof name) || !name[0] || !sq_json_take(&j, ':'))
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
        } while (sq_json_take(&j, ','));
        if (!sq_json_take(&j, '}'))
            return sq_error(SQ_FORMAT, "unterminated tensor directory");
    }
    sq_json_ws(&j);
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

sq_status sq_safetensors_named_data(const sq_safetensors *file, const char *name, const void **out,
                                    size_t *bytes) {
    const sq_tensor_info *info = NULL;
    sq_status status = sq_safetensors_find(file, name, strlen(name), &info);
    if (status != SQ_OK) {
        if (out)
            *out = NULL;
        if (bytes)
            *bytes = 0;
        return status;
    }
    return sq_safetensors_data(file, (size_t)(info - file->tensors), out, bytes);
}
