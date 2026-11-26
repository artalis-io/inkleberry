/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * C translation/rework of src/safetensors.zig at d10c0abb2c8990be81c08ba84c66a8333a691b33.
 * Changes: recoverable validation, dtypes/shapes, strict JSON and range checks,
 * injected allocation and memory-input support; no user-input assertions. */
#define _POSIX_C_SOURCE 200809L
#include "internal.h"
#include "json.h"
#include "file_map.h"
#include <string.h>

struct ifn_safetensors {
    ifn_options allocator;
    const unsigned char *data;
    size_t size, base, count;
    bool mapped;
    ifn_tensor_info *tensors;
    uint16_t table[4096];
};

static uint64_t ifn_safetensors_local_hash_name(const char *name, size_t length) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < length; i++)
        hash = (hash ^ (unsigned char)name[i]) * UINT64_C(1099511628211);
    return hash;
}

static size_t ifn_safetensors_local_lookup(const ifn_safetensors *file, const char *name,
                                           size_t length) {
    size_t slot = (size_t)(ifn_safetensors_local_hash_name(name, length) & 4095u);
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

static bool ifn_safetensors_local_tensor(ifn_json *j, ifn_tensor_info *info) {
    if (!ifn_json_take(j, '{'))
        return false;
    unsigned fields = 0;
    do {
        char key[32];
        if (!ifn_json_string(j, key, sizeof key) || !ifn_json_take(j, ':'))
            return false;
        unsigned bit;
        if (!strcmp(key, "dtype")) {
            char type[32];
            bit = 1;
            if (!ifn_json_string(j, type, sizeof type))
                return false;
            if (!strcmp(type, "U8"))
                info->dtype = IFN_U8;
            else if (!strcmp(type, "BF16"))
                info->dtype = IFN_BF16;
            else if (!strcmp(type, "F32"))
                info->dtype = IFN_F32;
            else if (!strcmp(type, "F8_E4M3"))
                info->dtype = IFN_F8_E4M3;
            else
                return false;
        } else if (!strcmp(key, "shape")) {
            bit = 2;
            if (!ifn_json_take(j, '['))
                return false;
            if (!ifn_json_take(j, ']')) {
                do {
                    if (info->rank == IFN_MAX_RANK || !ifn_json_u64(j, &info->shape[info->rank]))
                        return false;
                    info->rank++;
                } while (ifn_json_take(j, ','));
                if (!ifn_json_take(j, ']'))
                    return false;
            }
        } else if (!strcmp(key, "data_offsets")) {
            uint64_t end;
            bit = 4;
            if (!ifn_json_take(j, '[') || !ifn_json_u64(j, &info->offset) ||
                !ifn_json_take(j, ',') || !ifn_json_u64(j, &end) || !ifn_json_take(j, ']') ||
                end < info->offset)
                return false;
            info->bytes = end - info->offset;
        } else
            return false;
        if (fields & bit)
            return false;
        fields |= bit;
    } while (ifn_json_take(j, ','));
    return ifn_json_take(j, '}') && fields == 7;
}

static bool ifn_safetensors_local_metadata(ifn_json *j) {
    if (!ifn_json_take(j, '{'))
        return false;
    if (ifn_json_take(j, '}'))
        return true;
    do {
        if (!ifn_json_string(j, NULL, 0) || !ifn_json_take(j, ':') || !ifn_json_string(j, NULL, 0))
            return false;
    } while (ifn_json_take(j, ','));
    return ifn_json_take(j, '}');
}

static ifn_status ifn_safetensors_local_parse(ifn_safetensors *file) {
    if (file->size < 10)
        return ifn_error(IFN_FORMAT, "truncated safetensors file");
    uint64_t header = 0;
    for (unsigned i = 0; i < 8; i++)
        header |= (uint64_t)file->data[i] << (8 * i);
    if (header < 2 || header > IFN_MAX_HEADER || header > file->size - 8)
        return ifn_error(IFN_FORMAT, "invalid safetensors header length");
    file->base = 8 + (size_t)header;
    ifn_json j = {file->data + 8, (size_t)header, 0};
    if (!ifn_json_take(&j, '{'))
        return ifn_error(IFN_FORMAT, "expected tensor directory");
    bool seen_metadata = false;
    if (!ifn_json_take(&j, '}')) {
        do {
            char name[IFN_MAX_NAME + 1];
            if (!ifn_json_string(&j, name, sizeof name) || !name[0] || !ifn_json_take(&j, ':'))
                return ifn_error(IFN_FORMAT, "invalid tensor name/key");
            if (!strcmp(name, "__metadata__")) {
                if (seen_metadata || !ifn_safetensors_local_metadata(&j))
                    return ifn_error(IFN_FORMAT, "invalid metadata");
                seen_metadata = true;
                continue;
            }
            if (file->count == IFN_MAX_TENSORS ||
                ifn_safetensors_local_lookup(file, name, strlen(name)) != SIZE_MAX)
                return ifn_error(IFN_FORMAT, "duplicate tensor or tensor limit exceeded");
            ifn_tensor_info *info = &file->tensors[file->count];
            memset(info, 0, sizeof *info);
            memcpy(info->name, name, strlen(name) + 1);
            if (!ifn_safetensors_local_tensor(&j, info))
                return ifn_error(IFN_FORMAT, "invalid tensor schema/dtype/shape");
            size_t elements = 1;
            for (uint32_t i = 0; i < info->rank; i++) {
                if (info->shape[i] > SIZE_MAX ||
                    !ifn_size_mul(elements, (size_t)info->shape[i], &elements))
                    return ifn_error(IFN_FORMAT, "tensor shape overflow");
            }
            size_t expected;
            size_t item = info->dtype == IFN_F32 ? 4 : info->dtype == IFN_BF16 ? 2 : 1;
            if (!ifn_size_mul(elements, item, &expected) || expected != info->bytes ||
                info->offset > file->size - file->base ||
                info->bytes > file->size - file->base - info->offset)
                return ifn_error(IFN_FORMAT, "tensor size/offset mismatch");
            size_t slot = (size_t)(ifn_safetensors_local_hash_name(name, strlen(name)) & 4095u);
            while (file->table[slot] != UINT16_MAX)
                slot = (slot + 1) & 4095u;
            file->table[slot] = (uint16_t)file->count++;
        } while (ifn_json_take(&j, ','));
        if (!ifn_json_take(&j, '}'))
            return ifn_error(IFN_FORMAT, "unterminated tensor directory");
    }
    ifn_json_ws(&j);
    if (j.pos != j.size)
        return ifn_error(IFN_FORMAT, "trailing header data");
    uint64_t total = 0, payload = file->size - file->base;
    for (size_t i = 0; i < file->count; i++) {
        const ifn_tensor_info *a = &file->tensors[i];
        if (a->bytes > payload - total)
            return ifn_error(IFN_FORMAT, "overlapping tensor payload");
        total += a->bytes;
        for (size_t k = 0; k < i; k++) {
            const ifn_tensor_info *b = &file->tensors[k];
            if (a->bytes && b->bytes && a->offset < b->offset + b->bytes &&
                b->offset < a->offset + a->bytes)
                return ifn_error(IFN_FORMAT, "overlapping tensor payload");
        }
    }
    if (total != payload)
        return ifn_error(IFN_FORMAT, "unindexed tensor payload");
    return ifn_error(IFN_OK, "");
}

static ifn_status ifn_safetensors_local_create(const ifn_options *options, ifn_safetensors **out) {
    ifn_options allocator;
    ifn_status status = ifn_options_resolve(options, &allocator);
    if (status != IFN_OK)
        return status;
    ifn_safetensors *file = allocator.allocate(allocator.context, sizeof *file);
    if (!file)
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate tensor file");
    memset(file, 0, sizeof *file);
    file->allocator = allocator;
    memset(file->table, 0xff, sizeof file->table);
    file->tensors = allocator.allocate(allocator.context, IFN_MAX_TENSORS * sizeof *file->tensors);
    if (!file->tensors) {
        allocator.release(allocator.context, file);
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate tensor directory");
    }
    *out = file;
    return IFN_OK;
}

void ifn_safetensors_close(ifn_safetensors *file) {
    if (!file)
        return;
    ifn_options allocator = file->allocator;
    if (file->mapped)
        ifn_file_map_close(&(ifn_file_map){file->data, file->size});
    allocator.release(allocator.context, file->tensors);
    allocator.release(allocator.context, file);
}

ifn_status ifn_safetensors_open_memory(const void *data, size_t bytes, const ifn_options *options,
                                       ifn_safetensors **out) {
    if (out)
        *out = NULL;
    if (!out || !data)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing input/output");
    ifn_safetensors *file = NULL;
    ifn_status status = ifn_safetensors_local_create(options, &file);
    if (status != IFN_OK)
        return status;
    file->data = data;
    file->size = bytes;
    status = ifn_safetensors_local_parse(file);
    if (status != IFN_OK) {
        ifn_safetensors_close(file);
        return status;
    }
    *out = file;
    return IFN_OK;
}

ifn_status ifn_safetensors_open(const char *path, const ifn_options *options,
                                ifn_safetensors **out) {
    if (out)
        *out = NULL;
    if (!out || !path || !path[0])
        return ifn_error(IFN_INVALID_ARGUMENT, "missing path/output");
    ifn_options checked;
    ifn_status status = ifn_options_resolve(options, &checked);
    if (status != IFN_OK)
        return status;
    ifn_file_map map;
    status = ifn_file_map_open(path, 10, SIZE_MAX, &checked, &map);
    if (status != IFN_OK)
        return status;
    ifn_safetensors *file = NULL;
    status = ifn_safetensors_local_create(&checked, &file);
    if (status != IFN_OK) {
        ifn_file_map_close(&map);
        return status;
    }
    file->data = map.data;
    file->size = map.size;
    file->mapped = true;
    status = ifn_safetensors_local_parse(file);
    if (status != IFN_OK) {
        ifn_safetensors_close(file);
        return status;
    }
    *out = file;
    return IFN_OK;
}

size_t ifn_safetensors_count(const ifn_safetensors *file) {
    return file ? file->count : 0;
}

uint64_t ifn_safetensors_data_bytes(const ifn_safetensors *file) {
    return file ? file->size - file->base : 0;
}

ifn_status ifn_safetensors_at(const ifn_safetensors *file, size_t index,
                              const ifn_tensor_info **out) {
    if (out)
        *out = NULL;
    if (!file || !out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing file/output");
    if (index >= file->count)
        return ifn_error(IFN_NOT_FOUND, "tensor index not found");
    *out = &file->tensors[index];
    return ifn_error(IFN_OK, "");
}

ifn_status ifn_safetensors_find(const ifn_safetensors *file, const char *name, size_t length,
                                const ifn_tensor_info **out) {
    if (out)
        *out = NULL;
    if (!file || !out || !name || !length || length > IFN_MAX_NAME || memchr(name, 0, length))
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid lookup name/output");
    return ifn_safetensors_at(file, ifn_safetensors_local_lookup(file, name, length), out);
}

ifn_status ifn_safetensors_data(const ifn_safetensors *file, size_t index, const void **out,
                                size_t *bytes) {
    if (out)
        *out = NULL;
    if (bytes)
        *bytes = 0;
    if (!out || !bytes)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing data outputs");
    const ifn_tensor_info *info;
    ifn_status status = ifn_safetensors_at(file, index, &info);
    if (status != IFN_OK)
        return status;
    *out = file->data + file->base + (size_t)info->offset;
    *bytes = (size_t)info->bytes;
    return IFN_OK;
}

ifn_status ifn_safetensors_named_data(const ifn_safetensors *file, const char *name,
                                      const void **out, size_t *bytes) {
    const ifn_tensor_info *info = NULL;
    ifn_status status = ifn_safetensors_find(file, name, strlen(name), &info);
    if (status != IFN_OK) {
        if (out)
            *out = NULL;
        if (bytes)
            *bytes = 0;
        return status;
    }
    return ifn_safetensors_data(file, (size_t)(info - file->tensors), out, bytes);
}

ifn_status ifn_safetensors_sha256(const ifn_safetensors *file, unsigned char out[32]) {
    if (!file || !out)
        return ifn_error(IFN_INVALID_ARGUMENT, "missing checkpoint hash arguments");
    return ifn_sha256_hash(file->data, file->size, out)
               ? IFN_OK
               : ifn_error(IFN_LIMIT, "checkpoint hash length overflow");
}
