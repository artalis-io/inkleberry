/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#define _POSIX_C_SOURCE 200809L
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "line %d: %s (%s)\n", __LINE__, #x, sq_last_error());                  \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

typedef struct {
    size_t calls, fail_at, live;
} faults;

static void *allocate(void *context, size_t bytes) {
    faults *f = context;
    if (f->calls++ == f->fail_at)
        return NULL;
    void *p = malloc(bytes);
    if (p)
        f->live++;
    return p;
}

static void release(void *context, void *p) {
    faults *f = context;
    if (p)
        f->live--;
    free(p);
}

static size_t fixture(unsigned char *out, const char *header, size_t payload) {
    size_t n = strlen(header);
    for (unsigned i = 0; i < 8; i++)
        out[i] = (unsigned char)((uint64_t)n >> (8 * i));
    memcpy(out + 8, header, n);
    memset(out + 8 + n, 0x42, payload);
    return 8 + n + payload;
}

int main(void) {
    size_t n;
    CHECK(sq_size_add(1, 2, &n) && n == 3);
    CHECK(!sq_size_add(SIZE_MAX, 1, &n));
    CHECK(!sq_size_mul(SIZE_MAX, 2, &n));
    CHECK(sq_size_mul(SIZE_MAX, 0, &n) && n == 0);
    CHECK(sq_size_align(257, 256, &n) && n == 512);
    CHECK(!sq_size_align(SIZE_MAX, 256, &n));
    CHECK(!sq_size_align(1, 3, &n));
    CHECK(!sq_size_align(1, 0, &n));
    CHECK(!sq_size_add(0, 0, NULL));
    unsigned char bytes[2048];
    n = fixture(bytes, "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}", 4);
    sq_safetensors *file = NULL;
    CHECK(sq_safetensors_open_memory(bytes, n, NULL, &file) == SQ_OK);
    CHECK(sq_safetensors_count(file) == 1 && sq_safetensors_data_bytes(file) == 4);
    const sq_tensor_info *info = NULL;
    CHECK(sq_safetensors_find(file, "a", 1, &info) == SQ_OK);
    CHECK(info->dtype == SQ_BF16 && info->rank == 1 && info->shape[0] == 2);
    const void *data = NULL;
    size_t size = 0;
    CHECK(sq_safetensors_data(file, 0, &data, &size) == SQ_OK && size == 4);
    CHECK(!memcmp(data, "BBBB", 4));
    CHECK(sq_safetensors_find(file, "no", 2, &info) == SQ_NOT_FOUND && !info);
    CHECK(sq_safetensors_at(file, 0, &info) == SQ_OK && !sq_last_error()[0]);
    CHECK(sq_safetensors_find(file, "a\0b", 3, &info) == SQ_INVALID_ARGUMENT && !info);
    CHECK(sq_safetensors_data(file, 99, &data, &size) == SQ_NOT_FOUND && !data && !size);
    sq_safetensors_close(file);
    sq_safetensors_close(NULL);
    CHECK(sq_safetensors_count(NULL) == 0 && sq_safetensors_data_bytes(NULL) == 0);
    sq_options options = SQ_OPTIONS_INIT;
    options.api_version++;
    file = (sq_safetensors *)(uintptr_t)1;
    CHECK(sq_safetensors_open_memory(bytes, n, &options, &file) == SQ_INVALID_ARGUMENT && !file);
    options = (sq_options)SQ_OPTIONS_INIT;
    options.struct_size = 0;
    CHECK(sq_safetensors_open_memory(bytes, n, &options, &file) == SQ_INVALID_ARGUMENT && !file);
    options = (sq_options)SQ_OPTIONS_INIT;
    options.allocate = allocate;
    CHECK(sq_safetensors_open_memory(bytes, n, &options, &file) == SQ_INVALID_ARGUMENT && !file);
    CHECK(sq_safetensors_open_memory(NULL, 0, NULL, &file) == SQ_INVALID_ARGUMENT && !file);
    CHECK(sq_safetensors_open_memory(bytes, n, NULL, NULL) == SQ_INVALID_ARGUMENT);
    /* Every allocator site, both memory and mapped-file entrypoints. */
    char path[] = "/tmp/sq-test-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0 && write(fd, bytes, n) == (ssize_t)n && !close(fd));
    for (size_t fail = 0; fail < 3; fail++) {
        for (int mapped = 0; mapped < 2; mapped++) {
            faults f = {0, fail, 0};
            options = (sq_options)SQ_OPTIONS_INIT;
            options.allocate = allocate;
            options.release = release;
            options.context = &f;
            sq_status status = mapped ? sq_safetensors_open(path, &options, &file)
                                      : sq_safetensors_open_memory(bytes, n, &options, &file);
            CHECK(status == (fail < 2 ? SQ_OUT_OF_MEMORY : SQ_OK));
            if (status == SQ_OK)
                CHECK(file && f.live == 2);
            else
                CHECK(!file && !f.live);
            sq_safetensors_close(file);
            CHECK(!f.live);
        }
    }
    CHECK(!unlink(path));
    CHECK(sq_safetensors_open(path, NULL, &file) == SQ_IO && !file);
    /* Failed parse releases both owned allocations. */
    faults f = {0, SIZE_MAX, 0};
    options = (sq_options)SQ_OPTIONS_INIT;
    options.allocate = allocate;
    options.release = release;
    options.context = &f;
    bytes[8] = '[';
    CHECK(sq_safetensors_open_memory(bytes, n, &options, &file) == SQ_FORMAT && !file && !f.live);
    CHECK(sq_status_string(999) && sq_version());
    puts("host API, checked arithmetic, ownership and allocation failure tests passed");
    return 0;
}
