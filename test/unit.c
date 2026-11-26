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
            fprintf(stderr, "line %d: %s (%s)\n", __LINE__, #x, ifn_last_error());                 \
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
    CHECK(ifn_size_add(1, 2, &n) && n == 3);
    CHECK(!ifn_size_add(SIZE_MAX, 1, &n));
    CHECK(!ifn_size_mul(SIZE_MAX, 2, &n));
    CHECK(ifn_size_mul(SIZE_MAX, 0, &n) && n == 0);
    CHECK(ifn_size_align(257, 256, &n) && n == 512);
    CHECK(!ifn_size_align(SIZE_MAX, 256, &n));
    CHECK(!ifn_size_align(1, 3, &n));
    CHECK(!ifn_size_align(1, 0, &n));
    CHECK(!ifn_size_add(0, 0, NULL));
    unsigned char bytes[2048];
    n = fixture(bytes, "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}", 4);
    ifn_safetensors *file = NULL;
    CHECK(ifn_safetensors_open_memory(bytes, n, NULL, &file) == IFN_OK);
    CHECK(ifn_safetensors_count(file) == 1 && ifn_safetensors_data_bytes(file) == 4);
    const ifn_tensor_info *info = NULL;
    CHECK(ifn_safetensors_find(file, "a", 1, &info) == IFN_OK);
    CHECK(info->dtype == IFN_BF16 && info->rank == 1 && info->shape[0] == 2);
    const void *data = NULL;
    size_t size = 0;
    CHECK(ifn_safetensors_data(file, 0, &data, &size) == IFN_OK && size == 4);
    CHECK(!memcmp(data, "BBBB", 4));
    CHECK(ifn_safetensors_find(file, "no", 2, &info) == IFN_NOT_FOUND && !info);
    CHECK(ifn_safetensors_at(file, 0, &info) == IFN_OK && !ifn_last_error()[0]);
    CHECK(ifn_safetensors_find(file, "a\0b", 3, &info) == IFN_INVALID_ARGUMENT && !info);
    CHECK(ifn_safetensors_data(file, 99, &data, &size) == IFN_NOT_FOUND && !data && !size);
    ifn_safetensors_close(file);
    ifn_safetensors_close(NULL);
    CHECK(ifn_safetensors_count(NULL) == 0 && ifn_safetensors_data_bytes(NULL) == 0);
    ifn_options options = IFN_OPTIONS_INIT;
    options.api_version++;
    file = (ifn_safetensors *)(uintptr_t)1;
    CHECK(ifn_safetensors_open_memory(bytes, n, &options, &file) == IFN_INVALID_ARGUMENT && !file);
    options = (ifn_options)IFN_OPTIONS_INIT;
    options.struct_size = 0;
    CHECK(ifn_safetensors_open_memory(bytes, n, &options, &file) == IFN_INVALID_ARGUMENT && !file);
    options = (ifn_options)IFN_OPTIONS_INIT;
    options.allocate = allocate;
    CHECK(ifn_safetensors_open_memory(bytes, n, &options, &file) == IFN_INVALID_ARGUMENT && !file);
    CHECK(ifn_safetensors_open_memory(NULL, 0, NULL, &file) == IFN_INVALID_ARGUMENT && !file);
    CHECK(ifn_safetensors_open_memory(bytes, n, NULL, NULL) == IFN_INVALID_ARGUMENT);
    /* Every allocator site, both memory and mapped-file entrypoints. */
    char path[] = "/tmp/ifn-test-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0 && write(fd, bytes, n) == (ssize_t)n && !close(fd));
    for (size_t fail = 0; fail < 3; fail++) {
        for (int mapped = 0; mapped < 2; mapped++) {
            faults f = {0, fail, 0};
            options = (ifn_options)IFN_OPTIONS_INIT;
            options.allocate = allocate;
            options.release = release;
            options.context = &f;
            ifn_status status = mapped ? ifn_safetensors_open(path, &options, &file)
                                       : ifn_safetensors_open_memory(bytes, n, &options, &file);
            CHECK(status == (fail < 2 ? IFN_OUT_OF_MEMORY : IFN_OK));
            if (status == IFN_OK)
                CHECK(file && f.live == 2);
            else
                CHECK(!file && !f.live);
            ifn_safetensors_close(file);
            CHECK(!f.live);
        }
    }
    CHECK(!unlink(path));
    CHECK(ifn_safetensors_open(path, NULL, &file) == IFN_IO && !file);
    /* Failed parse releases both owned allocations. */
    faults f = {0, SIZE_MAX, 0};
    options = (ifn_options)IFN_OPTIONS_INIT;
    options.allocate = allocate;
    options.release = release;
    options.context = &f;
    bytes[8] = '[';
    CHECK(ifn_safetensors_open_memory(bytes, n, &options, &file) == IFN_FORMAT && !file && !f.live);
    CHECK(ifn_status_string(999) && ifn_version());
    puts("host API, checked arithmetic, ownership and allocation failure tests passed");
    return 0;
}
