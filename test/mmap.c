/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Sparse synthetic input proves metadata-only heap use; never touches models. */
#define _POSIX_C_SOURCE 200809L
#include "ifn.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "mmap:%d: %s\n", __LINE__, #x);                                        \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)
static size_t live, peak;

static void *small_allocate(void *context, size_t bytes) {
    (void)context;
    if (bytes > 2 * 1024 * 1024 - live)
        return NULL;
    size_t *p = malloc(bytes + sizeof(max_align_t));
    if (!p)
        return NULL;
    *p = bytes;
    live += bytes;
    if (live > peak)
        peak = live;
    return (unsigned char *)p + sizeof(max_align_t);
}

static void small_release(void *context, void *ptr) {
    (void)context;
    if (!ptr)
        return;
    size_t *p = (size_t *)((unsigned char *)ptr - sizeof(max_align_t));
    live -= *p;
    free(p);
}
#ifdef __linux__
static bool readonly_mapping(const void *data, uint64_t inode) {
    FILE *file = fopen("/proc/self/maps", "r");
    if (!file)
        return false;
    char line[1024], permissions[5];
    unsigned long begin, end;
    unsigned long long found_inode;
    bool found = false;
    while (fgets(line, sizeof line, file)) {
        if (sscanf(line, "%lx-%lx %4s %*s %*s %llu", &begin, &end, permissions, &found_inode) ==
                4 &&
            (uintptr_t)data >= begin && (uintptr_t)data < end && found_inode == inode &&
            !strcmp(permissions, "r--p"))
            found = true;
    }
    fclose(file);
    return found;
}
#endif
int main(void) {
    int failed = 0;
    char path[] = "/tmp/infer-mmap-XXXXXX";
    int fd = mkstemp(path);
    ifn_safetensors *file = NULL;
    CHECK(fd >= 0);
    const size_t payload = 64u * 1024u * 1024u;
    char json[192];
    int n = snprintf(json, sizeof json,
                     "{\"x\":{\"dtype\":\"U8\",\"shape\":[%zu],\"data_offsets\":[0,%zu]}}", payload,
                     payload);
    CHECK(n > 0 && (size_t)n < sizeof json);
    unsigned char length[8];
    for (unsigned i = 0; i < 8; i++)
        length[i] = (unsigned char)((uint64_t)n >> (8 * i));
    CHECK(write(fd, length, 8) == 8 && write(fd, json, (size_t)n) == n);
    off_t first = 8 + n, last = first + (off_t)payload - 1;
    CHECK(ftruncate(fd, last + 1) == 0);
    const unsigned char start = 42, finish = 37;
    CHECK(pwrite(fd, &start, 1, first) == 1 && pwrite(fd, &finish, 1, last) == 1);
    struct stat st;
    CHECK(!fstat(fd, &st) && st.st_blocks * 512 < 1024 * 1024);
    ifn_options options = IFN_OPTIONS_INIT;
    options.allocate = small_allocate;
    options.release = small_release;
    CHECK(ifn_safetensors_open(path, &options, &file) == IFN_OK);
    const void *data = NULL;
    size_t bytes = 0;
    CHECK(ifn_safetensors_data(file, 0, &data, &bytes) == IFN_OK && bytes == payload);
    CHECK(((const unsigned char *)data)[0] == 42 &&
          ((const unsigned char *)data)[payload - 1] == 37);
#ifdef __linux__
    CHECK(readonly_mapping(data, (uint64_t)st.st_ino));
#endif
    CHECK(unlink(path) == 0);
    CHECK(((const unsigned char *)data)[0] == 42 && peak < 2 * 1024 * 1024);
done:
    ifn_safetensors_close(file);
    if (fd >= 0)
        close(fd);
    (void)unlink(path);
    if (live)
        failed = 1;
    if (!failed)
        printf("read-only mapped 64 MiB payload; peak injected heap=%zu bytes; cleanup passed\n",
               peak);
    return failed;
}
