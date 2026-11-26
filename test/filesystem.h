/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_TEST_FILESYSTEM_H
#define IFN_TEST_FILESYSTEM_H
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <limits.h>

static inline int ifn_test_write(int fd, const void *data, size_t bytes) {
    return bytes > INT_MAX ? -1 : _write(fd, data, (unsigned)bytes);
}

static inline int ifn_test_close(int fd) {
    return _close(fd);
}

static inline int ifn_test_remove(const char *path) {
    return remove(path);
}

static inline int ifn_test_temp(char *path, size_t capacity) {
    char directory[MAX_PATH];
    if (capacity < MAX_PATH || !GetTempPathA(MAX_PATH, directory) ||
        !GetTempFileNameA(directory, "ifn", 0, path))
        return -1;
    return _open(path, _O_RDWR | _O_BINARY);
}

#else
#include <unistd.h>

static inline int ifn_test_write(int fd, const void *data, size_t bytes) {
    return (int)write(fd, data, bytes);
}

static inline int ifn_test_close(int fd) {
    return close(fd);
}

static inline int ifn_test_remove(const char *path) {
    return unlink(path);
}

static inline int ifn_test_temp(char *path, size_t capacity) {
    const char pattern[] = "/tmp/ifn-test-XXXXXX";
    if (capacity < sizeof pattern)
        return -1;
    memcpy(path, pattern, sizeof pattern);
    return mkstemp(path);
}
#endif
#endif
