/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * OS-local services follow OTTO's narrow read-only mapping/clock pattern. */
#define _POSIX_C_SOURCE 200809L
#ifndef _WIN32
#include "file_map.h"
#include "host_clock.h"
#include "internal.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

ifn_status ifn_file_map_open(const char *path, size_t minimum, size_t maximum,
                             const ifn_options *allocator, ifn_file_map *out) {
    (void)allocator;
    if (out)
        *out = (ifn_file_map){0};
    if (!out || !path || !path[0] || !minimum || maximum < minimum)
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid file mapping request");
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0)
        return ifn_os_error(IFN_IO, errno, "cannot open mapped file");
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        (uintmax_t)st.st_size < minimum || (uintmax_t)st.st_size > maximum) {
        (void)close(fd);
        return ifn_error(IFN_FORMAT, "invalid mapped file size/type");
    }
    size_t size = (size_t)st.st_size;
    void *data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    int error = errno;
    (void)close(fd);
    if (data == MAP_FAILED)
        return ifn_os_error(IFN_IO, error, "cannot map file");
    out->data = data;
    out->size = size;
    return IFN_OK;
}

void ifn_file_map_close(ifn_file_map *map) {
    if (map && map->data) {
        (void)munmap((void *)map->data, map->size);
        *map = (ifn_file_map){0};
    }
}

uint64_t ifn_host_now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) || t.tv_sec < 0 || t.tv_nsec < 0 ||
        t.tv_nsec >= 1000000000L || (uint64_t)t.tv_sec > UINT64_MAX / UINT64_C(1000000000))
        return 0;
    uint64_t seconds = (uint64_t)t.tv_sec * UINT64_C(1000000000);
    return UINT64_MAX - seconds < (uint64_t)t.tv_nsec ? 0 : seconds + (uint64_t)t.tv_nsec;
}
#else
typedef int ifn_host_posix_inactive;
#endif
