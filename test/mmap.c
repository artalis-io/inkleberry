/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Sparse synthetic input proves metadata-only heap use; never touches models. */
#define _POSIX_C_SOURCE 200809L
#include "allocation_alignment.h"
#include "ifn.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "filesystem.h"
#ifdef _WIN32
#include <winioctl.h>
#endif
#define IFN_MMAP_LOCAL_CHECK(x)                                                                    \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "mmap:%d: %s\n", __LINE__, #x);                                        \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)
static size_t ifn_mmap_local_live, ifn_mmap_local_peak;

static void *ifn_mmap_local_small_allocate(void *context, size_t bytes) {
    (void)context;
    if (bytes > 2 * 1024 * 1024 - ifn_mmap_local_live)
        return NULL;
    size_t *p = malloc(bytes + sizeof(ifn_allocation_alignment));
    if (!p)
        return NULL;
    *p = bytes;
    ifn_mmap_local_live += bytes;
    if (ifn_mmap_local_live > ifn_mmap_local_peak)
        ifn_mmap_local_peak = ifn_mmap_local_live;
    return (unsigned char *)p + sizeof(ifn_allocation_alignment);
}

static void ifn_mmap_local_small_release(void *context, void *ptr) {
    (void)context;
    if (!ptr)
        return;
    size_t *p = (size_t *)((unsigned char *)ptr - sizeof(ifn_allocation_alignment));
    ifn_mmap_local_live -= *p;
    free(p);
}
#ifdef __linux__
static bool ifn_mmap_local_readonly_mapping(const void *data, uint64_t inode) {
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
int main(int argc, char **argv) {
    int failed = 0;
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--large")))
        return 2;
    const size_t payload = argc == 2 ? (size_t)8 * 1024 * 1024 * 1024 : 64u * 1024u * 1024u;

    char path[512] = {0};
    int fd = ifn_test_temp(path, sizeof path);
    ifn_safetensors *file = NULL;
    IFN_MMAP_LOCAL_CHECK(fd >= 0);
    char json[192];
    int n = snprintf(json, sizeof json,
                     "{\"x\":{\"dtype\":\"U8\",\"shape\":[%zu],\"data_offsets\":[0,%zu]}}", payload,
                     payload);
    IFN_MMAP_LOCAL_CHECK(n > 0 && (size_t)n < sizeof json);
    unsigned char length[8];
    for (unsigned i = 0; i < 8; i++)
        length[i] = (unsigned char)((uint64_t)n >> (8 * i));
    IFN_MMAP_LOCAL_CHECK(ifn_test_write(fd, length, 8) == 8 &&
                         ifn_test_write(fd, json, (size_t)n) == n);
#ifdef _WIN32
    __int64 first = 8 + n, last = first + (__int64)payload - 1;
    DWORD returned = 0;
    HANDLE handle = (HANDLE)_get_osfhandle(fd);
    IFN_MMAP_LOCAL_CHECK(
        DeviceIoControl(handle, FSCTL_SET_SPARSE, NULL, 0, NULL, 0, &returned, NULL));
    LARGE_INTEGER end;
    end.QuadPart = last + 1;
    IFN_MMAP_LOCAL_CHECK(SetFilePointerEx(handle, end, NULL, FILE_BEGIN) && SetEndOfFile(handle));
    const unsigned char start = 42, finish = 37;
    IFN_MMAP_LOCAL_CHECK(_lseeki64(fd, first, SEEK_SET) == first && _write(fd, &start, 1) == 1);
    IFN_MMAP_LOCAL_CHECK(_lseeki64(fd, last, SEEK_SET) == last && _write(fd, &finish, 1) == 1);
    FILE_STANDARD_INFO storage;
    IFN_MMAP_LOCAL_CHECK(
        GetFileInformationByHandleEx(handle, FileStandardInfo, &storage, sizeof storage));
    IFN_MMAP_LOCAL_CHECK(storage.AllocationSize.QuadPart < 1024 * 1024);
    IFN_MMAP_LOCAL_CHECK(!ifn_test_close(fd));
    fd = -1;
#else
    off_t first = 8 + n, last = first + (off_t)payload - 1;
    IFN_MMAP_LOCAL_CHECK(ftruncate(fd, last + 1) == 0);
    const unsigned char start = 42, finish = 37;
    IFN_MMAP_LOCAL_CHECK(pwrite(fd, &start, 1, first) == 1 && pwrite(fd, &finish, 1, last) == 1);
    struct stat st;
    IFN_MMAP_LOCAL_CHECK(!fstat(fd, &st) && st.st_blocks * 512 < 1024 * 1024);
#endif
    ifn_options options = IFN_OPTIONS_INIT;
    options.allocate = ifn_mmap_local_small_allocate;
    options.release = ifn_mmap_local_small_release;
    IFN_MMAP_LOCAL_CHECK(ifn_safetensors_open(path, &options, &file) == IFN_OK);
    const void *data = NULL;
    size_t bytes = 0;
    IFN_MMAP_LOCAL_CHECK(ifn_safetensors_data(file, 0, &data, &bytes) == IFN_OK &&
                         bytes == payload);
    IFN_MMAP_LOCAL_CHECK(((const unsigned char *)data)[0] == 42 &&
                         ((const unsigned char *)data)[payload - 1] == 37);
#ifdef __linux__
    IFN_MMAP_LOCAL_CHECK(ifn_mmap_local_readonly_mapping(data, (uint64_t)st.st_ino));
#endif
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION region;
    IFN_MMAP_LOCAL_CHECK(VirtualQuery(data, &region, sizeof region) == sizeof region &&
                         region.Protect == PAGE_READONLY && region.Type == MEM_MAPPED);
    /* CRT/Windows versions differ in delete-pending support. Either outcome
     * must leave the existing view usable until explicit unmap. */
    (void)ifn_test_remove(path);
#else
    IFN_MMAP_LOCAL_CHECK(ifn_test_remove(path) == 0);
#endif
    IFN_MMAP_LOCAL_CHECK(((const unsigned char *)data)[0] == 42 &&
                         ifn_mmap_local_peak < 2 * 1024 * 1024);
done:
    ifn_safetensors_close(file);
    if (fd >= 0)
        ifn_test_close(fd);
    (void)ifn_test_remove(path);
    if (ifn_mmap_local_live)
        failed = 1;
    if (!failed)
        printf("read-only mapped %zu MiB payload; peak injected heap=%zu bytes; cleanup passed\n",
               payload / (1024 * 1024), ifn_mmap_local_peak);
    return failed;
}
