/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Mapping lifetime/clock shape adapted from OTTO; strict UTF-8, bounded files,
 * injected temporary allocation and native error preservation are IFN policy. */
#ifdef _WIN32
#include "file_map.h"
#include "host_clock.h"
#include "internal.h"
#include <windows.h>
#include <string.h>

ifn_status ifn_file_map_open(const char *path, size_t minimum, size_t maximum,
                             const ifn_options *allocator, ifn_file_map *out) {
    if (out)
        *out = (ifn_file_map){0};
    if (!out || !path || !path[0] || !minimum || maximum < minimum || !allocator ||
        !allocator->allocate || !allocator->release)
        return ifn_error(IFN_INVALID_ARGUMENT, "invalid file mapping request");
    /* A file API must not open a named pipe/device and wait for its peer. */
    if (!strncmp(path, "\\\\.\\", 4) || !strncmp(path, "//./", 4))
        return ifn_error(IFN_FORMAT, "device namespaces are not mapped files");
    int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
    if (!count)
        return ifn_win32_error(IFN_INVALID_ARGUMENT, GetLastError(), "invalid UTF-8 file path");
    size_t bytes;
    if (!ifn_size_mul((size_t)count, sizeof(wchar_t), &bytes))
        return ifn_error(IFN_LIMIT, "file path is too large");
    wchar_t *wide = allocator->allocate(allocator->context, bytes);
    if (!wide)
        return ifn_error(IFN_OUT_OF_MEMORY, "allocate file path");
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide, count)) {
        DWORD error = GetLastError();
        allocator->release(allocator->context, wide);
        return ifn_win32_error(IFN_INVALID_ARGUMENT, error, "invalid UTF-8 file path");
    }
    HANDLE file = CreateFileW(wide, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    DWORD error = GetLastError();
    allocator->release(allocator->context, wide);
    if (file == INVALID_HANDLE_VALUE)
        return ifn_win32_error(IFN_IO, error, "cannot open mapped file");
    LARGE_INTEGER size;
    if (GetFileType(file) != FILE_TYPE_DISK || !GetFileSizeEx(file, &size) || size.QuadPart < 0 ||
        (uint64_t)size.QuadPart < minimum || (uint64_t)size.QuadPart > maximum) {
        (void)CloseHandle(file);
        return ifn_error(IFN_FORMAT, "invalid mapped file size/type");
    }
    HANDLE mapping = CreateFileMappingW(file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping) {
        error = GetLastError();
        (void)CloseHandle(file);
        return ifn_win32_error(IFN_IO, error, "cannot create file mapping");
    }
    void *data = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    error = GetLastError();
    (void)CloseHandle(mapping);
    (void)CloseHandle(file);
    if (!data)
        return ifn_win32_error(IFN_IO, error, "cannot map file");
    out->data = data;
    out->size = (size_t)size.QuadPart;
    return IFN_OK;
}

void ifn_file_map_close(ifn_file_map *map) {
    if (map && map->data) {
        (void)UnmapViewOfFile(map->data);
        *map = (ifn_file_map){0};
    }
}

uint64_t ifn_host_now_ns(void) {
    LARGE_INTEGER ticks, frequency;
    uint64_t ns;
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0 ||
        !QueryPerformanceCounter(&ticks) || ticks.QuadPart < 0 ||
        !ifn_clock_from_ticks((uint64_t)ticks.QuadPart, (uint64_t)frequency.QuadPart, &ns))
        return 0;
    return ns;
}
#else
typedef int ifn_host_win_inactive;
#endif
