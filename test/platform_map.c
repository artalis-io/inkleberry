/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#define _POSIX_C_SOURCE 200809L
#include "file_map.h"
#include "internal.h"
#include "filesystem.h"
#define IFN_MAP_CHECK(x)                                                                           \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "mapping:%d: %s\n", __LINE__, #x);                                     \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

int main(void) {
    char path[512];
    int fd = ifn_test_temp(path, sizeof path);
    IFN_MAP_CHECK(fd >= 0 && ifn_test_write(fd, "test", 4) == 4 && !ifn_test_close(fd));
    ifn_options allocator;
    IFN_MAP_CHECK(ifn_options_resolve(NULL, &allocator) == IFN_OK);
    ifn_file_map map;
    IFN_MAP_CHECK(ifn_file_map_open(path, 5, SIZE_MAX, &allocator, &map) == IFN_FORMAT &&
                  !map.data && !map.size);
    IFN_MAP_CHECK(ifn_file_map_open(path, 1, 3, &allocator, &map) == IFN_FORMAT && !map.data);
#ifdef _WIN32
    /* A UTF-8 path must be decoded, never interpreted using the active codepage. */
    wchar_t original[512], unicode[512];
    IFN_MAP_CHECK(MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, original, 512));
    IFN_MAP_CHECK(swprintf(unicode, 512, L"%ls-\x0151-\x4e2d", original) > 0);
    IFN_MAP_CHECK(MoveFileW(original, unicode));
    IFN_MAP_CHECK(WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, unicode, -1, path, sizeof path,
                                      NULL, NULL));
    IFN_MAP_CHECK(ifn_file_map_open("\xff", 1, SIZE_MAX, &allocator, &map) ==
                      IFN_INVALID_ARGUMENT &&
                  !map.data);
    ifn_native_diagnostic d = ifn_native_diagnostic_get();
    IFN_MAP_CHECK(d.provider == 5 && d.code == ERROR_NO_UNICODE_TRANSLATION);
    IFN_MAP_CHECK(ifn_file_map_open("\\\\.\\pipe\\ifn-do-not-block", 1, SIZE_MAX, &allocator,
                                    &map) == IFN_FORMAT);
    DWORD before, after;
    IFN_MAP_CHECK(GetProcessHandleCount(GetCurrentProcess(), &before));
#endif
    for (unsigned i = 0; i < 1000; i++) {
        IFN_MAP_CHECK(ifn_file_map_open(path, 1, SIZE_MAX, &allocator, &map) == IFN_OK);
        IFN_MAP_CHECK(map.size == 4 && !memcmp(map.data, "test", 4));
        ifn_file_map_close(&map);
        IFN_MAP_CHECK(!map.data && !map.size);
        ifn_file_map_close(&map);
    }
#ifdef _WIN32
    IFN_MAP_CHECK(GetProcessHandleCount(GetCurrentProcess(), &after) && before == after);
    IFN_MAP_CHECK(DeleteFileW(unicode));
#else
    IFN_MAP_CHECK(!ifn_test_remove(path));
#endif
    ifn_file_map_close(NULL);
    puts("bounded mapping, UTF-8 paths and repeated cleanup passed");
    return 0;
}
