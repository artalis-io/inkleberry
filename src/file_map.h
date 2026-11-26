/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_FILE_MAP_H
#define IFN_FILE_MAP_H
#include "ifn_core.h"

typedef struct {
    const unsigned char *data;
    size_t size;
} ifn_file_map;

/* Private read-only view. The caller keeps backing bytes immutable until close.
 * Allocator is resolved; only temporary Windows path storage uses its callbacks. */
ifn_status ifn_file_map_open(const char *, size_t, size_t, const ifn_options *, ifn_file_map *);
void ifn_file_map_close(ifn_file_map *);
#endif
