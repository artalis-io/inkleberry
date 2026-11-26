/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sha256.h"
#include <stdio.h>

int main(void) {
    ifn_sha256 s;
    ifn_sha256_init(&s);
    unsigned char block[8192], digest[32];
    char hex[65];
    size_t n;
    while ((n = fread(block, 1, sizeof block, stdin)))
        if (!ifn_sha256_update(&s, block, n))
            return 1;
    if (ferror(stdin))
        return 1;
    ifn_sha256_final(&s, digest);
    ifn_sha256_hex(digest, hex);
    puts(hex);
    return 0;
}
