/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sq.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t bytes) {
    sq_tokenizer *t = NULL;
    sq_status status = sq_tokenizer_open_memory(data, bytes, NULL, &t);
    if (status == SQ_OK) {
        sq_tokenizer_scratch *s = NULL;
        if (sq_tokenizer_scratch_create(NULL, &s) == SQ_OK) {
            int32_t ids[256];
            unsigned char restored[256];
            size_t n = 0, at = 0, input = bytes < 256 ? bytes : 256;
            status = sq_tokenizer_encode(t, (const char *)data, input, 1, s, ids, 256, &n);
            if (status == SQ_OK) {
                for (size_t i = 0; i < n; i++) {
                    const void *part = NULL;
                    size_t size = 0;
                    if (sq_tokenizer_decode(t, ids[i], 0, &part, &size) != SQ_OK ||
                        size > sizeof restored - at)
                        abort();
                    memcpy(restored + at, part, size);
                    at += size;
                }
                if (at != input || memcmp(restored, data, input))
                    abort();
            } else if (status != SQ_FORMAT)
                abort(); /* a sampled UTF-8 suffix can be incomplete */
            sq_tokenizer_scratch_destroy(s);
        }
    } else if (t)
        abort();
    sq_tokenizer_close(t);
    return 0;
}
