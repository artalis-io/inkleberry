/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sq.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void *worker(void *context) {
    const sq_safetensors *file = context;
    for (size_t i = 0; i < 10000; i++) {
        const sq_tensor_info *out = NULL;
        const void *data = NULL;
        size_t bytes = 0;
        if (sq_safetensors_count(file) != 1 || sq_safetensors_find(file, "a", 1, &out) != SQ_OK ||
            !out || out->bytes != 1 || out->shape[0] != 1 ||
            sq_safetensors_data(file, 0, &data, &bytes) != SQ_OK || bytes != 1 ||
            *(const unsigned char *)data != 0x42)
            return (void *)(uintptr_t)1;
        if (sq_safetensors_at(NULL, 0, &out) != SQ_INVALID_ARGUMENT || !sq_last_error()[0])
            return (void *)(uintptr_t)1;
    }
    return NULL;
}

int main(void) {
    const char header[] = "{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
    unsigned char input[8 + sizeof header];
    for (unsigned i = 0; i < 8; i++)
        input[i] = (unsigned char)((uint64_t)(sizeof header - 1) >> (8 * i));
    memcpy(input + 8, header, sizeof header - 1);
    input[sizeof input - 1] = 0x42;
    sq_safetensors *file = NULL;
    if (sq_safetensors_open_memory(input, sizeof input, NULL, &file) != SQ_OK)
        return 1;
    pthread_t threads[4];
    /* This thread's error must survive other threads' successful and failed calls. */
    sq_safetensors *out = NULL;
    if (sq_safetensors_open("/nonexistent/squid-c/model", NULL, &out) != SQ_IO)
        return 1;
    char original[256];
    (void)snprintf(original, sizeof original, "%s", sq_last_error());
    size_t created = 0;
    int failed = 0;
    for (; created < 4; created++) {
        if (pthread_create(&threads[created], NULL, worker, file)) {
            failed = 1;
            break;
        }
    }
    for (size_t i = 0; i < created; i++) {
        void *result;
        if (pthread_join(threads[i], &result) || result)
            failed = 1;
    }
    if (strcmp(original, sq_last_error()))
        failed = 1;
    sq_safetensors_close(file);
    if (!failed)
        puts("concurrent immutable lookups and thread-local diagnostic isolation passed");
    return failed;
}
