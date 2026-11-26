/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void *ifn_concurrency_local_worker(void *context) {
    const ifn_safetensors *file = context;
    for (size_t i = 0; i < 10000; i++) {
        const ifn_tensor_info *out = NULL;
        const void *data = NULL;
        size_t bytes = 0;
        if (ifn_safetensors_count(file) != 1 ||
            ifn_safetensors_find(file, "a", 1, &out) != IFN_OK || !out || out->bytes != 1 ||
            out->shape[0] != 1 || ifn_safetensors_data(file, 0, &data, &bytes) != IFN_OK ||
            bytes != 1 || *(const unsigned char *)data != 0x42)
            return (void *)(uintptr_t)1;
        if (ifn_safetensors_at(NULL, 0, &out) != IFN_INVALID_ARGUMENT || !ifn_last_error()[0])
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
    ifn_safetensors *file = NULL;
    if (ifn_safetensors_open_memory(input, sizeof input, NULL, &file) != IFN_OK)
        return 1;
    pthread_t threads[4];
    /* This thread's error must survive other threads' successful and failed calls. */
    ifn_safetensors *out = NULL;
    if (ifn_safetensors_open("/nonexistent/inkleberry/model", NULL, &out) != IFN_IO)
        return 1;
    char original[256];
    (void)snprintf(original, sizeof original, "%s", ifn_last_error());
    size_t created = 0;
    int failed = 0;
    for (; created < 4; created++) {
        if (pthread_create(&threads[created], NULL, ifn_concurrency_local_worker, file)) {
            failed = 1;
            break;
        }
    }
    for (size_t i = 0; i < created; i++) {
        void *result;
        if (pthread_join(threads[i], &result) || result)
            failed = 1;
    }
    if (strcmp(original, ifn_last_error()))
        failed = 1;
    ifn_safetensors_close(file);
    if (!failed)
        puts("concurrent immutable lookups and thread-local diagnostic isolation passed");
    return failed;
}
