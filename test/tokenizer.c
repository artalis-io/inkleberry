/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn.h"
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define IFN_TOKENIZER_LOCAL_CHECK(x)                                                               \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "tokenizer:%d: %s (%s)\n", __LINE__, #x, ifn_last_error());            \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

typedef struct {
    size_t calls, fail, live;
} ifn_tokenizer_local_faults;

static void *ifn_tokenizer_local_alloc(void *p, size_t n) {
    ifn_tokenizer_local_faults *f = p;
    if (f->calls++ == f->fail)
        return NULL;
    void *x = malloc(n);
    if (x)
        f->live++;
    return x;
}

static void ifn_tokenizer_local_release(void *p, void *x) {
    ifn_tokenizer_local_faults *f = p;
    if (x)
        f->live--;
    free(x);
}

static int ifn_tokenizer_local_encode_checks(ifn_tokenizer *t, ifn_tokenizer_scratch *s) {
    int32_t ids[32] = {0};
    size_t n = 999;
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_encode(t, "abc aaaa<|eos|>", 16, 1, s, ids, 32, &n) ==
                              IFN_OK);
    IFN_TOKENIZER_LOCAL_CHECK(n == 5 && ids[0] == 257 && ids[1] == 32 && ids[2] == 259 &&
                              ids[3] == 300 && ids[4] == 0);
    /* 16 above includes the counted trailing NUL: binary roundtrip must retain it. */
    IFN_TOKENIZER_LOCAL_CHECK(
        ifn_tokenizer_encode(t, "abc aaaa<|eos|>", 15, 1, s, ids, 32, &n) == IFN_OK && n == 4);
    IFN_TOKENIZER_LOCAL_CHECK(
        ifn_tokenizer_encode(t, "abc", 3, 0, s, NULL, 0, &n) == IFN_BUFFER_TOO_SMALL && n == 1);
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_encode(t, "abc", 3, 0, s, ids, 1, &n) == IFN_OK &&
                              n == 1 && ids[0] == 257);
    IFN_TOKENIZER_LOCAL_CHECK(
        ifn_tokenizer_encode(t, "xyz", 3, 0, s, ids, 2, &n) == IFN_BUFFER_TOO_SMALL && n == 3);
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_encode(t, "\xff", 1, 0, s, ids, 32, &n) == IFN_FORMAT &&
                              n == 0);
    IFN_TOKENIZER_LOCAL_CHECK(
        ifn_tokenizer_encode(t, NULL, 1, 0, s, ids, 32, &n) == IFN_INVALID_ARGUMENT && n == 0);
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_encode(t, "", 0, 2, s, ids, 32, &n) ==
                              IFN_INVALID_ARGUMENT);
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_encode(t, NULL, 0, 0, s, NULL, 0, &n) == IFN_OK &&
                              n == 0);
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_encode(t, "<|eos|>", 7, 0, s, ids, 32, &n) == IFN_OK &&
                              n > 1);
    const void *data = NULL;
    size_t bytes = 0;
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_decode(t, 300, 0, &data, &bytes) == IFN_OK &&
                              bytes == 7 && !memcmp(data, "<|eos|>", 7));
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_decode(t, 300, 1, &data, &bytes) == IFN_OK &&
                              bytes == 0);
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_decode(t, 299, 0, &data, &bytes) == IFN_NOT_FOUND &&
                              !data && !bytes);
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_decode(t, -1, 0, &data, &bytes) == IFN_NOT_FOUND &&
                              !data);
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_decode(t, 0, 0, &data, &bytes) == IFN_OK &&
                              bytes == 1 && *(const unsigned char *)data == 0);
    return 0;
}

typedef struct {
    ifn_tokenizer *t;
    int result;
} ifn_tokenizer_local_worker_args;

static void *ifn_tokenizer_local_worker(void *p) {
    ifn_tokenizer_local_worker_args *a = p;
    ifn_tokenizer_scratch *s = NULL;
    a->result = 1;
    if (ifn_tokenizer_scratch_create(NULL, &s) == IFN_OK) {
        a->result = ifn_tokenizer_local_encode_checks(a->t, s);
        ifn_tokenizer_scratch_destroy(s);
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--invalid")) {
        ifn_tokenizer_local_faults f = {0, SIZE_MAX, 0};
        ifn_options o = IFN_OPTIONS_INIT;
        o.allocate = ifn_tokenizer_local_alloc;
        o.release = ifn_tokenizer_local_release;
        o.context = &f;
        ifn_tokenizer *bad = (ifn_tokenizer *)(uintptr_t)1;
        ifn_status status = ifn_tokenizer_open(argv[2], &o, &bad);
        IFN_TOKENIZER_LOCAL_CHECK((status == IFN_FORMAT || status == IFN_LIMIT) && !bad && !f.live);
        return 0;
    }
    IFN_TOKENIZER_LOCAL_CHECK(argc == 2);
    ifn_tokenizer *t = NULL;
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_open(argv[1], NULL, &t) == IFN_OK);
    ifn_tokenizer_scratch *s = NULL;
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_scratch_create(NULL, &s) == IFN_OK);
    IFN_TOKENIZER_LOCAL_CHECK(!ifn_tokenizer_local_encode_checks(t, s));
    FILE *input = fopen(argv[1], "rb");
    IFN_TOKENIZER_LOCAL_CHECK(input && !fseek(input, 0, SEEK_END));
    long end = ftell(input);
    IFN_TOKENIZER_LOCAL_CHECK(end > 0 && !fseek(input, 0, SEEK_SET));
    unsigned char *json = malloc((size_t)end);
    IFN_TOKENIZER_LOCAL_CHECK(json && fread(json, 1, (size_t)end, input) == (size_t)end &&
                              !fclose(input));
    ifn_tokenizer *copy = NULL;
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_open_memory(json, (size_t)end, NULL, &copy) == IFN_OK);
    memset(json, 0, (size_t)end);
    free(json);
    IFN_TOKENIZER_LOCAL_CHECK(!ifn_tokenizer_local_encode_checks(copy, s));
    ifn_tokenizer_close(copy);

    ifn_tokenizer_scratch_destroy(s);
    ifn_test_thread threads[2];
    ifn_tokenizer_local_worker_args args[2] = {{t, 1}, {t, 1}};
    for (size_t i = 0; i < 2; i++)
        IFN_TOKENIZER_LOCAL_CHECK(
            !ifn_test_thread_create(&threads[i], NULL, ifn_tokenizer_local_worker, &args[i]));
    for (size_t i = 0; i < 2; i++)
        IFN_TOKENIZER_LOCAL_CHECK(!ifn_test_thread_join(threads[i], NULL) && !args[i].result);
    ifn_tokenizer_close(t);
    size_t allocations = 5;
#ifdef _WIN32
    allocations++;
#endif
    for (size_t fail = 0; fail <= allocations; fail++) {
        ifn_tokenizer_local_faults f = {0, fail, 0};
        ifn_options o = IFN_OPTIONS_INIT;
        o.allocate = ifn_tokenizer_local_alloc;
        o.release = ifn_tokenizer_local_release;
        o.context = &f;
        t = NULL;
        ifn_status status = ifn_tokenizer_open(argv[1], &o, &t);
        IFN_TOKENIZER_LOCAL_CHECK(status ==
                                  (ifn_status)(fail < allocations ? IFN_OUT_OF_MEMORY : IFN_OK));
        IFN_TOKENIZER_LOCAL_CHECK(status == IFN_OK ? t != NULL : t == NULL);
        ifn_tokenizer_close(t);
        IFN_TOKENIZER_LOCAL_CHECK(!f.live);
    }
    for (size_t fail = 0; fail <= 4; fail++) {
        ifn_tokenizer_local_faults f = {0, fail, 0};
        ifn_options o = IFN_OPTIONS_INIT;
        o.allocate = ifn_tokenizer_local_alloc;
        o.release = ifn_tokenizer_local_release;
        o.context = &f;
        s = NULL;
        ifn_status status = ifn_tokenizer_scratch_create(&o, &s);
        IFN_TOKENIZER_LOCAL_CHECK(status == (ifn_status)(fail < 4 ? IFN_OUT_OF_MEMORY : IFN_OK));
        ifn_tokenizer_scratch_destroy(s);
        IFN_TOKENIZER_LOCAL_CHECK(!f.live);
    }
    /* Warm immutable tokenizer + scratch: encode/decode allocate nothing. */
    ifn_tokenizer_local_faults f = {0, SIZE_MAX, 0};
    ifn_options o = IFN_OPTIONS_INIT;
    o.allocate = ifn_tokenizer_local_alloc;
    o.release = ifn_tokenizer_local_release;
    o.context = &f;
    IFN_TOKENIZER_LOCAL_CHECK(ifn_tokenizer_open(argv[1], &o, &t) == IFN_OK &&
                              ifn_tokenizer_scratch_create(&o, &s) == IFN_OK);
    size_t before = f.calls;
    f.fail = before;
    for (unsigned i = 0; i < 100; i++)
        IFN_TOKENIZER_LOCAL_CHECK(!ifn_tokenizer_local_encode_checks(t, s));
    IFN_TOKENIZER_LOCAL_CHECK(f.calls == before);
    ifn_tokenizer_scratch_destroy(s);
    ifn_tokenizer_close(t);
    IFN_TOKENIZER_LOCAL_CHECK(!f.live);
    char out[256];
    size_t n = 0;
    IFN_TOKENIZER_LOCAL_CHECK(ifn_render_user_prompt("hello", 5, IFN_THINK_LOW, NULL, 0, &n) ==
                                  IFN_BUFFER_TOO_SMALL &&
                              n < sizeof out);
    IFN_TOKENIZER_LOCAL_CHECK(ifn_render_user_prompt("hello", 5, IFN_THINK_LOW, out, n, &n) ==
                              IFN_OK);
    IFN_TOKENIZER_LOCAL_CHECK(
        memcmp(out, "<|message_system|><|content_text|>Thinking effort level: 0.2", 57) == 0);
    IFN_TOKENIZER_LOCAL_CHECK(
        ifn_render_user_prompt("", 0, 99, out, sizeof out, &n) == IFN_INVALID_ARGUMENT && n == 0);
    ifn_tokenizer_close(NULL);
    ifn_tokenizer_scratch_destroy(NULL);
    puts("tokenizer BPE, buffers, UTF-8, allocation faults, no-per-call allocation and concurrency "
         "passed");
    return 0;
}
