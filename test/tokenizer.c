/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sq.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "tokenizer:%d: %s (%s)\n", __LINE__, #x, sq_last_error());             \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

typedef struct {
    size_t calls, fail, live;
} faults;

static void *alloc(void *p, size_t n) {
    faults *f = p;
    if (f->calls++ == f->fail)
        return NULL;
    void *x = malloc(n);
    if (x)
        f->live++;
    return x;
}

static void release(void *p, void *x) {
    faults *f = p;
    if (x)
        f->live--;
    free(x);
}

static int encode_checks(sq_tokenizer *t, sq_tokenizer_scratch *s) {
    int32_t ids[32] = {0};
    size_t n = 999;
    CHECK(sq_tokenizer_encode(t, "abc aaaa<|eos|>", 16, 1, s, ids, 32, &n) == SQ_OK);
    CHECK(n == 5 && ids[0] == 257 && ids[1] == 32 && ids[2] == 259 && ids[3] == 300 && ids[4] == 0);
    /* 16 above includes the counted trailing NUL: binary roundtrip must retain it. */
    CHECK(sq_tokenizer_encode(t, "abc aaaa<|eos|>", 15, 1, s, ids, 32, &n) == SQ_OK && n == 4);
    CHECK(sq_tokenizer_encode(t, "abc", 3, 0, s, NULL, 0, &n) == SQ_BUFFER_TOO_SMALL && n == 1);
    CHECK(sq_tokenizer_encode(t, "abc", 3, 0, s, ids, 1, &n) == SQ_OK && n == 1 && ids[0] == 257);
    CHECK(sq_tokenizer_encode(t, "xyz", 3, 0, s, ids, 2, &n) == SQ_BUFFER_TOO_SMALL && n == 3);
    CHECK(sq_tokenizer_encode(t, "\xff", 1, 0, s, ids, 32, &n) == SQ_FORMAT && n == 0);
    CHECK(sq_tokenizer_encode(t, NULL, 1, 0, s, ids, 32, &n) == SQ_INVALID_ARGUMENT && n == 0);
    CHECK(sq_tokenizer_encode(t, "", 0, 2, s, ids, 32, &n) == SQ_INVALID_ARGUMENT);
    CHECK(sq_tokenizer_encode(t, NULL, 0, 0, s, NULL, 0, &n) == SQ_OK && n == 0);
    CHECK(sq_tokenizer_encode(t, "<|eos|>", 7, 0, s, ids, 32, &n) == SQ_OK && n > 1);
    const void *data = NULL;
    size_t bytes = 0;
    CHECK(sq_tokenizer_decode(t, 300, 0, &data, &bytes) == SQ_OK && bytes == 7 &&
          !memcmp(data, "<|eos|>", 7));
    CHECK(sq_tokenizer_decode(t, 300, 1, &data, &bytes) == SQ_OK && bytes == 0);
    CHECK(sq_tokenizer_decode(t, 299, 0, &data, &bytes) == SQ_NOT_FOUND && !data && !bytes);
    CHECK(sq_tokenizer_decode(t, -1, 0, &data, &bytes) == SQ_NOT_FOUND && !data);
    CHECK(sq_tokenizer_decode(t, 0, 0, &data, &bytes) == SQ_OK && bytes == 1 &&
          *(const unsigned char *)data == 0);
    return 0;
}

typedef struct {
    sq_tokenizer *t;
    int result;
} worker_args;

static void *worker(void *p) {
    worker_args *a = p;
    sq_tokenizer_scratch *s = NULL;
    a->result = 1;
    if (sq_tokenizer_scratch_create(NULL, &s) == SQ_OK) {
        a->result = encode_checks(a->t, s);
        sq_tokenizer_scratch_destroy(s);
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--invalid")) {
        faults f = {0, SIZE_MAX, 0};
        sq_options o = SQ_OPTIONS_INIT;
        o.allocate = alloc;
        o.release = release;
        o.context = &f;
        sq_tokenizer *bad = (sq_tokenizer *)(uintptr_t)1;
        sq_status status = sq_tokenizer_open(argv[2], &o, &bad);
        CHECK((status == SQ_FORMAT || status == SQ_LIMIT) && !bad && !f.live);
        return 0;
    }
    CHECK(argc == 2);
    sq_tokenizer *t = NULL;
    CHECK(sq_tokenizer_open(argv[1], NULL, &t) == SQ_OK);
    sq_tokenizer_scratch *s = NULL;
    CHECK(sq_tokenizer_scratch_create(NULL, &s) == SQ_OK);
    CHECK(!encode_checks(t, s));
    FILE *input = fopen(argv[1], "rb");
    CHECK(input && !fseek(input, 0, SEEK_END));
    long end = ftell(input);
    CHECK(end > 0 && !fseek(input, 0, SEEK_SET));
    unsigned char *json = malloc((size_t)end);
    CHECK(json && fread(json, 1, (size_t)end, input) == (size_t)end && !fclose(input));
    sq_tokenizer *copy = NULL;
    CHECK(sq_tokenizer_open_memory(json, (size_t)end, NULL, &copy) == SQ_OK);
    memset(json, 0, (size_t)end);
    free(json);
    CHECK(!encode_checks(copy, s));
    sq_tokenizer_close(copy);

    sq_tokenizer_scratch_destroy(s);
    pthread_t threads[2];
    worker_args args[2] = {{t, 1}, {t, 1}};
    for (size_t i = 0; i < 2; i++)
        CHECK(!pthread_create(&threads[i], NULL, worker, &args[i]));
    for (size_t i = 0; i < 2; i++)
        CHECK(!pthread_join(threads[i], NULL) && !args[i].result);
    sq_tokenizer_close(t);
    for (size_t fail = 0; fail <= 5; fail++) {
        faults f = {0, fail, 0};
        sq_options o = SQ_OPTIONS_INIT;
        o.allocate = alloc;
        o.release = release;
        o.context = &f;
        t = NULL;
        sq_status status = sq_tokenizer_open(argv[1], &o, &t);
        CHECK(status == (fail < 5 ? SQ_OUT_OF_MEMORY : SQ_OK));
        CHECK(status == SQ_OK ? t != NULL : t == NULL);
        sq_tokenizer_close(t);
        CHECK(!f.live);
    }
    for (size_t fail = 0; fail <= 4; fail++) {
        faults f = {0, fail, 0};
        sq_options o = SQ_OPTIONS_INIT;
        o.allocate = alloc;
        o.release = release;
        o.context = &f;
        s = NULL;
        sq_status status = sq_tokenizer_scratch_create(&o, &s);
        CHECK(status == (fail < 4 ? SQ_OUT_OF_MEMORY : SQ_OK));
        sq_tokenizer_scratch_destroy(s);
        CHECK(!f.live);
    }
    /* Warm immutable tokenizer + scratch: encode/decode allocate nothing. */
    faults f = {0, SIZE_MAX, 0};
    sq_options o = SQ_OPTIONS_INIT;
    o.allocate = alloc;
    o.release = release;
    o.context = &f;
    CHECK(sq_tokenizer_open(argv[1], &o, &t) == SQ_OK &&
          sq_tokenizer_scratch_create(&o, &s) == SQ_OK);
    size_t before = f.calls;
    f.fail = before;
    for (unsigned i = 0; i < 100; i++)
        CHECK(!encode_checks(t, s));
    CHECK(f.calls == before);
    sq_tokenizer_scratch_destroy(s);
    sq_tokenizer_close(t);
    CHECK(!f.live);
    char out[256];
    size_t n = 0;
    CHECK(sq_render_user_prompt("hello", 5, SQ_THINK_LOW, NULL, 0, &n) == SQ_BUFFER_TOO_SMALL &&
          n < sizeof out);
    CHECK(sq_render_user_prompt("hello", 5, SQ_THINK_LOW, out, n, &n) == SQ_OK);
    CHECK(memcmp(out, "<|message_system|><|content_text|>Thinking effort level: 0.2", 57) == 0);
    CHECK(sq_render_user_prompt("", 0, 99, out, sizeof out, &n) == SQ_INVALID_ARGUMENT && n == 0);
    sq_tokenizer_close(NULL);
    sq_tokenizer_scratch_destroy(NULL);
    puts("tokenizer BPE, buffers, UTF-8, allocation faults, no-per-call allocation and concurrency "
         "passed");
    return 0;
}
