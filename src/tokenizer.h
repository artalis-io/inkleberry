/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_TOKENIZER_H
#define IFN_TOKENIZER_H
#include "internal.h"
#define IFN_TOK_IDS 204800u
#define IFN_TOK_MERGES 600000u
#define IFN_TOK_ARENA (8u * 1024u * 1024u)
#define IFN_TOK_JSON (64u * 1024u * 1024u)
#define IFN_TOK_PIECE 65536u
#define IFN_TOK_CHUNK (1024u * 1024u)

typedef struct {
    uint32_t offset, length;
    bool present, special;
} ifn_token_entry;

typedef struct {
    uint64_t pair;
    uint32_t rank;
    int32_t merged;
} ifn_merge;

struct ifn_tokenizer {
    ifn_options allocator;
    unsigned char *arena;
    ifn_token_entry *entries;
    uint32_t *vocab;
    ifn_merge *merges;
    size_t arena_used, arena_size, id_count, vocab_slots, merge_slots;
    int32_t byte_ids[256], specials[256];
    size_t special_count;
};

struct ifn_tokenizer_scratch {
    ifn_options allocator;
    uint32_t *codepoints, *offsets;
    int32_t *bpe;
};

uint64_t ifn_token_hash(const void *data, size_t size);
uint64_t ifn_pair_hash(uint64_t key);
int32_t ifn_special_lookup(const ifn_tokenizer *t, const void *bytes, size_t size);
int32_t ifn_vocab_lookup(const ifn_tokenizer *t, const void *bytes, size_t size);
bool ifn_utf8_next(const unsigned char *text, size_t bytes, size_t *pos, uint32_t *cp);
#endif
