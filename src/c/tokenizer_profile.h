/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Supported public Inkling ByteLevel profile; model/tokenizer data stay external. */
#ifndef SQ_TOKENIZER_PROFILE_H
#define SQ_TOKENIZER_PROFILE_H
#define PRETOKENIZER_PROFILE                                                                       \
    "{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":\"[^"    \
    "\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*[\\\\p{Ll}\\\\p{"  \
    "Lm}\\\\p{Lo}\\\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}" \
    "\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]+[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*(?i:'s|'t|'re|'"   \
    "ve|'m|'ll|'d)?|\\\\p{N}{1,3}| "                                                               \
    "?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n/"                                                       \
    "]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"},\"behavior\":\"Isolated\",\"invert\":false}" \
    ",{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":"     \
    "false}]}"
#endif
