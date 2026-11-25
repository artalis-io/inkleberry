/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Artalis
 * C host port of tensorfish/squid5090; see LICENSE and NOTICE.
 * Input parsing/tokenization and checkpoint binding; inference not ported. */
#ifndef SQ_H
#define SQ_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32) && defined(SQ_SHARED)
#if defined(SQ_BUILD_SHARED)
#define SQ_API __declspec(dllexport)
#else
#define SQ_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) && defined(SQ_SHARED)
#define SQ_API __attribute__((visibility("default")))
#else
#define SQ_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define SQ_API_VERSION 1u
#define SQ_MAX_TENSORS 2048u
#define SQ_MAX_RANK 8u
#define SQ_MAX_NAME 255u
#define SQ_MAX_HEADER (16u * 1024u * 1024u)
typedef uint32_t sq_status;

enum {
    SQ_OK,
    SQ_INVALID_ARGUMENT,
    SQ_IO,
    SQ_FORMAT,
    SQ_OUT_OF_MEMORY,
    SQ_NOT_FOUND,
    SQ_BUFFER_TOO_SMALL,
    SQ_LIMIT
};

typedef uint32_t sq_dtype;

enum { SQ_U8, SQ_BF16, SQ_F32, SQ_F8_E4M3 };

typedef void *(*sq_allocate_fn)(void *context, size_t bytes);
typedef void (*sq_release_fn)(void *context, void *allocation);

/* Callbacks must be paired, thread-safe when used concurrently, and valid
 * until every allocation using them is released. Allocation alignment must
 * satisfy max_align_t. No callback may reenter an active file operation. */
typedef struct {
    size_t struct_size;
    uint32_t api_version;
    void *context;
    sq_allocate_fn allocate;
    sq_release_fn release;
} sq_options;

/* Keep this aggregate macro stable across clang-format 18 patch releases. */
/* clang-format off */
#define SQ_OPTIONS_INIT {sizeof(sq_options), SQ_API_VERSION, NULL, NULL, NULL}
/* clang-format on */
typedef struct sq_safetensors sq_safetensors;

typedef struct {
    char name[SQ_MAX_NAME + 1];
    sq_dtype dtype;
    uint32_t rank;
    uint64_t shape[SQ_MAX_RANK];
    uint64_t offset;
    uint64_t bytes;
} sq_tensor_info;

SQ_API const char *sq_version(void);
SQ_API const char *sq_status_string(sq_status status);
/* Thread-local diagnostic, overwritten by the next status-reporting call on this thread. */
SQ_API const char *sq_last_error(void);
/* Opens a regular file read-only. The caller must keep it immutable while
 * mapped. out is cleared on failure. NULL options selects the default allocator. */
SQ_API sq_status sq_safetensors_open(const char *path, const sq_options *options,
                                     sq_safetensors **out);
/* Borrows the input bytes, which must stay alive and immutable until close.
 * This bounded memory entrypoint also supports fuzzing without filesystem IO. */
SQ_API sq_status sq_safetensors_open_memory(const void *data, size_t bytes,
                                            const sq_options *options, sq_safetensors **out);
/* NULL is accepted. Closing requires exclusive ownership; concurrent immutable
 * lookup is permitted only while the caller keeps the file alive. */
SQ_API void sq_safetensors_close(sq_safetensors *file);
SQ_API size_t sq_safetensors_count(const sq_safetensors *file);
SQ_API uint64_t sq_safetensors_data_bytes(const sq_safetensors *file);
/* Returned metadata and data spans are borrowed until close. Names are UTF-8
 * without embedded NUL. Output pointers are cleared on failure. */
SQ_API sq_status sq_safetensors_at(const sq_safetensors *file, size_t index,
                                   const sq_tensor_info **out);
SQ_API sq_status sq_safetensors_find(const sq_safetensors *file, const char *name, size_t length,
                                     const sq_tensor_info **out);
SQ_API sq_status sq_safetensors_data(const sq_safetensors *file, size_t index, const void **out,
                                     size_t *bytes);
/* Tokenizer matches the pinned Zig BPE/Unicode segmentation, not arbitrary
 * Hugging Face configurations. Loading copies its tables; input JSON may then
 * be released. Limits: 64 MiB JSON, 204800 IDs, 600000 merges, 8 MiB token bytes.
 * Immutable tokenizer reads may run concurrently using distinct scratch handles. */
typedef struct sq_tokenizer sq_tokenizer;
typedef struct sq_tokenizer_scratch sq_tokenizer_scratch;
SQ_API sq_status sq_tokenizer_open(const char *json_path, const sq_options *options,
                                   sq_tokenizer **out);
SQ_API sq_status sq_tokenizer_open_memory(const void *json, size_t bytes, const sq_options *options,
                                          sq_tokenizer **out);
SQ_API void sq_tokenizer_close(sq_tokenizer *tokenizer);
SQ_API sq_status sq_tokenizer_scratch_create(const sq_options *options, sq_tokenizer_scratch **out);
SQ_API void sq_tokenizer_scratch_destroy(sq_tokenizer_scratch *scratch);
/* No per-call allocation. text is counted valid UTF-8 (embedded NUL allowed).
 * parse_special must be 0 or 1. Output must not alias input/scratch/tokenizer.
 * On BUFFER_TOO_SMALL, needed is the full token count, output may be partial.
 * NULL ids with zero capacity is a sizing query. Other failures clear needed.
 * A scratch handle requires exclusive access until the call returns. */
SQ_API sq_status sq_tokenizer_encode(const sq_tokenizer *tokenizer, const char *text, size_t bytes,
                                     uint32_t parse_special, sq_tokenizer_scratch *scratch,
                                     int32_t *ids, size_t capacity, size_t *needed);
/* Returns borrowed raw token bytes, possibly an incomplete UTF-8 sequence.
 * A streaming caller must assemble bytes before decoding Unicode. */
SQ_API sq_status sq_tokenizer_decode(const sq_tokenizer *tokenizer, int32_t id,
                                     uint32_t skip_special, const void **out, size_t *bytes);
typedef uint32_t sq_thinking_effort;

enum {
    SQ_THINK_NONE,
    SQ_THINK_MINIMAL,
    SQ_THINK_LOW,
    SQ_THINK_MEDIUM,
    SQ_THINK_HIGH,
    SQ_THINK_MAX
};

/* Matches Zig's single-user template; no assistant prefix is added. Needed
 * counts bytes excluding NUL. Output is a counted span, not a C string. */
SQ_API sq_status sq_render_user_prompt(const char *text, size_t bytes, sq_thinking_effort effort,
                                       char *out, size_t capacity, size_t *needed);
/* Architecture-specific readonly binding barrier for the pinned Inkling text
 * engine. Opens/owns main and optional MTP maps; NULL MTP means AR-only, while
 * an explicitly requested missing/incompatible MTP is an error. Vision tensors
 * are not bound. Validates role, dtype, rank and exact shape before publication;
 * does not upload/pack or establish numerical validity of all weight values.
 * Files must remain immutable until close; borrowed views expire on close. */
typedef struct sq_inkling_checkpoint sq_inkling_checkpoint;
SQ_API sq_status sq_inkling_checkpoint_open(const char *main_path, const char *mtp_path,
                                            const sq_options *options, sq_inkling_checkpoint **out);
SQ_API void sq_inkling_checkpoint_close(sq_inkling_checkpoint *checkpoint);
SQ_API size_t sq_inkling_checkpoint_count(const sq_inkling_checkpoint *checkpoint);
/* source: 0 main, 1 MTP. Enumeration order is deterministic by engine role,
 * independent of serialized header order. */
SQ_API sq_status sq_inkling_checkpoint_at(const sq_inkling_checkpoint *checkpoint, size_t index,
                                          uint32_t *source, const sq_tensor_info **out);
SQ_API sq_status sq_inkling_checkpoint_data(const sq_inkling_checkpoint *checkpoint, size_t index,
                                            const void **out, size_t *bytes);
#ifdef __cplusplus
}
#endif
#endif
