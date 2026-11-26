/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Artalis
 * Inkleberry Finn C library; see LICENSE and NOTICE for attribution.
 * Host input API; optional CUDA AR execution is declared in ifn_engine.h. */
#ifndef IFN_CORE_H
#define IFN_CORE_H
#include "ifn_version.h"
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32) && defined(IFN_SHARED)
#if defined(IFN_BUILD_SHARED)
#define IFN_API __declspec(dllexport)
#else
#define IFN_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) && defined(IFN_SHARED)
#define IFN_API __attribute__((visibility("default")))
#else
#define IFN_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define IFN_API_VERSION 1u
#define IFN_MAX_TENSORS 2048u
#define IFN_MAX_RANK 8u
#define IFN_MAX_NAME 255u
#define IFN_MAX_HEADER (16u * 1024u * 1024u)
typedef uint32_t ifn_status;

enum {
    IFN_OK,
    IFN_INVALID_ARGUMENT,
    IFN_IO,
    IFN_FORMAT,
    IFN_OUT_OF_MEMORY,
    IFN_NOT_FOUND,
    IFN_BUFFER_TOO_SMALL,
    IFN_LIMIT,
    IFN_BUSY,
    IFN_GPU_ERROR,
    IFN_UNSUPPORTED,
    IFN_CANCELLED
};

typedef uint32_t ifn_dtype;

enum { IFN_U8, IFN_BF16, IFN_F32, IFN_F8_E4M3 };

typedef void *(*ifn_allocate_fn)(void *context, size_t bytes);
typedef void (*ifn_release_fn)(void *context, void *allocation);

/* Callbacks must be paired, thread-safe when used concurrently, and valid
 * until every allocation using them is released. Allocation alignment must
 * satisfy max_align_t. No callback may reenter an active file operation. */
typedef struct {
    size_t struct_size;
    uint32_t api_version;
    void *context;
    ifn_allocate_fn allocate;
    ifn_release_fn release;
} ifn_options;

/* Keep this aggregate macro stable across clang-format 18 patch releases. */
/* clang-format off */
#define IFN_OPTIONS_INIT {sizeof(ifn_options), IFN_API_VERSION, NULL, NULL, NULL}
/* clang-format on */
typedef struct ifn_safetensors ifn_safetensors;

typedef struct {
    char name[IFN_MAX_NAME + 1];
    ifn_dtype dtype;
    uint32_t rank;
    uint64_t shape[IFN_MAX_RANK];
    uint64_t offset;
    uint64_t bytes;
} ifn_tensor_info;

IFN_API const char *ifn_version(void);
IFN_API const char *ifn_status_string(ifn_status status);
/* Thread-local diagnostic, overwritten by the next status-reporting call on this thread. */
IFN_API const char *ifn_last_error(void);
/* Opens a regular file read-only. The caller must keep it immutable while
 * mapped. out is cleared on failure. NULL options selects the default allocator. */
IFN_API ifn_status ifn_safetensors_open(const char *path, const ifn_options *options,
                                        ifn_safetensors **out);
/* Borrows the input bytes, which must stay alive and immutable until close.
 * This bounded memory entrypoint also supports fuzzing without filesystem IO. */
IFN_API ifn_status ifn_safetensors_open_memory(const void *data, size_t bytes,
                                               const ifn_options *options, ifn_safetensors **out);
/* NULL is accepted. Closing requires exclusive ownership; concurrent immutable
 * lookup is permitted only while the caller keeps the file alive. */
IFN_API void ifn_safetensors_close(ifn_safetensors *file);
IFN_API size_t ifn_safetensors_count(const ifn_safetensors *file);
IFN_API uint64_t ifn_safetensors_data_bytes(const ifn_safetensors *file);
/* Returned metadata and data spans are borrowed until close. Names are UTF-8
 * without embedded NUL. Output pointers are cleared on failure. */
IFN_API ifn_status ifn_safetensors_at(const ifn_safetensors *file, size_t index,
                                      const ifn_tensor_info **out);
IFN_API ifn_status ifn_safetensors_find(const ifn_safetensors *file, const char *name,
                                        size_t length, const ifn_tensor_info **out);
IFN_API ifn_status ifn_safetensors_data(const ifn_safetensors *file, size_t index, const void **out,
                                        size_t *bytes);
/* Tokenizer matches the pinned Zig BPE/Unicode segmentation, not arbitrary
 * Hugging Face configurations. Loading copies its tables; input JSON may then
 * be released. Limits: 64 MiB JSON, 204800 IDs, 600000 merges, 8 MiB token bytes.
 * Immutable tokenizer reads may run concurrently using distinct scratch handles. */
typedef struct ifn_tokenizer ifn_tokenizer;
typedef struct ifn_tokenizer_scratch ifn_tokenizer_scratch;
IFN_API ifn_status ifn_tokenizer_open(const char *json_path, const ifn_options *options,
                                      ifn_tokenizer **out);
IFN_API ifn_status ifn_tokenizer_open_memory(const void *json, size_t bytes,
                                             const ifn_options *options, ifn_tokenizer **out);
IFN_API void ifn_tokenizer_close(ifn_tokenizer *tokenizer);
IFN_API ifn_status ifn_tokenizer_scratch_create(const ifn_options *options,
                                                ifn_tokenizer_scratch **out);
IFN_API void ifn_tokenizer_scratch_destroy(ifn_tokenizer_scratch *scratch);
/* No per-call allocation. text is counted valid UTF-8 (embedded NUL allowed).
 * parse_special must be 0 or 1. Output must not alias input/scratch/tokenizer.
 * On BUFFER_TOO_SMALL, needed is the full token count, output may be partial.
 * NULL ids with zero capacity is a sizing query. Other failures clear needed.
 * A scratch handle requires exclusive access until the call returns. */
IFN_API ifn_status ifn_tokenizer_encode(const ifn_tokenizer *tokenizer, const char *text,
                                        size_t bytes, uint32_t parse_special,
                                        ifn_tokenizer_scratch *scratch, int32_t *ids,
                                        size_t capacity, size_t *needed);
/* Returns borrowed raw token bytes, possibly an incomplete UTF-8 sequence.
 * A streaming caller must assemble bytes before decoding Unicode. */
IFN_API ifn_status ifn_tokenizer_decode(const ifn_tokenizer *tokenizer, int32_t id,
                                        uint32_t skip_special, const void **out, size_t *bytes);
typedef uint32_t ifn_thinking_effort;

enum {
    IFN_THINK_NONE,
    IFN_THINK_MINIMAL,
    IFN_THINK_LOW,
    IFN_THINK_MEDIUM,
    IFN_THINK_HIGH,
    IFN_THINK_MAX
};

/* Matches Zig's single-user template; no assistant prefix is added. Needed
 * counts bytes excluding NUL. Output is a counted span, not a C string. */
IFN_API ifn_status ifn_render_user_prompt(const char *text, size_t bytes,
                                          ifn_thinking_effort effort, char *out, size_t capacity,
                                          size_t *needed);
/* Architecture-specific readonly binding barrier for the pinned Inkling text
 * engine. Opens/owns main and optional MTP maps; NULL MTP means AR-only, while
 * an explicitly requested missing/incompatible MTP is an error. Vision tensors
 * are not bound. Validates role, dtype, rank and exact shape before publication;
 * does not upload/pack or establish numerical validity of all weight values.
 * Files must remain immutable until close; borrowed views expire on close. */
typedef struct ifn_inkling_checkpoint ifn_inkling_checkpoint;
IFN_API ifn_status ifn_inkling_checkpoint_open(const char *main_path, const char *mtp_path,
                                               const ifn_options *options,
                                               ifn_inkling_checkpoint **out);
IFN_API void ifn_inkling_checkpoint_close(ifn_inkling_checkpoint *checkpoint);
IFN_API size_t ifn_inkling_checkpoint_count(const ifn_inkling_checkpoint *checkpoint);
/* source: 0 main, 1 MTP. Enumeration order is deterministic by engine role,
 * independent of serialized header order. */
IFN_API ifn_status ifn_inkling_checkpoint_at(const ifn_inkling_checkpoint *checkpoint, size_t index,
                                             uint32_t *source, const ifn_tensor_info **out);
IFN_API ifn_status ifn_inkling_checkpoint_data(const ifn_inkling_checkpoint *checkpoint,
                                               size_t index, const void **out, size_t *bytes);
#ifdef __cplusplus
}
#endif
#endif
