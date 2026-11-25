/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Artalis
 * C host port of tensorfish/squid5090; see LICENSE and NOTICE.
 * Initial API: bounded checkpoint inspection, not inference. */
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

enum { SQ_OK, SQ_INVALID_ARGUMENT, SQ_IO, SQ_FORMAT, SQ_OUT_OF_MEMORY, SQ_NOT_FOUND };

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

#define SQ_OPTIONS_INIT {sizeof(sq_options), SQ_API_VERSION, NULL, NULL, NULL}
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
#ifdef __cplusplus
}
#endif
#endif
