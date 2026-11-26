/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef SQ_INFER_INTERNAL_H
#define SQ_INFER_INTERNAL_H
#include "sq_infer.h"
#include "sq_infer_squid.h"
#include "sq_chat.h"
#include "engine_control.h"
#include "internal.h"
#include <stdatomic.h>
#include <string.h>
#define SI_FEATURES                                                                                \
    (SQ_INFER_CAP_TEXT_CHAT | SQ_INFER_CAP_RAW_TEXT | SQ_INFER_CAP_RAW_TOKEN_IO |                  \
     SQ_INFER_CAP_REASONING_OUTPUT | SQ_INFER_CAP_REASONING_HISTORY | SQ_INFER_CAP_PREFIX_CACHE |  \
     SQ_INFER_CAP_SPECULATION | SQ_INFER_CAP_MEMORY_PLAN)
#define SI_CANCEL 1u
#define SI_SEALED 2u
#define SI_MAX_INPUT (32u * 1024u * 1024u)
#define SI_UNKNOWN_VENDOR UINT64_C(16)

typedef struct {
    sq_options upstream;
    atomic_uint_fast64_t live;
    uint64_t limit;
} si_budget;

struct sq_infer_model {
    sq_options allocator;
    si_budget budget;
    atomic_flag busy;
    uint32_t state, children;
    sq_weight_config weights_config;
    const char *profile;
    char device_id[11];
    sq_gpu_weights *weights;
    sq_tokenizer *tokenizer;
    sq_inkling_checkpoint *checkpoint;
    uint64_t required, device_budget;
};

struct sq_infer_session {
    sq_infer_model *model;
    sq_options allocator;
    si_budget budget;
    atomic_flag busy;
    uint32_t state, children;
    sq_session_config config;
    sq_session *native;
    const char *profile;
    uint64_t device_budget;
    unsigned char cache_scope[64];
    size_t cache_scope_bytes;
};

struct sq_infer_request {
    sq_infer_session *session;
    sq_options allocator;
    atomic_uint control, phase;
    atomic_bool running;
    atomic_uint_fast64_t generated, processed;
    int32_t *tokens;
    size_t token_count;
    sq_decode_options decode;
    sq_infer_request_config_v1 config;
    unsigned char scope[64];
    size_t scope_bytes;
    sq_stream *stream;
    sq_infer_emit_fn emit;
    void *context;
    uint64_t sequence, started;
    sq_status callback_status;
    uint32_t callback_action;
    sq_infer_result_v1 result;
    sq_infer_error_v1 terminal_error;
};

sq_status si_input(const void *, size_t);
sq_status si_output(void *, const void *, size_t);
sq_status si_error(sq_infer_error_v1 *, sq_status, uint32_t, const char *);
sq_status si_native_error(sq_infer_error_v1 *, sq_status, uint32_t);
bool si_span(sq_infer_bytes, size_t);
bool si_equal(sq_infer_bytes, const char *);
sq_infer_bytes si_text(const char *);
void si_budget_init(si_budget *, const sq_options *, uint64_t, sq_options *);
void si_free(const sq_options *, void *);
void *si_alloc(const sq_options *, size_t);
bool si_lock(atomic_flag *);
void si_unlock(atomic_flag *);
sq_status si_no_extensions(sq_infer_array);
uint64_t si_now(void);
sq_status si_defaults(uint32_t, void *, sq_infer_error_v1 *);
sq_status si_backend_info(sq_infer_backend_info_v1 *, sq_infer_error_v1 *);
sq_status si_model_create(const sq_infer_model_config_v1 *, sq_infer_model **, sq_infer_error_v1 *);
sq_status si_model_plan(sq_infer_model *, const sq_infer_source_v1 *, sq_infer_memory_plan_v1 *,
                        sq_infer_error_v1 *);
sq_status si_model_load(sq_infer_model *, const sq_infer_source_v1 *, sq_infer_error_v1 *);
sq_status si_model_info(sq_infer_model *, sq_infer_model_info_v1 *, sq_infer_error_v1 *);
sq_status si_model_close(sq_infer_model **, sq_infer_error_v1 *);
sq_status si_session_create(sq_infer_model *, const sq_infer_session_config_v1 *,
                            sq_infer_session **, sq_infer_error_v1 *);
sq_status si_session_plan(sq_infer_session *, sq_infer_memory_plan_v1 *, sq_infer_error_v1 *);
sq_status si_session_prepare(sq_infer_session *, sq_infer_error_v1 *);
sq_status si_session_info(sq_infer_session *, sq_infer_session_info_v1 *, sq_infer_error_v1 *);
sq_status si_session_clear(sq_infer_session *, sq_infer_error_v1 *);
sq_status si_session_close(sq_infer_session **, sq_infer_error_v1 *);
sq_status si_request_create(sq_infer_session *, const sq_infer_input_v1 *,
                            const sq_infer_request_config_v1 *, sq_infer_request **,
                            sq_infer_error_v1 *);
sq_status si_request_run(sq_infer_request *, sq_infer_emit_fn, void *, sq_infer_error_v1 *);
sq_status si_request_cancel(sq_infer_request *, uint32_t *, sq_infer_error_v1 *);
sq_status si_request_progress(const sq_infer_request *, sq_infer_progress_v1 *,
                              sq_infer_error_v1 *);
sq_status si_request_result(const sq_infer_request *, sq_infer_result_v1 *, sq_infer_error_v1 *);
sq_status si_request_close(sq_infer_request **, sq_infer_error_v1 *);
#endif
