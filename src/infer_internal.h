/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#ifndef IFN_INFER_INTERNAL_H
#define IFN_INFER_INTERNAL_H
#include "ifn_infer.h"
#include "ifn_infer_inkling.h"
#include "ifn_chat.h"
#include "engine_control.h"
#include "internal.h"
#include <stdatomic.h>
#include <string.h>
#define IFN_INFER_FEATURES                                                                         \
    (IFN_INFER_CAP_TEXT_CHAT | IFN_INFER_CAP_RAW_TEXT | IFN_INFER_CAP_RAW_TOKEN_IO |               \
     IFN_INFER_CAP_REASONING_OUTPUT | IFN_INFER_CAP_REASONING_HISTORY |                            \
     IFN_INFER_CAP_PREFIX_CACHE | IFN_INFER_CAP_SPECULATION | IFN_INFER_CAP_MEMORY_PLAN)
#define IFN_INFER_CANCEL 1u
#define IFN_INFER_SEALED 2u
#define IFN_INFER_MAX_INPUT (32u * 1024u * 1024u)
#define IFN_INFER_UNKNOWN_VENDOR UINT64_C(16)

typedef struct {
    ifn_options upstream;
    atomic_uint_fast64_t live;
    uint64_t limit;
} ifn_infer_budget;

struct ifn_infer_model {
    ifn_options allocator;
    ifn_infer_budget budget;
    atomic_flag busy;
    uint32_t state, children;
    ifn_weight_config weights_config;
    const char *profile;
    char device_id[11];
    ifn_gpu_weights *weights;
    ifn_tokenizer *tokenizer;
    ifn_inkling_checkpoint *checkpoint;
    uint64_t required, device_budget;
};

struct ifn_infer_session {
    ifn_infer_model *model;
    ifn_options allocator;
    ifn_infer_budget budget;
    atomic_flag busy;
    uint32_t state, children;
    ifn_session_config config;
    ifn_session *native;
    const char *profile;
    uint64_t device_budget;
    unsigned char cache_scope[64];
    size_t cache_scope_bytes;
};

struct ifn_infer_request {
    ifn_infer_session *session;
    ifn_options allocator;
    atomic_uint control, phase;
    atomic_bool running;
    atomic_uint_fast64_t generated, processed;
    int32_t *tokens;
    size_t token_count;
    ifn_decode_options decode;
    ifn_infer_request_config_v1 config;
    unsigned char scope[64];
    size_t scope_bytes;
    ifn_stream *stream;
    ifn_infer_emit_fn emit;
    void *context;
    uint64_t sequence, started;
    ifn_status callback_status;
    uint32_t callback_action;
    ifn_infer_result_v1 result;
    ifn_infer_error_v1 terminal_error;
};

ifn_status ifn_infer_input(const void *, size_t);
ifn_status ifn_infer_output(void *, const void *, size_t);
ifn_status ifn_infer_error(ifn_infer_error_v1 *, ifn_status, uint32_t, const char *);
ifn_status ifn_infer_native_error(ifn_infer_error_v1 *, ifn_status, uint32_t);
bool ifn_infer_span(ifn_infer_bytes, size_t);
bool ifn_infer_equal(ifn_infer_bytes, const char *);
ifn_infer_bytes ifn_infer_text(const char *);
void ifn_infer_budget_init(ifn_infer_budget *, const ifn_options *, uint64_t, ifn_options *);
void ifn_infer_free(const ifn_options *, void *);
void *ifn_infer_alloc(const ifn_options *, size_t);
bool ifn_infer_lock(atomic_flag *);
void ifn_infer_unlock(atomic_flag *);
ifn_status ifn_infer_no_extensions(ifn_infer_array);
uint64_t ifn_infer_now(void);
ifn_status ifn_infer_defaults(uint32_t, void *, ifn_infer_error_v1 *);
ifn_status ifn_infer_backend_info(ifn_infer_backend_info_v1 *, ifn_infer_error_v1 *);
ifn_status ifn_infer_model_create(const ifn_infer_model_config_v1 *, ifn_infer_model **,
                                  ifn_infer_error_v1 *);
ifn_status ifn_infer_model_plan(ifn_infer_model *, const ifn_infer_source_v1 *,
                                ifn_infer_memory_plan_v1 *, ifn_infer_error_v1 *);
ifn_status ifn_infer_model_load(ifn_infer_model *, const ifn_infer_source_v1 *,
                                ifn_infer_error_v1 *);
ifn_status ifn_infer_model_info(ifn_infer_model *, ifn_infer_model_info_v1 *, ifn_infer_error_v1 *);
ifn_status ifn_infer_model_close(ifn_infer_model **, ifn_infer_error_v1 *);
ifn_status ifn_infer_session_create(ifn_infer_model *, const ifn_infer_session_config_v1 *,
                                    ifn_infer_session **, ifn_infer_error_v1 *);
ifn_status ifn_infer_session_plan(ifn_infer_session *, ifn_infer_memory_plan_v1 *,
                                  ifn_infer_error_v1 *);
ifn_status ifn_infer_session_prepare(ifn_infer_session *, ifn_infer_error_v1 *);
ifn_status ifn_infer_session_info(ifn_infer_session *, ifn_infer_session_info_v1 *,
                                  ifn_infer_error_v1 *);
ifn_status ifn_infer_session_clear(ifn_infer_session *, ifn_infer_error_v1 *);
ifn_status ifn_infer_session_close(ifn_infer_session **, ifn_infer_error_v1 *);
ifn_status ifn_infer_request_create(ifn_infer_session *, const ifn_infer_input_v1 *,
                                    const ifn_infer_request_config_v1 *, ifn_infer_request **,
                                    ifn_infer_error_v1 *);
ifn_status ifn_infer_request_run(ifn_infer_request *, ifn_infer_emit_fn, void *,
                                 ifn_infer_error_v1 *);
ifn_status ifn_infer_request_cancel(ifn_infer_request *, uint32_t *, ifn_infer_error_v1 *);
ifn_status ifn_infer_request_progress(const ifn_infer_request *, ifn_infer_progress_v1 *,
                                      ifn_infer_error_v1 *);
ifn_status ifn_infer_request_result(const ifn_infer_request *, ifn_infer_result_v1 *,
                                    ifn_infer_error_v1 *);
ifn_status ifn_infer_request_close(ifn_infer_request **, ifn_infer_error_v1 *);
#endif
