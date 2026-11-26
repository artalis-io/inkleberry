/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Experimental versioned serving facade. See docs/inference-abi.md.
 * Multiple independent requests/streams; execution limits are discovered at runtime. */
#ifndef IFN_INFER_H
#define IFN_INFER_H
#include <ifn_core.h>
#include <string.h>
#if defined(_WIN32)
#define IFN_INFER_CALL __cdecl
#else
#define IFN_INFER_CALL
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define IFN_INFER_ABI_MAJOR 1u
#define IFN_INFER_ABI_MINOR 1u
#define IFN_INFER_RECORD_MAX_BYTES 4096u

/* Existing IFN_OK..IFN_CANCELLED retain their current values 0..11.
 * Facade-specific statuses append without renumbering the low-level API. */
enum { IFN_INFER_ABI_MISMATCH = 12, IFN_INFER_INVALID_STATE = 13, IFN_INFER_NOT_READY = 14 };
typedef struct ifn_infer_model ifn_infer_model;
typedef struct ifn_infer_session ifn_infer_session;
typedef struct ifn_infer_request ifn_infer_request;

/* Fixed-layout primitives: never append fields to these. */
typedef struct {
    const void *data;
    uint64_t bytes;
} ifn_infer_bytes;

typedef struct {
    const void *data;
    uint64_t count;
    uint32_t stride;
    uint32_t reserved;
} ifn_infer_array;

typedef struct {
    uint32_t struct_size;
    uint32_t flags;
} ifn_infer_in;

typedef struct {
    uint32_t struct_size;
    uint32_t written_size;
} ifn_infer_out;

/* Frozen primitive; does not embed the evolving low-level ifn_options record. */
typedef void *(IFN_INFER_CALL *ifn_infer_allocate_fn)(void *, size_t);
typedef void(IFN_INFER_CALL *ifn_infer_release_fn)(void *, void *);

typedef struct {
    void *context;
    ifn_infer_allocate_fn allocate;
    ifn_infer_release_fn release;
} ifn_infer_allocator;

/* A source helper, not a by-value exported ABI function. Supported records fit
 * in uint32_t and start with an input/output header. No side-effecting operands. */
#define IFN_INFER_INIT(record)                                                                     \
    do {                                                                                           \
        memset(&(record), 0, sizeof(record));                                                      \
        (record).hdr.struct_size = (uint32_t)sizeof(record);                                       \
    } while (0)

/* Error storage is caller-owned and survives all subsequent calls. */
#define IFN_INFER_ERR_SESSION_POISONED UINT32_C(1)
#define IFN_INFER_ERR_CLEANUP_PENDING UINT32_C(2)
#define IFN_INFER_ERR_OUTPUT_STARTED UINT32_C(4)

typedef struct {
    ifn_infer_out hdr;
    ifn_status status;
    uint32_t stage;
    uint32_t flags;
    uint32_t provider;
    int64_t provider_code;
    ifn_status cleanup_status;
    uint32_t message_bytes;
    int64_t cleanup_provider_code;
    char message[256];
} ifn_infer_error_v1;

enum {
    IFN_STAGE_NONE = 0,
    IFN_STAGE_VALIDATE = 1,
    IFN_STAGE_LOAD = 2,
    IFN_STAGE_PREPARE = 3,
    IFN_STAGE_PREFILL = 4,
    IFN_STAGE_DECODE = 5,
    IFN_STAGE_DRAIN = 6,
    IFN_STAGE_CLOSE = 7
};

enum {
    IFN_PROVIDER_NONE = 0,
    IFN_PROVIDER_OS = 1,
    IFN_PROVIDER_CUDA = 2,
    IFN_PROVIDER_CUBLAS = 3,
    IFN_PROVIDER_CUDA_DRIVER = 4
};

/* Capabilities describe supported contracts, not marketing architecture labels. */
#define IFN_INFER_CAP_TEXT_CHAT (UINT64_C(1) << 0)
#define IFN_INFER_CAP_RAW_TEXT (UINT64_C(1) << 1)
#define IFN_INFER_CAP_RAW_TOKEN_IO (UINT64_C(1) << 2)
#define IFN_INFER_CAP_REASONING_OUTPUT (UINT64_C(1) << 3)
#define IFN_INFER_CAP_REASONING_HISTORY (UINT64_C(1) << 4)
#define IFN_INFER_CAP_PREFIX_CACHE (UINT64_C(1) << 5)
#define IFN_INFER_CAP_SPECULATION (UINT64_C(1) << 6)
#define IFN_INFER_CAP_MEMORY_PLAN (UINT64_C(1) << 7)
#define IFN_INFER_CAP_ARTIFACT_VERIFY (UINT64_C(1) << 9)
#define IFN_INFER_CAP_LOGITS_EXTENSION (UINT64_C(1) << 8)

typedef struct {
    ifn_infer_out hdr;
    ifn_infer_bytes backend_id, engine_revision;
    uint64_t compiled_features;
    uint32_t pointer_bits;
    uint32_t limit_scope;
    uint32_t max_prepared_sessions;
    uint32_t max_active_runs;
} ifn_infer_backend_info_v1;

enum { IFN_LIMIT_PROVIDER_PROCESS = 1, IFN_LIMIT_DEVICE = 2 };

typedef struct {
    ifn_infer_in hdr;
    uint64_t id;
    uint32_t version;
    uint32_t payload_bytes;
    const void *payload;
} ifn_infer_extension_v1;

/* id is qualified by the backend identifier; unknown/duplicate/version-mismatched
 * requested extensions fail. Payloads are typed records, not arbitrary JSON. */

/* Optional per-model diagnostics. Callback spans are borrowed for this call.
 * Log callbacks must be prompt, thread-safe, must not throw/longjmp, and may
 * not call library APIs or close handles. No input text is logged. */
#define IFN_INFER_EXT_LOGGER UINT64_C(2)
#define IFN_LOG_ERROR 0u
typedef void(IFN_INFER_CALL *ifn_infer_log_fn)(void *, uint32_t, const ifn_infer_error_v1 *);

typedef struct {
    ifn_infer_in hdr;
    ifn_infer_log_fn callback;
    void *context;
} ifn_infer_logger_v1;

/* Explicit opt-in: SHA-256 is checked against the exact bytes consumed at load.
 * Expected hashes are copied from a trusted operator/release manifest. Full
 * content verification adds load-time work, never per-token work. */
#define IFN_INFER_EXT_ARTIFACT_MANIFEST UINT64_C(3)

typedef struct {
    ifn_infer_in hdr;
    ifn_infer_bytes tokenizer_sha256, weights_sha256, mtp_sha256;
} ifn_infer_artifact_manifest_v1;

typedef struct {
    ifn_infer_in hdr;
    ifn_infer_allocator allocator;
    ifn_infer_bytes device_id;
    ifn_infer_bytes weight_profile;
    uint64_t required_features;
    uint64_t max_owned_host_bytes;
    uint64_t max_owned_device_bytes;
    ifn_infer_array extensions;
} ifn_infer_model_config_v1;

typedef struct {
    ifn_infer_in hdr;
    ifn_infer_bytes directory;
    ifn_infer_bytes format_id;
} ifn_infer_source_v1;

enum {
    IFN_HANDLE_CREATED = 0,
    IFN_HANDLE_PREPARING = 1,
    IFN_HANDLE_READY = 2,
    IFN_HANDLE_FAILED = 3,
    IFN_HANDLE_POISONED = 4,
    IFN_HANDLE_CLOSING = 5
};

typedef struct {
    ifn_infer_out hdr;
    uint32_t state;
    uint32_t max_prepared_sessions;
    uint64_t supported_features;
    uint64_t enabled_features;
    uint64_t max_context_tokens;
    uint64_t max_messages;
    uint64_t max_input_bytes;
    uint32_t max_stops;
    uint32_t max_stop_bytes;
    uint32_t max_event_bytes;
    uint32_t fingerprint_verified;
    ifn_infer_bytes model_id, artifact_fingerprint, resolved_weight_profile;
    ifn_infer_bytes tokenizer_id, template_id, resolved_device_id;
} ifn_infer_model_info_v1;

/* Plans distinguish explicitly owned bytes from vendor/runtime estimates.
 * unknown_fields bits correspond to the five size fields below, in order. */
#define IFN_INFER_MEMORY_UNKNOWN_HOST UINT64_C(1)
#define IFN_INFER_MEMORY_UNKNOWN_MAPPED UINT64_C(2)
#define IFN_INFER_MEMORY_UNKNOWN_DEVICE UINT64_C(4)
#define IFN_INFER_MEMORY_UNKNOWN_PEAK UINT64_C(8)
#define IFN_INFER_MEMORY_UNKNOWN_VENDOR UINT64_C(16)

typedef struct {
    ifn_infer_out hdr;
    uint64_t owned_host_bytes;
    uint64_t mapped_file_bytes;
    uint64_t owned_device_bytes;
    uint64_t peak_owned_device_bytes;
    uint64_t vendor_device_estimate;
    uint64_t unknown_fields;
} ifn_infer_memory_plan_v1;

typedef struct {
    ifn_infer_in hdr;
    uint64_t context_tokens;
    uint64_t max_owned_host_bytes;
    uint64_t max_owned_device_bytes;
    ifn_infer_bytes execution_profile;
    ifn_infer_array extensions;
} ifn_infer_session_config_v1;

typedef struct {
    ifn_infer_out hdr;
    uint32_t state;
    uint32_t max_active_runs;
    uint64_t context_tokens;
    uint64_t enabled_features;
    ifn_infer_bytes resolved_execution_profile;
} ifn_infer_session_info_v1;

enum { IFN_INPUT_CHAT = 1, IFN_INPUT_RAW_TEXT = 2, IFN_INPUT_TOKENS = 3 };

enum { IFN_ROLE_SYSTEM = 1, IFN_ROLE_DEVELOPER = 2, IFN_ROLE_USER = 3, IFN_ROLE_ASSISTANT = 4 };

typedef struct {
    ifn_infer_in hdr;
    uint32_t role;
    uint32_t content_present;
    ifn_infer_bytes content, name, reasoning;
} ifn_infer_message_v1;

typedef struct {
    ifn_infer_in hdr;
    uint32_t kind;
    uint32_t reserved;
    ifn_infer_bytes text;
    ifn_infer_array messages;
    const int32_t *tokens;
    uint64_t token_count;
} ifn_infer_input_v1;

enum { IFN_OUTPUT_TEXT = 1, IFN_OUTPUT_RAW_TOKENS = 2, IFN_OUTPUT_DISCARD = 3 };

enum { IFN_HISTORY_OMIT_REASONING = 0, IFN_HISTORY_KEEP_REASONING = 1 };

#define IFN_INFER_OUTPUT_REASONING UINT32_C(1)

typedef struct {
    ifn_infer_in hdr;
    uint64_t max_output_tokens;
    uint64_t seed;
    float temperature;
    float top_p;
    int32_t top_k;
    uint32_t output_kind;
    uint32_t output_flags;
    uint32_t max_event_bytes;
    uint32_t history_policy;
    uint32_t reserved;
    uint64_t required_features;
    ifn_infer_array stops;       /* fixed-layout ifn_infer_bytes elements */
    ifn_infer_bytes cache_scope; /* empty disables cross-request reuse */
    ifn_infer_array extensions;
} ifn_infer_request_config_v1;

/* Library-resolved named defaults, so FFI consumers need not duplicate macros. */
enum { IFN_DEFAULT_MODEL = 1, IFN_DEFAULT_SESSION = 2, IFN_DEFAULT_REQUEST = 3 };

enum {
    IFN_REQUEST_PREPARED = 0,
    IFN_REQUEST_RUNNING = 1,
    IFN_REQUEST_COMPLETED = 2,
    IFN_REQUEST_CANCELLED = 3,
    IFN_REQUEST_FAILED = 4
};

enum {
    IFN_FINISH_EOS_V1 = 1,
    IFN_FINISH_OUTPUT_LIMIT_V1 = 2,
    IFN_FINISH_STOP_STRING_V1 = 3,
    IFN_FINISH_CALLBACK_V1 = 4,
    IFN_FINISH_CANCELLED_V1 = 5,
    IFN_FINISH_ERROR_V1 = 6
};

enum { IFN_CANCEL_ACCEPTED = 1, IFN_CANCEL_ALREADY_REQUESTED = 2, IFN_CANCEL_TOO_LATE = 3 };

enum { IFN_EVENT_CONTENT = 1, IFN_EVENT_REASONING = 2, IFN_EVENT_TOKENS = 3 };

typedef struct {
    ifn_infer_out hdr;
    uint32_t kind;
    uint32_t reserved;
    uint64_t sequence;
    uint64_t generated_tokens_so_far;
    ifn_infer_bytes text;
    const int32_t *tokens;
    uint64_t token_count;
} ifn_infer_event_v1;

enum { IFN_CALLBACK_CONTINUE = 0, IFN_CALLBACK_STOP = 1, IFN_CALLBACK_CANCEL = 2 };

typedef uint32_t(IFN_INFER_CALL *ifn_infer_emit_fn)(void *, const ifn_infer_event_v1 *);

typedef struct {
    ifn_infer_out hdr;
    uint32_t state;
    uint32_t cancel_requested;
    uint64_t prompt_tokens_processed;
    uint64_t generated_tokens;
} ifn_infer_progress_v1;

#define IFN_INFER_TIME_PREPARE UINT32_C(1)
#define IFN_INFER_TIME_RUN UINT32_C(2)
#define IFN_INFER_TIME_PREFILL UINT32_C(4)
#define IFN_INFER_TIME_FIRST_CONTENT UINT32_C(8)
#define IFN_INFER_TIME_FIRST_REASONING UINT32_C(16)

typedef struct {
    ifn_infer_out hdr;
    uint32_t finish_reason;
    uint32_t cancel_requested;
    ifn_status status;
    uint32_t timing_valid;
    uint64_t input_tokens, generated_tokens, prefix_reused_tokens;
    uint64_t content_bytes, reasoning_bytes, utf8_replacements;
    uint64_t prepare_ns, run_ns, prefill_ns;
    uint64_t first_content_ns, first_reasoning_ns;
    ifn_infer_bytes resolved_execution_profile;
} ifn_infer_result_v1;

/* Static read-only provider table. Append functions only in minor versions.
 * Core entries are non-NULL; unsupported optional work returns UNSUPPORTED. */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_major;
    uint32_t abi_minor;
    uint32_t reserved;
    ifn_status(IFN_INFER_CALL *backend_info)(ifn_infer_backend_info_v1 *, ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *defaults)(uint32_t kind, void *record, ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *model_create)(const ifn_infer_model_config_v1 *, ifn_infer_model **,
                                             ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *model_plan)(ifn_infer_model *, const ifn_infer_source_v1 *,
                                           ifn_infer_memory_plan_v1 *, ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *model_load)(ifn_infer_model *, const ifn_infer_source_v1 *,
                                           ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *model_info)(ifn_infer_model *, ifn_infer_model_info_v1 *,
                                           ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *model_close)(ifn_infer_model **, ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *session_create)(ifn_infer_model *,
                                               const ifn_infer_session_config_v1 *,
                                               ifn_infer_session **, ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *session_plan)(ifn_infer_session *, ifn_infer_memory_plan_v1 *,
                                             ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *session_prepare)(ifn_infer_session *, ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *session_info)(ifn_infer_session *, ifn_infer_session_info_v1 *,
                                             ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *session_clear_cache)(ifn_infer_session *, ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *session_close)(ifn_infer_session **, ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *request_create)(ifn_infer_session *, const ifn_infer_input_v1 *,
                                               const ifn_infer_request_config_v1 *,
                                               ifn_infer_request **, ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *request_run)(ifn_infer_request *, ifn_infer_emit_fn, void *,
                                            ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *request_cancel)(ifn_infer_request *, uint32_t *disposition,
                                               ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *request_progress)(const ifn_infer_request *, ifn_infer_progress_v1 *,
                                                 ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *request_result)(const ifn_infer_request *, ifn_infer_result_v1 *,
                                               ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *request_close)(ifn_infer_request **, ifn_infer_error_v1 *);
    ifn_status(IFN_INFER_CALL *query_extension)(uint64_t id, uint32_t min_version,
                                                const void **table, ifn_infer_error_v1 *);
} ifn_infer_api_v1;

/* Sole required dynamic symbol. Use a real const void* variable for out_table;
 * do not cast a T** to void**. Clear output on mismatch/failure. */
IFN_API ifn_status IFN_INFER_CALL ifn_infer_get_api(uint32_t major, uint32_t min_minor,
                                                    const void **out_table);
#ifdef __cplusplus
}
#endif
#endif
