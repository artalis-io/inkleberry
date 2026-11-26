/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn.h"
#include <stdio.h>
#include <string.h>

static uint32_t ifn_generate_local_output(void *context, const ifn_infer_event_v1 *event) {
    (void)context;
    if (event->kind == IFN_EVENT_CONTENT && event->text.bytes &&
        fwrite(event->text.data, 1, (size_t)event->text.bytes, stdout) != event->text.bytes)
        return IFN_CALLBACK_CANCEL;
    return IFN_CALLBACK_CONTINUE;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fputs("usage: generate MODEL_DIR PROMPT\n", stderr);
        return 2;
    }
    const void *raw = NULL;
    if (ifn_infer_get_api(1, 0, &raw) != IFN_OK)
        return 1;
    const ifn_infer_api_v1 *api = raw;
    ifn_infer_model *model = NULL;
    ifn_infer_session *session = NULL;
    ifn_infer_request *request = NULL;
    ifn_infer_model_config_v1 mc;
    ifn_infer_session_config_v1 sc;
    ifn_infer_request_config_v1 rc;
    ifn_infer_source_v1 source;
    ifn_infer_input_v1 input;
    ifn_infer_message_v1 message;
    IFN_INFER_INIT(mc);
    IFN_INFER_INIT(sc);
    IFN_INFER_INIT(rc);
    IFN_INFER_INIT(source);
    IFN_INFER_INIT(input);
    IFN_INFER_INIT(message);
    int failed = 1;
    ifn_infer_error_v1 error;
    IFN_INFER_INIT(error);
#define IFN_EXAMPLE_TRY(call)                                                                      \
    do {                                                                                           \
        ifn_status code = (call);                                                                  \
        if (code) {                                                                                \
            fprintf(stderr, "Inkleberry inference: %s (stage %u, provider %u, code %lld): %.*s\n", \
                    ifn_status_string(code), error.stage, error.provider,                          \
                    (long long)error.provider_code, (int)error.message_bytes, error.message);      \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)
    IFN_EXAMPLE_TRY(api->defaults(IFN_DEFAULT_MODEL, &mc, &error));
    IFN_EXAMPLE_TRY(api->defaults(IFN_DEFAULT_SESSION, &sc, &error));
    IFN_EXAMPLE_TRY(api->defaults(IFN_DEFAULT_REQUEST, &rc, &error));
    source.directory = (ifn_infer_bytes){argv[1], strlen(argv[1])};
    source.format_id = (ifn_infer_bytes){"inkling-safetensors", sizeof("inkling-safetensors") - 1};
    message.role = IFN_ROLE_USER;
    message.content_present = 1;
    message.content = (ifn_infer_bytes){argv[2], strlen(argv[2])};
    input.kind = IFN_INPUT_CHAT;
    input.messages = (ifn_infer_array){&message, 1, sizeof message, 0};
    rc.max_output_tokens = 128;
    IFN_EXAMPLE_TRY(api->model_create(&mc, &model, &error));
    IFN_EXAMPLE_TRY(api->model_load(model, &source, &error));
    IFN_EXAMPLE_TRY(api->session_create(model, &sc, &session, &error));
    IFN_EXAMPLE_TRY(api->session_prepare(session, &error));
    IFN_EXAMPLE_TRY(api->request_create(session, &input, &rc, &request, &error));
    IFN_EXAMPLE_TRY(api->request_run(request, ifn_generate_local_output, NULL, &error));
    failed = 0;
done:
#define IFN_EXAMPLE_CLOSE(call)                                                                    \
    do {                                                                                           \
        ifn_status code = (call);                                                                  \
        if (code) {                                                                                \
            fprintf(stderr, "Cleanup: %s (stage %u, provider %u, code %lld): %.*s\n",              \
                    ifn_status_string(code), error.stage, error.provider,                          \
                    (long long)error.provider_code, (int)error.message_bytes, error.message);      \
            failed = 1;                                                                            \
        }                                                                                          \
    } while (0)
    IFN_EXAMPLE_CLOSE(api->request_close(&request, &error));
    IFN_EXAMPLE_CLOSE(api->session_close(&session, &error));
    IFN_EXAMPLE_CLOSE(api->model_close(&model, &error));
    return failed;
}
