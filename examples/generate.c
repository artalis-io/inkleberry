/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "sq.h"
#include <stdio.h>
#include <string.h>

static uint32_t output(void *context, const sq_infer_event_v1 *event) {
    (void)context;
    if (event->kind == SQ_EVENT_CONTENT && event->text.bytes &&
        fwrite(event->text.data, 1, (size_t)event->text.bytes, stdout) != event->text.bytes)
        return SQ_CALLBACK_CANCEL;
    return SQ_CALLBACK_CONTINUE;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fputs("usage: generate MODEL_DIR PROMPT\n", stderr);
        return 2;
    }
    const void *raw = NULL;
    if (sq_infer_get_api(1, 0, &raw) != SQ_OK)
        return 1;
    const sq_infer_api_v1 *api = raw;
    sq_infer_model *model = NULL;
    sq_infer_session *session = NULL;
    sq_infer_request *request = NULL;
    sq_infer_model_config_v1 mc;
    sq_infer_session_config_v1 sc;
    sq_infer_request_config_v1 rc;
    sq_infer_source_v1 source;
    sq_infer_input_v1 input;
    sq_infer_message_v1 message;
    SQ_INFER_INIT(mc);
    SQ_INFER_INIT(sc);
    SQ_INFER_INIT(rc);
    SQ_INFER_INIT(source);
    SQ_INFER_INIT(input);
    SQ_INFER_INIT(message);
    int failed = 1;
    if (api->defaults(SQ_DEFAULT_MODEL, &mc, NULL) ||
        api->defaults(SQ_DEFAULT_SESSION, &sc, NULL) ||
        api->defaults(SQ_DEFAULT_REQUEST, &rc, NULL))
        goto done;
    source.directory = (sq_infer_bytes){argv[1], strlen(argv[1])};
    source.format_id = (sq_infer_bytes){"inkling-safetensors", sizeof("inkling-safetensors") - 1};
    message.role = SQ_ROLE_USER;
    message.content_present = 1;
    message.content = (sq_infer_bytes){argv[2], strlen(argv[2])};
    input.kind = SQ_INPUT_CHAT;
    input.messages = (sq_infer_array){&message, 1, sizeof message, 0};
    rc.max_output_tokens = 128;
    if (api->model_create(&mc, &model, NULL) || api->model_load(model, &source, NULL) ||
        api->session_create(model, &sc, &session, NULL) || api->session_prepare(session, NULL) ||
        api->request_create(session, &input, &rc, &request, NULL) ||
        api->request_run(request, output, NULL, NULL))
        goto done;
    failed = 0;
done:
    if (api->request_close(&request, NULL))
        failed = 1;
    if (api->session_close(&session, NULL))
        failed = 1;
    if (api->model_close(&model, NULL))
        failed = 1;
    if (failed)
        fputs("Generation or cleanup failed\n", stderr);
    return failed;
}
