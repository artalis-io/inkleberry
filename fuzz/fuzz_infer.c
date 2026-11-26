/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Host-only ABI validation; execution is the explicit test mock. */
#include "infer_internal.h"

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size) {
    if (size < 8)
        return 0;
    ifn_infer_model_config_v1 mc;
    ifn_infer_session_config_v1 sc;
    ifn_infer_request_config_v1 rc;
    IFN_INFER_INIT(mc);
    IFN_INFER_INIT(sc);
    IFN_INFER_INIT(rc);
    (void)si_defaults(IFN_DEFAULT_MODEL, &mc, NULL);
    (void)si_defaults(IFN_DEFAULT_SESSION, &sc, NULL);
    (void)si_defaults(IFN_DEFAULT_REQUEST, &rc, NULL);
    sc.context_tokens = 512;
    rc.output_kind = IFN_OUTPUT_RAW_TOKENS;
    rc.max_output_tokens = data[0];
    rc.max_event_bytes = data[1];
    rc.top_k = (int32_t)data[2] - 64;
    rc.top_p = (float)data[3] / 255.0f;
    rc.temperature = (float)data[4] / 16.0f;
    rc.output_flags = data[5];
    rc.required_features = (uint64_t)data[6] << 60;
    rc.hdr.flags = data[7];
    ifn_infer_model *m = NULL;
    ifn_infer_session *s = NULL;
    ifn_infer_request *r = NULL;
    if (si_model_create(&mc, &m, NULL))
        return 0;
    if (ifn_gpu_weights_create(&m->weights_config, &m->allocator, &m->weights))
        goto done;
    m->state = IFN_HANDLE_READY;
    if (si_session_create(m, &sc, &s, NULL) || si_session_prepare(s, NULL))
        goto done;
    int32_t tokens[32];
    size_t count = size - 8 > 32 ? 32 : size - 8;
    for (size_t i = 0; i < count; i++)
        tokens[i] = data[8 + i];
    ifn_infer_input_v1 input;
    IFN_INFER_INIT(input);
    input.kind = IFN_INPUT_TOKENS;
    input.tokens = tokens;
    input.token_count = count;
    (void)si_request_create(s, &input, &rc, &r, NULL);
    (void)si_request_close(&r, NULL);
done:
    (void)si_session_close(&s, NULL);
    (void)si_model_close(&m, NULL);
    return 0;
}
