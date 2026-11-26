/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "ifn.h"
#include "ifn_infer.h"
#include <stdio.h>
#define IFN_ABI_SIZE(type) printf("\"%s\":%zu,", #type, sizeof(type))
#define IFN_ABI_OFFSET(type, member)                                                               \
    printf("\"%s.%s\":%zu,", #type, #member, offsetof(type, member))

int main(void) {
    if (sizeof(void *) != 8 || sizeof(size_t) != 8)
        return 2;
    putchar('{');
    IFN_ABI_SIZE(ifn_options);
    IFN_ABI_SIZE(ifn_tensor_info);
    IFN_ABI_SIZE(ifn_infer_bytes);
    IFN_ABI_SIZE(ifn_infer_array);
    IFN_ABI_SIZE(ifn_infer_error_v1);
    IFN_ABI_SIZE(ifn_infer_model_config_v1);
    IFN_ABI_SIZE(ifn_infer_source_v1);
    IFN_ABI_SIZE(ifn_infer_session_config_v1);
    IFN_ABI_SIZE(ifn_infer_request_config_v1);
    IFN_ABI_SIZE(ifn_infer_input_v1);
    IFN_ABI_SIZE(ifn_infer_result_v1);
    IFN_ABI_SIZE(ifn_infer_api_v1);
    IFN_ABI_OFFSET(ifn_infer_api_v1, request_run);
    IFN_ABI_OFFSET(ifn_infer_api_v1, request_cancel);
    IFN_ABI_OFFSET(ifn_infer_error_v1, provider_code);
    IFN_ABI_OFFSET(ifn_infer_model_config_v1, extensions);
    printf("\"IFN_OK\":%u,\"IFN_GPU_ERROR\":%u,\"IFN_CANCELLED\":%u,\"IFN_INFER_ABI_MISMATCH\":%u,"
           "\"IFN_INFER_NOT_READY\":%u}",
           IFN_OK, IFN_GPU_ERROR, IFN_CANCELLED, IFN_INFER_ABI_MISMATCH, IFN_INFER_NOT_READY);
    return 0;
}
