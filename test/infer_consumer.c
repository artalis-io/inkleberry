/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Public headers only; usable by a generic server, C++ or a proprietary host. */
#include "ifn_infer.h"
#include <stdio.h>

int main(void) {
    const void *raw = NULL;
    if (ifn_infer_get_api(1, 0, &raw) != IFN_OK)
        return 1;
    const ifn_infer_api_v1 *api = (const ifn_infer_api_v1 *)raw;
    if (api->struct_size < sizeof *api)
        return 2;
    ifn_infer_backend_info_v1 info;
    IFN_INFER_INIT(info);
    if (api->backend_info(&info, NULL) != IFN_OK || !info.max_active_runs)
        return 3;
    ifn_infer_model_config_v1 config;
    IFN_INFER_INIT(config);
    if (api->defaults(IFN_DEFAULT_MODEL, &config, NULL) != IFN_OK)
        return 4;
    ifn_infer_model *model = NULL;
    if (api->model_create(&config, &model, NULL) != IFN_OK)
        return 5;
    ifn_infer_model_info_v1 metadata;
    IFN_INFER_INIT(metadata);
    if (api->model_info(model, &metadata, NULL) != IFN_OK)
        return 6;
    if (api->model_close(&model, NULL) != IFN_OK || model)
        return 7;
    printf("public consumer: %.*s, maximum active runs=%u\n", (int)info.backend_id.bytes,
           (const char *)info.backend_id.data, info.max_active_runs);
    return 0;
}
