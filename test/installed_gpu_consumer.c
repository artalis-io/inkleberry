/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include <ifn_engine.h>

int main(void) {
    ifn_session *session = NULL;
    ifn_decode_options options = IFN_DECODE_OPTIONS_INIT;
    ifn_prefix_info info = IFN_PREFIX_INFO_INIT;
    if (ifn_session_prefix_cache(NULL, 1) != IFN_INVALID_ARGUMENT ||
        ifn_session_prefix_info(NULL, &info) != IFN_INVALID_ARGUMENT)
        return 1;
    ifn_decode_stats stats;
    if (ifn_session_generate_ex(NULL, NULL, 0, 0, &options, NULL, 0, NULL, NULL, &stats) !=
        IFN_INVALID_ARGUMENT)
        return 1;
    if (ifn_session_create(NULL, NULL, NULL, &session) != IFN_INVALID_ARGUMENT || session ||
        ifn_session_destroy(&session) != IFN_OK)
        return 1;
    ifn_weight_config config = IFN_WEIGHT_CONFIG_INIT;
    ifn_gpu_weights *weights = NULL;
    ifn_weight_plan plan;
    if (ifn_gpu_weights_create(&config, NULL, &weights) != IFN_OK ||
        ifn_gpu_weights_get_plan(weights, &plan) != IFN_OK || !plan.arena_bytes)
        return 1;
    return ifn_gpu_weights_destroy(&weights) != IFN_OK || weights != NULL;
}
