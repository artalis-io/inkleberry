/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include <sq_engine.h>

int main(void) {
    sq_session *session = NULL;
    sq_decode_options options = SQ_DECODE_OPTIONS_INIT;
    sq_prefix_info info = SQ_PREFIX_INFO_INIT;
    if (sq_session_prefix_cache(NULL, 1) != SQ_INVALID_ARGUMENT ||
        sq_session_prefix_info(NULL, &info) != SQ_INVALID_ARGUMENT)
        return 1;
    sq_decode_stats stats;
    if (sq_session_generate_ex(NULL, NULL, 0, 0, &options, NULL, 0, NULL, NULL, &stats) !=
        SQ_INVALID_ARGUMENT)
        return 1;
    if (sq_session_create(NULL, NULL, NULL, &session) != SQ_INVALID_ARGUMENT || session ||
        sq_session_destroy(&session) != SQ_OK)
        return 1;
    sq_weight_config config = SQ_WEIGHT_CONFIG_INIT;
    sq_gpu_weights *weights = NULL;
    sq_weight_plan plan;
    if (sq_gpu_weights_create(&config, NULL, &weights) != SQ_OK ||
        sq_gpu_weights_get_plan(weights, &plan) != SQ_OK || !plan.arena_bytes)
        return 1;
    return sq_gpu_weights_destroy(&weights) != SQ_OK || weights != NULL;
}
