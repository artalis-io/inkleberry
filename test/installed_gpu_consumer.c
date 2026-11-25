/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include <sq_engine.h>

int main(void) {
    sq_session *session = NULL;
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
