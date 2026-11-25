/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include <sq_gpu.h>

int main(void) {
    sq_weight_config config = SQ_WEIGHT_CONFIG_INIT;
    sq_gpu_weights *weights = NULL;
    sq_weight_plan plan;
    if (sq_gpu_weights_create(&config, NULL, &weights) != SQ_OK ||
        sq_gpu_weights_get_plan(weights, &plan) != SQ_OK || !plan.arena_bytes)
        return 1;
    return sq_gpu_weights_destroy(&weights) != SQ_OK || weights != NULL;
}
