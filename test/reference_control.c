/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis
 * Source-weight/BF16-KV conditional-prefix control; never changes production defaults. */
#include "ifn_engine.h"
#include "sha256.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define IFN_REFERENCE_TRY(x)                                                                       \
    do {                                                                                           \
        if ((x)) {                                                                                 \
            failed = 1;                                                                            \
            goto done;                                                                             \
        }                                                                                          \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 4 || strlen(argv[1]) > 1500 || strlen(argv[3]) > 1500)
        return 2;
    int failed = 0;
    ifn_inkling_checkpoint *checkpoint = NULL;
    ifn_gpu_weights *weights = NULL;
    ifn_session *session = NULL;
    FILE *input = NULL;
    int32_t *tokens = malloc(32768 * sizeof(int32_t));
    float *logits = malloc(200058 * sizeof(float)), *baseline = malloc(200058 * sizeof(float));
    char path[1800];
    unsigned seen = 0;
    IFN_REFERENCE_TRY(!tokens || !logits || !baseline);
    snprintf(path, sizeof path, "%s/model-00001-of-00001.safetensors", argv[1]);
    IFN_REFERENCE_TRY(ifn_inkling_checkpoint_open(path, NULL, NULL, &checkpoint));
    ifn_weight_config wc = IFN_WEIGHT_CONFIG_INIT;
    wc.mode = 0;
    wc.with_mtp = 0;
    IFN_REFERENCE_TRY(ifn_gpu_weights_create(&wc, NULL, &weights));
    IFN_REFERENCE_TRY(ifn_gpu_weights_load(weights, checkpoint));
    ifn_inkling_checkpoint_close(checkpoint);
    checkpoint = NULL;
    ifn_session_config sc = IFN_SESSION_CONFIG_INIT;
    sc.context_tokens = 32768;
    sc.kv_format = IFN_KV_BF16;
    sc.use_graphs = 0;
    IFN_REFERENCE_TRY(ifn_session_create(weights, &sc, NULL, &session));
    input = fopen(argv[2], "r");
    IFN_REFERENCE_TRY(!input);
    unsigned which, prompt, total;
    while (fscanf(input, "%u %u %u", &which, &prompt, &total) == 3) {
        IFN_REFERENCE_TRY(which < 4 || which > 6 || !prompt || prompt >= total || total >= 32768 ||
                          total - prompt > 96 || (seen & (1u << (which - 4))));
        seen |= 1u << (which - 4);
        for (unsigned i = 0; i < total; i++)
            IFN_REFERENCE_TRY(fscanf(input, "%d", tokens + i) != 1 || tokens[i] < 0 ||
                              tokens[i] >= 200058);
        ifn_sha256 hash;
        unsigned char sum[32];
        char input_sha256[65];
        ifn_sha256_init(&hash);
        for (unsigned i = 0; i < total; i++) {
            uint32_t value = (uint32_t)tokens[i];
            unsigned char le[4] = {(unsigned char)value, (unsigned char)(value >> 8),
                                   (unsigned char)(value >> 16), (unsigned char)(value >> 24)};
            IFN_REFERENCE_TRY(!ifn_sha256_update(&hash, le, sizeof le));
        }
        ifn_sha256_final(&hash, sum);
        ifn_sha256_hex(sum, input_sha256);
        for (unsigned run = 0; run < 2; run++) {
            IFN_REFERENCE_TRY(ifn_session_reset(session));
            ifn_decode_options options = IFN_DECODE_OPTIONS_INIT;
            options.seed = 42;
            int32_t ignored;
            ifn_decode_stats stats;
            /* Only prompt tokens are consumed; the one sampled output is not fed.
             * Subsequent logits calls append the exact supplied preceding outputs. */
            IFN_REFERENCE_TRY(ifn_session_generate_ex(session, tokens, prompt, 1, &options,
                                                      &ignored, 1, NULL, NULL, &stats));
            for (unsigned i = prompt; i < total; i++)
                IFN_REFERENCE_TRY(ifn_session_logits(session, tokens + i, 1, 0, logits, 200058));
            unsigned top = 0;
            for (unsigned i = 0; i < 200058; i++) {
                IFN_REFERENCE_TRY(!isfinite(logits[i]));
                if (logits[i] > logits[top])
                    top = i;
            }
            if (!run)
                memcpy(baseline, logits, 200058 * sizeof(float));
            else
                IFN_REFERENCE_TRY(memcmp(baseline, logits, 200058 * sizeof(float)));
            printf("{\"case\":%u,\"run\":%u,\"top1\":%u,\"prompt_tokens\":%u,\"total_tokens\":%u,"
                   "\"input_sha256\":\"%s\"}\n",
                   which, run, top, prompt, total, input_sha256);
            fflush(stdout);
        }
        snprintf(path, sizeof path, "%s/case-%u.logits.f32", argv[3], which);
        FILE *out = fopen(path, "wbx");
        IFN_REFERENCE_TRY(!out);
        int write_failed = fwrite(logits, sizeof(float), 200058, out) != 200058;
        if (fclose(out))
            write_failed = 1;
        IFN_REFERENCE_TRY(write_failed);
    }
    IFN_REFERENCE_TRY(!feof(input) || seen != 7);
done:
    if (failed)
        fprintf(stderr, "reference control failed: %s\n", ifn_last_error());
    if (input && fclose(input))
        failed = 1;
    free(tokens);
    free(logits);
    free(baseline);
    if (ifn_session_destroy(&session))
        failed = 1;
    if (ifn_gpu_weights_destroy(&weights))
        failed = 1;
    ifn_inkling_checkpoint_close(checkpoint);
    return failed;
}
