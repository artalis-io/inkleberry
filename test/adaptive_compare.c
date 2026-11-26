/* SPDX-License-Identifier: Apache-2.0; Copyright (c) 2026 Artalis */
#include "decode_compare.c"
static size_t adaptive_requests, lookup_hits, branch_counts[3];
static uint32_t adaptive_limit = 32;
static int copy_fixture, requested_lookup;
static int32_t adaptive_output[512];

int ifn_adaptive_start(const char *dir, uint32_t mode, uint32_t kv, uint32_t n, uint32_t extra) {
    return ifn_decode_start(dir, mode, kv, 1, n, 7, extra);
}

uint32_t ifn_adaptive_token_count(void) {
    return (uint32_t)token_count;
}

int ifn_adaptive_copy_fixture(const char *directory) {
    const char *text =
        "Copy the passage between START and END exactly, with no explanation.\nSTART\n"
        "On Tuesday morning, a blue delivery van crossed the wooden bridge near the old library. "
        "Its driver carried twelve green boxes, three silver lamps, and a small yellow notebook. "
        "The village clock showed nine fifteen when the baker opened the eastern gate. "
        "Beyond the orchard, seven white birds rested beside the quiet river. "
        "A handwritten sign directed visitors past the red barn toward the stone fountain. "
        "Every package had a clear label and every recipient checked the list twice. "
        "At noon the driver returned along the same road, stopping briefly to photograph the "
        "mountains. "
        "The notebook remained safely inside the front compartment until the journey ended.\nEND";
    char path[1200], rendered[8192];
    size_t bytes = 0, count = 0;
    ifn_tokenizer *t = NULL;
    ifn_tokenizer_scratch *scratch = NULL;
    (void)snprintf(path, sizeof path, "%s/tokenizer.json", directory);
    ifn_status rc = ifn_tokenizer_open(path, NULL, &t);
    if (rc == IFN_OK)
        rc = ifn_tokenizer_scratch_create(NULL, &scratch);
    if (rc == IFN_OK)
        rc = ifn_render_user_prompt(text, strlen(text), IFN_THINK_NONE, rendered, sizeof rendered,
                                    &bytes);
    if (rc == IFN_OK)
        rc = ifn_tokenizer_encode(t, rendered, bytes, 1, scratch, tokens, token_count, &count);
    ifn_tokenizer_scratch_destroy(scratch);
    ifn_tokenizer_close(t);
    if (rc != IFN_OK || !count)
        return 1;
    token_count = count;
    adaptive_limit = 256;
    copy_fixture = 1;
    return 0;
}

int ifn_adaptive_check(float temperature, float top_p, int32_t k, uint64_t seed, float threshold,
                       uint32_t lookup, const int32_t *expected, uint32_t count, uint32_t steps,
                       uint32_t total, uint32_t accepted, uint32_t lookups) {
    ifn_decode_options options = IFN_DECODE_OPTIONS_INIT;
    options.temperature = temperature;
    options.top_p = top_p;
    options.top_k = k;
    options.seed = seed;
    options.draft_tokens = 7;
    options.adaptive = 1;
    options.prompt_lookup = lookup;
    options.spec_threshold = threshold;
    ifn_decode_stats stats;
    ifn_decode_info info = IFN_DECODE_INFO_INIT;
    ifn_status rc = ifn_session_generate_ex(session, tokens, token_count, adaptive_limit, &options,
                                            adaptive_output, 512, NULL, NULL, &stats);
    if (rc != IFN_OK || ifn_session_decode_info(session, &info) != IFN_OK ||
        stats.generation.generated_tokens != count ||
        memcmp(adaptive_output, expected, (size_t)count * 4) || stats.speculative_steps != steps ||
        stats.total_drafts != total || stats.accepted_drafts != accepted ||
        info.lookup_steps != lookups || info.width2 + info.width4 + info.width8 != steps ||
        info.width2 + 3 * info.width4 + 7 * info.width8 != total ||
        info.lookup_steps > info.width8) {
        fprintf(stderr,
                "adaptive mismatch T=%g threshold=%g lookup=%u C=%u/%u/%u/%u Zig=%u/%u/%u/%u rc=%u "
                "%s\n",
                (double)temperature, (double)threshold, lookup, stats.generation.generated_tokens,
                stats.speculative_steps, stats.total_drafts, info.lookup_steps, count, steps, total,
                lookups, rc, ifn_last_error());
        return 1;
    }
    requested_lookup = (int)lookup;
    adaptive_requests++;
    lookup_hits += info.lookup_steps;
    branch_counts[0] += info.width2;
    branch_counts[1] += info.width4;
    branch_counts[2] += info.width8;
    return 0;
}

int ifn_adaptive_finish(uint32_t success) {
    if (success && copy_fixture && requested_lookup && !lookup_hits) {
        fputs("copy fixture did not exercise lookup\n", stderr);
        success = 0;
    }
    int rc = ifn_decode_finish(0);
    if (copy_fixture && requested_lookup && !lookup_hits)
        rc = 1;
    if (!rc && success)
        printf("{\"adaptive_parity\":true,\"prompt_tokens\":%zu,\"requests\":%zu,\"lookup_steps\":%"
               "zu,\"width2\":%zu,"
               "\"width4\":%zu,\"width8\":%zu}\n",
               token_count, adaptive_requests, lookup_hits, branch_counts[0], branch_counts[1],
               branch_counts[2]);
    return rc;
}
