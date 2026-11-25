/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Pinned adaptive graph scheduling: width switch, lookup verification and
 * next-draft overlap. Every exit drains work before reset or resource release. */
#include "engine_internal.h"

static void capture_finish(sq_session *s, void **graph) {
    sq_ar_check(s, squid_cuda_graph_end_instantiate(s->stream, graph),
                "finish adaptive component capture");
}

static void prepare(sq_session *s, float threshold) {
    if (!s->draft_graph) {
        sq_ar_check(s, squid_cuda_graph_begin(s->stream), "begin draft graph");
        if (s->status != SQ_OK)
            return;
        sq_spec_draft(s, 7, s->d_conf);
        capture_finish(s, &s->draft_graph);
    }
    if (s->status != SQ_OK)
        return;
    if (!s->lookup_graph) {
        sq_ar_check(s, squid_cuda_graph_begin(s->stream), "begin lookup graph");
        if (s->status != SQ_OK)
            return;
        sq_spec_verify(s, 7);
        if (s->status == SQ_OK)
            sq_ar_check(s, squid_cuda_memcpy_d2h_async(s->h_count, s->d_num_acc, 4, s->stream),
                        "capture lookup count");
        if (s->status == SQ_OK)
            sq_ar_check(s, squid_cuda_memcpy_d2h_async(s->h_accepted, s->d_accepted, 32, s->stream),
                        "capture lookup output");
        capture_finish(s, &s->lookup_graph);
    }
    if (s->status != SQ_OK || s->adaptive_threshold == threshold)
        return;
    sq_ar_check(s, sq_adaptive_cleanup(s->adaptive_state), "retire old adaptive threshold");
    if (s->status != SQ_OK)
        return;
    s->adaptive_threshold = 0;
    sq_ar_check(s,
                sq_adaptive_prepare(s->adaptive_state, s->d_anchor, s->d_drafts, s->d_conf,
                                    threshold, s->d_tokens, s->d_width, s->d_accepted,
                                    s->h_accepted, s->d_num_acc, s->h_count, s->h_width, s->stream),
                "prepare conditional graph");
    const uint32_t widths[3] = {2, 4, 8};
    for (int i = 0; i < 3 && s->status == SQ_OK; i++) {
        sq_ar_check(s, sq_adaptive_branch_begin(s->adaptive_state, i, s->stream),
                    "begin conditional branch");
        if (s->status != SQ_OK)
            break;
        sq_spec_target(s, widths[i] - 1);
        sq_ar_check(s, sq_adaptive_branch_end(s->adaptive_state), "end conditional branch");
    }
    if (s->status == SQ_OK)
        sq_ar_check(s, sq_adaptive_instantiate(s->adaptive_state), "instantiate conditional graph");
    if (s->status == SQ_OK)
        s->adaptive_threshold = threshold;
}

static void draft(sq_session *s) {
    if (s->status != SQ_OK)
        return;
    /* Conservatively retain the in-flight marker if launch/sync fails. */
    s->draft_in_flight = true;
    int e = squid_cuda_graph_launch(s->draft_graph, s->stream);
    sq_ar_check(s, e, "launch overlapped draft");
    if (!e)
        s->decode_info.draft_launches++;
}

void sq_adaptive_step(sq_session *s, const sq_decode_options *options, bool lookup) {
    if (s->status != SQ_OK)
        return;
    prepare(s, options->spec_threshold);
    if (s->status != SQ_OK)
        return;
    if (lookup) {
        int32_t one = 1;
        const sq_weight_backend *b = &s->weights->backend;
        sq_ar_check(s, b->copy(b->context, s->d_draft_is_delta, &one, 4, 0, s->stream),
                    "set Dirac proposal");
        if (s->status == SQ_OK)
            sq_ar_check(
                s, b->copy(b->context, s->d_drafts, (int32_t *)s->pinned + 10, 28, 0, s->stream),
                "upload lookup proposal");
        if (s->status == SQ_OK)
            sq_ar_check(s, squid_cuda_graph_launch(s->lookup_graph, s->stream),
                        "verify lookup proposal");
        if (s->status == SQ_OK)
            sq_ar_check(s, b->sync(b->context, s->stream), "finish lookup verification");
        if (s->status == SQ_OK)
            s->draft_in_flight = false;
        return;
    }
    if (!s->draft_in_flight)
        draft(s);
    if (s->status == SQ_OK)
        sq_ar_check(s, sq_adaptive_launch(s->adaptive_state, s->stream),
                    "launch adaptive verification");
    if (s->status == SQ_OK)
        sq_ar_check(s, squid_cuda_event_record(s->done_event, s->stream),
                    "record verified-output event");
    if (s->status == SQ_OK)
        draft(s);
    if (s->status == SQ_OK)
        sq_ar_check(s, squid_cuda_event_sync(s->done_event), "await verified outputs");
}

void sq_adaptive_drain(sq_session *s) {
    if (!s->draft_in_flight)
        return;
    int e = s->weights->backend.sync(s->weights->backend.context, s->stream);
    sq_ar_check(s, e, "drain overlapped draft");
    if (!e)
        s->draft_in_flight = false;
}
