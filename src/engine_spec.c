/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 tensorfish; Copyright (c) 2026 Artalis
 * Fixed-width DSpark: same draft/assemble/target/verify order as pinned Zig.
 * Commit is serialized on the owned stream; no work remains overlapped on return. */
#include "engine_internal.h"

void ifn_spec_commit(ifn_session *s, const float *taps, int32_t batch, const int32_t *position) {
    if (s->use_mtp && s->status == IFN_OK)
        squid_launch_dspark_commit(&s->model.dspark, taps, s->ds_k, s->ds_v, s->ds_ks, s->ds_vs,
                                   s->ds_cap, (uint8_t)s->config.kv_format, s->ds_f_ring, s->ds_mix,
                                   s->ds_kv, batch, position, s->stream);
}

void ifn_spec_draft(ifn_session *s, uint32_t drafts, float *confidence) {
    const void *ks[3] = {s->ds_ks[0], s->ds_ks[1], s->ds_ks[2]};
    const void *vs[3] = {s->ds_vs[0], s->ds_vs[1], s->ds_vs[2]};
    squid_launch_dspark_draft(s->model.embed, s->model.embed_norm, &s->model.dspark, s->d_anchor,
                              s->d_pos, (int32_t)drafts, s->ds_k, s->ds_v, ks, vs, s->ds_cap,
                              (uint8_t)s->config.kv_format, s->split_buf, s->ds_f_ring, s->ds_ws,
                              s->d_drafts, confidence, s->d_sample_cfg, s->d_draft_probs,
                              s->d_draft_q_sel, s->d_draft_is_delta, s->stream);
    ifn_ar_check(s, ifn_cuda_launch_status(), "DSpark draft launch");
}

void ifn_spec_verify(ifn_session *s, uint32_t drafts) {
    squid_launch_assemble_verify_tokens(s->d_anchor, s->d_drafts, s->d_tokens, (int32_t)drafts,
                                        s->stream);
    ifn_spec_target(s, drafts);
}

void ifn_spec_target(ifn_session *s, uint32_t drafts) {
    s->verifying = true;
    ifn_ar_forward(s, (int32_t)drafts + 1, true);
    s->verifying = false;
    if (s->status == IFN_OK)
        squid_launch_spec_verify(s->d_next_tokens, s->d_drafts, (int32_t)drafts + 1, s->d_accepted,
                                 s->d_num_acc, s->d_pos, s->d_anchor, s->stream);
    ifn_ar_check(s, ifn_cuda_launch_status(), "DSpark draft/verify launch");
}

void ifn_spec_step(ifn_session *s, uint32_t drafts) {
    if (s->status != IFN_OK)
        return;
    if (!s->config.use_graphs) {
        ifn_spec_draft(s, drafts, NULL);
        if (s->status == IFN_OK)
            ifn_spec_verify(s, drafts);
        return;
    }
    if (!s->spec_graph[drafts]) {
        ifn_ar_check(s, squid_cuda_graph_begin(s->stream), "begin DSpark graph");
        if (s->status != IFN_OK)
            return;
        ifn_spec_draft(s, drafts, NULL);
        if (s->status == IFN_OK)
            ifn_spec_verify(s, drafts);
        ifn_ar_check(s, squid_cuda_graph_end_instantiate(s->stream, &s->spec_graph[drafts]),
                     "instantiate DSpark graph");
    }
    if (s->status == IFN_OK)
        ifn_ar_check(s, squid_cuda_graph_launch(s->spec_graph[drafts], s->stream),
                     "launch DSpark graph");
}
