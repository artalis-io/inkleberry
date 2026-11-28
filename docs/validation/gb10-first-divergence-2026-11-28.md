# GB10 first-divergence analysis — 28 November 2026

## Executed historical analysis

The extractor validates all 28 case/profile groups, one warm-up and five measured
repetitions per group on each device, duplicate/missing rows, token bounds, output
counts, and within-device repeatability. It reproduces 17 matching and 11 divergent
groups from the 26 November sweeps. Nine differences are greedy/speculative outcomes;
two are sampled outcomes. Sampling differences are not greedy argmax disagreements.

Indices below are zero-based generated-token positions, excluding prompt tokens.

| Case | Profile | First difference | RTX token | GB10 token |
| --- | --- | ---: | ---: | ---: |
| 4 | adaptive | 5 | 151644 | 149878 |
| 4 | ar | 5 | 151644 | 149878 |
| 4 | fixed | 5 | 151644 | 149878 |
| 5 | adaptive | 74 | 174520 | 200010 |
| 5 | ar | 36 | 154573 | 153667 |
| 5 | fixed | 74 | 174520 | 200010 |
| 5 | sampled | 74 | 174520 | 170289 |
| 6 | adaptive | 43 | 150925 | 150409 |
| 6 | ar | 42 | 149917 | 150397 |
| 6 | fixed | 42 | 149917 | 150397 |
| 6 | sampled | 4 | 151470 | 149884 |

Cases 4, 5 and 6 are code editing, JSON extraction and long-context retrieval.
The [machine-readable record](gb10-first-divergence-2026-11-28.json) includes exact
source artifact checksums and the shared generated tokens preceding each difference.
It does not include prompt token IDs: historical equal prompt counts alone do not
prove prompt-byte identity. Reconstruct and verify them before numerical comparison.

## Next native experiment

1. Use the same pinned model/tokenizer bytes on RTX and GB10. Record source,
   executable, toolkit, driver and runtime identities.
2. Reconstruct the original prompt and verify token count and exact IDs. Capture
   the same prompt plus shared generated prefix at each divergence boundary.
3. First reproduce each original path (AR, fixed speculation, adaptive/lookup,
   sampled) with the original prefill/KV/graph settings. Record any failure to
   reproduce the historical boundary on newer source. Do not silently substitute
   eager singleton prefill for the original batched generation path.
4. Capture full finite post-softcap FP32 logits at the boundary without changing
   kernels, KV precision, sampling, or goldens. Compare top tokens, margins, ranks,
   absolute/relative differences and softmax probability differences. Full-vector
   comparison is necessary; matching top-1 alone does not establish distributions.
5. Verify an independent reference supports `Inkling2LoopForConditionalGeneration`
   (`inkling2_loop`) and the exact ModelOpt NVFP4 checkpoint representation. A
   different quantization or merely the same engine on another device is not an
   equivalent trusted reference. An unavailable equivalent must remain explicit.
6. Separate near-tie path divergence from material distribution changes using
   measured logits and reference results; do not infer quality from token mismatch
   counts. Follow with held-out task quality.

Public `ifn_session_logits` is available, but feeding the whole prompt in batches
of one/eight is not proof of matching the original prefill path. Original-path
capture/reproduction is a prerequisite for interpreting such diagnostic outputs.

## Current availability and scope

The qualification stack is merged into `main` at `c18401e`. Main post-merge CI
is tracked separately from the passing pre-merge suites. Both `spark-7468` and
`spark-3eb0` have active `VLLM::EngineCore` processes. No inference service has
been stopped. Permission to temporarily pause/restore the named Spark service
has been requested before GPU capture. No new GB10 logit or trusted-reference
results are claimed. Existing kernel and golden files are unchanged.

Reproduce historical extraction:

```sh
python3 tools/first_divergence.py /external/rtx-engine.jsonl \
  /external/gb10-engine.jsonl --out /external/new-first-divergence.json
python3 test/first_divergence.py
```
