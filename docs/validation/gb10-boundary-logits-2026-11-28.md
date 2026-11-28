# RTX / GB10 boundary-logit capture — 28 November 2026

## Measured result

All 11 historical first-divergence cases were reproduced on both devices at
capture source `4de7f2a`. Each case ran once without capture and once with capture;
the complete generated token sequences were identical and matched that device’s
historical benchmark. Exact prompt token IDs and the shared generated prefix
match across devices. Both used identical model/tokenizer/MTP file hashes.

All nine greedy/speculative differences exchange the top two candidates. This
is consistent with close competing choices, but does not establish that the
whole distribution is equivalent or either device is closer to a reference.
Full-vocabulary temperature-1 total variation is about 0.0187–0.1523 across those
cases. It would be misleading to label every observed difference negligible.

| Case | Profile | Output index | RTX batch / row | GB10 batch / row | RTX top-2 gap | GB10 top-2 gap | TV at T=1 |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| 4 | adaptive | 5 | 2 / 0 | 2 / 0 | 0.010178 | 0.037029 | 0.043965 |
| 4 | ar | 5 | 1 / 0 | 1 / 0 | 0.020830 | 0.037745 | 0.043396 |
| 4 | fixed | 5 | 8 / 0 | 8 / 0 | 0.010178 | 0.037029 | 0.043965 |
| 5 | adaptive | 74 | 4 / 1 | 2 / 1 | 0.007156 | 0.067648 | 0.018726 |
| 5 | ar | 36 | 1 / 0 | 1 / 0 | 0.104576 | 0.193367 | 0.098679 |
| 5 | fixed | 74 | 8 / 1 | 8 / 1 | 0.007156 | 0.067648 | 0.018726 |
| 5 | sampled | 74 | 1 / 0 | 1 / 0 | 0.765623 | 0.653614 | 0.047817 |
| 6 | adaptive | 43 | 8 / 5 | 8 / 5 | 0.080139 | 0.324389 | 0.097552 |
| 6 | ar | 42 | 1 / 0 | 1 / 0 | 0.033810 | 0.126509 | 0.146499 |
| 6 | fixed | 42 | 8 / 6 | 8 / 6 | 0.163067 | 0.162134 | 0.152282 |
| 6 | sampled | 4 | 1 / 0 | 1 / 0 | 1.621847 | 2.434050 | 0.146612 |

Cases 4/5/6 are code editing, JSON extraction and long-context retrieval. Output
indices are zero-based. The two sampled cases retain the same top-1 on both devices;
different sampled tokens are not greedy argmax failures. T=1 probabilities/TV/KL
are diagnostic distributions, not the sampled profile’s temperature-0.7/top-p-0.95
delivery distribution. The adaptive JSON case uses widths 4 versus 2; its result
includes that policy-path difference. Toolkits, vendor libraries and CPUs also
differ, so these are complete-platform comparisons, not isolated silicon causation.

## Capture method and checks

The shared benchmark fixture preserves the original prompts. Execution uses the
original generation API, NVFP4+MTP weights, FP8 KV, graphs, the four original decode
profiles, seed 42, 32K allocation and a 96-token budget. Actual longest prompt is
8,844 tokens. No engine/kernel numerical bodies, sampling policy or goldens change.

Greedy generation deliberately skips full-logit stores. At the selected callback,
the test synchronizes its stream, saves the original target final-normalized hidden
states, and replays only the existing LM-head kernel at the original batch width
into separate output and scratch buffers with full-logit stores enabled. It does
not replace original prefill or force the whole prompt through singleton batches.
The replay adds observation work and is not a performance benchmark.

The client records selected verification rows, prompt IDs, full finite FP32 logits
(200,058 values), final hidden states and the complete generated output. It checks
the shared prefix while running and compares entire output with/without capture.
The independent comparator additionally checks historical sequence reproduction,
exact cross-device prompt identity, shapes, finite values and agreement between
each actual greedy selection and the replayed top logit. It records full-vector
differences, top-ten values, winner margins and candidate ranks/probabilities.

Local `make check` and lint pass for the capture client; four comparator adversarial
tests cover missing/duplicate records, changed prefixes/outputs, malformed dimensions,
truncation, NaN/Inf, wrong greedy row/winner and known probability identities.

## Reproduction

```sh
make -j8 CUDA_ARCH=120a .build/logit-boundary  # 121a on GB10
mkdir /external/new-capture
INKLEBERRY_TEST_GPU_UUID=GPU-APPROVED-ID \
  python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 \
  .build/logit-boundary /external/model /external/requests.txt /external/new-capture \
  > /external/new-capture/capture.jsonl
python3 test/boundary_logits.py
```

Each of the 11 request records contains case number, profile number (AR=0,
fixed=1, adaptive=2, sampled=3), zero-based output index, prompt-token count, then
the shared generated prefix IDs. Generate these from the checked-in historical
first-divergence JSON; preserve `requests.txt` and its recorded checksum.

```sh
python3 tools/compare_boundary_logits.py /external/rtx /external/gb10 \
  --left-benchmark /external/rtx-engine.jsonl \
  --right-benchmark /external/gb10-engine.jsonl \
  --divergences docs/validation/gb10-first-divergence-2026-11-28.json \
  --out /external/new-comparison.json
```

[Machine-readable evidence](gb10-boundary-logits-2026-11-28.json) records runtime
metadata, source/client/model hashes, per-case comparisons and artifact checksums.
Raw binary vectors and logs stay outside Git at
`/Users/mark/artalis.io/data/inkleberry-boundary-2026-11-28/` and isolated
`/data/scratch/inkleberry-boundary-2026-11-28` directories on both hosts.

## Remaining qualification

An independent full-model reference with equivalent weight representation and
verified matching inputs has not run. These measurements establish what differs,
not which implementation is more accurate. Keep kernels and goldens unchanged;
compare reference logits before numerical changes, then run held-out task quality.
Do not substitute repeated stability or output plausibility for that reference.

## Service restoration

Only the approved `qwen3_8-flash-next-fp4-local` vLLM service on `spark-7468`
was paused. It was restarted after capture and independently verified healthy:
HTTP `/health` 200, API model identity matched, prior boot enablement (disabled)
preserved, and all eight unrelated Cursus container IDs unchanged. Cursus
`http://127.0.0.1:8080/health/ready` returned 200.

The initial readiness waiter treated the warming health command’s exit status 12
as fatal after successfully starting the service. A bounded replacement waiter
accepted the structured warming response and completed health/state verification;
no second restart or service-configuration change was made. The original failure
and recovery are retained in the external lifecycle and supervisor logs.
