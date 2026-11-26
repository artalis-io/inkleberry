# Single-GPU RTX Blackwell and GB10 measurements — 26 November 2026

## Result and qualification boundary

Inkleberry Finn runs natively on both the RTX PRO 6000 Blackwell Max-Q and
GB10 after two narrow compatibility fixes: unified-memory idle preflight and
admitting compute capability 12.1 in the host. The original 12.0-only host
rejected GB10 with CUDA status 801 before loading weights.

Both engine and public inference-ABI sweeps completed successfully. All five
measured repetitions of every case were deterministic on each platform. RTX
matched **28/28** existing profile-specific golden sequences on each interface;
GB10 matched **17/28** on each. Engine and public-ABI outputs match each other
for all 28 cases on each platform. Original goldens remain unchanged.

These are performance measurements, not full numerical/quality qualification of
GB10. Its 11 divergent cases require further numerical investigation. Differences
were not traced to a specific cause; architecture, CUDA/cuBLAS version and host
math implementation differ. No claim of cross-platform quality equivalence is
made from spot-checking coherent outputs.

## Reproducible configuration

Measured source: `cf211032ed41df4de743576b1284cc929661a43f`.
Both platforms use the same source and native compilation, without emulation.
Architecture-specific binaries differ. No CUDA kernel math, launch geometry,
precision, sampling policy or model files were changed.

| Property | RTX platform | GB10 platform |
| --- | --- | --- |
| Host | `core` (`rtx-85b4`) | `spark-7468` |
| GPU count used | 1 of 2 installed cards | 1 |
| Device | RTX PRO 6000 Blackwell **Max-Q**, 96 GB, configured 300 W | NVIDIA GB10, unified system memory |
| Compute capability / SMs | 12.0 / 188 | 12.1 / 48 |
| CPU architecture | AMD64, Threadripper 9980X | ARM64, Cortex-X925/A725 |
| CUDA toolkit | 13.5.37 | 13.0.88 |
| NVIDIA driver | 595.104.02 | 580.178.04 |
| Native target | `CUDA_ARCH=120a` | `CUDA_ARCH=121a` |

Model: `thinkingmachines/Inkling-2-Mini`, pinned Hugging Face revision
`c9d028302cd0f035fdf8e7e91f089a4ce94d2195`. Both sets of model, MTP and
tokenizer bytes were checksum-verified against the existing manifest. Artifact
fingerprint: `2cfbc10c85ba0949cbaddba1fa1c76f1609a89dcffa160438888de595925b9f9`.

`test/native_benchmark.c` supplies seven workloads and four profiles:
autoregressive greedy (`ar`), fixed seven-draft speculative (`fixed`), adaptive
seven-draft plus prompt lookup (`adaptive`), and seeded non-speculative sampling
(`sampled`, temperature 0.7, top-p 0.95, seed 42). NVFP4 weights, MTP loaded,
FP8 KV, CUDA graphs enabled, 32,768-token context allocation, at most 96 generated
tokens. Each case has one excluded warm-up and five measured runs, resetting
KV/prefix state between requests. The retrieval case really prefills **8,844**
tokens; the other prompts have 24–36 tokens. Allocation of 32K does not establish
32K or million-token inference performance.

The comparison comprises 672 generations: 112 warm-ups and 560 measured runs,
across two devices, two interfaces and 28 workload/profile combinations.

## Decode throughput

Tokens/second, median (minimum–maximum) over five measured runs. These selected
cases have **identical prompt and generated tokens** across devices and all
produce 96 output tokens. Ratios describe these complete platforms, including
their differing CPUs, drivers and toolkit versions, not an isolated GPU factor.

| Workload | Profile | RTX tokens/s | GB10 tokens/s | RTX / GB10 |
| --- | --- | --- | --- | --- |
| Explanation | ar | 239.1 (239.0–239.1) | 70.9 (70.7–71.0) | 3.37× |
| Explanation | adaptive | 323.7 (323.7–323.9) | 80.7 (80.5–80.8) | 4.01× |
| Prose | ar | 239.9 (239.8–240.0) | 70.9 (70.5–71.1) | 3.38× |
| Prose | adaptive | 295.6 (295.6–295.8) | 74.9 (74.5–74.9) | 3.95× |
| Code generation | ar | 238.9 (238.8–238.9) | 70.8 (70.6–70.9) | 3.37× |
| Code generation | adaptive | 332.8 (332.7–332.9) | 83.4 (83.2–83.5) | 3.99× |

The metric is `(generated_tokens - 1) / (completion_time - first_token_time)`.
TTFT is measured to the first emitted token. Speculative callbacks may deliver
bursts; this is emitted-token throughput, not per-token kernel latency.
Model loading, tokenization and session creation precede the per-request timer.

The public inference ABI produces comparable results:

| Workload | RTX public API tokens/s | GB10 public API tokens/s |
| --- | --- | --- |
| Explanation, adaptive | 323.7 | 80.7 |
| Code generation, adaptive | 332.6 | 83.9 |

The complete JSON/CSV contains all 28 combinations, including fixed speculation
and sampling, plus TTFT, minimum, maximum and coefficient of variation. For
nonmatching output sequences, comparisons are only approximate and are marked
with `golden_match=false`.

## Resources and correctness

- `rtx-engine`: 80.4 s whole-process wall time; peak process RSS 18.76 GiB; maximum decode-throughput CV 0.05%.
- `rtx-facade`: 81.1 s whole-process wall time; peak process RSS 18.60 GiB; maximum decode-throughput CV 0.08%.
- `gb10-engine`: 355.8 s whole-process wall time; peak process RSS 18.64 GiB; maximum decode-throughput CV 0.64%.
- `gb10-facade`: 349.9 s whole-process wall time; peak process RSS 18.47 GiB; maximum decode-throughput CV 1.29%.

Whole-process time includes loading, setup, warm-ups and all measured requests;
it is not a request latency. RSS includes mapped model pages and is not GPU
memory. RTX sampled GPU memory peaked at 17,130 MiB (16.73 GiB). GB10's NVML
VRAM counter is unavailable; no fabricated per-GPU memory figure is substituted.
System-memory headroom before its successful sweep was 123,685,318,656 bytes.
The existing Cursus services remained resident during the GB10 run.

NVML telemetry was sampled approximately twice per second. Across the engine
sweep, RTX reported mean/peak power 251.4/296.8 W and GB10 42.1/69.3 W, including
initial loading/idle periods. These counters do not establish comparable
whole-machine energy efficiency. No PCIe traffic, wall power, kernel bandwidth,
per-token p95/p99 latency, or pure prefill throughput was measured.

GB10 golden mismatches: `4/ar`, `5/ar`, `6/ar`, `4/fixed`, `5/fixed`, `6/fixed`,
`4/adaptive`, `5/adaptive`, `6/adaptive`, `5/sampled`, `6/sampled` (editing,
extraction and retrieval). The same set occurs through the public ABI. At the
96-token cap, some outputs stop in the thinking/code phase. Spot checks showed
the correct None/default-list repair approach on both devices; GB10 also emitted
the requested Acme/17 JSON within the cap. This does not replace a quality suite
or teacher-forced logit comparison, which was not run.

Device instruction/control-word comparisons against each architecture's
pre-fix objects passed for all **423 GPU functions**. This demonstrates that the
host capability fix did not change device instructions; it does not assert that
12.0 and 12.1 binaries are identical. Host/API/install, extracted package, lint,
14 idle-guard regression tests and capability acceptance/rejection tests passed.

## Commands

In separate source checkouts of the measured revision:

```sh
# RTX: 120a. GB10: substitute 121a; both builds use native host compilers.
make -j2 CUDA_ROOT=/usr/local/cuda CUDA_ARCH=120a \
  .build/native-benchmark .build/facade-benchmark

# Supply the correct UUID, and only run after reserving/freeing that device.
export INKLEBERRY_TEST_GPU_UUID=GPU-APPROVED-ID
export INKLEBERRY_TEST_TIMEOUT_SECONDS=3600
python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 \
  .build/native-benchmark /external/verified/model > engine.jsonl
python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 \
  .build/facade-benchmark /external/verified/model > facade.jsonl
python3 tools/model_release.py verify engine.jsonl \
  --golden test/fixtures/inkling-golden-v1.json
```

The last command passes on RTX and currently fails on GB10; do not refresh the
expected sequences to hide that distinction. The guard is an idle check, not a
scheduler. On GB10, require no compute process, zero utilization and sufficient
host memory; unavailable VRAM alone is not evidence that the device is free.

## Operational outcome and evidence

Only the explicitly approved `qwen3_8-flash-next-fp4-local` vLLM service on
`spark-7468` was temporarily stopped. The supervisor restored it and confirmed
`/health` and `/v1/models`; configuration verification passed and its prior
systemd enabled/disabled state was retained. Cursus, PostgreSQL, Velo and Carta
retained their eight original container IDs and remained healthy. No services
on `spark-3eb0`, drivers, system dependencies, routing or power limits were changed.

Two earlier Spark attempts aborted before model inference: a supervisor relative
path error, then the existing 12.0-only host gate. Both restored vLLM; their error
and lifecycle records are retained. The successful run used absolute paths and
the capability fix. A temporary model-transfer HTTP server was stopped.

Raw JSONL, telemetry, model identities, binaries' build logs, supervisor/download
scripts, lifecycle/restore evidence, decoded spot-checks, and full statistical
summaries are retained outside Git at:

`/Users/mark/artalis.io/data/inkleberry-blackwell-gb10-20261126T124414Z`

Remote experiment directory on both hosts:
`/data/scratch/inkleberry-benchmark-cd9f2bc` (its directory name records the initial
revision; the successful measurement uses the newer pinned source above).
Source weights and bulk benchmark records are not committed.

During the run, the Cloudflare SSH aliases returned handshake errors. The
controller refreshed its existing `core` Access login and used LAN SSH for the
Spark with its existing key and verified host identity. No SSH configuration or
Cloudflare resources were changed; the Spark Access alias needs separate review.
