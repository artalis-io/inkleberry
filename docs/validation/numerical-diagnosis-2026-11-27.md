# Numerical diagnosis — 27 November 2026

PR #9 merged into the Ampere branch as `ce923e2` after green CI. This pass adds
diagnostic tests and analysis. **No kernel, golden, default precision, sampling
rule or production math policy changed.** GB202 and GB10 measurements are complete;
Ampere runtime measurements await an approved desktop reservation.

## Controlled inputs and software

All three hosts have identical SHA-256 checksums for tokenizer, main weights and
MTP weights from Inkling-2-Mini `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`.
Primary cross-device traces use source
`879712614f41ff16edb554dcc82bf2e774d20613`, NVFP4/MTP weights, FP8 global KV,
eager target execution and the same eight valid token IDs. Local KV retains its
existing representation. The initial GB202 pilot used `566c0fc`; a repeat at the
frozen revision matched every packed record and all 10,848 tensor files exactly.

The real-generation prefill follow-up uses `9ce816d1aff9ce7fb4ac67b17fe3a2c2c75cdf82`
on both Linux hosts. It uses the public 36-token extraction benchmark prompt,
32K context and one greedy output token. The FP64 reference follow-up uses
`7ef9e79` on GB202; its actual operand hashes match both original prefill runs.
Changes between these diagnostic revisions do not change library mathematics.

| Host | Runtime measurements | Toolkit / loaded cuBLAS |
| --- | --- | --- |
| GB202-class RTX PRO 6000 Blackwell Max-Q, `core` | Completed | CUDA 13.5.37 / cuBLAS 13.10.0.20 |
| GB10, Linux ARM64, `spark-7468` | Completed; approved vLLM pause/restore | CUDA 13.0.88 / cuBLAS 13.1.1.3 |
| RTX 3090 Ti, Windows x64, `edge` | Native MSVC/CUDA build prepared at the frozen primary revision; GPU runs pending | Installed native toolkit; no driver/toolchain change |

See [machine-readable evidence](numerical-diagnosis-2026-11-27.json) for model
checksums, retained binary checksums, measured results and lifecycle evidence.
Pilot executable hashes were not captured before replacement; they are not
presented as measured hashes of the later reference executable. Exact source
revisions, build flags and toolchains remain recorded.

## Packing and short eager execution

The tracer hashes initialized weight payload spans without alignment padding and
records view geometry, offsets and exact global-scale bits. It does not save model
weights into Git. It records stage values and physical visit numbers, distinguishing
reused layers. This finds the earliest *observed* boundary, not an independent
oracle for every operator.

GB10 and GB202 match all **1,690 packed span/view records**. On both singleton and
batch-8 execution, all **2,712 stage/position tensors** are bit-identical for the
frozen short probe. The same 76 library source files were also fingerprinted and
matched across the Linux hosts in the prefill follow-up.

There is a separate execution-shape issue even within GB202:

| Weight profile | Batch 1 against batches 2/4/8 | Earliest observed difference |
| --- | --- | --- |
| `exact` | All recorded tensors identical in this fixture | None |
| `fp8` | 2,696 differing tensors | Visit 0 Q/R/gate/sink projection; maximum absolute difference 0.00000667572021484375 |
| `nvfp4` | 2,696 differing tensors | Same projection; maximum absolute difference 0.0021953582763671875 |

Top-1 matched at all eight positions. `exact` retains mixed checkpoint expert
formats; it is not a full-precision oracle. A controlled arithmetic-prompt probe
with a singleton-built 27-token common prefix and 11 forced generated tokens also
preserved top-1 at every position. That probe did **not** reproduce the historical
AR/speculative sequence difference. Public full-logits evaluation is not proof
of equivalence to every optimized greedy LM-head or MTP draft/commit path.

## Cross-device boundary: real batched prefill

In the ordinary generation campaign, prompts of 27, 28, 24 and 31 tokens match the
GB202 goldens on GB10; longer prompts use the separate batched-prefill path. The
focused extraction probe exercises that actual path through
`ifn_session_generate_ex`, rather than simulating it with repeated short forwards.

Both hosts have identical packed payloads. At the **first cuBLAS GEMM** they also
have identical actual dequantized FP16 weight and activation matrices, dimensions
and scale bits. Embedding and initial attention-input values match. The first
observed difference is the Q/R/gate/sink projection, visit 0, position 0:

| Measurement | Value |
| --- | --- |
| GEMM rows / columns / batch | 6,552 / 2,304 / 32 |
| Maximum absolute result difference at position 0 | 0.000019073486328125 |
| RMS result difference at position 0 | 0.0000023595823704510036 |
| Differing coordinates at position 0 | 6,489 / 6,552 |
| Packed records that differ | 0 / 1,690 |
| Actual GEMM operand/scale identities | Equal |

The difference propagates to 10,552 of 10,632 observed tensors. Both runs produce
the same first generated token; that alone does not explain later text differences.
The comparator orders prefill stages before tail execution, so later low-numbered
tail visits cannot be mislabeled as the original cause.

An independent binary16 decoder and FP64 dot product were evaluated for three
first-token coordinates using the actual represented GEMM inputs. Row 3277 was
chosen from the observed worst cross-device discrepancy; row 0 and the last row
are controls. No raw weight matrix was stored.

| Row | FP64 reference | GB202 | GB10 |
| --- | ---: | ---: | ---: |
| 0 | 0.2442909849586768 | 0.2442911267280579 | 0.2442909330129623 |
| 3277 | 10.599749733712805 | 10.599742889404297 | 10.599723815917969 |
| 6551 | 6.569727543900758 | 6.569722652435303 | 6.569715976715088 |

These selected references assess GEMM arithmetic on the represented FP16 inputs,
not quantization quality or a whole-model error bound. Neither GPU is uniformly
closer at the sampled coordinates.

The measured boundary is inside the vendor GEMM result, rather than a dataset,
packing or first-GEMM operand mismatch. It cannot be attributed to silicon alone:
toolkit, cuBLAS version, CPU architecture and SM configuration also differ.
NVIDIA's [cuBLAS reproducibility guarantee](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)
is scoped to matching toolkit, GPU architecture and SM configuration; it does not
promise bit-identical outputs across this deployment matrix. This explains why
the observation is possible, not every resulting generation difference.

## Generation, graph and performance regression

Each uninstrumented client ran seven workload families across AR, fixed
speculative, adaptive speculative and sampled generation: one warm-up and five
measured repetitions per group. Engine/facade tokens match in all 28 groups on
each device. Both devices pass their own historical token/performance gates.

| Device | Engine / facade matches against existing GB202 goldens | AR | Fixed | Adaptive | Sampled |
| --- | --- | ---: | ---: | ---: | ---: |
| GB202 | 28/28 each | 239.13 | 331.70 | 413.14 | 225.71 |
| GB10 | 17/28 each | 71.36 | 80.23 | 89.01 | 69.24 |

Rates are pooled median decode tokens/s from eligible groups, not per-context
latency guarantees. Since cross-device outputs differ, the table is not an
apples-to-apples speedup claim. Profile-specific golden parity is a regression
check, not an independent semantic-quality or distribution-preservation proof.

Existing AR/speculative sequence differences on GB202 remain at generated index
11 in arithmetic and 78 in retrieval/copying. No goldens were rewritten to hide
them or the GB10 differences. Synthetic linear-shape tests also pass five eager
and five graph repetitions per shape on both Linux targets, including resource
preparation in independent CUDA contexts.

## Proposed smallest next experiment — no fix implemented

Prioritize the **vendor prefill GEMM contract** before changing custom decode
kernels. In an isolated test, evaluate explicit `CUBLAS_COMPUTE_32F_PEDANTIC` and
compare the same represented operands, FP64 references, generation and prefill
latency. This is a candidate precision control, not a promise of cross-device
bit identity. A matched-toolkit experiment would separately control version
selection; use an isolated environment, not a host driver/toolchain upgrade.
Do not prescribe a workspace setting or fixed algorithm number as a universal
cross-version numerical guarantee.

The within-device singleton/batched projection discrepancy is a separate axis.
Its first fused projection dispatch is in `ifn_cuda_dispatch_qkvrg_b`; changing
only generic linear GEMV would not fix that first boundary. Only after the
numerical contract is agreed should a shared existing fused-MMA specialization
for B=1 be evaluated, with scale-folding, layout and performance checks. Do not
rewrite all kernels, change the default to `exact`, fit a permissive tolerance,
or update goldens. No corrective arithmetic policy has been implemented here.

## Runtime safety and pending Ampere work

Only the previously approved `qwen3_8-flash-next-fp4-local` vLLM service on
`spark-7468` was temporarily stopped. Both bounded supervisors restored it and
verified `/health`, `/v1/models`, original enablement and all eight original Cursus
container IDs. The focused prefill window began only after the first restoration
was verified. Cursus, PostgreSQL, Velo and Carta stayed unchanged; `spark-3eb0`
was not modified. No GPU settings, driver or global dependencies changed.

The current `edge` desktop requires a new operator-reviewed expiring reservation.
It has not been admitted automatically. Its model checksums match and the frozen
primary diagnostic builds natively, but no new Ampere GPU result is claimed.
Three-device numerical qualification remains incomplete.

## Reproduction and checks

Keep every output in a new directory outside the repository:

```sh
make -j4 .build/numerical
mkdir /external/new-short-run
INKLEBERRY_TEST_GPU_UUID=GPU-APPROVED-ID \
  python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 \
  .build/numerical /external/model /external/new-short-run 2

mkdir /external/new-prefill-run
INKLEBERRY_TEST_GPU_UUID=GPU-APPROVED-ID \
  python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 \
  .build/numerical /external/model /external/new-prefill-run 2 \
  test/fixtures/numerical-extraction-prefix.txt generation
python3 tools/compare_numerics.py /external/gb202-prefill /external/gb10-prefill \
  --left-batch 0 --right-batch 0 --out /external/comparison.json
```

Use the recorded target source revision and native build directory/suffix for
Windows or experimental profiles. The guard must still admit the reserved GPU.
The small checked-in extraction fixture contains public input token IDs, not
expected outputs or model bytes. Optional token files contain a common-prefix
length followed by IDs, up to 128 tokens. `generation` requires a zero common
prefix and more than 32 tokens. New completed runs contain `run.json`; the
comparator rejects changed inputs/configuration, duplicate/missing stages,
truncated payloads, incomplete GEMM identities and non-finite values (even equal
NaNs). Legacy fixed-eight-token pilot outputs remain readable as that fixture.

Local host/install/contract checks, pin/namespace/lint checks, reference-free and
extracted packages pass; comparator regression tests cover input identity,
missing/duplicate data, finiteness, GEMM operands and prefill execution order.
Native compilation and model source checks were performed on all three hosts;
GPU execution is Linux GB202/GB10 only. Default Linux forward **and prefill**
objects match their pre-diagnostic object bytes exactly with tracing disabled.

Raw controller artifacts are under `data/inkleberry-numerics-2026-11-27/`.
Linux workspaces are `/data/scratch/inkleberry-numerics-2026-11-27`; Windows uses
`C:/data/cache/inkleberry/numerics-2026-11-27`. Preserve sources and artifacts when
continuing Ampere. Review PR CI separately; do not report pending runs as passed.
