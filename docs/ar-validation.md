# C autoregressive execution

The C host now executes the pinned Inkling forward pass, batched prefill and
greedy autoregressive decode. It does not call Zig in production. Zig is a test
oracle; CUDA math sources remain unchanged at
`d10c0abb2c8990be81c08ba84c66a8333a691b33`.

## Contract and ownership

The optional `libsq-gpu.a` exports `include/sq_engine.h`. Sessions use an explicit
size/version configuration, caller-selected context limit (64–1,048,576),
BF16/FP8/Q4 KV and eager/graph decode. “Exact” retains checkpoint-quantized expert
weights. Unknown options fail; the API never silently enables speculation.

One live C session per process is accepted initially. A session retains its GPU
weight image; image destruction returns `SQ_BUSY` until session teardown
succeeds. Sessions own the runtime arena, stream, graph and cuBLAS handle. They
borrow no model mappings after upload. Callers keep allocator contexts, handles
and input/output spans alive until the synchronous call returns, and prevent
new calls when destruction can release a handle.

`create` validates host inputs before selecting the GPU. Device initialization
failure can publish a partially initialized handle: always destroy a non-NULL
output. Destruction retains failed resources for retry. Graphs and cuBLAS are
retired before their stream/arena; no reset of the device or caller context is
used. The first operation error survives subsequent cleanup failures.

`generate` starts an independent request, validates all IDs and context/output
bounds, resets recurrent state and fills caller-provided output storage. Greedy
sampling is the only exposed policy. EOS, length exhaustion, callback stop and
cancellation have distinct finish reasons. Emitted EOS is counted, matching Zig.
There is no prefix-cache reuse. `logits` accepts 1–8 teacher-forced IDs and returns
the last position's entire 200,058-value post-softcap FP32 vector; reset=0 can
continue an existing teacher-forced sequence.

An overlapping generation/reset/destroy returns `SQ_BUSY`. Cancellation is the
only supported reentrant operation and may come from a callback or another
thread. Cancellation waits for the current GPU chunk, synchronizes and resets
recurrent state; it cannot preempt a running kernel. Already emitted tokens
remain in caller output with accurate counts. Validation errors leave the
session usable; CUDA execution/restoration failures poison it until teardown.
Callbacks must preserve the CUDA device/current context and must not reset or
destroy the active session. Cancellation is not a poisoned state. Device loss can require worker restart;
the library does not repair it by resetting shared hardware.

## Execution and cuBLAS boundary

`engine_forward.c` preserves the original 48 visits, loop passes, controller
selection, projections, local/global attention, router/MoE, residual/depth-norm,
position accounting and output head. Ablation, LoopCD and DSpark are disabled
rather than silently approximated. `engine_prefill.c` preserves the original
2,048-token maximum Tensor Core chunks, 256-token local-attention subchunks and
8-token tail handling. Target-forward and batched-prefill scratch have distinct
lifetimes inside one checked, 256-byte aligned runtime arena.

The original cuBLAS prefill wrappers use a process-global handle and terminate
on some errors. The C path never calls them. `checked_prefill.cu` includes the
unchanged CUDA prefill source to reuse its kernels and adds error-returning
GEMM/controller/global-convolution dispatch using the session's own cuBLAS handle.
Math mode, GEMM arguments and dequantization kernels match Zig. CUDA and cuBLAS
statuses are surfaced in bounded diagnostics; cuBLAS raw statuses use the
`10000 + status` domain (allocation failure maps to `SQ_OUT_OF_MEMORY`).

The included source supplies original entrypoints for the oracle too. Do not
also link the original `kernels_prefill` object into `libsq-gpu.a`; the Makefile
recreates that archive to prevent removed members surviving incremental builds.
cuBLAS creation is lazy at the first batched prefill; the decode graph is captured
on first generation. Those operations and cuBLAS internal workspace allocation
are first-use initialization, not a claim of zero allocation across an entire
cold request. The steady decode loop has no explicit per-token host/device
allocation.

## Reproduce

```sh
make -j8 check install-check
make lint                         # clang-format 18
make sanitize
make thread-sanitize
make fuzz-smoke
make -j8 CUDA_ROOT=/usr/local/cuda-13.5 squid-c-generate gpu-install-check
make CUDA_ROOT=/usr/local/cuda-13.5 ZIG=/path/to/zig ar-oracle
export SQ_TEST_GPU_UUID=GPU-REPLACE-WITH-APPROVED-IDLE-UUID
python3 tools/validate_ar.py --model /external/model-directory \
  --cuda-lib /usr/local/cuda-13.5/lib64
python3 tools/validate_ar.py --long --model /external/model-directory \
  --cuda-lib /usr/local/cuda-13.5/lib64
```

The model directory contains official external `tokenizer.json` and
`model-00001-of-00001.safetensors`. MTP is not loaded. The oracle harness targets
the audited Zig 0.17.1 toolchain. Every test selects the approved idle device
through the UUID guard. No downloads, clock/power changes or termination of other
workloads occur. The default per-process GPU test timeout is 180 seconds; long
qualification explicitly raises it to a bounded 3,600 seconds.

C-only memory checking (no Zig runtime or its global cuBLAS handle):

```sh
SQ_TEST_TIMEOUT_SECONDS=3600 python3 tools/run_cuda_smoke.py \
  /usr/local/cuda-13.5/lib64 /usr/local/cuda-13.5/bin/compute-sanitizer \
  --tool memcheck --leak-check full --error-exitcode 99 \
  ./squid-c-generate /external/model-directory \
  'Explain in two sentences why the sky appears blue during the day and red during sunset, using simple language for a child who is learning about light and the atmosphere.' \
  64 nvfp4 fp8 graph
```

`gpu-install-check` stages headers and both archives inside `.build/gpu-stage`,
then links/runs C and C++ consumers. It never installs into a system directory.
Host CI compiles and exercises the lifecycle against a private injected backend;
it does not require CUDA, a driver or model files.

## Qualification and limits

Executed results are recorded in `benchmarks/ar-2026-11-25.json`. Short native
coverage comprises 27 cases: three weight modes × three KV formats × eager/graph,
and lengths 1, 7, 8, 9, 31, 32, 2047, 2048 and 2049 in NVFP4/FP8-KV graph mode.
Teacher-forced logits are compared bit-for-bit over the entire vocabulary at the
last position of each eight-token batch. Every generated ID is compared against
actual Zig greedy generation, including its production batched-prefill path.
No numerical tolerance was widened to accept a divergence.

The additional real 8,192- and 131,072-token NVFP4/FP8-KV graph runs passed.
They compare 204,859,392 and 3,277,750,272 FP32 values respectively, plus all
32 generated IDs. During the 8K run, cross-thread prefill cancellation returned
135 ms after the request in the observed trial; repeating the request afterward
reproduced the uncancelled result. This observation is not a worst-case bound.
The complete oracle qualification took about 23 seconds at 8K (including the
cancellation/reuse checks) and 355 seconds at 131K. Those wall times include both
engines, comparisons and initialization, not standalone prefill/decode throughput.

C-only native memcheck passed with zero errors and zero leaked allocations for
a 46-token prompt, batched prefill and 64 graph-decoded tokens, including cuBLAS
teardown. A separate natural-language request reached EOS after 37 tokens.
Both staged C and C++ consumers passed. Linux GCC/Clang compilation, macOS ARM64
host checks, ASan/UBSan, TSan and two 3,000-iteration parser fuzz smoke campaigns
passed. The private host simulation additionally exercises acquisition/cleanup
faults, memory preflight, invalid lengths before scanning IDs, retained images,
poisoning, callback reentrancy, concurrent cancellation and cancellation combined
with failed context restoration. These simulations do not validate kernel math.

The first implementation matched target-forward logits but disagreed with Zig
generation at 33 tokens because it lacked the Tensor Core prefill branch. That
was treated as a blocker: the real batched path and checked cuBLAS dispatch were
ported before rerunning the matrix. Final parity records are emitted only after
all checks and cleanup succeed.

The fixtures repeat real tokenized text and perform actual prefill, not zeroed
KV allocation at simulated positions. They qualify execution consistency, not
long-context retrieval quality, broad task accuracy, stochastic distributions,
serving viability or speedups. Timings are single qualification runs, not a
five-repetition performance study. Only tested context/mode combinations should
be called qualified. The configured maximum alone establishes no usable-context
claim.

Remaining work: stochastic sampling, DSpark/speculation, prefix-cache ownership,
independent long-context retrieval/quality workloads, request/stop-string
adapters, serving and full performance characterization. Kernel tuning and
upstream upgrades remain separate changes.
