# Adaptive DSpark and verified prompt lookup

The C host now translates the pinned Zig adaptive schedule without changing its
CUDA mathematics. The original source manifest remains hash-verified at
`d10c0abb2c8990be81c08ba84c66a8333a691b33`. This is a stacked slice after
[sampling/fixed-width speculation](speculation-validation.md).

## Contract

Use `SQ_DECODE_OPTIONS_INIT`, then set `draft_tokens=7`, `adaptive=1`,
`spec_threshold=0.8f` and optionally `prompt_lookup=1`. Adaptive execution requires
CUDA graphs and MTP weights. A finite threshold in (0,1] is mandatory; invalid
combinations fail before GPU mutation. Existing AR/fixed-width defaults remain
unchanged. The previous options prefix ending at `reserved` is still accepted;
new fields default to adaptive/lookup disabled. The original statistics layout
is unchanged. Initialize `sq_decode_info` with `SQ_DECODE_INFO_INIT` and query
`sq_session_decode_info` after return for lookup steps, verification-width counts
and successful draft launches. It returns BUSY during an active operation.

Each request remains independent: there is no prefix cache or HTTP service.
The selector chooses verification widths 2, 4 or 8 using the original confidence
kernel. A threshold change rebuilds only the conditional graph. Fixed graphs,
draft graph and lookup graph are separate, session-owned resources.

The C lookup index preserves the pinned four-token hash, sixteen-token suffix
confirmation, two candidates and generation-stamped 8,192-token history. It
proposes seven tokens, but the target verifies them through the original Dirac
proposal acceptance path. A lookup hit never authorizes unverified output.
Generated and prompt tokens feed only the current request's index. The index is
bounded host bookkeeping, not an unbounded cache.

## Ownership and cancellation

Adaptive state uses the injected host allocator. MTP sessions reserve confidence
and width buffers in their existing arena, one 68-byte pinned output/proposal
block and one completion event. No graph-owner C++ heap allocation is hidden
behind the C interface. The additive checked CUDA bridge retains partial graph
handles on error and leaves the original CUDA/Zig sources untouched.

For neural proposals the completion event marks verified output, followed by the
next draft on the same stream. The host waits for that event, allowing drafting
to overlap token publication. Every return path drains any trailing draft before
reset, the B=1 context-boundary tail, or caller reuse. Callback stop, EOS, output
length and cancellation therefore cannot leave an orphaned draft. Cancellation
is cooperative, not CUDA kernel preemption. A drain failure poisons the session;
cleanup retains failed resource handles for a retry. Graphs retire before their
referenced arena/pinned outputs and stream. The existing single-live-session
restriction remains in force.

The target-tap commit remains serialized on the main stream, as in the earlier
C slice. This change establishes behavioral parity, not an inference speedup.
Acceptance counters include completed verification even when publication stops
before all verified tokens are emitted.

## Reproduce

CPU-only tests require neither provider weights nor CUDA:

```sh
make -j8 check install-check
make lint                     # clang-format 18
make sanitize
make thread-sanitize
make fuzz-smoke
```

On a reserved idle Blackwell GPU with the external model files already present:

```sh
export SQ_TEST_GPU_UUID=GPU-REPLACE-WITH-APPROVED-IDLE-UUID
make -j8 CUDA_ROOT=/usr/local/cuda-13.5 squid-c-generate
python3 tools/run_cuda_smoke.py /usr/local/cuda-13.5/lib64 \
  ./squid-c-generate /external/model-directory 'Explain why the sky is blue.' \
  128 nvfp4 fp8 graph --temperature 0.8 --top-p 0.95 --top-k 40 \
  --seed 42 --drafts 7 --adaptive-threshold 0.8 --prompt-lookup 1
make CUDA_ROOT=/usr/local/cuda-13.5 ZIG=/path/to/zig adaptive-oracle
python3 tools/validate_adaptive.py --model /external/model-directory \
  --cuda-lib /usr/local/cuda-13.5/lib64
python3 tools/validate_adaptive.py --long --model /external/model-directory \
  --cuda-lib /usr/local/cuda-13.5/lib64
make CUDA_ROOT=/usr/local/cuda-13.5 .build/adaptive-recovery .build/adaptive-graph
python3 tools/run_cuda_smoke.py /usr/local/cuda-13.5/lib64 \
  .build/adaptive-recovery /external/model-directory
SQ_TEST_TIMEOUT_SECONDS=3600 python3 tools/run_cuda_smoke.py \
  /usr/local/cuda-13.5/lib64 /usr/local/cuda-13.5/bin/compute-sanitizer \
  --tool memcheck --leak-check full --error-exitcode 99 .build/adaptive-graph
SQ_TEST_TIMEOUT_SECONDS=3600 python3 tools/run_cuda_smoke.py \
  /usr/local/cuda-13.5/lib64 /usr/local/cuda-13.5/bin/compute-sanitizer \
  --tool memcheck --leak-check full --error-exitcode 99 \
  ./squid-c-generate /external/model-directory 'Explain why the sky is blue.' \
  64 nvfp4 fp8 graph --temperature 0.8 --top-p 0.95 --top-k 40 \
  --seed 42 --drafts 7 --adaptive-threshold 0.8 --prompt-lookup 1
```

The oracle imports the actual pinned Zig Engine and ModelWeights. It compares
every emitted ID and speculative/acceptance/lookup counters for seven sampler
configurations. Within a session it changes confidence thresholds and repeats an
earlier sampler after other requests. The short matrix crosses all three weight
modes, all three global-KV formats and lookup on/off. It includes local-window,
prefill-boundary and context-tail cases, plus a copy workload that must produce
actual verified lookup hits. Long runs perform real 8K/131K prefills; merely
allocating KV does not qualify a context. Each case is a single qualification
run, not five-repetition performance characterization or an independent quality
benchmark. Matching Zig preserves the pinned implementation's known numerical
limitations; it does not establish universal stochastic quality equivalence.

The native recovery test covers callback stop, callback cancellation, independent
thread cancellation during prefill, adaptive/fixed/AR switching, threshold graph
rebuild and exact replay after reuse. A separate real-device graph fixture forces
all three branches at three thresholds and retires partial/capturing graphs.
CPU tests cover lookup confirmation/collisions, ring/generation wrap, exact old
options-prefix reads, invalid options, BUSY queries, constructor failure,
conditional-graph cleanup retention, pinned/event release failure and retry.

## Executed evidence

On core-1, all **280 adaptive C/Zig requests** passed (38 short/copy cases plus
real 8K and 131K cases, seven samplers each). All nine weight/KV copy combinations
produced verified lookup hits. The copy fixture contains **158 actual prompt
tokens**; its 1,024-token allocation is not a long-prefill claim. Fourteen AR/fixed
regression requests also matched. All three memcheck runs reported zero errors and
zero leaked bytes, including a C-only 256-token copy run with 18 actual lookup
steps. Its prompt is the literal in `sq_adaptive_copy_fixture`
(`test/adaptive_compare.c`), rendered by the CLI; it uses the default greedy
sampler, NVFP4/FP8, seven drafts, threshold 0.8 and lookup enabled. Native prefill cancellation returned in 0.153337 seconds in
one observation; this is not a cancellation-latency guarantee. Linux host checks,
ASan/UBSan, ThreadSanitizer, 3,000 fuzz iterations per parser and installed C/C++
consumers passed; macOS ARM64 ordinary host checks passed.

See [the qualification manifest](../benchmarks/adaptive-2026-11-25.json) for
executed results, toolchain/model identities and limitations. Prefix restoration,
HTTP serving, million-token independent retrieval and full performance/energy
characterization remain separate release gates. No production service or global
GPU configuration is changed by these tests.
