# Stochastic generation and fixed-width DSpark

This slice adds `sq_session_generate_ex` and `sq_decode_options` to the optional
C GPU API. It preserves the existing greedy `sq_session_generate` contract.
Original CUDA and Zig sources remain unchanged and hash-verified at
`d10c0abb2c8990be81c08ba84c66a8333a691b33`.

## Public contract

Options have size/version fields and a reserved-zero field. Defaults select
greedy AR: temperature=0, top_p=0.95, top_k=0, seed=0, draft_tokens=0.
Temperature must be finite and nonnegative, top_p finite in (0,1], top_k between
0 and 200058, and draft_tokens between 0 and 7. Invalid values return
`SQ_INVALID_ARGUMENT` before device selection or mutation. The kernel's existing
threshold is retained: temperature below 1e-4 selects greedy, as does top_k=1.

Seed zero is a deterministic seed, not an instruction to read a clock. All 64
seed bits are preserved. Each request uploads a fresh CUDA sampler configuration;
cached graphs reference that device buffer rather than capturing old host values.
Changing seed, temperature or filters does not rebuild the graph or reuse RNG
state from another request. Sampling mathematics, histogram-based top-p/top-k
filtering, Gumbel sampling and stochastic verification remain the original kernels.

Zero draft tokens selects AR. Positive values explicitly select a **fixed**
DSpark width; the verification batch is draft_tokens+1. MTP weights must already
have been loaded in the image. Missing MTP returns `SQ_UNSUPPORTED`; there is no
silent fallback. Draft width, precision and KV format are explicit in the test
manifest. That initial slice did not expose adaptive width or prompt lookup. The subsequent
[adaptive slice](adaptive-validation.md) appends explicit options while accepting
the original options prefix; fixed-width behavior remains unchanged.

The extended statistics include ordinary generation counts/finish reason plus
speculative steps, proposed drafts and accepted drafts. These last counts cover
completed verification work, including verified tokens that were not emitted
because output length, EOS, callback stop or cancellation ended publication.
They are not interchangeable with the number of published tokens.

Callbacks, input/output lifetimes, BUSY handling, poisoning, context restoration
and retryable destruction follow [the AR contract](ar-validation.md). Every
request starts independently; there is no prefix cache. Cancellation is
cooperative between completed chunks, not kernel preemption. A restoration error
overrides normal cancellation. No device reset is used.

## Execution and ownership

The weight binder materializes the already-validated full-vocabulary DSpark
descriptors. An MTP image's session reserves three draft KV pairs, its recurrent
ring, target taps, mixing/work buffers, probability vectors and verification
outputs during arena construction. No draft workspace is allocated per step.
Local MTP caches use BF16; the global cache uses the configured format and retains
the pinned source's 64K draft-cache cap for contexts above 524288. That cap does
not clamp the target model's context.

Prefill commits target taps into the draft caches. Each speculative step runs
DSpark drafting, assembles anchor/proposals, runs the target verification batch,
then invokes the original acceptance/position-update kernel. The host checks
accepted-count bounds before updating counters or publishing tokens. Near the
context boundary it uses the original B=1 tail behavior, preventing verification
from wrapping global KV indices.

Graphs are cached separately for fixed widths 1–7 and destroyed before the arena
and stream. The AR graph remains separate. This initial C path serializes DSpark
commit and the LM head on the session stream instead of using Zig's auxiliary
stream overlap. Their buffers are independent; mode-matched numerical and token
parity are required despite the scheduling change. There is no trailing
asynchronous draft to drain on return. Overlap/adaptive optimization is separate
work, not an advertised speedup in this change.

## Reproduce

CPU-only CI needs no model, GPU, CUDA headers or SDK:

```sh
make -j8 check install-check
make lint                           # clang-format 18
make sanitize
make thread-sanitize
make fuzz-smoke
```

Build and run on an explicitly reserved Blackwell GPU:

```sh
make -j8 CUDA_ROOT=/usr/local/cuda-13.5 squid-c-generate gpu-install-check
export SQ_TEST_GPU_UUID=GPU-REPLACE-WITH-APPROVED-IDLE-UUID
python3 tools/run_cuda_smoke.py /usr/local/cuda-13.5/lib64 \
  ./squid-c-generate /external/model-directory 'Explain why the sky is blue.' \
  128 nvfp4 fp8 graph --temperature 0.8 --top-p 0.95 --top-k 40 \
  --seed 42 --drafts 7
```

The external directory contains `tokenizer.json`,
`model-00001-of-00001.safetensors` and, for positive drafts, `mtp.safetensors`.
No download occurs. Omit the flags for the existing greedy workflow. CLI flags
may follow the prompt or the optional positional settings; seed parsing preserves
full unsigned decimal uint64 values. There is no implicit random seeding.

Model-backed oracle qualification uses audited Zig 0.17.1:

```sh
make CUDA_ROOT=/usr/local/cuda-13.5 ZIG=/path/to/zig decode-oracle
python3 tools/validate_decode.py --model /external/model-directory \
  --cuda-lib /usr/local/cuda-13.5/lib64
python3 tools/validate_decode.py --long --model /external/model-directory \
  --cuda-lib /usr/local/cuda-13.5/lib64
```

The oracle imports actual pinned `Engine`/`ModelWeights`. It explicitly disables
Zig's adaptive graph and prompt lookup so both engines exercise the same fixed
width. It compares every generated ID and all three acceptance counters. Seven
sampler configurations include greedy, unrestricted stochastic sampling, filtered
sampling, top-k=1, the greedy threshold, maximum uint64 seed, and an earlier seed
repeated after other requests. No C RNG or duplicated decoder is the oracle.

Small distribution diagnostics invoke the unchanged target-sampling kernel with
synthetic two-token distributions:

```sh
make CUDA_ROOT=/usr/local/cuda-13.5 sampling-kernel-check
make CUDA_ROOT=/usr/local/cuda-13.5 .build/decode-recovery
python3 tools/run_cuda_smoke.py /usr/local/cuda-13.5/lib64 \
  .build/decode-recovery /external/model-directory
```

Memory checking, including C-only MTP upload, fold, prefill, speculative graph
execution and teardown:

```sh
SQ_TEST_TIMEOUT_SECONDS=3600 python3 tools/run_cuda_smoke.py \
  /usr/local/cuda-13.5/lib64 /usr/local/cuda-13.5/bin/compute-sanitizer \
  --tool memcheck --leak-check full --error-exitcode 99 \
  ./squid-c-generate /external/model-directory \
  'Explain briefly why the sky is blue in the daytime and red at sunset, using simple language for a young child who is learning about sunlight and the atmosphere.' \
  64 nvfp4 fp8 graph --temperature 0.8 --top-p 0.95 --top-k 40 \
  --seed 42 --drafts 7
```

The idle UUID guard is required for all GPU commands. It checks process/VRAM
state but is not a scheduler reservation. No driver, clock, power or unrelated
service changes are made. Long qualification explicitly raises the bounded
per-process timeout to 3600 seconds.

## Evidence scope

On core-1's RTX PRO 6000 Blackwell, the 95 ordinary settings passed all
665 mode-matched C/Zig requests. This covers three weight modes, three KV formats,
eager/graph execution, every draft width, prompt boundaries and context-tail
fallback. The additional 8K and 131K cases each passed all seven sampler
configurations, for 679 matched requests in total. Generated token IDs and
speculative counters matched exactly; no C/Zig tolerance was widened.

C-only memcheck passed with zero errors and zero leaked allocations for MTP
upload/folding, a 47-token prefill, 64 emitted tokens, 28 verification steps and
teardown. The four sampling diagnostics also passed under memcheck with zero
errors/leaks. Direct sampling observed P(0)=0.703699, stochastic rejection
0.701624 and Dirac-proposal rejection 0.701990 against target 0.7; top-k=1 was
fully deterministic. The `matching_proposals` diagnostic is a token-match count;
it represents accepted proposals only in the rejection-sampling cases.

The C-only 8K recovery suite passed callback stop/cancellation after four tokens,
AR/DSpark and draft-width switching, cross-thread prefill cancellation and exact
post-cancellation token/counter reproduction. Cancellation returned about 148 ms
after the request in that trial, not a worst-case guarantee. The legacy greedy
oracle remained bit-exact on its checked logits and generated IDs. Installed C
and C++ consumers, Linux/macOS host checks, ASan/UBSan, TSan and two 3000-run
parser fuzz smoke campaigns passed. CPU fault tests cover invalid sampler
options, missing MTP, accepted-count bounds, poisoning and retryable cleanup of
multiple speculative graphs; those simulations are not kernel-numerics tests.

Executed results and exact source/model/binary identities are recorded in
`benchmarks/decode-2026-11-25.json`. Timings are qualification wall times, not
standalone throughput measurements or speedup claims. Context fixtures perform
actual prefill of repeated tokenized text; fitting or zeroing a KV cache is not
treated as long-context inference evidence.

The statistical diagnostics use 8191 counted positions per case and a preselected
0.035 tolerance. They exercise direct P=(0.7,0.3), rejection with Q=(0.2,0.8),
Dirac proposals, and deterministic top-k=1. They are finite regression diagnostics,
not proof of every distribution/filter combination. Mode-matched C/Zig parity
is distinct from comparing FP8 speculation with FP8 AR: the pinned implementation
already has path-dependent near-tie behavior. No tolerance is widened to mask a
C-versus-Zig divergence.

Remaining P3 work includes adaptive widths, prompt lookup, overlapping drafts,
prefix snapshots/shielding and independent long-context retrieval/quality
qualification. HTTP serving remains a later milestone, after these execution and
lifecycle contracts. No OpenAI-compatible serving or broad quality/performance
claim follows from this slice.
