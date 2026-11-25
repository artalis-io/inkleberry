# Session-private prefix caching

This slice translates the pinned Zig four-checkpoint prefix cache and bounded
side-request shield into the C session. Original CUDA/Zig files remain
hash-verified at `d10c0abb2c8990be81c08ba84c66a8333a691b33`. It builds on the
[adaptive/lookup slice](adaptive-validation.md), without changing kernel math.

## Public API and scope

Caching is disabled by default. Enable with
`sq_session_prefix_cache(session, 1)` and disable with zero. Enable is idempotent;
allocation failure frees partial allocations and leaves caching disabled. The
allocator/context belong to the session's existing injected allocator. Neither
enable nor disable mutates GPU data. Calls return BUSY during active operations.
There are no allocations of cache buffers during generation.

Initialize `sq_decode_options` with `SQ_DECODE_OPTIONS_INIT`. The new
`system_prefix_tokens` is an advisory checkpoint boundary, honored only for an
enabled cache with `128 <= hint < prompt length`. `prefix_reserved` must be zero.
Both earlier options layouts (the original 40-byte prefix and adaptive 56-byte
prefix on the supported 64-bit platforms) remain accepted with new fields zero.
Generation and decode statistics keep their existing layouts.

Initialize `sq_prefix_info` with `SQ_PREFIX_INFO_INIT` and call
`sq_session_prefix_info` after a request. It reports reused input tokens, shielded
global-KV tokens, local checkpoint restores and allocated host bytes. Live reuse
has nonzero reused tokens and zero checkpoint restores. It is a read-only BUSY
query during execution. Diagnostics describe work for the completed request,
including interrupted requests; they do not authorize reuse after cancellation.

The cache is owned by one session with immutable model image, precision, KV
format and visit schedule. Exact input token history and AR-versus-MTP identity
must match before reuse. Sampler changes do not invalidate token-conditioned KV;
each request still gets a fresh sampler configuration. Unsupported LoopCD and
other model algorithms are not added here. A cache hit never skips the final
prompt token needed for fresh logits/sampling.

Keep a cache within one application trust scope. Reset/disable when reassigning
a session to another principal; no multi-tenant isolation or cache encryption
is implied. Nothing persists to disk or survives session destruction. There is
still only one live C session per process and no HTTP server.

## Checkpoints and state ownership

| Slot | Snapshot point |
|---|---|
| 0 | Requested shared/system prefix boundary |
| 1 | Last complete 2K boundary strictly before the end of the prompt |
| 2 | End of prompt, before decoding changes local state |
| 3 | End of committed model execution, including verified internal tokens |

Snapshots include all local BF16 KV, target convolution rings and MTP local KV
and recurrent ring when loaded. Global target KV stays in the existing device
arena. The largest valid checkpoint at or before the exact common prefix can be
restored, or an exact live position reused. Checkpoints beyond a branch point
are invalidated before prefill overwrites global KV. The target context is not
clamped to DSpark's separate proposal-cache capacity.

A short unrelated request can preserve a long main history: when main history
has at least 1,024 tokens, common prefix is below 512 and
`prompt length + maximum output + 16 <= 4096`, save the affected global KV prefix,
execute the side request from reset state, then restore that KV region. Keep
main token/checkpoint metadata and mark live local state unusable; the next main
request restores a host checkpoint. FP8/Q4 scales are included. The shield is
bounded by the actual session context; it never reads past a smaller arena.

Generation tracks tokens actually fed to the model, not just published tokens.
Speculative verification may commit more tokens than an output limit/callback
publishes. Checkpoint positions and token identity therefore follow verified
state, preventing reuse of a position with the wrong history.

Snapshots become valid only after all copies and synchronization succeed.
Adaptive trailing work is drained before final snapshots or reuse. Cancellation
and execution/context-restoration errors invalidate all reuse metadata. Explicit
session reset and direct logits mutations do likewise. Failed CUDA execution
poisons the session as before. Host snapshots are freed only after session GPU
resources have been retired, so failed destruction remains retryable.

This preserves the pinned cache policy. It is not an LRU across conversations,
a disk cache, or a new scheduler. At the 8K-test configuration (8,256 context slots, FP8/MTP), the buffers
consume 1,031,766,272 host bytes (about 984 MiB); local snapshot sizes dominate. No extra device arena
is allocated. Enabling caching for a one-shot CLI request offers no persistent
reuse; it is an explicit ownership/test option.

## Reproduce

```sh
make -j8 check install-check
make lint                         # clang-format 18
make sanitize
make thread-sanitize
make fuzz-smoke
make -j8 CUDA_ROOT=/usr/local/cuda-13.5 ZIG=/path/to/zig prefix-oracle
export SQ_TEST_GPU_UUID=GPU-REPLACE-WITH-APPROVED-IDLE-UUID
python3 tools/validate_prefix.py --model /external/model-directory \
  --cuda-lib /usr/local/cuda-13.5/lib64
python3 tools/validate_prefix.py --edges --model /external/model-directory \
  --cuda-lib /usr/local/cuda-13.5/lib64
python3 tools/validate_prefix.py --long --model /external/model-directory \
  --cuda-lib /usr/local/cuda-13.5/lib64
make CUDA_ROOT=/usr/local/cuda-13.5 .build/prefix-recovery
python3 tools/run_cuda_smoke.py /usr/local/cuda-13.5/lib64 \
  .build/prefix-recovery /external/model-directory
SQ_TEST_TIMEOUT_SECONDS=3600 python3 tools/run_cuda_smoke.py \
  /usr/local/cuda-13.5/lib64 /usr/local/cuda-13.5/bin/compute-sanitizer \
  --tool memcheck --leak-check full --error-exitcode 99 \
  .build/prefix-recovery /external/model-directory
```

All model files stay external; commands download nothing. The GPU guard requires
an explicitly selected idle device. Zig 0.17.1 is used only for oracle tests.
The C library, CLI and recovery executable have no Zig runtime dependency.

The oracle imports the real pinned Engine. It checks every emitted ID,
acceptance counters and reused-token counts across cold, identical, appended,
branched and truncated prompts, live versus restored state, side requests,
sampler/mode changes and explicit invalidation. The separate edge matrix checks
129-token checkpoint boundaries, greedy/stochastic changes and MTP side requests. The test explicitly invalidates
Zig's host metadata for reset parity, because upstream Engine.reset alone does
not clear that metadata. No original implementation files are patched.

The CPU suite injects every cache-buffer allocation failure, checks rollback and
idempotency, BUSY queries, old options-prefix reads, live/side-request selection,
failed restore invalidation and destruction ownership. The C-only native suite
checks callback stop, decode/prefill cancellation, replay after cancellation,
reset, direct logits and disable/re-enable, including full, stopped and cancelled
side requests. CPU simulations do not prove GPU
numerics or memory safety.

## Executed evidence

All **942 prefix C/Zig request comparisons** passed: 36 ordinary configurations
of 21 requests, 18 edge configurations of eight requests, and two long
configurations of 21 requests. The long runs demonstrate **131,040 reused
tokens** in FP8 and Q4. Another 28 cache-disabled AR/fixed/adaptive comparisons
passed. Native C recovery, installed C/C++ consumers, host checks and sanitizers
passed; the final patched C-only full recovery run reported zero memcheck errors
and zero leaked bytes.

Final review found and fixed cancellation arriving during final checkpoint
publication. The deterministic CPU regression fails without the invalidation
guard; patched recovery, cache-disabled regressions and full memcheck were run
after the fix. The guard changes cancellation cleanup, with the successful
inference path and CUDA mathematics preserved. Native prefill cancellation
returned in 0.171993 seconds in one observation; instrumented timing is recorded
separately and is not a native latency measurement.

The checked-in [manifest](../benchmarks/prefix-2026-11-25.json) identifies executed
cases, source/model/toolchain hashes and limitations. Qualification timings are
single-run wall times, not throughput/TTFT benchmarks or evidence of a universal
speedup. Million-token independent retrieval, serving and full performance and
quality characterization remain separate release gates.
