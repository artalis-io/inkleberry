# Inference ABI validation — 25 November 2026 UTC

This is execution evidence for the experimental facade, not HTTP/harness
certification. No server layer was added and no inference service was restarted.
The source manifest in [the receipt](../benchmarks/infer-2026-11-25.json) identifies
the tested library sources; the pinned upstream files still verify unchanged.

## Executed checks

| Check | Result |
| --- | --- |
| macOS ARM64 `make check install-check lint` | Passed; host-only, no GPU claim |
| Linux AMD64 host checks/install checks | Passed |
| Linux ASan/UBSan full host checks | Passed, including model-load allocation faults |
| Linux ThreadSanitizer | Passed, including 100 completion/cancellation races |
| Fuzz smoke | Four targets × 3000 runs; no findings |
| Public-only C and C++ consumers | Passed, using installed headers/static facade |
| Python ctypes against mock and real shared provider | Passed negotiation/discovery |
| Dynamic provider export allowlist and SONAME | Passed; only `sq_infer_get_api`, `SQ_INFER_1.0` |
| Staged GPU provider installation/pkg-config/shared consumer | Passed; no system install |
| Native static/shared consumers on reserved RTX PRO 6000 | Passed |
| Final shared workflow under CUDA Compute Sanitizer memcheck | Zero errors; zero bytes leaked |
| Existing native conversation/streaming regression | Passed |
| Existing native prefix recovery regression | Passed; prefill cancellation returned in 0.173 s |

Host execution tests deliberately use a fake native backend. Model-loading fault
tests reuse the existing sparse metadata fixtures; their payloads are not real
weights. These tests establish validation, ownership and concurrency behavior.
Only the separate native tests establish the recorded GPU behavior.

Host regressions include ABI mismatches, output-tail canaries, zero/nonzero input
tails, malformed array extents, required-feature rejection, memory budgets before
GPU acquisition, parent/child ownership, retryable close, refusing work after failed
close, copied input/stops, independent streams, queued and active cancellation,
stale cancellation during a subsequent stream, late completion cancellation,
callback stop/error, cache-scope changes and poisoning after failed cancellation
cleanup. Exhaustive allocator failures are injected through model loading.

## Native setup and observations

Controller: macOS ARM64. GPU host: `core` / `rtx-85b4`, Linux AMD64. Isolated build
and host-check directories under `/data/scratch/squid-c-infer*-20261125`. Existing
external Inkling checkpoint directory reused; no weights copied into the repo.
GPU: `GPU-5a215cb3-57b0-0a6e-d511-e90bb05c0e05`, RTX PRO 6000 Blackwell Max-Q.
All native tests used `tools/run_cuda_smoke.py` with the explicit idle-UUID guard.
GPU 0 returned to 2 MiB afterward; GPU 1 remained at 26 MiB and was not used.

The native public-ABI consumer prepared six independent requests before execution.
It used the actual chat template, NVFP4+MTP weights, FP8 KV, graphs, adaptive seven-
draft decoding and prompt lookup. It overwrote caller-side input/options after
preparation, then verified:

- Multi-turn arithmetic answer `7`; 876 input tokens and 3 generated target IDs.
- Repeated response equality and 846 reused prompt tokens within the same scope.
- A competing stream returns BUSY without consuming that request.
- Cancelling an old completed request does not cancel the currently active stream.
- Accepted active cancellation terminates as CANCELLED; replay is correct and
  cannot reuse the cancelled state.
- Changing cache scope forces zero reuse while preserving the answer.
- A textual `7` stop is withheld and produces STOP_STRING with no visible content.

CPU arena planning reported 16,645,882,628 peak owned weight bytes and
1,180,898,816 owned session-arena bytes for that configuration. These are allocation
plans, not total process/device memory measurements. An uninstrumented earlier
shared run took about 0.385 s for the short first generation; final memcheck took
about 9.548 s. Neither is a throughput benchmark or a multi-stream capacity claim.

## Reproduction

Host:

```sh
make -j4 check install-check
make lint CLANG_FORMAT=/path/to/clang-format-18
make sanitize
make fuzz-smoke
make thread-sanitize
```

On this core host, libFuzzer's C++ runtime lookup needed the installed GCC 13
location explicitly selected. No global toolchain/package changes were made:

```sh
make sanitize SAN_CC='clang --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13' \
  SAN_CXX='clang++ --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'
make fuzz-smoke SAN_CC='clang --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'
make thread-sanitize SAN_CC='clang --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13' \
  SAN_CXX='clang++ --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'
```

Native, with an approved idle GPU UUID and external model path:

```sh
make -j8 .build/infer-native .build/infer-native-shared .build/chat-native .build/prefix-recovery
make infer-provider-check
SQ_TEST_GPU_UUID="$APPROVED_GPU_UUID" SQ_TEST_TIMEOUT_SECONDS=900 \
  python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 \
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 91 .build/infer-native-shared "$MODEL_DIR"
SQ_TEST_GPU_UUID="$APPROVED_GPU_UUID" SQ_TEST_TIMEOUT_SECONDS=600 \
  python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 .build/chat-native "$MODEL_DIR"
SQ_TEST_GPU_UUID="$APPROVED_GPU_UUID" SQ_TEST_TIMEOUT_SECONDS=600 \
  python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 .build/prefix-recovery "$MODEL_DIR"
make DESTDIR="$PWD/.build/provider-stage" PREFIX=/usr/local install-infer-backend
```

The first memcheck invocation selected `/usr/bin/compute-sanitizer` (2022.4.1),
which failed before testing with a missing injection library. Using the installed
CUDA 13.5 toolkit's explicit binary path resolved it. The initial native consumer
also caught an incorrect counted format-ID length in the test; it now derives the
length from the literal. These failed attempts are not counted as passing checks.

## Remaining qualification

No real concurrent GPU generations, continuous batching, multi-GPU execution,
other model architectures, external proprietary/open harness integrations, HTTP
server, long-context matrix or performance comparison was tested for this ABI
slice. Artifact content hashing is not implemented by the facade and is reported
unverified. Future minor-version compatibility cannot be proven against an
implementation that does not yet exist. GitHub CI is reported separately in the PR.
