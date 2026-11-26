# Zig-independent CUDA ABI — validation

The native C/CUDA declaration source is now `abi/cuda.json`. This change leaves
CUDA mathematics, host inference code and public interfaces unchanged. All three
generated artifact payloads match pre-change commit `0c59cd3` exactly after the
leading attribution comment; only that comment now identifies the JSON source.
The compatibility hashes are recorded in `test/fixtures/cuda_abi_v1.json`.

## Executed

| Check | Result |
| --- | --- |
| macOS ARM64 host/check/install/lint | Passed |
| Linux AMD64 host/check/install | Passed |
| Isolated C checks/install checks without any Zig files | Passed on macOS and Linux |
| Schema unit tests | Five groups passed: frozen payloads, types/arrays, malformed schemas, malformed JSON, isolated generator/failure behavior |
| Production provenance rejects altered CUDA input | Passed in isolated fixture; full verification still requires missing Zig files |
| CUDA compiler/probe against real bridge | 76 prototypes, seven records, 72 layout checks passed |
| Wrong return type | Rejected by CUDA C++ compiler |
| Wrong C record layout | Rejected by independent C/C++ size/offset comparison |
| Native serving-ABI workflow | Passed with normal checkout and without reference files |
| Optional C/Zig AR comparison | Passed: 1,000,290 logits and 32 generated tokens |
| Linux ASan/UBSan | Passed full host suite |
| Fuzz smoke | Four targets, 3000 runs each; passed |

The isolated generator test copies only the generator and schema into a fresh
tree, with no upstream manifest/helper, reference source, CUDA tree or preexisting
outputs. Generation and read-only verification succeed. Stale outputs and invalid
schemas fail without overwriting existing artifacts.

`make reference-free-check` copies only production/build/test inputs, excluding
both the reference directory and every `*.zig` harness. It runs the actual C
`check` and `install-check` targets. In that disposable copy it also verifies that
production provenance rejects altered CUDA input and that full provenance still
requires the pinned Zig sources. The Linux GCC CI job now runs this check.

## Native hardware and commands

Tests ran on `core` / `rtx-85b4` using the approved idle GPU UUID
`GPU-5a215cb3-57b0-0a6e-d511-e90bb05c0e05` (RTX PRO 6000 Blackwell Max-Q).
Build directories were isolated under `/data/scratch/squid-c-schema*-20261126`.
The original external Inkling checkpoint was reused; no weights entered the repo.

Normal native checkout:

```sh
make -j8 check install-check cuda-abi-negative-check .build/infer-native-shared \
  ar-oracle ZIG=/path/to/zig
SQ_TEST_GPU_UUID="$APPROVED_GPU_UUID" SQ_TEST_TIMEOUT_SECONDS=600 \
  python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 \
  .build/infer-native-shared "$MODEL_DIR"
SQ_TEST_GPU_UUID="$APPROVED_GPU_UUID" SQ_TEST_TIMEOUT_SECONDS=600 \
  python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 \
  .build/ar-oracle "$MODEL_DIR" 2 1 1 33 'What is 2 + 2? Answer briefly.'
```

A separate fresh native source tree excluded `reference/` and every `*.zig` file.
With `ZIG=/not-installed/zig`, this executed successfully:

```sh
make -j8 .build/infer-native-shared cuda-abi-negative-check ZIG=/not-installed/zig
SQ_TEST_GPU_UUID="$APPROVED_GPU_UUID" SQ_TEST_TIMEOUT_SECONDS=600 \
  python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 \
  .build/infer-native-shared "$MODEL_DIR"
```

Both native workflows returned `7`, with 876 prompt tokens, three generated target
IDs and 846 prefix tokens reused. The suite verifies adaptive speculation/lookup,
independent streams, BUSY handling, stale/active cancellation, cache invalidation,
scope isolation, stop withholding and replay. GPU 0 returned to 2 MiB; GPU 1 stayed
at 26 MiB and was not used. No services or system configuration changed.

Sanitizer/fuzz builds used the existing installed GCC 13 search-path selection:
`SAN_CC='clang --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'` and matching
`SAN_CXX` for the sanitizer build. No compiler/package installation was performed.

The first reference-free test-helper attempt accidentally copied optional Zig
harnesses along with the tests; its own no-Zig assertion caught this before a
build. The helper now excludes every `*.zig`, and both reference-free suites
passed. This was a test-isolation fix, not a production inference change.

## Scope

The original pinned Zig comparison oracle remains available, protected by
`make reference-check`. Removing that oracle requires independent comparison
coverage and is separate from this dependency removal. GPU memcheck, TSan and the
full long-context/precision benchmark matrix were not repeated for this generator/
build-check change. Earlier qualification is preserved; no new performance,
model-quality or platform-support claims are made. GitHub CI status is reported
separately on the PR.
