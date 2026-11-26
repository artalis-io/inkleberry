# Inkleberry Finn — agent engineering guide

Inkleberry Finn is a correctness-first C11 library and native GPU inference host
for Inkling-2-Mini. Follow the small-library engineering approach used in Jev Bush:
explicit ownership, narrow typed interfaces, modular reviewable code, generated
amalgamation, installed-consumer checks and evidence before performance claims.
Do not import Jev Bush's model algorithms or imply its CPU inference applies here.

## Read first

- `README.md`, `docs/build.md`, `docs/inference-abi.md` and `docs/embedding.md`.
- `UPSTREAM.json` and `THIRD_PARTY.md` before touching retained dependencies.

`main` is the integration/release branch. Use focused feature branches based on
`main` and require Linux CI before merging. Dependency provenance remains recorded in `UPSTREAM.json`.
Never replace maintained inference code with unqualified upstream changes.

## Architecture and naming

| Path | Responsibility |
| --- | --- |
| `include/ifn*.h` | Public API, types and ownership contracts |
| `src/` | Canonical C host, readers, tokenizer, inference and execution state |
| `src/cuda/` | Checked adapters around the retained native implementation |
| `cuda/` | Namespace-adapted CUDA C++20 kernels and bridge |
| `abi/cuda.json` | Internal bridge schema; generated declarations/layout checks |
| `config/sources.mk` | Canonical source groups for modular/amalgamated builds |
| `dist/ifn.h`, `dist/ifn.c` | Generated embedding distribution |
| `cli/`, `examples/` | Thin clients of the library |
| `test/`, `fuzz/` | Public consumers, regression tests and malformed-input checks |
| `tools/` | Maintained Python standard-library build/validation helpers |
| `reference/` | Historical designs and recorded qualification evidence only |

Public functions/types use `ifn_`, macros/constants `IFN_`, environment controls
`INKLEBERRY_`. The CLI is `inkleberry`; library names begin with `libinkleberry`.
The backend identity is `artalis.inkleberry`. There are no pre-release `sq_` API
aliases. All project-owned native symbols, including internal helpers, CUDA
namespaces, kernel functions and bridge records, use `ifn_` / `IFN_`. Local
variables/members and identifiers required by CUDA, the standard library, `main`
and libFuzzer retain their external conventions.

## Library contract

- Use opaque owned handles, typed requests, explicit status returns and initialized
  error outputs. Document borrowed spans, allocator lifetime and thread safety.
- No process termination, assertions on external input, hidden process-global
  device ownership, shell commands or hidden allocation in new library code.
- Route owned allocations through the injected allocator. Check all size, offset,
  shape, narrowing and allocation arithmetic before use; bound parser work.
- Validate checkpoint tensor roles/shapes before device upload. Keep mapped files
  immutable for the full mapping lifetime. mmap is not zero-copy GPU upload.
- Keep cancellation request-owned, cleanup complete and terminal callbacks unique.
  Callbacks borrow spans and must return promptly. Do not silently change sampling,
  KV precision, graph semantics, speculative acceptance or prefix-cache isolation.
- Multiple request streams do not imply parallel GPU execution. The current
  provider permits one prepared GPU session and one active generation per process.
- Keep CLI policy, scheduling and future HTTP transport out of model mathematics.
  Do not add frameworks or dependencies for routine library changes.

## Generated and pinned files

Edit canonical modules and regenerate with `make amalgamation`. Do not hand-edit
`dist/`. Update `config/sources.mk` when source membership changes.
Generate the internal CUDA contract with `tools/generate_cuda_abi.py` from
`abi/cuda.json`; never hand-edit generated declarations or layout assertions.
`make generated-check` verifies retained dependency pins and schema output.
CUDA files carry a modification notice; `tools/native_namespace.py` reverses
`config/native-namespace.json` and the exact reviewed host-only error adapters in
`config/native-host-adaptations.json` to reproduce the original pinned bytes.
Do not expand the map to hide algorithm changes. A dependency update requires
an explicit reviewed change, never a silent hash refresh.
Preserve Apache-2.0 LICENSE, NOTICE and translation attribution.

## Validation

For host/API/build changes run:

```sh
make -j4 check install-check install-contract-check
make lint                       # clang-format 18, configurable CLANG_FORMAT
make reference-free-check
make release-check
make sanitize
make fuzz-smoke amalgamation-fuzz-smoke
```

Tests must exercise behavior and failure boundaries: allocation faults, malformed
inputs, ownership, concurrency, cancellation, mapping lifetime, symbol visibility
and extracted C/C++ consumers. Keep existing regression coverage. Add focused
regressions for meaningful bugs; avoid tests that merely mirror implementation.
Changes to inference, kernels or GPU lifetime also require model-backed native
qualification. CPU tests and compilation cannot establish numerical equivalence,
GPU resource safety, output quality or speed. Report executed and pending checks
separately, including toolchain/model/device/configuration used.

GPU tests require `INKLEBERRY_TEST_GPU_UUID` identifying an explicitly approved
idle GPU and `tools/run_cuda_smoke.py`'s guard. Never stop unrelated workloads,
change clocks/power limits, install global toolchains or modify inference services.
Use isolated workspaces and external model inputs. If no idle device is available,
complete independent checks and report native verification pending.

## Style and distribution

Use C11, four-space indentation, snake_case and explicit bounds/error handling.
Handwritten C follows clang-format 18; do not format pinned CUDA or generated files.
Keep patches focused; no unrelated algorithm, precision or tuning changes.

Releases use explicit allowlists in `config/package.json`, checksums and extracted
artifact tests. Update the allowlist for maintained file additions/renames. Keep
reference material out of production packages. Never commit model weights, keys,
private data, local build outputs or bulk evaluation artifacts. Runtime packages
need no Python/Zig; CUDA/driver dependencies remain explicit host prerequisites.
Do not claim correctness, performance, context support or quality beyond measured
qualification. Keep outstanding validation gaps visible in documentation and PRs.
