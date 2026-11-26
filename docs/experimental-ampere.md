# Experimental Ampere compatibility layer

Implementation status: host-tested; **CUDA compilation and GPU execution of
this implementation are deferred**. The owner requested that `edge` remain
untouched while it is being used for gaming. No SSH connection, remote build,
GPU test, benchmark, model load or service change was performed there for this
implementation. The earlier [audit](design/ampere-compatibility.md) compiled the
unchanged baseline; that result does not qualify the new code.

## Behavior

The default remains Blackwell `CUDA_ARCH=120a`; `121a` selects the GB10 image.
An explicitly experimental Ampere build requires both `CUDA_ARCH=86` and
`EXPERIMENTAL_AMPERE=1`. Unsupported targets, missing opt-in and host/device
architecture mismatches fail closed. Runtime admission matches the selected
profile exactly: a 120a image does not admit 12.1, and an 86 image admits only
8.6. The public C ABI and all CUDA bridge layouts/prototypes are unchanged.

Blackwell retains its existing PDL launch configuration. Ampere leaves those
attributes absent, using ordinary ordered stream launches with the same kernels,
dimensions, arguments and stream. Device conversion fallbacks are reused; there
is no new weight format, quantizer, CPU generation backend or numerical rewrite.

For Ampere, all eight grouped-prefill specializations prepare their shared-memory
opt-in in the current session context before cuBLAS creation and graph capture.
Preparation checks static plus requested shared memory and function thread limits,
returns the first CUDA failure, and performs no device-wide synchronization.
There is no successful-result cache shared across devices/contexts. Ampere's
captured launch path does not make those preparation queries inside capture.
Blackwell's existing prefill preparation path remains unchanged.

The retained-source changes are exact host hooks in
`config/native-platform-adaptations.json`. Reversing those hooks, then the older
error adapters and identifier map, must reconstruct the original upstream hashes.
Device numerical edits remain detectable. The new private headers own launch
policy and checked resource preparation; kernel implementations are shared.

## Build and artifact contract

These are **build-only commands for later use**, not commands executed on `edge`:

```sh
# Native Windows: source scripts/msvc-env.sh first; choose CC=cl or clang-cl.
make -j2 CUDA_ARCH=86 EXPERIMENTAL_AMPERE=1 gpu-build
make CUDA_ARCH=86 EXPERIMENTAL_AMPERE=1 print-build-dir
make CUDA_ARCH=86 EXPERIMENTAL_AMPERE=1 runtime-gpu-package
```

`gpu-build` compiles/packages dependencies without launching inference or GPU
tests. It does not establish CUDA compatibility until it has actually been run
with a toolkit, nor numerical correctness after compilation.

POSIX Ampere objects, archives, provider, host utilities and generator live below
`.build/sm86-experimental`; a selected `BUILD_DIR` becomes its base directory.
Native Windows retains separate compiler/architecture output trees, such as
`.build/windows-msvc-x64-release-cuda86`. Experimental clean/build commands do
not delete or overwrite default Blackwell artifacts. Source membership remains
in `config/sources.mk`, using the existing Make graph.

GPU builds emit `cuda-profile.json`. GPU runtime packages require that identity,
include it as `CUDA-PROFILE.json`, and add the CUDA target to the archive name;
Ampere additionally carries `experimental`. The metadata explicitly says that
qualification is not established by the build. Old GPU builds without this new
identity must be rebuilt before packaging. Host-only packages retain their
existing naming and do not acquire a CUDA runtime dependency.

## Executed host validation

The local `cuda-policy-check` uses fake CUDA host APIs, never a GPU:

- Positive host/device profiles and compile-time rejection of unsupported,
  unapproved or mismatched targets.
- Exact argument, dimension and stream forwarding; PDL attributes only on
  Blackwell, no attributes on Ampere; first launch error preservation.
- Resource arithmetic boundaries, invalid inputs, function thread limits, fresh
  device/context queries and failures at every API boundary.
- All eight real preparation-list entries, their kernel identities and audited
  memory sizes, with early termination at each possible failed opt-in.
- Make build/clean/package isolation, opt-in validation, explicit profile metadata
  and distinct package names, using small non-confidential packaging fixtures.
- Reconstruction of pinned upstream bytes and rejection of missing, ambiguous
  or device-body platform transformations.

These tests are part of ordinary host CI, including hosted Windows MSVC/clang-cl.
They do not contact `edge` or use a CUDA driver. Existing public ABI, allocator,
concurrency, cancellation, parser and distribution regressions remain in place.

## Deferred acceptance gates

Before enabling this for actual use: compile/link with real NVCC under both
Windows host compilers, inspect target images/exports, run native layout checks,
then qualify primitives, eager/graph AR, fixed/adaptive speculation, prefix reuse,
real cancellation and cleanup on a reserved device. Verify conditional SWITCH
graph support and resource opt-in through real execution. Keep Blackwell goldens
and performance gates intact; report any numerical drift explicitly.

Do not bypass the existing idle-GPU guard, change the desktop's driver/watchdog
settings or interrupt gaming to satisfy these gates. Real inference, benchmarks,
maximum context and performance claims remain on hold for separate authorization.
