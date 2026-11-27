# Experimental Ampere compatibility layer

Implementation status: native Windows build/ABI checks, real inference,
conditional graphs, recovery and the 168-run benchmark completed on `edge`'s
RTX 3090 Ti. **Cross-device numerical equivalence did not pass**: 11/28 complete
benchmark token sequences matched the Blackwell goldens, despite deterministic
Ampere repetitions. Keep this experimental. See the
[validation report](validation/ampere-windows-2026-11-27.md) for exact scope,
measurements, differences and remaining gates.

The user approved a Windows desktop reservation after the dedicated-device guard
correctly refused the desktop baseline. Tests used the explicit, expiring process
snapshot described below. No unrelated apps, services, drivers, power settings
or watchdogs were changed.

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

Build-only commands (executed on `edge` with both `CC=cl` and `CC=clang-cl`):

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

## Remaining acceptance gates

Windows primitives, eager/graph AR, fixed/adaptive speculation, conditional graphs,
prefix reuse, cancellation and recovery have executed successfully. That is not
complete numerical/quality qualification. Investigate cross-device and within-device
AR/speculative differences with logits and independent references; preserve the
existing goldens. Qualify broader semantic behavior, longer contexts, additional
failure injection, Linux Ampere and extracted native SDK consumers. Run fresh
Blackwell performance regression checks before a broader release.

Do not bypass the guard, alter driver/watchdog settings or interrupt gaming to
satisfy these gates. Build identity never means runtime qualification.

## Explicit Windows desktop reservation

The default dedicated-GPU guard is unchanged. An operator-approved Windows
maintenance window can instead set `INKLEBERRY_TEST_WINDOWS_RESERVATION` to an
external JSON snapshot, alongside `INKLEBERRY_TEST_GPU_UUID`. This mode is native
Windows only. The operator must inspect every listed process before authorizing
it; never capture an unknown or inference workload as a desktop baseline.

The snapshot contains `format_version: 1`, `uuid`, Unix `issued_at`/`expires_at`,
`max_vram_mib`, and a `processes` map keyed by decimal PID. Each process records
its exact NVIDIA `process_name` as `name` and CIM UTC creation timestamp as
`created`. Creation timestamps use
`CreationDate.ToUniversalTime().ToString('o')`. Keep snapshots outside the repo;
they contain local process identities and expire after at most two hours.

The memory ceiling must be greater than 256 and at most 4096 MiB; use the observed
desktop baseline plus a small explicit allowance. Tests still require 0% GPU
activity in two samples, known memory use, only approved live process identities,
and a fresh process/memory recheck after sampling. New processes, recycled PIDs,
identity changes, expired reservations and unobservable counters fail closed.
The guard adds the supplied CUDA binary directory to the Windows child PATH.
It does not close desktop applications, disable the watchdog or alter GPU settings.

This is a maintenance-window preflight, not a GPU scheduler or exclusivity lock.
The operator must keep the device reserved throughout the run. Another process
can still race the final check; do not claim isolation from this guard alone.

With a freshly reviewed reservation and approved GPU UUID exported, execute GPU
targets sequentially (MSYS2 shell with `scripts/msvc-env.sh` sourced):

```sh
make -j1 CC=cl PYTHON=python CUDA_ARCH=86 EXPERIMENTAL_AMPERE=1 \
  conversion-oracle-check gpu-packing-check sampling-kernel-check
```

The guard is mandatory; avoid parallel GPU targets. Runtime DLL discovery uses
`CUDA_RUNTIME_PATH` (`bin` on Windows, `lib64` on POSIX). Creating a reservation
requires a new review of the live desktop process identities; do not reuse the
expired or revoked snapshot from a previous test campaign.
