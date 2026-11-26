# Native Windows portability — design for review

Status: proposed; discovery and source review only. No implementation, compiler
probe builds, GPU execution, model downloads, service changes or host provisioning
were performed for this design.

Scope confirmed by the owner: **Windows portability first; Ampere qualification
is a separate workstream.** This milestone covers the native Windows host, build,
ABI, test and packaging contracts. It does not widen GPU capability admission or
change CUDA numerical paths. The remaining technical recommendations below are
still presented for review.

## Outcome and boundaries

Support native Windows x64 C consumers using **MSVC cl** and **LLVM clang-cl**,
with modular/static, host DLL, generated amalgamation and versioned inference
provider builds. Keep the C host C11 and the existing CUDA implementation C++20.
Keep numerical algorithms, model formats, quantization, graph behavior, prefix
caching, request ownership, cancellation and execution limits unchanged.

A Windows port is justified. It is not yet equivalent to running inference on
`edge`: that PC's GPU is Ampere, while the currently admitted devices are
Blackwell. OS, host compiler and GPU capability are independent axes.

Do not add an HTTP server, worker pool, scheduler, CPU model backend, new model
format, CMake project or generic platform framework to accomplish this port.
Do not compile the C host as C++ to bypass C11/compiler issues. No dependency on
Keel, Jev Bush or bitnet.c is introduced by studying their engineering patterns.

## Discovery evidence

Reviewed Inkleberry revision: `fc50d29817ccbd902ea355f84d8ce0e322b8ca53`.

Read-only SSH discovery on `edge` found:

| Item | Observed |
| --- | --- |
| OS | Windows 11 Pro x64, version 10.0.26300 |
| CPU | Ryzen 9 5950X, 16 cores / 32 logical processors |
| GPU | RTX 3090 Ti, 24,564 MiB, compute capability **8.6** |
| Driver | 621.01 |
| CUDA | 13.3.73, installed beneath `CUDA_PATH` |
| MSVC | VS 2022 Build Tools 17.14.41; cl 19.44.35229.0 |
| LLVM | clang / clang-cl 23.1.1, target x86_64-pc-windows-msvc |
| Other tools | GNU Make and Bash in C:/msys64/usr/bin; native Python 3.14 via py; CMake 4.4.3 and Ninja 1.13.2 also installed |
| Disk | About 319 GB free on C: at discovery |

`cl` is installed but not in the ordinary SSH PATH; use an imported VS developer
environment for future builds. The installed CUDA host_config.h accepts
1920 <= _MSC_VER < 1960, covering this cl version. That is a prerequisite check,
not proof that this program compiles. C11 atomics headers are present; their
actual operations and TLS still require compile/run probes during implementation.

The GPU reported roughly 1.2 GiB used and numerous WDDM process records with
unavailable per-process memory. Neither zero utilization nor N/A proves that it
is free. No processes were stopped or reset.

### Patterns from the other libraries

| Checkout and inspected revision | Useful pattern | Limits to copying it |
| --- | --- | --- |
| Jev Bush, `85fd51f8c99b1b8b7ea26d33e02413e681a9271d` | Native cl command and Windows selftest CI; CreateFile/MapView ownership; QueryPerformanceCounter; public DLL annotations; generated distribution | Its mapper uses CreateFileA and can fall back to a whole-file allocation. Inkleberry should use strict UTF-8 conversion and preserve bounded-heap mapping. Its CLI error strategy and CPU/CUDA backend are different. |
| Keel, `accb8c330106b8745448041079e37b43e6b2bff3` | One Make graph plus mk/toolchain.mk; native MSVC environment setup; separate OS/compiler seams; native thread tests; derived suite parity and import checks | Do not import networking, event backends or its runtime threading layer. Its compiler profile recognizes cl/cl.exe; Inkleberry must explicitly account for clang-cl too. |
| bitnet.c, `ab9e8e8661371ee33fb086980b675aae88246de5` | Narrow platform file/time boundary; model/session/backend ownership boundaries | This checkout's platform implementation is POSIX/Emscripten, its Makefile is GCC/Clang-oriented, and CI has no Windows job. It is architectural guidance, not evidence of MSVC readiness. Do not copy its whole-file fallback, allocation/error policy or model/backend framework. |

Relevant inspected files: Jev Bush `jb.c`, `README.md`, Windows CI; Keel
`mk/toolchain.mk`, `scripts/msvc-env.sh`, `src/kl_atomic.h`, platform thread
implementation and `docs/msvc_build.md`; bitnet.c `include/platform.h`,
`src/platform.c`, `AGENTS.md`, `Makefile` and CI.

### Follow-up: OTTO and current Keel mapping/clock review

Read-only review requested specifically for the OS adapters:

| Source | Observed pattern | Inkleberry decision |
| --- | --- | --- |
| OTTO `ad177d22752cb8f30dc4666496984e4056c52397`, shared/include/sh_pal.h and shared/src/sh_pal_windows.c | ShFileMap is pointer + size; read-only view survives closure of file and mapping handles; null-safe unmap clears both fields | Adapt this minimal ownership model; no long-lived Windows handles are needed. |
| Same OTTO revision, shared/src/sh_pal_posix.c | Read-only private mmap; descriptor closed after mapping; empty file rejected; platform-local cleanup | Preserve this shape but retain Inkleberry's stricter regular-file, bound, nonblocking and inheritance checks. |
| Same OTTO revision, sh_monotonic_ns | QPC division split into whole seconds and remainder; POSIX CLOCK_MONOTONIC with a wall-time fallback | Reuse the division shape with checked arithmetic; retain Inkleberry's no-wall-fallback failure policy. |
| Keel `ff25928069ee1822c3529479f840a248999f644d`, src/platform_win.c, platform_posix.c and platform.h | Narrow OS services, separate OS TUs, QPC milliseconds with GetTickCount64 fallback | Reuse OS separation, not the public millisecond API or fallback policy. |
| Same Keel revision, src/file_io.c | Async-file-I/O construction is a NULL-returning stub, not a mapped-file implementation | Do not attribute a mapping backend to Keel or import its event/runtime model. |

OTTO vendors a separate Keel revision
`5916635132e69423cc6c1f8735f7ce29d4485b80`; the direct Keel checkout above was
reviewed separately. Earlier build/atomic findings retain their recorded revision.
No sibling repository was modified and no new test result is claimed from
reading these implementations. OTTO's shared PAL tests exercise its millisecond
clock, but this review did not identify dedicated mapping-lifetime or synthetic
QPC-conversion edge cases there; Inkleberry needs its own authoritative coverage.

## Inkleberry review: concrete gaps

| Priority | Current source | Finding / required treatment |
| --- | --- | --- |
| P0 | src/safetensors.c; src/tokenizer_load.c | Duplicated direct open/fstat/mmap/munmap, O_CLOEXEC/O_NONBLOCK and POSIX headers block native Windows. Extract a private mapped-file seam; do not change parsers. |
| P0 | Makefile | GNU driver/output/archive flags, POSIX shell tools, ELF SONAME/version-script flags, symlinks and Linux CUDA library paths are built in. Normalize toolchain intent once and retain the source graph. |
| P0 | src/infer.exports; include/ifn_core.h | Linux restricts provider exports with an ELF version script. A Windows .def file does not hide functions already marked dllexport. Provider and host-DLL compilation profiles must differ. |
| P0 | src/cuda/capability.h | Only 12.0/12.1 are admitted. edge's 8.6 is correctly rejected today. Do not widen this as part of an OS port. |
| P1 | src/infer.c | CLOCK_MONOTONIC needs a private Windows QPC implementation with checked conversion. |
| P1 | src/foundation.c; weights/infer internals | _Thread_local and C11 atomics need real cl/clang-cl probes. Atomics are private already; preserve memory ordering and nonblocking cancellation. |
| P1 | test/mock_gpu.h; test/mmap.c; pthread-based tests | Sparse no-access reservations, real read-only mapping checks, temporary files, threads and yielding are POSIX-specific. Port the harness semantics rather than omit the tests. |
| P1 | tools/release.py; test/native_release.py; export/install checks | GPU packaging explicitly rejects non-Linux. .so/.dylib, ELF tools and GNU consumer commands need Windows artifact handling. |
| P1 | diagnostics; source paths | OS family currently means errno; raw Win32 error codes need a distinct domain. File paths have no explicit cross-platform UTF-8 contract. Clarify both without changing record layouts. |
| P1 | tools/run_cuda_smoke.py | WDDM process/memory reporting does not justify weakening the idle guard. Windows GPU qualification needs an independently verified usable device. |

## Proposed architecture

### 1. Private mapped-file service

Use the narrow mapping pattern already implemented by OTTO's
`shared/include/sh_pal.h` and `shared/src/sh_pal_{posix,windows}.c`. Introduce a
private `src/file_map.h` contract, implemented in the two OS translation units
described below. Both parsers use the same small service:

- Open a nonempty regular disk file, with caller-specified minimum/maximum size.
- Return a read-only borrowed view and an owned pointer/length mapping record.
- Close an initialized record safely; partial failures unwind in reverse order.
- Resolve any path-conversion allocations through the existing caller allocator;
  do not hide a full-file heap copy or introduce a new allocator.

Preserve Inkleberry's POSIX read-only MAP_PRIVATE behavior, nonblocking open,
close-on-exec, regular-file validation and checked size limits. On Windows adapt
OTTO's CreateFile/GetFileSizeEx/CreateFileMapping/MapViewOfFile sequence, using
strict UTF-8 -> UTF-16 conversion and the W entrypoints instead of OTTO's ANSI
entrypoints. Native file/mapping handles are local construction resources; the
successful view retains its backing reference after both handles are closed.
Close those handles before publishing success, retaining only pointer and length,
as OTTO does. Capture the primary OS error before cleanup can overwrite it.

This revises the earlier retained-handle proposal: there is no need to retain
HANDLE fields in a long-lived mapping just to keep the view alive. It also avoids
promising a persistent share-deny-write lock after the handle closes. Checkpoint
immutability remains the caller/source-lifetime contract on both operating
systems; read-only mapping is not a snapshot or a lock against external writers.

Safetensors retains the mapping until close; open_memory still borrows external
bytes and never unmaps/frees them. Tokenizer verified-open hashes and parses the
same mapped bytes, then releases the temporary mapping as it does today.
Zero length, oversized input, directories and non-disk handles fail. Preserve the
existing distinction between malformed size/type and OS I/O failures. Device and
pipe namespaces must not turn an ordinary file-open API into blocking pipe I/O.

Proposed path policy: strict UTF-8 for Windows path entrypoints and native byte
paths on POSIX, with no locale/code-page fallback. Support relative paths and
ordinary drive/UNC paths; specify extended/long-path handling explicitly during
implementation. Do not silently change the facade's existing 4,000-byte directory
limit. Test Hungarian accents, supplementary Unicode, spaces and invalid UTF-8.

### 2. Private monotonic clock

Introduce a private `src/host_clock.h` contract, implemented beside the file
mapping functions in the appropriate OS translation unit. Use it from the facade
and benchmark/test timing sites. Use CLOCK_MONOTONIC on POSIX and QPC on Windows.
Follow OTTO's `sh_monotonic_ns` quotient/remainder conversion to avoid multiplying
the full tick count by 1e9, adding explicit range/frequency/overflow checks.
Do not assume MSVC long double provides extra precision. Test long uptime,
invalid frequency and overflow independently of the OS call.

Preserve Inkleberry's existing zero-on-failure convention and nanosecond unit.
Keel's clock is millisecond-based and can fall back to GetTickCount64; OTTO's
POSIX clock can fall back to gettimeofday. Do not inherit either fallback:
wall time is not monotonic, and switching clock families is not the current
Inkleberry timing contract. No public clock API or clock-domain conversion is
needed for this port.

### Implementation layout after OTTO/Keel review

Keep file mapping and timing as **separate private function contracts**, without
a platform vtable. Define them in `src/host_posix.c` and `src/host_win.c`, following
the siblings' one-OS-per-translation-unit separation. OS headers and branches
stay outside parsers and inference algorithms. Grouping their small definitions
by OS does not introduce shared mutable state or couple the two contracts.

List both self-guarded OS units once in HOST, following OTTO's approach; an
inactive unit uses a harmless declaration rather than an invalid empty C TU.
The compiler sees exactly one implementation. The amalgamation generator embeds
the same guarded source and needs no second platform source inventory or new
manifest group. Continue its existing collision checks, and verify no Windows
headers enter POSIX builds or public headers. Do not add shared/, Keel networking,
clock injection, a worker runtime or a new dependency to Inkleberry.

### 3. Compiler and build intent

Follow Keel's approach with a small `mk/toolchain.mk`, adapting only needed intent:
C/C++ standard selection, warning policy, object/executable output, archiver,
shared linking, dependency files and platform library spellings. OS selection
stays separate from compiler selection. `config/sources.mk` remains the single
source list used by builds and amalgamation; no second list in a PowerShell or
CMake project.

Initial compiler profiles:

| Profile | C host | CUDA translation units | ABI/runtime |
| --- | --- | --- | --- |
| Windows MSVC | cl /TC /std:c11, measured warnings treated as errors | NVCC with supported MSVC C++ host, existing C++20/device options | x64 Microsoft ABI, release /MD |
| Windows Clang | clang-cl /TC, C11, measured warnings treated as errors | Same NVCC + MSVC CUDA objects, not a new Clang-CUDA backend | x64 Microsoft ABI, release /MD |
| Existing Linux/macOS | Existing GCC/Clang flags | Existing opt-in NVCC route on Linux | Existing artifact and runtime contracts |

Clang here means LLVM clang-cl, not MinGW Clang and not compiling C as C++.
Clang-generated C objects and MSVC-hosted CUDA objects meet only through the
existing C ABI. Direct clang.exe and Clang-CUDA can be separate future profiles;
do not claim them tested by a clang-cl job.

Use native C11 atomics with cl's /experimental:c11atomics where required, as in
Jev Bush/Keel. Probe the actual types/operations used, including 64-bit CAS,
atomic_flag and acquire/release cancellation. Do not replace atomics with
volatile or silently add locks. Keep any necessary TLS spelling adaptation in
one private compiler header, only after a probe demonstrates the need.
The implementation does not add POSIX signal-handler requirements: Inkleberry
currently has no console/signal handler or internal background worker.

Use separate output directories for OS, host compiler, CRT mode and CUDA target.
Switching toolchains must never reuse stale incompatible objects, import libs,
PDBs or a shared output archive. Preserve GNU Make flag expansion on existing
platforms with before/after dry-run checks and actual Linux regression runs.
Do not call build variables LIB, INCLUDE, LINK or CL: they are MSVC environment
inputs. Correctly quote paths with spaces and keep the real MSVC link.exe ahead
of MSYS's unrelated link.exe.

An MSYS2 shell may host GNU Make, as it does for Keel. It is a build tool only:
shipped libraries/executables must import no msys-2.0, cygwin or winpthread DLL.
Adopt the existing VS environment only in the build process; do not alter global
PATH, install packages or modify SSH configuration. No automatic unsupported-NVCC
compiler override. Header dependency tracking must be correct or force a clean
rebuild with a documented check; silently stale headers are unacceptable.

### 4. DLL and allocator boundaries

Keep frozen public record sizes, offsets, enums, calling conventions, vtable
negotiation and opaque ownership. Windows LLP64 must pass the existing ABI
fingerprint and standalone C/C++ header consumers. Do not add Windows fields or
atomics to the public ABI, and do not pack structs globally.

Compile the host DLL with its existing IFN_SHARED/IFN_BUILD_SHARED annotations.
Compile provider-private objects without the broad low-level export annotations;
a Windows .def exports **only ifn_infer_get_api**. Reuse the same source modules
with a separate object profile, not a second implementation. Verify PE exports
with dumpbin or llvm-readobj, including absence of internals and C++ mangling.

Proposed Windows artifacts: `inkleberry.lib` static host;
`inkleberry-1.dll` with distinct `inkleberry-1.lib` import library;
`inkleberry_backend-1.dll` with its import library; distinct static inference,
GPU and CUDA archives. Keep POSIX names unchanged. Consumers need explicit
runtime DLL location without process-global search-path mutation.

Qualify /MD consistently for release objects and /MDd separately for debug.
Allocation and release stay in the owning module or the supplied callback pair;
never ask a caller to free library-owned memory across CRT boundaries. No FILE*,
C++ objects, exception state or CRT-owned strings cross the inference ABI.

Proposed diagnostic extension: keep provider family 1 for errno; append a
**Win32 provider family 5** for raw GetLastError values. No existing codes or
layouts change. Preserve the original error before cleanup and avoid a library-
global invalid-parameter handler. This is a review decision, not implemented API.

### 5. Tests and distribution

Production does not use pthreads; its synchronization is C11 atomics. Introduce
only test-side thread/yield/temp-file adapters. Prefer _beginthreadex plus join
and CloseHandle for CRT-using test workers. Preserve thread-local diagnostics,
stale-cancellation isolation, close-while-running, allocator faults and bounded
waits. Do not copy Keel's production thread-pool abstraction into Inkleberry.

Mock GPU arenas must remain sparse reservations: VirtualAlloc MEM_RESERVE with
PAGE_NOACCESS / VirtualFree on Windows, equivalent to the current PROT_NONE
mappings. Replacing them with malloc could commit many GiB and change the tests.
Use VirtualQuery for Windows mapping protection checks instead of /proc/self/maps;
verify partial-open cleanup, immediate construction-handle release, view validity
after that release, and repeated mapping lifetime. Keep native
read-only mapping and small-heap tests, including a sparse file beyond 4 GiB.

Derive Windows test enrollment from shared test inventory, with named exclusions
and reasons, following Keel. Do not create a handpicked smoke-only Windows suite.
Use platform-appropriate PE/import/ABI checks rather than skipping ELF checks
and calling that parity. TSan remains a Linux gate; do not invent a Windows TSan
pass. Add supported Windows sanitizers as additional checks, not replacements
for lifetime/concurrency tests.

Package explicit .exe/.dll/.lib artifacts without POSIX symlinks; retain allowlists,
checksums, deterministic ordering and extracted-source/SDK consumers. Declare
Windows CUDA/driver prerequisites instead of copying SDK/driver DLLs indiscriminately.
Source and embedding packages must exercise both cl and clang-cl consumers.

## GPU portability is a separate workstream

edge cannot establish Windows Blackwell inference correctness or speed. Its
3090 Ti is 8.6; today's native gate admits only 12.0/12.1. Do not widen that gate
or claim support solely because NVCC compiles or links a binary.

An Ampere study may be smaller than a new numerical backend: the retained
`cuda/kernels_common.cuh` already has software FP4 and FP8 conversion fallbacks,
and device PDL instructions are guarded for architecture >= 9.0. However its
**host** ifn_cuda_launch_pdl helper unconditionally adds the programmatic stream
serialization launch attribute. Other launch/attention/packing requirements,
working memory and correctness remain unqualified on 8.6.

If Ampere work is approved, first compile/audit the existing fallback paths for
sm_86, establish required launch capabilities and investigate serial-launch
alternatives. Device capability selection belongs in CUDA adapters, never in
_WIN32 or _MSC_VER branches. Retain the fast Blackwell path, quantization meaning
and original pins; any retained-source modification must use an explicit reviewed
provenance adaptation, not a hash refresh. Do not expand all weights to BF16 by
default: that could exceed 24 GiB. Measure actual memory and conversion numerics.

Gate Ampere admission on real model-backed checks, cancellation/context cleanup,
graph replay, speculation, teacher-forced logits and output-quality/performance
validation. The GB10 experiment's existing 17/28 cross-platform token matches
remain an unresolved numerical review item; an additional backend must not hide
that by replacing goldens. A Windows build neither fixes nor worsens it by definition.

## Implementation sequence and gates

1. **Approve boundaries.** Decide the four items in the review checklist below.
   Inventory/probe the actual MSVC/clang-cl C11, atomic, TLS and CUDA toolchain
   capabilities in an isolated edge directory. Do not install or upgrade tools.
2. **Host portability.** Add private mapping/time adapters and compiler intent;
   keep parsers, public ABI and inference state machines intact. Run modular,
   amalgamated, negative-input, file-lifetime and static/shared consumer tests
   under cl and clang-cl. Keep Linux regression checks mandatory.
3. **Provider and packaging parity.** Exercise mocked real ABI requests,
   concurrent cancellation, error ownership, export allowlists, CRT boundaries,
   installation and extracted artifacts on both compiler profiles.
4. **CUDA build/link validation.** Compile unchanged Blackwell CUDA kernels with
   installed NVCC/MSVC and consume them from each C host toolchain. Run ABI/schema
   probes that do not execute kernels. On edge, verify clean unsupported-device
   behavior with a lightweight scoped probe, not a model-sized allocation.
5. **Native model qualification.** Use a Windows Blackwell device for Blackwell
   runtime validation. Ampere qualification is outside this milestone. Use identical
   model hashes and explicit CPU/GPU/toolkit identities. Measure short and long
   real prefills, graphs, cancellation, prefix reuse and both speculative modes;
   compare numerical outputs, not only process exit and speed. Never alter edge's
   existing GPU workloads or desktop to force a guard pass.
6. **CI policy.** Add native Windows MSVC and clang-cl host jobs as required checks
   when they are green and the support contract is established. Keep the five
   current Linux checks required; optional macOS stays optional. GPU qualification
   remains an explicitly reserved manual gate, not an untrusted self-hosted PR job.

No stage may describe host-only Windows tests as successful native GPU inference.
No merge, deployment or host mutation is implied by accepting this design.

## Critical decisions for joint review

| Decision | Recommendation | Why it matters |
| --- | --- | --- |
| Windows versus edge's GPU — confirmed | Windows host/build/ABI first; Ampere qualification deferred to a separate workstream | Avoid mixing OS portability with a numerical backend change. Windows Blackwell runtime still needs suitable hardware. |
| Build/compiler contract | One Make/source graph; cl and clang-cl C11 host profiles; NVCC + MSVC CUDA host | Preserve C-first architecture and an unmodified CUDA implementation, without a competing build system or C++ host rewrite. |
| Mapping/path/diagnostic contract | Read-only bounded-heap mappings, strict UTF-8 Windows paths, explicit Win32 error domain | Preserve lifetime, memory, Unicode and error fidelity rather than merely compiling. Review the family-5 addition and path limits explicitly. |
| Support/acceptance claim | Windows host/ABI support and GPU support have separate gates; retain all numerical evidence | An RTX 3090 Ti cannot certify Windows Blackwell performance. Green CPU CI cannot certify either GPU backend. |

## Primary documentation consulted

- Microsoft: [C language standard switches](https://learn.microsoft.com/en-us/cpp/build/reference/std-specify-language-standard-version)
  and [MSVC C11 atomics support](https://learn.microsoft.com/en-gb/cpp/overview/what-s-new-for-visual-cpp-in-visual-studio?view=msvc-170).
- NVIDIA: [Windows CUDA toolchain guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
  and [Ampere application compatibility](https://docs.nvidia.com/cuda/ampere-compatibility-guide/).
- Local installed CUDA 13.3 host_config.h and compiler/header inventories take
  precedence over inferring compatibility from a newer online guide's table.
