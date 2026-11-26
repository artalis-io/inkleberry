# Native Windows portability — 26 November 2026

This milestone qualifies native Windows host/build/SDK behavior separately from
GPU architecture and model-output qualification. The existing Blackwell device
gate and all retained CUDA source bytes remain unchanged.

## Machine and toolchains

The isolated checkout on `edge` was `C:/Users/Mark/inkleberry-windows-20261126`.
No global tools, drivers, services, power settings or workloads were changed.

- Windows 11 Pro x64, build 10.0.26300.
- Ryzen 9 5950X, 16 cores / 32 logical processors.
- RTX 3090 Ti, 24,564 MiB, compute capability 8.6; driver 621.01.
- CUDA 13.3.73; NVCC target `sm_120a`, host compiler MSVC.
- Visual Studio 2022 Build Tools 17.14.41; MSVC 19.44.35229, toolset 14.44.35207.
- LLVM clang-cl 23.1.1, GNU Make from MSYS2; native Python 3.14.
- Approximately 319 GB free on the workspace drive during initial preflight.

The GPU is Ampere and has display/workload allocations. It was not reserved or
used for model inference. Successful compilation/linking for Blackwell does not
qualify inference on this GPU or on Windows Blackwell hardware.

## Executed checks

From MSYS2 Bash, using the checked-in environment helper:

```sh
source scripts/msvc-env.sh
make -j4 CC=cl PYTHON=python check install-check install-contract-check windows-sdk-check
make -j4 CC=clang-cl PYTHON=python check install-check install-contract-check windows-sdk-check
make -j2 CC=cl PYTHON=python windows-cuda-build-check
make -j2 CC=clang-cl PYTHON=python windows-cuda-build-check
```

The host suites exercise ordinary and amalgamated implementations: parser/BPE
rejection cases, allocation failures, immutable checkpoint admission, SHA-256,
Unicode/stop streaming, mock GPU ownership and cleanup, independent request
streams, cross-thread cancellation, TLS diagnostics, ABI negotiation and frozen
public record sizes/offsets. These are host/mock checks, not inference evidence.

Additional portability regressions cover strict UTF-8 filenames, rejected device
paths, bounded file sizes, 1,000 mapping lifecycles with unchanged Windows handle
count, read-only view protection, 64 MiB and 8 GiB sparse mapped inputs, monotonic
clocks and 156 tick conversions against Python's independent integer arithmetic.
The 8 GiB mapping retains metadata-only allocation (peak injected heap 712,792
bytes), without materializing the payload.

Native static/shared installed C/C++ consumers, custom-prefix non-destructive
uninstallation, ctypes ABI consumers, deterministic source/embedding/runtime
archives and extracted C/C++ consumers run under both native compiler profiles.
PE checks enforce ASLR/DEP, private-symbol boundaries and absence of MSYS,
Cygwin, winpthread or libgcc runtime imports.

Both native compiler CUDA checks build the native GPU archive, generator and provider DLL,
verify its single export and negotiate its ABI without preparing a GPU
session. The CPU-only C/NVCC ABI probe checks 76 prototypes, seven records and
72 layout facts.

## Defects found and corrected during qualification

- Native archive output syntax and accidental selection of MSYS `link.exe`.
- A CLI import library overwriting the static archive; distinct artifacts and
  ordinary CLI objects now prevent that collision.
- MSVC C's missing `max_align_t`, implicit codepage decoding, text-mode stdin,
  duplicated clang-cl language switches and generated Windows path separators.
- Extra temporary UTF-16 allocations now included in allocator fault coverage.
- Native sparse fixture creation, physical-size verification and privilege-free
  fixture aliases replace POSIX-only assumptions.
- CUDA's driver import library contains UUID helpers requesting LIBCMT. Owned
  C/CUDA objects use `/MD`; `/NODEFAULTLIB:LIBCMT` resolves those helpers through
  the same selected CRT. Linker warnings remain fatal.
- Failed build outputs are removed, and configuration fingerprints include native
  linker/CUDA options to prevent stale-success artifacts.

## Remaining qualification

Windows model loading/generation, numerical equivalence, throughput, cancellation
under actual native GPU execution and CUDA graphs require a supported, reserved
Blackwell GPU on Windows. Ampere support is a separate task. Existing GB10
numerical qualification gaps remain unchanged. No Windows model-quality or
performance claim follows from these checks.

Linux GCC/Clang, ASan/fuzz, TSan and lint remain the existing CI gates. Two native
Windows compiler jobs add host/SDK coverage; CUDA compilation remains an explicit
operator-run check on a machine with an existing toolkit.
