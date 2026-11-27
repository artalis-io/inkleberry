# Windows Ampere build validation — 27 November 2026

Tested source: `f9f7fc36d133b5ed7567aa7cb5ba007fa959e412`.
88 source/build inputs on the remote workspace matched local SHA-256 values.
This report records compilation and host-side checks, **not GPU qualification**.

## Environment

- `edge`: Windows 11 Pro, version 10.0.26300, x86-64.
- NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, 24,564 MiB VRAM.
- NVIDIA driver 621.01; CUDA toolkit/NVCC 13.3.73.
- MSVC compiler 19.44.35229 (toolchain directory 14.44.35207); LLVM clang-cl 23.1.1.
- Approximately 293 GiB free on the target drive before downloading assets.
- C hosts built separately with MSVC and clang-cl. NVCC used its MSVC host
  compiler in both profiles; this does not claim NVCC clang host support.

## Executed commands and results

From the isolated source workspace, after `source scripts/msvc-env.sh`, for each
of `CC=cl` and `CC=clang-cl`:

```sh
make -j2 CC=cl PYTHON=python CUDA_ARCH=86 EXPERIMENTAL_AMPERE=1 gpu-build
make -j2 CC=cl PYTHON=python CUDA_ARCH=86 EXPERIMENTAL_AMPERE=1 windows-cuda-build-check runtime-gpu-package
# Repeat both commands with CC=clang-cl.
```

Both compiler profiles passed:

- Static archives, native generator and shared provider compilation/linking.
- Only `ifn_infer_get_api` exported from the provider; no POSIX runtime imports.
- PE ASLR/DEP checks, ctypes ABI negotiation and capability discovery.
- 76 CUDA bridge prototypes, 7 records and 72 native layout checks.
- Runtime GPU packaging with explicit `sm86-experimental` identity.
- `cuobjdump --list-elf`: five `sm_86` images per provider; no Blackwell images.

The runtime package identifies ordered launches and experimental status, with
`qualification=not-established-by-build`. Packaging is not an execution test.

A Linux CPU CI failure was corrected in the dry-run isolation test: recursive
Make directory messages must not be parsed as `rm` operands. The local 14-profile
compiler/build test passed after that fix; hosted CI was still queued at reporting.

## Unexecuted GPU gates and actual blockers

The user released the machine for testing. Read-only NVIDIA checks reported 0%
GPU utilization, but approximately 1,518 MiB of desktop VRAM and Windows/UI
processes in the compute-app process listing. The unchanged guard returned
`GPU idle VRAM preflight failed`; its test child did not execute. The current
contract permits at most 256 MiB and no listed processes. No threshold was raised,
process allowlist added, GPU guard bypassed or unrelated process stopped.

A narrowly scoped Windows desktop reservation mode was proposed to the owner;
that policy change is pending a decision. GPU primitives, graph replay, inference,
conditional SWITCH graphs, cancellation, prefix reuse, numerical agreement and
performance have therefore **not run**. Linux Ampere is also unqualified.

The official model revision `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`
was checked against existing golden SHA-256 values. The model download stalled
before receiving payload bytes; the owned download process was stopped. No
model bytes were published as valid and no existing model was overwritten.
Model acquisition must finish with size/checksum verification before execution.

No driver/toolchain installation, desktop app/service termination, clock/power
change, watchdog change or inference service mutation was performed.
