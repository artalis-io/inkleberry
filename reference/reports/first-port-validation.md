# First C port slice — validation

25 November 2026. This is checkpoint/container and CUDA boundary validation,
not C model inference qualification. The pinned oracle/kernel revision is
`d10c0abb2c8990be81c08ba84c66a8333a691b33`; source hashes are in `UPSTREAM.json`.

## Implemented boundary

The C library owns its directory allocations and read-only mapping, returns
recoverable errors and has no CUDA dependency for checkpoint inspection. Its
memory entrypoint borrows immutable bytes. Tensor pointers remain borrowed until
close; callers must prevent close racing with reads. Failed calls clear output
pointers. Every injected allocation site is exercised under failure for both
memory and file entrypoints.

The CUDA declarations are generated from the pinned Zig ABI. An NVCC-compiled
probe includes the actual unchanged upstream bridge, checks all 76 declarations
against the CUDA C++ types and compares C/C++ size/alignment/field offsets for all
7 records (72 checks). A separate C program executes `extract_block` eagerly,
then replays a captured CUDA graph twice with changing inputs. Copies use the
same stream and explicit synchronization. Exact copied float data is compared;
this is not a numerical GEMV, attention or full-model oracle.

## Executed checks

| Environment | Check | Result |
|---|---|---|
| Local macOS ARM64 | `make -j4 check install-check` | Passed: unit/API/checked arithmetic/fault cleanup, immutable concurrent lookup + TLS errors, 46 format and 3 filesystem cases, C/C++ and static/shared consumers |
| Local macOS ARM64 | `make lint` with LLVM clang-format 18.1.8 | Passed: handwritten formatting, library invariants, pin/generated checks |
| core-1 Linux AMD64, GCC 13.3 | `make -j8 check install-check` | Passed; same host contract and cases |
| core-1, Clang 18.1.3 | `ASAN_OPTIONS=detect_leaks=1 make sanitize` | Passed ASan, UBSan, leak checks, malformed inputs and failure paths |
| core-1, Clang 18.1.3 | `make fuzz-smoke` with explicit installed GCC 13 toolchain | Passed 3,000 libFuzzer iterations under ASan/UBSan, 46 synthetic seeds |
| core-1, Clang 18.1.3 | `make thread-sanitize` | Passed four concurrent readers, 10,000 lookup/data/error iterations each |
| core-1, CUDA 13.5.37 | `make -j8 CUDA_ROOT=/usr/local/cuda-13.5 cuda-abi-check` | Passed 76 prototypes, 7 records, 72 layout checks |
| core-1, RTX PRO 6000 Blackwell Max-Q | `SQ_TEST_GPU_UUID=… make -j8 CUDA_ROOT=/usr/local/cuda-13.5 cuda-check` | Passed eager + two graph replays from C host |
| Same reserved GPU | SDK Compute Sanitizer memcheck, `--leak-check full --error-exitcode 99` | Zero errors, zero bytes leaked for this kernel/graph smoke test |
| core-1, official external Inkling main + MTP files | `tools/check_checkpoint.py` | Every name/dtype/rank/shape/offset/size and lookup matched independent Python JSON: 1,244 main + 54 MTP tensors |

The GPU is `GPU-5a215cb3-57b0-0a6e-d511-e90bb05c0e05`, a 97,887 MiB reported
RTX PRO 6000 Blackwell Max-Q Workstation Edition, 300 W power limit. Before each
GPU test it reported 2 MiB used and no compute processes. GPU 1 and existing
inference/evaluation installations were not modified. Work was isolated under
`/data/scratch/squid-c-port-20261125`. No models were downloaded or copied into
the repository; existing public-model files were opened read-only.

The descriptor check compares header identity and metadata, not entire payload
checksums or weight values. Main file: 19,526,019,168 bytes, header SHA-256
`4b1fa8297cb1a0f6484c707bef69e7f0f81d08325ccf2c5b25b86d528f64acf2`.
MTP: 635,801,416 bytes, header SHA-256
`753b21c80ea24bce28d63e40a64e49fc27b4a9c59b855837b99e563868dc4bc1`.

## Reproduce GPU checks

Use the build/check commands in README. An additional instrumented run is:

```sh
SQ_TEST_GPU_UUID=GPU-REPLACE-WITH-APPROVED-IDLE-UUID \
  python3 tools/run_cuda_smoke.py /usr/local/cuda-13.5/lib64 \
  /usr/local/cuda-13.5/bin/compute-sanitizer \
  --tool memcheck --leak-check full --error-exitcode 99 .build/cuda-smoke
```

Run sanitizer targets separately: they clean `.build`, so do not run them
concurrently with ordinary builds. CPU CI runs on hosted runners and never
schedules untrusted PR code on core-1. GPU qualification remains an explicit gate.

## Environment failures encountered

- Local macOS ASan binaries spun inside sanitizer runtime initialization before
  reaching `main`, with both Apple Clang and Homebrew LLVM 18. Those processes
  were stopped; no macOS sanitizer pass is claimed. Normal ARM64 checks passed;
  Linux sanitizer and fuzz checks above completed.
- Core-1's Clang auto-selected an incomplete GCC 14 installation for C++ runtime
  linkage. The C++ header consumer needs no C++ runtime and now compiles as C++
  and links with the C compiler. libFuzzer, which needs that runtime, was run with
  `SAN_CC='clang --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'`.
- `/usr/bin/compute-sanitizer` could not locate its injection library. The
  versioned installed SDK executable completed successfully. No host packages,
  global toolchains or driver configuration were changed.

## Remaining gates

No C tokenization, model-role/shape validation, weight upload/packing, full
forward/prefill/decode, sampling, speculative decoding, cancellation, API server,
full-inference leak safety, quality or throughput claim is made. The retained
bridge's original process-global cuBLAS lifetime still needs an explicit ownership
solution before library/session inference qualification. The next slice should
add tokenizer/weight binding with independent real-checkpoint parity, then a
non-speculative C forward/decode path against the pinned Zig oracle.
