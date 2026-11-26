# C library hardening qualification — 26 November 2026

The inference implementation at `affd0d9afb15b485a8e8a1a93ce50be8d585a6af`
was built from a clean Git archive and compared with the pristine namespace
baseline `cb2b71773ab1a04dc574a644e01ad270fc6b6bd4`. Subsequent commits change
only package metadata, packaging tests and documentation; runtime sources,
headers, amalgamation, kernels and compiler flags are identical.

## Native correctness and performance

Device: one NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition on `core`,
UUID `GPU-e5f713d1-46ae-9f99-c16d-65de9e9df32a`, driver 595.104.02,
300 W configured limit. Pop!_OS 24.04, GCC 13.3.0, CUDA 13.5.37,
`sm_120a`, `-O3 --use_fast_math`; no GPU settings changed.
The guard checked the selected device was idle before each GPU consumer.

Both the engine and versioned facade ran seven public workloads across four
profiles (AR, fixed speculative, adaptive lookup and seeded sampling). Each group
had one warm-up and five measured repetitions, at most 96 generated tokens and
up to 8,844 actual prefilled tokens. Configuration: NVFP4 weights, FP8 KV, CUDA
graphs, context capacity 32,768, seed 42; sampled profile temperature 0.7.
The exact tokenizer, main and MTP SHA-256 identities are maintained in
`test/fixtures/inkling-golden-v1.json`.

All 28 groups matched baseline prompt counts and exact output tokens in all
five measured repetitions through each API. This is regression evidence against
the existing implementation, not an independent assertion of model accuracy or
maximum context support.

Ratios below are candidate/baseline medians across the 28 groups; decode includes
only groups generating at least 16 tokens. No threshold was relaxed.

| Metric | Engine ratio range | Facade ratio range |
| --- | --- | --- |
| End-to-end generation latency | 0.9980–1.0052 | 0.9968–1.0291 |
| First-token latency | 0.9983–1.0062 | 0.9965–1.0334 |
| Decode tokens/second | 0.9992–1.0020 | 0.9894–1.0032 |

Both performance gates passed (latency tolerance +10%, decode tolerance −5%).
The largest observed median decode reduction was approximately 1.06% through
the facade; it is not evidence of a throughput improvement. Per-group receipt
summaries retain minimum, maximum and coefficient of variation.

Compiled GPU instructions/control words match the baseline exactly: 423 functions,
592,416 words across GEMV, Inkling, DSpark, checked-prefill and checked-weights.
NVCC anonymous-namespace hash suffixes in function labels are normalized; actual
instruction/control words are not. Original upstream bytes still reconstruct
through the reviewed identifier map and explicit host-only error adapters.
Frozen Zig reference files remain unchanged.

The real C verification example loaded the exact model and reported fingerprint
`2cfbc10c85ba0949cbaddba1fa1c76f1609a89dcffa160438888de595925b9f9`, matching
Python's independently computed artifact manifest.

## Host and failure boundaries

Executed successfully:

- macOS ARM64: host/API and ABI snapshot checks, C/C++ installed consumers,
  configurable install/uninstall, lint with clang-format 18, reference-free build,
  deterministic extracted source/embedding/runtime archives.
- Linux AMD64: host checks, install contracts, ELF hardening inspection,
  ASan/UBSan and TSan with Clang 18.1.3.
- Seven modular/amalgamated fuzz harnesses: approximately 11 seconds each,
  5,028,552 aggregate executions, no detected crash/sanitizer failure.
- Native error-injection mocks: preserved first launch error through caller-context
  restoration, cancellation/reset cleanup diagnostics, logger with NULL error
  output, bounded extension parsing and tokenizer checksum rejection before
  native upload. Host macro tests exercise the actual retained CUDA error macro
  without deliberately damaging a GPU context.
- Clean-source native release gate: exact identity verification, engine/facade
  golden and performance gates, extracted source/provider and amalgamated model
  consumers, and packaged generation command.

The added static pkg-config consumer first exposed a missing host archive in the
runtime SDK. The packager now includes it and static metadata explicitly selects
archives even when shared libraries are installed.

Final static SDK package retest at
`142980723bf7ae5903e24b8ad83a7f3eee231fe9`: passed, including actual model
execution through extracted source, installed static pkg-config, amalgamated C
and the packaged CLI.

## Reproduction and remaining scope

The full command is documented in `docs/library-integration.md` and automated by
`tools/run_model_release.py`. Evidence is external to Git under
`/data/scratch/inkleberry-hardening-qualified-evidence` on `core`, with the final
static SDK package log at
`/data/scratch/inkleberry-hardening-final-packages-fixed.log`.
Do not reuse baseline timings across different devices or toolchain configurations.

The trusted GitHub GPU workflow has not been executed as a workflow and needs an
operator-provisioned self-hosted runner. Its underlying command was executed
manually on the reserved device. These checks do not qualify other GPUs, models,
quantizations, arbitrary long contexts or every possible output distribution.
