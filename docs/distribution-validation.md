# Distribution validation

Validation recorded on 2026-11-26. The modular implementation remains canonical.

## Executed

- macOS: host checks, install/consumer checks, lint and reference-free rebuild.
- Linux x86-64 on core: host checks, install checks, sanitizer checks and fuzz smoke.
- Generated host: parser, tokenizer, conversation, streaming, concurrency and mmap
  regressions; public/global symbol parity with the modular host (71 symbols).
- Modular and generated readers: sparse 64 MiB safetensors fixture with a 2 MiB
  injected heap budget; measured peak injected allocation 712,792 bytes. Linux
  additionally verifies a read-only private file mapping. This is a memory-use
  regression, not a throughput benchmark or a zero-copy GPU claim.
- Generator and package-policy regression tests.
- Deterministic source archives rebuilt after extraction without the reference
  tree; extracted two-file SDK compiled and consumed by C and C++ applications;
  extracted host-runtime executable exercised against a public synthetic fixture.
- Both modular and amalgamated parser fuzz smoke: 3,000 runs per harness.
- CUDA-enabled amalgamated native acceptance executable and CUDA runtime archive
  built on core using the existing native kernel toolchain.

## Pending native qualification

No inference request was executed for the new amalgamated host or extracted GPU
SDK. The idle-device guard refused the run: both GPUs were occupied by unrelated
vLLM workloads using approximately 82 GiB each. Those workloads were left intact.
Compilation alone does not establish numerical or generation equivalence.

When a device is available, run the guarded native acceptance and
`make native-release-check MODEL_DIR=/external/model SQ_TEST_GPU_UUID=GPU-...`
with the documented CUDA toolchain. This checks extracted source, GPU runtime and
embedding artifacts. Keep the distribution change in draft until that native
qualification succeeds. Model inputs remain external to every package.
