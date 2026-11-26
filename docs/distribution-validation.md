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

## Native qualification after the Inkleberry Finn rename

Executed on core on 2026-11-26 with CUDA 13.5 / sm_120a and the external
`inkling-2-mini-c9d028302cd0` checkpoint. The selected idle device was GPU 1,
`GPU-e5f713d1-46ae-9f99-c16d-65de9e9df32a`, an NVIDIA RTX PRO 6000 Blackwell
Max-Q Workstation Edition. GPU 0's existing inference workload was untouched.
The earlier busy-GPU blockage is resolved for this qualification.

```sh
INKLEBERRY_TEST_GPU_UUID=GPU-e5f713d1-46ae-9f99-c16d-65de9e9df32a \
  make native-release-check \
  MODEL_DIR=/data/models/safetensors/inkling-2-mini-c9d028302cd0
```

The helper rebuilt the extracted source, produced/extracted the GPU runtime and
embedding archives, then ran the real native acceptance suite through both the
shared modular provider and the independently compiled amalgamated host linked
to the packaged CUDA archive. Both passed: answer parity, adaptive/lookup modes,
prefix reuse, independent streams, BUSY handling, stale/active cancellation,
cache invalidation, scope isolation, stop filtering and replay. The packaged
`inkleberry-generate` CLI also produced the expected arithmetic answer.

Machine-readable result:

```json
{"extracted_source_native": true, "extracted_amalgamation_native": true, "packaged_cli": true, "reference_files_required": false}
```

The controller's run log is `/data/scratch/inkleberry-native-release-20261126.log`
on core. The shared provider exports `ifn_infer_get_api@@IFN_INFER_1.0` with the
remaining implementation hidden. All 77 canonical implementation/adapter/test
C files were checked for unchanged logic tokens modulo the deliberate rename
and formatting. CUDA pins, LICENSE and NOTICE are unchanged.

This is functional qualification of the renamed distributions on the listed
hardware/model, not a new performance campaign or a claim about other models.
Model inputs remain external to every package.
