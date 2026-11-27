# Architecture consistency and native shape diagnostic — 27 November 2026

Audit PR #8 was merged into `feat/ampere-compatibility` as `4f9715b` after
all seven CI jobs passed on its exact head commit. It was not merged into main.
The implementation branch is `feat/architecture-consistency`, based on that
merge. This pass changes private contracts and qualification tooling; it does
not change device arithmetic, sampling, public ABI, precision defaults or goldens.

## Small implementation boundaries

- `src/inkling_geometry.h` replaces duplicated C geometry enums and key layout/
  schedule literals. Checked CUDA adapters assert agreement with the unchanged
  pinned constants; this is a fixed model contract, not a model registry.
- `src/inkling_quant.h` names the existing expert/MTP representation choices.
  Facade profiles use explicit entries rather than positional modulo arithmetic.
  All six weight profiles and six execution profiles retain their meanings.
- Prefill resource preparation is checked per context before capture on every
  target. PDL only selects launch ordering. The old process-static, unchecked
  shared-memory opt-in is removed through the narrowly reversible host adapter.
- New benchmark rows identify exact canonical little-endian input token bytes
  and resolved request settings. Repetition validation rejects mixed identities,
  prompt lengths, duplicate run numbers, invalid tokens and malformed timings.
- The release runner requires an actual clean checkout, verifies any supplied
  revision against HEAD, fingerprints allowlisted source contents, records actual
  binary/config identities and rechecks sources at completion. Make supplies
  platform paths and selected tool commands. An extracted tree cannot establish
  commit provenance from an operator-supplied string; extracted-package testing
  remains separate. Historical benchmark files remain readable without inventing
  identities they never recorded.

There is no generic backend framework, model registry, new runtime option,
precision fallback or numerical repair.

## Native measurement

Executed on `core`, RTX PRO 6000 Blackwell Max-Q (`sm_120a`), CUDA 13.5.37.
The synthetic test source at implementation revision `4cf8966` links the normal
CUDA archive, uses the unchanged production dispatch and needs no model files.
Exact compiled source/binary hashes, compiler/driver/device and raw rows are in
[the evidence JSON](architecture-consistency-2026-11-27.json).

The test prepares/destroys checked prefill resources in two independent CUDA
contexts. For each width it evaluates 32 rows with identical FP4 weights, unit
scales and exactly representable FP16 inputs at batch sizes 1, 2, 4 and 8. A
host double-precision reference sums the same represented operands. Five eager
and five graph executions must agree exactly within each shape and remain finite.

| Width | Singleton maximum absolute reference error | Batches 2/4/8 maximum absolute reference error | Eager/graph repeatability |
| --- | ---: | ---: | --- |
| 2304 | 0.1455078125 | 0 | Exact, all five repetitions per mode |
| 3072 | 0.18994140625 | 0 | Exact, all five repetitions per mode |

Every batched element in this fixture differed from the corresponding singleton
result. This confirms that execution shape can change arithmetic even on the
same GPU: the singleton NVFP4 path rounds FP16 partial sums, while the batched
path uses FP32 MMA accumulation. It does **not** establish the cause of historical
GB10/Ampere model-output differences, a production quality error, or an approved
error budget. No arithmetic path was changed to hide the difference.

The guard admitted an idle GPU; it returned to 2 MiB / 0% utilization afterwards.
No inference service was stopped and no GPU settings were changed.

Reproduce on an explicitly approved idle device:

```sh
make -j4 .build/linear-shapes
INKLEBERRY_TEST_GPU_UUID=GPU-APPROVED-ID make linear-shape-check
```

Use the existing target flags for GB10 or experimental Ampere; do not infer
qualification from building the executable. The ordinary `check` target never
runs this GPU diagnostic.

## Validation and limits

Executed: host `check install-check install-contract-check`; all nine weight
layouts against the independent byte oracle; exact 48-visit schedule; all expert/
MTP precision choices and facade profiles; C/CUDA geometry mismatch rejection;
all three stub launch/resource policies including context failures; benchmark/
release negative cases; lint, namespace and upstream reconstruction; isolated
reference-free build and extracted-package checks. Both benchmark host variants
compiled. The native synthetic probe and actual prefill preparation passed.

The default Apple ASan runtime stalled in initialization before main; that
attempt was stopped and is not a passing sanitizer result. The complete ASan/UBSan
retry passed with the already-installed LLVM 22.1.8 toolchain, as did modular and
amalgamated fuzz smoke (3,000 runs per case). Final CI results are recorded in the PR.

Not established by this pass: a fresh model-backed performance/quality campaign,
GB10/Ampere native measurements, full-model teacher-forced traces, a cross-device
numerical tolerance or complete runtime vendor-library identity. Existing goldens
remain unchanged; historical mismatches remain open.

## Next numerical step

Freeze one source/input/representation configuration across targets. Compare
packed weights/scales first, then teacher-forced singleton versus batched logits
and the earliest divergent operator. Classify FP16 intermediate rounding, reduction
order and format conversion separately. Agree an operation-specific numerical
contract before changing any kernel or acceptance threshold. Preserve existing
throughput and quality evidence throughout.
