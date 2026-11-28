# Independent conditional-prefix reference — 28 November 2026

## Result

The pinned independent Hugging Face Transformers implementation loaded the same
checkpoint with no missing, unexpected or mismatched keys. All three selected AR
boundaries completed twice with bit-identical full-vocabulary logits. A separate
native source-weight/BF16-KV control also completed twice per case with identical
vectors and SHA-256 checks of the exact consumed prompt plus preceding output IDs.

The control agrees with the reference top token in all three cases. The production
RTX configuration agrees in two, and production GB10 in none. These are **three
preselected divergent prefixes**, not a representative accuracy score. Do not use
those counts to rank overall model quality.

| Boundary | Reference top token | Production RTX top token | Production GB10 top token | BF16 native control top token |
| --- | ---: | ---: | ---: | ---: |
| Case 4 | 151644 | 151644 | 149878 | 151644 |
| Case 5 | 151791 | 154573 | 153667 | 151791 |
| Case 6 | 149917 | 149917 | 150397 | 149917 |

Full-vocabulary total variation at diagnostic temperature 1:

| Boundary | Production RTX vs reference | Production GB10 vs reference | Native BF16 control vs reference |
| --- | ---: | ---: | ---: |
| Case 4 | 0.071309 | 0.072536 | 0.075360 |
| Case 5 | 0.278606 | 0.186919 | 0.020245 |
| Case 6 | 0.163500 | 0.192655 | 0.093201 |

The JSON control improves TV from approximately 27.9% to 2.0%, and retrieval
from 16.4% to 9.3%, relative to production RTX. Code-editing TV does not improve
(7.1% versus 7.5%), although its logit RMS difference decreases. Full distributions
remain different. There is no inferred pass threshold or end-to-end quality claim.

## Representation and execution boundary

| Configuration | Linear matrices | Experts | K/V cache | Execution |
| --- | --- | --- | --- | --- |
| Production captures | Additional native NVFP4 conversion, including checkpoint BF16 unembed | Native checkpoint formats/arithmetic | FP8 | CUDA graphs |
| Native control | Checkpoint BF16 linears retained | Native checkpoint formats/arithmetic | BF16 | Eager conditional-prefix logits |
| Independent reference | Checkpoint BF16 linears retained | Official conversion dequantizes NVFP4/MXFP8 experts into model BF16 | BF16 model K/V | Eager PyTorch attention/experts |

This is an original-checkpoint BF16 reference, **not an exactly quantization-matched
oracle for the default production profile**. The official converter rounds the
decoded expert values into the parameter dtype; native experts retain their existing
packed formats and computation order. The native control changes linears, KV and
graph execution together, so it does not isolate their individual effects. It must
not be used to claim that all remaining differences are harmless rounding or that
either complete platform is numerically correct in every regime.

Reference source: [Transformers b62646bb](https://github.com/huggingface/transformers/tree/b62646bbda836d467060b7e3d64028202079b77d).
The reviewed [conversion mapping](https://github.com/huggingface/transformers/blob/b62646bbda836d467060b7e3d64028202079b77d/src/transformers/conversion_mapping.py)
and [expert decoder](https://github.com/huggingface/transformers/blob/b62646bbda836d467060b7e3d64028202079b77d/src/transformers/core_model_loading.py)
explicitly dequantize Inkling experts. Only the in-memory unsupported ModelOpt
quantizer envelope is removed; checkpoint bytes and upstream source stay unchanged.
No custom quantization/conversion implementation or replacement kernel is introduced.

## Execution and provenance

Reference: Transformers 5.25.0.dev0 at the exact commit above, borrowed installed
Torch 2.13.0+cu130, explicitly eager attention/experts, hub kernels disabled, local
files only, offline mode, TF32 disabled and reduced-precision BF16 reduction disabled.
The model loads on CPU and moves to the selected RTX GPU. Existing service packages,
drivers and global environments are unchanged; no new dependency was installed.

An initial `device_map` attempt stopped before model execution because `accelerate`
was absent. CPU loading plus GPU transfer removed that optional dependency. The
failed attempt log remains in the external evidence. Subsequent reference runs
completed successfully. Native control links the previously built, unchanged native
engine archives (source archive and both library hashes are recorded). Its C source,
executable and input-file hashes are separately recorded; this is operator-attested
build association, not a signed build attestation.

Both experiments ran sequentially on the explicitly guarded idle RTX PRO 6000 UUID
`GPU-5a215cb3-57b0-0a6e-d511-e90bb05c0e05`. The independent reference used at most
59,930,209,792 Torch-allocated bytes. The final sample is 2 MiB / 0% on that device;
the other GPU remains at its 26 MiB / 0% starting baseline. No Spark service was
paused for this experiment. All model files were rehashed after execution, and
the pinned reference checkout remains clean.

Prompt chunking reproduces the native prefill partition: 40+7 tokens, 32+4 tokens,
and 2048+2048+2048+2048+648+4 tokens for the three cases. The shared preceding
output tokens are then forced one at a time. No processor/chat template rewrites
the saved input IDs. The native control first consumes just the prompt via a
one-output generation call; that sampled output is not fed, and the public logits
API appends only the supplied preceding tokens.

Nine new host tests reject missing/duplicate cases and repetitions, changed model
identity, changed contexts/control input hashes, changed payloads even with updated
digests, malformed IDs and incorrect prompt boundaries. Full local `make check`
and clang-format-18 lint pass for the final local tree. Kernels, goldens,
production defaults and public APIs remain unchanged.

[Machine-readable evidence](independent-reference-2026-11-28.json) contains the
comparison, input identities, model/source/artifact hashes, reference loading report,
repeat records and post-run verification. Raw logits and complete manifests remain
outside Git at `/Users/mark/artalis.io/data/inkleberry-reference-2026-11-28/` and
`/data/scratch/inkleberry-reference-2026-11-28` on core-1. The executed collector
is retained externally as `executed-collector.py` with its recorded hash.

## Reproduction

Use an isolated clean checkout of the pinned Transformers source and a compatible
Torch runtime. Do not install into an existing service environment. All input paths
are external. Run through the UUID/idle guard with offline/cache environment flags
as shown in the retained command/log evidence; normal builds do not need Torch.

```sh
PYTHONPATH=/external/transformers/src HF_HUB_OFFLINE=1 \
  INKLEBERRY_TEST_GPU_UUID=GPU-APPROVED-ID INKLEBERRY_TEST_TIMEOUT_SECONDS=1800 \
  python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 \
  /external/torch-runtime/bin/python tools/collect_reference_logits.py \
  --model /external/model --captures /external/rtx-capture \
  --divergences docs/validation/gb10-first-divergence-2026-11-28.json \
  --reference-source /external/transformers \
  --reference-revision b62646bbda836d467060b7e3d64028202079b77d \
  --output /external/new-reference
make -j8 CUDA_ARCH=120a .build/reference-control
```

The native control input has three records: case, prompt-token count, total-token
count, then exact prompt and shared-prefix IDs. Generate it using the collector’s
`contexts()` helper from the same report/capture. Preserve its checksum. Run the
control executable through the same idle guard with model directory, control input
file and a new output directory. Save stdout as `control.jsonl` in that directory.

```sh
python3 tools/compare_reference_logits.py \
  --reference /external/reference --control /external/control \
  --boundaries /external/paired-boundary-captures \
  --divergences docs/validation/gb10-first-divergence-2026-11-28.json \
  --out /external/new-reference-comparison.json
```

## Next qualification

Keep kernels/defaults/goldens unchanged. Run held-out task quality under the actual
production profiles. If tighter numerical attribution is needed, vary linears, KV
and execution mode independently, then evaluate expert representation/arithmetic
differences. Do not silently label this BF16 control a precision-matched proof.
