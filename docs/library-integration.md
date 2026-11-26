# Library integration and qualification

Inkleberry Finn provides modular C11 headers through `include/ifn.h` and the same
API through generated `dist/ifn.h`. `dist/ifn.c` includes the host implementation;
native inference additionally requires the CUDA archive and documented runtime
libraries. `examples/generate.c` demonstrates streaming generation and checked
cleanup; `examples/verify.c` demonstrates exact model identity verification.
Neither example supplies a server or a scheduler.

## ABI and installation

`VERSION` generates `include/ifn_version.h` and supplies package/pkg-config versions.
The host shared library uses ABI-major-1 SONAME/install names, with a versioned
real file and development link. The inference vtable remains ABI major 1, minor 1:
existing structure sizes, offsets and enum values are checked against the v1.0
fixture. Optional logger and artifact-manifest extensions do not enlarge existing
records. Request unsupported required features explicitly rather than guessing.

```sh
make check install-check install-contract-check
make install DESTDIR=/temporary/stage PREFIX=/usr/local \
  LIBDIR=/usr/local/lib64 INCLUDEDIR=/usr/local/include \
  BINDIR=/usr/local/bin PKGCONFIGDIR=/usr/local/lib64/pkgconfig
make install-infer-backend DESTDIR=/temporary/stage PREFIX=/usr/local \
  LIBDIR=/usr/local/lib64 INCLUDEDIR=/usr/local/include \
  BINDIR=/usr/local/bin PKGCONFIGDIR=/usr/local/lib64/pkgconfig
PKG_CONFIG_PATH=/temporary/stage/usr/local/lib64/pkgconfig \
  pkg-config --static --libs inkleberry-static-infer
```

Use identical directory overrides for all install/uninstall commands.
`make uninstall-infer-backend` removes the ABI-1 shared provider and its
`inkleberry-infer.pc`, preserving the host/static SDK. `make uninstall` removes
the complete installed SDK for this build. Neither deletes unrelated files or
other versioned real library files. Ordinary development
builds enable stack protection; Linux builds also enable fortification, RELRO,
immediate binding, non-executable stack and CLI PIE. `make harden-check` inspects
actual ELF outputs. `HARDEN=0` is an explicit portability escape hatch.

## Recoverable errors and caller-controlled logging

The retained host CUDA/cuBLAS helpers record errors instead of terminating the
process. Checked launch boundaries consume the first recorded error and preserve
its native code. GPU execution failure can poison a session: close it and create a
new session; returning an error does not promise that a failed GPU is reusable.

`ifn_infer_error_v1` is caller-owned and contains a copied message, stage, provider
and code. Provider families distinguish OS errno, CUDA runtime, cuBLAS and CUDA
driver errors. Native primary codes are unencoded within their provider family.
The existing `cleanup_provider_code` field uses bridge encoding for secondary
cleanup failures: runtime codes unchanged, cuBLAS 10000 + code, driver 20000 + code.
The first operation failure remains authoritative when cleanup also fails.
Successful later calls may clear the thread-local low-level diagnostic; copy it
before issuing further low-level operations. Not every nested cleanup failure can
be enumerated by the fixed-size record.

Attach `ifn_infer_logger_v1` through model configuration extensions to receive
native facade errors even when the call's error output is NULL. Logging is opt-in,
contains no prompt text and runs on the calling thread. Callback records are
borrowed: copy before returning. Callbacks must return promptly, be thread-safe,
and must not throw, longjmp, reenter the library or close its handles. There is no
per-token success logging and no global logger.

## Exact artifact identity

Default model loading retains its existing cost and behavior. To verify exact
assets, attach `ifn_infer_artifact_manifest_v1` with SHA-256 hex spans for tokenizer,
main checkpoint and (for MTP) draft checkpoint. These are hashes of the actual
mapped bytes, checked before parsing/tokenizer admission or GPU weight upload.
Keep mapped files immutable throughout their mapping lifetime.

Request `IFN_INFER_CAP_ARTIFACT_VERIFY` when verification is mandatory. A missing
manifest, malformed hash or mismatch fails loading; an ordinary unverified load
never claims verified identity. After successful verified loading, model-info
reports `fingerprint_verified` and a fingerprint equal to SHA-256 of the bytes
`ifn-artifacts-v1`, followed by the binary tokenizer, main and optional MTP hashes,
in that order. Verification adds a full file read at load time, not during decode.
This identifies artifacts; it does not establish model quality or trust in a
manifest supplied by an untrusted caller.

## Trusted model-backed release gate

CPU CI includes ABI/installed-consumer checks, injected error/concurrency tests,
ASan/UBSan, TSan, and timed parser fuzzing of modular and amalgamated code. Those
checks cannot establish numerical inference correctness. Native releases require
an explicitly reserved idle GPU and external model files:

```sh
python3 tools/run_model_release.py \
  --model /external/inkling-model --gpu GPU-APPROVED-ID \
  --baseline-engine /external/baseline-engine.jsonl \
  --baseline-facade /external/baseline-facade.jsonl \
  --out /external/release-evidence
```

The runner verifies the maintained model/tokenizer hashes, checks C/Python artifact
fingerprints, executes deterministic goldens through both the engine and versioned
facade, compares performance and tests extracted source, amalgamation and native
runtime packages. Receipts include source revision, model identities, toolchains,
binary hashes, device, flags, output equality and timing distributions. Each attempt
atomically replaces the current receipt with a new run ID and `passed: false`
before preflight; all validation/build failures publish failed evidence. Consumers
must check that current receipt and the command exit status, rather than trusting
leftover logs from earlier runs. Supply
`--source-revision` when running an exact extracted source tree without `.git`.
Never label modified sources with an unrelated commit.

Maintain comparable baselines on the same reserved device/toolchain with the same
model, profiles and power configuration. `test/native_benchmark.c` runs seven
public workloads across AR, fixed speculative, adaptive lookup and seeded
sampling, one warm-up and five measured repetitions. The longest actual prefill
is 8,844 tokens; this suite does not establish the model's maximum context or
exhaustive quality. Exact output tokens are required; median total/first-token
latency may not regress over 10%, and decode throughput may not regress over 5%
for outputs of at least 16 tokens. Short outputs retain latency gates. Inspect
variability and rerun controlled A/B measurements when a gate fails; do not waive
quality failures or silently relax thresholds.

`.github/workflows/model-release.yml` exposes the same command manually on a
trusted self-hosted runner restricted to `artalis`. It requires operator-provided
local assets and reserved GPU; no runner is installed by this repository. It is
not an automatic PR GPU job. Never run untrusted pull requests on a model host.

For host-adapter changes, `tools/check_device_code.py BASE_BUILD CANDIDATE_BUILD
--out evidence.json` compares actual compiled GPU instructions, normalizing only
NVCC anonymous-namespace hash suffixes in function names. Provenance reconstruction
separately reverses reviewed host adapters and identifier maps to verify the
original upstream hashes. Neither mechanism permits silent kernel math changes.
