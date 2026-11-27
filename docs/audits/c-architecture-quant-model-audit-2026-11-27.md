# Inkleberry Finn: C, architecture, quantization and model audit

Date: 27 November 2026. Audited revision:
`5553ef371e2e0342f3ea3dc546b74fcb1e9364dc` (`feat/ampere-compatibility`).
**Analysis only. No implementation changes, GPU runs or SSH/device-host access in
this pass.** The prior implementation was already committed and its remote
branch SHA matched at the start of this audit.

## Assessment

The C library foundation is sound enough to evolve incrementally: bounded
checkpoint admission, injected allocation, explicit ownership, a versioned facade,
private backend state, generated bridge layouts, and unusually useful failure and
consumer tests already exist. A rewrite, generic model framework or new scheduler
would add risk without solving the current problem.

The next architectural work should establish authoritative model/quant/execution
contracts and reliable qualification identity. Numerical work should then locate
the earliest divergence, starting with packed weights and teacher-forced execution.
Free-running text and repeatability alone cannot locate or exonerate a numerical
error.

Nine findings: **4 high, 4 medium, 1 informational**. High here denotes an
architecture/qualification blocker for the next milestone, not a newly proven
remote-code-execution or memory-corruption vulnerability. No new critical/high
C memory-safety defect was established in the reviewed paths. This is a targeted
source audit, not formal verification of every CUDA instruction.

## Methods and comparative baselines

- Applied bitnet.c's `.claude/skills/c-audit/SKILL.md`: memory, untrusted parsing,
  integer arithmetic, vector boundaries, concurrency, teardown, dead code and tests.
  GGUF-specific checks were adapted to safetensors; no automatic replacement or
  fix mode was used. Comparator revision: `ab9e8e8661371ee33fb086980b675aae88246de5`.
- Applied Keel's `.claude/skills/axis-audit/SKILL.md` map/trace/ownership method to
  model, quant, execution/backend and platform axes. Its networking backends and
  operational commands are not applicable. No Keel source was changed.
- Inspected Jev Bush `85fd51f8c99b1b8b7ea26d33e02413e681a9271d`: `README.md`,
  `tools/check_strict_invariance.py`, `tools/check_model_golden.py` and manifest
  practices. This checkout has no separate AGENTS.md or additional audit skill;
  no unavailable Jev Bush skill is claimed to have run.
- Inspected bitnet.c's `AGENTS.md`, `CLAUDE.md`,
  `docs/transformer-behavior-map.md`, `docs/coherence.md`, quant/model/backend
  ownership and test strategy.

Jev Bush supplies a useful explicit strict-versus-fast contract and
cache/batch/monolithic invariance tests. Its scalar decision-model implementation
is not an Inkling oracle. bitnet.c supplies useful ownership and behavior maps;
its broad model registry and fixed coherence tolerances should not be copied into
this specialized engine.

The baseline contains about 19,596 lines across native implementation/public
headers and 174 files under source/include/test/tools/CUDA. Review concentrated on
admission, load/plan/bind, session/request lifetime, prefill/decode/speculation,
platform seams, qualification tooling and the tests exercising them. Generated
amalgamation is checked through canonical-source/provenance tools rather than
counted as a second independent implementation. Kernel review was targeted, not
an exhaustive memory proof of all retained numerical bodies.

## Current architecture and ownership

| Layer | Files and types | Current responsibility |
| --- | --- | --- |
| Host services | `foundation.c`, `host_posix.c`, `host_win.c`, `host_clock.*` | Checked arithmetic, allocator/error contracts, file mapping, clock |
| Source interpretation | `safetensors.c`, `json.c`, `tokenizer_load.c`, `inkling.c` | Bytes, tensor roles/shapes, tokenizer admission; no device upload |
| Loaded GPU representation | `weights_recipe.c`, `weights_runtime.c`, `weights_internal.h`, `gpu_cuda.c` | Two-pass layout, representation conversion, arena and scales; private injected upload backend |
| Inkling execution binding | `engine_bind.c`, `engine_internal.h` | Borrowed device views, fixed layer/visit schedule |
| Session execution | `engine_layout.c`, `engine_forward.c`, `engine_prefill.c`, `engine_runtime.c` | Session-local KV/rings/scratch, eager/captured work, failure state |
| Decode policy | `sampling.c`, `engine_spec.c`, `engine_adaptive.c`, `engine_prefix.c` | Request options, target/draft execution, verification, prefix state |
| Native bridge | `abi/cuda.json`, generated headers, `src/cuda/*`, `cuda/*` | C/CUDA layouts, checked wrappers, native operations and kernels |
| Public facade | `infer*.c`, `ifn_infer.h`, `ifn_infer_inkling.h` | Version negotiation, model/session/request handles, budgets, streaming and cancellation |
| Qualification/distribution | `tools/run_model_release.py`, `model_release.py`, `release.py`, tests | Artifact validation, workloads, evidence, explicit package contents |

An `ifn_ar_model` in a session is a borrowed execution view, not another owned
weight image. `ifn_gpu_weights` owns the device representation and retains the
checkpoint during upload; sessions retain that image through `session_users`.
Session KV, activations, rings, graph handles and prefix snapshots are mutable
session state. Requests own copied tokens/options and cancellation/result state.
There is one prepared native session per provider process, enforced and advertised;
independent request streams do not mean concurrent GPU execution.

The upload mock seam is deliberately smaller than an entire execution backend.
Most execution calls are direct CUDA bridge calls. That is consistent with the
current CUDA-only scope; it must not be described as a fully replaceable backend.

## Representative traces

1. **Admission and upload:** `ifn_infer_model_load` validates source paths and
   optional expected hashes; `ifn_inkling_checkpoint_open` binds exact source
   role/dtype/shape specifications; `ifn_gpu_weights_load` enters the selected
   device and checks memory; `ifn_weights_recipe` materializes the same planned
   spans and views via checked copies/conversions. Mappings remain borrowed input,
   not device memory or a CPU copy of the entire checkpoint.
2. **Prepare:** facade session planning calls `ifn_session_owned_plan`; prepare
   calls `ifn_session_create`, retains weights, binds device views and creates
   session resources. Low-level create intentionally may return a cleanup-owned
   handle on partial initialization failure (`ifn_engine.h:78`); the facade
   separates create from prepare. This is documented, not a constructor leak.
3. **AR:** `ifn_infer_request_run` holds the session lease, selects a cache scope,
   and calls `ifn_session_generate_controlled`; the native generator chooses
   batched prefill or 1..8-token forward, then eager/graph decode and token events.
4. **Speculation:** `ifn_spec_draft` produces proposals and probabilities;
   `ifn_spec_target` verifies a batch of `drafts+1`; the adaptive path chooses
   graph branches. These batch shapes can differ numerically from single-token
   target evaluation, even before testing acceptance arithmetic.
5. **Cancellation/close:** a request-local atomic cancellation flag feeds native
   cooperative checks. The atomic seal in `infer_request.c:453` linearizes
   cancellation versus completion. Error/cancel paths invalidate state before
   unlocking. Close keeps failed resources for retry and parents reject live
   children. Arbitrary concurrent close versus API entry is explicitly outside
   the caller-lifetime contract, not a promised reference-counted handle system.

## Findings

| ID | Severity / axis | Evidence | Consequence and smallest proposed change |
| --- | --- | --- | --- |
| A1 | High / model anatomy | `src/inkling.c:9`, `weights_recipe.c:9`, `engine_internal.h:50`, `engine_bind.c:245`, `cuda/config.hpp:13` | Geometry, expert split and visit schedule have multiple authorities. A model update can change admission without matching layout/execution. Establish one private Inkling contract and cross-language assertions/derived tables, preserving current order and constants. |
| A2 | High / quant policy | `weights_recipe.c:106`, `:182`, `:264`; `infer_model.c:110`; `infer_session.c:34`; `engine_layout.c:25`; `cuda/config.hpp:81` | Model roles, quant decisions, physical packing, FP8/NVFP4 scales, MTP folding and ordinal profile selection are intertwined. Define resolved per-role representation metadata, then let the existing recipe consume it. Do not equate the mixed `exact` profile with an unquantized numerical oracle. |
| A3 | Medium / launch versus resource preparation | `src/cuda/target_profile.h:16`, `prefill_resources.hpp:14`, `checked_prefill.cu:36`, `cuda/kernels_prefill.cu:832` | `IFN_CUDA_USE_PDL` controls both launch ordering and whether resource preflight is checked per context. Blackwell still has the old static, ignored `cudaFuncSetAttribute` result. Separate policies and context ownership. Treat changing error/preparation behavior as a separate correctness patch, not an invisible structural refactor. |
| A4 | High / numerical contract | `Makefile:34`, `checked_prefill.cu:72`, `cuda/kernels_common.cuh:109`, validation reports | Fast math is unconditional; cuBLAS algorithms and native conversion/reduction paths can differ across devices/toolkits. There is no resolved numerical-policy identity or controlled diagnostic path. Specify and record rounding, reduction, compute type, quant and vendor choices before setting equivalence gates. |
| A5 | High / qualification provenance | `tools/run_model_release.py:84`, `:87`, `:93`, `:96`, `:101` | A supplied 40-hex revision need not equal the checked-out source; dirty status is recorded but not rejected. Toolchain/path queries and build targets hardcode Linux/default locations. A passing receipt is not sufficient proof of the claimed three-device source/configuration. Bind evidence to verified source contents and actual build outputs; resolve platform paths from existing Make contracts. |
| A6 | Medium / repeatability verifier | `tools/model_release.py:14-25`, `:73-77` | Repetitions check equal output tokens but not equal prompt identity/settings; golden verification examines the first row's prompt count. There are no per-run input-token/config/packing fingerprints. Require identical workload identity across repetitions and bind records to a run manifest, with negative tests for mixed or forged records. |
| A7 | Medium / diagnostic coverage | `src/engine_runtime.c:303`, `test/conversion_oracles.cu:11`, `test/native_benchmark.c:120` | Small conversion oracles and token goldens do not locate the first divergent layer or batch path. Existing logits API covers 1..8-token forward and returns the final position; it does not expose the full batched-prefill trace. Extend a private test observer and fixed teacher-forced probes; preserve the public ABI and uninstrumented fast path. |
| A8 | Medium / observable execution identity | `src/infer.c:211`, `infer_internal.h:9`, `tools/cuda_profile.py:12`, `release.py:150` | Build metadata honestly says build is not qualification, but runtime/profile information lacks a complete recipe, math, toolchain and packed-image identity. Same API/version/profile name cannot establish numerical comparability. Add a private diagnostic/run manifest first; only consider a versioned public query if an actual consumer needs it. |
| A9 | Informational / backend ownership | `weights_internal.h:12`, `gpu_cuda.c:81`, `engine_runtime.c:81`, `engine_prefill.c:13` | The backend vtable covers upload/lifetime, while execution is CUDA-specific. This is legitimate specialization. Name/document the seam accurately; do not introduce a full operation vtable, model registry or alternate backend merely to mimic bitnet.c. |

### A4: execution-shape arithmetic candidate

`cuda/kernels_gemv.cu:1156` dispatches supported NVFP4 single-token linear
operations to `ifn_cuda_linear_nvfp4_b1_pdl_kernel`, whose partial products and
sums use FP16 `__hmul2`/`__hfma2`/`__hadd2` before FP32 accumulation
(`:234-243`). Batched operations select the MMA kernel, using
`ifn_cuda_mma_m16n8k16_f32_f16` (`:332`). These paths have different intermediate
rounding and reduction contracts even on one device. This is a concrete candidate
for singleton-versus-verification diagnostics, **not** an established root cause
of the historical cross-device or AR/speculative differences. Preserve both paths
in the architecture pass; investigate with identical inputs and declared budgets
before proposing arithmetic changes.

### A5: concrete evidence failure scenario

An operator can supply a syntactically valid revision for an extracted or modified
checkout. The runner records that string and a dirty flag, then may continue to
success if the workloads pass. The build/compiler/CUDA queries may also describe
default tools rather than the selected Make toolchain. This is an evidence-integrity
problem, not permission to execute untrusted receipts. Proposed remedies must retain
the current fail-closed invalidation of stale success before any attempt.

### Review items, not established defects

- Retained CUDA helpers use `reinterpret_cast`-based representation punning
  (`kernels_common.cuh:125`, `:142`, `:163`). This deserves a compiler/aliasing
  audit and adversarial fixtures; passing Ampere conversion oracles is not a
  language-level proof. No change is proposed without a concrete case and
  provenance review.
- Some recipe multiplications (`weights_recipe.c:120`, `:150`, `:166`) are not
  individually overflow-checked. Current callers use fixed, admitted geometry;
  no external-input-reachable overflow was established. A future geometry
  contract must preserve those bounds or add checks before generalizing.
- The prefill position array has 256 entries (`engine_prefill.c:38`), but the
  generator bounds chunks to 2,048 and uses eight-token groups. It is not an
  unbounded VLA or demonstrated overflow. This relationship should become an
  explicit invariant test rather than be lost during structural work.

## C-audit disposition

| Category | Reviewed evidence and conclusion |
| --- | --- |
| Parser safety | `safetensors.c:108-178` bounds header/work, checks dtype/rank and products, validates offsets using subtraction, rejects duplicates/overlap/unindexed bytes. `inkling.c:196` adds exact role shapes before upload. No raw pattern match was promoted to a parser vulnerability. |
| Strings / allocation | Default malloc/free are isolated behind allocator callbacks; budget allocation checks header addition and limits. Internal recipe names are generated bounded strings. Benchmark `strcat` calls use fixed fixture text and large fixed arrays; not evidence of untrusted-input overflow. |
| Mapping lifetime | POSIX read-only private mmap and Windows read-only file mapping are explicit; source immutability is a caller obligation. Hash verification is optional in the API but must be mandatory in numerical qualification. mmap is not an immutable content snapshot or zero-copy GPU execution. |
| GPU lifetimes | Checked resource handles and retained cleanup are present; sessions retain weights and may be poisoned after native failure. Raw internal bridge calls are not independent safe APIs for arbitrary pointers/shapes. |
| Threading | Session/request leases and atomic cancellation are explicit. Arbitrary concurrent close is excluded by documentation. One-session process scope is deliberate and tested; do not remove it while legacy shared helpers remain. |
| Bounds / vectors | Fixed geometry establishes divisibility for many kernels. New model geometry cannot inherit that guarantee automatically. Exhaustive small conversion tests do not cover arbitrary GEMM/attention/MoE layouts. |
| Errors | Native family/codes are normalized with first-error preservation in reviewed paths. Mapping these codes is duplicated between weight/session orchestration and merits eventual private consolidation, without changing public error semantics. |
| Dead/stale code | `if (true)` blocks in binding and old Zig-oracle comments are cleanup candidates only. Retained legacy prefill entrypoints/global handle are not called by the C session; do not confuse them with its session-owned cuBLAS handle. |
| Tests / hardening | Existing parser, allocation, concurrency, fuzz, ABI and installed-consumer suites are substantive. CPU checks cannot discharge device memory, numerical or GPU-context proof obligations. No sanitizer or model campaign was rerun in this design-only pass. |

## Existing qualification, not new experiments

| Target | Existing evidence | Still open |
| --- | --- | --- |
| GB202-class RTX / `sm_120a` | Historical recorded RTX PRO 6000 Max-Q: 28/28 profile-specific goldens, engine/facade parity | Fresh baseline at the same audited revision/toolchain matrix; not live-verified today |
| GB10 / `sm_121a` | Historical 17/28 goldens; per-device repeatability and engine/facade parity | 11 differing sequences, first-divergence analysis |
| Ampere RTX 3090 Ti / experimental `sm_86` | 11/28 goldens, deterministic repetitions, real recovery, 18 format smokes | 17 differing sequences; some same-device AR/speculative differences; broader quality |

The older RTX/GB10 campaign used a different source revision and CUDA toolkit on
each host; Ampere used Windows/MSVC and yet another toolkit. These are platform
observations, not an experiment isolating GPU silicon. Confirm exact GPU family,
UUID, target and software for a new campaign instead of inferring them from an alias.

## Checks executed in this audit pass

- `python3 tools/verify_upstream.py`: passed.
- `python3 tools/lint.py`: passed.
- `python3 test/native_namespace.py`: 8 tests passed.
- `make -j4 check install-check install-contract-check`: passed on the local CPU host.
- A6 counterexample: an ephemeral copy of historical records with one non-first
  repetition's prompt count changed was still accepted as all 28 groups. Original
  artifacts were unchanged; no test implementation was added.
- Read-only source, documentation and historical-artifact inspection.

No new ASan/fuzz/GPU/performance run, model acquisition, toolchain installation,
SSH/device-host connection or production mutation was performed. Prior results are cited
from `docs/validation/blackwell-gb10-2026-11-26.md` and
`docs/validation/ampere-windows-2026-11-27.md`, not relabeled as this pass's results.

## Proposed order

1. Approve the ownership/identity contracts in the companion design.
2. Add architecture invariants and consolidate fixed model/quant policy without
   changing resolved plans, numerical bodies, ABI or allocation ordering.
3. Fix evidence identity/platform resolution and independent resource preparation
   in isolated, reviewed changes with explicit compatibility tests.
4. Capture packed representations and teacher-forced traces on one fixed build
   per target; then isolate toolchain/OS/device effects.
5. Establish acceptance budgets and only then pursue numerical repairs or tuning.

See [architecture and numerical consistency design](../design/architecture-and-numerical-consistency.md).
