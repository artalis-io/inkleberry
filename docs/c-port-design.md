# C host port of squid5090

Status: accepted for implementation; first C host slice is in progress.
Date: 25 November 2026.
Repository: `artalis-io/squid-c`; public symbol prefix: `sq_`.

## 1. Recommendation

Build a C11 host library and CLI around the existing CUDA implementation. Preserve the CUDA mathematics, model representation, graph schedule and sampling behavior during the port. Establish behavioral parity first, then make separately tested ownership, cancellation and serving improvements.

This is tractable because the existing Zig implementation already calls a C ABI. Reimplementing the kernels is unnecessary. However, the host is substantial: the evaluated revision contains **9,385 lines of Zig** and **10,667 lines across the CUDA sources/headers**. Loading, tokenization, allocation, graph capture, recurrent state, speculative verification and prefix restoration all affect correctness.

The main benefit is a maintainable C library that fits Artalis's tooling and embedding conventions. Changing the host language alone does not establish an inference speedup.

**Recommended sequence:** pinned kernels and ABI → input/weight/tokenizer parity → autoregressive execution → speculative/cache parity → safe library lifecycle → optional HTTP service. Kernel optimization and upstream kernel upgrades stay separate from the language port.

## 2. Sources actually reviewed

| Source | Pinned revision | Review scope |
|---|---|---|
| Evaluated squid5090 | `d10c0abb2c8990be81c08ba84c66a8333a691b33` | Zig host modules, CUDA ABI/configuration, build, LICENSE/NOTICE, completed hardware evaluation |
| Current squid5090 HEAD observed during design | `51b85dd17e719a45e93e77a9e49df94c50144b30` | Six-commit comparison; LICENSE, NOTICE, build, CUDA declarations/bridge, constants and server snapshots; **not benchmarked here** |
| Jev Bush | `85fd51f8c99b1b8b7ea26d33e02413e681a9271d` | README, public header, modular source organization, foundation/API ownership, Makefile, CI/release gates, error/parallel-region checks, fault-injection tests and amalgamation tooling |

Jev Bush is the single repository [artalis-io/jev-bush](https://github.com/artalis-io/jev-bush/tree/85fd51f8c99b1b8b7ea26d33e02413e681a9271d), inspected locally at `/Users/mark/artalis.io/src/jev-bush`. Its local and remote HEADs matched at review time.

The original evaluation checkout and evidence remain separate from this repository. Review-only newer-head files and their hashes are under [the pinned upstream comparison](https://github.com/tensorfish/squid5090/compare/d10c0abb2c8990be81c08ba84c66a8333a691b33...51b85dd17e719a45e93e77a9e49df94c50144b30). No evaluated source, benchmark receipt or running installation was replaced.

### Upstream moved: choose the baseline deliberately

The observed HEAD is six commits beyond the evaluated revision. The comparison includes substantial CUDA rewrites, cluster attention/controller work, different drafting and sampling paths, and new `cublasLt` linkage. Its Zig CUDA surface has 79 extern functions versus 76 in the evaluated revision. NOTICE also changed.

**Start the mechanical port from the evaluated revision.** Preserve an upstream remote and exact revision/patch manifest. After the C host passes parity, upgrade kernels and their corresponding bridge/configuration as a separate change with a fresh Zig baseline on the RTX PRO 6000. Never combine new-head CUDA with old-head host layouts or reuse the old performance/quality claims for it.

If starting from newer kernels is preferred, rerun baseline validation on that exact revision before porting. The language-port and upstream-upgrade gates remain independent.

## 3. Licensing and publication

**Yes: Apache-2.0 permits this derivative and publication under `artalis-io`, including commercial use.** The inspected old and newer squid LICENSE files both declare Apache-2.0. Distribution must include the license, retain applicable attribution/copyright notices, preserve relevant NOTICE content, and identify changed files. A translated host is derived work; identify its upstream source rather than describing it as an unrelated clean-room implementation. Apache's trademark and patent provisions still apply. [Apache license, sections 2–6](https://www.apache.org/licenses/LICENSE-2.0), [upstream LICENSE](https://github.com/tensorfish/squid5090/blob/d10c0abb2c8990be81c08ba84c66a8333a691b33/LICENSE).

Use **Apache-2.0 for this fork**. Jev Bush's 0BSD license does not remove the Apache obligations attached to translated squid code and retained kernels. Retain upstream's license text and add Artalis attribution for modifications; keep original copyright statements. Jev-derived utility code, if actually reused, should have recorded provenance even though its 0BSD grant is permissive. [Jev Bush LICENSE](https://github.com/artalis-io/jev-bush/blob/85fd51f8c99b1b8b7ea26d33e02413e681a9271d/LICENSE).

There are separate materials to account for:

| Material | Treatment |
|---|---|
| Squid host translations and retained CUDA | Apache-2.0; keep the pin-specific LICENSE/NOTICE and prominent change notices |
| Adapted third-party work named in squid NOTICE | Preserve its attribution chain; audit actual copied files and their licenses before publication. NOTICE is not a substitute for a dependency's license |
| Jev Bush design and any copied utilities | Record the referenced revision and actual copied material; apply its 0BSD terms to those portions |
| Inkling checkpoint, tokenizer and model materials | External downloads; the pinned card declares Apache-2.0 and links a separate provider AUP. Do not bundle weights or imply the engine license replaces model-use terms |
| CUDA compiler/runtime/cuBLAS | NVIDIA SDK terms remain separate. Source releases initially require an installed toolkit. Binary/container distribution must check allowed redistributables and required notices rather than copying the whole toolkit |

The model AUP includes use restrictions and end-user disclosure obligations. Preserve links to the applicable model card/policy in model provenance. CUDA's redistribution list includes runtime/BLAS components, subject to its distribution requirements. [Pinned model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/c9d028302cd0f035fdf8e7e91f089a4ce94d2195/README.md), [model AUP](https://thinkingmachines.ai/model-acceptable-use-policy/), [CUDA EULA and Attachment A](https://docs.nvidia.com/cuda/eula/index.html).

Publication artifacts: `LICENSE`, `NOTICE`, `THIRD_PARTY.md`, a source/patch provenance manifest, and per-file attribution for translated/modified code. Choose an Artalis project identity; upstream attribution does not imply endorsement. No model weights, credentials, private prompts or evaluation output dumps belong in the source repository.

## 4. What to adopt from Jev Bush

| Observed Jev practice | Application here |
|---|---|
| Opaque `jb_model` / `jb_session`; immutable model, mutable workspace | Opaque C model/session handles with explicit ownership, reference retention and exclusive mutation |
| Typed library calls; JSON adapters call the engine directly | Token IDs and typed generation options are authoritative; CLI/HTTP parse into them without JSON serialization inside the engine |
| Fixed-width public discriminants, `struct_size`, API-version checks | Stable C/C++-consumable public header with explicit initialization and extension rules |
| Allocator/logger contracts and owned results | Host allocator fault injection; result lifetime independent of active inference; bounded diagnostic errors |
| Failed execution poisons a session while invalid input remains recoverable | Explicit ready/busy/cancelling/poisoned states and tested reuse rules |
| Focused source modules and a generated distribution file | Small responsibility-based C modules. A host amalgamation can be generated later; CUDA still requires separate compilation |
| Scalar/SIMD, cache/batch and model-backed golden checks | Matched C-versus-Zig token/logit/cache checks using the same kernel revision and configuration |
| Allocation-failure, concurrency, parser fuzz and error-boundary checks | Exhaustive failure injection and ownership tests, plus real CUDA memory checking |
| Static/shared library and installed-consumer tests | Test the actual installed public artifact, exported symbols and C/C++ consumers |
| Separate fast host CI and model-backed release gate | CPU-only validation on ordinary runners; protected real-GPU parity/release evidence |

These are concrete references: [public API](https://github.com/artalis-io/jev-bush/blob/85fd51f8c99b1b8b7ea26d33e02413e681a9271d/jb.h), [API implementation](https://github.com/artalis-io/jev-bush/blob/85fd51f8c99b1b8b7ea26d33e02413e681a9271d/src/api.inc), [allocation failures](https://github.com/artalis-io/jev-bush/blob/85fd51f8c99b1b8b7ea26d33e02413e681a9271d/test/library_model_faults.c), [CI](https://github.com/artalis-io/jev-bush/blob/85fd51f8c99b1b8b7ea26d33e02413e681a9271d/.github/workflows/ci.yml), [model release](https://github.com/artalis-io/jev-bush/blob/85fd51f8c99b1b8b7ea26d33e02413e681a9271d/.github/workflows/model-release.yml).

### Practices that need adaptation

- This engine generates autoregressive text; Jev's bounded diffusion candidate scoring is a different algorithm and model. Its transformer/tokenizer kernels are not drop-in components.
- Jev has a strict scalar CPU oracle. This project has no CPU inference implementation initially. Its oracle is the pinned Zig path plus the independent reference measurements.
- Preserve upstream CUDA fast-math flags during parity. Do not advertise Jev-like universal bit-exact strict semantics: evaluated FP8 AR/speculation already diverges near ties.
- Jev's optional CUDA backend dynamically loads driver/NVRTC and documents process-lifetime resources. Here, reuse the evaluated NVCC-built kernels and make resource teardown a release requirement. Do not import a process-lifetime-only design accidentally.
- Adopt Jev's recoverable-error discipline, but use explicit status propagation and cleanup labels for the new host. Do not longjmp across CUDA/C++ frames or transplant its error machinery mechanically.
- Jev permits elaborate logger reentrancy. Use a simpler explicit callback contract initially; reentrant destruction is outside this release.

## 5. Product boundary and repository shape

### Initial supported scope

- Linux AMD64 host, one Blackwell `sm_120a` GPU; the tested RTX PRO 6000 Max-Q is the first validation target.
- Official pinned Inkling-2-Mini text safetensors and its tokenizer/template.
- Existing dense precision and global-KV modes, AR, DSpark hybrid speculative decoding, CUDA graphs, prefix/reset behavior and streaming token callbacks.
- C11 public library, static/shared builds and a CLI. Optional HTTP adapter follows core parity.
- The same model semantics and measured configuration defaults, including capacity heuristics. No speculative automatic retuning based on VRAM size.

The host is C; the retained GPU implementation is **CUDA C++20**, compiled by NVCC and linked with the relevant CUDA libraries and C++ runtime. This is a C API / C host project, not an entirely ISO-C or dependency-free GPU binary.

CPU parser/ownership tests can run without CUDA. That does not provide CPU generation or establish GPU support on macOS, Windows, AMD/ROCm or DGX Spark. Arm64/other Blackwell targets need separate ABI/build/instruction validation; the current `sm_120a` result does not prove GB10 compatibility.

Initial exclusions: arbitrary Hugging Face architectures, GGUF loading, vision/audio, tensor parallelism, continuous batching, a distributed scheduler, training, optional ablation/refusal tooling and LoopCD behavior changes. Reject unsupported requested modes explicitly. Retained optional kernel fields remain disabled/zeroed. Models stay outside the checkout.

### Proposed structure

```text
LICENSE / NOTICE / THIRD_PARTY.md
UPSTREAM.json                     # exact upstream revision, file hashes, local patches
include/sq.h                      # only installed public header
src/
  internal/                       # checked arithmetic, model layout, private ABI
  platform.c / error.c / arena.c
  json.c / safetensors.c
  tokenizer.c / chat_template.c
  weights.c / cuda_backend.c
  engine.c / graphs.c / sampling.c / prefix_cache.c
  api.c
cuda/                             # retained source; one canonical kernel copy
cli/main.c
server/main.c                     # separate optional transport target
examples/library.c
test/                             # unit, ABI, fault, integration and GPU cases
benchmarks/fixtures/               # small synthetic prompts, not weights
benchmarks/manifests/
tools/                            # upstream verification, oracle comparisons, evidence
Makefile / sq.pc.in
.github/workflows/                 # host checks and protected model release gate
docs/                             # design, API ownership, numerics, model support
```


Prefer an actual Git fork that retains upstream history, with `origin` under Artalis and an `upstream` remote. Preserve the original revision for oracle builds. During transition Zig reference files can coexist; the released C runtime must not require Zig. The oracle is a separate checkout/worktree of the pinned upstream commit, not a second maintained product implementation.

Use ordinary C translation units initially. Jev's generated amalgamation is useful packaging, but forcing the first port into one giant editable file would make review harder. If introduced later, generated `dist/sq.c` is never hand-edited and does not include CUDA compilation units.

## 6. CUDA boundary: reuse without ABI guesswork

The evaluated `src/cuda.zig` declares **76 C-callable functions and seven extern records**. The existing `cuda/cuda_abi.cu` already exports C linkage; `weights.hpp` and the kernel headers remain C++ internals. A C host can call these functions directly.

Required contracts:

1. Establish a private C-compatible ABI header with fixed-width scalars, opaque handles and the existing POD layouts. No CUDA SDK headers or C++ classes in the installed public API.
2. Initially verify the header against existing C++ and Zig layouts. Extract duplicate declarations into one shared private header only as a reviewed bridge change. Preserve symbol signatures and numerical bodies.
3. Test `sizeof`, alignment and **every field offset** for all seven records, including nested arrays, padding and sampler seed placement. Test pointer width, enum values, config dimensions and the 48-visit schedule. Do not use packed structs to force a match.
4. Compare exported-symbol/prototype inventories, not just symbol names: C linkage does not validate calling convention, pointee type or ownership.
5. Check arena-size calculation against actual allocations. GPU scratch capacity and launch geometry are coupled; the rejected split experiment demonstrated why changing one constant alone is unsafe.
6. Keep all bridge-owned C++ allocations visible to C ownership. Graph wrappers use allocation despite the bridge's thin-wrapper description. Catch allocation exceptions inside the C++ boundary; none may cross into C.

The kernel-math source stays frozen for the parity stages. Header extraction, status handling and explicit resource-release entry points are allowed as individually reviewed bridge/lifecycle patches afterward. Changing a `.cu` wrapper or resource handle is distinguished from changing a `__global__` computation.

## 7. Public C API and ownership

Provisional API sketch; names and complete layouts require review before coding:

```c
typedef struct sq_model sq_model;
typedef struct sq_session sq_session;
typedef struct sq_generation sq_generation;
typedef uint32_t sq_status;

sq_status sq_model_load(const char *directory,
                        const sq_model_options *options, sq_model **out);
sq_status sq_session_create(sq_model *model,
                            const sq_session_options *options, sq_session **out);
sq_status sq_generate(sq_session *session, const sq_request *request,
                       sq_token_fn callback, void *user, sq_generation **out);
sq_status sq_session_cancel(sq_session *session);
sq_status sq_session_reset(sq_session *session);
sq_status sq_session_destroy(sq_session *session);
void sq_generation_free(sq_generation *generation);
void sq_model_retain(sq_model *model);
void sq_model_release(sq_model *model);
```

Provide separate bounded tokenize/decode/template functions and version/capability/error queries. Typed generation consumes validated token IDs directly. JSON adapters parse into typed structures; generation never serializes a request to JSON and reparses it internally.

Public structures carry size/version fields, explicit defaults, reserved-zero fields and documented extension rules. Discriminants use fixed-width typedefs. Strings are counted byte spans; buffer limits and UTF-8 handling are defined. Private CUDA layouts are not public ABI or serialized file formats.

### Ownership

- A model owns immutable tokenizer/configuration and weight mappings/device images. Sessions retain it. Releasing the caller's model reference cannot free a live session's weights.
- A session owns streams/events, graphs, sampler state, KV/recurrent state, arenas, prefix snapshots, cancellation state and callback workspace.
- Input spans are borrowed until the synchronous call returns. Token callbacks receive borrowed data valid only during that callback.
- A generation result owns its output and statistics. It survives session destruction and is released through the library's allocator family. Caller-provided allocator context must outlive all allocations using it.
- Only cancellation is allowed reentrantly on the active session. Callbacks must not destroy it, reset it, recursively generate or alter model state. Logger callbacks have the same restriction.
- An attempt to destroy/reset an active session returns `BUSY` without freeing resources. Success invalidates the handle. Access to already destroyed handles remains a caller programming error.

### Concurrency: conservative first release

Expose the model/session separation now, but initially support **one GPU-resident model and one live session per process**. Report this in capabilities; reject a second live GPU session/model with `BUSY` or an explicit unsupported-capacity status. Host-only immutable queries can run concurrently when documented.

One active generation per session is mandatory. A second generation returns `BUSY`, never silently queues. Cancellation is atomic and callable from another thread.

This restriction is necessary because upstream's process-global cuBLAS handle and capture/stream assumptions do not establish Jev-like independent GPU sessions. Before lifting it, replace/audit every process-global mutable CUDA resource, prove per-instance capture/workspace isolation, and add real concurrent-model/session tests. An immutable model pointer alone does not prove GPU thread safety.

## 8. Errors, cleanup and cancellation

Expected user/runtime failures return statuses: invalid argument, invalid model, unsupported format/device, context limit, I/O, host/device OOM, busy, cancelled and CUDA failure. Keep raw backend error information in a bounded diagnostic record. No library `exit`, abort or unconditional stderr printing for malformed data, missing files or ordinary allocation failures.

Use explicit cleanup paths with zero-initialized handles and tracked initialization counts. Release partially created graphs/events/streams/mappings/arenas exactly once. Preserve the first failure while reporting cleanup problems. Internal invariant failures remain assertions active in release builds, but public validation must make malformed input a normal error path.

State transitions:

```text
READY → PREFILL → DECODE → READY
                 ↓
             CANCELLING → synchronize → invalidate/reset → READY
GPU execution failure → POISONED → teardown
```

Validation failures leave the session reusable. An asynchronous device execution failure poisons it; subsequent inference is rejected. Device loss may require the embedding application to restart its worker. The library must not reset a shared device or terminate the host process to repair it.

Cancellation checks occur before/after prefill chunks, before decode verification/commit, and at streaming boundaries. A callback stop is distinguished from cancellation, EOS and length exhaustion. Cancellation is cooperative: an already executing CUDA kernel cannot be promised immediate preemption. Measure the longest non-preemptible region and publish the observed bound under stated conditions.

Before reusing a cancelled session, synchronize the main and auxiliary streams, retire outstanding drafting work, and invalidate partially updated prefix/recurrent state. An unrelated subsequent request must match a fresh session. Do not claim cancelled work published no state until these tests pass.

The evaluated cuBLAS exit leaks are a release blocker for the embedding lifecycle. With the initial single-session restriction, an explicit release routine can safely close the owned handle at the final runtime teardown; future concurrency requires per-instance ownership. Destroy graphs before their captured workspaces. Never call `cudaDeviceReset`, reset another library's context, or destroy resources borrowed from the caller. Preserve the caller's thread/device context as specified by the backend contract.

## 9. Input, memory and numerical invariants

### Model and tokenizer

Retain the current checkpoint format. Validate tensor names, dtypes, shapes, ranges, duplicate keys, file/shard identity, index offsets and exact byte counts. Check integer multiplication/addition/alignment before allocation or pointer arithmetic. Reject unknown layouts; do not silently guess a quantization representation.

Treat checkpoint paths as local configuration, validate the fixed supported shard layout, and reject index references outside its allowed artifact paths. Loading never executes checkpoint-supplied code or fetches remote assets. Memory-map files read-only; retain mappings while referenced. Preserve two-pass weight/runtime planning, alignment and exact-fill checks, including additional library allocations. Query available device memory for planning, then still handle allocation failure: another process can allocate between the query and allocation.

Port the actual Inkling BPE, Unicode splitting, special-token handling and chat template. Jev's DiffusionGemma tokenizer is different. Preserve counted lengths, little-endian decoding and Zig's explicit wrapping arithmetic with unsigned C operations. Avoid signed overflow, aliasing violations and unaligned typed loads. Untrusted text cannot become trusted control tokens merely by spelling them.

### Execution

Preserve the visit schedule, local/global cache layout, ring positions, convolution state, target/drafter synchronization, vocabulary mapping, sampling configuration, random-counter progression, acceptance/commit order and position accounting. Centralize layout definitions or enforce mirror equivalence in compiled ABI/config tests.

Prefix entries are valid only for matching model/tokenizer/template identities, relevant execution configuration and exact token history. Keep caches session-owned. Reset/cancel/mode changes cannot leave stale validity metadata. Server authentication and cache scope are separate concerns; no multi-tenant security claim follows from prefix reuse tests.

Plan and reserve core workspaces before generation. The core hot path must not perform per-token host/device allocation. Server admission may allocate bounded request storage outside the hot path. Benchmark allocation counts as well as throughput.

### Numerical policy

Preserve CUDA compiler flags and weight packing during the port; keep host fast-math off initially. The source's `exact` mode means BF16 dense weights with the checkpoint's existing quantized experts, not an entirely unquantized or universally bit-exact engine.

Compare C AR with Zig AR, and C speculative with Zig speculative, under identical mode/KV/seed/history/cache settings. **Do not demand that C FP8 speculation match Zig AR when that cross-mode comparison already diverges upstream.** C-versus-Zig divergences block parity until explained; existing upstream numerical limits remain documented rather than becoming excuses to loosen port tolerances.

Use identical frozen CUDA objects where feasible for matched host comparisons; retain the independently built clean Zig oracle as well. Hash source, objects, shared libraries and toolchain. A different binary hash caused by paths/debug metadata is distinct from functional divergence.

## 10. CLI and optional serving

First deliver library-backed `generate`, `bench`, `logits` and `kld`. Preserve useful upstream flags through a documented mapping; reject unsupported behavior. Synthetic zero-KV diagnostics remain visibly separate from real-context benchmarks. The benchmark command reports each run and actual generated counts, not just a best token rate.

The HTTP adapter is a separate target and dependency boundary. Prefer an established Artalis C HTTP component such as Keel after checking its exact revision/license and adapter requirements. Do not mechanically transplant the synchronous socket loop or make the core depend on networking.

Minimum server release requirements:

- Control/event-loop thread stays responsive while one inference worker owns the session. Admission is bounded; excess requests receive a documented busy response.
- Liveness, readiness, busy and draining states are distinct. Busy alone is not device failure. Shutdown stops admission, cancels/drains safely, then releases resources.
- Localhost bind by default, explicit authentication configuration, redacted logs and bounded body/depth/message/tool/stop counts.
- Empty message arrays and malformed/unsupported requests fail clearly. Limit violations never silently drop trailing conversation history or tool calls.
- One shared tested stop/UTF-8 stream filter handles text/chat, streaming/nonstreaming paths, including stop strings across token and network boundaries.
- SSE usage and finish reasons agree with committed tokens; role events are not reported as useful-answer TTFT. Client disconnect requests cancellation; slow clients cannot cause unlimited buffering.
- Template/tool-call semantics must have fixtures before being advertised. No claim of full OpenAI compatibility from a small endpoint smoke suite.

The latest upstream's conversation-limit fix is relevant evidence for a regression test, not a reason to merge its unrelated kernels during host translation.

## 11. Verification and release gates

All checks below are **proposed for the C implementation**. Existing Zig/reference measurements inform fixtures and limits; this design review did not execute a C port.

| Layer | Required evidence |
|---|---|
| Public ABI | C and C++ consumers, supported struct sizes/versions, hidden internal symbols, static/shared/installed usage |
| CUDA ABI | All seven records' field offsets/alignment; prototype/symbol and dimension/schedule inventories; no SDK types in public header |
| Parsers | Real-layout metadata fixtures, truncated/duplicate/overflow/bad-dtype inputs; Unicode and special-token vectors; fuzzed JSON/safetensors |
| Ownership | Fail every host allocation and injected CUDA acquisition step; zero leaked owned resources or invalid frees; lifetime/refcount/busy/callback contracts |
| Packing | C/Zig planned and actual sizes match; per-tensor packed-byte checks for BF16/FP8/NVFP4 scales, expert transforms and optional MTP vocabulary |
| Core execution | Same-ID greedy fixtures and streamed teacher-forced logits against pinned Zig; graph/eager, AR/spec and mode-specific comparisons |
| State isolation | A/B/A, long/short/long, changed prefix, reset, cancellation in prefill/decode, sampler/mode switching, invalid input then valid call |
| Speculation | Same-seed path replay against Zig; acceptance/commit/RNG counters; retain stochastic distribution diagnostics and known FP8 limitations |
| GPU memory | Compute Sanitizer graph/eager and AR/full-B8 cases crossing the local window; model/session churn and stable memory; actual owned cuBLAS cleanup |
| Long context | Real 8K/131K and million-window FP8/Q4 independent-code retrieval, three depths, EOS, HTTP cache/reset where serving is built |
| Transport | Stop/Unicode regressions, auth/limits, malformed/empty inputs, disconnect/slow client, readiness during long prefill, bounded overload/shutdown |
| Performance | Matched input/mode/seed/kernel hashes, cold versus cache hit, six ordinary runs; expensive long cases explicitly fewer; memory/latency/energy evidence |

Use existing validated fixtures and scripts where possible. Promote portable, small synthetic fixtures and manifests into the new repository; keep model weights and large raw artifacts external. Preserve the distinction between the formal v2 patterned throughput prompts and the stronger v3 independent-code retrieval inputs.

The oracle ladder is: primitive CPU tests → ABI/packing parity → pinned Zig execution → independent official Transformers logits/functional checks → complete real-context workflows. One oracle cannot replace all others.

### Regression rules

- No missing test is a pass; host-only or mocked CUDA CI is not GPU evidence.
- No unexplained C/Zig token, cache, stop or ownership regression is accepted. Establish logit tolerances from repeated baseline behavior before evaluating the port, not afterward to rescue a failure.
- Check each workload/backend mode. An aggregate speedup cannot hide a speculative regression. A reproducible median slowdown over 3% on an ordinary sentinel requires investigation and explicit review; compare interleaved controls and noise before attributing it to the port.
- Extra allocations or memory growth require explanation; principal memory planning should match the baseline, with documented bounded overhead.
- Do not label `--leak-check no` as a full leak pass. Resolve owned leaks; any vendor suppression must be narrowly justified and pinned.
- Full context/workload characterization runs before the first release and after meaningful kernel, tokenizer, packing, sampling or graph changes. Smaller 8K/131K sentinels protect routine host changes.

Public CI runs on hosted machines without model secrets: GCC/Clang C11, sanitizers, host concurrency, ABI consumers, source/lint rules and parser fuzz smoke. Real-model/GPU jobs run on protected trusted revisions with artifacts and manifests. Do not run arbitrary public PR code on `core` with credentials. Provisioning a GitHub runner or changing Herding Sparks is a separate operational action, not part of this design.

## 12. Build and provenance

Use a small Makefile consistent with Jev's operator experience: static/shared library, CLI, `check`, host sanitizer/fuzz targets, `gpu-check`, `install` and `pkg-config`. Exact target names are provisional.

Compile host C11 and the five retained CUDA translation units separately. Keep NVCC's evaluated `-O3`, C++20, `--use_fast_math` and `sm_120a` settings during parity. Capture NVCC, host compiler, CUDA runtime/cuBLAS, driver, build flags and dependency hashes. The evaluated toolkit is 13.5.37 on driver 595.104.02; successful tested minor-version compatibility is not a guarantee for every newer CUDA API. A newer upstream pin introduces `cublasLt`, which belongs to that upgrade's manifest.

Dependencies are explicitly versioned; no floating UI downloads or silent model acquisition during startup. Ordinary generation is offline with prepared assets. Downloads are separate tools with pinned revisions/checksums and atomic publication.

Every release manifest records:

- Artalis source commit and original upstream revision;
- exact upstream file hashes, local patches and CUDA/object/library hashes;
- model/tokenizer/template identities and precision/cache configuration;
- compiler/driver/GPU identity, clocks/power conditions and workload manifest;
- executed/skipped/failed tests, numerical comparisons and per-run benchmark evidence.

## 13. Implementation milestones

| Milestone | Work | Exit gate |
|---|---|---|
| P0: baseline and repository | Resolve name; preserve Git history, license/notices and upstream pin; external models; freeze oracle/fixtures; scaffolding only | Rebuild oracle, identify ABI, reproduce a short tested case; no kernel upgrade |
| P1: host foundation and inputs | C ownership/errors, checked arenas, model/container parsing, tokenizer/template, C ABI layout checks | CPU/fault/fuzz/ABI checks and token/packing parity |
| P2: autoregressive engine | Runtime planning, streams, schedule, recurrent/KV state, graphs, typed generate, logits/kld and CLI | C-versus-Zig AR and graph/eager parity; real 8K/131K; cleanup under failure |
| P3: speculative execution and cache | DSpark, hybrid lookup, adaptive graphs, RNG/commit, exact prefix/reset behavior | Mode-matched parity, stochastic diagnostics, independent 1M retrieval and memory checks |
| P4: embeddable lifecycle | Safe final cuBLAS/resource teardown, reference lifetimes, busy states, cooperative cancellation | Repeated load/generate/cancel/destroy; partial-init failures; no owned leaks; context preservation |
| P5: optional server and release | Separate HTTP adapter, validation/stop/auth/health/overload/shutdown, installed artifacts | API regressions and complete real-model release matrix with final evidence |
| P6: upstream upgrade/tuning | Rebaseline chosen newer kernels, matching ABI/config/NOTICE, one-variable experiments | Same correctness/quality gates plus controlled performance evidence; independently reviewed change |

Land reviewable slices. Each stage has working tests and a usable narrower artifact; do not postpone the first real-model comparison until the whole Zig host is translated. No calendar estimate is reliable before P1/P2 expose the porting and packing differences. This is a moderate engineering project; GPU reuse saves the largest algorithmic rewrite but does not remove host state-machine work.

## 14. Decisions for the review

Recommended defaults:

1. Apache-2.0 Artalis fork retaining upstream history and notices.
2. Evaluated `d10c0abb…` as the mechanical-port baseline; newer kernels get their own rebaseline/change.
3. C11 host/public API plus retained CUDA C++20, Linux AMD64/`sm_120a` first.
4. Library and CLI before HTTP; explicit single-model/session capability initially.
5. Stable kernel math/default heuristics and numerical modes during translation; no automatic performance claims.
6. The Jev-style ownership, faults, golden fixtures, installed consumers and protected model-backed gates are mandatory parts of the port.

Selected for kickoff: squid-c / sq_, the evaluated upstream revision, and library/CLI first. HTTP and a newer-kernel upgrade remain later milestones. See the root README for implemented capabilities.

## Implementation progress — 25 November 2026

The stacked C AR slice implements the P2 forward/prefill/greedy-decode path,
including scoped cuBLAS, graphs, three KV formats, logits and cooperative
cancellation. See [AR validation](ar-validation.md) for the exact tested matrix,
limits and reproducible commands. The subsequent sampling/fixed-width DSpark slice is documented in
[speculation validation](speculation-validation.md); the larger P3/P4/P5
release gates remain separate; the implemented typed API uses an explicit GPU
weight-image handle rather than presenting an unfinished all-in-one model API.
One live C session is enforced while immutable weight images remain separately
owned and retained. No serving or broad performance qualification is implied.
