# Ampere compatibility: kernel audit and dispatch design

Status: **proposed for review; steps 1–2 complete**. This change contains design
and audit evidence only. It does not enable Ampere, change kernels, loosen the
GPU guard, launch GPU work, download models or interrupt existing workloads.

Audited main: `e0341c8a030e35481c34d50bdef025e9bc1b3c27` (Windows portability,
PR #5). Retained CUDA source: `d10c0abb2c8990be81c08ba84c66a8333a691b33`.
See the [machine-readable audit](../validation/ampere-compile-audit-2026-11-26.json).

## Recommendation

Reuse the existing packed-weight kernels, tensor-core implementation and C
engine. Introduce a small **compile-time target profile**, with ordinary ordered
stream launches for experimental `sm_86` and the existing PDL path for Blackwell.
Do not introduce another inference backend, per-kernel vtable, model format,
quantizer, scheduler or serving dependency.

The source is less Blackwell-exclusive than its comments suggest. The unchanged
GPU library and provider DLL already compile and link for `sm_86`. What remains
is host launch policy, precise device/image admission, resource/error checks,
and real numerical/lifecycle/performance qualification. Compilation alone does
not establish that inference works.

## 1. Executed audit

### Scope and evidence

Reviewed all retained CUDA files, their checked adapters, weight packing/layout,
C execution/graph paths, capability admission, Make/compiler profiles, provenance
checks and idle-device guard. CUDA source declares 95 `__global__` definitions
across the five implementation files; templates expand into more entry points.

On `edge`, without source edits or GPU execution:

```sh
source scripts/msvc-env.sh
make -j2 CC=cl PYTHON=python gpu infer-shared CUDA_ARCH=86
```

This succeeded with MSVC 19.44.35229 and NVCC 13.3.73 using the existing
`-O3 --use_fast_math -std=c++20` settings. The 72 compiled source/header/build
inputs were SHA-256 compared with audited main and all matched. Output stayed
under `.build/windows-msvc-x64-release-cuda86` in the existing isolated checkout.
The existing Blackwell-only runtime gate was not changed or exercised.

Offline `cuobjdump` inspection of the resulting provider found five `sm_86`
cubins and 423 function resource records. Maximum reported registers/thread:
176; static shared memory: 42,820 bytes; stack: 512 bytes. Twenty-nine records
have nonzero stack. `LOCAL:0` is not proof of zero spills or low memory traffic.
Dynamic shared-memory requests must be assessed separately.

Read-only telemetry reported RTX 3090 Ti, CC 8.6, driver 621.01, 24,564 MiB total,
1,697 MiB used and 22,613 MiB free. Zero instantaneous utilization does not mean
exclusive ownership. The current 256 MiB idle threshold would reject this GPU.
No attempt was made to bypass it.

Raw build/resource logs and the source hash manifest are outside Git:
`/Users/mark/artalis.io/data/inkleberry-ampere-audit-2026-11-26`.

### Kernel and integration inventory

References below are paths in the audited revision, not promises about future
line numbers. Operation names, not comments saying “Blackwell”, determine the
actual hardware dependency.

| Area | Existing source behavior | Ampere assessment / required action |
| --- | --- | --- |
| FP4 unpack | `cuda/kernels_common.cuh`, `ifn_cuda_cvt_e2m1x2_to_half2` and x8 helper: hardware conversion above the architecture guard; a 16-entry signed-value lookup fallback below it | Fallback already compiles for 8.6. Exhaustively test all packed bytes, signed zero and pair ordering; retain it initially |
| FP8 unpack | Same header, `ifn_cuda_cvt_e4m3x2_to_half2`: hardware conversion guarded at 8.9; CUDA FP8 type conversion below | Existing fallback compiles. Check every byte/pair, finite values and NaN classification against an independent oracle |
| FP8 packing / NVFP4 packing | `cuda/cuda_abi.cu` and `src/cuda/checked_weights.cu`: SDK FP8 conversion and explicit software E2M1 quantization | No new weight format or quantizer needed; verify packed outputs, scales and boundary rounding |
| MXFP8 scales | `ifn_cuda_cvt_ue8m0_to_float` uses explicit exponent bits | Preserve special-value handling and current scale interpretation |
| FP16/BF16 tensor-core math | Common helpers use `mma.sync.aligned.m16n8k16` with FP16/BF16 inputs and FP32 accumulation | These forms are Ampere-capable. Packed FP4/FP8 storage does not require native FP4/FP8 tensor cores here [S1] |
| Prefill transfers / matrix fragments | `cuda/kernels_prefill.cu`: ordinary `cp.async`, `ldmatrix`, FP16 MMA | Not TMA, WGMMA or `tcgen05`; reuse initially. Current `sm_86` compilation succeeds |
| Cache-control loads | Common helpers use noncoherent immutable-weight loads and coherent dependent-activation loads, including L2 prefetch hints | Compiler accepts the exact forms for 8.6; retain coherent dependency loads and weight immutability |
| PDL device instructions | Common `ifn_cuda_pdl_*` helpers guard `griddepcontrol` at architecture 9.0 | Already become no-ops on Ampere; this alone does not select a suitable host launch policy |
| PDL host launches | `ifn_cuda_launch_pdl` always requests `cudaLaunchAttributeProgrammaticStreamSerialization`; 50 textual call sites across GEMV, Inkling and DSpark files | Required integration seam. Use ordinary same-stream dependencies for 8.6. PDL overlap requires CC 9.0+ [S2]; exact current attribute behavior on 8.6 was not executed |
| Attention, routing, sampling, convolution, reductions | Shared kernels and the C orchestration reuse shuffle, half/BF16 and ordinary arithmetic paths | No separate Ampere algorithms indicated by source audit; numerical tests still required |
| DSpark / prefix / cancellation | Existing C ownership, generation and cache paths call the same CUDA bridge | Preserve them; qualify eager, graph, fixed-speculative and adaptive modes separately |
| Conditional graphs | `cuda/cuda_abi.cu`: device selector calls `cudaGraphSetConditional`; host constructs a three-body SWITCH | Compiles for 8.6. Toolkit/driver graph support and execution must be tested; do not infer support or lack of support from PDL availability [S3] |
| Runtime admission | `src/cuda/capability.h` admits only CC 12.0/12.1 | Keep closed until qualified. Future admission must match both the selected build profile and actual device image |

The source scan found no inline `tcgen05`, `wgmma`, TMA bulk-copy or cluster-launch
instructions in the retained implementation. NVIDIA documents the instruction
families in [S1]; actual `sm_86` compilation is stronger evidence for these exact
source branches than a comment-based classification.

### Resources and performance risks

The grouped prefill configuration computes shared memory from `FMT`, `K` and
`TM` in `ifn_cuda_wg_cfg`. The dispatched `K` values are 2304 and 1152; `TM` is
32 for B<=512 and 128 above it. Evaluating those existing expressions gives:

| Format | K | TM=32 bytes | TM=128 bytes |
| --- | ---: | ---: | ---: |
| NVFP4 | 2304 | 92,160 | 98,304 |
| NVFP4 | 1152 | 82,944 | 89,088 |
| MXFP8 | 2304 | 93,184 | 99,328 |
| MXFP8 | 1152 | 88,576 | 94,720 |

The largest request is below the documented 99 KiB (101,376 bytes) per-block
opt-in limit for CC 8.6 [S4]. This is a static fit calculation, not successful
resource negotiation or a performance measurement. Account for static allocation,
actual function limits and register usage before launching.

`ifn_cuda_dispatch_wgemm_tm` currently caches a `cudaFuncSetAttribute` result in a
function-local static and discards the error. That is a checked-launch review
item: neither failed opt-in nor a result cached from another device/context may
be treated as successful preparation. Validate the actual context/function
configuration before capture; surface the original error and avoid launching
an invalid configuration. Do not add device-wide synchronization to solve it.

Other measured/static hotspots worth profiling *after* correctness:

- B=8 NVFP4 LM-head specialization reports 176 registers/thread. The 680-CTA
  persistent LM-head constant is tuned independently of the actual SM count.
- The 1024-thread controller reports 63 registers/thread, 28,176 static shared
  bytes and 120 stack bytes. Do not reduce its block size: warp ownership and
  barriers are part of the implementation.
- KV split counts, prefill tile/stage counts and fixed launch geometries are
  tuning choices. Preserve them for the first comparison; no SM-count scaling
  or alternate fused/unfused controller until measurements justify it.
- SDK conversion fallback cost may change register pressure and latency. No
  throughput projection follows from additional/decreased VRAM or compilation.

### Memory feasibility

Existing CPU-only `weight-plan` results, in GiB:

| Mode | AR resident / load peak | MTP resident / load peak |
| --- | ---: | ---: |
| NVFP4 | 14.280 / 15.138 | 14.549 / 15.503 |
| FP8 | 16.559 / 17.418 | 16.828 / 17.782 |
| Exact mixed checkpoint | 17.588 / 18.446 | 18.276 / 19.230 |

These are weight plans, not total inference memory. Session KV, prefill scratch,
CUDA graphs, cuBLAS/driver allocations and existing desktop allocations are
additional. `src/engine_layout.c` allocates global KV by requested context and
local KV separately; DSpark adds its own cache. Do not claim a maximum context
from the weight table. Start future execution with a small explicit context and
budget, then grow using the real planners and measured peaks.

Keep original packed weights and decode in registers/shared tiles. Expanding the
whole model to FP16 would be an unnecessary memory-policy and format change.

## 2. Proposed minimal dispatch design

### A. Explicit build profile, separate from runtime device facts

| Profile | Device image | Admission | Launch policy |
| --- | --- | --- | --- |
| Existing Blackwell 12.0 | `sm_120a` | Matching 12.0 device and qualified modes | Existing PDL path |
| Existing Blackwell 12.1 | `sm_121a` | Matching 12.1 device and its qualification status | Existing PDL path |
| Experimental Ampere | `sm_86` | Exactly 8.6, explicit experimental build, then qualification | Ordinary ordered stream launches |

Do not turn the gate into `major >= 8`, admit 8.0/Ada/Hopper without tests, or
assume an architecture-specific Blackwell cubin runs on another family. Physical
capability, a matching compiled image and qualified feature behavior are distinct
checks. GB10's existing numerical qualification gap remains open.

Have the existing Make/compiler profile emit one private target definition for
host CUDA compilation; keep `__CUDA_ARCH__` for device conversion selection.
**Do not use `__CUDA_ARCH__` to choose host launch behavior**: it is undefined
in NVCC's host compilation pass. Check inconsistent target definitions at build
time and inspect emitted image architecture in qualification.

Blackwell remains the default. Ampere support stays experimental and explicitly
selected until runtime acceptance passes. Existing model/sampling/KV options,
public C ABI, CUDA bridge prototypes and resource ownership do not change.

### B. One launch helper, compile-time policy

Keep the existing launch sites and kernel arguments. Select policy inside their
shared host launch helper:

- Blackwell: retain the current launch attributes and error-recording path.
- Ampere: construct the same launch dimensions/stream but leave PDL attributes
  absent (`numAttrs=0`, null attributes); use the existing extensible launch API.
- Device-side PDL helpers retain their existing architecture guards. Normal
  same-stream completion dependencies provide ordering on Ampere, including
  ordinary dependency edges when captured into a graph.
- Do not add per-token device queries, runtime kernel vtables, hidden stream
  changes, host/device synchronization between every operation, or a process-wide
  mutable architecture selector.

A compile-time branch avoids a new hot-path decision on Blackwell. Capability
and resource validation belongs to device/session preparation, against its owned
CUDA context, outside graph capture. Scope restoration and recoverable errors
must still work if preparation fails.

### C. Preserve numerical source and provenance

The anticipated retained-source edit is a small, explicit host launch-policy hook
in `cuda/kernels_common.cuh`; no rewrite of numerical bodies is proposed. Resource
error handling may require another narrowly reviewed host-only change in the
prefill dispatcher. Keep these separate from any later performance work.

Record those exact host changes as reviewed platform adaptations, with original
upstream hashes preserved and reconstruction checked. A separate platform-patch
manifest is preferable to disguising architecture work in the existing error-only
adaptation list. Unexpected edits to device mathematics must still fail provenance
verification. Do not fork full copies of the GEMV/attention/MoE kernels or use
preprocessor symbol-renaming tricks to hide dispatch.

### D. Fit into the existing build/ABI

| Location | Intended responsibility during implementation |
| --- | --- |
| `mk/toolchain.mk`, `Makefile` | Explicit target/profile definition; architecture-separated output and runtime package roots on POSIX as well as Windows |
| `config/sources.mk` | Remains the only source inventory; add a private helper only if necessary |
| `src/cuda/capability.h`, checked adapters | Match compiled profile/device; preserve context restoration and error propagation |
| `cuda/kernels_common.cuh` | Minimal host policy seam; existing device fallbacks and kernel callers reused |
| `src/cuda/checked_prefill.cu` / host prefill dispatcher | Validate resource opt-in/function limits before capture and execution; no ignored preparation error |
| Provenance tooling/config | Reviewed, exact host-platform patch reconstruction and mutation rejection |
| Existing C tests/native qualification | Extend matrices and failure tests; keep Blackwell goldens and gates intact |

Windows outputs are already isolated by compiler and `CUDA_ARCH`. POSIX currently
uses shared artifact filenames even when its configuration stamp forces rebuilds.
Separate experimental Ampere archives, provider, generator and package output so
switching targets cannot overwrite or accidentally combine Blackwell artifacts.
Reuse the Make graph and packaging implementation, not a second build system.

Keep feature discovery honest. Compiled features are not proof of a particular
runtime's support. Requested graph/adaptive features must either work under the
selected profile or fail explicitly before execution. Do not silently switch
adaptive decoding to fixed decoding, disable graphs, change KV precision, or
alter quantization to make a test pass.

## Qualification contract for the later implementation

1. **CPU/build first:** both Windows compiler profiles, existing Linux gates,
   frozen C/CUDA layouts, static/shared/amalgamated consumers, isolated packages,
   exact target-image checks, unsupported-device rejection and context restoration.
2. **Primitive oracles:** exhaustive FP4/FP8 decoding, pack boundaries and scales;
   FP16/BF16 matrix operations, non-multiple dimensions, normalization, attention,
   convolution, routing/top-k ties and sampling. Use independent expected values,
   not only comparisons between two implementations sharing the same bug.
3. **Ordered execution:** producer/consumer chains across all replaced PDL call
   families; eager versus captured graph outputs; poisoned output buffers and
   alternating requests to expose stale reads. Check launch errors and opt-in
   failures; cross-device/context reuse must not inherit stale policy/resource
   success. Verify Ampere emits no PDL launch attributes.
4. **Real model:** reserved 8.6 device, pinned weights/tokenizer and configuration;
   AR eager first, then AR graphs, fixed speculation, adaptive conditional SWITCH
   graphs, prefix reuse and long-context tests. Validate actual conditional graph
   creation/instantiation/execution rather than assuming support from compilation.
5. **Failure/lifecycle:** real cancellation, callbacks returning stop/cancel,
   prefill cancellation, poisoned sessions, interrupted graph preparation,
   allocation failures and repeat cleanup. Measure cancellation latency and
   Windows watchdog behavior; do not disable TDR or change machine settings.
6. **Numerics:** compare against qualified Blackwell outputs and independent
   tensor references with predeclared operation-specific tolerances. Keep existing
   deterministic goldens; explain any token/logit drift rather than overwriting
   expected values. Test speculative probability/acceptance behavior, not just
   coherent-looking text. Compilation or an intact ABI cannot satisfy this gate.
7. **Performance only after correctness:** same weights, quantization, context,
   seeds/sampling, output count and speculative mode; warm-up and at least five
   ordinary repetitions. Measure prefill/decode, TTFT, latency tails, peak memory,
   power and acceptance separately. Preserve Blackwell performance gates and its
   current default flags. Profile before changing launch geometry or fusion.

An initial small-context NVFP4 AR workload is a practical first execution case,
not a new default or a reason to omit exact/FP8/MTP validation. Runtime support
should be reported per OS/compiler/device/mode; Windows 8.6 success alone would
not independently qualify Linux 8.6 or resolve GB10 differences.

### Operational prerequisite

`edge` currently fails the existing idle-memory check. A later GPU run needs an
explicitly reserved maintenance window/device and valid idle evidence. Do not
raise the threshold, treat WDDM N/A process counters as zero, or terminate apps
as a side effect of qualification. If an unavoidable desktop baseline requires
a Windows-specific reservation contract, review that separately with the owner.
This audit authorizes none of those changes.

## Decisions to review before coding

- Start with exactly `sm_86`, existing compact formats and ordinary stream launches.
- Preserve Blackwell device code and launch policy; no new public ABI or backend.
- Make host platform adaptations explicit and provenance-verifiable.
- Treat graph/resource behavior and numerical correctness as acceptance gates;
  do not open the production capability gate based on this successful build.
- Defer SM-count tuning, alternate controller fusion and conversion optimization
  until the shared-kernel baseline is measured.

## Primary references

- [S1 — NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html): conversion and tensor/memory instruction target requirements.
- [S2 — NVIDIA Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html): launch and dependency semantics.
- [S3 — NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html): conditional handle/SWITCH construction and body requirements.
- [S4 — NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/13.1.0/ampere-tuning-guide/index.html): CC 8.6 resource limits and async-copy support.
