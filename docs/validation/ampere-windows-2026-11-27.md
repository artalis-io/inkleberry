# Windows Ampere validation — 27 November 2026

Inkleberry Finn runs real Inkling inference on `edge`'s RTX 3090 Ti using the
experimental `sm_86` profile. Native execution and recovery passed. Cross-device
numerical equivalence has **not** passed: 11/28 benchmark token sequences match
the existing Blackwell goldens. Keep Ampere experimental and preserve the goldens.
The [machine-readable report](ampere-windows-2026-11-27.json) records per-case
measurements, variability and the differing sequences' keys.

## Revisions and environment

Initial build checks used `f9f7fc36d133b5ed7567aa7cb5ba007fa959e412`, with 88
source/build input hashes checked. Runtime tests used
`0d41de5cda1229053bf0627e30936ddb7f94f8d1`; 92 build/guard/test inputs matched
that revision on the remote workspace. Later changes document these results,
quiet PowerShell progress output and select the correct Windows CUDA DLL path.
No device numerical bodies or public ABI layouts were changed.

- Windows 11 Pro 10.0.26300, x86-64.
- NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, 24,564 MiB VRAM.
- NVIDIA driver 621.01; CUDA toolkit/NVCC 13.3.73.
- MSVC compiler 19.44.35229 (toolchain directory 14.44.35207); clang-cl 23.1.1.
- Both C host/compiler profiles were built separately. NVCC used MSVC as its
  CUDA host compiler in both; this does not claim NVCC clang host support.
- About 293 GiB free before model acquisition; Wi-Fi link reported 216 Mbps.
- An explicitly approved desktop reservation recorded 25 existing Windows/UI
  processes, their full names and creation times, GPU UUID, a 1,800 MiB preflight
  memory ceiling and a two-hour expiry. Baseline usage was 1,544 MiB at 0% activity.
  This reservation is a preflight contract, not a GPU scheduler/exclusivity lock.

The official model revision is `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`.
Tokenizer, main checkpoint and MTP checkpoint SHA-256 values were verified on
both controller and target against `test/fixtures/inkling-golden-v1.json`.
Weights remain external; none entered Git or a source package.

## Build and host checks

After `source scripts/msvc-env.sh`, for each of `CC=cl` and `CC=clang-cl`:

```sh
make -j2 CC=cl PYTHON=python CUDA_ARCH=86 EXPERIMENTAL_AMPERE=1 gpu-build
make -j2 CC=cl PYTHON=python CUDA_ARCH=86 EXPERIMENTAL_AMPERE=1 windows-cuda-build-check runtime-gpu-package
# Repeat with CC=clang-cl.
```

Both passed compilation/linking, export/PE checks, ctypes ABI discovery and
76 bridge prototypes, 7 records and 72 native layout checks. Each provider has
five `sm_86` images, with no Blackwell image. Runtime packages carry explicit
`sm86-experimental` identity and `qualification=not-established-by-build`.

The Linux dry-run regression was fixed to ignore recursive Make directory
messages when checking cleanup operands. The native acceptance fixture's thread
handle is initialized, resolving an MSVC warning; CPU CI now compiles that real
fixture. Local host/install/distribution/provenance/lint checks passed. The
Windows reservation guard has 34 mocked tests, including expiry, recycled PIDs,
new processes during sampling, unknown activity and excess memory. Linux and
Windows CI passed for the runtime-test revision.

## Executed GPU correctness and recovery

Each command was launched through `tools/run_cuda_smoke.py`, with the approved
GPU UUID and external Windows reservation. The ordinary dedicated-GPU guard
remains unchanged. Its earlier refusal was verified before the reservation was
approved; no guard bypass or unrelated process termination was used.

| Check | Result |
| --- | --- |
| Native C eager kernel and two captured replays with changing inputs | Exact data checks passed |
| Checked/original packing and conversion parity | 5 cases passed |
| Independent FP4/FP8 conversion equations | 65,536 packed pairs; vector lanes, signed zero/NaN, poisoned eager/graph/replay passed |
| Sampling distributions, proposal acceptance, Dirac residual and top-k=1 | 4 diagnostics, 8,191 draws each, passed |
| Compute Sanitizer memcheck on packing | 0 errors |
| NVFP4/FP8-KV CLI: eager AR, graph AR, fixed and adaptive speculation | Same 32-token smoke output |
| clang-cl generator versus MSVC generator | Same smoke output |
| Weight/KV format matrix | 18/18 full-answer arithmetic checks reached EOS with final `17`; all 9 eager/graph text pairs matched |
| Real inference facade | Expected final answer `7`; 876 input tokens, 846 prefix tokens reused |
| Facade lifecycle | Copied chat, independent streams, BUSY, scope isolation, cancellation, stop filtering and replay passed |
| Fixed and adaptive recovery | 8,192-token prompt; context restoration, callback stop, cancellation, replay passed |

The format matrix used NVFP4, FP8 and the existing mixed "exact" weight profile,
BF16/FP8/Q4 KV, and eager/graph execution, with a 128-token output limit. It is a
format/admission smoke test, not broad quality qualification for every precision.
The existing "exact" profile is not universally full-precision weights.

The native facade's first answer generation took 0.789 seconds, excluding model
loading. Its CLI smoke tests deliberately stopped at 32 tokens and include
reasoning text; those are execution/parity tests, not complete-answer quality
checks. The facade test separately verifies the final answer.

Cancellation return after the request during real 8,192-token prefill measured
0.536448 seconds for fixed decoding and 0.530868 seconds for adaptive decoding.
These are individual measurements, not a worst-case latency guarantee.
Adaptive execution exercised conditional graph paths and reported drafting widths
2/4/8. No graph or precision fallback was introduced to pass the tests.

The correctness campaign's sampled peak VRAM was 17,838 MiB, including the desktop.
Samples were approximately one second apart and do not prove an instantaneous peak.

## Benchmark

The unmodified `native-benchmark` workload ran 28 groups: seven workloads times
AR, fixed speculation, adaptive speculation and sampled AR. Each group had one
warm-up and five measured repetitions: 168 total runs, 140 measured. Weights were
NVFP4 with MTP, KV was FP8, graphs enabled, seed 42. Greedy modes used temperature
zero; sampled AR used temperature 0.7. The output cap was 96 tokens.

Session capacity was 32,768 tokens; **actual** prompts ranged up to 8,844 tokens.
This validates real prefill at that tested length, not 32K or million-token context.

The following rates pool 35 measured runs across seven different workloads per
profile. They are workload-mixture summaries, not a latency SLO or equal-output
speedup claim. Per-case median/min/max/CV are in the JSON report.

| Profile | Median decode tok/s | Run range tok/s | Median TTFT ms |
| --- | ---: | ---: | ---: |
| AR | 112.21 | 98.09–117.07 | 79.55 |
| Fixed speculation | 143.74 | 93.47–183.44 | 81.70 |
| Adaptive speculation | 176.11 | 128.30–198.58 | 80.16 |
| Sampled AR | 110.29 | 97.60–110.54 | 78.40 |

The campaign completed in about 208 seconds, including setup. Observed sampled
peaks: 18,288 MiB VRAM including desktop, 331.32 W, 80°C. No power/clock/watchdog
settings were changed. Power samples include setup/prefill/decode; they do not
establish decode-only joules per token or PCIe/bandwidth utilization.

## Numerical and quality boundary

All five repetitions in every group produced identical tokens on this Ampere
configuration. The strict existing golden verifier nevertheless failed, first
at `4/ar`; only 11/28 complete sequences matched Blackwell. No goldens were
rewritten or tolerance silently relaxed.

Within Ampere, AR and speculative greedy sequences match for 4/7 workloads;
C generation, Python editing and JSON extraction differ. The existing Blackwell
reference also has AR/speculative differences for 2/7 workloads. These observations
do not prove that every difference is harmless rounding, nor establish exact
distribution preservation. Logit-level and independent-reference investigation
remain necessary.

Decoded inspection found coherent reasoning; the JSON extraction example retained
`name=Acme` and `quantity=17` despite different reasoning. Several code and RAG
outputs hit the 96-token limit before a full final answer. Do not count them as
successful code validation or completed long-context retrieval. The benchmark
measures execution and repeatability, not a comprehensive quality evaluation.

## Remaining qualification

Linux Ampere, multi-GPU behavior, long-context limits beyond the tested 8,844-token
prefill, broad semantic quality, cross-device logit agreement, failure injection,
extracted native GPU SDK execution and a fresh Blackwell performance regression
campaign remain unexecuted. Keep this experimental and the implementation PR in
draft while numerical equivalence is unresolved. There is no HTTP server here.

Interrupted transfer artifacts were rejected by checksum. Fresh downloads were
staged separately and published only after target-side verification. Windows can
report stale directory-entry sizes for open files; progress was checked using an
open shared read handle rather than treating a displayed zero as a stalled transfer.
No existing model or unrelated workload was overwritten or removed.

## End-of-run state

The final guarded Make target `conversion-oracle-check` passed on Windows using
`CUDA_RUNTIME_PATH` pointing to the toolkit's `bin` directory. The POSIX default
remains `lib64`; existing guarded Make targets now use the same platform variable.

After all tests, the GPU reported 1,544 MiB desktop usage, 22,766 MiB free, 0%
utilization and approximately 8.24 W. No qualification processes remained. The
completed reservation snapshot was removed, so another maintenance window needs
a fresh reviewed snapshot. Only the task-created, checksum-rejected transfer
partial was cleaned up; validated model files and unrelated data remain intact.
