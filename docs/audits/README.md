# Repeatable audits

Inkleberry Finn adopts audit practices from Jev Bush rather than its model or
execution assumptions. The reference examined was `artalis-io/jev-bush` commit
`85fd51f8c99b1b8b7ea26d33e02413e681a9271d`. Its BSD-0-Clause call-graph traversal
informed `tools/check_audit_boundaries.py`. The repository-local
[skill](../../.claude/skills/c-audit/SKILL.md) also adapts bitnet.c's C-audit
checklist to safetensors, explicit status returns and the CUDA execution contract.
No sibling checkout is needed to use these audits.

| Jev Bush practice | Inkleberry implementation | Contract difference |
| --- | --- | --- |
| Error-frame reachability | Existing C/CUDA fatal/allocation lint; new transitive control-path guard | Status/error returns; no setjmp/longjmp frames to transplant |
| Parallel-region safety | Existing CUDA launch/resource policy and source-pin reconstruction tests | No OpenMP transformer regions; zero OpenMP regions is not CUDA safety evidence |
| Library concurrency | Existing concurrency, ABI admission, ownership and TSan tests; new cancellation/progress call-path guard | One prepared GPU session/active run per process, explicit BUSY rejection |
| Strict invariance | Existing `compare_numerics.py`, repetition/identity checks and per-device release gates | No cross-device or singleton/batch bit-identity promise |
| Model golden release | Existing `model_release.py`, `run_model_release.py`, native/facade and extracted SDK checks | Preserve current goldens and report known deviations |
| Accuracy/task scoring | New semantic acceptance suite and actual compiled-code tests | Task-level pass is separate from exact-token/quality-distribution equivalence |
| Run manifest | Existing release preflight and semantic receipts | Distinguish verified source from a caller-declared revision |
| Amalgamation/installed library/ELF hardening | Existing generated, installed-consumer, export and hardening gates | Native Windows/CUDA layout and runtime profiles remain separate checks |

## Commands

```sh
make audit-check             # pins, generated ABI, direct policies, new call paths
make lint                    # format, namespace and allocation/error boundaries
make check install-check install-contract-check
make reference-free-check release-check
make sanitize
make fuzz-smoke amalgamation-fuzz-smoke
```

`audit-boundary-check` is also part of ordinary `make check`, so mandatory Linux
and native Windows CI run it. It scans all 29 canonical C modules, checks required
control roots transitively for known allocation/blocking/termination calls, keeps
OS primitives in their platform implementations, and rejects public platform SDK
includes. Its mutation tests cover indirect allocator member names, helper chains,
cycles, platform alternatives, missing roots and comments/literals.

This is a lexical regression guard: macros, arbitrary function-pointer targets,
external libraries and CUDA device bodies require their existing tests and manual
review. The checker merges mutually exclusive platform definitions conservatively.
It does not introduce a C parser, orchestration framework or strict-math mode.

For a full `/c-audit`, follow the local skill and produce a findings report.
GPU runs remain explicit, guarded operations. Do not repeat expensive hardware
campaigns for documentation-only audit migration or present old runs as new ones.
