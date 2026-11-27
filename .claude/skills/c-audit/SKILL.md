---
name: c-audit
description: Audit Inkleberry Finn's C API, parsers, ownership, concurrency, CUDA boundaries, numerical contracts and qualification evidence.
user-invocable: true
---

# Inkleberry Finn C audit

Adapted from bitnet.c's `.claude/skills/c-audit/SKILL.md` and Jev Bush's error,
parallel-region, concurrency, invariance and run-manifest audits. See
`docs/audits/README.md` for the migration map and source revisions. This skill
is self-contained; sibling repositories are not runtime dependencies.

`/c-audit [paths]` audits without modifying runtime code. Default scope is
canonical `src/`, `include/`, `cuda/`, their build contracts and validation tools.
`dist/` is generated evidence, not a second codebase to fix manually.

## Procedure

1. Read `AGENTS.md`, `UPSTREAM.json`, `THIRD_PARTY.md`, the public ownership/API
   docs and the exact current source revision. Record dirty state and scope.
2. Run `make audit-check` and `make lint`. These are lexical/source-policy guards,
   not a comprehensive C/C++ analyzer or proof about callback targets.
3. Review memory safety: checked size arithmetic before allocation, offsets and
   alignment; allocator pairing; initialization; output clearing; partial cleanup;
   borrowed/mapped lifetime; bounds on strings, tokens, positions and scratch.
   Do not mechanically replace malloc with calloc: overflow and allocator policy
   need independent validation. Distinguish public inputs from private prevalidated
   bridge records. Check stack budgets in test/CLI code separately from libraries.
4. Treat safetensors/tokenizer/JSON as untrusted inputs: bounded header/counts,
   tensor dtype/rank/shape/byte ranges, duplicate/overlapping metadata, UTF-8,
   tokenizer-profile support and resource limits. Inkleberry has no GGUF reader;
   do not invent GGUF acceptance or import bitnet-specific guards.
5. Review GPU ownership: device/context entry and restoration, session/weight
   leases, arenas, streams, cuBLAS handles, graph resources, retryable cleanup,
   cancellation and cache invalidation. SDK errors must return through checked
   adapters; C callers must validate shapes before private launches.
6. Review concurrency: atomic ordering for cancellation and terminal publication,
   progress/result reads, callback reentrancy, immutable tables, allocator/log
   callbacks and handle lifetime. One prepared native session/active run per
   process is the current contract. Do not import Jev Bush's concurrent-session
   scheduling model or mistake BUSY admission for GPU parallelism.
7. Review model/quantization/launch separation: geometry and packing invariants,
   scale finiteness, layout/alignment, actual hardware admission and preparation
   outside graph capture. Inspect retained device code read-only. Do not change
   kernels, goldens, math defaults, rounding, sampling or speculative acceptance
   to make an audit pass. Pinned reconstruction is not a kernel-correctness proof.
8. Check evidence separately: exact model and executable identities, toolchain,
   source verification versus caller declarations, real prefill, repetitions,
   EOS/truncation, cache reset, semantic answers and performance configuration.
   Fixed-device token regression, cross-device numerical comparison and semantic
   task acceptance are different checks. Existing cross-device/batch differences
   remain visible; do not promise Jev Bush strict-mode invariance here.
9. Check build/package/ABI: modular/amalgamated agreement, C/C++ installed
   consumers, symbol exports, Linux hardening, MSVC/clang-cl, sanitizers and fuzz
   failure paths. Use existing tests; add focused regressions for new audit checks.
10. Report confirmed findings by severity, exact file:line, triggering conditions,
    evidence, impact and smallest suggested fix. Separate analyzer candidates,
    rejected false positives, known limitations and checks not executed. A grep
    match is not a confirmed vulnerability. Run tests appropriate to changed code.

## Execution limits

GPU execution requires the existing UUID guard and current reservation policy.
Audit does not authorize stopping inference services, installing toolchains,
changing clocks or GPU settings, or obtaining private model/customer inputs.
Reuse prior qualification as prior evidence, clearly dated; do not relabel it as
new execution. If hardware is unavailable, complete independent checks and report
GPU qualification pending. Do not create goals or delegate solely because this
skill is invoked.

`--fix` additionally authorizes small reviewed fixes after recording findings;
follow canonical source generation and required checks. Architectural/numerical
changes require an explicit design/review. Without `--fix`, implement requested
audit tooling and documentation, and leave runtime findings reviewable.

## Report

Write `docs/audits/YYYY-MM-DD-c-audit.md` (and a compact JSON companion when useful):
source/scope; findings table; category coverage; tool outputs and false-positive
triage; executed versus prior/unexecuted tests; kernel/golden preservation; next
steps. Link evidence and avoid copying model weights, secrets or bulk logs to Git.

The adapted bitnet.c checklist retains its MIT notice in `LICENSE.bitnet`;
Inkleberry-specific additions follow the repository license.
