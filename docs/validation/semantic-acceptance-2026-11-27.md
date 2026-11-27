# Semantic acceptance — 27 November 2026

This is a task-level smoke suite, separate from numerical identity and timed
benchmarks. The native client uses source `a421d780967a2bd22e804214c80a3e09a5f80852`.
Seven cases run once in each of AR, fixed speculation, adaptive speculation and
sampled decoding, using 2,048 output tokens, seed 42, NVFP4/MTP, FP8 global KV,
32K context and graphs. No kernel, production source, math default or existing
golden fixture changed.

| Device | Completed checks | Task acceptance |
| --- | ---: | --- |
| GB202, `core` | 28 | Pass |
| Ampere RTX 3090 Ti, `edge` | 28 | Pass |
| GB10, `spark-7468` | 28 | Pass |

## What was checked

- Exact arithmetic: `455` and `-53`.
- JSON: exactly `name`, integer `quantity` and boolean `urgent`, with expected
  values and no duplicate keys or extra prose.
- Generated C: GCC 14.2 compilation with warnings as errors and executable tests
  for negative values, singleton input, integer extrema, duplicates and maximum
  position. The restricted compiler container is pinned to
  `gcc@sha256:b99b86a28812b1e6453a231a947dc43d76fe192788a12f344a9b568bf9f5d24c`.
  Separate deliberately incorrect and syntactically invalid programs were rejected.
- Exact identifier copying and middle-of-context record retrieval.
- EOS termination, a complete final-answer channel, nonempty valid text, a
  bounded single paragraph and no detected repeated eight-word loops.

All 12 final prose paragraphs across the three devices were read as well: they describe
coherent quiet autumn scenes with complete sentences and no obvious repetition.
The automated prose checks are structural; they do not establish general quality.
Wording and implementation differences are accepted when the task contract holds.
For example, one generated maximum function guards empty input, another assumes
the documented `n >= 1`; both satisfy the requested contract and executable tests.

## Failure handling and review

The first prose validator mistakenly required the literal words “autumn” and
“morning”. Valid descriptions need not repeat prompt words. That invalid check
was removed and regression-tested without changing prompts or regenerating model
responses. Topic and coherence remain explicit manual-review responsibilities.

One Windows collection stopped before `json-fixed` because the idle guard observed
desktop GPU activity. It remains an incomplete, non-passing collection. A fresh
collection subsequently completed all 28 checks; the guard was not bypassed.
The desktop reservation was closed after completion. No desktop apps or services
were stopped. GB10 vLLM was restored healthy: `/health` returns 200 and `/v1/models`
reports the expected model. All eight Cursus container IDs and the original
service enablement match their pre-test state. No Cursus deployment changed.

The validator rejects truncated/cancelled generation, incorrect task answers,
malformed/duplicate JSON, code compile/runtime failures, repetition, missing or
duplicate campaign entries and stale successful assessments. Generated code runs
only in a disposable offline, non-root, read-only container with bounded resources;
timeout cleanup removes it. The GPU host needs no Docker.

## Evidence and scope

See the [machine-readable report](semantic-acceptance-2026-11-27.json) for executable,
model, validator and compiler-image identities. Raw prompts, responses and receipts
remain outside Git under `data/inkleberry-semantic-2026-11-27/`. Existing Blackwell,
GB10 and Ampere token-golden differences remain unchanged. Passing these seven
cases is evidence of task usefulness, not proof of broad semantic equivalence,
speculative distribution preservation or all-context numerical accuracy.

Executed local checks: host/install/contract tests, nine semantic-validator
regression tests, lint, reference-free/extracted packages, ASan/UBSan, modular and
amalgamated fuzz smoke tests, plus actual restricted-container positive/negative
code tests. All seven CI jobs passed for code commit `ddfb1ca`; final evidence-commit CI
must be checked separately. All three devices pass 28/28 checks (84 total),
with matching source model-file checksums. No response hit the output limit.

Reproduction commands and exact acceptance rules are in
[semantic acceptance](../semantic-acceptance.md). Collection uses the documented
`run` command with `--profiles ar fixed adaptive sampled`; assessment uses the
pinned compiler image above. A completed collection is not a passing assessment.
